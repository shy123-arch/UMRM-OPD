"""Isolated functional tests: no production writes, OSS reads or GPU use."""
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from openpyxl import load_workbook
from core import Store, Conflict, binary, probe, sha
from server import create_app


class DeskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media_tmp=tempfile.TemporaryDirectory()
        cls.video=Path(cls.media_tmp.name)/'source.mp4'
        subprocess.run([binary('ffmpeg'),'-nostdin','-v','error','-f','lavfi','-i',
                        'testsrc2=size=320x180:rate=25:duration=6','-c:v','libx264',
                        '-threads','1','-pix_fmt','yuv420p','-movflags','+faststart',str(cls.video)],check=True)

    @classmethod
    def tearDownClass(cls):cls.media_tmp.cleanup()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.store=Store(self.tmp.name)
        self.source=dict(id='fixture_1',dataset='goalstep',video_uid='fixture_video',
            source_path=str(self.video),proxy_path=str(self.video),source_metadata=probe(self.video),
            window_start=0.,window_end=6.,suggested_start=0.,anchor=3.,
            reaction_original='Pick up the box.',reaction_cn='拿起盒子。',reason_original='A test goal.',
            pretraining_seen=True,probe_overlap=False)
        self.store.add(self.source);self.client=create_app(self.store).test_client()
        self.data=dict(version=0,action='cut',annotator_id='QA',start=.2,end=2.2,
            reaction='Pick up the box.',reaction_cn='拿起盒子。',state='',note='',category='',
            boundary_ok=True,support_ok=True)

    def tearDown(self):self.tmp.cleanup()

    def test_boundaries_and_confirmation(self):
        for fields in [dict(end=float('nan')),dict(start=-1),dict(end=7),dict(end=.4),
                       dict(boundary_ok=False),dict(reaction_cn='English only')]:
            with self.assertRaises(ValueError):self.store.save('fixture_1',dict(self.data,**fields))
        self.assertEqual(self.store.get('fixture_1')['version'],0)

    def test_concurrent_edits_history_and_immutable_source(self):
        self.store.save('fixture_1',self.data)
        with self.assertRaises(Conflict):self.store.save('fixture_1',self.data)
        self.assertFalse(self.store.add(self.source))
        with self.assertRaises(Conflict):self.store.add(dict(self.source,reaction_original='Changed'))
        self.assertEqual(self.store.get('fixture_1')['source']['reaction_original'],'Pick up the box.')
        self.assertEqual(len(self.client.get('/api/history/fixture_1').json['items']),1)

    def test_http_range_and_write_guard(self):
        r=self.client.get('/media/fixture_1',headers={'Range':'bytes=0-99'})
        self.assertEqual(r.status_code,206);self.assertEqual(len(r.data),100)
        r.close()
        self.assertEqual(self.client.get('/media/fixture_1',headers={'Range':'bytes=999999999-'}).status_code,416)
        self.assertEqual(self.client.get('/exports/../../etc/passwd').status_code,404)
        self.assertEqual(self.client.post('/api/items/fixture_1',json=self.data).status_code,403)
        self.assertEqual(self.client.post('/api/items/fixture_1',json=self.data,
                         headers={'X-Ego-Desk':'1','Origin':'https://other.test'}).status_code,403)
        self.assertEqual(self.client.post('/api/items/fixture_1',json=self.data,headers={'X-Ego-Desk':'1'}).status_code,200)

    def test_source_frame_timestamp(self):
        r=self.client.get('/api/frame/fixture_1?time=2.001')
        self.assertEqual(r.status_code,200)
        self.assertTrue(r.data.startswith(b'\xff\xd8'))
        self.assertAlmostEqual(float(r.headers['X-Frame-Source-Time']),2.04,places=3)

    def test_real_cut_and_compatible_export(self):
        self.store.save('fixture_1',self.data);self.assertTrue(self.store.process_one())
        r=self.store.get('fixture_1');self.assertEqual(r['status'],'ready')
        output=json.loads(r['output']);self.assertAlmostEqual(output['duration'],2.,places=2)
        z=self.store.export(); archive=Path(self.tmp.name)/'exports'/z['name']
        self.assertTrue(archive.is_file())
        folder=archive.with_suffix('')
        wb=load_workbook(folder/'reaction_clips.xlsx'); rows=list(wb.active.values);wb.close()
        index={k:i for i,k in enumerate(rows[0])}
        for k in ['视频名称','开始时间(秒)','结束时间(秒)','reaction','reaction_ch','标注来源ID']:self.assertIn(k,index)
        self.assertEqual(rows[1][index['开始时间(秒)']],0)
        self.assertAlmostEqual(rows[1][index['结束时间(秒)']],2.)
        self.assertEqual(rows[1][index['源视频开始时间(秒)']],.2)
        with zipfile.ZipFile(folder/'videos.zip') as v:self.assertEqual(v.namelist(),[rows[1][0]])
        m=json.loads((folder/'manifest.json').read_text())
        self.assertEqual(m['items'][0]['status'],'pending_independent_review')
        self.assertEqual(m['items'][0]['split'],'unassigned')
        self.assertEqual(sha(Path(output['path'])),m['items'][0]['clip']['sha256'])
        self.store.save('fixture_1',dict(self.data,version=1,action='draft'))
        self.assertEqual(self.client.get('/clip/fixture_1?version=1').status_code,409)
        with self.assertRaises(ValueError):self.store.export()

    def test_old_worker_cannot_overwrite_new_revision(self):
        self.store.save('fixture_1',self.data)
        def during_cut(*args,**kwargs):
            self.store.save('fixture_1',dict(self.data,version=1,action='draft'))
            raise RuntimeError('old worker failed')
        with patch('core.cut',side_effect=during_cut):self.store.process_one()
        r=self.store.get('fixture_1');self.assertEqual(r['version'],2);self.assertEqual(r['status'],'draft')


if __name__=='__main__':unittest.main()
