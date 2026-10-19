"""Explicit small-batch import from PT-C manifest; no model calls or GPU.

Selection JSON: [{id, reaction_cn, note?}]. Source reaction/reason remain immutable.
Default is a dry-run; --execute creates only local short proxies and new desk rows.
"""
import argparse
import json
from pathlib import Path
from core import Store, DEFAULT_DATA, cut, probe, sha, dump, now


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    m=json.loads(a.manifest.read_text()); selection=json.loads(a.selection.read_text())
    assert 1<=len(selection)<=30, 'small batch limit: 30'
    train={x['id']:x for x in m['train']}
    excluded=set(m['eval_video_uids']); seen=set(); plan=[]
    for item in selection:
        x=train[item['id']]
        assert x['src'] in ('egonormia','goalstep'), 'Why QA is not a ready reaction label'
        assert x['video_uid'] not in excluded and x['video_uid'] not in seen
        assert Path(x['path']).is_file() and item['reaction_cn'].strip()
        seen.add(x['video_uid'])
        plan.append(dict(id=x['id'],dataset=x['src'],video_uid=x['video_uid'],anchor=x['end']))
    print(json.dumps(dict(count=len(plan),gpu=0,ffmpeg_threads=2,max_proxy_seconds=30,
                         output=str(a.data_dir),items=plan),ensure_ascii=False,indent=2),flush=True)
    if not a.execute: return
    store=Store(a.data_dir)
    lock=store.root/'import_lock.json'
    identity=dict(manifest_sha256=sha(a.manifest),selection_sha256=sha(a.selection))
    if lock.exists():
        assert json.loads(lock.read_text())==identity,'inputs changed; use another data directory'
    else: dump(lock,identity)
    for i,item in enumerate(selection):
        x=train[item['id']]; ident=x['id']; path=Path(x['path'])
        metadata=store.root/'media'/(ident+'.json')
        proxy=store.root/'media'/(ident+'.mp4')
        if metadata.exists():
            source=json.loads(metadata.read_text())
            assert proxy.exists() and sha(proxy)==source['proxy_sha256']
        else:
            info=probe(path); anchor=float(x['end'])
            start=max(0,anchor-20); end=min(info['duration'],anchor+10)
            pi=cut(path,proxy,start,end-start,proxy=True)
            reason,reaction=x['target'].split('\n<next> ',1)
            stat=path.stat()
            source=dict(id=ident,dataset=x['src'],video_uid=x['video_uid'],source_path=str(path),
                source_stat=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns),source_metadata=info,
                source_manifest_sha256=identity['manifest_sha256'],source_item_id=ident,
                anchor=anchor,anchor_kind='annotation_timestamp_not_verified_onset',
                window_start=start,window_end=end,suggested_start=float(x['start']),
                reaction_original=reaction.strip(),reason_original=reason.removeprefix('<reason> ').strip(),
                reaction_cn=item['reaction_cn'],translation_method='Codex faithful translation; no new action facts',
                import_note=item.get('note',''),proxy_path=str(proxy),proxy_sha256=sha(proxy),proxy_metadata=pi,
                pretraining_seen=True,probe_overlap=False,original_split='pt_c_train',
                candidate_status='pending_boundary_and_reaction_review')
            dump(metadata,source)
        store.add(source)
        print(f'{i+1}/{len(selection)} imported {ident}',flush=True)
    dump(store.root/'batch_summary.json',dict(created_at=now(),count=len(selection),
         dataset_counts={s:sum(x['dataset']==s for x in plan) for s in ('egonormia','goalstep')},
         excluded_probe_videos=len(excluded),unique_videos=len(seen),**identity))


if __name__=='__main__':main()
