"""Single-process, threaded CPU-only annotation server with sequential cut worker."""
import argparse
import fcntl
import io
import json
import math
import re
import subprocess
import threading
from pathlib import Path
from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.serving import make_server
from core import Store, DEFAULT_DATA, HERE, Conflict, binary


def create_app(store):
    app=Flask(__name__,static_folder=None)
    app.config['MAX_CONTENT_LENGTH']=64*1024
    frame_lock=threading.BoundedSemaphore(1)
    export_lock=threading.Lock()

    @app.before_request
    def same_origin():
        if request.method=='POST':
            # No credentials/authentication: deploy only through the existing private platform ingress.
            if request.headers.get('X-Ego-Desk')!='1':
                return jsonify(error='missing desk request header'),403
            if request.headers.get('Origin') and request.headers['Origin'].rstrip('/')!=request.host_url.rstrip('/'):
                return jsonify(error='cross-origin writes denied'),403

    @app.after_request
    def headers(resp):
        resp.headers['X-Content-Type-Options']='nosniff'
        resp.headers['Referrer-Policy']='same-origin'
        if request.path.startswith('/api/'):
            resp.headers['Cache-Control']='no-store'
        return resp

    @app.errorhandler(Conflict)
    def conflict(exc): return jsonify(error=str(exc)),409

    @app.errorhandler(ValueError)
    def invalid(exc): return jsonify(error=str(exc)),400

    @app.errorhandler(KeyError)
    def missing(exc): return jsonify(error='条目不存在'),404

    @app.get('/')
    def index(): return send_from_directory(HERE/'web','index.html')

    @app.get('/assets/<path:name>')
    def asset(name): return send_from_directory(HERE/'web',name)

    @app.get('/health')
    def health():
        rows=store.items()
        return jsonify(status='ok',name='ego数据反应标注台',count=len(rows),
                       counts={s:sum(r['status']==s for r in rows) for s in ('pending','draft','queued','cutting','ready','rejected','failed')})

    def public(row):
        row=dict(row);s=dict(row['source'])
        for k in ('source_path','proxy_path'):s.pop(k,None)
        row['source']=s
        if row['output']:
            o=json.loads(row['output']);o.pop('path',None);row['output']=o
        row['media_url']='/media/'+row['id']
        row['clip_url']=f"/clip/{row['id']}?version={row['version']}" if row['status']=='ready' else None
        return row

    @app.get('/api/items')
    def items(): return jsonify(items=[public(r) for r in store.items()])

    @app.get('/api/items/<ident>')
    def item(ident): return jsonify(public(store.get(ident)))

    @app.post('/api/items/<ident>')
    def save(ident):
        payload=request.get_json()
        if not isinstance(payload,dict): raise ValueError('JSON object required')
        return jsonify(public(store.save(ident,payload)))

    @app.get('/api/history/<ident>')
    def history(ident):
        store.get(ident)
        with store.db() as db:
            rows=[dict(r) for r in db.execute('SELECT version,actor,action,created_at FROM history WHERE id=? ORDER BY seq DESC',(ident,))]
        return jsonify(items=rows)

    @app.get('/media/<ident>')
    def media(ident):
        return send_file(store.get(ident)['source']['proxy_path'],mimetype='video/mp4',conditional=True)

    @app.get('/clip/<ident>')
    def clip(ident):
        r=store.get(ident)
        if r['status']!='ready' or str(r['version'])!=request.args.get('version'):
            raise Conflict('切片版本已变化，请刷新')
        return send_file(json.loads(r['output'])['path'],mimetype='video/mp4',conditional=True)

    @app.get('/api/frame/<ident>')
    def frame(ident):
        s=store.get(ident)['source']; t=float(request.args.get('time','nan'))
        if not math.isfinite(t) or not s['window_start']<=t<s['window_end']:
            raise ValueError('frame time outside current window')
        if not frame_lock.acquire(blocking=False):
            return jsonify(error='上一帧正在读取，请稍后'),429
        try:
            # Decode from original source, not the playback proxy. Output first frame at/after requested timestamp.
            result=subprocess.run([binary('ffmpeg'),'-nostdin','-v','info','-threads','1','-ss',f'{t:.6f}',
               '-copyts','-i',s['source_path'],'-frames:v','1','-an','-vf',"showinfo,scale=w='min(960,iw)':h=-2",
               '-threads','1','-f','image2pipe','-vcodec','mjpeg','pipe:1'],capture_output=True,timeout=45,check=True)
            found=re.search(rb'pts_time:([0-9.e+-]+)',result.stderr)
            resp=send_file(io.BytesIO(result.stdout),mimetype='image/jpeg',max_age=0)
            resp.headers['X-Frame-Requested-Time']=f'{t:.6f}'
            if found: resp.headers['X-Frame-Source-Time']=found.group(1).decode()
            return resp
        finally: frame_lock.release()

    @app.post('/api/export')
    def export():
        if not export_lock.acquire(blocking=False):
            return jsonify(error='正在生成导出包，请稍后'),409
        try:return jsonify(store.export())
        finally:export_lock.release()

    @app.get('/exports/<name>')
    def download(name):
        if not re.fullmatch(r'ego_[A-Za-z0-9_]+\.zip',name):raise KeyError(name)
        return send_from_directory(store.root/'exports',name,as_attachment=True)

    return app


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=5132)
    a=p.parse_args();store=Store(a.data_dir)
    lock=(store.root/'server.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with store.db() as db:db.execute("UPDATE items SET status='queued' WHERE status='cutting'")
    stop=threading.Event()
    def worker():
        while not stop.is_set():
            try:
                if not store.process_one():stop.wait(1)
            except Exception as exc:
                print('worker error',type(exc).__name__,flush=True);stop.wait(3)
    threading.Thread(target=worker,daemon=True).start()
    server=make_server(a.host,a.port,create_app(store),threaded=True)
    print(f'ego数据反应标注台 listening on {a.host}:{a.port}',flush=True)
    try:server.serve_forever()
    finally:stop.set();server.server_close();lock.close()


if __name__=='__main__':main()
