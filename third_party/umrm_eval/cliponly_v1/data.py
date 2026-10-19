"""Preparation only; encoder consumes a separate allowlisted manifest."""
import hashlib,json,re
from pathlib import Path
from urllib.parse import urlparse,parse_qs
import numpy as np
ROOT=Path('/mnt/workspace/Wilson/ck/umrm')
RUN=ROOT/'umrm_model/outputs/cliponly_v1_20260918'
BASE=ROOT/'benchmark/judge_exploration/evaluations/astra_motion_loop/baseline636_20260915'
NAMES=['Pelvis','Left_Hip','Right_Hip','Spine1','Left_Knee','Right_Knee','Spine2','Left_Ankle','Right_Ankle','Spine3','Left_Foot','Right_Foot','Neck','Left_Collar','Right_Collar','Head','Left_Shoulder','Right_Shoulder','Left_Elbow','Right_Elbow','Left_Wrist','Right_Wrist','Left_Hand','Right_Hand']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def encoder_record(row):return {k:row[k] for k in ('id','clip_path')}
def source_keys(row):
    link=str(row.get('src_link') or '').strip(); sid=str(row.get('src_id') or '').strip()
    keys=[]
    for value in [link,sid]:
        bv=re.search(r'BV[0-9A-Za-z]+',value)
        if bv: keys.append('bilibili:'+bv[0]);continue
        u=urlparse(value)
        if 'youtu.be' in u.netloc: keys.append('youtube:'+u.path.strip('/'));continue
        if 'youtube.com' in u.netloc and parse_qs(u.query).get('v'):keys.append('youtube:'+parse_qs(u.query)['v'][0]);continue
        if value:keys.append(('url:'+u.netloc.lower()+u.path.rstrip('/')) if u.netloc else 'id:'+value)
    return keys

def build_split(rows):
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={};unknown=set()
    for i,r in enumerate(rows):
        keys=source_keys(r)
        if not keys:unknown.add(i)
        keys+=['clip:'+r['clip_sha256']]
        for k in keys:
            if k in seen:parent[find(i)]=find(seen[k])
            else:seen[k]=i
    unknown_groups={find(i) for i in unknown};groups={}
    for i,r in enumerate(rows):groups.setdefault(find(i),[]).append(r['id'])
    known=[g for g in groups if g not in unknown_groups]
    known.sort(key=lambda g:hashlib.sha256(('42:'+','.join(sorted(groups[g]))).encode()).hexdigest())
    n=max(1,round(len(known)*.15));split={}
    for j,g in enumerate(known):
        label='test' if j<n else 'val' if j<2*n else 'train'
        for qid in groups[g]:split[qid]=label
    for g in unknown_groups:
        for qid in groups[g]:split[qid]='unknown_source'
    return split

def canonical(x):
    x=np.asarray(x,dtype=np.float64).copy();origin=x[0,0].copy();origin[1]=0;x-=origin
    right=x[0,2]-x[0,1];right[1]=0
    if np.linalg.norm(right)<1e-5:raise ValueError('degenerate initial hip axis')
    right/=np.linalg.norm(right);up=np.array([0.,1.,0.])
    x=x@np.stack([right,up,np.cross(right,up)],axis=1)
    # Initial floor only, no future-dependent floor alignment.
    x[:,:,1]-=x[0,:,1].min()
    return x.astype(np.float32)

def resample_motion(x,t,horizon=88,fps=20):
    if len(t)!=len(x) or len(t)<2 or np.any(np.diff(t)<=0):raise ValueError('bad timestamps')
    new=np.arange(horizon)/fps
    y=np.stack([np.interp(new,t,x[:,j,c]) for j in range(24) for c in range(3)],axis=-1).reshape(horizon,24,3)
    return y.astype(np.float32),new<=t[-1]+1e-9

def prepare():
    if (RUN/'data.npz').exists():raise FileExistsError('Preparation already exists; refuse overwrite')
    formal=ROOT/'benchmark/umrm_final_636/items.jsonl'
    assert sha(formal)=='e037a788d7cade34477d1dc9bdbac8fadf2c4d3af066d71e675d2de0faac3c7f'
    rows=[json.loads(x) for x in formal.read_text().splitlines()]
    clips=json.loads((BASE/'clip_inputs_636.json').read_text())['items']
    refs={r['id']:r for r in json.loads((BASE/'pico_baseline_636.json').read_text())['items']}
    assert len(rows)==len(clips)==len(refs)==636
    by_id={r['qid']:r for r in rows};meta=[];targets=[];masks=[];durations=[]
    for c in clips:
        qid=c['id'];r=refs[qid];p=r['primary_recording']
        rec=next(v for v in r['recordings'] if v['path']==p)
        assert sha(p)==rec['sha256'] and sha(c['clip_path'])==r['clip_sha256']
        with np.load(p,allow_pickle=False) as d:
            names=d['body_joint_names'].tolist();x=d['raw_body_poses'][:,[names.index(n) for n in NAMES],:3]
            ns=d['raw_motion_timestamp_ns'].astype(np.int64);t=(ns-ns[0])*1e-9
        assert np.isfinite(x).all()
        y,m=resample_motion(canonical(x),t);targets.append(y);masks.append(m);durations.append(float(t[-1]))
        meta.append(dict(id=qid,src_link=by_id[qid].get('src_link'),src_id=by_id[qid].get('src_id'),clip_sha256=r['clip_sha256'],reference_sha256=rec['sha256']))
    split=build_split(meta)
    RUN.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(RUN/'data.npz',targets=np.stack(targets),masks=np.stack(masks),ids=[c['id'] for c in clips],durations=durations)
    write(RUN/'encoder_inputs.json',[encoder_record(c) for c in clips]);write(RUN/'split.json',split);write(RUN/'data_provenance.json',meta)
    report=dict(records=636,counts={s:list(split.values()).count(s) for s in sorted(set(split.values()))},benchmark_sha256=sha(formal),target_shape=list(np.stack(targets).shape),fps=20,horizon=88,joint_names=NAMES,split_policy='experimental known source + exact clip union; unknown source groups excluded from fit and primary evaluation',coordinates='initial-body axes, initial-frame floor, meters; forward -z',output='joint positions only; orientations not generated')
    write(RUN/'data_report.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':prepare()
