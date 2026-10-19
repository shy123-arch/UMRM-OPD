"""Fetch pinned official TMR sources; import only checksum-verified model archive."""
import argparse
import os
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get('VHO_TMR_ASSETS', str(HERE / 'tmr_assets_20261012')))
REVISION = '6d74688730d15d43b0a755ce2b0e1f2d76138fc1'
ARCHIVE_MD5 = '7b6d8814f9c1ca972f62852ebb6c7a6f'
FILES = [
    'LICENSE', 'src/model/actor.py', 'src/model/text_encoder.py',
    'src/guofeats/__init__.py', 'src/guofeats/common/quaternion.py',
    'src/guofeats/common/skeleton.py', 'src/guofeats/motion_representation.py',
    'src/guofeats/paramUtil.py', 'src/guofeats/skeleton_example_h3d.npy',
    'stats/humanml3d/guoh3dfeats/mean.pt', 'stats/humanml3d/guoh3dfeats/std.pt',
    'configs/model/temos.yaml', 'configs/model/tmr.yaml',
    'configs/data/_base.yaml', 'configs/data/motion_loader/guoh3dfeats.yaml',
    'src/data/motion.py', 'prepare/download_pretrain_models.sh',
]


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch_sources():
    ASSETS.mkdir(exist_ok=True)
    journal = ASSETS/'sources.json'
    records = json.loads(journal.read_text()) if journal.exists() else {}
    for name in FILES:
        path = ASSETS/'official'/name
        if name in records:
            assert digest(path) == records[name]['sha256']
            continue
        if path.exists():
            raise FileExistsError(path)
        url = f'https://cdn.jsdelivr.net/gh/Mathux/TMR@{REVISION}/{name}'
        with urllib.request.urlopen(url, timeout=25) as response:
            data = response.read(4*1024*1024+1)
        if len(data)>4*1024*1024:
            raise ValueError('Source size bound exceeded')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:
            f.write(data)
        records[name] = {'url':url,'revision':REVISION,'bytes':len(data),'sha256':digest(path)}
        journal.write_text(json.dumps(records,indent=2))
        print(name,len(data),flush=True)


def import_archive(archive):
    archive = Path(archive)
    if digest(archive,'md5') != ARCHIVE_MD5:
        raise ValueError('Archive does not match official TMR published MD5; refusing extraction')
    dest = ASSETS/'pretrained'
    if dest.exists():
        raise FileExistsError(dest)
    with tarfile.open(archive) as tar:
        selected=[]
        for member in tar.getmembers():
            p=PurePosixPath(member.name)
            if p.is_absolute() or '..' in p.parts or member.issym() or member.islnk():
                raise ValueError('Unsafe archive entry')
            if member.isfile() and 'tmr_humanml3d_guoh3dfeats' in p.parts:
                if member.size>100*1024*1024:
                    raise ValueError('Unexpected per-file weight size')
                selected.append(member)
        if not selected or sum(m.size for m in selected)>500*1024*1024:
            raise ValueError('Unexpected model archive scope')
        dest.mkdir()
        for member in selected:
            path=dest/member.name
            path.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src, path.open('xb') as out:
                shutil.copyfileobj(src,out)
    (ASSETS/'archive_provenance.json').write_text(json.dumps({'archive_sha256':digest(archive),'official_md5':ARCHIVE_MD5,'source':'https://drive.google.com/uc?id=1n6kRb-d2gKsk8EXfFULFIpaUKYcnaYmm','files':{str(p.relative_to(dest)):digest(p) for p in dest.rglob('*') if p.is_file()}},indent=2))


def fetch_text_backbone():
    repo='distilbert/distilbert-base-uncased'
    revision='12040accade4e8a0f71eabdb258fecc2e7e948be'
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true',timeout=25) as response:
        meta=json.load(response)
    wanted={'config.json','tokenizer.json','tokenizer_config.json','vocab.txt','model.safetensors'}
    files=[f for f in meta['siblings'] if f['rfilename'] in wanted]
    if len(files)!=5 or sum(f['size'] for f in files)>270*1024*1024:
        raise ValueError('Unexpected DistilBERT download scope')
    dest=ASSETS/'distilbert-base-uncased';dest.mkdir(exist_ok=True)
    rows=[]
    for info in files:
        path=dest/info['rfilename']
        if not path.exists():
            url=f'https://huggingface.co/{repo}/resolve/{revision}/{info["rfilename"]}'
            temporary=path.with_suffix(path.suffix+'.partial')
            if temporary.exists():
                raise FileExistsError(temporary)
            with urllib.request.urlopen(url,timeout=60) as response, temporary.open('xb') as out:
                size=0
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk:break
                    size+=len(chunk)
                    if size>info['size']:raise ValueError('Unexpected file size')
                    out.write(chunk)
            if temporary.stat().st_size!=info['size']:raise ValueError('Incomplete download')
            temporary.rename(path)
        if 'lfs' in info:
            assert digest(path)==info['lfs']['sha256']
        else:
            data=path.read_bytes()
            assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==info['blobId']
        rows.append({'name':path.name,'sha256':digest(path),'size':path.stat().st_size})
        print('DistilBERT',path.name,path.stat().st_size,flush=True)
    (ASSETS/'text_provenance.json').write_text(json.dumps({'repo':repo,'revision':revision,'files':rows},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--fetch-sources',action='store_true')
    p.add_argument('--archive',type=Path)
    p.add_argument('--fetch-text',action='store_true')
    args=p.parse_args()
    if args.fetch_sources:
        fetch_sources()
    if args.archive:
        import_archive(args.archive)
    if args.fetch_text:
        fetch_text_backbone()
