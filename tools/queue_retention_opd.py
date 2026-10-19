"""ER19: inspect by default; --execute runs the preregistered, detached-compatible queue.

No downloads, no old-result replacement, no automatic SFT restarts. Uses the existing
GPU reservation protocol and stops at the first failed stage. See docs/prereg/ego_retention_opd_20261019.md.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
EXP = ROOT / 'umrm_model/experiments/vt_head_opd_v1197aug'
LL = EXP.parent / 'vt_lastlayer_v1197aug'
RUNS = EXP / 'runs'
CONTROL = RUNS / 'er19_control'
LOGS = EXP / 'logs/er19'
PIPE = REPO / 'pipeline'
DATA = ROOT / 'release/umrm_vho_data_v1197aug'
CACHE = Path('/tmp/vho1197aug_cache')
EGO_CACHE = Path('/tmp/vho_ego_cache')
MODEL = Path('/tmp/vho_models/Qwen3-VL-4B-Instruct')
ENVS = Path('/mnt/workspace/Wilson/miniconda3/envs')
Q = ENVS / 'qwen35_532/bin/python'
H = ENVS / 'hero/bin/python'
K = ENVS / 'kimodo/bin/python'
TEACHER = RUNS / 'sft_teacher_r/epoch1'
PT = RUNS / 'pt_c/epoch1'
TEACHER_PRED = 'er19_teacher_r'


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def env_for(gpu=None, ego=False, hero=False):
    e = os.environ.copy()
    for key in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'VHO_UNSEAL_SEALED', 'VHO_LIMIT'):
        e.pop(key, None)
    e.update(VHO_RUNS=str(RUNS), VHO_DATA=str(DATA), VHO_QWEN=str(MODEL),
             VHO_CACHE=str(EGO_CACHE if ego else CACHE), VHO_ZMODE='mr',
             VHO_SPLIT='val', VHO_NFRAMES='16', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if gpu is not None:
        e['CUDA_VISIBLE_DEVICES'] = str(gpu)
    if hero:
        e.pop('PYTORCH_CUDA_ALLOC_CONF', None)
        e['LD_LIBRARY_PATH'] = str(ENVS / 'hero/lib/python3.10/site-packages/torch/lib') + ':/usr/local/cuda/lib64'
    return e


def sft_cmd(arm, smoke=False):
    name = f'er19_sft_{arm}' if not smoke else 'er19_smoke_sft'
    cmd = [Q, '-u', PIPE / 'sft.py', '--mode', 'student', '--task', 'answer', '--out', name,
           '--lr', '2e-5', '--epochs', '3', '--stop-after', '1', '--accum', '8', '--seed', '0',
           '--checkpoint-steps', '1' if smoke else '104']
    if arm == 'ego' or smoke:
        cmd += ['--init-adapter', PT]
    if smoke:
        cmd += ['--limit', '16']
    return [str(x) for x in cmd]


def opd_cmd(arm, smoke=False):
    name = f'er19_opd_{arm}' if not smoke else 'er19_smoke_opd'
    student = RUNS / (f'er19_sft_{arm}/epoch1' if not smoke else 'sft_student/epoch1')
    cmd = [Q, '-u', PIPE / 'opd.py', '--out', name, '--student', student, '--teacher', TEACHER,
           '--epochs', '1', '--k', '8', '--lr', '5e-5', '--accum', '4',
           '--lam_fwd', '0', '--kl-span', 're', '--seed', '0', '--mr-filter', 'human',
           '--gen-videos', '1', '--fwd-chunk', '2', '--save-every', '25', '--skip-e0']
    if smoke:
        cmd += ['--limit', '4']
    return [str(x) for x in cmd]


def preflight():
    m = read(DATA / 'data_manifest.json'); old = read(RUNS / 'data_manifest.json')
    assert m['train'] == old['train'] and m['val'] == old['val']
    assert len(m['train']) == 1652 and len(m['val']) == 125
    assert len(set(m['train'] + m['val'])) == 1777
    assert all(m['rows'][k]['target'] == old['rows'][k]['target'] for k in m['train'] + m['val'])
    assert all(m['rows'][k]['group'] == split for split in ('train', 'val') for k in m[split])
    human = [k for k in m['train'] if m['rows'][k].get('mental_reasoning')
             and not str(m['rows'][k].get('mr_src', 'human')).startswith('generated')]
    old_human = [k for k in old['train'] if old['rows'][k].get('mental_reasoning')
                 and not str(old['rows'][k].get('mr_src', 'human')).startswith('generated')]
    assert human == old_human and len(human) == 832
    assert all(m['rows'][k]['mental_reasoning'] == old['rows'][k]['mental_reasoning'] for k in human)
    ego = read(RUNS / 'ego_manifest.json')
    assert len(ego['probe']) == 1235 and len({x['id'] for x in ego['probe']}) == 1235
    assert not ({x['video_uid'] for x in ego['train']} & set(ego['eval_video_uids']))
    required = [Q, H, K, MODEL / 'config.json', MODEL / 'model.safetensors.index.json',
                TEACHER / 'adapter_model.safetensors', TEACHER / 'adapter_config.json',
                PT / 'adapter_model.safetensors', PT / 'adapter_config.json',
                LL / 'runs/head_A36/best.pt', LL / 'textenc.py', LL / 'head.py', LL / 'eval_head.py']
    required += [CACHE / 'video' / f'{k}.pt' for k in m['train'] + m['val']]
    required += [EGO_CACHE / 'video' / f"{x['id']}.pt" for x in ego['probe']]
    weights = sorted(set(read(MODEL / 'model.safetensors.index.json')['weight_map'].values()))
    required += [MODEL / n for n in weights]
    missing = [str(p) for p in required if not p.is_file()]
    assert not missing, f'missing required assets: {missing[:20]}'
    assert read(TEACHER.parent / 'args.json')['zmode'] == 'mr'
    assert read(TEACHER.parent / 'selection.json')['selected'] == 'epoch1'
    teacher_infer = read(RUNS / 'sft_teacher_r_zgreedy.args.json')
    assert teacher_infer['prompt'] == 'teacher' and teacher_infer['adapter'] == 'runs/sft_teacher_r/epoch1'
    assert teacher_infer['k'] == 1 and teacher_infer['split'] == 'val' and teacher_infer['limit'] == 0
    pred = [json.loads(l) for l in (RUNS / 'sft_teacher_r_zgreedy.shard0.jsonl').read_text().splitlines() if l]
    assert len(pred) == 125 and {r['id'] for r in pred} == set(m['val'])
    files = [DATA / 'data_manifest.json', RUNS / 'data_manifest.json', RUNS / 'ego_manifest.json',
             TEACHER / 'adapter_model.safetensors', TEACHER / 'adapter_config.json',
             PT / 'adapter_model.safetensors', PT / 'adapter_config.json', LL / 'runs/head_A36/best.pt',
             RUNS / 'sft_teacher_r_zgreedy.shard0.jsonl',
             RUNS / 'sft_teacher_r_zgreedy.args.json', TEACHER.parent / 'args.json',
             REPO / 'docs/prereg/ego_retention_opd_20261019.md', Path(__file__),
             REPO / 'analysis/analyze_retention_opd.py']
    files += [PIPE / f for f in ('common.py', 'vt_io.py', 'sft.py', 'opd.py', 'ego_probe.py', 'infer.py')]
    files += [LL / f for f in ('common.py', 'textenc.py', 'head.py', 'eval_common.py', 'eval_head.py')]
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), n_train=1652, n_val=125,
                n_opd=832, n_probe=1235, split_sha256=m['split_sha256'],
                opd_ids_sha256=hashlib.sha256('\n'.join(human).encode()).hexdigest(),
                input_sha256={str(p): sha(p) for p in files}, model_shards={n: (MODEL / n).stat().st_size for n in weights},
                commands={f'{kind}_{arm}': fun(arm) for kind, fun in [('sft', sft_cmd), ('opd', opd_cmd)] for arm in ('base', 'ego')},
                resource_limit=dict(own_parallel_gpus=2, shared_registered_cap=3, estimate_gpu_hours=[20, 30]))


def status(stage, **extra):
    r = dict(time_utc=datetime.now(timezone.utc).isoformat(), stage=stage, **extra)
    print(json.dumps(r, ensure_ascii=False), flush=True)
    write(CONTROL / 'status.json', r)


@contextmanager
def gpu_lease(name, free_gb):
    reservation = RUNS / 'opd_inuse' / name
    deadline = time.monotonic() + 6 * 3600
    gpu = None
    while gpu is None:
        with (RUNS / '.acquire.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            reserved = []
            for p in (RUNS / 'opd_inuse').iterdir():
                fields = p.read_text().strip().split(':', 1)
                assert len(fields) == 2, f'unknown reservation: {p}'
                reserved += [int(x) for x in fields[1].split()]
            if len(reserved) < 3:
                q = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
                candidates = []
                for line in q.stdout.strip().splitlines():
                    i, used, total, util = [int(x.strip()) for x in line.split(',')]
                    free = (total - used) / 1024
                    if i not in reserved and free >= free_gb:
                        candidates.append((util // 25, -free, i))
                if candidates:
                    gpu = min(candidates)[2]
                    with reservation.open('x') as f:
                        f.write(f'{name}:{gpu}\n')
        if gpu is None:
            if time.monotonic() > deadline:
                raise TimeoutError(f'{name}: GPU wait exceeded six hours')
            time.sleep(60)
    print(f'{name}: acquired GPU {gpu}', flush=True)
    try:
        yield gpu
    finally:
        reservation.unlink()


def call(name, cmd, gpu=None, ego=False, hero=False, cwd=EXP):
    # Unique logs per invocation; preserve earlier failures and partial outputs.
    stamp = time.time_ns()
    with (LOGS / f'{name}.{stamp}.log').open('x') as log:
        log.write(json.dumps([str(x) for x in cmd]) + '\n'); log.flush()
        subprocess.run([str(x) for x in cmd], cwd=cwd, env=env_for(gpu, ego, hero),
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def validate_probe(name):
    p = read(RUNS / f'probe_{name}.json'); expected = {x['id'] for x in read(RUNS / 'ego_manifest.json')['probe']}
    assert p['n_missing_cache'] == 0 and set(p['items']) == expected


def probe(name, adapter, gpu):
    if not (RUNS / f'probe_{name}.json').exists():
        call('probe_' + name, [Q, '-u', PIPE / 'ego_probe.py', '--name', name, '--adapter', adapter], gpu, ego=True)
    validate_probe(name)


def link_once(src, dst):
    if dst.is_symlink() or dst.exists():
        assert dst.resolve() == src.resolve(), f'existing unrelated output {dst}'
    else:
        dst.symlink_to(src)


def readout(name, gpu, adapter=None):
    if adapter is not None and not (RUNS / f'{name}.args.json').exists():
        call('infer_' + name, [Q, '-u', PIPE / 'infer.py', '--adapter', adapter, '--name', name], gpu)
    rows = [json.loads(x) for x in (RUNS / f'{name}.shard0.jsonl').read_text().splitlines() if x]
    assert len(rows) == 125 and {r['id'] for r in rows} == set(read(DATA / 'data_manifest.json')['val'])
    link_once(RUNS / f'{name}.shard0.jsonl', LL / 'runs' / f'{name}.shard0.jsonl')
    if not (CACHE / 'textfeat' / f'pred_{name}.meta.json').exists():
        call('text_' + name, [Q, '-u', LL / 'textenc.py', '--corpus', 'pred', '--name', name], gpu, cwd=LL)
    npz = LL / 'runs/gen' / f'{name}__A36.npz'
    if not npz.exists():
        call('head_' + name, [H, '-u', LL / 'head.py', 'gen', '--ckpt', 'runs/head_A36/best.pt',
                             '--corpus', f'pred_{name}', '--out', f'gen/{name}__A36.npz', '--samples', '3'], gpu, hero=True, cwd=LL)
    dst = LL / 'runs' / f'eval_head_{name}__A36.json'
    if not dst.exists():
        call('eval_' + name, [K, '-u', LL / 'eval_head.py', '--npz', npz, '--name', name + '__A36'], gpu, cwd=LL)
    assert set(read(dst)['per_gen']) == set(read(DATA / 'data_manifest.json')['val'])


def smoke():
    if (CONTROL / 'smoke.done.json').exists():
        return
    with gpu_lease('er19_smoke', 42) as gpu:
        for name, cmd in [('er19_smoke_sft', sft_cmd('ego', True)), ('er19_smoke_opd', opd_cmd('base', True))]:
            if not (RUNS / name / 'selection.json').exists():
                call(name, cmd, gpu)
            assert read(RUNS / name / 'selection.json')['selected'] == 'epoch1'
        assert (RUNS / 'er19_smoke_sft/step1/adapter_model.safetensors').is_file()
        selection = read(RUNS / 'er19_smoke_sft/selection.json')
        assert set(selection['val_nll']) == {'1'}, 'diagnostic checkpoint entered model selection'
    write(CONTROL / 'smoke.done.json', {'passed': True})


def train_sft(arm):
    name = f'er19_sft_{arm}'
    with gpu_lease(name, 24) as gpu:
        if not (RUNS / name / 'selection.json').exists():
            if (RUNS / name / 'log.jsonl').exists():
                raise RuntimeError(f'{name}: incomplete SFT kept for inspection; no automatic restart')
            call(name, sft_cmd(arm), gpu)
        assert read(RUNS / name / 'selection.json')['selected'] == 'epoch1'


def retention(arm):
    name = f'er19_sft_{arm}'
    with gpu_lease(name + '_eval', 36) as gpu:
        probe(name + '_half', RUNS / name / 'step104', gpu)
        probe(name, RUNS / name / 'epoch1', gpu)
        readout(name, gpu, RUNS / name / 'epoch1')


def teacher_readout():
    link_once(RUNS / 'sft_teacher_r_zgreedy.shard0.jsonl', RUNS / f'{TEACHER_PRED}.shard0.jsonl')
    with gpu_lease('er19_teacher_eval', 36) as gpu:
        readout(TEACHER_PRED, gpu)


def train_opd(arm):
    name = f'er19_opd_{arm}'
    with gpu_lease(name, 42) as gpu:
        if not (RUNS / name / 'selection.json').exists():
            cmd = opd_cmd(arm)
            if (RUNS / name / 'log.jsonl').exists():
                if not (RUNS / name / 'resume.pt').exists():
                    raise RuntimeError(f'{name}: interrupted before first resume checkpoint; outputs preserved')
                cmd += ['--resume']
            call(name, cmd, gpu)
        assert read(RUNS / name / 'selection.json')['selected'] == 'epoch1'
        probe(name, RUNS / name / 'epoch1', gpu)
        readout(name, gpu, RUNS / name / 'epoch1')


def paired(fn):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fn, arm) for arm in ('base', 'ego')]
        errors = []
        for f in futures:
            try:
                f.result()
            except Exception as e:
                errors.append(e)
        if errors:
            raise RuntimeError('; '.join(str(e) for e in errors)) from errors[0]


def analyze():
    call('analysis', [Q, REPO / 'analysis/analyze_retention_opd.py', '--experiment', EXP, '--lastlayer', LL,
                      '--data', DATA, '--out', CONTROL / 'analysis'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--smoke-only', action='store_true', help='with --execute, stop after GPU smoke')
    a = ap.parse_args()
    CONTROL.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    with (CONTROL / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        p = preflight()
        frozen = CONTROL / 'execution_inputs.json'
        if a.execute and frozen.exists():
            assert read(frozen)['input_sha256'] == p['input_sha256'], 'inputs/code changed since execution; review before resuming'
        write(CONTROL / 'preflight.json', p)
        if not a.execute:
            print(json.dumps({'preflight': 'passed', 'gpu_started': False, 'plan': str(CONTROL / 'preflight.json')}, indent=2))
            return
        if not frozen.exists():
            write(frozen, p)
        try:
            status('gpu_smoke'); smoke()
            if a.smoke_only:
                status('smoke_complete'); return
            status('sft'); paired(train_sft)
            status('retention_probes_and_A36'); paired(retention)
            status('fixed_teacher_A36'); teacher_readout()
            analyze()
            # No result-based checkpoint/teacher switching: same predeclared epoch1 and teacher for both arms.
            status('matched_opd'); paired(train_opd)
            analyze()
            status('complete'); write(CONTROL / 'done.json', {'completed_utc': datetime.now(timezone.utc).isoformat()})
        except Exception as e:
            status('failed', error=str(e)); raise


if __name__ == '__main__':
    main()
