"""CPU-only descriptive analysis of existing §9 artifacts; never launches models.

Writes a separate report directory, without changing registered gates or run outputs.
New comparisons/subgroups are exploratory; cluster-bootstrap intervals are unadjusted.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def read(p):
    return json.loads(p.read_text())


def lines(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def boot(a, b, groups, n=4000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    groups = np.asarray(groups)
    us = np.unique(groups)
    ix = [np.flatnonzero(groups == u) for u in us]
    d = a - b
    rng = np.random.default_rng(0)
    bs = [d[np.concatenate([ix[j] for j in rng.integers(len(us), size=len(us))])].mean() for _ in range(n)]
    return dict(n=len(d), n_clusters=len(us), A=float(a.mean()), B=float(b.mean()),
                diff=float(d.mean()), ci95=np.percentile(bs, [2.5, 97.5]).tolist(),
                increased=int((d > 1e-10).sum()), decreased=int((d < -1e-10).sum()),
                unchanged=int((abs(d) <= 1e-10).sum()))


def margin(x):
    return x['scores'][x['gold']] - max(v for i, v in enumerate(x['scores']) if i != x['gold'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', type=Path, required=True)
    ap.add_argument('--lastlayer', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    run = a.experiment / 'runs'
    a.out.mkdir(parents=True, exist_ok=True)
    used = []

    def artifact(p):
        used.append(p)
        return read(p)

    m = artifact(run / 'ego_manifest.json')
    probes = {n: artifact(run / f'probe_{n}.json') for n in ['base', 'sft', 'pt_c', 'sft_ptc']
              if (run / f'probe_{n}.json').exists()}
    expected = {x['id']: x for x in m['probe']}
    for n, p in probes.items():
        assert set(p['items']) == set(expected), (n, 'incomplete probe')
        for k, v in p['items'].items():
            assert v['gold'] == expected[k]['gold']
            assert len(v['scores']) == len(expected[k]['options'])
            assert v['pred'] == int(np.argmax(v['scores']))
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Existing completed artifacts only; no GPU inference. New diagnostics are exploratory.',
              'probe_pending': [n for n in ['base', 'sft', 'pt_c', 'sft_ptc'] if n not in probes],
              'registered_gates': artifact(run / 'gates_e.json')}
    report['probe_acc'] = {n: p['acc'] for n, p in probes.items()}
    comparisons = {}
    cases = {}
    pairs = [('pt_c', 'base'), ('pt_c', 'sft'), ('sft', 'base'), ('sft_ptc', 'pt_c'), ('sft_ptc', 'sft')]
    for an, bn in pairs:
        if an not in probes or bn not in probes:
            continue
        A, B = probes[an]['items'], probes[bn]['items']
        cs = {}
        for st in sorted({x['set'] for x in expected.values()}) + ['egotom_pooled']:
            ks = [k for k, x in expected.items() if x['set'] == st or (st == 'egotom_pooled' and x['set'].startswith('egotom_'))]
            aa = [A[k]['pred'] == A[k]['gold'] for k in ks]
            bb = [B[k]['pred'] == B[k]['gold'] for k in ks]
            cs[st] = boot(aa, bb, [expected[k]['video_uid'] for k in ks])
            cs[st]['correct_A'] = sum(aa)
            cs[st]['correct_B'] = sum(bb)
            cs[st]['both_correct'] = sum(x and y for x, y in zip(aa, bb))
            cs[st]['both_wrong'] = sum(not x and not y for x, y in zip(aa, bb))
            cs[st]['prediction_changed'] = sum(A[k]['pred'] != B[k]['pred'] for k in ks)
            cs[st]['mean_gold_margin_A'] = float(np.mean([margin(A[k]) for k in ks]))
            cs[st]['mean_gold_margin_B'] = float(np.mean([margin(B[k]) for k in ks]))
            if st == 'egonormia_verified':
                for typ, want_a, want_b in [('fixed', True, False), ('broken', False, True)]:
                    picked = [k for k, x, y in zip(ks, aa, bb) if x == want_a and y == want_b]
                    cases[f'{an}_vs_{bn}_{typ}'] = [
                        dict(id=k, gold=expected[k]['options'][A[k]['gold']],
                             prediction_A=expected[k]['options'][A[k]['pred']],
                             prediction_B=expected[k]['options'][B[k]['pred']],
                             margin_A=margin(A[k]), margin_B=margin(B[k])) for k in picked]
        comparisons[f'{an}_vs_{bn}'] = cs
    report['probe_comparisons'] = comparisons

    train_uids = {x['video_uid'] for x in m['train']}
    eval_uids = set(m['eval_video_uids'])
    report['data_audit'] = dict(manifest_stats=m['stats'],
                              train_eval_video_overlap=sorted(train_uids & eval_uids),
                              unique_train_ids=len({x['id'] for x in m['train']}))
    assert not train_uids & eval_uids
    norm = lambda s: re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()
    train_next = {norm(x['target'].split('<next>', 1)[1]) for x in m['train'] if '<next>' in x['target']}
    report['data_audit']['egonormia_gold_exact_training_next_match'] = sum(
        norm(x['options'][x['gold']]) in train_next for x in m['probe'] if x['set'] == 'egonormia_verified')
    report['data_audit']['window_seconds'] = {}
    for st in sorted({x['set'] for x in m['probe']}):
        lens = [x['end']-x['start'] for x in m['probe'] if x['set'] == st]
        report['data_audit']['window_seconds'][st] = dict(median=float(np.median(lens)), minimum=min(lens), maximum=max(lens))

    report['pretraining'] = artifact(run / 'pt_c/args.json')
    ptlogs = []
    for fn in ['log.jsonl', 'log_rank1.jsonl', 'log_rank2.jsonl']:
        p = run / 'pt_c' / fn
        used.append(p)
        ptlogs += lines(p)
    report['pretraining_loss_windows'] = {}
    for src in ['goalstep', 'egonormia', 'egore_why']:
        fld = 'loss_' + src
        report['pretraining_loss_windows'][src] = {
            label: float(np.mean([r[fld] for r in ptlogs if lo <= r['step'] <= hi and fld in r]))
            for label, lo, hi in [('steps_10_100', 10, 100), ('steps_2310_2400', 2310, 2400)]}
    report['pretraining_loss_note'] = 'Unweighted means of logged per-rank window means, not exact sample-weighted losses; training only.'
    sft_logs = {}
    report['sft_validation'] = {}
    for n in ['sft_student', 'sft_ptc']:
        p = run / n / 'log.jsonl'
        used.append(p)
        sft_logs[n] = lines(p)
        report['sft_validation'][n] = [r for r in sft_logs[n] if 'val_nll_per_tok' in r]

    manifest = artifact(a.data / 'data_manifest.json')
    ks = manifest['val']
    rows = {k: manifest['rows'][k] for k in ks}
    assert len(ks) == 125 and all(r['group'] == 'val' for r in rows.values())
    base_eval = artifact(a.lastlayer / 'runs/eval_head_sft__A36.json')
    pt_eval = artifact(a.lastlayer / 'runs/eval_head_sft_ptc_greedy__A36.json')
    assert set(base_eval['per_gen']) == set(pt_eval['per_gen']) == set(ks)
    sources = [rows[k]['source'] for k in ks]

    def values(e, metric):
        return {k: float(np.mean([v['orig_m2t_rank'] <= 10 if metric == 'R@10' else v[metric]
                                 for v in e['per_gen'][k]])) for k in ks}

    report['motion'] = {}
    av, bv = {}, {}
    for metric in ['R@10', 'orig_t2m_sim', 'orig_m2t_rank', 'mpjpe_m', 'root_relative_mpjpe_m', 'end_err_m', 'static', 'body_speed']:
        av[metric], bv[metric] = values(pt_eval, metric), values(base_eval, metric)
        report['motion'][metric] = boot([av[metric][k] for k in ks], [bv[metric][k] for k in ks], sources)
        assert abs(report['motion'][metric]['A']-pt_eval['summary']['H'][metric]) < 1e-8
        assert abs(report['motion'][metric]['B']-base_eval['summary']['H'][metric]) < 1e-8
    assert abs(report['motion']['R@10']['diff'] - report['registered_gates']['G-E1']['R@10']['diff']) < 5e-5
    report['motion']['R@10']['hits_A'] = sum(v['orig_m2t_rank'] <= 10 for vs in pt_eval['per_gen'].values() for v in vs)
    report['motion']['R@10']['hits_B'] = sum(v['orig_m2t_rank'] <= 10 for vs in base_eval['per_gen'].values() for v in vs)
    report['motion_by_category_descriptive'] = {}
    for cat in sorted({r['category'] for r in rows.values()}):
        ids = [k for k in ks if rows[k]['category'] == cat]
        report['motion_by_category_descriptive'][cat] = dict(n=len(ids),
            A=float(np.mean([av['R@10'][k] for k in ids])), B=float(np.mean([bv['R@10'][k] for k in ids])),
            t2m_diff=float(np.mean([av['orig_t2m_sim'][k]-bv['orig_t2m_sim'][k] for k in ids])))

    pred = {}
    for n in ['sft_student_greedy', 'sft_ptc_greedy']:
        p = run / f'{n}.shard0.jsonl'
        used.append(p)
        pred[n] = {x['id']: x['samples'][0] for x in lines(p)}
        assert set(pred[n]) == set(ks)
    A, B = pred['sft_ptc_greedy'], pred['sft_student_greedy']
    report['text'] = {f'{fld}_exact_same': sum(A[k][fld] == B[k][fld] for k in ks) for fld in ['reaction', 'exec', 'plan']}
    report['plan_fields'] = {}
    for f in ['locomotion', 'direction', 'turn', 'posture', 'duration']:
        r = boot([A[k]['plan'].get(f) == rows[k]['plan'][f] for k in ks],
                 [B[k]['plan'].get(f) == rows[k]['plan'][f] for k in ks], sources)
        r['counts_A'] = dict(Counter(A[k]['plan'].get(f) for k in ks))
        r['counts_B'] = dict(Counter(B[k]['plan'].get(f) for k in ks))
        r['counts_gold'] = dict(Counter(rows[k]['plan'][f] for k in ks))
        report['plan_fields'][f] = r
    cases['umrm_val_all'] = [dict(id=k, source=rows[k]['source'], category=rows[k]['category'],
        reference_reaction=rows[k]['reaction'], reference_exec=rows[k]['exec'],
        old_exec=B[k]['exec'], new_exec=A[k]['exec'],
        old_plan=B[k]['plan'], new_plan=A[k]['plan'], gold_plan=rows[k]['plan'],
        r10_diff=av['R@10'][k]-bv['R@10'][k],
        t2m_diff=av['orig_t2m_sim'][k]-bv['orig_t2m_sim'][k]) for k in ks]

    report['provenance'] = [dict(path=str(p), bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in used]
    (a.out/'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    (a.out/'cases.json').write_text(json.dumps(cases, indent=2, ensure_ascii=False)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for src in ['goalstep', 'egonormia', 'egore_why']:
        by_step = defaultdict(list)
        for r in ptlogs:
            if 'loss_'+src in r:
                by_step[r['step']].append(r['loss_'+src])
        ss = sorted(by_step)
        yy = np.array([np.mean(by_step[s]) for s in ss])
        if len(yy) >= 10:
            axes[0].plot(ss[9:], np.convolve(yy, np.ones(10)/10, mode='valid'), label=src)
    axes[0].set(xlabel='Optimizer step', ylabel='Training token NLL', title='Stage A: smoothed training loss')
    axes[0].legend(fontsize=8)
    for n, label in [('sft_student','Base → SFT'), ('sft_ptc','Ego PT → SFT')]:
        ls = report['sft_validation'][n]
        axes[1].plot([x['epoch'] for x in ls], [x['val_nll_per_tok'] for x in ls], marker='o', label=label)
    axes[1].set(xlabel='Completed SFT epoch', ylabel='UMRM VAL token NLL', title='Stage B: validation loss', xticks=[0,1,2])
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out/'learning_curves.png', dpi=180)
    plt.close(fig)
    print(json.dumps({k:report[k] for k in ['probe_pending','probe_comparisons','pretraining_loss_windows','sft_validation','motion','motion_by_category_descriptive','plan_fields','text','data_audit']},ensure_ascii=False))


if __name__ == '__main__':
    main()
