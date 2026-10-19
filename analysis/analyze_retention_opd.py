"""CPU-only ER19 paired retention and difference-in-differences analysis.

No fitting, checkpoint selection, gate rewriting, or model calls. Missing arms are
explicitly pending. Outputs can be refreshed after each preregistered stage.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ARMS = {'A': 'er19_sft_base', 'B': 'er19_opd_base', 'C': 'er19_sft_ego', 'D': 'er19_opd_ego'}


def bootstrap(d, groups, n=4000):
    d, groups = np.asarray(d, float), np.asarray(groups)
    assert d.ndim == 1 and len(d) == len(groups) and np.isfinite(d).all() and len(d)
    us = np.unique(groups)
    sums = np.array([d[groups == u].sum() for u in us])
    sizes = np.array([(groups == u).sum() for u in us])
    rng = np.random.default_rng(0)
    draw = rng.integers(len(us), size=(n, len(us)))
    means = sums[draw].sum(1) / sizes[draw].sum(1)
    return {'n': len(d), 'n_clusters': len(us), 'diff': float(d.mean()),
            'ci95': np.percentile(means, [2.5, 97.5]).tolist(),
            'increased': int((d > 1e-10).sum()), 'decreased': int((d < -1e-10).sum()),
            'unchanged': int((abs(d) <= 1e-10).sum())}


def main():
    ap = argparse.ArgumentParser()
    for name in ('experiment', 'lastlayer', 'data', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args(); used = {}

    def read(p):
        b = p.read_bytes(); used[str(p)] = hashlib.sha256(b).hexdigest()
        return json.loads(b)

    runs = a.experiment / 'runs'
    m = read(a.data / 'data_manifest.json'); keys = m['val']; rows = m['rows']
    assert len(keys) == 125 and all(rows[k]['group'] == 'val' for k in keys)
    groups = [rows[k]['source'] for k in keys]
    expected = {x['id']: x for x in read(runs / 'ego_manifest.json')['probe']}
    assert len(expected) == 1235
    r = {'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'ER19; VAL and fixed external probes; one training seed',
         'probe_comparisons': {}, 'sft_nll': {}, 'motion': {}, 'plan': {}, 'opd_logs': {}, 'pending': []}
    probes = {}
    names = ['base', 'sft', 'pt_c', 'sft_ptc', *ARMS.values(), 'er19_sft_base_half', 'er19_sft_ego_half']
    for name in names:
        p = runs / f'probe_{name}.json'
        if not p.exists():
            r['pending'].append(p.name); continue
        probe = read(p)
        assert probe['n_missing_cache'] == 0 and set(probe['items']) == set(expected)
        for k, x in probe['items'].items():
            assert x['gold'] == expected[k]['gold'] and x['pred'] == int(np.argmax(x['scores']))
            assert x['video_uid'] == expected[k]['video_uid']
        probes[name] = probe['items']
    r['probe_accuracy'] = {}
    sets = sorted({x['set'] for x in expected.values()}) + ['egotom_pooled']
    pairs = [('er19_sft_ego_half', 'pt_c'), ('er19_sft_ego', 'pt_c'), ('er19_sft_ego', 'sft_ptc'),
             ('er19_sft_base', 'sft'), ('er19_sft_ego_half', 'er19_sft_base_half'),
             ('er19_sft_ego', 'er19_sft_base'), ('er19_opd_base', 'er19_sft_base'),
             ('er19_opd_ego', 'er19_sft_ego')]
    for st in sets:
        ks = [k for k, x in expected.items() if x['set'] == st or (st == 'egotom_pooled' and x['set'].startswith('egotom_'))]
        pg = [expected[k]['video_uid'] for k in ks]
        vals = {name: np.array([p[k]['pred'] == p[k]['gold'] for k in ks], float) for name, p in probes.items()}
        r['probe_accuracy'][st] = {name: {'n': len(v), 'correct': int(v.sum()), 'acc': float(v.mean())} for name, v in vals.items()}
        for an, bn in pairs:
            if an in vals and bn in vals:
                r['probe_comparisons'].setdefault(an + '_vs_' + bn, {})[st] = bootstrap(vals[an] - vals[bn], pg)
        if all(n in vals for n in ARMS.values()):
            d = (vals[ARMS['D']] - vals[ARMS['C']]) - (vals[ARMS['B']] - vals[ARMS['A']])
            r['probe_comparisons'].setdefault('interaction', {})[st] = bootstrap(d, pg)
    for arm in ('base', 'ego'):
        p = runs / f'er19_sft_{arm}/log.jsonl'
        if p.exists():
            used[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
            r['sft_nll'][arm] = [x for x in map(json.loads, p.read_text().splitlines()) if 'val_nll_per_tok' in x]
        p = runs / f'er19_opd_{arm}/log.jsonl'
        if p.exists():
            used[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
            ls = [x for x in map(json.loads, p.read_text().splitlines()) if 'rev_kl' in x]
            # Resume may replay logged steps newer than its checkpoint; keep the final occurrence.
            ls = sorted({x['step']: x for x in ls}.values(), key=lambda x: x['step'])
            fields = ['rev_kl', 'fwd_kl', 'student_entropy', 'sample_strict', 'distilled_tokens']
            r['opd_logs'][arm] = {'logged_steps': len(ls), 'first5': {f: float(np.mean([x[f] for x in ls[:5]])) for f in fields},
                                 'last5': {f: float(np.mean([x[f] for x in ls[-5:]])) for f in fields}} if ls else {'logged_steps': 0}
    evals, predictions = {}, {}
    for name in [*ARMS.values(), 'er19_teacher_r']:
        p = a.lastlayer / 'runs' / f'eval_head_{name}__A36.json'
        if p.exists():
            e = read(p); assert set(e['per_gen']) == set(keys)
            assert all(len(e['per_gen'][k]) == 3 for k in keys)
            evals[name] = e['per_gen']
        else:
            r['pending'].append(p.name)
        p = runs / f'{name}.shard0.jsonl'
        if p.exists():
            used[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
            rec = [json.loads(x) for x in p.read_text().splitlines() if x]
            if len(rec) == 125 and {x['id'] for x in rec} == set(keys):
                predictions[name] = {x['id']: x['samples'][0] for x in rec}
    def metric(e, field):
        return np.array([np.mean([float(v['orig_m2t_rank'] <= 10) if field == 'R@10' else
                                 float(v['orig_m2t_rank_samecat'] <= 3) if field == 'R@3_samecat' else v[field]
                                 for v in e[k]]) for k in keys])
    for field in ['R@10', 'orig_t2m_sim', 'R@3_samecat', 'mpjpe_m']:
        vals = {name: metric(e, field) for name, e in evals.items()}
        result = {'absolute': {n: float(v.mean()) for n, v in vals.items()}}
        for an, bn in [('B', 'A'), ('D', 'C'), ('D', 'B'), ('C', 'A')]:
            if ARMS[an] in vals and ARMS[bn] in vals:
                result[an + '-' + bn] = bootstrap(vals[ARMS[an]] - vals[ARMS[bn]], groups)
        if all(n in vals for n in ARMS.values()):
            result['interaction'] = bootstrap((vals[ARMS['D']] - vals[ARMS['C']]) - (vals[ARMS['B']] - vals[ARMS['A']]), groups)
        if 'er19_teacher_r' in vals:
            human = np.array([bool(rows[k].get('mental_reasoning')) and not str(rows[k].get('mr_src', 'human')).startswith('generated') for k in keys])
            for arm in ('A', 'C'):
                if ARMS[arm] in vals:
                    d = vals['er19_teacher_r'] - vals[ARMS[arm]]
                    result['teacher-' + arm] = bootstrap(d, groups)
                    result['teacher-' + arm + '_human_reason_val'] = bootstrap(d[human], np.array(groups)[human])
        r['motion'][field] = result
    for name, pred in predictions.items():
        fields = ['locomotion', 'direction', 'turn', 'posture', 'duration']
        q = {'strict_rate': float(np.mean([bool(pred[k]['strict']) for k in keys])), 'fields': {}}
        for f in fields:
            ct = Counter(str(rows[k]['plan'][f]) for k in keys)
            cp = Counter(str(pred[k]['plan'].get(f, '__missing__')) for k in keys)
            q['fields'][f] = {'accuracy': float(np.mean([pred[k]['plan'].get(f) == rows[k]['plan'][f] for k in keys])),
                             'counts': dict(cp), 'truth_counts': dict(ct),
                             'tv': sum(abs(cp[v] - ct[v]) for v in cp.keys() | ct.keys()) / (2 * len(keys))}
        if 'er19_teacher_r' in predictions:
            t = predictions['er19_teacher_r']
            q['teacher_exact_agreement'] = {
                f: float(np.mean([bool(pred[k].get(f)) and pred[k].get(f) == t[k].get(f)
                                  and (f != 'plan' or all(x in pred[k][f] for x in fields)) for k in keys]))
                for f in ['reaction', 'exec', 'plan']}
        r['plan'][name] = q
    primary = r['motion'].get('R@10', {})
    if 'interaction' in primary:
        r['primary_support'] = bool(primary['interaction']['ci95'][0] > 0 and primary['D-C']['ci95'][0] > 0)
    else:
        r['primary_support'] = None
    r['input_sha256'] = used
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'metrics.json').write_text(json.dumps(r, indent=2, ensure_ascii=False) + '\n')
    lines = ['# ER19 能力保留与匹配 OPD', '', f"更新 UTC：{r['created_utc']}", '',
             '缺失产物：' + str(len(r['pending'])) + '；尚未生成的臂不作零分处理。', '',
             '| 探针 | PT-C | 旧 SFT | 旧 PT-C→SFT | A | C | B | D |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for st, vals in r['probe_accuracy'].items():
        cols = [f"{vals[n]['acc']:.2%}" if n in vals else '待完成' for n in ['pt_c', 'sft', 'sft_ptc', ARMS['A'], ARMS['C'], ARMS['B'], ARMS['D']]]
        lines.append('| ' + st + ' | ' + ' | '.join(cols) + ' |')
    lines += ['', 'A/C 为相同低学习率 SFT；B/D 分别在 A/C 上接受相同固定教师 OPD。', '', '## A36 R@10', '']
    for name, v in primary.get('absolute', {}).items():
        lines.append(f'- {name}：{v:.4f}')
    for name in ['B-A', 'D-C', 'D-B', 'interaction']:
        if name in primary:
            v = primary[name]; lo, hi = v['ci95']
            lines.append(f"- {name}：{v['diff']:+.4f}，95% CI [{lo:+.4f}, {hi:+.4f}]")
    lines += ['', '主要比较是否支持：' + ('待完成' if r['primary_support'] is None else str(r['primary_support'])), '',
              '完整的半程保留、配对统计、教师优势、NLL、计划分布、KL/熵及输入哈希见 [metrics.json](metrics.json)。', '',
              '来源/视频 UID 聚类 bootstrap 4,000 次。一个训练 seed；次要分析未多重校正。低学习率是否保留能力必须结合探针与任务质量判断。', '',
              '本轮尚未证明能力保留时，不能以 OPD 负结果否定充分保留能力后的条件假设。']
    (a.out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'pending': r['pending'], 'primary_support': r['primary_support']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
