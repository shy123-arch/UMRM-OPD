"""Paired comparison of two VAL eval files (route-6 or this experiment), per item = mean over seeds/samples.
Source-cluster bootstrap 95% CI (resampling video sources) + item-level Wilcoxon, same as route6 compare.py.
Spec: <eval json path or name in runs/>:<arm>   (own-head files: arm H). Usage: compare.py A:TPH B:TPH [--json out]"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

METS = ["R@10", "orig_t2m_sim", "R@3_samecat", "orig_m2t_rank", "orig_cos", "end_err_m", "mpjpe_m", "root_relative_mpjpe_m", "static"]
DEC = ["reaction_vs_human_cos", "exec_vs_gt3p_cos", "plan_acc_mean"]


def per_item(spec):
    p, arm = spec.rsplit(":", 1); p = Path(p) if Path(p).exists() else RUNS / f"{p}.json"
    r = rjson(p); out = {}
    f = {"R@10": lambda x: x["orig_m2t_rank"] <= 10, "R@3_samecat": lambda x: x["orig_m2t_rank_samecat"] <= 3}
    for key, L in r["per_gen"].items():
        k, a_ = key.split("|") if "|" in key else (key, "H")
        if a_ == arm:
            out[k] = {m: float(np.mean([f[m](x) if m in f else x[m] for x in L])) for m in METS}
    for k, d in r.get("decision", {}).items():
        if k in out:
            out[k].update({m: d[m] for m in DEC if m in d})
    return out


def cmp(a, b, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed); man = manifest(); A, B = per_item(a), per_item(b); ks = sorted(set(A) & set(B))
    src = np.array([man["rows"][k]["source"] for k in ks]); us = np.unique(src); idx = {s: np.where(src == s)[0] for s in us}
    res = {"A": a, "B": b, "n": len(ks), "n_sources": len(us), "metrics": {}}
    for m in METS + DEC:
        if not all(m in A[k] and m in B[k] for k in ks):
            continue
        d = np.array([A[k][m] - B[k][m] for k in ks]); bs = []
        for _ in range(n_boot):
            ii = np.concatenate([idx[s] for s in rng.choice(us, len(us))]); bs.append(d[ii].mean())
        try:
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        except ValueError:
            p = 1.0
        res["metrics"][m] = {"A": float(np.mean([A[k][m] for k in ks])), "B": float(np.mean([B[k][m] for k in ks])), "diff": float(d.mean()),
                             "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "wilcoxon_p": p}
    return res


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]; r = cmp(args[0], args[1])
    print(f"A={r['A']}  B={r['B']}  n={r['n']} sources={r['n_sources']}")
    for m, v in r["metrics"].items():
        flag = "*" if v["ci95"][0] > 0 or v["ci95"][1] < 0 else " "
        print(f" {flag} {m:24s} A {v['A']:.3f}  B {v['B']:.3f}  diff {v['diff']:+.3f}  CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  p={v['wilcoxon_p']:.3f}")
    if "--json" in sys.argv:
        wjson(Path(sys.argv[sys.argv.index("--json") + 1]), r)
