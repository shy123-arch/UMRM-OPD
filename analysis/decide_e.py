"""Mechanical application of README §9 (CPU). Writes runs/gates_e.json with whatever parts exist so far.
G-E1 (primary): sft_ptc greedy vs SFT-aug greedy, A36 head R@10 (source-cluster bootstrap CI lower bound > 0).
G-E0 (diagnostic): probe accuracy pt_c vs base (and all pairs), paired bootstrap over video_uid."""
import json
from pathlib import Path
import numpy as np
from common import *
import decide_r as DR

ex = lambda p: Path(p).exists()


def probe_cmp(a, b, n=4000, seed=0):
    A, B = rjson(RUNS / f"probe_{a}.json")["items"], rjson(RUNS / f"probe_{b}.json")["items"]; out = {}
    for st in sorted({v["set"] for v in A.values()}):
        ks = [k for k in A if A[k]["set"] == st and k in B]
        d = np.array([float(A[k]["pred"] == A[k]["gold"]) - float(B[k]["pred"] == B[k]["gold"]) for k in ks])
        vu = np.array([A[k]["video_uid"] for k in ks]); us = np.unique(vu); idx = {u: np.where(vu == u)[0] for u in us}
        rng = np.random.default_rng(seed); bs = [d[np.concatenate([idx[u] for u in rng.choice(us, len(us))])].mean() for _ in range(n)]
        out[st] = {"n": len(ks), "A": round(float(np.mean([A[k]["pred"] == A[k]["gold"] for k in ks])), 4),
                   "B": round(float(np.mean([B[k]["pred"] == B[k]["gold"] for k in ks])), 4), "diff": round(float(d.mean()), 4),
                   "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
    return out


def main():
    man = manifest(); rows = man["rows"]; keys = list(man["val"]); G = {"rule": "README §9"}
    sp = DR.preds(DR.SFT_PRED); r = DR.arm("sft_ptc_greedy", rows, keys, sp)
    if r:
        G["sft_ptc_greedy"] = r; x = r["A36_vs_SFT"]["R@10"]; G["G-E1"] = {"pass": bool(x["ci95"][0] > 0), "R@10": x}
        G["val_nll"] = {"sft_ptc": rjson(RUNS / "sft_ptc/selection.json")["val_nll"], "sft_aug": rjson(RUNS / "sft_student/selection.json")["val_nll"]}
    P = {n: rjson(RUNS / f"probe_{n}.json")["acc"] for n in ("base", "sft", "pt_c", "sft_ptc") if ex(RUNS / f"probe_{n}.json")}
    if P:
        G["probe_acc"] = P
    if ex(RUNS / "probe_pt_c.json") and ex(RUNS / "probe_base.json"):
        G["G-E0_pt_c_vs_base"] = probe_cmp("pt_c", "base")
    for a, b in (("sft_ptc", "sft"), ("sft", "base"), ("sft_ptc", "base")):
        if ex(RUNS / f"probe_{a}.json") and ex(RUNS / f"probe_{b}.json"):
            G[f"probe_{a}_vs_{b}"] = probe_cmp(a, b)
    if ex(RUNS / "pt_c/log.jsonl"):
        L = [json.loads(l) for l in open(RUNS / "pt_c/log.jsonl")]
        G["pt_c_loss"] = {"first": L[0] if L else None, "last": L[-1] if L else None, "args": rjson(RUNS / "pt_c/args.json")}
    wjson(RUNS / "gates_e.json", G); print(json.dumps({k: v for k, v in G.items() if k.startswith(("G-", "probe_acc"))}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
