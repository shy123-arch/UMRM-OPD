"""Mechanical application of README §8 (CPU). Writes runs/gates_r.json (whatever parts exist so far).
Part 1: reasoning as the student's OUTPUT (R-CoT primary, R-obs, R-MT) vs SFT-aug greedy, A36 head.
Part 2: G-T8 teacher gate for the R-chain privileged teacher (stronger AND plan accuracy not raised) -> OPD v8 (G4-v8)."""
import json, random, re
from pathlib import Path
import numpy as np
from common import *
from compare import cmp, per_item

LL = EXP / "vt_lastlayer_v1197aug/runs"; SFT_A36 = LL / "eval_head_sft__A36.json"; SFT_PRED = "sft_student_greedy"
ex = lambda p: Path(p).exists()
STOP = set("i the a an to and my so of it that is in on me for this with as be was am at they their them i'm are there".split())
cw = lambda s: set(re.findall(r"[a-z]+", (s or "").lower())) - STOP


def summ(r, ms=("R@10", "orig_t2m_sim", "R@3_samecat", "orig_m2t_rank")):
    return {m: {"A": round(r["metrics"][m]["A"], 4), "B": round(r["metrics"][m]["B"], 4), "diff": round(r["metrics"][m]["diff"], 4),
                "ci95": [round(x, 4) for x in r["metrics"][m]["ci95"]], "p": round(r["metrics"][m]["wilcoxon_p"], 4)} for m in ms if m in r["metrics"]}


def preds(name):
    out = {}
    for l in open(RUNS / f"{name}.shard0.jsonl"):
        r = json.loads(l); out[r["id"]] = r["samples"][0]
    return out


def boot(a, b, keys, rows, n=4000, seed=0):
    """paired source-cluster bootstrap of mean(a - b) over keys."""
    rng = np.random.default_rng(seed); src = np.array([rows[k]["source"] for k in keys]); us = np.unique(src)
    idx = {s: np.where(src == s)[0] for s in us}; d = np.array([a[k] - b[k] for k in keys])
    bs = [d[np.concatenate([idx[s] for s in rng.choice(us, len(us))])].mean() for _ in range(n)]
    return {"A": round(float(np.mean([a[k] for k in keys])), 4), "B": round(float(np.mean([b[k] for k in keys])), 4), "diff": round(float(d.mean()), 4),
            "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], "n": len(keys)}


def tv(pred, keys, rows, f):
    P = [pred[k]["plan"].get(f) for k in keys]; G = [rows[k]["plan"][f] for k in keys]; cats = set(P) | set(G)
    return round(0.5 * sum(abs(P.count(c) - G.count(c)) / len(keys) for c in cats), 3)


def arm(name, rows, keys, sft_p, first_tag=None):
    e = LL / f"eval_head_{name}__A36.json"
    if not (ex(e) and ex(RUNS / f"{name}.shard0.jsonl")):
        return None
    p = preds(name); pa = {k: plan_acc(p[k]["plan"], rows[k]["plan"]) for k in keys}; sa = {k: plan_acc(sft_p[k]["plan"], rows[k]["plan"]) for k in keys}
    out = {"A36_vs_SFT": summ(cmp(str(e) + ":H", str(SFT_A36) + ":H")), "plan_acc_vs_SFT": boot(pa, sa, keys, rows),
           "strict_rate": round(float(np.mean([bool(p[k]["strict"]) for k in keys])), 4),
           "plan_tv_vs_truth": {f: {"arm": tv(p, keys, rows, f), "sft": tv(sft_p, keys, rows, f)} for f in ("direction", "duration")}}
    if first_tag:  # consistency of the model's own reasoning line with its own reaction (vs a random other reaction)
        line = lambda raw: next((l[len(first_tag) + 2:].strip() for l in raw.splitlines() if l.strip().lower().startswith(f"<{first_tag}>")), "")
        rs = {k: line(p[k]["raw"]) for k in keys}; ov = lambda a, b: len(cw(a) & cw(b)) / max(1, len(cw(b)))
        rng = random.Random(0); sh = keys[:]; rng.shuffle(sh)
        out["reason_line_rate"] = round(float(np.mean([bool(rs[k]) for k in keys])), 4)
        out["consistency_own_minus_random"] = round(float(np.mean([ov(rs[k], p[k]["reaction"]) - ov(rs[k], p[j]["reaction"]) for k, j in zip(keys, sh)])), 4)
        hum = [k for k in keys if rows[k].get("mr_src") != "generated"]
        out["reason_vs_human_reason_overlap"] = round(float(np.mean([ov(rs[k], rows[k]["mental_reasoning"]) for k in hum])), 4)
    return out


def main():
    man = manifest(); rows = man["rows"]; keys = list(man["val"]); sft_p = preds(SFT_PRED); G = {"rule": "README §8"}
    for nm, tag, g in (("sft_rcot_greedy", "reason", "G-R1"), ("sft_robs_greedy", "observe", "G-R2"), ("sft_rmt_greedy", None, "G-R3")):
        r = arm(nm, rows, keys, sft_p, tag)
        if r:
            G[nm] = r; x = r["A36_vs_SFT"]["R@10"]; G[g] = {"pass": x["ci95"][0] > 0, "R@10": x}
    t = arm("sft_teacher_chain_zgreedy", rows, keys, sft_p)
    if t:
        x = t["A36_vs_SFT"]["R@10"]; pa = t["plan_acc_vs_SFT"]
        stronger = x["ci95"][0] > 0; no_leak = pa["diff"] <= 0.02 and pa["ci95"][0] <= 0
        hum = [k for k in keys if rows[k].get("mr_src") != "generated"]
        A = {k: v["R@10"] for k, v in per_item(str(LL / "eval_head_sft_teacher_chain_zgreedy__A36.json") + ":H").items()}
        B = {k: v["R@10"] for k, v in per_item(str(SFT_A36) + ":H").items()}
        tp = preds("sft_teacher_chain_zgreedy")
        G["sft_teacher_chain_zgreedy"] = t
        G["G-T8"] = {"pass": bool(stronger and no_leak), "stronger": bool(stronger), "plan_not_raised": bool(no_leak), "R@10": x, "plan_acc": pa,
                     "descriptive_human_reason_subset": {"R@10": boot(A, B, hum, rows),
                                                         "plan_acc": boot({k: plan_acc(tp[k]["plan"], rows[k]["plan"]) for k in hum},
                                                                          {k: plan_acc(sft_p[k]["plan"], rows[k]["plan"]) for k in hum}, hum, rows)}}
    o = arm("opd_v8_e1_greedy", rows, keys, sft_p)
    if o:
        x = o["A36_vs_SFT"]["R@10"]; tvd = o["plan_tv_vs_truth"]
        G["opd_v8_e1_greedy"] = o
        G["G4-v8"] = {"pass": bool(x["ci95"][0] > 0 and all(tvd[f]["arm"] - tvd[f]["sft"] <= 0.1 for f in tvd)), "R@10": x, "plan_tv": tvd}
    wjson(RUNS / "gates_r.json", G); print(json.dumps({k: v for k, v in G.items() if k.startswith("G")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
