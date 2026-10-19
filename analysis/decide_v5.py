"""Mechanical application of README §5 (CPU). Writes runs/gates_v5.json."""
import json
from pathlib import Path
import numpy as np
from common import *
from compare import cmp, per_item

A = lambda n: RUNS / f"{n}.json"; ex = lambda n: A(n).exists()
SFT_AB, SFT_A, SFT_K = "eval_head_sft_student_greedy__head_AB_full_best", "eval_head_sft_student_greedy__head_A_full_best", "eval_kimodo_sft_student_greedy"


def summ(r, ms=("R@10", "orig_t2m_sim", "R@3_samecat", "static", "plan_acc_mean")):
    return {m: {"A": round(r["metrics"][m]["A"], 4), "B": round(r["metrics"][m]["B"], 4), "diff": round(r["metrics"][m]["diff"], 4),
                "ci95": [round(x, 4) for x in r["metrics"][m]["ci95"]], "p": round(r["metrics"][m]["wilcoxon_p"], 4)} for m in ms if m in r["metrics"]}


def arm(n):
    ab, a, k = f"eval_head_{n}_e1_greedy__head_AB_full_best", f"eval_head_{n}_e1_greedy__head_A_full_best", f"eval_kimodo_{n}_e1_greedy"
    if not (ex(ab) and ex(k)):
        return None
    r = cmp(str(A(ab)) + ":H", str(A(SFT_AB)) + ":H"); ps = lambda x: rjson(A(x))["summary"]["plan_shift"]
    tv = {f: [round(ps(SFT_K)[f]["tv_distance"], 3), round(ps(k)[f]["tv_distance"], 3)] for f in ("direction", "duration")}
    top = [round(ps(SFT_K)["direction"]["top_label_share"], 3), round(ps(k)["direction"]["top_label_share"], 3)]
    out = {"AB_primary": summ(r), "pass_R@10": r["metrics"]["R@10"]["ci95"][0] > 0, "plan_tv_sft_vs_opd": tv, "direction_top_share_sft_vs_opd": top,
           "tv_ok": all(b - a <= 0.1 for a, b in tv.values()), "kimodo_TPH": summ(cmp(str(A(k)) + ":TPH", str(A(SFT_K)) + ":TPH"))}
    if ex(a):
        out["A_full"] = summ(cmp(str(A(a)) + ":H", str(A(SFT_A)) + ":H"))
    out["pass"] = out["pass_R@10"] and out["tv_ok"]
    return out


def sub_boot(a, b, keys, rows, m="R@10"):
    P, Q = per_item(str(A(a)) + ":H"), per_item(str(A(b)) + ":H"); d = np.array([P[k][m] - Q[k][m] for k in keys])
    src = np.array([rows[k]["source"] for k in keys]); us = np.unique(src); ix = {u: np.where(src == u)[0] for u in us}; rng = np.random.default_rng(0)
    bs = [d[np.concatenate([ix[u] for u in rng.choice(us, len(us))])].mean() for _ in range(4000)]
    return {"n": len(keys), "diff": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}


def main():
    man = manifest(); rows = man["rows"]; G = {}
    for t in ("r", "o"):
        g = {}
        if ex(f"gate_GT2_opd_v5{t}"):
            g["kl"] = rjson(A(f"gate_GT2_opd_v5{t}"))
        if ex(f"gate_GT5{t}_text"):
            x = rjson(A(f"gate_GT5{t}_text")); g["text_human_MR_subset"] = x.get("subset_human"); g["text_all"] = {k: x[k] for k in ("exec_gain", "reaction_gain", "n")}
        G[f"G-T5_teacher_{t}"] = g
    if ex("eval_head_sft_teacher_o_zgreedy__head_AB_full_best"):
        G["teacher_o_ceiling_AB"] = {k: round(v, 4) for k, v in rjson(A("eval_head_sft_teacher_o_zgreedy__head_AB_full_best"))["summary"]["H"].items() if k in ("R@10", "orig_t2m_sim", "R@3_samecat")}
    for n in ("opd_v5o", "opd_v5r", "opd_v5o_mix"):
        G[n] = arm(n)
    o = G.get("opd_v5o")
    if o:
        G["G4-v5"] = {"pass": o["pass"], "R@10": o["AB_primary"]["R@10"], "tv_ok": o["tv_ok"]}
        tv = o["plan_tv_sft_vs_opd"]["direction"]; top = o["direction_top_share_sft_vs_opd"]
        G["H-v5a_no_collapse_under_reverse_KL"] = {"direction_tv": tv, "top_share": top, "supported": (tv[1] - tv[0] <= 0.1) and (top[1] - top[0] <= 0.1)}
        hk = [k for k in man["val"] if rows[k].get("mental_reasoning") and rows[k].get("mr_src") != "generated"]
        if ex("eval_head_opd_v4_e1_greedy__head_AB_full_best"):
            b = sub_boot("eval_head_opd_v5o_e1_greedy__head_AB_full_best", "eval_head_opd_v4_e1_greedy__head_AB_full_best", hk, rows)
            G["H-v5b_v5o_minus_v4_on_human_MR_VAL"] = {**b, "supported": b["ci95"][0] > 0}
        if G.get("opd_v5o_mix"):
            G["secondary_mix_minus_v5o_AB"] = summ(cmp(str(A("eval_head_opd_v5o_mix_e1_greedy__head_AB_full_best")) + ":H", str(A("eval_head_opd_v5o_e1_greedy__head_AB_full_best")) + ":H"))
    wjson(RUNS / "gates_v5.json", G); print(json.dumps(G, indent=1, ensure_ascii=False)[:5000])


if __name__ == "__main__":
    main()
