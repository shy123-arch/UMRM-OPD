"""Mechanical application of README §6 (CPU). Writes runs/gates_v6.json."""
import json
from pathlib import Path
from common import *
from compare import cmp

A = lambda n: RUNS / f"{n}.json"; LL = EXP / "vt_lastlayer_v1197aug/runs"; ex = lambda p: Path(p).exists()
SFT = {"A36": LL / "eval_head_sft__A36.json", "AB": A("eval_head_sft_student_greedy__head_AB_full_best"), "A": A("eval_head_sft_student_greedy__head_A_full_best"),
       "K": A("eval_kimodo_sft_student_greedy")}
def files(n):
    return {"A36": LL / f"eval_head_{n}__A36.json", "AB": A(f"eval_head_{n}__head_AB_full_best"), "A": A(f"eval_head_{n}__head_A_full_best"), "K": A(f"eval_kimodo_{n}")}
def summ(r, ms=("R@10", "orig_t2m_sim", "R@3_samecat", "static", "plan_acc_mean")):
    return {m: {"A": round(r["metrics"][m]["A"], 4), "B": round(r["metrics"][m]["B"], 4), "diff": round(r["metrics"][m]["diff"], 4),
                "ci95": [round(x, 4) for x in r["metrics"][m]["ci95"]], "p": round(r["metrics"][m]["wilcoxon_p"], 4)} for m in ms if m in r["metrics"]}
def vs(n, base=SFT):
    f = files(n); out = {}
    for h in ("A36", "AB", "A", "K"):
        if ex(f[h]) and ex(base[h]):
            arm = ":TPH" if h == "K" else ":H"; out[h] = summ(cmp(str(f[h]) + arm, str(base[h]) + arm))
    return out
def plan(n):
    ps = lambda p: rjson(p)["summary"]["plan_shift"]; s, o = ps(SFT["K"]), ps(files(n)["K"])
    return {"tv": {f: [round(s[f]["tv_distance"], 3), round(o[f]["tv_distance"], 3)] for f in ("direction", "duration")},
            "top_dir": [round(s["direction"]["top_label_share"], 3), round(o["direction"]["top_label_share"], 3)]}
def main():
    G = {}
    sel = rjson(RUNS / "sft_teacher_8b/selection.json"); ss = rjson(RUNS / "sft_student/selection.json")
    G["G-T6"] = {"val_nll": {"teacher_8b": sel["val_nll"], "student_4b": ss["val_nll"], "selected": [sel["selected"], ss["selected"]]}}
    if ex(files("sft_teacher_8b_greedy")["K"]):
        G["G-T6"]["teacher_vs_student"] = vs("sft_teacher_8b_greedy")
    T = G["G-T6"].get("teacher_vs_student", {})
    for n in ("opd_v6", "opd_v6_re"):
        f = files(f"{n}_e1_greedy")
        if not (ex(f["A36"]) and ex(f["K"])):
            continue
        r = vs(f"{n}_e1_greedy"); p = plan(f"{n}_e1_greedy"); G[n] = {**r, "plan": p}
        G[n]["transfer_ratio"] = {h: round(r[h]["R@10"]["diff"] / T[h]["R@10"]["diff"], 3) for h in r if h in T and abs(T[h]["R@10"]["diff"]) > 1e-9}
        G[n]["args"] = {k: rjson(RUNS / n / "args.json").get(k) for k in ("n_train", "k", "lam_fwd", "kl_span", "teacher_base", "world_size")}
    if "opd_v6" in G:
        o = G["opd_v6"]; tv = o["plan"]["tv"]; top = o["plan"]["top_dir"]
        G["G4-v6"] = {"pass": o["A36"]["R@10"]["ci95"][0] > 0 and all(b - a <= 0.1 for a, b in tv.values()), "A36_R@10": o["A36"]["R@10"], "tv": tv}
        G["H-v6_no_collapse"] = {"direction_tv": tv["direction"], "top_share": top, "supported": tv["direction"][1] - tv["direction"][0] <= 0.1 and top[1] - top[0] <= 0.1}
        if "opd_v6_re" in G:
            G["v6_minus_v6_re"] = {h: summ(cmp(str(files("opd_v6_e1_greedy")[h]) + (":TPH" if h == "K" else ":H"), str(files("opd_v6_re_e1_greedy")[h]) + (":TPH" if h == "K" else ":H")))
                                   for h in ("A36", "K")}
    old = {}
    for n in ("opd_v3_e1_greedy", "opd_v4_e1_greedy", "opd_v5r_e1_greedy", "opd_v5o_e1_greedy", "opd_v5o_mix_e1_greedy"):
        if ex(LL / f"eval_head_{n}__A36.json"):
            old[n] = summ(cmp(str(LL / f"eval_head_{n}__A36.json") + ":H", str(SFT["A36"]) + ":H"), ("R@10", "orig_t2m_sim"))
    for n in ("v1197_opd_v2_e2_greedy", "v1197_opd_v2_s1_e2_greedy"):
        if ex(LL / f"eval_head_{n}__A36.json") and ex(LL / "eval_head_v1197_sft_student_greedy__A36.json"):
            old[n] = summ(cmp(str(LL / f"eval_head_{n}__A36.json") + ":H", str(LL / "eval_head_v1197_sft_student_greedy__A36.json") + ":H"), ("R@10", "orig_t2m_sim"))
    G["exploratory_old_OPD_through_A36"] = old
    wjson(RUNS / "gates_v6.json", G); print(json.dumps(G, indent=1, ensure_ascii=False)[:6000])
if __name__ == "__main__":
    main()
