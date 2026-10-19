"""Mechanical application of README §2 gates for v1197aug (CPU). Writes runs/gates_aug.json."""
import json, os
from pathlib import Path
from common import *
from compare import cmp

A = lambda n: RUNS / f"{n}.json"; V = lambda n: V1197 / "runs" / f"{n}.json"
ex = lambda p: Path(p).exists()


def lo(r, m):
    return r["metrics"][m]["ci95"][0]


def summ(r, ms=("R@10", "orig_t2m_sim", "R@3_samecat", "static", "plan_acc_mean")):
    return {m: {"diff": round(r["metrics"][m]["diff"], 4), "ci95": [round(x, 4) for x in r["metrics"][m]["ci95"]],
                "p": round(r["metrics"][m]["wilcoxon_p"], 4), "A": round(r["metrics"][m]["A"], 4), "B": round(r["metrics"][m]["B"], 4)}
            for m in ms if m in r["metrics"]}


def main():
    G = {}
    # ---- confirmation of +0.077 (README §2.2)
    c = {}
    if ex(V("eval_head_opd_v2_s1_e2_greedy__head_AB_full_best")):
        r = cmp(str(V("eval_head_opd_v2_s1_e2_greedy__head_AB_full_best")) + ":H", str(V("eval_head_sft_student_greedy__head_AB_full_best")) + ":H")
        c["C1_seed1_VAL125_AB"] = {**summ(r), "pass": lo(r, "R@10") > 0}
        if ex(V("eval_kimodo_opd_v2_s1_e2_greedy")):
            c["C1_secondary_kimodo_TPH"] = summ(cmp(str(V("eval_kimodo_opd_v2_s1_e2_greedy")) + ":TPH", str(V("eval_kimodo_sft_student_greedy")) + ":TPH"))
        if ex(V("eval_head_opd_v2_s1_e2_greedy__head_A_full_best")):
            c["C1_secondary_A_full"] = summ(cmp(str(V("eval_head_opd_v2_s1_e2_greedy__head_A_full_best")) + ":H", str(V("eval_head_sft_student_greedy__head_A_full_best")) + ":H"))
    if ex(A("cmp_C2_AB_full")):
        r = rjson(A("cmp_C2_AB_full")); c["C2_sealed50_AB"] = {**summ(r), "n": r["n"], "n_sources": r["n_sources"], "pass": lo(r, "R@10") > 0}
        for f, nm in (("cmp_C2_A_full", "C2_secondary_A_full"), ("cmp_C2_kimodo_TPH", "C2_secondary_kimodo_TPH")):
            if ex(A(f)):
                c[nm] = summ(rjson(A(f)))
    if "C1_seed1_VAL125_AB" in c and "C2_sealed50_AB" in c:
        k = c["C1_seed1_VAL125_AB"]["pass"] + c["C2_sealed50_AB"]["pass"]
        c["verdict"] = {2: "reproduced", 1: "partially reproduced", 0: "not reproduced"}[k]
    G["confirm_+0.077"] = c
    # ---- G-T3
    if ex(A("gate_GT3_text")):
        t = rjson(A("gate_GT3_text")); kl = rjson(A("gate_GT2_opd_v3")) if ex(A("gate_GT2_opd_v3")) else {}
        G["G-T3"] = {"generated_MR_subset": t.get("subset_generated"), "human_MR_subset": t.get("subset_human"), "kl": kl,
                     "pass": (t.get("subset_generated") or {}).get("exec_gain", -1) >= 0.05}
    # ---- G4-v3
    if ex(A("eval_head_opd_v3_e1_greedy__head_AB_full_best")) and ex(A("eval_kimodo_opd_v3_e1_greedy")):
        r = cmp(str(A("eval_head_opd_v3_e1_greedy__head_AB_full_best")) + ":H", str(A("eval_head_sft_student_greedy__head_AB_full_best")) + ":H")
        ps = lambda n: rjson(A(n))["summary"]["plan_shift"]
        tv = {f: [ps("eval_kimodo_sft_student_greedy")[f]["tv_distance"], ps("eval_kimodo_opd_v3_e1_greedy")[f]["tv_distance"]] for f in ("direction", "duration")}
        ok = all(b - a <= 0.1 for a, b in tv.values())
        G["G4-v3"] = {"AB_full_primary": summ(r), "plan_tv_sft_vs_opd": tv, "tv_ok": ok, "pass": lo(r, "R@10") > 0 and ok,
                      "kimodo_TPH": summ(cmp(str(A("eval_kimodo_opd_v3_e1_greedy")) + ":TPH", str(A("eval_kimodo_sft_student_greedy")) + ":TPH")),
                      "A_full": summ(cmp(str(A("eval_head_opd_v3_e1_greedy__head_A_full_best")) + ":H", str(A("eval_head_sft_student_greedy__head_A_full_best")) + ":H"))
                      if ex(A("eval_head_opd_v3_e1_greedy__head_A_full_best")) else None}
        if ex(A("eval_head_opd_v3_e1_greedy__head_AB_full_best_visshift1")):
            G["G4-v3"]["AB_vs_mismatched_video(opd)"] = summ(cmp(str(A("eval_head_opd_v3_e1_greedy__head_AB_full_best")) + ":H",
                                                                str(A("eval_head_opd_v3_e1_greedy__head_AB_full_best_visshift1")) + ":H"))
    # ---- G4-v4 / H-v4 (README §4)
    if ex(A("eval_head_opd_v4_e1_greedy__head_AB_full_best")) and ex(A("eval_kimodo_opd_v4_e1_greedy")):
        r = cmp(str(A("eval_head_opd_v4_e1_greedy__head_AB_full_best")) + ":H", str(A("eval_head_sft_student_greedy__head_AB_full_best")) + ":H")
        ps = lambda n: rjson(A(n))["summary"]["plan_shift"]
        tv = {f: [ps("eval_kimodo_sft_student_greedy")[f]["tv_distance"], ps("eval_kimodo_opd_v4_e1_greedy")[f]["tv_distance"]] for f in ("direction", "duration")}
        ok = all(b_ - a_ <= 0.1 for a_, b_ in tv.values())
        G["G4-v4"] = {"AB_full_primary": summ(r), "plan_tv_sft_vs_opd": tv, "tv_ok": ok, "pass": lo(r, "R@10") > 0 and ok,
                      "kimodo_TPH": summ(cmp(str(A("eval_kimodo_opd_v4_e1_greedy")) + ":TPH", str(A("eval_kimodo_sft_student_greedy")) + ":TPH")),
                      "A_full": summ(cmp(str(A("eval_head_opd_v4_e1_greedy__head_A_full_best")) + ":H", str(A("eval_head_sft_student_greedy__head_A_full_best")) + ":H"))}
        if ex(A("eval_head_opd_v4_e1_greedy__head_AB_full_best_visshift1")):
            G["G4-v4"]["AB_vs_mismatched_video(opd)"] = summ(cmp(str(A("eval_head_opd_v4_e1_greedy__head_AB_full_best")) + ":H",
                                                                str(A("eval_head_opd_v4_e1_greedy__head_AB_full_best_visshift1")) + ":H"))
        import numpy as np
        from compare import per_item
        man = manifest(); rows = man["rows"]; hk = [k for k in man["val"] if rows[k].get("mental_reasoning") and rows[k].get("mr_src") != "generated"]
        P4, P3 = per_item(str(A("eval_head_opd_v4_e1_greedy__head_AB_full_best")) + ":H"), per_item(str(A("eval_head_opd_v3_e1_greedy__head_AB_full_best")) + ":H")
        PS = per_item(str(A("eval_head_sft_student_greedy__head_AB_full_best")) + ":H")
        src = np.array([rows[k]["source"] for k in hk]); us = np.unique(src); ix = {u: np.where(src == u)[0] for u in us}; rng = np.random.default_rng(0)
        def boot(d):
            bs = [d[np.concatenate([ix[u] for u in rng.choice(us, len(us))])].mean() for _ in range(4000)]
            return {"diff": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
        hv = {f"v4_minus_v3_{m}": boot(np.array([P4[k][m] - P3[k][m] for k in hk])) for m in ("R@10", "orig_t2m_sim")}
        hv.update({f"v4_minus_sft_{m}": boot(np.array([P4[k][m] - PS[k][m] for k in hk])) for m in ("R@10", "orig_t2m_sim")})
        G["H-v4"] = {"n": len(hk), "n_sources": len(us), **hv, "supported": hv["v4_minus_v3_R@10"]["ci95"][0] > 0}
    # ---- exploratory
    e = {}
    if ex(A("eval_kimodo_sft_student_greedy")):
        e["SFTaug_vs_SFTv1197_kimodo_TPH"] = summ(cmp(str(A("eval_kimodo_sft_student_greedy")) + ":TPH", str(V("eval_kimodo_sft_student_greedy")) + ":TPH"))
    if ex(A("eval_head_sft_student_greedy__head_AB_full_best")) and ex(A("eval_head_sft_student_greedy__head_A_full_best")):
        e["AB_vs_A(sft)"] = summ(cmp(str(A("eval_head_sft_student_greedy__head_AB_full_best")) + ":H", str(A("eval_head_sft_student_greedy__head_A_full_best")) + ":H"))
        if ex(A("eval_head_sft_student_greedy__head_AB_full_best_visshift1")):
            e["AB_vs_mismatched_video(sft)"] = summ(cmp(str(A("eval_head_sft_student_greedy__head_AB_full_best")) + ":H",
                                                       str(A("eval_head_sft_student_greedy__head_AB_full_best_visshift1")) + ":H"))
        L = [json.loads(l) for l in (RUNS / "head_AB_full/train.jsonl").read_text().splitlines()]; b = rjson(RUNS / "head_AB_full/result.json")["best_step"]
        e["AB_gates_at_selected_step"] = next({k: v for k, v in x.items() if k.startswith("gate")} for x in L if x["step"] == b)
    if ex(A("eval_head_opd_v3_half_greedy__head_AB_full_best")):
        e["OPDv3_half_vs_SFT_AB"] = summ(cmp(str(A("eval_head_opd_v3_half_greedy__head_AB_full_best")) + ":H", str(A("eval_head_sft_student_greedy__head_AB_full_best")) + ":H"))
    G["exploratory"] = e
    wjson(RUNS / "gates_aug.json", G); print(json.dumps(G, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
