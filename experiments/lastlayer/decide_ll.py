"""Mechanical application of README §3 (CPU). Writes runs/gates_ll.json."""
import json
from pathlib import Path
from common import *
from compare import cmp

E = lambda n: str(RUNS / f"eval_head_{n}.json") + ":H"
ARMS = ("A4", "A36", "S36", "V36")


def verdict(r, m="R@10"):
    lo, hi = r["metrics"][m]["ci95"]
    return "better" if lo > 0 else ("worse" if hi < 0 else "n.s.")


def summ(r):
    return {m: {"A": round(r["metrics"][m]["A"], 4), "B": round(r["metrics"][m]["B"], 4), "diff": round(r["metrics"][m]["diff"], 4),
                "ci95": [round(x, 4) for x in r["metrics"][m]["ci95"]], "p": round(r["metrics"][m]["wilcoxon_p"], 4)}
            for m in ("R@10", "orig_t2m_sim", "R@3_samecat", "orig_m2t_rank") if m in r["metrics"]}


def main():
    G = {"P1_V36_vs_A4": None, "decomposition": {}, "sanity_vs_train_mean": {}, "hml3d_selfcheck": {}, "exploratory_oracle": {}, "arms": {}}
    have = {a: Path(E("sft__" + a)[:-2]).exists() for a in ARMS}
    for a in ARMS:
        if have[a]:
            x = rjson(RUNS / f"eval_head_sft__{a}.json")["summary"]["H"]; G["arms"][a] = {k: round(x[k], 4) for k in ("R@10", "orig_t2m_sim", "R@3_samecat", "median_rank", "mpjpe_m", "static")}
            r = cmp(E("sft__" + a), str(RUNS / "eval_base_train_mean.json") + ":H"); G["sanity_vs_train_mean"][a] = {**summ(r), "verdict": verdict(r)}
    if have["V36"] and have["A4"]:
        r = cmp(E("sft__V36"), E("sft__A4")); G["P1_V36_vs_A4"] = {**summ(r), "verdict_R@10": verdict(r), "verdict_t2m": verdict(r, "orig_t2m_sim")}
    for nm, a, b in (("D1_layer(A36-A4)", "A36", "A4"), ("D2_sft_weights(S36-A36)", "S36", "A36"), ("D3_video_context(V36-S36)", "V36", "S36")):
        if have[a] and have[b]:
            r = cmp(E("sft__" + a), E("sft__" + b)); G["decomposition"][nm] = {**summ(r), "verdict_R@10": verdict(r), "verdict_t2m": verdict(r, "orig_t2m_sim")}
    for a in ("A4", "A36", "S36"):
        p = RUNS / f"hml3d_rprec_{a}.json"
        if p.exists():
            x = rjson(p); G["hml3d_selfcheck"][a] = {"gen_R@1/2/3": x["gen_R@1/2/3"], "gt_R@3": x["gt_vqpath_R@1/2/3"][2], "ratio_R@3": round(x["G-B_ratio_R@3"], 3)}
    for a in ARMS:
        p = RUNS / f"eval_head_oracle__{a}.json"
        if p.exists():
            x = rjson(p)["summary"]["H"]; G["exploratory_oracle"][a] = {k: round(x[k], 4) for k in ("R@10", "orig_t2m_sim", "R@3_samecat")}
    for a in ARMS:
        p = RUNS / f"head_{a}/result.json"
        if p.exists():
            G["arms"].setdefault(a, {})["finetune_best_step"] = rjson(p)["best_step"]
        p = RUNS / (f"head_pre_{a}/result.json" if a in ("A36", "S36") else "nonexistent")
        if p.exists():
            G["arms"][a]["pretrain_best_val_ce"] = round(rjson(p)["best_val_sum_ce"], 4)
    ref = EXP / "vt_head_opd_v1197aug/runs/head_A_full/result.json"
    if (RUNS / "head_A4/result.json").exists() and ref.exists():
        G["code_consistency_A4_vs_aug_head_A_full"] = {"this_dir": rjson(RUNS / "head_A4/result.json")["best_val_sum_ce"], "aug": rjson(ref)["best_val_sum_ce"]}
    wjson(RUNS / "gates_ll.json", G); print(json.dumps(G, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
