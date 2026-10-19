"""Applies the pre-registered gates of README §2.3 mechanically to the eval files (CPU). Writes runs/gates.json."""
import json
from pathlib import Path
from common import *
from compare import cmp

E = lambda n: str(RUNS / f"{n}.json")


def ci_pos(r, m):
    return r["metrics"][m]["ci95"][0] > 0


def summ(r, ms=("R@10", "orig_t2m_sim", "R@3_samecat", "static", "plan_acc_mean")):
    return {m: {"diff": round(r["metrics"][m]["diff"], 4), "ci95": [round(x, 4) for x in r["metrics"][m]["ci95"]], "p": round(r["metrics"][m]["wilcoxon_p"], 4)}
            for m in ms if m in r["metrics"]}


def main():
    G = {}
    f = lambda n: Path(E(n)).exists()
    if f("eval_head_sft_student_greedy__head_AB_full_best") and f("eval_head_sft_student_greedy__head_A_full_best"):
        r = cmp(E("eval_head_sft_student_greedy__head_AB_full_best") + ":H", E("eval_head_sft_student_greedy__head_A_full_best") + ":H")
        G["G-V"] = {"AB_vs_A": summ(r), "pass": ci_pos(r, "R@10") or ci_pos(r, "orig_t2m_sim")}
        if f("eval_head_sft_student_greedy__head_AB_full_best_visshift1"):
            c = cmp(E("eval_head_sft_student_greedy__head_AB_full_best") + ":H", E("eval_head_sft_student_greedy__head_AB_full_best_visshift1") + ":H")
            G["G-V"]["AB_vs_AB_mismatched_video"] = summ(c); G["G-V"]["uses_item_specific_video"] = ci_pos(c, "R@10") or ci_pos(c, "orig_t2m_sim")
        ck = RUNS / "head_AB_full/train.jsonl"
        if ck.exists():
            L = [json.loads(l) for l in ck.read_text().splitlines()]; b = rjson(RUNS / "head_AB_full/result.json")["best_step"]
            G["G-V"]["gates_at_selected_step"] = next({k: v for k, v in x.items() if k.startswith("gate")} for x in L if x["step"] == b)
            a0 = json.loads((RUNS / "head_A_full/train.jsonl").read_text().splitlines()[0])["val_sum_ce"]
            G["G-V"]["init_equivalence_step0_val_ce(AB, A)"] = [L[0]["val_sum_ce"], a0]
    if f("eval_head_sft_student_greedy__head_ABfrz_full_best"):  # EXPLORATORY, not a pre-registered gate
        r = cmp(E("eval_head_sft_student_greedy__head_ABfrz_full_best") + ":H", E("eval_head_sft_student_greedy__head_A_full_best") + ":H")
        G["exploratory_ABfrz"] = {"ABfrz_vs_A": summ(r)}
        if f("eval_head_sft_student_greedy__head_ABfrz_full_best_visshift1"):
            G["exploratory_ABfrz"]["ABfrz_vs_mismatched_video"] = summ(cmp(E("eval_head_sft_student_greedy__head_ABfrz_full_best") + ":H",
                                                                             E("eval_head_sft_student_greedy__head_ABfrz_full_best_visshift1") + ":H"))
        L = [json.loads(l) for l in (RUNS / "head_ABfrz_full/train.jsonl").read_text().splitlines()]; b = rjson(RUNS / "head_ABfrz_full/result.json")["best_step"]
        G["exploratory_ABfrz"]["gates_at_selected_step"] = next({k: v for k, v in x.items() if k.startswith("gate")} for x in L if x["step"] == b)
    if f("eval_head_sft_student_greedy__head_A_full_best") and f("eval_head_sft_student_greedy__head_A_cap_best"):
        r = cmp(E("eval_head_sft_student_greedy__head_A_full_best") + ":H", E("eval_head_sft_student_greedy__head_A_cap_best") + ":H")
        G["E1-confirm"] = {"full_vs_cap": summ(r), "pass": ci_pos(r, "R@10") or ci_pos(r, "orig_t2m_sim")}
    if f("gate_GT2_opd_v2") and f("gate_GT2_text"):
        k, t = rjson(E("gate_GT2_opd_v2")), rjson(E("gate_GT2_text"))
        G["G-T2"] = {"kl": k, "text": t, "pass": k["ratio"] >= 1.5 and t["exec_gain"] >= 0.05}
    if f("eval_kimodo_opd_v2_e2_greedy") and f("eval_kimodo_sft_student_greedy"):
        r = cmp(E("eval_kimodo_opd_v2_e2_greedy") + ":TPH", E("eval_kimodo_sft_student_greedy") + ":TPH")
        ps = lambda n: rjson(E(n))["summary"]["plan_shift"]
        tv = {fl: [ps("eval_kimodo_sft_student_greedy")[fl]["tv_distance"], ps("eval_kimodo_opd_v2_e2_greedy")[fl]["tv_distance"]] for fl in ("direction", "duration")}
        ok_tv = all(b - a <= 0.1 for a, b in tv.values())
        G["G4-v2"] = {"kimodo_TPH": summ(r), "plan_tv_sft_vs_opd": tv, "tv_ok": ok_tv, "pass": (ci_pos(r, "R@10") or ci_pos(r, "orig_t2m_sim")) and ok_tv}
        for sub in ("mr", "v860"):
            G["G4-v2"][f"kimodo_TPH_subset_{sub}"] = summ(cmp(E("eval_kimodo_opd_v2_e2_greedy") + ":TPH", E("eval_kimodo_sft_student_greedy") + ":TPH", subset=sub))
        if f("eval_head_opd_v2_e2_greedy__head_A_full_best"):
            G["G4-v2"]["own_head_A_full"] = summ(cmp(E("eval_head_opd_v2_e2_greedy__head_A_full_best") + ":H", E("eval_head_sft_student_greedy__head_A_full_best") + ":H"))
    save = RUNS / "gates.json"; wjson(save, G); print(json.dumps(G, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
