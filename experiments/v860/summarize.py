"""Summary table of every VAL eval of this experiment (+ route-6 references). CPU. Writes runs/summary_val.md."""
import json
from pathlib import Path
from common import *

R6E = R6RUNS
COLS = ["R@10", "orig_t2m_sim", "R@3_samecat", "median_rank", "end_err_m", "mpjpe_m", "root_relative_mpjpe_m", "static"]


def row(label, path, arm):
    p = Path(path)
    if not p.exists():
        return None
    d = rjson(p); x = d["summary"][arm]; dec = d["summary"].get("decision", {})
    tv = d["summary"].get("plan_shift", {}).get("duration", {}).get("tv_distance")
    return [label] + [f"{x[c]:.3f}" if c != "median_rank" else f"{x[c]:.1f}" for c in COLS] + \
           [f"{dec['plan_acc_mean']:.3f}" if dec else "", f"{tv:.2f}" if tv is not None else ""]


def main():
    V = ["sft4b_student_greedy", "opd_v1_e1_greedy", "opd_v1_e2_greedy", "sft4b_teacher_zgreedy"]
    lab = {"sft4b_student_greedy": "4B SFT", "opd_v1_e1_greedy": "4B OPD e1", "opd_v1_e2_greedy": "4B OPD e2 (pre-registered final)",
           "sft4b_teacher_zgreedy": "*diag: 4B teacher + z (uses GT motion info)*"}
    R = []
    for v in V:
        for arm in ("T", "TPH"):
            R.append(row(f"{lab[v]} → Kimodo {arm}", RUNS / f"eval_kimodo_{v}.json", arm))
        for h, hl in (("head_pre", "own head (HumanML3D pretrain only)"), ("head_ft", "own head (PICO ft, pre-registered)"), ("head_ft_e1", "own head (PICO ft, full event E1)")):
            R.append(row(f"{lab[v]} → {hl}", RUNS / f"eval_head_{v}__{h}_best.json", "H"))
    for h, hl in (("head_pre", "pretrain only"), ("head_ft", "PICO ft"), ("head_ft_e1", "PICO ft E1")):
        R.append(row(f"*oracle gt3p + GT duration → own head ({hl})*", RUNS / f"eval_head_oracle__{h}_best.json", "H"))
    R += [row("route6 7B SFT → Kimodo TPH", R6E / "eval_sft_greedy.json", "TPH"), row("route6 7B DPO → Kimodo TPH", R6E / "eval_dpo_final_greedy.json", "TPH"),
          row("*route6 oracle gt3p → Kimodo O_T*", R6E / "eval_oracle.json", "O_T"), row("*route6 oracle gt3p + GT plan → Kimodo O_TPH*", R6E / "eval_oracle.json", "O_TPH_lab"),
          row("route6 HERO-FT (video end-to-end)", R6E / "eval_hero_ft_B.json", "H"), row("kNN1 TC-CLIP (real motion)", R6E / "eval_baselines_val.json", "knn1"),
          row("train mean (real motion)", R6E / "eval_baselines_val.json", "train_mean")]
    hdr = ["method / executor", "R@10↑", "t2m↑", "同类R@3↑", "中位排名↓", "终点误差m↓", "MPJPE↓", "根相对↓", "静止率↓", "计划准确率↑", "时长TV↓"]
    md = "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "".join("| " + " | ".join(r) + " |\n" for r in R if r)
    (RUNS / "summary_val.md").write_text("VAL 96 条（随机 R@10 = 0.104）。斜体 = 用到真值信息的诊断/上限，不可部署。\n\n" + md); print(md)


if __name__ == "__main__":
    main()
