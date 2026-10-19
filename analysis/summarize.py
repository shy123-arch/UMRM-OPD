"""Summary table of every v1197 VAL eval (125 items; random R@10 = 0.08). CPU. Writes runs/summary_val.md."""
from pathlib import Path
from common import *
COLS = ["R@10", "orig_t2m_sim", "R@3_samecat", "median_rank", "end_err_m", "mpjpe_m", "root_relative_mpjpe_m", "static"]


def row(label, path, arm):
    p = Path(path)
    if not p.exists():
        return None
    d = rjson(p); x = d["summary"][arm]; dec = d["summary"].get("decision", {}); ps = d["summary"].get("plan_shift", {})
    tv = lambda f: f"{ps[f]['tv_distance']:.2f}" if f in ps else ""
    return [label] + [f"{x[c]:.1f}" if c == "median_rank" else f"{x[c]:.3f}" for c in COLS] + \
           [f"{dec['plan_acc_mean']:.3f}" if dec else "", tv("direction"), tv("duration")]


def main():
    P = [("sft_student_greedy", "SFT"), ("opd_v2_e1_greedy", "OPD v2 e1"), ("opd_v2_e2_greedy", "OPD v2 e2（预登记）"),
         ("sft_teacher_v2_zgreedy", "*诊断：教师 v2 + z*"), ("oracle_gt3p", "*上限：gt3p + 真值计划*")]
    Hd = [("head_A_cap_best", "自有头 A（截断）"), ("head_A_full_best", "自有头 A（完整事件）"), ("head_AB_full_best", "自有头 A+B 视觉残差"),
          ("head_AB_full_best_visshift1", "自有头 A+B（错配视频，对照）"),
          ("head_ABfrz_full_best", "自有头 A冻结+B（探索性）"), ("head_ABfrz_full_best_visshift1", "自有头 A冻结+B（探索性，错配视频）")]
    R = []
    for n, l in P:
        for arm in ("T", "TPH"):
            R.append(row(f"{l} → Kimodo {arm}", RUNS / f"eval_kimodo_{n}.json", arm))
        for h, hl in Hd:
            R.append(row(f"{l} → {hl}", RUNS / f"eval_head_{n}__{h}.json", "H"))
    R.append(row("训练集平均动作", RUNS / "eval_base_train_mean.json", "H"))
    hdr = ["方法 → 执行器", "R@10↑", "t2m↑", "同类R@3↑", "中位排名↓", "终点误差m↓", "MPJPE↓", "根相对↓", "静止率↓", "计划准确率↑", "方向TV↓", "时长TV↓"]
    md = "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "".join("| " + " | ".join(r) + " |\n" for r in R if r)
    (RUNS / "summary_val.md").write_text("v1197 VAL 125 条（随机 R@10 = 0.080）。斜体 = 用到真值信息的诊断/上限，不可部署。\n\n" + md); print(md)


if __name__ == "__main__":
    main()
