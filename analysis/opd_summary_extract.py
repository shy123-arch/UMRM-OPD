import json, os, sys, importlib.util
from pathlib import Path
X = Path("/mnt/workspace/Wilson/ck/umrm/umrm_model/experiments")
def load_cmp(d):
    os.chdir(X / d); sys.path.insert(0, str(X / d))
    for m in ("common", "compare"):
        sys.modules.pop(m, None)
    spec = importlib.util.spec_from_file_location("cmp_" + d, X / d / "compare.py"); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    sys.path.pop(0); return M.cmp
def rj(p): return json.load(open(p))
def d(r, m):
    x = r["metrics"].get(m); return None if x is None else (round(x["diff"], 3), [round(v, 3) for v in x["ci95"]], round(x["A"], 3), round(x["B"], 3))
rows = {}
def arm(name, d_, opd, sft, head_opd, head_sft, k_opd, k_sft, ab=None):
    cmp = load_cmp(d_); R = X / d_ / "runs"
    o = {"head": d(cmp(str(R / f"eval_head_{head_opd}.json") + ":H", str(R / f"eval_head_{head_sft}.json") + ":H"), "R@10"),
         "head_t2m": d(cmp(str(R / f"eval_head_{head_opd}.json") + ":H", str(R / f"eval_head_{head_sft}.json") + ":H"), "orig_t2m_sim")}
    rk = cmp(str(R / f"eval_kimodo_{k_opd}.json") + ":TPH", str(R / f"eval_kimodo_{k_sft}.json") + ":TPH")
    o.update(kimodo=d(rk, "R@10"), plan=d(rk, "plan_acc_mean"), static=d(rk, "static"))
    ps = lambda n: rj(R / f"eval_kimodo_{n}.json")["summary"]["plan_shift"]
    o["tv_dir"] = [round(ps(k_sft)["direction"]["tv_distance"], 2), round(ps(k_opd)["direction"]["tv_distance"], 2)]
    o["tv_dur"] = [round(ps(k_sft)["duration"]["tv_distance"], 2), round(ps(k_opd)["duration"]["tv_distance"], 2)]
    o["top_dir"] = [round(ps(k_sft)["direction"]["top_label_share"], 2), round(ps(k_opd)["direction"]["top_label_share"], 2)]
    if ab:
        o["ab"] = d(cmp(str(R / f"eval_head_{ab[0]}.json") + ":H", str(R / f"eval_head_{ab[1]}.json") + ":H"), "R@10")
    rows[name] = o
arm("v1", "vt_head_opd_v860", "", "", "opd_v1_e2_greedy__head_ft_e1_best", "sft4b_student_greedy__head_ft_e1_best", "opd_v1_e2_greedy", "sft4b_student_greedy")
arm("v2", "vt_head_opd_v1197", "", "", "opd_v2_e2_greedy__head_A_full_best", "sft_student_greedy__head_A_full_best", "opd_v2_e2_greedy", "sft_student_greedy",
    ab=("opd_v2_e2_greedy__head_AB_full_best", "sft_student_greedy__head_AB_full_best"))
arm("v2_C1", "vt_head_opd_v1197", "", "", "opd_v2_s1_e2_greedy__head_A_full_best", "sft_student_greedy__head_A_full_best", "opd_v2_s1_e2_greedy", "sft_student_greedy",
    ab=("opd_v2_s1_e2_greedy__head_AB_full_best", "sft_student_greedy__head_AB_full_best"))
for v in ("v3", "v4", "v5r", "v5o", "v5o_mix"):
    arm(v, "vt_head_opd_v1197aug", "", "", f"opd_{v}_e1_greedy__head_A_full_best", "sft_student_greedy__head_A_full_best", f"opd_{v}_e1_greedy", "sft_student_greedy",
        ab=(f"opd_{v}_e1_greedy__head_AB_full_best", "sft_student_greedy__head_AB_full_best"))
# teacher strength
T = {}
nll = lambda d_, r: rj(X / d_ / "runs" / r / "selection.json")
s860, t860 = nll("vt_head_opd_v860", "sft_student"), nll("vt_head_opd_v860", "sft_teacher")
s1197, t1197 = nll("vt_head_opd_v1197", "sft_student"), nll("vt_head_opd_v1197", "sft_teacher_v2")
sa = nll("vt_head_opd_v1197aug", "sft_student")
sel = lambda s: s["val_nll"][s["selected"].replace("epoch", "")]
T["v1"] = {"stu": sel(s860), "tea": sel(t860)}; T["v2"] = {"stu": sel(s1197), "tea": sel(t1197)}
for v, t in (("v3", "sft_teacher_v3"), ("v5r", "sft_teacher_r"), ("v5o", "sft_teacher_o")):
    T[v] = {"stu": sel(sa), "tea": sel(nll("vt_head_opd_v1197aug", t))}
G = {"v1": X / "vt_head_opd_v860/runs/gate_GT_decision.json", "v2": X / "vt_head_opd_v1197/runs/gate_GT2_opd_v2.json",
     "v3": X / "vt_head_opd_v1197aug/runs/gate_GT2_opd_v3.json", "v5r": X / "vt_head_opd_v1197aug/runs/gate_GT2_opd_v5r.json", "v5o": X / "vt_head_opd_v1197aug/runs/gate_GT2_opd_v5o.json"}
for v, p in G.items():
    g = rj(p); T[v]["kl"] = (round(g["kl_with_z"], 3), round(g["kl_without_z"], 3), g.get("n"))
# teacher plan acc + teacher own-head / kimodo upper bounds
def ev(d_, f, arm_="H"):
    p = X / d_ / "runs" / f"{f}.json"; return round(rj(p)["summary"][arm_]["R@10"], 3) if p.exists() else None
def dec(d_, f):
    p = X / d_ / "runs" / f"{f}.json"; return round(rj(p)["summary"]["decision"]["plan_acc_mean"], 3) if p.exists() else None
T["v1"].update(tA=ev("vt_head_opd_v860", "eval_head_sft4b_teacher_zgreedy__head_ft_e1_best"), sA=ev("vt_head_opd_v860", "eval_head_sft4b_student_greedy__head_ft_e1_best"),
               tK=ev("vt_head_opd_v860", "eval_kimodo_sft4b_teacher_zgreedy", "TPH"), sK=ev("vt_head_opd_v860", "eval_kimodo_sft4b_student_greedy", "TPH"), tplan=dec("vt_head_opd_v860", "eval_kimodo_sft4b_teacher_zgreedy"), splan=dec("vt_head_opd_v860", "eval_kimodo_sft4b_student_greedy"))
T["v2"].update(tA=ev("vt_head_opd_v1197", "eval_head_sft_teacher_v2_zgreedy__head_A_full_best"), sA=ev("vt_head_opd_v1197", "eval_head_sft_student_greedy__head_A_full_best"),
               tK=ev("vt_head_opd_v1197", "eval_kimodo_sft_teacher_v2_zgreedy", "TPH"), sK=ev("vt_head_opd_v1197", "eval_kimodo_sft_student_greedy", "TPH"), tplan=dec("vt_head_opd_v1197", "eval_kimodo_sft_teacher_v2_zgreedy"), splan=dec("vt_head_opd_v1197", "eval_kimodo_sft_student_greedy"))
A = "vt_head_opd_v1197aug"
T["v3"].update(tAB=ev(A, "eval_head_sft_teacher_v3_zgreedy__head_AB_full_best"), sAB=ev(A, "eval_head_sft_student_greedy__head_AB_full_best"), tK=ev(A, "eval_kimodo_sft_teacher_v3_zgreedy", "TPH"), sK=ev(A, "eval_kimodo_sft_student_greedy", "TPH"), tplan=dec(A, "eval_kimodo_sft_teacher_v3_zgreedy"), splan=dec(A, "eval_kimodo_sft_student_greedy"))
T["v5o"].update(tAB=ev(A, "eval_head_sft_teacher_o_zgreedy__head_AB_full_best"), sAB=ev(A, "eval_head_sft_student_greedy__head_AB_full_best"))
for v, g in (("v5r", "gate_GT5r_text"), ("v5o", "gate_GT5o_text"), ("v3", "gate_GT3_text")):
    x = rj(X / A / "runs" / f"{g}.json")["subset_human"]; T[v]["plan_human"] = (round(x["plan_acc_b"], 2), round(x["plan_acc_a"], 2)); T[v]["exec_gain_human"] = round(x["exec_gain"], 3)
x = rj(X / "vt_head_opd_v1197/runs/gate_GT2_text.json"); T["v2"]["exec_gain_human"] = round(x["exec_gain"], 3); T["v2"]["plan_human"] = (round(x["plan_acc_b"], 2), round(x["plan_acc_a"], 2))
# training sizes
for v, (d_, r) in {"v1": ("vt_head_opd_v860", "opd_v1"), "v2": ("vt_head_opd_v1197", "opd_v2"), "v2_C1": ("vt_head_opd_v1197", "opd_v2_s1"), "v3": (A, "opd_v3"), "v4": (A, "opd_v4"), "v5r": (A, "opd_v5r"), "v5o": (A, "opd_v5o"), "v5o_mix": (A, "opd_v5o_mix")}.items():
    a = rj(X / d_ / "runs" / r / "args.json"); rows[v]["args"] = {k: a.get(k) for k in ("n_train", "k", "epochs", "lam_fwd", "seed", "zmode", "mr_filter")}
print(json.dumps({"rows": rows, "teacher": T}, indent=1, ensure_ascii=False))
