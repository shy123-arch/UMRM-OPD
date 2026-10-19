"""G-T2 semantic part (kimodo env, official HumanML3D TMR text encoder only): on VAL items with mental_reasoning,
teacher-v2 greedy (with z) vs student greedy: TMR text cosine of <exec> vs gt3p and of <reaction> vs the human reaction.
Usage: gate_text.py --a sft_teacher_v2_zgreedy --b sft_student_greedy --out gate_GT2_text"""
import argparse, json
import numpy as np
from eval_common import C, R6K


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    man = C.manifest(); rows = man["rows"]; keys = [k for k in man["val"] if rows[k].get("mental_reasoning")]
    A, B = C.load_jsonl_shards(a.a, C.RUNS), C.load_jsonl_shards(a.b, C.RUNS)
    E = R6K.Engine(with_kimodo=False); T = lambda xs: E.otmr_text([x or "" for x in xs])
    gt, hr = T([rows[k]["exec"] for k in keys]), T([rows[k]["reaction"] for k in keys])
    res = {}
    for nm, R in (("a", A), ("b", B)):
        ex, rx = T([R[k]["samples"][0]["exec"] for k in keys]), T([R[k]["samples"][0]["reaction"] for k in keys])
        res[nm] = {"exec_vs_gt3p": [R6K.cos(ex[i], gt[i]) for i in range(len(keys))], "reaction_vs_human": [R6K.cos(rx[i], hr[i]) for i in range(len(keys))],
                   "plan_acc": [C.plan_acc(R[k]["samples"][0]["plan"], rows[k]["plan"]) for k in keys]}
    out = {"a": a.a, "b": a.b, "n": len(keys), **{f"{m}_{nm}": float(np.mean(res[nm][m])) for nm in "ab" for m in res[nm]}}
    out["exec_gain"] = out["exec_vs_gt3p_a"] - out["exec_vs_gt3p_b"]; out["reaction_gain"] = out["reaction_vs_human_a"] - out["reaction_vs_human_b"]
    out["pass_text_part(exec_gain>=0.05)"] = out["exec_gain"] >= 0.05
    for sub in ("generated", "human"):  # v1197aug: G-T3 is defined on the generated-MR VAL items
        idx = [i for i, k in enumerate(keys) if (rows[k].get("mr_src") == "generated") == (sub == "generated")]
        if idx:
            out[f"subset_{sub}"] = {"n": len(idx), **{f"{m}_{nm}": float(np.mean([res[nm][m][i] for i in idx])) for nm in "ab" for m in res[nm]}}
            out[f"subset_{sub}"]["exec_gain"] = out[f"subset_{sub}"]["exec_vs_gt3p_a"] - out[f"subset_{sub}"]["exec_vs_gt3p_b"]
    C.save_json(C.RUNS / f"{a.out}.json", out); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
