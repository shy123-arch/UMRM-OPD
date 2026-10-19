"""VLM text -> frozen Kimodo, scored by the route-6 independent evaluator (same 3 seeds, arms T / TPH, same compiler
with TRAIN-only stats). Identical logic to route6 eval_motion.py; predictions are read from THIS experiment's runs/.
--stage encode (LLM2Vec, run once per prediction file) | eval. Usage: eval_kimodo.py --name sft_student_greedy"""
import argparse, json, time
import numpy as np
from eval_common import C, R6C, R6K, summarize

SEEDS = [6001, 6002, 6003]; FALLBACK = "A person stands still."


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--stage", default="eval"); ap.add_argument("--arms", default="T,TPH")
    a = ap.parse_args(); man = R6C.manifest(); rows = man["rows"]; keys = man["val"]
    if a.stage == "eval" and (C.RUNS / f"eval_kimodo_{a.name}.json").exists():
        print("exists, skip", a.name); return
    R = C.load_jsonl_shards(a.name, C.RUNS); miss = set(keys) - set(R); assert not miss, f"{len(miss)} missing"
    outs = {k: R[k]["samples"][0] for k in keys}
    if a.stage == "encode":
        print("encoded", R6K.encode_texts([FALLBACK] + [rows[k]["exec"] for k in keys] + [o["exec"] for o in outs.values() if o["exec"]], mode="eqlen")); return
    E = R6K.Engine(); unenc = [k for k in keys if outs[k]["exec"] and outs[k]["exec"] not in E.tc.d]
    if unenc:
        raise SystemExit(f"{len(unenc)} exec texts not LLM2Vec-encoded; run --stage encode")
    comp = R6C.Compiler(R6C.rjson(R6C.RUNS / "compiler_stats.json")); M = R6K.Metrics(E, keys, rows); th = R6K.theta0(E)
    jobs = []
    for k in keys:
        o = outs[k]; text = o["exec"] if o["exec"] else FALLBACK; n = comp.nframes(o["plan"])
        for arm in a.arms.split(","):
            jobs.append((k, arm, text, n, None if arm == "T" else (comp.constraints(o["plan"], n, th, heading=True) or None)))
    t0 = time.time(); res = {}
    for sd in SEEDS:
        for j, g in zip(jobs, R6K.gen(E, [j[2] for j in jobs], [j[3] for j in jobs], [j[4] for j in jobs], seed=sd)):
            res.setdefault(f"{j[0]}|{j[1]}", []).append(M.motion(j[0], g))
        print("seed", sd, round(time.time() - t0), flush=True)
    ex = E.otmr_text([outs[k]["exec"] or FALLBACK for k in keys]); gt = E.otmr_text([rows[k]["exec"] for k in keys])
    rx = E.otmr_text([outs[k]["reaction"] or "" for k in keys]); hr = E.otmr_text([rows[k]["reaction"] for k in keys])
    dec = {k: {"exec_vs_gt3p_cos": R6K.cos(ex[i], gt[i]), "reaction_vs_human_cos": R6K.cos(rx[i], hr[i]), "strict": float(bool(outs[k].get("strict"))),
               "plan_acc_mean": R6C.plan_acc(outs[k]["plan"], rows[k]["plan"]),
               **{f"acc_{f}": float(outs[k]["plan"].get(f) == rows[k]["plan"][f]) for f in R6C.PLAN_FIELDS}} for i, k in enumerate(keys)}
    summ = {arm: summarize({k: res[f"{k}|{arm}"] for k in keys}, keys) for arm in a.arms.split(",")}
    summ["decision"] = {m: float(np.mean([dec[k][m] for k in keys])) for m in dec[keys[0]]}
    import importlib.util
    sp = importlib.util.spec_from_file_location("r6_eval_motion", C.R6 / "eval_motion.py"); EM = importlib.util.module_from_spec(sp); sp.loader.exec_module(EM)
    summ["plan_shift"] = EM.plan_shift(outs, rows, keys)
    C.save_json(C.RUNS / f"eval_kimodo_{a.name}.json", {"name": a.name, "split": "val", "seeds": SEEDS, "n": len(keys), "theta0": th,
                                                       "summary": summ, "per_gen": res, "decision": dec,
                                                       "chosen": {k: {x: outs[k].get(x) for x in ("reaction", "exec", "plan")} for k in keys}})
    print(json.dumps({k: v for k, v in summ.items() if k != "plan_shift"}, indent=1))


if __name__ == "__main__":
    main()
