"""Independent evaluation on VAL (kimodo env, 1 GPU). Frozen Kimodo, 3 seeds, arms T (text + predicted duration) and TPH
(+ compiled Root2D path + terminal heading). Evaluator = official HumanML3D TMR + geometry ONLY (TMR-SOMA is the reward
model and is never used here). Sample choice for multi-sample files: --select first | mbr (GT-free minimum-Bayes-risk:
utility = LLM2Vec cos(exec_i, exec_j) + plan agreement; equal-budget control). --oracle: gt3p arms.
--stage encode (LLM2Vec texts, run once) | --stage eval."""
import argparse, collections, time
import numpy as np
from r6_common import *

SEEDS = [6001, 6002, 6003]
FALLBACK = "A person stands still."


def outputs(a, man, keys):
    rows = man["rows"]
    if a.oracle:
        return {k: {"exec": rows[k]["exec"], "reaction": rows[k]["reaction"], "plan": rows[k]["plan"], "strict": True} for k in keys}
    R = load_jsonl_shards(a.name); miss = set(keys) - set(R)
    assert not miss, f"{len(miss)} missing"
    if a.select == "first":
        return {k: R[k]["samples"][0] for k in keys}
    from r6_kimodo import TextCache, cos
    tc = TextCache(); out = {}
    for k in keys:
        S = [s for s in R[k]["samples"] if s["exec"]] or R[k]["samples"]
        if len(S) == 1:
            out[k] = S[0]; continue
        e = [tc.d[s["exec"]] if s["exec"] in tc.d else None for s in S]
        U = np.zeros(len(S))
        for i in range(len(S)):
            for j in range(len(S)):
                if i != j:
                    U[i] += (cos(e[i], e[j]) if e[i] is not None and e[j] is not None else 0) + plan_acc(S[i]["plan"], {f: S[j]["plan"].get(f) for f in PLAN_FIELDS})
        out[k] = {**S[int(U.argmax())], "mbr_index": int(U.argmax()), "mbr_n": len(S)}
    return out


def plan_shift(outs, rows, keys):
    res = {}
    for f in PLAN_FIELDS:
        pr = collections.Counter(outs[k]["plan"].get(f, "<missing>") for k in keys); gt = collections.Counter(rows[k]["plan"][f] for k in keys)
        labs = set(pr) | set(gt); n = len(keys)
        ent = lambda c: float(-sum((v / n) * np.log(v / n) for v in c.values() if v))
        res[f] = {"pred": dict(pr), "gt": dict(gt), "tv_distance": float(0.5 * sum(abs(pr[l] / n - gt[l] / n) for l in labs)),
                  "entropy_pred": ent(pr), "entropy_gt": ent(gt), "top_label_share": max(pr.values()) / n}
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name"); ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--select", default="first", choices=["first", "mbr"]); ap.add_argument("--split", default="val")
    ap.add_argument("--arms", default="T,TPH"); ap.add_argument("--stage", default="eval"); ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    if a.split == "test" and os.environ.get("R6_UNSEAL_TEST") != "1":
        raise SystemExit("TEST is sealed")
    man = manifest(); rows = man["rows"]; keys = man[a.split]
    if a.stage == "encode":
        from r6_kimodo import encode_texts
        T = [FALLBACK] + [rows[k]["exec"] for k in keys]
        if not a.oracle:
            T += [s["exec"] for r in load_jsonl_shards(a.name).values() for s in r["samples"] if s["exec"]]
        print("encoded", encode_texts(T), flush=True); return
    from r6_kimodo import Engine, Metrics, gen, theta0, cos
    tag = a.tag or ("oracle" if a.oracle else a.name + ("_mbr" if a.select == "mbr" else ""))
    comp = Compiler(rjson(RUNS / "compiler_stats.json")); outs = outputs(a, man, keys)
    E = Engine()
    unenc = [k for k in keys if outs[k]["exec"] and outs[k]["exec"] not in E.tc.d]
    if unenc:  # never silently replace an unencoded prediction by the fallback text
        raise SystemExit(f"{len(unenc)} predicted exec texts are not LLM2Vec-encoded; run s_encode.py first")
    M = Metrics(E, keys, rows); th = theta0(E)
    arms = ["O_T", "O_TPH_lab"] if a.oracle else a.arms.split(",")
    jobs = []
    for k in keys:
        o = outs[k]; text = o["exec"] if o["exec"] and o["exec"] in E.tc.d else FALLBACK
        for arm in arms:
            if arm == "O_T":
                n, sp = E.gt_len(k), None
            else:
                n = comp.nframes(o["plan"]); sp = None if arm == "T" else (comp.constraints(o["plan"], n, th, heading=True) or None)
            jobs.append((k, arm, text, n, sp))
    t0 = time.time(); res = {}
    for sd in SEEDS:
        G = gen(E, [j[2] for j in jobs], [j[3] for j in jobs], [j[4] for j in jobs], seed=sd)
        for j, g in zip(jobs, G):
            res.setdefault(f"{j[0]}|{j[1]}", []).append(M.motion(j[0], g))
        print("seed", sd, round(time.time() - t0), flush=True)
    ex = E.otmr_text([outs[k]["exec"] or FALLBACK for k in keys]); gt = E.otmr_text([rows[k]["exec"] for k in keys])
    rx = E.otmr_text([outs[k]["reaction"] or "" for k in keys]); hr = E.otmr_text([rows[k]["reaction"] for k in keys])
    dec = {k: {"exec_vs_gt3p_cos": cos(ex[i], gt[i]), "reaction_vs_human_cos": cos(rx[i], hr[i]), "strict": float(bool(outs[k].get("strict"))),
               "has_exec": float(bool(outs[k]["exec"])), "plan_acc_mean": plan_acc(outs[k]["plan"], rows[k]["plan"]),
               **{f"acc_{f}": float(outs[k]["plan"].get(f) == rows[k]["plan"][f]) for f in PLAN_FIELDS}} for i, k in enumerate(keys)}
    mets = ["orig_cos", "orig_t2m_sim", "orig_m2t_rank", "orig_m2t_rank_samecat", "end_err_m", "yaw_err_deg", "mpjpe_m",
            "root_relative_mpjpe_m", "static", "body_speed"]
    summ = {}
    for arm in arms:
        per = {k: res[f"{k}|{arm}"] for k in keys}
        s = {m: float(np.mean([np.mean([x[m] for x in per[k]]) for k in keys])) for m in mets}
        s["R@10"] = float(np.mean([np.mean([x["orig_m2t_rank"] <= 10 for x in per[k]]) for k in keys]))
        s["R@3_samecat"] = float(np.mean([np.mean([x["orig_m2t_rank_samecat"] <= 3 for x in per[k]]) for k in keys]))
        s["median_rank"] = float(np.median([np.mean([x["orig_m2t_rank"] for x in per[k]]) for k in keys]))
        summ[arm] = s
    summ["decision"] = {m: float(np.mean([dec[k][m] for k in keys])) for m in dec[keys[0]]}
    summ["plan_shift"] = plan_shift(outs, rows, keys)
    summ["chance"] = {"R@10": 10 / len(keys), "median_rank": (len(keys) + 1) / 2,
                      "R@3_samecat": float(np.mean([min(1, 3 / M.cat.tolist().count(rows[k]["category"])) for k in keys]))}
    save_json(RUNS / f"eval_{tag}.json", {"name": tag, "split": a.split, "arms": arms, "seeds": SEEDS, "theta0": th, "n": len(keys),
                                         "select": a.select, "elapsed_s": time.time() - t0, "summary": summ, "per_gen": res, "decision": dec,
                                         "chosen": {k: {x: outs[k].get(x) for x in ("reaction", "exec", "plan", "mbr_index")} for k in keys}})
    print(json.dumps({k: v for k, v in summ.items() if k != "plan_shift"}, indent=1))


if __name__ == "__main__":
    main()
