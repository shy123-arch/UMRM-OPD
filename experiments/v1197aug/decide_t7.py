"""Mechanical application of README §7 (teacher gates G-T7a self-consistency / G-T7b 32 frames; CPU). Writes runs/gates_t7.json.
Selections are made from the per-sample A36 readouts (eval_head_sc16_<m>_kKK__A36.json); no new generation."""
import json
from pathlib import Path
import numpy as np
from common import *
from compare import cmp

LL = EXP / "vt_lastlayer_v1197aug/runs"; TF = Path("/tmp/vho1197aug_cache/textfeat"); N = 16; L36 = 3  # LAYERS (9, 18, 27, 36)
GREEDY = {"stu": LL / "eval_head_sft__A36.json", "t8b": LL / "eval_head_sft_teacher_8b_greedy__A36.json"}
GREEDY_PRED = {"stu": "sft_student_greedy", "t8b": "sft_teacher_8b_greedy"}
ex = lambda p: Path(p).exists()
ok = lambda s: bool(s.get("strict")) and bool(s.get("exec"))


def summ(r, ms=("R@10", "orig_t2m_sim", "R@3_samecat", "orig_m2t_rank")):
    return {m: {"A": round(r["metrics"][m]["A"], 4), "B": round(r["metrics"][m]["B"], 4), "diff": round(r["metrics"][m]["diff"], 4),
                "ci95": [round(x, 4) for x in r["metrics"][m]["ci95"]], "p": round(r["metrics"][m]["wilcoxon_p"], 4)} for m in ms if m in r["metrics"]}


def load(m):
    S = {}
    for l in open(RUNS / f"sc16_{m}.shard0.jsonl"):
        r = json.loads(l); S[r["id"]] = r["samples"]
    E = [rjson(LL / f"eval_head_sc16_{m}_k{k:02d}__A36.json")["per_gen"] for k in range(N)]
    emb = []
    for k in range(N):
        meta = rjson(TF / f"pred_sc16_{m}_k{k:02d}.meta.json"); F = np.load(TF / f"pred_sc16_{m}_k{k:02d}.feats.npy", mmap_mode="r")
        assert tuple(meta["layers"])[L36] == 36
        emb.append({it["id"]: np.asarray(F[meta["offsets"][it["tid"]]:meta["offsets"][it["tid"]] + meta["lengths"][it["tid"]], L36], dtype=np.float32).mean(0)
                    for it in meta["items"]})
    return S, E, emb


def mbr_scores(C, i, emb, mu):
    V = np.stack([emb[k][i] - mu for k in C]); V /= np.linalg.norm(V, axis=1, keepdims=True)
    return (V @ V.T).sum(1) - 1.0  # sum of cosines to the other candidates


def select(ids, S, E, emb):
    mu = np.stack([emb[k][i] for i in ids for k in range(N) if ok(S[i][k])]).mean(0)
    sel = {"mbr_exec": {}, "mbr_dur": {}, "oracle_naive": {}, "oracle_xfit": {}}
    for i in ids:
        C = [k for k in range(N) if ok(S[i][k])] or [0]
        sc = mbr_scores(C, i, emb, mu); sel["mbr_exec"][i] = C[int(np.argmax(sc))]  # argmax = first max = lowest index on ties
        dur = [S[i][k]["plan"].get("duration") for k in C]; cnt = {d: dur.count(d) for d in dur}; top = max(cnt.values())
        tied = [d for d in cnt if cnt[d] == top]
        d_star = tied[0] if len(tied) == 1 else dur[max((j for j in range(len(C)) if dur[j] in tied), key=lambda j: (sc[j], -j))]
        Cd = [k for k, d in zip(C, dur) if d == d_star]; scd = mbr_scores(Cd, i, emb, mu); sel["mbr_dur"][i] = Cd[int(np.argmax(scd))]
        hit = lambda g: float(g["orig_m2t_rank"] <= 10)
        sel["oracle_naive"][i] = min(range(N), key=lambda k: (-np.mean([hit(g) for g in E[k][i]]), np.mean([g["orig_m2t_rank"] for g in E[k][i]]), k))
        sel["oracle_xfit"][i] = min(range(N), key=lambda k: (E[k][i][0]["orig_m2t_rank"], k))  # chosen on generation 1 only
    return sel


def synth(m, rule, ids, E, sel=None):
    if rule == "mean16":
        pg = {i: sum((E[k][i] for k in range(N)), []) for i in ids}
    elif rule == "oracle_xfit":
        pg = {i: E[sel[i]][i][1:] for i in ids}  # scored on generations 2-3 only
    else:
        pg = {i: E[sel[i]][i] for i in ids}
    p = RUNS / f"eval_sel_sc16_{m}_{rule}__A36.json"; wjson(p, {"name": f"sc16_{m}_{rule}", "per_gen": pg}); return p


def plan_acc_of(pred, ids, rows):
    return round(float(np.mean([plan_acc(pred[i]["plan"], rows[i]["plan"]) for i in ids])), 4)


def main():
    G = {}; man = manifest(); rows = man["rows"]; ids = list(man["val"])
    vs = lambda a, b: summ(cmp(str(a) + ":H", str(b) + ":H"))
    for m in ("stu", "t8b"):
        if not all(ex(LL / f"eval_head_sc16_{m}_k{k:02d}__A36.json") for k in range(N)):
            continue
        S, E, emb = load(m); sel = select(ids, S, E, emb)
        files = {r: synth(m, r, ids, E, sel[r]) for r in sel}; files["mean16"] = synth(m, "mean16", ids, E)
        gp = {}
        for l in open(RUNS / f"{GREEDY_PRED[m]}.shard0.jsonl"):
            r = json.loads(l); gp[r["id"]] = r["samples"][0]
        G[m] = {"strict_rate": round(float(np.mean([ok(s) for i in ids for s in S[i]])), 4),
                "plan_acc": {"greedy": plan_acc_of(gp, ids, rows), **{r: plan_acc_of({i: S[i][sel[r][i]] for i in ids}, ids, rows) for r in ("mbr_exec", "mbr_dur")},
                             "mean16": round(float(np.mean([plan_acc(S[i][k]["plan"], rows[i]["plan"]) for i in ids for k in range(N)])), 4)},
                "vs_own_greedy": {r: vs(files[r], GREEDY[m]) for r in files}}
        if m == "t8b":
            G[m]["vs_student_greedy"] = {r: vs(files[r], GREEDY["stu"]) for r in ("mbr_exec", "mbr_dur", "mean16")}
    if "stu" in G:
        r = G["stu"]["vs_own_greedy"]["mbr_exec"]["R@10"]; xf = G["stu"]["vs_own_greedy"]["oracle_xfit"]["R@10"]["diff"]
        G["G-T7a"] = {"pass": r["ci95"][0] > 0, "student_mbr_exec_minus_greedy_R@10": r,
                      "reading": ("consensus teacher is significantly stronger -> OPD v7 worth registering" if r["ci95"][0] > 0 else
                                  "answer is in the distribution but this selector does not find it" if xf >= 0.15 else
                                  "consensus teacher not stronger; little headroom in the student's own samples")}
    if ex(LL / "eval_head_sft_f32_greedy__A36.json"):
        r = vs(LL / "eval_head_sft_f32_greedy__A36.json", GREEDY["stu"]); gp = {}
        for l in open(RUNS / "sft_f32_greedy.shard0.jsonl"):
            x = json.loads(l); gp[x["id"]] = x["samples"][0]
        G["G-T7b"] = {"pass": r["R@10"]["ci95"][0] > 0, "f32_minus_f16_greedy": r,
                      "val_nll": {"f32": rjson(RUNS / "sft_f32/selection.json")["val_nll"], "f16": rjson(RUNS / "sft_student/selection.json")["val_nll"],
                                  "selected_f32": rjson(RUNS / "sft_f32/selection.json")["selected"]},
                      "plan_acc": {"f32": plan_acc_of(gp, ids, rows)}}
    wjson(RUNS / "gates_t7.json", G); print(json.dumps(G, indent=1, ensure_ascii=False)[:5000])


if __name__ == "__main__":
    main()
