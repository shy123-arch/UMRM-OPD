"""B2 (CPU): final v1197 experimental manifest + compiler stats (TRAIN only; route-6 definitions).
reaction: v860 text | uploader English original | Qwen3-8B translation (B1, 3 x 羽毛球 fixed in B1b).
exec (gt3p): inherited v860 gt3p | Qwen3-8B rewrite (B1) after fidelity repair (B1b no-addition gate, B1c no-omission).
TEST / test_extra / excluded rows carry no text (never needed now). Output: runs/data_manifest.json, runs/compiler_stats.json"""
import collections, sys
import numpy as np
from common import *
sys.path.insert(0, str(R6))
import r6_common as R6C  # noqa: E402
import plan as PL  # noqa: E402


def main():
    if (RUNS / "data_manifest.json").exists():
        raise FileExistsError(RUNS / "data_manifest.json")
    inv = rjson(RUNS / "inventory.json"); rows_in = inv["rows"]
    b1, b1b, b1c = rjson(RUNS / "b1_llm.json"), rjson(RUNS / "b1b_fidelity.json"), rjson(RUNS / "b1c_omission.json")
    trans, gt3p = b1b["translations"], b1c["gt3p_final"]
    rows = {}
    for k, r in rows_in.items():
        g = r["group"]; base = {x: r[x] for x in ("id", "group", "cohort", "category", "category_src", "category_confidence", "source",
                                                  "staged_film", "prev_group", "in_v860_val") if x in r}
        if r.get("eval_exclude"):
            base["eval_exclude"] = r["eval_exclude"]
        if g not in ("train", "val", "sealed"):
            rows[k] = base; continue
        reaction = r["reaction"] if r.get("reaction") else trans[k]
        if r.get("exec"):
            exec_, exec_src = r["exec"], r["exec_src"]
        else:
            exec_ = gt3p[k]; exec_src = "qwen3_8b_rewrite(route6 prompt)" + ("+strict" if k in b1b["regenerated"] else "") + ("+keep_verbs" if k in b1c["regenerated"] else "")
        rows[k] = {**base, "clip": r["clip"], "reaction": reaction, "reaction_src": r["reaction_src"], "exec": exec_, "exec_src": exec_src,
                   "plan": r["plan"], "plan_raw": r["plan_raw"], "mental_reasoning": r.get("mental_reasoning", ""),
                   "target": target_text(reaction, exec_, r["plan"])}
    keys = {g: sorted(k for k in rows if rows[k]["group"] == g) for g in ("train", "val", "test", "test_extra", "sealed", "excluded_conflict")}
    z = np.load(DATA / "data.npz"); pos = {str(k): i for i, k in enumerate(z["ids"])}
    P = {k: PL.extract(z["targets"][pos[k]], z["masks"][pos[k]]) for k in keys["train"]}
    med = lambda v: float(np.median(v)) if len(v) else 0.0
    stats = {"dist": {l: med([P[k]["distance_m"] for k in P if P[k]["locomotion"] == l]) for l in ("stay", "step", "walk", "run")},
             "yaw": {t: med([abs(P[k]["yaw_deg"]) for k in P if P[k]["turn"] == t]) for t in ("left", "right", "around")},
             "onset_frac": med([P[k]["onset_s"] / P[k]["duration_s"] for k in P]), "settle_frac": med([P[k]["settle_s"] / P[k]["duration_s"] for k in P]),
             "dur_median": med([P[k]["duration_s"] for k in P]),
             "label_counts": {f: dict(collections.Counter(rows[k]["plan"][f] for k in keys["train"])) for f in PLAN_FIELDS}}
    split = {"data_version": "v1197", "inherit_from": inv["inherit_from"], "rules": inv["rules"], "item_group": {k: rows[k]["group"] for k in rows}}
    save_json(RUNS / "split.json", split); save_json(RUNS / "compiler_stats.json", stats)
    rep = {g: {"n": len(v), "cohort": dict(collections.Counter(rows[k]["cohort"] for k in v)),
               "category": dict(sorted(collections.Counter(rows[k]["category"] for k in v).items()))} for g, v in keys.items()}
    rep["text_provenance_trainval"] = {"reaction_src": dict(collections.Counter(rows[k]["reaction_src"].split("->")[-1][:60] for g in ("train", "val") for k in keys[g])),
                                       "exec_src": dict(collections.Counter(rows[k]["exec_src"][:60] for g in ("train", "val") for k in keys[g]))}
    save_json(RUNS / "data_manifest.json", {"data_version": "v1197", "split_sha256": sha(RUNS / "split.json"), **keys, "test_sealed": True,
                                             "report": rep, "rows": rows})
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
