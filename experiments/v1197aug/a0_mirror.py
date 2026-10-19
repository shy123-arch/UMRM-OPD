"""A0 (hero env, CPU): left-right MIRROR augmentation of the v1197 TRAIN items (read-only on all inputs).
For each real TRAIN item k -> 'k_M' (group train, mirror_of=k) when all of its texts can be mirrored unambiguously:
  motion : canonical PICO joints x -> -x (canonical x = rightward axis) + swap left/right joint pairs; 88-frame target from
           the frozen data_v1197 targets, full event from the raw recording (as v1197 b3); HumanML263 via pico_to_humanml.
  plan   : re-extracted from the mirrored 88-frame target with the route-6 extractor (agreement with label-swapping reported)
  text   : reaction / exec / mental_reasoning with lateral words swapped (left<->right, clockwise<->counterclockwise...);
           an item is NOT mirrored if any field contains 'left'/'right' that the rule cannot classify as lateral
           (e.g. 'I left the room', 'right away') -> listed in report.
  video  : horizontally flipped frames (prep_video_cache.py, key 'k_M').
VAL / sealed / TEST are never mirrored. Output: runs/data_manifest.json (+ split.json, compiler_stats.json copied from
v1197 = real TRAIN only), runs/motion/{humanml263_capped,humanml263_full}.npz (real + mirrored), runs/mirror_report.json"""
import importlib.util, json, re, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
_cs = importlib.util.spec_from_file_location("vho_common", HERE / "common.py"); C = importlib.util.module_from_spec(_cs); _cs.loader.exec_module(C)
sp = importlib.util.spec_from_file_location("p2h", C.EXP / "hero_finetune/pico_to_humanml.py"); P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)
sys.path.insert(0, str(C.EXP / "harness_v1")); sys.path.insert(0, str(C.EXP / "cliponly_v1"))
import plan as PL  # noqa: E402
from data import canonical, NAMES  # noqa: E402

PAIRS = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23)]
PERM = list(range(24))
for a, b in PAIRS:
    PERM[a], PERM[b] = b, a
LAT = {"left": "right", "right": "left", "leftward": "rightward", "rightward": "leftward", "leftwards": "rightwards", "rightwards": "leftwards",
       "clockwise": "counterclockwise", "counterclockwise": "clockwise", "counter-clockwise": "clockwise", "anticlockwise": "clockwise",
       "anti-clockwise": "clockwise"}
PRE = set("the my his her their your its a one's to turn turns turned turning veer veers veered veering swerve swerves swerved swerving "
          "step steps stepped stepping look looks looked looking glance glances glanced glancing lean leans leaned leaning shift shifts "
          "shifted shifting dodge dodges dodged dodging move moves moved moving slide slides slid sliding sidestep sidesteps pivot pivots "
          "pivoted pivoting rotate rotates rotated rotating head heads headed heading go goes went going swing swings swung swinging "
          "lower upper far front back rear both from toward towards and or slightly sharply quickly".split())
POST = set("hand hands arm arms leg legs foot feet side sides shoulder shoulders knee knees wrist wrists elbow elbows lane lanes turn "
           "and or hip hips ear ears eye eye-line palm fist rear front corner".split())
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def mirror_text(t):
    """-> (mirrored text, ambiguous: bool)"""
    if not t:
        return t, False
    toks = list(WORD.finditer(t)); out = []; last = 0; amb = False
    for i, m in enumerate(toks):
        w = m.group(0); lw = w.lower()
        if lw in LAT:
            if lw in ("left", "right"):
                prev = {toks[j].group(0).lower() for j in range(max(0, i - 2), i)}
                nxt = toks[i + 1].group(0).lower() if i + 1 < len(toks) else ""
                if not (prev & PRE or nxt in POST):
                    amb = True
            rep = LAT[lw]; rep = rep.capitalize() if w[0].isupper() else rep
            out.append(t[last:m.start()] + rep); last = m.end()
    out.append(t[last:])
    return "".join(out), amb


def mirror_joints(x):
    y = np.asarray(x, np.float64)[..., PERM, :].copy(); y[..., 0] *= -1; return y


SWAP_DIR = {"left": "right", "right": "left", "forward-left": "forward-right", "forward-right": "forward-left", "back-left": "back-right", "back-right": "back-left"}


def swap_plan(p):
    q = dict(p); q["direction"] = SWAP_DIR.get(p["direction"], p["direction"]); q["turn"] = {"left": "right", "right": "left"}.get(p["turn"], p["turn"]); return q


def main():
    if (C.RUNS / "data_manifest.json").exists():
        raise FileExistsError(C.RUNS / "data_manifest.json")
    m = C.rjson(C.V1197 / "runs/data_manifest.json"); rows = {k: dict(r) for k, r in m["rows"].items()}
    z = np.load(C.DATA / "data.npz"); pos = {str(k): i for i, k in enumerate(z["ids"])}
    sys.path.insert(0, str(C.R6)); import r6_common as R6C  # noqa
    BASE = C.UMRM / "benchmark/judge_exploration/evaluations/astra_motion_loop/baseline636_20260915"
    refs = {r["id"]: r["primary_recording"] for r in json.loads((BASE / "pico_baseline_636.json").read_text())["items"]}
    refs.update({json.loads(l)["id"]: json.loads(l)["pico_file"] for l in (C.DATA / "provenance_new.jsonl").read_text().splitlines()})
    P.configure(); rep = {"excluded_ambiguous": {}, "plan_label_swap_agreement": {}, "n_real_train": len(m["train"])}
    new, cap_f, cap_l, full_f, full_l, agree = [], [], [], [], [], []
    for k in m["train"]:
        r = rows[k]; mt = {}; amb = []
        for f in ("reaction", "exec", "mental_reasoning"):
            mt[f], a = mirror_text(r.get(f, "")); amb += [f] if a else []
        if amb:
            rep["excluded_ambiguous"][k] = amb; continue
        i = pos[k]; msk = z["masks"][i].astype(bool); L = int(msk.sum())
        tgt = mirror_joints(z["targets"][i]); p = PL.extract(tgt, msk); plan = R6C.plan_labels(p)
        agree.append({f: plan[f] == swap_plan(r["plan"])[f] for f in C.PLAN_FIELDS})
        cf = P.to_features(tgt[:L, :22]); assert np.isfinite(cf).all(), k
        with np.load(refs[k], allow_pickle=False) as d:
            names = d["body_joint_names"].tolist(); x = d["raw_body_poses"][:, [names.index(n) for n in NAMES], :3]
            ns = d["raw_motion_timestamp_ns"].astype(np.int64); t = (ns - ns[0]) * 1e-9
        x = canonical(x); n = min(196, int(np.floor(t[-1] * 20 + 1e-9)) + 1); g = np.arange(n) / 20
        y = np.stack([np.interp(g, t, x[:, j, c]) for j in range(24) for c in range(3)], -1).reshape(n, 24, 3)
        ff = P.to_features(mirror_joints(y)[:, :22]); assert np.isfinite(ff).all(), k
        km = k + "_M"; new.append(km)
        rows[km] = {**{x_: r[x_] for x_ in ("cohort", "category", "category_src", "category_confidence", "source", "staged_film", "clip") if x_ in r},
                    "id": km, "group": "train", "mirror_of": k, "prev_group": None, "in_v860_val": False,
                    "reaction": mt["reaction"], "reaction_src": r["reaction_src"] + "+mirror", "exec": mt["exec"], "exec_src": r["exec_src"] + "+mirror",
                    "mental_reasoning": mt["mental_reasoning"], "plan": plan,
                    "plan_raw": {x_: float(p[x_]) for x_ in ("duration_s", "distance_m", "yaw_deg", "forward_m", "right_m")},
                    "target": C.target_text(mt["reaction"], mt["exec"], plan)}
        cap_f.append(cf); cap_l.append(L); full_f.append(ff); full_l.append(n)
    # motion files: real (v1197 runs/motion) + mirrored
    out = C.RUNS / "motion"; out.mkdir(parents=True, exist_ok=True)
    for kind, F, Ls, T in (("capped", cap_f, cap_l, 88), ("full", full_f, full_l, 196)):
        with np.load(C.V1197 / f"runs/motion/humanml263_{kind}.npz") as zz:
            ids = [str(x_) for x_ in zz["ids"]]; feats, lens = zz["features"], zz["lengths"]
        add = np.zeros((len(new), T, 263), np.float32)
        for j, f in enumerate(F):
            add[j, :len(f)] = f
        np.savez(out / f"humanml263_{kind}.npz", ids=np.array(ids + new), features=np.concatenate([feats, add]), lengths=np.concatenate([lens, np.array(Ls)]))
    rep["n_mirrored"] = len(new); rep["n_excluded"] = len(rep["excluded_ambiguous"])
    rep["plan_label_swap_agreement"] = {f: float(np.mean([a[f] for a in agree])) for f in C.PLAN_FIELDS}
    rep["n_train_total"] = len(m["train"]) + len(new)
    examples = [(rows[k]["exec"], rows[k + "_M"]["exec"]) for k in m["train"] if k + "_M" in rows and re.search(r"\b(left|right)\b", rows[k]["exec"], re.I)][:8]
    rep["examples_lateral"] = examples
    keys = {g: list(m[g]) for g in ("val", "test", "test_extra", "sealed", "excluded_conflict")}; keys["train"] = list(m["train"]) + new
    C.save_json(C.RUNS / "split.json", {"data_version": "v1197aug", "base": str(C.V1197 / "runs/split.json"), "item_group": {k: rows[k]["group"] for k in rows}})
    C.save_json(C.RUNS / "compiler_stats.json", C.rjson(C.V1197 / "runs/compiler_stats.json"))
    C.save_json(C.RUNS / "data_manifest.json", {"data_version": "v1197aug", "split_sha256": C.sha(C.RUNS / "split.json"), **keys, "test_sealed": True,
                                               "report": rep, "rows": rows})
    C.save_json(C.RUNS / "mirror_report.json", rep); print(json.dumps({k_: v for k_, v in rep.items() if k_ != "excluded_ambiguous"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
