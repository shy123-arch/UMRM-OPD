"""E1 data (VPP2-inspired, 'complete event'): v860 TRAIN+VAL PICO targets WITHOUT the 88-frame (4.4 s) cap, at 20 Hz,
capped only at 196 frames (9.8 s, the HumanML3D/MoMask maximum), converted to HumanML263 with EXACTLY the hero_ft_v860
conversion (experiments/hero_finetune/pico_to_humanml.py, no uniform_skeleton). Same canonicalisation as cliponly_v1.
TEST never read. hero env, CPU. Output: runs/e1_full_event/humanml263_full.npz {ids, features (N,196,263), lengths, raw_dur_s}
+ check that the first 88 frames reproduce hero_ft_v860's features for uncapped items."""
import csv, importlib.util, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
_cs = importlib.util.spec_from_file_location("vho_common", HERE / "common.py"); C = importlib.util.module_from_spec(_cs); _cs.loader.exec_module(C)
sp = importlib.util.spec_from_file_location("p2h", C.EXP / "hero_finetune/pico_to_humanml.py"); P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)
sys.path.insert(0, str(C.EXP / "cliponly_v1"))
from data import canonical, NAMES  # noqa: E402

MAXF = 196


def main():
    out = C.RUNS / "e1_full_event"; f = out / "humanml263_full.npz"
    if f.exists():
        raise FileExistsError(f)
    out.mkdir(parents=True, exist_ok=True); P.configure(); man = C.manifest()
    BASE = C.UMRM / "benchmark/judge_exploration/evaluations/astra_motion_loop/baseline636_20260915"
    refs = {r["id"]: r["primary_recording"] for r in json.loads((BASE / "pico_baseline_636.json").read_text())["items"]}
    NEW = C.UMRM_MODEL / "outputs/data_umrm925_20260927/umrm925_approved_20260927_132408"
    refs.update({r["id"]: str(NEW / r["pico_file"]) for r in csv.DictReader(open(NEW / "metadata/approved_manifest.csv", encoding="utf-8-sig"))})
    keys = man["train"] + man["val"]; feats = np.zeros((len(keys), MAXF, 263), np.float32); L = np.zeros(len(keys), int); dur = np.zeros(len(keys))
    for i, k in enumerate(keys):
        with np.load(refs[k], allow_pickle=False) as d:
            names = d["body_joint_names"].tolist(); x = d["raw_body_poses"][:, [names.index(n) for n in NAMES], :3]
            ns = d["raw_motion_timestamp_ns"].astype(np.int64); t = (ns - ns[0]) * 1e-9
        x = canonical(x); n = min(MAXF, int(np.floor(t[-1] * 20 + 1e-9)) + 1); g = np.arange(n) / 20
        y = np.stack([np.interp(g, t, x[:, j, c]) for j in range(24) for c in range(3)], -1).reshape(n, 24, 3)[:, :22].astype(np.float64)
        ft = P.to_features(y); assert np.isfinite(ft).all(), k
        feats[i, :n] = ft; L[i] = n; dur[i] = t[-1]
    with np.load(C.HVRUNS / "hero_data/humanml263.npz") as z:
        pos = {k: j for j, k in enumerate(z["ids"].tolist())}; ref, rl = z["features"], z["lengths"]
    diffs = [float(np.abs(feats[i, :rl[pos[k]] - 1] - ref[pos[k], :rl[pos[k]] - 1]).max()) for i, k in enumerate(keys) if rl[pos[k]] < 88]
    np.savez(f, ids=np.array(keys), features=feats, lengths=L, raw_dur_s=dur)
    rep = {"n": len(keys), "cap_frames": MAXF, "n_longer_than_88": int((L > 88).sum()), "n_at_196_cap": int((L == MAXF).sum()),
           "len_percentiles": {p: float(np.percentile(L, p)) for p in (50, 75, 90, 95, 99)},
           "max_abs_diff_vs_hero_ft_v860_on_uncapped(first L-1 frames)": float(np.max(diffs))}
    C.wjson(out / "report.json", rep); print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
