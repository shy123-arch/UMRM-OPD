"""B3 (hero env, CPU): HumanML263 motion targets for v1197 TRAIN+VAL (TEST / test_extra / sealed never read):
  capped: the frozen 88-frame 20 Hz targets of data_v1197 (= hero_ft_v860 conversion, checked on the v860 items)
  full  : complete events from the raw PICO recordings (no 88-frame cap; <= 196 frames = 9.8 s), same canonicalisation
          (cliponly_v1.canonical) and conversion (hero_finetune/pico_to_humanml.py, no uniform_skeleton).
Output: runs/motion/{humanml263_capped.npz, humanml263_full.npz, report.json}"""
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
    out = C.RUNS / "motion"
    if (out / "report.json").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True); P.configure()
    inv = C.rjson(C.RUNS / "inventory.json")["rows"]; keys = sorted(k for k, r in inv.items() if r["group"] in ("train", "val"))
    BASE = C.UMRM / "benchmark/judge_exploration/evaluations/astra_motion_loop/baseline636_20260915"
    refs = {r["id"]: r["primary_recording"] for r in json.loads((BASE / "pico_baseline_636.json").read_text())["items"]}
    refs.update({json.loads(l)["id"]: json.loads(l)["pico_file"] for l in (C.DATA / "provenance_new.jsonl").read_text().splitlines()})
    z = np.load(C.DATA / "data.npz"); pos = {str(k): i for i, k in enumerate(z["ids"])}
    cap = np.zeros((len(keys), 88, 263), np.float32); Lc = np.zeros(len(keys), int)
    full = np.zeros((len(keys), MAXF, 263), np.float32); Lf = np.zeros(len(keys), int); dur = np.zeros(len(keys))
    for i, k in enumerate(keys):
        m = z["masks"][pos[k]].astype(bool); L = int(m.sum()); f = P.to_features(z["targets"][pos[k], :L, :22].astype(np.float64))
        assert np.isfinite(f).all(), k; cap[i, :L] = f; Lc[i] = L
        with np.load(refs[k], allow_pickle=False) as d:
            names = d["body_joint_names"].tolist(); x = d["raw_body_poses"][:, [names.index(n) for n in NAMES], :3]
            ns = d["raw_motion_timestamp_ns"].astype(np.int64); t = (ns - ns[0]) * 1e-9
        x = canonical(x); n = min(MAXF, int(np.floor(t[-1] * 20 + 1e-9)) + 1); g = np.arange(n) / 20
        y = np.stack([np.interp(g, t, x[:, j, c]) for j in range(24) for c in range(3)], -1).reshape(n, 24, 3)[:, :22].astype(np.float64)
        ff = P.to_features(y); assert np.isfinite(ff).all(), k; full[i, :n] = ff; Lf[i] = n; dur[i] = t[-1]
    with np.load(C.HVRUNS / "hero_data/humanml263.npz") as h:
        hp = {str(k): j for j, k in enumerate(h["ids"])}; hf, hl = h["features"], h["lengths"]
    d860 = [float(np.abs(cap[i, :Lc[i]] - hf[hp[k], :Lc[i]]).max()) for i, k in enumerate(keys) if k in hp]
    dfull = [float(np.abs(full[i, :Lc[i] - 1] - cap[i, :Lc[i] - 1]).max()) for i in range(len(keys)) if Lc[i] < 88]
    np.savez(out / "humanml263_capped.npz", ids=np.array(keys), features=cap, lengths=Lc)
    np.savez(out / "humanml263_full.npz", ids=np.array(keys), features=full, lengths=Lf, raw_dur_s=dur)
    rep = {"n": len(keys), "capped_vs_hero_ft_v860_maxdiff(v860 items)": max(d860), "n_v860_checked": len(d860),
           "full_vs_capped_maxdiff(uncapped items, first L-1 frames)": max(dfull), "n_longer_than_88": int((Lf > 88).sum()),
           "n_at_196": int((Lf == MAXF).sum()), "full_len_pct": {p: float(np.percentile(Lf, p)) for p in (50, 90, 99)}}
    C.wjson(out / "report.json", rep); print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
