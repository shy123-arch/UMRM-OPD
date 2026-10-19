"""Score own-head generations (VAL) with the route-6 independent evaluator, exactly like route6 eval_hero.py (official
HumanML3D TMR on the generated frames, geometry on 88 frames held after the end, SMPL24 hands completed along the forearm
with the TRAIN-median ratio) but with per-item generated lengths. Input npz {val_ids, val_joints (K,N,88,22,3), lengths}.
Usage: eval_head.py --npz runs/gen/xxx.npz --name xxx"""
import argparse, json, os, time
import numpy as np
from eval_common import C, R6C, R6K, summarize

ELB, WR, HA = (18, 19), (20, 21), (22, 23)


def complete(j22, ratio):
    x = np.zeros(j22.shape[:-2] + (24, 3), np.float32); x[..., :22, :] = j22
    for s in (0, 1):
        x[..., HA[s], :] = j22[..., WR[s], :] + ratio[s] * (j22[..., WR[s], :] - j22[..., ELB[s], :])
    return x


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); ap.add_argument("--name", required=True); a = ap.parse_args()
    man = R6C.manifest(); rows = man["rows"]; keys = man[os.environ.get("VHO_SPLIT", "val")]
    z = np.load(C.resolve(a.npz)); ids = list(z["val_ids"]); J = z["val_joints"]; L = z["lengths"] if "lengths" in z.files else np.full(len(ids), 88)
    assert sorted(ids) == sorted(keys)
    E = R6K.Engine(with_kimodo=False); M = R6K.Metrics(E, keys, rows); ratio = []
    for s in (0, 1):
        r = [np.linalg.norm(E.tg[E.pos[k]][E.mk[E.pos[k]]][:, HA[s]] - E.tg[E.pos[k]][E.mk[E.pos[k]]][:, WR[s]], axis=-1).mean() /
             np.linalg.norm(E.tg[E.pos[k]][E.mk[E.pos[k]]][:, WR[s]] - E.tg[E.pos[k]][E.mk[E.pos[k]]][:, ELB[s]], axis=-1).mean() for k in man["train_real"]]
        ratio.append(float(np.median(r)))
    pos = {k: i for i, k in enumerate(ids)}; res = {}; t0 = time.time()
    for k in keys:
        valid = np.arange(88) < int(L[pos[k]])
        res[k] = [M.canonical(k, complete(J[s, pos[k]], ratio), valid) for s in range(J.shape[0])]
    summ = {"H": summarize(res, keys), "chance": {"R@10": 10 / len(keys), "median_rank": (len(keys) + 1) / 2}}
    C.save_json(C.RUNS / f"eval_head_{a.name}.json", {"name": a.name, "split": "val", "n": len(keys), "samples": int(J.shape[0]), "source_npz": a.npz,
                                                     "hand_ratio": ratio, "elapsed_s": time.time() - t0, "summary": summ, "per_gen": res})
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
