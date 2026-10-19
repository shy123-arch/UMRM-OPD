"""EXPLORATORY (post hoc, not pre-registered; README §7 note): full TMR similarity matrices for the G-T7a per-sample A36 readouts,
so that the rank of ANY generated motion against ANY VAL caption is available (needed for a random-candidate null of oracle@16).
Recomputes exactly what eval_head.py / Metrics.canonical do (hand completion, valid frames, official TMR), checks that the own-caption
ranks reproduce per_gen, and writes runs/t7_simmat_<m>.npy  [16 samples, 3 gens, 125 items, 125 captions] (float32).
Usage (kimodo env, 1 GPU): t7_simmat.py stu t8b"""
import json, sys, time
import numpy as np
from eval_common import C, R6C, R6K
from eval_head import complete, ELB, WR, HA


def main():
    man = R6C.manifest(); rows = man["rows"]; keys = man["val"]
    E = R6K.Engine(with_kimodo=False); M = R6K.Metrics(E, keys, rows); ratio = []
    for s in (0, 1):
        r = [np.linalg.norm(E.tg[E.pos[k]][E.mk[E.pos[k]]][:, HA[s]] - E.tg[E.pos[k]][E.mk[E.pos[k]]][:, WR[s]], axis=-1).mean() /
             np.linalg.norm(E.tg[E.pos[k]][E.mk[E.pos[k]]][:, WR[s]] - E.tg[E.pos[k]][E.mk[E.pos[k]]][:, ELB[s]], axis=-1).mean() for k in man["train_real"]]
        ratio.append(float(np.median(r)))
    for m in sys.argv[1:]:
        out = np.zeros((16, 3, len(keys), len(keys)), np.float32); t0 = time.time(); bad = 0
        for k in range(16):
            name = f"sc16_{m}_k{k:02d}"; z = np.load(C.RUNS / "gen" / f"{name}__A36.npz"); pos = {x: i for i, x in enumerate(z["val_ids"])}
            J, L = z["val_joints"], z["lengths"]; ref = json.load(open(C.RUNS / f"eval_head_{name}__A36.json"))["per_gen"]
            for ii, key in enumerate(keys):
                valid = np.arange(88) < int(L[pos[key]])
                eo = np.stack(E.otmr_motion([complete(J[s, pos[key]], ratio)[valid] for s in range(J.shape[0])]))
                S = eo @ M.Ot.T; out[k, :, ii] = S
                for s in range(3):
                    bad += int(1 + int((S[s] > S[s, ii]).sum()) != ref[key][s]["orig_m2t_rank"])
            print(m, k, round(time.time() - t0), "s; rank mismatches so far", bad, flush=True)
        np.save(C.RUNS / f"t7_simmat_{m}.npy", out); print(m, "saved; total own-rank mismatches", bad, "of", 16 * 3 * len(keys))


if __name__ == "__main__":
    main()
