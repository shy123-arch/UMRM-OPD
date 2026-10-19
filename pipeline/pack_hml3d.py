"""Pack HumanML3D new_joint_vecs (read-only source) into one local file CACHE/hml3d_motion.npz:
names, offsets, lengths, feats (float32 [sum T, 263]). CPU, threaded (CPFS small-file latency)."""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from common import *


def main():
    out = CACHE / "hml3d_motion.npz"
    if out.exists():
        print("exists"); return
    names = sorted(set(l.strip() for s in ("train", "val", "test") for l in (HML3D / f"{s}.txt").read_text().splitlines() if l.strip()))
    t0 = time.time()
    def rd(n):
        p = HML3D / "new_joint_vecs" / f"{n}.npy"
        return n, (np.load(p).astype(np.float32) if p.exists() else None)
    with ThreadPoolExecutor(48) as ex:
        R = [x for x in ex.map(rd, names) if x[1] is not None and np.isfinite(x[1]).all()]
    lens = np.array([len(x[1]) for x in R]); off = np.concatenate([[0], np.cumsum(lens)[:-1]])
    np.savez(out, names=np.array([x[0] for x in R]), offsets=off, lengths=lens, feats=np.concatenate([x[1] for x in R]))
    print("packed", len(R), "of", len(names), "frames", int(lens.sum()), "s", round(time.time() - t0))


if __name__ == "__main__":
    main()
