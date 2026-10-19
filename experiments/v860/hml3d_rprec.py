"""G-B self-check (kimodo env): official HumanML3D TMR R-precision (motion->text, batches of 32: 1 GT caption + 31 others,
standard protocol) for own-head generations vs the same-VQ-path GT motions on a fixed HumanML3D test subset.
Input npz from head.py hml3d_gen {texts, lengths, gen (N,196,22,3), gt}. Usage: hml3d_rprec.py --npz runs/x.npz --name x"""
import argparse, json
import numpy as np
from eval_common import C, R6K


def rprec(M, T, bs=32, seed=0):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(M)); r = np.zeros(3); n = 0
    for s in range(0, len(idx) - bs + 1, bs):
        ii = idx[s:s + bs]; S = M[ii] @ T[ii].T
        rank = (S > np.diag(S)[:, None]).sum(1)
        r += [(rank < k).sum() for k in (1, 2, 3)]; n += len(ii)
    return (r / n).tolist()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); ap.add_argument("--name", required=True); a = ap.parse_args()
    z = np.load(C.HERE / a.npz); L = z["lengths"]; E = R6K.Engine(with_kimodo=False)
    T = E.otmr_text(list(z["texts"])); G = E.otmr_motion([z["gen"][i, :L[i]] for i in range(len(L))]); R = E.otmr_motion([z["gt"][i, :L[i]] for i in range(len(L))])
    nz = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    T, G, R = nz(T), nz(G), nz(R)
    out = {"n": int(len(L)), "gen_R@1/2/3": rprec(G, T), "gt_vqpath_R@1/2/3": rprec(R, T), "random_R@1/2/3": [1 / 32, 2 / 32, 3 / 32],
           "gen_matching_cos": float((G * T).sum(1).mean()), "gt_matching_cos": float((R * T).sum(1).mean())}
    out["G-B_ratio_R@3"] = out["gen_R@1/2/3"][2] / out["gt_vqpath_R@1/2/3"][2]
    C.save_json(C.RUNS / f"hml3d_rprec_{a.name}.json", out); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
