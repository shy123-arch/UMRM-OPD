"""EXPLORATORY (post hoc; README §7 note): random-candidate null for oracle@16. For VAL item i the 16 candidates are motions generated
from samples of 16 random OTHER VAL items (different source than i); their ranks against caption i give the oracle@16 that pure
"16 shots" would reach without any item-specific information. Same naive / cross-fitted rules as decide_t7.py. Writes runs/t7_null.json."""
import json
import numpy as np
from common import *

LL = EXP / "vt_lastlayer_v1197aug/runs"; N = 16; D = 200


def ranks(S, i, j, k):  # ranks (3 gens) of the motions of item j / sample k against caption i
    s = S[k, :, j]; return 1 + (s > s[:, [i]]).sum(1)


def main():
    man = manifest(); ids = list(man["val"]); src = np.array([man["rows"][k]["source"] for k in ids]); n = len(ids); out = {}
    for m in ("stu", "t8b"):
        S = np.load(LL / f"t7_simmat_{m}.npy", mmap_mode="r"); S = np.asarray(S)
        own_r = np.stack([[ranks(S, i, i, k) for k in range(N)] for i in range(n)])          # [item, sample, gen]
        own_naive = (own_r <= 10).mean(2).max(1)
        own_xfit = np.array([(own_r[i, np.lexsort((np.arange(N), own_r[i, :, 0]))[0], 1:] <= 10).mean() for i in range(n)])
        own_single = (own_r <= 10).mean((1, 2))
        rng = np.random.default_rng(0); nn, nx, ns = np.zeros((D, n)), np.zeros((D, n)), np.zeros((D, n))
        for d in range(D):
            for i in range(n):
                pool = np.where(src != src[i])[0]; js = rng.choice(pool, N, replace=False); ks = rng.integers(0, N, N)
                r = np.stack([ranks(S, i, j, k) for j, k in zip(js, ks)])                    # [cand, gen]
                nn[d, i] = (r <= 10).mean(1).max(); nx[d, i] = (r[np.lexsort((np.arange(N), r[:, 0]))[0], 1:] <= 10).mean(); ns[d, i] = (r <= 10).mean()
        null_naive, null_xfit, null_single = nn.mean(0), nx.mean(0), ns.mean(0)
        us = np.unique(src); idx = {s: np.where(src == s)[0] for s in us}; b = np.random.default_rng(1)
        def ci(x):
            bs = [x[np.concatenate([idx[s] for s in b.choice(us, len(us))])].mean() for _ in range(4000)]
            return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
        out[m] = {"own": {"naive": round(float(own_naive.mean()), 4), "xfit": round(float(own_xfit.mean()), 4), "single_mean16": round(float(own_single.mean()), 4)},
                  "null_random_other_items": {"naive": round(float(null_naive.mean()), 4), "xfit": round(float(null_xfit.mean()), 4), "single": round(float(null_single.mean()), 4),
                                              "xfit_draw_sd": round(float(nx.mean(1).std()), 4)},
                  "excess_over_null": {"naive": [round(float((own_naive - null_naive).mean()), 4), ci(own_naive - null_naive)],
                                       "xfit": [round(float((own_xfit - null_xfit).mean()), 4), ci(own_xfit - null_xfit)],
                                       "single": [round(float((own_single - null_single).mean()), 4), ci(own_single - null_single)]}}
        print(m, json.dumps(out[m]), flush=True)
    wjson(RUNS / "t7_null.json", {"note": "exploratory, post hoc; random-candidate null for oracle@16 (README §7 note)", "draws": D, **out})


if __name__ == "__main__":
    main()
