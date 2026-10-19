"""Train-mean baseline on the v1197 VAL (kimodo env; = route6 b1_baselines 'train_mean': frame-wise mean of hold-padded
TRAIN motions, valid length = median TRAIN length). Writes runs/eval_base_train_mean.json in the eval_head format (arm H)."""
import json
import numpy as np
from eval_common import C, R6C, R6K, summarize


def main():
    man = R6C.manifest(); rows = man["rows"]; keys = man["val"]
    E = R6K.Engine(with_kimodo=False); M = R6K.Metrics(E, keys, rows)

    def tg(k):
        x = E.tg[E.pos[k]].copy(); m = E.mk[E.pos[k]]; L = int(m.sum()); x[L:] = x[L - 1]; return x
    mean = np.mean([tg(k) for k in man["train_real"]], 0); Lmed = int(np.median([E.mk[E.pos[k]].sum() for k in man["train_real"]])); vm = np.arange(88) < Lmed
    res = {k: [M.canonical(k, mean, vm)] for k in keys}
    C.save_json(C.RUNS / "eval_base_train_mean.json", {"name": "train_mean", "n": len(keys), "summary": {"H": summarize(res, keys)}, "per_gen": res})
    print(json.dumps(summarize(res, keys), indent=1))


if __name__ == "__main__":
    main()
