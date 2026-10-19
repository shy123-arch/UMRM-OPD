"""Median PICO bone lengths used by the route-6 evaluator geometry (engine.pico_eval: first 400 ids of data_v860),
saved so the released evaluator reproduces the original numbers exactly (VHO_EVAL_LP). kimodo env, CPU."""
import sys
from pathlib import Path
import numpy as np
T = Path(__file__).resolve().parents[1] / "third_party/umrm_eval"
sys.path.insert(0, str(T / "kimodo_ft_v1")); sys.path.insert(0, str(T / "harness_v1"))
import kimodo_ft_common as KC  # noqa: E402
z = np.load("/mnt/workspace/Wilson/ck/umrm/umrm_model/outputs/data_v860_20260927/data.npz")
tg, mk = z["targets"].astype(np.float32), z["masks"].astype(bool)
Lp = np.median(np.stack([KC.bone_lengths(tg[i], valid=mk[i]) for i in range(400)]), 0)
out = Path(sys.argv[1]); out.parent.mkdir(parents=True, exist_ok=True); np.save(out, Lp); print(out, Lp.shape, Lp[:5])
