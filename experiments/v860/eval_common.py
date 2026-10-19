"""kimodo-env glue: import the route-6 independent evaluator (official HumanML3D TMR + geometry; read-only) and redirect
every write (LLM2Vec text cache, eval json) into this experiment's runs/ directory."""
import importlib.util, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
_cs = importlib.util.spec_from_file_location("vho_common", HERE / "common.py"); C = importlib.util.module_from_spec(_cs); _cs.loader.exec_module(C)
sys.path.insert(0, str(C.R6))
import r6_common as R6C  # noqa: E402
import r6_kimodo as R6K  # noqa: E402

R6K.RO_CACHES = list(R6K.RO_CACHES) + [R6K.MY_CACHE]          # route-6 cache: read only
R6K.MY_CACHE = C.RUNS / "texts_cache_llm2vec.npz"              # new texts are appended here, never to route 6
METS = ["orig_cos", "orig_t2m_sim", "orig_m2t_rank", "orig_m2t_rank_samecat", "end_err_m", "yaw_err_deg", "mpjpe_m",
        "root_relative_mpjpe_m", "static", "body_speed"]


def summarize(per, keys):
    s = {m: float(sum(sum(x[m] for x in per[k]) / len(per[k]) for k in keys) / len(keys)) for m in METS}
    import numpy as np
    s["R@10"] = float(np.mean([np.mean([x["orig_m2t_rank"] <= 10 for x in per[k]]) for k in keys]))
    s["R@3_samecat"] = float(np.mean([np.mean([x["orig_m2t_rank_samecat"] <= 3 for x in per[k]]) for k in keys]))
    s["median_rank"] = float(np.median([np.mean([x["orig_m2t_rank"] for x in per[k]]) for k in keys]))
    return s
