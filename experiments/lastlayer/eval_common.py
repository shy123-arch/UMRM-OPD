"""kimodo-env glue for the v1197 VAL: route-6 independent evaluator (official HumanML3D TMR + geometry; read-only) with
  * manifest -> this experiment's v1197 manifest (TRAIN/VAL rows only; categories from the frozen v1197 categories.csv),
  * Engine targets -> frozen data_v1197 data.npz (first 860 rows byte-identical to v860; evaluator constants unchanged),
  * LLM2Vec text cache writes -> this experiment's runs/ (route6 and v860 caches read-only)."""
import importlib.util, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
_cs = importlib.util.spec_from_file_location("vho_common", HERE / "common.py"); C = importlib.util.module_from_spec(_cs); _cs.loader.exec_module(C)
sys.path.insert(0, str(C.R6))
import r6_common as R6C  # noqa: E402
import r6_kimodo as R6K  # noqa: E402

R6K.RO_CACHES = list(R6K.RO_CACHES) + [R6K.MY_CACHE, C.V860 / "runs/texts_cache_llm2vec.npz"]
R6K.MY_CACHE = C.RUNS / "texts_cache_llm2vec.npz"
_MAN = C.manifest()
R6C.manifest = lambda: _MAN
COMPILER_STATS = C.RUNS / "compiler_stats.json"
_Base = R6K.Engine


class Engine(_Base):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        z = np.load(C.DATA / "data.npz"); ids = [str(x) for x in z["ids"]]
        assert ids[:len(self.ids)] == list(self.ids)
        self.ids = ids; self.pos = {k_: i for i, k_ in enumerate(ids)}; self.tg = z["targets"].astype(np.float32); self.mk = z["masks"].astype(bool)


R6K.Engine = Engine
METS = ["orig_cos", "orig_t2m_sim", "orig_m2t_rank", "orig_m2t_rank_samecat", "end_err_m", "yaw_err_deg", "mpjpe_m",
        "root_relative_mpjpe_m", "static", "body_speed"]


def summarize(per, keys):
    s = {m: float(sum(sum(x[m] for x in per[k]) / len(per[k]) for k in keys) / len(keys)) for m in METS}
    s["R@10"] = float(np.mean([np.mean([x["orig_m2t_rank"] <= 10 for x in per[k]]) for k in keys]))
    s["R@3_samecat"] = float(np.mean([np.mean([x["orig_m2t_rank_samecat"] <= 3 for x in per[k]]) for k in keys]))
    s["median_rank"] = float(np.median([np.mean([x["orig_m2t_rank"] for x in per[k]]) for k in keys]))
    return s
