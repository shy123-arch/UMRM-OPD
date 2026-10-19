import os
"""Shared paths/helpers for the TRAINING_DESIGN small-scale validation pilot (2026-10-13).
Read-only on all formal / historical artifacts; every output goes under ./runs/."""
import hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
UMRM = EXP.parents[1]
OUT = Path(os.environ.get("UMRM_OUTPUTS", "/mnt/workspace/Wilson/ck/umrm/umrm_model/outputs"))
RUNS = HERE / "runs"
FT = Path(os.environ.get("VHO_EVAL_FT", str(OUT / "kimodo_ft_v1_v860_20260927")))  # texts.npz (LLM2Vec cache), fit/soma_fit.npz
TMR_SOMA = Path(os.environ.get("VHO_TMR_SOMA", "/mnt/workspace/Wilson/ck/umrm/umrm_model/experiments/design_pilot_20261013/assets/TMR-SOMA-RP-v1"))
SALT = "umrm-design-pilot-20261013"
sys.path.insert(0, str(EXP / "harness_v1"))


def h(k):
    return hashlib.sha256((SALT + k).encode()).hexdigest()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_items():
    split = json.loads((OUT / "data_v860h_20260927/split.json").read_text())
    items = {json.loads(l)["qid"]: json.loads(l) for l in (OUT / "data_v860_20260927/items.jsonl").read_text().splitlines()}
    clips = {c["id"]: c["clip_path"] for c in json.loads((OUT / "data_v860_20260927/clips.json").read_text())["items"]}
    prompts = json.loads((FT / "prompts.json").read_text())
    return split, items, clips, prompts


def save_json(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1))


def wjson(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
