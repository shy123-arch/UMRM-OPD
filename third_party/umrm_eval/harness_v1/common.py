"""Versioned data access for the UMRM clip -> reaction harness.

A *data version* bundles: item ids, experimental split, canonical PICO targets (88 frames @20 Hz,
PICO24, canonical() frame: +x body right, +y up, -z forward), masks, scene category (A-E),
source video id, annotation text and clip path. Formal benchmark files are read only.
New versions (e.g. the ~1000 / 3000 item releases) are added to VERSIONS once their motion cache
exists; every script takes --data-version so reruns need no code change.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "umrm_model/outputs"
FPS = 20
HORIZON = 88

VERSIONS = {
    "v636": {
        "motion_cache": OUT / "cliponly_v1_20260918/data.npz",
        "split": OUT / "cliponly_v1_20260918/split.json",
        "items": ROOT / "benchmark/umrm_final_636/items.jsonl",
        "manifest": ROOT / "benchmark/umrm_final_636_export/UMRM-636-v1_manifest.jsonl",
        "clips": ROOT / "benchmark/judge_exploration/evaluations/astra_motion_loop/baseline636_20260915/clip_inputs_636.json",
        "tcclip": OUT / "tcclip_cache_636_20260926/features",
        "note": "experimental split 384/56/84/112; test84 already viewed -> development evidence only",
    },
    "v860": {  # 636 unchanged + 224 approved new mocap records (umrm925 export 2026-09-27), all in group train_new
        "motion_cache": OUT / "data_v860_20260927/data.npz",
        "split": OUT / "data_v860_20260927/split.json",
        "items": OUT / "data_v860_20260927/items.jsonl",
        "manifest": OUT / "data_v860_20260927/manifest.jsonl",
        "clips": OUT / "data_v860_20260927/clips.json",
        "tcclip": OUT / "tcclip_cache_636_20260926/features",  # new items not encoded yet
        "note": "val/test identical to v636; new 224 -> train_new (12 source groups, no overlap with 636)",
    },
}
VERSIONS["v860h"] = {**VERSIONS["v860"], "split": OUT / "data_v860h_20260927/split.json",
                     "note": "v860 data; new-data groups desk5124:r1 + upload914:230cb7 + legacy singletons -> val_new (54)"}

_spec = importlib.util.spec_from_file_location("_cliponly_data", Path(__file__).resolve().parents[1] / "cliponly_v1/data.py")
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
canonical = _mod.canonical


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def score(motion, reference, valid):
    """Same definition as compare_hero_val.score (common22, masked frames)."""
    motion, reference = motion[:, :22], reference[:, :22]
    err = np.linalg.norm(motion[valid] - reference[valid], axis=-1).mean()
    local = np.linalg.norm((motion - motion[:, :1])[valid] - (reference - reference[:, :1])[valid], axis=-1).mean()
    root = np.linalg.norm(motion[valid, 0] - reference[valid, 0], axis=-1).mean()
    return {"mpjpe_m": float(err), "root_relative_mpjpe_m": float(local), "root_error_m": float(root)}


class Dataset:
    def __init__(self, version="v636"):
        cfg = VERSIONS[version]
        self.version, self.cfg = version, cfg
        with np.load(cfg["motion_cache"]) as z:
            self.ids = z["ids"].tolist()
            self.targets = z["targets"].astype(np.float32)
            self.masks = z["masks"].astype(bool)
        self.pos = {k: i for i, k in enumerate(self.ids)}
        self.split = json.loads(Path(cfg["split"]).read_text())
        items = {json.loads(l)["qid"]: json.loads(l) for l in Path(cfg["items"]).read_text().splitlines()}
        man = {json.loads(l)["item_id"]: json.loads(l) for l in Path(cfg["manifest"]).read_text().splitlines()}
        clips = {c["id"]: c["clip_path"] for c in json.loads(Path(cfg["clips"]).read_text())["items"]}
        self.meta = {}
        for k in self.ids:
            it, m = items[k], man[k]
            self.meta[k] = {
                "id": k, "split": self.split[k], "category": it["scene"], "scene_name": it["scene_name"],
                "source": m["source_video_id"], "clip_path": clips[k],
                "reason": it["mental_reasoning_en"], "reaction": it["reaction_en"],
                "motion_prompt": it["motion_prompt_en"], "reaction_cn": it.get("reaction_cn"), "options": it["options"],
                "option_type": it.get("option_type"), "correct": it["correct"],
            }

    def keys(self, *groups):
        return [k for k in self.ids if self.split[k] in groups]

    def target(self, k):
        i = self.pos[k]
        return self.targets[i], self.masks[i]


def to_eval_frame(joints24, fps_in):
    """Any generated PICO24-ordered motion (T,24,3, y-up, metres) -> canonical 88x20Hz."""
    x = canonical(joints24)
    t = np.arange(len(x)) / fps_in
    new = np.arange(HORIZON) / FPS
    new_c = np.minimum(new, t[-1])  # hold last pose past the end
    y = np.stack([np.interp(new_c, t, x[:, j, c]) for j in range(x.shape[1]) for c in range(3)], -1)
    return y.reshape(HORIZON, x.shape[1], 3).astype(np.float32)
