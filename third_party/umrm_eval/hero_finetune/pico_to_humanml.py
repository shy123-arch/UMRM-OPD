"""Convert the audited 20 Hz PICO cache (SMPL22 subset) into HumanML3D 263-dim features.

No uniform_skeleton retargeting: HumanML's target skeleton (sample 000021) is not
available locally, and keeping the PICO skeleton keeps the ground-truth scale so the
feature -> joints round trip is (numerically) exact.  HERO's VQ is fine-tuned on these
features, so the model adapts to PICO bone lengths.

The last valid frame is duplicated once so that T valid frames produce T feature rows
(process_file returns T-1 rows).  Output is read-only derived data in a new directory.
Run: /usr/bin/python run_hero_env.py hero_finetune/pico_to_humanml.py --output DIR
"""
import argparse
import os
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERO = Path(os.environ.get("VHO_HERO", str(ROOT / "HERO")))
RUN = ROOT / "umrm_model/outputs/cliponly_v1_20260918"
DATA_SHA = "570873890d578a19f41b4b21f11b61d47a4761d8a2563d391e79bb6e2c3bbdc1"
SPLIT_SHA = "ad135b2f91820f85121bde04988b3a557c66d7b62e2c5382ada1a2c93665c6fb"
HERO_ITEMS = ROOT / "benchmark/judge_exploration/evaluations/hero_astra_636_20260918/items"
sys.path.insert(0, str(HERO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cliponly_v1"))

import torch  # noqa: E402
import utils.motion_process as mp  # noqa: E402
from utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain  # noqa: E402
from data import canonical  # noqa: E402

EDGES = [(c[i], c[i + 1]) for c in t2m_kinematic_chain for i in range(len(c) - 1)]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def configure():
    mp.l_idx1, mp.l_idx2 = 5, 8
    mp.fid_r, mp.fid_l = [8, 11], [7, 10]
    mp.face_joint_indx = [2, 1, 17, 16]
    mp.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    mp.kinematic_chain = t2m_kinematic_chain
    mp.tgt_offsets = None
    mp.uniform_skeleton = lambda positions, target: positions  # keep PICO skeleton


def to_features(joints):
    """joints (T,22,3) y-up meters -> (T,263) HumanML features."""
    padded = np.concatenate([joints, joints[-1:]], 0).astype(np.float64)
    data, _, _, _ = mp.process_file(padded.copy(), 0.002)
    assert data.shape == (len(joints), 263), data.shape
    return data.astype(np.float32)


def to_joints(features):
    return mp.recover_from_ric(torch.from_numpy(np.asarray(features)).float(), 22).numpy()


def bone_lengths(x):
    return np.stack([np.linalg.norm(x[..., a, :] - x[..., b, :], axis=-1) for a, b in EDGES], -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    if sha(RUN / "data.npz") != DATA_SHA or sha(RUN / "split.json") != SPLIT_SHA:
        raise ValueError("audited cache or split changed")
    configure()
    split = json.loads((RUN / "split.json").read_text())
    with np.load(RUN / "data.npz", allow_pickle=False) as z:
        ids = z["ids"].tolist()
        targets = z["targets"][:, :, :22].astype(np.float64)
        masks = z["masks"].astype(bool)
    n = len(ids)
    feats = np.zeros((n, 88, 263), np.float32)
    lengths = masks.sum(1).astype(np.int64)
    errors = []
    for i in range(n):
        L = lengths[i]
        f = to_features(targets[i, :L])
        if not np.isfinite(f).all():
            raise ValueError(f"non-finite features {ids[i]}")
        feats[i, :L] = f
        rec = canonical(to_joints(f))
        errors.append(float(np.linalg.norm(rec - targets[i, :L], axis=-1).mean()))
    errors = np.array(errors)
    # Bone length comparison: PICO GT vs HERO zero-shot candidate0 (HumanML skeleton).
    pico_bones = np.concatenate([bone_lengths(targets[i, :lengths[i]]) for i in range(n)]).mean(0)
    hero_bones = []
    for key in ids[:100]:
        folder = HERO_ITEMS / hashlib.sha256(key.encode()).hexdigest()[:20]
        hero_bones.append(bone_lengths(np.load(folder / "candidate0.npy")))
    hero_bones = np.median(np.concatenate(hero_bones), 0)
    out.mkdir(parents=True)
    np.savez(out / "humanml263.npz", features=feats, lengths=lengths, ids=np.array(ids))
    groups = {g: [k for k in ids if split[k] == g] for g in ["train", "val", "test", "unknown_source"]}
    train_idx = [i for i, k in enumerate(ids) if split[k] == "train"]
    stack = np.concatenate([feats[i, :lengths[i]] for i in train_idx])
    np.savez(out / "pico_train_stats.npz", mean=stack.mean(0), std=stack.std(0))
    report = {
        "source_data_sha256": DATA_SHA, "split_sha256": SPLIT_SHA,
        "features_sha256": sha(out / "humanml263.npz"),
        "uniform_skeleton": False, "feet_threshold": 0.002, "last_frame_duplicated": True,
        "counts": {g: len(v) for g, v in groups.items()},
        "roundtrip_mpjpe_m": {"mean": float(errors.mean()), "max": float(errors.max())},
        "bones": {"edges": EDGES, "pico_mean_m": pico_bones.round(4).tolist(),
                  "hero_candidate0_median_m": hero_bones.round(4).tolist(),
                  "pico_over_hero_ratio_median": float(np.median(pico_bones / hero_bones))},
        "length_frames": {"min": int(lengths.min()), "median": float(np.median(lengths)), "max": int(lengths.max())},
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ["roundtrip_mpjpe_m", "counts", "length_frames"]}),
          report["bones"]["pico_over_hero_ratio_median"])


if __name__ == "__main__":
    main()
