"""Shared helpers for the Kimodo fine-tune (kimodo env)."""
import os, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness_v1"))
import kimodo_tool as KT  # noqa: E402

os.environ.update(CHECKPOINT_DIR=str(KT.CHECKPOINT.parent), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LOCAL_CACHE="true")
sys.path.insert(0, str(KT.KIMODO_REPO))
BODY = torch.tensor(KT.BODY)
LH, RH = [15, 19, 24, 29, 34], [43, 47, 52, 57, 62]


class NoText:
    """Placeholder text encoder: all prompts are pre-encoded (texts.npz)."""
    def __call__(self, texts):
        raise RuntimeError("use cached text features")

    def to(self, *a, **k):
        return self

    def eval(self):
        return self


def load_kimodo(device="cuda:0"):
    from kimodo import load_model
    return load_model("Kimodo-SOMA-RP-v1.1", device=device, text_encoder=NoText())


def map_joints_t(x77):
    """torch version of kimodo_tool.map_joints: (...,77,3) -> (...,24,3)."""
    return torch.cat([x77[..., BODY.to(x77.device), :], x77[..., LH, :].mean(-2, keepdim=True),
                      x77[..., RH, :].mean(-2, keepdim=True)], -2)


def canon_to_kimodo(x):
    """canonical (+x right, -z forward) -> Kimodo world (heading 0 faces +Z): negate x and z."""
    y = np.array(x, dtype=np.float32, copy=True); y[..., 0] *= -1; y[..., 2] *= -1
    return y


def resample(x, valid, fps_in=20, fps_out=30):
    T = int(valid.sum()); t = np.arange(T) / fps_in
    n = int(np.floor((T - 1) / fps_in * fps_out + 1e-6)) + 1; tn = np.arange(n) / fps_out
    J = x.shape[1]
    y = np.stack([np.interp(tn, t, x[:T, j, c]) for j in range(J) for c in range(3)], -1).reshape(n, J, 3)
    return y.astype(np.float32)


# ---------------------------------------------------------------- bone-length retargeting (PICO24 tree)
# SMPL-22 parents + hand proxies (22 <- L wrist 20, 23 <- R wrist 21)
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def bone_lengths(J, valid=None):
    """J (...,T,24,3) -> median bone length per joint (24,), [0]=0."""
    J = np.asarray(J); J = J[valid] if valid is not None else J.reshape(-1, 24, 3)
    L = np.zeros(24, np.float32)
    for j, p in enumerate(PARENTS):
        if p >= 0:
            L[j] = np.median(np.linalg.norm(J[:, j] - J[:, p], axis=-1))
    return L


def retarget(J, L):
    """Rebuild a (T,24,3) motion along the PICO24 tree with bone lengths L, keeping the pelvis trajectory and
    every bone direction; then shift y so the frame-0 lowest joint keeps its height (floor contact)."""
    J = np.asarray(J, np.float32); out = J.copy()
    for j, p in enumerate(PARENTS):
        if p >= 0:
            d = J[:, j] - J[:, p]; d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-8)
            out[:, j] = out[:, p] + L[j] * d
    out[..., 1] += J[0, :, 1].min() - out[0, :, 1].min()
    return out


def soma_lengths(model):
    """Bone lengths of the Kimodo SOMA body measured through map_joints in the rest pose."""
    R = torch.eye(3, device=model.device).repeat(1, 30, 1, 1)
    sk = model.skeleton
    _, P, _ = sk.somaskel77.fk(sk.to_SOMASkeleton77(R), torch.tensor([[0, 0.9, 0.]], device=model.device))
    return bone_lengths(map_joints_t(P).cpu().numpy())


def pico_lengths(d):
    """PICO24 bone lengths (median over train items; the PICO data use one fixed skeleton, std < 2 mm)."""
    return np.median(np.stack([bone_lengths(*d.target(k)[:1], valid=d.target(k)[1]) for k in d.keys("train")]), 0)


def to_pico_eval(joints24_kimodo, L_pico, fps=30):
    """Generated (T,24,3) Kimodo-world PICO-ordered joints -> retargeted onto the PICO skeleton -> eval frame."""
    from common import to_eval_frame
    return to_eval_frame(retarget(joints24_kimodo, L_pico), fps)
