"""Generator tool: Kimodo-SOMA-RP-v1.1 (text [+ root-2D trajectory constraint]) -> PICO24 motion.

Run with the kimodo env on ONE visible GPU (LLM2Vec multiprocessing deadlocks with several GPUs).
Reuses the pinned checkpoint of the audited astra_kimodo pilot (read only) and the local
Llama-3-8B + LLM2Vec adapter (fp32), same as benchmark/.../astra_kimodo_harness/backend.py.
Frames: canonical (x=body right, z=backward) -> Kimodo world (heading 0 faces +Z, right = -X):
x_k = -x_c, z_k = -z_c (verified by smoke()).
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("UMRM_ROOT", "/mnt/workspace/Wilson/ck/umrm"))
CHECKPOINT = Path(os.environ.get("VHO_KIMODO_CKPT", str(ROOT / "benchmark/judge_exploration/evaluations/astra_kimodo_pilot5_20260918/assets/Kimodo-SOMA-RP-v1.1")))
KIMODO_REPO = Path(os.environ.get("VHO_KIMODO_REPO", str(ROOT / "kimodo")))
TEXT_BASE = Path(os.environ.get("VHO_LLM2VEC_BASE", str(ROOT / "models/Meta-Llama-3-8B-Instruct")))
TEXT_ADAPTER = Path(os.environ.get("VHO_LLM2VEC_ADAPTER", str(ROOT / "models/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised")))
BODY = [0, 67, 72, 1, 68, 73, 2, 69, 74, 3, 70, 75, 4, 11, 39, 6, 12, 40, 13, 41, 14, 42]
FPS = 30
_MODEL = None


def map_joints(x):
    return np.concatenate([x[:, BODY], x[:, [15, 19, 24, 29, 34]].mean(1, keepdims=True),
                           x[:, [43, 47, 52, 57, 62]].mean(1, keepdims=True)], axis=1)


def load():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    os.environ.update(CHECKPOINT_DIR=str(CHECKPOINT.parent), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LOCAL_CACHE="true")
    sys.path.insert(0, str(KIMODO_REPO))
    from kimodo.model.llm2vec import LLM2VecEncoder
    from kimodo import load_model
    text = LLM2VecEncoder(str(TEXT_BASE), str(TEXT_ADAPTER), dtype="float32", llm_dim=4096, device="cuda:0")
    _MODEL = load_model("Kimodo-SOMA-RP-v1.1", device="cuda:0", text_encoder=text)
    assert _MODEL.fps == FPS
    return _MODEL


def root_constraint(path_c, frame_idx):
    """path_c: (K,2) canonical (x_right, z_back) root positions at 30 fps frame indices."""
    import torch
    from kimodo.constraints import Root2DConstraintSet
    m = load()
    xy = torch.tensor(-np.asarray(path_c), dtype=torch.float32)  # -> kimodo (x, z)
    return Root2DConstraintSet(m.skeleton, torch.tensor(frame_idx, dtype=torch.long), xy).to(m.device)


def generate(prompts, n_frames, paths=None, seed=0, steps=100, cfg=(2.0, 2.0), post=True):
    """prompts: list[str]; n_frames: list[int] @30fps; paths: optional list of None or (frame_idx, (K,2)).
    Returns list of (T_i, 24, 3) float32 PICO-ordered joints (y-up, metres, Kimodo world)."""
    import torch
    m = load()
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    cons = None
    if paths is not None and any(p is not None for p in paths):
        cons = [[root_constraint(p[1], p[0])] if p is not None else [] for p in paths]
    with torch.inference_mode():
        out = m(prompts=list(prompts), num_frames=list(n_frames), num_denoising_steps=steps,
                constraint_lst=cons or [], cfg_weight=list(cfg), post_processing=post,
                return_numpy=True, progress_bar=lambda x: x)
    pj = np.asarray(out["posed_joints"])
    return [map_joints(pj[i, :n_frames[i]]).astype(np.float32) for i in range(len(prompts))]


def smoke():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import plan
    from common import to_eval_frame
    fr = np.arange(0, 132, 6)
    tgt = np.stack([np.zeros(len(fr)), -np.linspace(0, 2.0, len(fr))], -1)  # 2 m forward
    tgt_r = np.stack([np.linspace(0, 1.5, len(fr)), np.zeros(len(fr))], -1)  # 1.5 m to the right
    outs = generate(["A person walks.", "A person walks.", "A person walks forward."], [132, 132, 132],
                    [(fr, tgt), (fr, tgt_r), None])
    for name, o in zip(["fwd2m", "right1.5m", "free"], outs):
        e = to_eval_frame(o, FPS); p = plan.extract(e, np.ones(88, bool))
        print(name, {k: round(p[k], 2) for k in ("forward_m", "right_m", "yaw_deg", "duration_s")}, flush=True)


if __name__ == "__main__":
    smoke()
