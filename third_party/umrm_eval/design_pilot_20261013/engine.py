"""Kimodo-env engine: batched Kimodo generation from cached LLM2Vec features, TMR-SOMA (reward evaluator),
official HumanML3D TMR (independent evaluator) and plan/geometry metrics. One visible GPU only."""
import os, sys, time
import numpy as np
import torch
from pilot_common import *

sys.path.insert(0, str(EXP / "kimodo_ft_v1"))
sys.path.insert(0, str(EXP / "generation_benchmark_v1"))
import kimodo_ft_common as KC  # sets CHECKPOINT_DIR for Kimodo, kimodo on sys.path
import kimodo_tool as KT
import plan as PL
from common import to_eval_frame, score

DEV = "cuda:0"


class TextCache:
    """text -> pooled LLM2Vec (fp32, sanitized) vector. Sources: FT cache (read-only) + pilot cache."""
    def __init__(self):
        z = np.load(FT / "texts.npz")
        self.d = {str(t): e for t, e in zip(z["texts"], z["emb"])}
        self.pilot = RUNS / "texts_pilot.npz"
        if self.pilot.exists():
            z = np.load(self.pilot)
            self.d.update({str(t): e for t, e in zip(z["texts"], z["emb"])})

    def missing(self, texts):
        return sorted({t for t in texts if t not in self.d})

    def get(self, texts):
        return np.stack([self.d[t] for t in texts])


def encode_new_texts(texts):
    """Encode texts with LLM2Vec (same config as kimodo_ft_v1/encode_texts.py); append to pilot cache."""
    tc = TextCache(); new = tc.missing(texts)
    if not new:
        return 0
    from kimodo.model.llm2vec import LLM2VecEncoder
    from kimodo.sanitize import sanitize_texts
    enc = LLM2VecEncoder(str(KT.TEXT_BASE), str(KT.TEXT_ADAPTER), dtype="float32", llm_dim=4096, device=DEV)
    E = []
    for i in range(0, len(new), 32):
        e, _ = enc(sanitize_texts(new[i:i + 32])); E.append(e[:, 0].float().cpu().numpy())
    del enc; torch.cuda.empty_cache()
    old_t, old_e = [], np.zeros((0, 4096), np.float32)
    if tc.pilot.exists():
        z = np.load(tc.pilot); old_t, old_e = list(z["texts"]), z["emb"]
    np.savez(tc.pilot, texts=np.array(list(old_t) + new), emb=np.concatenate([old_e, np.concatenate(E)]).astype(np.float32))
    return len(new)


class Engine:
    def __init__(self, with_kimodo=True):
        from kimodo import load_model
        from kimodo.skeleton import build_skeleton
        self.tc = TextCache()
        self.m = KC.load_kimodo(DEV) if with_kimodo else None
        # TMR-SOMA built directly from its config WITHOUT a raw text encoder (load_model would overwrite the
        # top text encoder when text_encoder= is passed, or instantiate Llama-3-8B otherwise). Texts are fed as
        # cached pooled LLM2Vec vectors through tmr.encode_text.
        from omegaconf import OmegaConf
        from kimodo.model.loading import instantiate_from_dict
        conf = OmegaConf.merge(OmegaConf.load(TMR_SOMA / "config.yaml"), OmegaConf.create({"checkpoint_dir": str(TMR_SOMA)}))
        conf = OmegaConf.to_container(conf, resolve=True); conf.pop("checkpoint_dir", None)
        self.tmr = instantiate_from_dict(conf, overrides={"device": DEV}).eval()
        assert self.tmr.raw_text_encoder is None
        self.sk77 = build_skeleton(77)
        from tmr_adapter import OfficialTMR, preprocess
        from setup_tmr import ASSETS
        self.otmr = OfficialTMR(ASSETS / "distilbert-base-uncased", device=DEV); self.prep = preprocess
        z = np.load(os.environ.get("VHO_EVAL_TARGETS", str(OUT / "data_v860_20260927/data.npz")))
        self.ids = list(z["ids"]); self.pos = {k: i for i, k in enumerate(self.ids)}
        self.tg, self.mk = z["targets"].astype(np.float32), z["masks"].astype(bool)
        f = np.load(FT / "fit/soma_fit.npz")
        self.fit = {k: i for i, k in enumerate(f["ids"])}; self.f = {k: f[k] for k in ("lengths", "local_rot6d", "root")}
        self.Lp = None

    # ------------------------------------------------------------------ generation
    def gen(self, texts, nframes, paths=None, seed=0, steps=100, cfg=(2.0, 2.0), bs=32):
        """Returns list of dicts {j77 (T,77,3) Kimodo world, j24 (T,24,3)}; 30 fps."""
        from kimodo.constraints import Root2DConstraintSet
        from kimodo.postprocess import post_process_motion
        from kimodo.motion_rep.feature_utils import length_to_mask
        m, mr = self.m, self.m.motion_rep
        paths = paths or [None] * len(texts); out_all = []
        for s in range(0, len(texts), bs):
            tx, nf, pa = texts[s:s + bs], list(nframes[s:s + bs]), paths[s:s + bs]
            torch.manual_seed(seed + s); B = len(tx)
            tfeat = torch.tensor(self.tc.get(tx))[:, None]; tmask = torch.ones(B, 1, dtype=torch.bool)
            lengths = torch.tensor(nf, device=DEV); pad = length_to_mask(lengths)
            cons = [[Root2DConstraintSet(m.skeleton, torch.tensor(p[0], dtype=torch.long),
                                         torch.tensor(p[1], dtype=torch.float32)).to(DEV)] if p is not None else [] for p in pa]
            obs, msk = (mr.create_conditions_from_constraints_batched(cons, lengths, to_normalize=True, device=DEV)
                        if any(cons) else (None, None))
            with torch.inference_mode():
                mot = m._generate(None, int(lengths.max()), steps, pad, torch.zeros(B, device=DEV), msk, obs, list(cfg),
                                  text_feat=tfeat.to(DEV), text_pad_mask=tmask.to(DEV), progress_bar=lambda x: x)
                o = mr.inverse(mot, is_normalized=True)
                o.update(post_process_motion(o["local_rot_mats"], o["root_positions"], o["foot_contacts"],
                                             m.skeleton, cons, root_margin=0.04))
                o = m.skeleton.output_to_SOMASkeleton77(o)
                P = o["posed_joints"]; J = KC.map_joints_t(P)
            P, J = P.float().cpu().numpy(), J.float().cpu().numpy()
            out_all += [{"j77": P[i, :nf[i]], "j24": J[i, :nf[i]]} for i in range(B)]
        return out_all

    # ------------------------------------------------------------------ ground truth
    def gt_len(self, k):
        return int(self.f["lengths"][self.fit[k]])

    def gt_j77(self, k):
        from kimodo.geometry import cont6d_to_matrix
        i = self.fit[k]; n = int(self.f["lengths"][i]); sk = self.m.skeleton if self.m else None
        if sk is None:
            raise RuntimeError("needs kimodo skeleton")
        r6 = torch.tensor(self.f["local_rot6d"][i, :n], device=DEV); rt = torch.tensor(self.f["root"][i, :n], device=DEV)
        R77 = sk.to_SOMASkeleton77(cont6d_to_matrix(r6))
        _, P, _ = sk.somaskel77.fk(R77, rt)
        return P.float().cpu().numpy()

    def gt_c24(self, k):
        i = self.pos[k]; return self.tg[i][self.mk[i]]  # canonical 20 Hz, valid frames

    # ------------------------------------------------------------------ TMR-SOMA (reward)
    def soma_motion(self, j77s):
        E = []
        for j in j77s:
            x = torch.tensor(j, device=DEV)[None]
            E.append(self.tmr.encode_motion(x, original_skeleton=self.sk77, lengths=torch.tensor([len(j)], device=DEV)).float().cpu().numpy()[0])
        return np.stack(E)

    def soma_text(self, texts):
        x = torch.tensor(self.tc.get(texts), device=DEV)[:, None]
        with torch.no_grad():
            e = self.tmr.encode_text({"x": x, "mask": torch.ones(len(texts), 1, dtype=torch.bool, device=DEV)})
        return e.float().cpu().numpy()

    # ------------------------------------------------------------------ official TMR (independent)
    @staticmethod
    def to20(j24_world30):
        x = np.asarray(j24_world30, np.float64); T = len(x); t = np.arange(T) / 30.0
        g = np.arange(int(np.floor(t[-1] * 20 + 1e-6)) + 1) / 20.0
        return np.stack([np.interp(g, t, x[:, j, c]) for j in range(24) for c in range(3)], -1).reshape(len(g), 24, 3)

    def otmr_motion(self, c24_20hz_list):
        return np.stack([self.otmr.encode_features(self.prep(np.asarray(x)[:, :22])) for x in c24_20hz_list])

    def otmr_text(self, texts):
        return np.concatenate([self.otmr.encode_text(texts[i:i + 64]) for i in range(0, len(texts), 64)])

    # ------------------------------------------------------------------ geometry
    def pico_eval(self, j24):
        if self.Lp is None and os.environ.get("VHO_EVAL_LP") and os.path.exists(os.environ["VHO_EVAL_LP"]):
            self.Lp = np.load(os.environ["VHO_EVAL_LP"])  # median PICO bone lengths of the original evaluator (released)
        if self.Lp is None:
            ks = [k for k in self.ids][:400]
            self.Lp = np.median(np.stack([KC.bone_lengths(self.tg[self.pos[k]], valid=self.mk[self.pos[k]]) for k in ks]), 0)
        return KC.to_pico_eval(j24, self.Lp)

    def geom(self, ev88, k):
        """ev88: canonical 88x24 20 Hz (held after end). Compare to GT k."""
        tgt, msk = self.tg[self.pos[k]], self.mk[self.pos[k]]
        s = score(ev88, tgt, msk)
        pg, pt = PL.extract(ev88, msk), PL.extract(tgt, msk)
        end_err = float(np.hypot(pg["forward_m"] - pt["forward_m"], pg["right_m"] - pt["right_m"]))
        yaw_err = float(abs((pg["yaw_deg"] - pt["yaw_deg"] + 180) % 360 - 180))
        speed = float(np.linalg.norm(np.diff(ev88[msk], axis=0), axis=-1).mean() * 20)
        return {**s, "end_err_m": end_err, "yaw_err_deg": yaw_err, "body_speed": speed}


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
