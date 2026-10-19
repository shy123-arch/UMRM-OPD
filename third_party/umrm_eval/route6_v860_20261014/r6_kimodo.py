"""Kimodo-env helpers (1 visible GPU): LLM2Vec text cache, constrained generation, reward-side (TMR-SOMA) and independent
(official HumanML3D TMR + geometry) metrics. Wraps the read-only pilot engine; caches of v1 / pilot are read, never written."""
import fcntl, sys
import numpy as np, torch
from r6_common import *
from r6_common import _local_ok
sys.path.insert(0, str(PILOT))
import engine as EN
from engine import Engine, cos, DEV

MY_CACHE = RUNS / "texts_cache.npz"
RO_CACHES = [V1 / "runs/texts_pilot.npz", PILOT / "runs/texts_pilot.npz"]


class TextCache(EN.TextCache):
    def __init__(self):
        z = np.load(FT / "texts.npz"); self.d = {str(t): e for t, e in zip(z["texts"], z["emb"])}
        for p in RO_CACHES + [MY_CACHE]:
            if p.exists():
                z = np.load(p); self.d.update({str(t): e for t, e in zip(z["texts"], z["emb"])})
        self.pilot = MY_CACHE


EN.TextCache = TextCache  # Engine() now uses the merged cache
# byte-identical local copies of the LLM2Vec base/adapter (top-level files only) when complete (CPFS FUSE reads are slow)
_LB, _LA = "/tmp/r6_models/llm2vec/base", "/tmp/r6_models/llm2vec/adapter"
if _local_ok(EN.KT.TEXT_BASE, _LB) and _local_ok(EN.KT.TEXT_ADAPTER, _LA):
    EN.KT.TEXT_BASE, EN.KT.TEXT_ADAPTER = Path(_LB), Path(_LA)


def _encode_eqlen(enc, sani, max_bs=32):
    """Batch only texts with IDENTICAL token length (no padding -> avoids the padding corruption that forced batch_size=1;
    validated against batch_size=1 embeddings by check_eqlen.py before use)."""
    tok = enc.model.tokenizer; L = [len(tok(t)["input_ids"]) for t in sani]
    out = [None] * len(sani); by = {}
    for i, l in enumerate(L):
        by.setdefault(l, []).append(i)
    done = 0
    for l, idx in by.items():
        for c in range(0, len(idx), max_bs):
            ii = idx[c:c + max_bs]
            with torch.no_grad():
                e = enc.model.encode([sani[i] for i in ii], batch_size=len(ii), show_progress_bar=False, device=DEV)
            e = np.asarray(e.float().cpu() if torch.is_tensor(e) else e, np.float32)
            for j, i in enumerate(ii):
                out[i] = e[j]
            done += len(ii)
        print("enc", done, "/", len(sani), flush=True)
    return np.stack(out)


def encode_texts(texts, dtype="bfloat16", mode=None):
    """LLM2Vec-encode missing texts. mode 'bs1' (default in v1; internal batching with padding corrupts embeddings) or
    'eqlen' (equal-token-length batches, no padding). Default from env R6_ENC (bs1 unless validated)."""
    mode = mode or os.environ.get("R6_ENC", "bs1")
    tc = TextCache(); new = tc.missing([t for t in texts if t])
    if not new:
        return 0
    from kimodo.model.llm2vec import LLM2VecEncoder
    from kimodo.sanitize import sanitize_texts
    enc = LLM2VecEncoder(str(EN.KT.TEXT_BASE), str(EN.KT.TEXT_ADAPTER), dtype=dtype, llm_dim=4096, device=DEV)
    if mode == "eqlen":
        E = [_encode_eqlen(enc, sanitize_texts(new))]
    else:
        E = []
        for i in range(0, len(new), 64):
            with torch.no_grad():
                e = enc.model.encode(sanitize_texts(new[i:i + 64]), batch_size=1, show_progress_bar=False, device=DEV)
            E.append(np.asarray(e.float().cpu() if torch.is_tensor(e) else e, np.float32)); print("enc", min(i + 64, len(new)), "/", len(new), flush=True)
    del enc; torch.cuda.empty_cache(); E = np.concatenate(E)
    MY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(str(MY_CACHE) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        old_t, old_e = [], np.zeros((0, 4096), np.float32)
        if MY_CACHE.exists():
            z = np.load(MY_CACHE); old_t, old_e = list(z["texts"]), z["emb"]
        have = set(old_t); keep = [i for i, t in enumerate(new) if t not in have]
        tmp = str(MY_CACHE) + ".tmp.npz"
        np.savez(tmp, texts=np.array(list(old_t) + [new[i] for i in keep]), emb=np.concatenate([old_e, E[keep]]).astype(np.float32))
        os.replace(tmp, MY_CACHE); fcntl.flock(lk, fcntl.LOCK_UN)
    return len(new)


def gen(E, texts, nframes, specs, seed, steps=100, cfg=(2.0, 2.0), bs=32):
    """Identical to v1 rs_kimodo.gen. specs[i]: list of (frame_idx, xz, heading or None)."""
    from kimodo.constraints import Root2DConstraintSet
    from kimodo.postprocess import post_process_motion
    from kimodo.motion_rep.feature_utils import length_to_mask
    m, mr = E.m, E.m.motion_rep; out = []
    for s in range(0, len(texts), bs):
        tx, nf, sp = texts[s:s + bs], list(nframes[s:s + bs]), specs[s:s + bs]
        torch.manual_seed(seed + s); B = len(tx)
        tfeat = torch.tensor(E.tc.get(tx))[:, None]; tmask = torch.ones(B, 1, dtype=torch.bool)
        lengths = torch.tensor(nf, device=DEV); pad = length_to_mask(lengths)
        cons = [[Root2DConstraintSet(m.skeleton, torch.tensor(fr, dtype=torch.long), torch.tensor(xz, dtype=torch.float32),
                                     global_root_heading=None if hd is None else torch.tensor(hd, dtype=torch.float32)).to(DEV)
                 for fr, xz, hd in (c or [])] for c in sp]
        obs, msk = (mr.create_conditions_from_constraints_batched(cons, lengths, to_normalize=True, device=DEV)
                    if any(cons) else (None, None))
        with torch.inference_mode():
            mot = m._generate(None, int(lengths.max()), steps, pad, torch.zeros(B, device=DEV), msk, obs, list(cfg),
                              text_feat=tfeat.to(DEV), text_pad_mask=tmask.to(DEV), progress_bar=lambda x: x)
            o = mr.inverse(mot, is_normalized=True)
            o.update(post_process_motion(o["local_rot_mats"], o["root_positions"], o["foot_contacts"], m.skeleton, cons, root_margin=0.04))
            o = m.skeleton.output_to_SOMASkeleton77(o)
            P = o["posed_joints"]; J = EN.KC.map_joints_t(P)
        P, J = P.float().cpu().numpy(), J.float().cpu().numpy()
        out += [{"j77": P[i, :nf[i]], "j24": J[i, :nf[i]]} for i in range(B)]
    return out


def native_heading(j77):
    from kimodo.constraints import compute_global_heading
    from kimodo.skeleton import SOMASkeleton77
    hh = compute_global_heading(torch.tensor(j77[:1]), SOMASkeleton77()).reshape(-1, 2)[0].numpy()
    return float(np.arctan2(hh[1], hh[0]))


def theta0(E):
    return native_heading(gen(E, ["A person stands still."], [60], [None], seed=1)[0]["j77"])


class Metrics:
    """INDEPENDENT evaluator (never used for training): official HumanML3D TMR + geometry. Retrieval pool = gt3p texts of
    the evaluated split (full pool) and same-category sub-pool."""
    def __init__(self, E, keys, rows):
        self.E, self.pool, self.rows = E, list(keys), rows
        self.Ot = E.otmr_text([rows[k]["exec"] for k in self.pool]); self.idx = {k: i for i, k in enumerate(self.pool)}
        self.GTo = dict(zip(self.pool, E.otmr_motion([E.gt_c24(k) for k in self.pool])))
        self.gtspeed = {k: E.geom(E.tg[E.pos[k]], k)["body_speed"] for k in self.pool}
        self.cat = np.array([rows[k]["category"] for k in self.pool])

    def _sem(self, k, eo):
        S = self.Ot @ eo; i = self.idx[k]; same = self.cat == self.rows[k]["category"]
        return {"orig_m2t_rank": 1 + int((S > S[i]).sum()), "orig_m2t_rank_samecat": 1 + int((S[same] > S[i]).sum()),
                "samecat_pool": int(same.sum()), "orig_cos": cos(eo, self.GTo[k]), "orig_t2m_sim": float(S[i])}

    def motion(self, k, g):
        """g = Kimodo output {j77, j24 @30 fps}."""
        eo = self.E.otmr_motion([self.E.to20(g["j24"])])[0]; geo = self.E.geom(self.E.pico_eval(g["j24"]), k)
        return {**self._sem(k, eo), **geo, "static": float(geo["body_speed"] < 0.2 * self.gtspeed[k])}

    def canonical(self, k, ev88, valid):
        """ev88 = canonical 88x24x3 @20 Hz PICO-frame motion (retrieval / mean baselines)."""
        eo = self.E.otmr_motion([ev88[valid]])[0]; geo = self.E.geom(ev88, k)
        return {**self._sem(k, eo), **geo, "static": float(geo["body_speed"] < 0.2 * self.gtspeed[k])}
