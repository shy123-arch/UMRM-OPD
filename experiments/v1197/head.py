"""T->M action head (hero env, 1 GPU): HERO/MoMask-style frozen RVQ (6 layers x 512 codes, /4 temporal) + masked
transformer + residual transformer (initialised from HERO ViMo checkpoints; HERO/ is imported read-only), conditioned on
a frozen Qwen3-VL-4B text encoder (textenc.py caches) through QwenAdapter: per-layer LayerNorm -> softmax layer mix ->
MLP to 512 -> masked attention pooling into 16 memory slots + masked-mean global vector (HERO's (global, memory) slots).
CFG fix: HERO drops only the global vector for the unconditional branch while memory stays visible; here the drop mask
also zeroes memory (train-time drop and inference null condition are the same).
Motion = HumanML263 @20 Hz normalised with the ViMo mean/std (same as hero_ft_v860), decoded joints = canonical
HumanML22 in the PICO frame (FH.to_canonical_joints), i.e. directly consumable by the route-6 independent evaluator.
Stages:
  vqcheck  : recon MPJPE of candidate VQs on HumanML3D val / v860 val
  pretrain : HumanML3D train (captions incl. #from#to segments); select lowest val (mask CE + res CE)
  finetune : v860 TRAIN with gt3p <exec> features; select lowest v860 VAL (mask CE + res CE)   [TEST never loaded]
  gen      : generate VAL from a text-feature corpus (gt3p or VLM predictions) with given lengths -> npz
  hml3d_gen: generate a fixed HumanML3D test subset (G-B self-check) -> npz"""
import argparse, importlib.util, json, math, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
# load our common.py under another name: HERO has its own top-level package called "common"
_cs = importlib.util.spec_from_file_location("vho_common", HERE / "common.py"); _C = importlib.util.module_from_spec(_cs); _cs.loader.exec_module(_C)
globals().update({k: v for k, v in vars(_C).items() if not k.startswith("__")})

_spec = importlib.util.spec_from_file_location("FH", EXP / "hero_finetune/finetune_hero.py")
FH = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(FH)  # NOTE: chdirs into HERO (read-only)
torch = FH.torch; nn = torch.nn; F = torch.nn.functional
import models.mask_transformer.transformer_memo_cross as TMC  # noqa: E402  (HERO, read-only)

VQ_CANDIDATES = {"vimo": None, "v860": HVRUNS / "vq/best.pt"}
LAYERS = (9, 18, 27, 36)


# ------------------------------------------------------------------ CFG-consistent condition dropout (patch)
def _mask_cond(self, cond, force_mask=False):
    bs = cond.shape[0]
    if force_mask:
        m = torch.ones(bs, 1, device=cond.device)
    elif self.training and self.cond_drop_prob > 0:
        m = torch.bernoulli(torch.full((bs, 1), float(self.cond_drop_prob), device=cond.device))
    else:
        m = torch.zeros(bs, 1, device=cond.device)
    self._keep = 1.0 - m
    return cond * self._keep


def _m_trans_forward(self, motion_ids, cond, padding_mask, force_mask=False, memory=None):
    cond = self.global_local_crossatt_block(cond.unsqueeze(1), memory).squeeze(1)
    cond = self.mask_cond(cond, force_mask=force_mask)
    memory = memory * self._keep[:, :, None]
    x = self.input_process(self.token_emb(motion_ids))
    cond = self.cond_emb(cond).unsqueeze(0); memory = self.cond_emb(memory).permute((1, 0, 2))
    x = self.position_enc(x); xseq = torch.cat([cond, x], dim=0)
    padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1)
    output = self.seqTransDecoder(tgt=xseq, memory=memory, tgt_key_padding_mask=padding_mask)[1:]
    return self.output_process(output)


def _r_trans_forward(self, motion_codes, qids, cond, padding_mask, force_mask=False, memory=None):
    cond = self.global_local_crossatt_block(cond.unsqueeze(1), memory).squeeze(1)
    cond = self.mask_cond(cond, force_mask=force_mask)
    memory = memory * self._keep[:, :, None]
    x = self.input_process(motion_codes)
    q_emb = self.quant_emb(self.encode_quant(qids).float().to(x.device)).unsqueeze(0)
    cond = self.cond_emb(cond).unsqueeze(0); memory = self.cond_emb(memory).permute((1, 0, 2))
    x = self.position_enc(x); xseq = torch.cat([cond, q_emb, x], dim=0)
    padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:2]), padding_mask], dim=1)
    output = self.seqTransDecoder(tgt=xseq, memory=memory, tgt_key_padding_mask=padding_mask)[2:]
    return self.output_process(output)


TMC.MaskTransformer.mask_cond = _mask_cond; TMC.MaskTransformer.trans_forward = _m_trans_forward
TMC.ResidualTransformer.mask_cond = _mask_cond; TMC.ResidualTransformer.trans_forward = _r_trans_forward


class QwenAdapter(nn.Module):
    def __init__(self, use_layers=LAYERS, d_in=2560, d=512, n_slots=16):
        super().__init__()
        self.sel = [LAYERS.index(l) for l in use_layers]
        self.mix = nn.Parameter(torch.zeros(len(self.sel)))
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        self.q = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 8, batch_first=True)
        self.ln_mem = nn.LayerNorm(d); self.glob = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def forward(self, H, mask):
        """H (b, L, nL_cached, d_in) float, mask (b, L) bool -> global (b, d), memory (b, n_slots, d)"""
        H = H[:, :, self.sel].float(); H = F.layer_norm(H, H.shape[-1:])
        x = (H * torch.softmax(self.mix, 0)[None, None, :, None]).sum(2)
        h = self.proj(x); b = h.shape[0]
        q = self.q[None].expand(b, -1, -1)
        mem, _ = self.attn(q, h, h, key_padding_mask=~mask)
        g = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.glob(g), self.ln_mem(mem + q)


class VisAdapter(QwenAdapter):
    """Visual-semantic branch B: hidden states of the frozen SFT student (video + prompt + answer, teacher-forced):
    frame-pooled video tokens (type 0) + <exec> line tokens (type 1)."""
    def __init__(self, use_layers=LAYERS, d=512):
        super().__init__(use_layers, d=d); self.type_emb = nn.Embedding(2, d)

    def forward(self, H, mask, typ):
        H = H[:, :, self.sel].float(); H = F.layer_norm(H, H.shape[-1:])
        x = (H * torch.softmax(self.mix, 0)[None, None, :, None]).sum(2)
        h = self.proj(x) + self.type_emb(typ); b = h.shape[0]
        q = self.q[None].expand(b, -1, -1)
        mem, _ = self.attn(q, h, h, key_padding_mask=~mask)
        g = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.glob(g), self.ln_mem(mem + q)


class Cond(nn.Module):
    """A (text re-encoding by the frozen base; HumanML3D-pretrained) + optional B (visual semantics) through per-channel
    ZERO-initialised gates: at init the conditioning is exactly A's."""
    def __init__(self, use_layers=LAYERS, vis=False, d=512):
        super().__init__(); self.a = QwenAdapter(use_layers); self.vis = vis
        if vis:
            self.b = VisAdapter(use_layers); self.gate_g = nn.Parameter(torch.zeros(d)); self.gate_m = nn.Parameter(torch.zeros(d))

    def forward(self, H, mk, V=None, vm=None, vt=None):
        g, m = self.a(H, mk)
        if self.vis:
            gb, mb = self.b(V, vm, vt); g = g + self.gate_g * gb; m = m + self.gate_m * mb
        return g, m

    def vis_state(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("a.")} if self.vis else None


class VisFeats:
    """CACHE/visfeat/<name>.{feats.npy [Ntok, nL, 2560] fp16, meta.json {ids, offsets, lengths, n_video}}"""
    def __init__(self, name):
        d = CACHE / "visfeat"; self.meta = rjson(d / f"{name}.meta.json"); self.f = np.load(d / f"{name}.feats.npy", mmap_mode="r")
        assert tuple(self.meta["layers"]) == LAYERS
        self.idx = {k: i for i, k in enumerate(self.meta["ids"])}

    def batch(self, keys, device):
        ii = [self.idx[k] for k in keys]; Ls = [self.meta["lengths"][i] for i in ii]; L = max(Ls)
        H = np.zeros((len(keys), L) + self.f.shape[1:], np.float16); m = np.zeros((len(keys), L), bool); t = np.zeros((len(keys), L), np.int64)
        for j, i in enumerate(ii):
            o, n, nv = self.meta["offsets"][i], self.meta["lengths"][i], self.meta["n_video"][i]
            H[j, :n] = self.f[o:o + n]; m[j, :n] = True; t[j, nv:n] = 1
        return torch.from_numpy(H).to(device), torch.from_numpy(m).to(device), torch.from_numpy(t).to(device)


class TextFeats:
    def __init__(self, name):
        d = CACHE / "textfeat"; self.meta = rjson(d / f"{name}.meta.json"); self.f = np.load(d / f"{name}.feats.npy", mmap_mode="r")
        assert tuple(self.meta["layers"]) == LAYERS
        self.off, self.len = self.meta["offsets"], self.meta["lengths"]

    def batch(self, tids, device):
        L = max(self.len[t] for t in tids); H = np.zeros((len(tids), L) + self.f.shape[1:], np.float16); m = np.zeros((len(tids), L), bool)
        for i, t in enumerate(tids):
            H[i, :self.len[t]] = self.f[self.off[t]:self.off[t] + self.len[t]]; m[i, :self.len[t]] = True
        return torch.from_numpy(H).to(device), torch.from_numpy(m).to(device)


def build(device, vq_name, init=None, use_layers=LAYERS, vis=False):
    vq_ckpt = VQ_CANDIDATES[vq_name]
    if vq_ckpt is None:  # original ViMo VQ state
        vo, _, _ = FH.vq_opts(device); FH.gen.opt = __import__("types").SimpleNamespace(device=device, name=FH.M_NAME)
        vq0, _ = FH.gen.load_vq_model(vo); vq_ckpt = CACHE / "vq_vimo_state.pt"
        if not vq_ckpt.exists():
            torch.save({"vq_model": vq0.state_dict()}, vq_ckpt)
    vq, trans, res, mean, std = FH.load_models(device, vq_ckpt)
    ad = Cond(use_layers, vis=vis).to(device)
    if init:
        ck = torch.load(init, map_location="cpu")
        trans.load_state_dict(ck["trans"], strict=False); res.load_state_dict(ck["res"], strict=False); ad.a.load_state_dict(ck["adapter"])
        if vis and ck.get("adapter_vis"):
            miss = ad.load_state_dict({**{"a." + k: v for k, v in ck["adapter"].items()}, **ck["adapter_vis"]}, strict=True)
    return vq, trans, res, ad, mean, std


def crop_len(L, rng):
    coin = rng.choice(["single", "single", "double"])
    return max((min(L, 196) // 4 - (1 if coin == "double" else 0)) * 4, 4)


class HML3D:
    def __init__(self, split, mean, std):
        z = np.load(CACHE / "hml3d_motion.npz"); self.pos = {n: i for i, n in enumerate(z["names"])}
        self.off, self.lens, self.feats = z["offsets"], z["lengths"], ((z["feats"] - mean) / std).astype(np.float32)
        self.tf = TextFeats(f"hml3d_{split}")
        self.items = [x for x in self.tf.meta["items"] if x["motion"] in self.pos]
        by = {}
        for i, x in enumerate(self.items):
            by.setdefault(x["motion"], []).append(i)
        self.by = list(by.values())

    def sample(self, bs, rng, device, idx=None):
        if idx is None:
            idx = [self.by[j][rng.integers(len(self.by[j]))] for j in rng.integers(0, len(self.by), bs)]
        its = [self.items[i] for i in idx]; ml = [crop_len(x["end"] - x["start"], rng) for x in its]
        X = np.zeros((len(its), max(ml), 263), np.float32)
        for b, (x, m) in enumerate(zip(its, ml)):
            p = self.pos[x["motion"]]; seg = self.feats[self.off[p] + x["start"]:self.off[p] + x["end"]]
            s = rng.integers(0, len(seg) - m + 1); X[b, :m] = seg[s:s + m]
        H, mk = self.tf.batch([x["tid"] for x in its], device)
        return torch.from_numpy(X).to(device), torch.tensor(ml, device=device), H, mk, None, None, None


class V1197:
    def __init__(self, group, mean, std, full_event=False, vis=None):
        """v1197 TRAIN/VAL motions (runs/motion: capped 88-frame or full event <= 196) + gt3p text feats (+ vis feats)."""
        man = manifest()
        with np.load(RUNS / "motion" / ("humanml263_full.npz" if full_event else "humanml263_capped.npz")) as z:
            ids = [str(x) for x in z["ids"]]; feats, lens = z["features"], z["lengths"]
        pos = {k: i for i, k in enumerate(ids)}; self.keys = man[group]
        self.feats = ((feats[[pos[k] for k in self.keys]] - mean) / std).astype(np.float32); self.lens = lens[[pos[k] for k in self.keys]]
        self.tf = TextFeats("gt3p_trainval"); tid = {x["id"]: x["tid"] for x in self.tf.meta["items"]}; self.tids = [tid[k] for k in self.keys]
        self.vis = VisFeats(vis) if vis else None

    def sample(self, bs, rng, device, idx=None):
        idx = rng.choice(len(self.keys), bs, replace=False) if idx is None else idx
        ml = [crop_len(int(self.lens[i]), rng) for i in idx]; X = np.zeros((len(idx), max(ml), 263), np.float32)
        for b, (i, m) in enumerate(zip(idx, ml)):
            s = rng.integers(0, int(self.lens[i]) - m + 1); X[b, :m] = self.feats[i, s:s + m]
        H, mk = self.tf.batch([self.tids[i] for i in idx], device)
        V = self.vis.batch([self.keys[i] for i in idx], device) if self.vis else (None, None, None)
        return (torch.from_numpy(X).to(device), torch.tensor(ml, device=device), H, mk) + tuple(V)


def losses(vq, trans, res, ad, X, ml, H, mk, V=None, vm=None, vt=None):
    with torch.no_grad():
        codes, _ = vq.encode(X)
    g, m = ad(H, mk, V, vm, vt); tl = ml // 4
    l1, _, acc = trans(codes[..., 0], g, tl, m); l2, _, _ = res(codes, g, tl, m)
    return l1, l2, acc


@torch.no_grad()
def val_ce(vq, trans, res, ad, data, n, device):
    trans.eval(); res.eval(); ad.eval(); rng = np.random.default_rng(0); torch.manual_seed(0)
    N = len(data.items) if hasattr(data, "items") else len(data.keys); idx = np.arange(N)[:n] if N <= n else np.random.default_rng(1).choice(N, n, replace=False)
    s = np.zeros(3); c = 0
    for j in range(0, len(idx), 64):
        l1, l2, acc = losses(vq, trans, res, ad, *data.sample(None, rng, device, idx=list(idx[j:j + 64])))
        s += np.array([float(l1), float(l2), float(acc)]) * len(idx[j:j + 64]); c += len(idx[j:j + 64])
    trans.train(); res.train(); ad.train()
    return {"val_mask_ce": s[0] / c, "val_res_ce": s[1] / c, "val_mask_acc": s[2] / c, "val_sum_ce": (s[0] + s[1]) / c}


def train(a, data_tr, data_va, n_val):
    device = torch.device("cuda:0"); out = RUNS / a.out
    if (out / "train.jsonl").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True); wjson(out / "args.json", {k: str(v) for k, v in vars(a).items()})
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    vq, trans, res, ad, mean, std = build(device, a.vq, a.init, tuple(int(x) for x in a.layers.split(",")), vis=bool(a.vis))
    dtr, dva = data_tr(mean, std), data_va(mean, std)
    params = list(ad.parameters()) + trans.parameters_wo_clip() + res.parameters_wo_clip()
    if getattr(a, "freeze_base", False):  # EXPLORATORY (ControlNet-style): trunk + text adapter A frozen, train only branch B + gates
        assert a.vis
        for p_ in params:
            p_.requires_grad_(False)
        params = list(ad.b.parameters()) + [ad.gate_g, ad.gate_m]
        for p_ in params:
            p_.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in params if p.requires_grad], lr=a.lr, betas=(0.9, 0.99), weight_decay=1e-5)
    warm = a.warmup; sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 if s >= 0.7 * a.steps else 1.0))
    trans.train(); res.train(); ad.train(); log = open(out / "train.jsonl", "w"); best = float("inf"); t0 = time.time(); run = []
    for step in range(a.steps + 1):
        if step > 0:
            l1, l2, acc = losses(vq, trans, res, ad, *dtr.sample(a.batch_size, rng, device))
            opt.zero_grad(); (l1 + l2).backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            run.append([float(l1), float(l2), float(acc)])
        if step % a.eval_every == 0:
            r = {"step": step, **val_ce(vq, trans, res, ad, dva, n_val, device), "lr": sched.get_last_lr()[0], "s": round(time.time() - t0)}
            if run:
                r.update(dict(zip(["tr_mask_ce", "tr_res_ce", "tr_mask_acc"], np.mean(run, 0).tolist()))); run = []
            if a.vis:
                r["gate_g_absmean"] = float(ad.gate_g.abs().mean()); r["gate_m_absmean"] = float(ad.gate_m.abs().mean())
            log.write(json.dumps(r) + "\n"); log.flush(); print(json.dumps(r), flush=True)
            st = {"trans": FH.no_clip(trans.state_dict()), "res": FH.no_clip(res.state_dict()), "adapter": ad.a.state_dict(),
                  "adapter_vis": ad.vis_state(), "vis": bool(a.vis), "step": step, "vq": a.vq, "layers": a.layers, **r}
            if r["val_sum_ce"] < best and step > 0:
                best = r["val_sum_ce"]; torch.save(st, out / "best.pt")
    torch.save(st, out / "last.pt")
    wjson(out / "result.json", {"selection": "lowest val (mask CE + res CE), teacher-forced, fixed masks", "best_val_sum_ce": best,
                                "best_step": torch.load(out / "best.pt", map_location="cpu")["step"], "steps": a.steps})


@torch.no_grad()
def generate(vq, trans, res, ad, mean, std, tf, tids, nframes, device, seed, bs=48, cond_scale=4, res_scale=5, timesteps=18, vf=None, vkeys=None):
    """vf/vkeys: VisFeats + the item ids whose visual features condition each sample (shuffle control = permuted ids)."""
    trans.eval(); res.eval(); ad.eval(); out = []
    for s in range(0, len(tids), bs):
        torch.manual_seed(seed * 100003 + s)
        H, mk = tf.batch(tids[s:s + bs], device); V = vf.batch(vkeys[s:s + bs], device) if vf is not None else (None, None, None)
        g, m = ad(H, mk, *V)
        nf = [int(n) // 4 * 4 for n in nframes[s:s + bs]]; lens = torch.tensor([n // 4 for n in nf], device=device)
        tok = trans.generate(g, lens, timesteps=timesteps, cond_scale=cond_scale, temperature=1, topk_filter_thres=.9, gsample=False, memory=m)
        tok = res.generate(tok, g, lens, temperature=1, cond_scale=res_scale, memory=m)
        mot = vq.forward_decoder(tok).cpu().numpy()
        out += [FH.to_canonical_joints(mot[i, :nf[i]], mean, std) for i in range(len(nf))]
    return out


def stage_gen(a):
    """--corpus: textfeat corpus name with items {'id', 'tid', 'nframes'?}; lengths: item 'nframes' or --nframes fixed."""
    device = torch.device("cuda:0"); outp = RUNS / a.out
    if outp.exists():
        raise FileExistsError(outp)
    ck = torch.load(a.ckpt, map_location="cpu")
    vq, trans, res, ad, mean, std = build(device, ck["vq"], a.ckpt, tuple(int(x) for x in ck["layers"].split(",")), vis=ck.get("vis", False))
    tf = TextFeats(a.corpus); man = manifest(); keys = man["val"]
    vf = VisFeats(a.vis_corpus) if ck.get("vis") else None
    assert (vf is None) == (not a.vis_corpus), "vis checkpoint needs --vis-corpus (and only then)"
    vkeys = list(np.roll(np.array(keys), a.vis_shift * max(1, len(keys) // 6))) if a.vis_shift else list(keys)
    item = {x["id"]: x for x in tf.meta["items"]}; assert set(keys) <= set(item), "corpus must cover VAL"
    nfr = [int(item[k].get("nframes", a.nframes)) for k in keys]
    J = np.zeros((a.samples, len(keys), 88, 22, 3), np.float32); L = np.array([min(88, n // 4 * 4) for n in nfr])
    for s in range(a.samples):
        G = generate(vq, trans, res, ad, mean, std, tf, [item[k]["tid"] for k in keys], [min(88, n) for n in nfr], device, seed=1000 + s,
                     cond_scale=a.cond_scale, vf=vf, vkeys=vkeys)
        for j, x in enumerate(G):
            J[s, j, :len(x)] = x; J[s, j, len(x):] = x[-1]  # hold last pose (only for fixed-shape storage; eval uses lengths)
    outp.parent.mkdir(parents=True, exist_ok=True); np.savez(outp, val_ids=np.array(keys), val_joints=J, lengths=L)
    wjson(str(outp) + ".json", {"ckpt": str(a.ckpt), "step": ck["step"], "corpus": a.corpus, "vis_corpus": a.vis_corpus, "vis_shift": a.vis_shift,
                                "samples": a.samples, "cond_scale": a.cond_scale,
                                "lengths_hist": {int(k): int(v) for k, v in zip(*np.unique(L, return_counts=True))}})
    print("saved", outp)


def stage_hml3d_gen(a):
    device = torch.device("cuda:0"); outp = RUNS / a.out
    if outp.exists():
        raise FileExistsError(outp)
    ck = torch.load(a.ckpt, map_location="cpu"); vq, trans, res, ad, mean, std = build(device, ck["vq"], a.ckpt, tuple(int(x) for x in ck["layers"].split(",")))
    d = HML3D("test", mean, std); rng = np.random.default_rng(7)
    first = [b[0] for b in d.by]; idx = sorted(rng.choice(len(first), min(a.n, len(first)), replace=False)); idx = [first[i] for i in idx]
    its = [d.items[i] for i in idx]; nf = [min(196, x["end"] - x["start"]) // 4 * 4 for x in its]
    G = generate(vq, trans, res, ad, mean, std, d.tf, [x["tid"] for x in its], nf, device, seed=7, cond_scale=a.cond_scale)
    GT = []
    for x, n in zip(its, nf):
        p = d.pos[x["motion"]]; seg = d.feats[d.off[p] + x["start"]:d.off[p] + x["start"] + n]; GT.append(FH.to_canonical_joints(seg, mean, std))
    pad = lambda A: np.stack([np.concatenate([q, np.repeat(q[-1:], 196 - len(q), 0)]) for q in A]).astype(np.float32)
    np.savez(outp, texts=np.array([d.tf.meta["texts"][x["tid"]] for x in its]), lengths=np.array(nf), gen=pad(G), gt=pad(GT))
    print("saved", outp, len(its))


@torch.no_grad()
def stage_vqcheck(a):
    device = torch.device("cuda:0"); res_ = {}
    for name in VQ_CANDIDATES:
        vq, _, _, _, mean, std = build(device, name)
        d = HML3D("val", mean, std); rng = np.random.default_rng(0); e = []
        for j in range(0, 512, 64):
            X, ml = d.sample(None, rng, device, idx=list(range(j, j + 64)))[:2]
            codes, _ = vq.encode(X); rec = vq.forward_decoder(codes).cpu().numpy(); Xn = X.cpu().numpy()
            for i in range(len(ml)):
                n = int(ml[i]); e.append(np.linalg.norm(FH.to_canonical_joints(rec[i, :n], mean, std) - FH.to_canonical_joints(Xn[i, :n], mean, std), axis=-1).mean())
        v = V1197("val", mean, std); e2 = []
        for i in range(len(v.keys)):
            n = int(v.lens[i]) // 4 * 4; X = torch.from_numpy(v.feats[i:i + 1, :n]).to(device)
            codes, _ = vq.encode(X); rec = vq.forward_decoder(codes).cpu().numpy()[0]
            e2.append(np.linalg.norm(FH.to_canonical_joints(rec, mean, std) - FH.to_canonical_joints(v.feats[i, :n], mean, std), axis=-1).mean())
        res_[name] = {"hml3d_val512_recon_mpjpe_m": float(np.mean(e)), "v860_val_recon_mpjpe_m": float(np.mean(e2))}
        print(name, res_[name], flush=True)
    wjson(RUNS / "vqcheck.json", res_)


def main():
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=["vqcheck", "pretrain", "finetune", "gen", "hml3d_gen"])
    p.add_argument("--out"); p.add_argument("--init"); p.add_argument("--ckpt"); p.add_argument("--vq", default="v860")
    p.add_argument("--layers", default="9,18,27,36"); p.add_argument("--steps", type=int, default=60000); p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--warmup", type=int, default=1000); p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=3407); p.add_argument("--corpus"); p.add_argument("--nframes", type=int, default=88)
    p.add_argument("--samples", type=int, default=3); p.add_argument("--cond-scale", type=float, default=4); p.add_argument("--n", type=int, default=1024)
    p.add_argument("--full-event", action="store_true", help="finetune on un-truncated (complete event) targets")
    p.add_argument("--vis", default=None, help="vis-feat corpus of TRAIN+VAL (GT answer) -> adds the zero-gated visual branch B")
    p.add_argument("--vis-corpus", default=None, help="gen: vis-feat corpus of the evaluated predictions")
    p.add_argument("--vis-shift", type=int, default=0, help="gen control: condition B on other VAL items' visual features")
    p.add_argument("--freeze-base", action="store_true", help="EXPLORATORY: freeze trunk + adapter A (init from an A-only head), train only B + gates")
    a = p.parse_args(); torch.set_num_threads(4)
    for n in ("init", "ckpt"):  # FH chdirs into HERO on import: resolve relative paths against the launch directory
        v = getattr(a, n)
        if v and not Path(v).is_absolute():
            setattr(a, n, str((FH.ORIG_CWD / v).resolve()))
    if a.stage == "vqcheck":
        stage_vqcheck(a)
    elif a.stage == "pretrain":
        train(a, lambda m, s: HML3D("train", m, s), lambda m, s: HML3D("val", m, s), 1024)
    elif a.stage == "finetune":
        train(a, lambda m, s: V1197("train", m, s, a.full_event, a.vis), lambda m, s: V1197("val", m, s, a.full_event, a.vis), 125)
    elif a.stage == "gen":
        stage_gen(a)
    else:
        stage_hml3d_gen(a)


if __name__ == "__main__":
    main()
