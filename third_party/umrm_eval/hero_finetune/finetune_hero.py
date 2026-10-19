"""Fair HERO baseline: fine-tune HERO (ViMo checkpoints) on the same 384 PICO train items.

Stages (each writes into its own new output directory):
  vq    : fine-tune rvq_bs256_finetune_ep10 on PICO HumanML263 features (HERO trainer loss,
          20-frame windows, ViMo mean/std kept), select by val reconstruction MPJPE.
  trans : with the fine-tuned VQ frozen, fine-tune the memo-cross mask transformer
          (from net_best_acc) and residual transformer (from net_best_fid) using HERO's own
          forward losses, with cached frozen TC-CLIP (global -> cond, memory -> at_features).
          Every --eval-every steps, generate the 56 val items with the candidate0 settings
          (18 steps, CFG 4, temp 1, top-k .9, residual CFG 5, 88 frames, per-id seed rule)
          and keep the pair with the lowest val global MPJPE (same criterion as our model).
  gen   : generate val+test (K seeds) with the selected pair; save canonical joints.
The HERO repo is imported read-only; nothing under HERO/ is written.
Run with: /usr/bin/python run_hero_env.py hero_finetune/finetune_hero.py STAGE ...
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERO = Path(os.environ.get("VHO_HERO", str(ROOT / "HERO")))
RUN = ROOT / "umrm_model/outputs/cliponly_v1_20260918"
TCCLIP = ROOT / "umrm_model/outputs/tcclip_cache_636_20260926"
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("cliponly_data", Path(__file__).resolve().parents[1] / "cliponly_v1/data.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
canonical = _mod.canonical

ORIG_CWD = Path.cwd()
os.chdir(HERO)
sys.path.insert(0, str(HERO))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import gen  # noqa: E402
from utils.get_opt import get_opt  # noqa: E402
from utils.motion_process import recover_from_ric  # noqa: E402
from utils.fixseed import fixseed  # noqa: E402

VQ_NAME = "rvq_bs256_finetune_ep10"
M_NAME = "mtrans_memo_cross_l6_bs64_ep200"
R_NAME = "rtrans_memo_cross_l6_bs64_ep200"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_for(ident, k):
    return int(hashlib.sha256(f"hero636-v1:{ident}:{k}".encode()).hexdigest()[:8], 16) % 2147483647


def score(motion, reference, valid):
    err = np.linalg.norm(motion[valid] - reference[valid], axis=-1).mean()
    local = np.linalg.norm((motion - motion[:, :1])[valid] - (reference - reference[:, :1])[valid], axis=-1).mean()
    root = np.linalg.norm(motion[valid, 0] - reference[valid, 0], axis=-1).mean()
    return {"mpjpe_m": float(err), "root_relative_mpjpe_m": float(local), "root_error_m": float(root)}


def load_all(data_dir):
    split = json.loads((RUN / "split.json").read_text())
    with np.load(RUN / "data.npz", allow_pickle=False) as z:
        ids = z["ids"].tolist()
        target = z["targets"][:, :, :22].astype(np.float32)
        mask = z["masks"].astype(bool)
    with np.load(data_dir / "humanml263.npz", allow_pickle=False) as z:
        assert z["ids"].tolist() == ids
        feats = z["features"]
        lengths = z["lengths"]
    idx = {g: np.array([i for i, k in enumerate(ids) if split[k] == g]) for g in ("train", "val", "test")}
    assert [len(idx[g]) for g in ("train", "val", "test")] == [384, 56, 84]
    return SimpleNamespace(ids=ids, target=target, mask=mask, feats=feats, lengths=lengths, idx=idx)


def vq_opts(device):
    vo = get_opt(f"checkpoints/vimo/{VQ_NAME}/opt.txt", device)
    vo.dim_pose = 263
    mean = np.load(f"checkpoints/vimo/{VQ_NAME}/meta/mean.npy").astype(np.float32)
    std = np.load(f"checkpoints/vimo/{VQ_NAME}/meta/std.npy").astype(np.float32)
    return vo, mean, std


def to_canonical_joints(motion_norm, mean, std):
    motion = motion_norm * std + mean
    return canonical(recover_from_ric(torch.from_numpy(motion).float(), 22).numpy())


@torch.no_grad()
def vq_val_mpjpe(vq, D, mean, std, device, group="val"):
    vq.eval()
    errs = []
    for i in D.idx[group]:
        L = int(D.lengths[i]) // 4 * 4
        x = torch.from_numpy((D.feats[i, :L] - mean) / std)[None].to(device)
        codes, _ = vq.encode(x)
        rec = vq.forward_decoder(codes).cpu().numpy()[0]
        joints = to_canonical_joints(rec, mean, std)
        errs.append(score(joints, D.target[i, :L], np.ones(L, bool))["mpjpe_m"])
    vq.train()
    return float(np.mean(errs))


def stage_vq(a):
    device = torch.device("cuda:0")
    out = a.output.resolve()
    out.mkdir(parents=True)
    D = load_all(a.data)
    vo, mean, std = vq_opts(device)
    vq, _ = gen.load_vq_model(vo)
    vq.to(device).train()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    windows = [(i, s) for i in D.idx["train"] for s in range(int(D.lengths[i]) - vo.window_size + 1)]
    normed = (D.feats - mean) / std
    crit = torch.nn.SmoothL1Loss()
    opt = torch.optim.AdamW(vq.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(a.steps * 0.7)], gamma=0.05)
    log = open(out / "train.jsonl", "w")
    best = vq_val_mpjpe(vq, D, mean, std, device)
    zero_shot = best
    torch.save({"vq_model": vq.state_dict(), "step": 0, "val_mpjpe_m": best}, out / "best.pt")
    log.write(json.dumps({"step": 0, "val_recon_mpjpe_m": best}) + "\n")
    best_step = 0
    for step in range(1, a.steps + 1):
        if step < vo.warm_up_iter:
            for g in opt.param_groups:
                g["lr"] = a.lr * (step + 1) / (vo.warm_up_iter + 1)
        pick = rng.integers(0, len(windows), a.batch_size)
        batch = np.stack([normed[windows[p][0], windows[p][1]:windows[p][1] + vo.window_size] for p in pick])
        x = torch.from_numpy(batch).to(device)
        pred, commit, perplexity = vq(x)
        rec = crit(pred, x)
        explicit = crit(pred[..., 4:67], x[..., 4:67])
        loss = rec + vo.loss_vel * explicit + vo.commit * commit
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            val = vq_val_mpjpe(vq, D, mean, std, device)
            row = {"step": step, "loss": float(loss), "rec": float(rec), "commit": float(commit),
                   "perplexity": float(perplexity), "val_recon_mpjpe_m": val}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
            if val < best:
                best, best_step = val, step
                torch.save({"vq_model": vq.state_dict(), "step": step, "val_mpjpe_m": val}, out / "best.pt")
    test = None
    ck = torch.load(out / "best.pt", map_location=device)
    vq.load_state_dict(ck["vq_model"])
    test = vq_val_mpjpe(vq, D, mean, std, device, "test")
    result = {"zero_shot_val_recon_mpjpe_m": zero_shot, "best_step": best_step, "best_val_recon_mpjpe_m": best,
              "best_test_recon_mpjpe_m": test, "init_checkpoint_sha256": sha(f"checkpoints/vimo/{VQ_NAME}/model/net_best_fid.tar"),
              "data_sha256": sha(a.data / "humanml263.npz"), "args": {k: str(v) for k, v in vars(a).items()},
              "normalization": "ViMo mean/std from rvq_bs256_finetune_ep10/meta"}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)


def load_tc(ids):
    G, M = [], []
    for k in ids:
        with np.load(TCCLIP / "features" / f"{k}.npz") as z:
            G.append(z["global"].astype(np.float32))
            M.append(z["memory"].astype(np.float32))
    return np.stack(G), np.stack(M)


def load_models(device, vq_ckpt, trans_state=None, res_state=None):
    opt = SimpleNamespace(device=device, name=M_NAME)
    gen.opt = opt
    vo, mean, std = vq_opts(device)
    vq, _ = gen.load_vq_model(vo)
    vq.load_state_dict(torch.load(vq_ckpt, map_location="cpu")["vq_model"])
    mo = get_opt(f"checkpoints/vimo/{M_NAME}/opt.txt", device)
    mo.num_tokens, mo.num_quantizers, mo.code_dim = vo.nb_code, vo.num_quantizers, vo.code_dim
    ro = get_opt(f"checkpoints/vimo/{R_NAME}/opt.txt", device)
    res = gen.load_res_model(ro, vo, opt)
    trans = gen.load_trans_model(mo, opt, "net_best_acc.tar")
    if trans_state is not None:
        trans.load_state_dict(trans_state, strict=False)
    if res_state is not None:
        res.load_state_dict(res_state, strict=False)
    for m in (vq, trans, res):
        m.to(device)
    vq.eval()
    for p in vq.parameters():
        p.requires_grad_(False)
    return vq, trans, res, mean, std


def no_clip(state):
    return {k: v for k, v in state.items() if not k.startswith("clip_model.")}


@torch.no_grad()
def generate(vq, trans, res, mean, std, G, M, ids, device, k=0):
    trans.eval()
    res.eval()
    out = []
    lens = torch.tensor([22], device=device)
    for j, key in enumerate(ids):
        vg = torch.from_numpy(G[j:j + 1]).to(device)
        vl = torch.from_numpy(M[j:j + 1]).to(device)
        fixseed(seed_for(key, k))
        tokens = trans.generate(vg, lens, timesteps=18, cond_scale=4, temperature=1, topk_filter_thres=.9,
                                gsample=False, memory=vl)
        tokens = res.generate(tokens, vg, lens, temperature=1, cond_scale=5, memory=vl)
        motion = vq.forward_decoder(tokens).cpu().numpy()[0]
        out.append(to_canonical_joints(motion, mean, std))
    trans.train()
    res.train()
    return np.stack(out)


def stage_trans(a):
    device = torch.device("cuda:0")
    out = a.output.resolve()
    out.mkdir(parents=True)
    D = load_all(a.data)
    vq, trans, res, mean, std = load_models(device, a.vq_checkpoint)
    G, M = load_tc(D.ids)
    tr, va = D.idx["train"], D.idx["val"]
    normed = ((D.feats - mean) / std).astype(np.float32)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    opts = [torch.optim.AdamW(m.parameters_wo_clip(), lr=a.lr, betas=(0.9, 0.99), weight_decay=1e-5) for m in (trans, res)]
    scheds = [torch.optim.lr_scheduler.MultiStepLR(o, milestones=[int(a.steps * 0.7)], gamma=0.1) for o in opts]
    trans.train()
    res.train()

    def batch(indices, train=True):
        x = np.zeros((len(indices), 88, 263), np.float32)
        m_lens = []
        for b, i in enumerate(indices):
            L = int(D.lengths[i])
            coin = rng.choice(["single", "single", "double"]) if train else "single"
            m = (L // 4 - (1 if coin == "double" else 0)) * 4
            m = max(m, 4)
            s = rng.integers(0, L - m + 1) if train else 0
            x[b, :m] = normed[i, s:s + m]
            m_lens.append(m)
        x = torch.from_numpy(x).to(device)
        with torch.no_grad():
            codes, _ = vq.encode(x)
        return (codes, torch.tensor(m_lens, device=device) // 4,
                torch.from_numpy(G[indices]).to(device), torch.from_numpy(M[indices]).to(device))

    def val_ce():
        trans.eval(); res.eval()
        with torch.no_grad():
            torch.manual_seed(0)
            codes, ml, g, m = batch(va, train=False)
            c1, _, acc1 = trans(codes[..., 0], g, ml, m)
            c2, _, acc2 = res(codes, g, ml, m)
        trans.train(); res.train()
        return float(c1), float(acc1), float(c2), float(acc2)

    def val_mpjpe():
        pred = generate(vq, trans, res, mean, std, G[va], M[va], [D.ids[i] for i in va], device)
        return float(np.mean([score(pred[j], D.target[i], D.mask[i])["mpjpe_m"] for j, i in enumerate(va)])), \
            float(np.mean([score(pred[j], D.target[i], D.mask[i])["root_relative_mpjpe_m"] for j, i in enumerate(va)]))

    log = open(out / "train.jsonl", "w")
    best, best_step = float("inf"), None
    t0 = time.time()
    for step in range(0, a.steps + 1):
        if step > 0:
            if step < 250:
                for o in opts:
                    for g_ in o.param_groups:
                        g_["lr"] = a.lr * (step + 1) / 251
            codes, ml, g, m = batch(rng.choice(tr, a.batch_size, replace=False))
            l1, _, acc1 = trans(codes[..., 0], g, ml, m)
            l2, _, acc2 = res(codes, g, ml, m)
            for o, l in zip(opts, (l1, l2)):
                o.zero_grad()
                l.backward()
                o.step()
            for s_ in scheds:
                s_.step()
        if step % a.eval_every == 0:
            mp, rr = val_mpjpe()
            ce = val_ce()
            row = {"step": step, "val_mpjpe_m": mp, "val_root_rel_m": rr, "val_mask_ce": ce[0], "val_mask_acc": ce[1],
                   "val_res_ce": ce[2], "val_res_acc": ce[3], "elapsed_s": round(time.time() - t0, 1)}
            if step > 0:
                row.update(train_mask_ce=float(l1), train_res_ce=float(l2))
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
            if mp < best:
                best, best_step = mp, step
                torch.save({"trans": no_clip(trans.state_dict()), "res": no_clip(res.state_dict()), "step": step,
                            "val_mpjpe_m": mp}, out / "best.pt")
    result = {"best_step": best_step, "best_val_mpjpe_m": best, "vq_checkpoint": str(a.vq_checkpoint),
              "vq_checkpoint_sha256": sha(a.vq_checkpoint), "data_sha256": sha(a.data / "humanml263.npz"),
              "args": {k: str(v) for k, v in vars(a).items()},
              "selection": "lowest val(56) global MPJPE of candidate0-setting generation"}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)


def stage_gen(a):
    device = torch.device("cuda:0")
    out = a.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    D = load_all(a.data)
    ck = torch.load(a.trans_checkpoint, map_location="cpu")
    vq, trans, res, mean, std = load_models(device, a.vq_checkpoint, ck["trans"], ck["res"])
    G, M = load_tc(D.ids)
    result = {"trans_checkpoint_sha256": sha(a.trans_checkpoint), "vq_checkpoint_sha256": sha(a.vq_checkpoint),
              "step": ck["step"], "samples": a.samples, "permute_shift": a.permute_shift}
    arrays = {}
    for group in ("val", "test"):
        ix = D.idx[group]
        keys = [D.ids[i] for i in ix]
        src = np.roll(ix, a.permute_shift * max(1, len(ix) // 6)) if a.permute_shift else ix  # clip-permutation control
        preds = np.stack([generate(vq, trans, res, mean, std, G[src], M[src], keys, device, k) for k in range(a.samples)])
        arrays[f"{group}_ids"] = np.array(keys)
        arrays[f"{group}_joints"] = preds.astype(np.float32)
        s0 = [score(preds[0, j], D.target[i], D.mask[i]) for j, i in enumerate(ix)]
        result[group] = {k: float(np.mean([r[k] for r in s0])) for k in s0[0]}
        print(group, json.dumps(result[group]), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    Path(str(out) + ".json").write_text(json.dumps(result, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["vq", "trans", "gen"])
    p.add_argument("--data", type=Path, default=ROOT / "umrm_model/outputs/hero_ft_data_20260926")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--vq-checkpoint", type=Path)
    p.add_argument("--trans-checkpoint", type=Path)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--permute-shift", type=int, default=0, help="gen only: feed other clips' features (control)")
    a = p.parse_args()
    for name in ("data", "output", "vq_checkpoint", "trans_checkpoint"):
        value = getattr(a, name)
        if value is not None and not value.is_absolute():
            setattr(a, name, (ORIG_CWD / value).resolve())
    if a.stage != "gen" and a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(4)
    {"vq": stage_vq, "trans": stage_trans, "gen": stage_gen}[a.stage](a)


if __name__ == "__main__":
    main()
