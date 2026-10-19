"""README §9 stage A: egocentric causal pre-training (PT-C) of a NEW LoRA r16 on Qwen3-VL-4B (same modules as sft.py) on
runs/ego_manifest.json["train"] (video -> <reason>/<next>, or EgoRe why-question -> answer). Teacher-forced NLL on answer tokens
(same loss as sft.py), lr 1e-4 cosine, 3% warm-up, global update every --accum items, 1 epoch, final checkpoint.
Multi-GPU: WORLD_SIZE/RANK env (gloo, LoRA grads summed on CPU -> identical to single-GPU), see run_ego_pt.sh.
Resume: runs/<out>/resume.pt every --save-every updates. Items whose cache file is missing are skipped (logged).
Run with VHO_CACHE=/tmp/vho_ego_cache."""
import argparse, datetime, json, math, os, random, time
import torch
import torch.distributed as dist
from common import *
from vt_io import load_model, encode, seq_logp


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--accum", type=int, default=8); ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--resume", action="store_true"); a = ap.parse_args()
    ws, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0")); dev = "cuda:0"
    if ws > 1:
        dist.init_process_group("gloo", timeout=datetime.timedelta(minutes=90))
    barrier = (lambda: dist.barrier()) if ws > 1 else (lambda: None)
    out = RUNS / a.out
    if rank == 0:
        if (out / "log.jsonl").exists() and not a.resume:
            raise FileExistsError(out)
        out.mkdir(parents=True, exist_ok=True)
    barrier()
    m = rjson(RUNS / "ego_manifest.json"); its = m["train"]
    miss = [x["id"] for x in its if not (CACHE / "video" / f"{x['id']}.pt").exists()]
    its = [x for x in its if (CACHE / "video" / f"{x['id']}.pt").exists()]
    if a.limit:
        its = its[:a.limit]
    if rank == 0:
        wjson(out / "args.json", {**vars(a), "world_size": ws, "n_items": len(its), "n_missing_cache": len(miss), "missing_ids": miss[:200],
                                  "by_src": {s: sum(x["src"] == s for x in its) for s in ("goalstep", "egonormia", "egore_why")},
                                  "cache": str(CACHE), "qwen": QWEN, "nframes": NFRAMES})
    torch.manual_seed(a.seed); random.seed(a.seed)
    model = load_model(trainable=True, r=a.r, device=dev); model.train()
    params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = math.ceil(len(its) / a.accum) * a.epochs; warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    step, start = 0, (1, 0)
    if a.resume and (out / "resume.pt").exists():
        ck = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["lora"], strict=False); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, start = ck["step"], tuple(ck["next"]); random.setstate(ck["py_rng"])
        print(f"rank{rank} resumed at step {step}, next {start}", flush=True)
    log = open(out / ("log.jsonl" if rank == 0 else f"log_rank{rank}.jsonl"), "a"); t0 = time.time(); run = {}
    for ep in range(1, a.epochs + 1):
        rs = random.Random(a.seed + ep); order = its[:]; rs.shuffle(order)  # identical order on all ranks and on resume
        if ep < start[0]:
            continue
        for g0 in range(0, len(order), a.accum):
            if (ep, g0) < start:
                continue
            grp = order[g0:g0 + a.accum]; mine = grp[rank::ws]
            for x in mine:
                b = encode(x["id"], x["prompt"], x["target"]); lp, c = seq_logp(model, b, dev)
                loss = -lp / c; (loss / len(grp)).backward(); run.setdefault(x["src"], []).append(float(loss))
            if ws > 1:
                flat = torch.cat([(p_.grad if p_.grad is not None else torch.zeros_like(p_)).float().reshape(-1) for p_ in params]).cpu()
                dist.all_reduce(flat); o_ = 0
                for p_ in params:
                    n_ = p_.numel(); g_ = flat[o_:o_ + n_].view_as(p_).to(p_.device, p_.dtype)
                    if p_.grad is None:
                        p_.grad = g_
                    else:
                        p_.grad.copy_(g_)
                    o_ += n_
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 10 == 0:
                r = {"epoch": ep, "step": step, "of": total, **{f"loss_{k}": round(sum(v) / len(v), 4) for k, v in run.items()},
                     "lr": sched.get_last_lr()[0], "s": round(time.time() - t0), "mem_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
                run = {}; log.write(json.dumps(r) + "\n"); log.flush()
                if rank == 0:
                    print(r, flush=True)
            if rank == 0 and step % a.save_every == 0:
                lora = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
                nxt = (ep, g0 + a.accum) if g0 + a.accum < len(order) else (ep + 1, 0)
                torch.save({"lora": lora, "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "next": list(nxt), "py_rng": random.getstate()},
                           out / "resume.tmp"); os.replace(out / "resume.tmp", out / "resume.pt")
        if rank == 0:
            model.save_pretrained(out / f"epoch{ep}")
        barrier()
    if rank == 0:
        wjson(out / "selection.json", {"rule": "pre-registered: final epoch (README §9)", "selected": f"epoch{a.epochs}"})
    barrier()


if __name__ == "__main__":
    main()
