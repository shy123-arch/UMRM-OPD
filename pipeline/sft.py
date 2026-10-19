"""S1 SFT of Qwen3-VL-4B (LoRA r16 on all LLM linear layers; vision tower frozen). Recipe identical to route 6
(lr 1e-4, 3 epochs, accum 8, cosine, 3% warm-up; pre-registered selection = lowest VAL teacher-forced NLL per token).
--mode student : prompt = PROMPT (video only)                      -> the SFT baseline and the OPD student init
--mode teacher : prompt = PROMPT + privileged hint z (PICO motion)  -> the frozen OPD teacher (training-time only)
Usage: sft.py --mode student --out sft_student"""
import argparse, math, random, time
import torch
from torch.utils.data import Dataset, DataLoader
from common import *
from vt_io import load_model, encode, seq_logp


class DS(Dataset):
    """items = list of (key, task); task 'answer' with mode teacher uses the privileged teacher prompt (VHO_ZMODE)."""
    def __init__(self, items, rows, mode):
        self.items, self.rows, self.mode = items, rows, mode

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        k, task = self.items[i]; r = self.rows[k]
        if self.mode == "teacher":
            return encode(r["id"], teacher_prompt(r), r["target"])
        return encode(r["id"], *task_io(r, task))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["student", "teacher"], required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--r", type=int, default=16); ap.add_argument("--base", default=None, help="base model path or '8b' (default Qwen3-VL-4B)"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--task", choices=["answer", "cot", "obs", "mt"], default="answer", help="README §8: reasoning as output (cot/obs) or multi-task (mt)")
    ap.add_argument("--init-adapter", default=None, help="README §9: continue training this LoRA (e.g. pt_c/epoch1) instead of a new one")
    ap.add_argument("--stop-after", type=int, default=0, help="stop after this epoch (LR schedule still over --epochs); selection among completed epochs")
    ap.add_argument("--checkpoint-steps", type=int, nargs="*", default=[], help="save and evaluate diagnostic optimizer-step checkpoints; never used for epoch selection")
    a = ap.parse_args(); out = RUNS / a.out
    if (out / "log.jsonl").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    man = manifest(); rows = man["rows"]; tr, ev = man["train"], man["val"]
    if a.limit:
        tr, ev = tr[:a.limit], ev[:max(2, a.limit // 4)]
    tr_items = ([(k, "answer") for k in tr] + [(k, "reason") for k in tr]) if a.task == "mt" else [(k, a.task) for k in tr]
    ev_items = [(k, "answer" if a.task == "mt" else a.task) for k in ev]  # mt: selection on the answer task only
    wjson(out / "args.json", {**vars(a), "n_train_items": len(tr_items), "zmode": ZMODE, "qwen": QWEN, "nframes": NFRAMES, "cache": str(CACHE), "split_sha256": man["split_sha256"], "n_train": len(tr), "n_val": len(ev)})
    torch.manual_seed(a.seed); random.seed(a.seed)
    model = load_model(adapter=resolve(a.init_adapter) if a.init_adapter else None, trainable=True, r=a.r, base=a.base)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = math.ceil(len(tr_items) / a.accum) * a.epochs; warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    g = torch.Generator(); g.manual_seed(a.seed)
    dl_tr = lambda: DataLoader(DS(tr_items, rows, a.mode), batch_size=None, shuffle=True, num_workers=0, generator=g)
    dl_ev = DataLoader(DS(ev_items, rows, a.mode), batch_size=None, shuffle=False, num_workers=0)
    log = open(out / "log.jsonl", "a"); step = 0; t0 = time.time()

    def evaluate(ep, diagnostic_step=None):
        model.eval(); s = n = 0.0
        with torch.no_grad():
            for b in dl_ev:
                lp, c = seq_logp(model, b); s += float(lp); n += float(c)
        model.train(); r = {"epoch": ep, "val_nll_per_tok": -s / n, "s": round(time.time() - t0)}
        if diagnostic_step is not None:
            r.update(diagnostic_only=True, step=diagnostic_step, checkpoint=f"step{diagnostic_step}")
        log.write(json.dumps(r) + "\n"); log.flush(); print(r, flush=True); return r
    evaluate(0); model.train()
    for ep in range(1, a.epochs + 1):
        run = []
        for i, b in enumerate(dl_tr()):
            lp, c = seq_logp(model, b); (-lp / c / a.accum).backward(); run.append(float(-lp / c))
            if (i + 1) % a.accum == 0 or i + 1 == len(tr_items):
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step in a.checkpoint_steps:
                    # The validation DataLoader can consume RNG state. Preserve it so diagnostics do not
                    # change the subsequent shuffle/dropout trajectory relative to an uninstrumented run.
                    cpu_rng, cuda_rng, loader_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all(), g.get_state()
                    model.save_pretrained(out / f"step{step}")
                    evaluate(ep - 1 + (i + 1) / len(tr_items), diagnostic_step=step)
                    torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng); g.set_state(loader_rng)
                if step % 5 == 0:
                    r = {"epoch": ep, "step": step, "loss": sum(run) / len(run), "lr": sched.get_last_lr()[0], "s": round(time.time() - t0),
                         "mem_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
                    run = []; log.write(json.dumps(r) + "\n"); log.flush(); print(r, flush=True)
        model.save_pretrained(out / f"epoch{ep}"); evaluate(ep)
        if a.stop_after and ep >= a.stop_after:
            break
    log.close()
    L = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    ev_ = {r["epoch"]: r["val_nll_per_tok"] for r in L if "val_nll_per_tok" in r and r["epoch"] > 0 and not r.get("diagnostic_only")}
    rule = "lowest val NLL per token" + (f" among completed epochs 1-{a.stop_after} (stopped after epoch {a.stop_after}; LR schedule over {a.epochs})" if a.stop_after else "")
    wjson(out / "selection.json", {"rule": rule, "val_nll": ev_, "selected": f"epoch{min(ev_, key=ev_.get)}"})


if __name__ == "__main__":
    main()
