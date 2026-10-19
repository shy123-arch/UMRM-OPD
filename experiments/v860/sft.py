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
    def __init__(self, keys, rows, mode):
        self.keys, self.rows, self.mode = keys, rows, mode

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        r = self.rows[self.keys[i]]
        return encode(r["id"], teacher_prompt(r) if self.mode == "teacher" else PROMPT, r["target"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["student", "teacher"], required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--r", type=int, default=16); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); out = RUNS / a.out
    if (out / "log.jsonl").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    man = manifest(); rows = man["rows"]; tr, ev = man["train"], man["val"]
    if a.limit:
        tr, ev = tr[:a.limit], ev[:max(2, a.limit // 4)]
    wjson(out / "args.json", {**vars(a), "qwen": QWEN, "split_sha256": man["split_sha256"], "n_train": len(tr), "n_val": len(ev)})
    torch.manual_seed(a.seed); random.seed(a.seed)
    model = load_model(trainable=True, r=a.r)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = math.ceil(len(tr) / a.accum) * a.epochs; warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    g = torch.Generator(); g.manual_seed(a.seed)
    dl_tr = lambda: DataLoader(DS(tr, rows, a.mode), batch_size=None, shuffle=True, num_workers=2, generator=g)
    dl_ev = DataLoader(DS(ev, rows, a.mode), batch_size=None, shuffle=False, num_workers=2)
    log = open(out / "log.jsonl", "a"); step = 0; t0 = time.time()

    def evaluate(ep):
        model.eval(); s = n = 0.0
        with torch.no_grad():
            for b in dl_ev:
                lp, c = seq_logp(model, b); s += float(lp); n += float(c)
        model.train(); r = {"epoch": ep, "val_nll_per_tok": -s / n, "s": round(time.time() - t0)}
        log.write(json.dumps(r) + "\n"); log.flush(); print(r, flush=True); return r
    evaluate(0); model.train()
    for ep in range(1, a.epochs + 1):
        run = []
        for i, b in enumerate(dl_tr()):
            lp, c = seq_logp(model, b); (-lp / c / a.accum).backward(); run.append(float(-lp / c))
            if (i + 1) % a.accum == 0 or i + 1 == len(tr):
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 5 == 0:
                    r = {"epoch": ep, "step": step, "loss": sum(run) / len(run), "lr": sched.get_last_lr()[0], "s": round(time.time() - t0),
                         "mem_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
                    run = []; log.write(json.dumps(r) + "\n"); log.flush(); print(r, flush=True)
        model.save_pretrained(out / f"epoch{ep}"); evaluate(ep)
    L = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    ev_ = {r["epoch"]: r["val_nll_per_tok"] for r in L if "val_nll_per_tok" in r and r["epoch"] > 0}
    wjson(out / "selection.json", {"rule": "lowest val NLL per token", "val_nll": ev_, "selected": f"epoch{min(ev_, key=ev_.get)}"})


if __name__ == "__main__":
    main()
