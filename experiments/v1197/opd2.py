"""OPD v2: privileged-information on-policy self-distillation (qwen env, 1 GPU). Pre-registered in README §2.
teacher = Qwen3-VL-4B + sft_teacher_v2 LoRA (merged, FROZEN); prompt = PROMPT + kinematic hint + my recorded inner
          reasoning (mental_reasoning; 'what/why'). TRAINING ONLY.
student = Qwen3-VL-4B + sft_student LoRA (trainable); prompt = PROMPT (video only).
Training items: TRAIN items that have mental_reasoning (formal 636 part; items without 'what' information are not used).
Per video: K on-policy samples (T=1, merged sampler refreshed after every optimiser step). Per response token t (full vocab):
  loss_t = 0.5 * KL(pi_S || pi_T) + 0.5 * KL(pi_T || pi_S)
only on tokens of the <reaction> and <exec> lines: the <plan> line and the end token are EXCLUDED (v860: reverse KL on the
plan line collapsed direction/duration to the majority labels). Sampling path is stop-grad (GKD-style). No generator calls.
Checkpoint = final epoch (pre-registered). VAL greedy predictions are written after every epoch.
--gate: G-T2 KL part (VAL items with mental_reasoning): masked KL(student-sample) against teacher-with-z vs teacher-without-z."""
import argparse, math, random, time
import numpy as np, torch
from common import *
from vt_io import load_model, encode, generate, generate_multi, refresh_sampler, to_dev, tokenizer


def sel(run):
    if "/epoch" in run:
        return HERE / "runs" / run
    return HERE / "runs" / run / rjson(HERE / "runs" / run / "selection.json")["selected"]


def batch_with_responses(k, prompt, resp_ids):
    b = encode(k, prompt); P = b["n_prompt"]; R = max(len(r) for r in resp_ids); K = len(resp_ids); pad = tokenizer().pad_token_id
    ids = torch.full((K, P + R), pad, dtype=torch.long); am = torch.zeros_like(ids)
    for i, r in enumerate(resp_ids):
        ids[i, :P] = b["input_ids"][0]; ids[i, P:P + len(r)] = torch.tensor(r); am[i, :P + len(r)] = 1
    mm = torch.zeros_like(ids); mm[:, :P] = b["mm_token_type_ids"][0]
    return {"input_ids": ids, "attention_mask": am, "mm_token_type_ids": mm,
            "pixel_values_videos": b["pixel_values_videos"].repeat(K, 1), "video_grid_thw": b["video_grid_thw"].repeat(K, 1)}, R


def loss_mask(resp_ids, R):
    """(K,R) True on response tokens BEFORE the <plan> line (i.e. <reaction> and <exec> lines); plan line + end excluded."""
    tok = tokenizer(); m = torch.zeros(len(resp_ids), R, dtype=torch.bool)
    for i, r in enumerate(resp_ids):
        body = r[:-1]  # last = end token
        text = tok.decode(body); cut = text.find("<plan")
        n = len(body)
        if cut >= 0:
            n = next(j for j in range(len(body)) if len(tok.decode(body[:j + 1])) > cut)
            # drop a trailing newline-only token before the tag from the distilled span (keeps the span = two lines)
        m[i, :n] = True
    return m


def resp_logits(model, inp, R, dev):
    return model(**to_dev(inp, dev), use_cache=False, logits_to_keep=R + 1).logits[:, :-1].float()


def kls(ls, lt):
    lps, lpt = torch.log_softmax(ls, -1), torch.log_softmax(lt, -1)
    rev = (lps.exp() * (lps - lpt)).sum(-1); fwd = (lpt.exp() * (lpt - lps)).sum(-1)
    return rev, fwd


@torch.no_grad()
def val_greedy(student, keys, rows, name):
    student.eval(); f = open(RUNS / f"{name}.shard0.jsonl", "w"); acc = []
    for k in keys:
        p = parse(generate(student, encode(k, PROMPT))[0]["raw"]); f.write(json.dumps({"id": k, "samples": [p]}, ensure_ascii=False) + "\n")
        acc.append(plan_acc(p["plan"], rows[k]["plan"]))
    f.close(); student.train(); return {"val_plan_acc": float(np.mean(acc))}


def gate(a, dev):
    man = manifest(); rows = man["rows"]; keys = [k for k in man["val"] if rows[k].get("mental_reasoning")]
    eos = tokenizer().convert_tokens_to_ids("<|im_end|>")
    sampler = load_model(adapter=sel(a.student), merge=True, device=dev); teacher = load_model(adapter=sel(a.teacher), merge=True, device=dev)
    torch.manual_seed(0); res = {"kl_with_z": [], "kl_without_z": [], "rev_with_z": [], "rev_without_z": []}
    for g0 in range(0, len(keys), 4):
        grp = keys[g0:g0 + 4]
        for k, G in zip(grp, generate_multi(sampler, [encode(k, PROMPT) for k in grp], n=a.k, sample=True)):
            resp = [g["ids"][:159] + [eos] for g in G]
            with torch.no_grad():
                inp, R = batch_with_responses(k, PROMPT, resp); m = loss_mask(resp, R).to(dev); ls = resp_logits(sampler, inp, R, dev)
                for nm, pr in (("with_z", teacher_prompt(rows[k])), ("without_z", PROMPT)):
                    it, _ = batch_with_responses(k, pr, resp); rev, fwd = kls(ls, resp_logits(teacher, it, R, dev))
                    res["kl_" + nm].append(float(((0.5 * rev + 0.5 * fwd) * m).sum() / m.sum())); res["rev_" + nm].append(float((rev * m).sum() / m.sum()))
    out = {k_: float(np.mean(v)) for k_, v in res.items()}; out["n"] = len(keys); out["ratio"] = out["kl_with_z"] / max(out["kl_without_z"], 1e-9)
    save_json(RUNS / f"gate_GT2_{a.out}.json", out); print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--student", default="sft_student"); ap.add_argument("--teacher", default="sft_teacher_v2")
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--k", type=int, default=4); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--accum", type=int, default=4); ap.add_argument("--lam_fwd", type=float, default=0.5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--gate", action="store_true")
    a = ap.parse_args(); dev = "cuda:0"
    if a.gate:
        return gate(a, dev)
    out = RUNS / a.out
    if (out / "log.jsonl").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True); man = manifest(); rows = man["rows"]
    tr = [k for k in man["train"] if rows[k].get("mental_reasoning")]; va = man["val"]
    if a.limit:
        tr, va = tr[:a.limit], va[:max(4, a.limit // 4)]
    wjson(out / "args.json", {**vars(a), "student_init": str(sel(a.student)), "teacher": str(sel(a.teacher)), "n_train": len(tr), "split_sha256": man["split_sha256"]})
    torch.manual_seed(a.seed); random.seed(a.seed)
    student = load_model(adapter=sel(a.student), trainable=True, device=dev); teacher = load_model(adapter=sel(a.teacher), merge=True, device=dev)
    for p in teacher.parameters():
        p.requires_grad_(False)
    sampler = load_model(device=dev); refresh_sampler(sampler, student)
    eos = tokenizer().convert_tokens_to_ids("<|im_end|>")
    params = [p for p in student.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = math.ceil(len(tr) / a.accum) * a.epochs; warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    log = open(out / "log.jsonl", "a"); t0 = time.time(); step = 0; run = []
    r0 = {"epoch": 0, **val_greedy(student, va, rows, f"{a.out}_e0_greedy"), "s": round(time.time() - t0)}; log.write(json.dumps(r0) + "\n"); log.flush(); print(r0, flush=True)
    for ep in range(1, a.epochs + 1):
        order = tr[:]; random.shuffle(order)
        for g0 in range(0, len(order), a.accum):
            grp = order[g0:g0 + a.accum]
            for k, G in zip(grp, generate_multi(sampler, [encode(k, PROMPT) for k in grp], n=a.k, sample=True, temperature=1.0)):
                resp = [g["ids"][:159] + [eos] for g in G]
                with torch.no_grad():
                    it, R = batch_with_responses(k, teacher_prompt(rows[k]), resp); lt = resp_logits(teacher, it, R, dev)
                inp, R2 = batch_with_responses(k, PROMPT, resp); assert R2 == R
                m = loss_mask(resp, R).to(dev); ls = resp_logits(student, inp, R, dev)
                rev, fwd = kls(ls, lt); n = m.sum().clamp(min=1)
                loss = (((1 - a.lam_fwd) * rev + a.lam_fwd * fwd) * m).sum() / n
                (loss / len(grp)).backward()
                with torch.no_grad():
                    ent = -(torch.softmax(ls, -1) * torch.log_softmax(ls, -1)).sum(-1)
                run.append([float(loss), float((rev * m).sum() / n), float((fwd * m).sum() / n), float((ent * m).sum() / n),
                            float(m.sum(1).float().mean()), float(np.mean([parse(g["raw"])["strict"] for g in G]))])
                del ls, lt, rev, fwd
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            refresh_sampler(sampler, student)
            if step % 5 == 0:
                r = {"epoch": ep, "step": step, **dict(zip(["loss", "rev_kl", "fwd_kl", "student_entropy", "distilled_tokens", "sample_strict"],
                                                         np.mean(run, 0).tolist())), "lr": sched.get_last_lr()[0], "s": round(time.time() - t0),
                     "mem_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
                run = []; log.write(json.dumps(r) + "\n"); log.flush(); print(r, flush=True)
        student.save_pretrained(out / f"epoch{ep}")
        r = {"epoch": ep, **val_greedy(student, va, rows, f"{a.out}_e{ep}_greedy"), "s": round(time.time() - t0)}; log.write(json.dumps(r) + "\n"); log.flush(); print(r, flush=True)
    wjson(out / "selection.json", {"rule": "pre-registered: final epoch", "selected": f"epoch{a.epochs}"})


if __name__ == "__main__":
    main()
