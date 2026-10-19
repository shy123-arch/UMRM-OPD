"""S3 privileged-information on-policy self-distillation (OPD), reverse KL (qwen env, 1 GPU).
teacher = Qwen3-VL-4B + sft_teacher LoRA (merged, frozen), prompt = PROMPT + privileged hint z (PICO motion of the real
          reaction; TRAINING ONLY)
student = Qwen3-VL-4B + sft_student LoRA (trainable), prompt = PROMPT (video only)
Per train video: student samples K responses y ~ pi_S(.|x) at temperature 1 (on-policy, no grad), then on every response
token t:  KL_t = sum_v pi_S(v|x,y<t) [log pi_S(v|x,y<t) - log pi_T(v|x,z,y<t)]   (full 151,936-token vocab, exact),
loss = token-mean KL (sampling path stop-grad, GKD-style). No reward model, no Kimodo calls during training.
Pre-registered: checkpoint = final epoch; VAL greedy predictions written after every epoch for the independent evaluator.
--gate: G-T diagnostics on VAL before training (does the teacher use z? KL with vs without z).
Usage: opd.py --out opd_v1 [--epochs 2 --k 4 --lr 5e-5 --accum 4]"""
import argparse, math, random, time
import numpy as np, torch
from common import *
from vt_io import load_model, encode, generate, generate_multi, refresh_sampler, to_dev, tokenizer


def sel(run):
    """'sft_student' -> its pre-registered selected epoch; 'sft_student/epoch1' -> that adapter explicitly."""
    if "/epoch" in run:
        return HERE / "runs" / run
    return HERE / "runs" / run / rjson(HERE / "runs" / run / "selection.json")["selected"]


def batch_with_responses(k, prompt, resp_ids, eos):
    """K sequences = same prompt + each response (+eos), right-padded. Returns model inputs, R (max resp len), mask (K,R)."""
    b = encode(k, prompt); P = b["n_prompt"]; R = max(len(r) for r in resp_ids)
    K = len(resp_ids); pad = tokenizer().pad_token_id
    ids = torch.full((K, P + R), pad, dtype=torch.long); am = torch.zeros_like(ids); m = torch.zeros(K, R, dtype=torch.bool)
    for i, r in enumerate(resp_ids):
        ids[i, :P] = b["input_ids"][0]; ids[i, P:P + len(r)] = torch.tensor(r); am[i, :P + len(r)] = 1; m[i, :len(r)] = True
    mm = torch.zeros_like(ids); mm[:, :P] = b["mm_token_type_ids"][0]
    inp = {"input_ids": ids, "attention_mask": am, "mm_token_type_ids": mm,
           "pixel_values_videos": b["pixel_values_videos"].repeat(K, 1), "video_grid_thw": b["video_grid_thw"].repeat(K, 1)}
    return inp, R, m


def resp_logits(model, inp, R, dev):
    out = model(**to_dev(inp, dev), use_cache=False, logits_to_keep=R + 1)
    return out.logits[:, :-1].float()  # (K, R, V): position j predicts response token j


def rev_kl(ls, lt, m):
    """per-token reverse KL (K,R) from student / teacher logits."""
    lps, lpt = torch.log_softmax(ls, -1), torch.log_softmax(lt, -1)
    return (lps.exp() * (lps - lpt)).sum(-1) * m


def sample(student, k, K, eos):
    student.eval()
    G = generate(student, encode(k, PROMPT), n=K, sample=True, temperature=1.0)
    student.train()
    return [g["ids"][:159] + [eos] for g in G], G


@torch.no_grad()
def val_greedy(student, keys, rows, name):
    student.eval(); f = open(RUNS / f"{name}.shard0.jsonl", "w"); acc = []; strict = []
    for k in keys:
        g = generate(student, encode(k, PROMPT))[0]; p = parse(g["raw"])
        f.write(json.dumps({"id": k, "samples": [p]}, ensure_ascii=False) + "\n"); acc.append(plan_acc(p["plan"], rows[k]["plan"])); strict.append(p["strict"])
    f.close(); student.train()
    return {"val_plan_acc": float(np.mean(acc)), "val_strict": float(np.mean(strict))}


def gate(a, dev):
    """G-T: on 32 VAL videos, KL(student-sample || teacher with z) vs (|| teacher without z); teacher must use z."""
    man = manifest(); rows = man["rows"]; keys = man["val"][:32]; eos = tokenizer().convert_tokens_to_ids("<|im_end|>")
    student = load_model(adapter=sel(a.student), merge=True, device=dev); teacher = load_model(adapter=sel(a.teacher), merge=True, device=dev)
    torch.manual_seed(0); res = {"kl_with_z": [], "kl_without_z": [], "kl_student_vs_base_teacher_prompt": []}
    for k in keys:
        resp, _ = sample(student, k, a.k, eos); student.eval()
        with torch.no_grad():
            inp, R, m = batch_with_responses(k, PROMPT, resp, eos); ls = resp_logits(student, inp, R, dev)
            for name, pr in (("kl_with_z", teacher_prompt(rows[k])), ("kl_without_z", PROMPT)):
                it, Rt, mt = batch_with_responses(k, pr, resp, eos); lt = resp_logits(teacher, it, Rt, dev)
                res[name].append(float(rev_kl(ls, lt, m.to(dev)).sum() / m.sum()))
    out = {k_: float(np.mean(v)) for k_, v in res.items() if v}; out["n"] = len(keys)
    wjson(RUNS / f"gate_GT_{a.out}.json", out); print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--student", default="sft_student"); ap.add_argument("--teacher", default="sft_teacher")
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--k", type=int, default=4); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--accum", type=int, default=4); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gate", action="store_true"); ap.add_argument("--val-every-epoch", type=int, default=1)
    a = ap.parse_args(); dev = "cuda:0"
    if a.gate:
        return gate(a, dev)
    out = RUNS / a.out
    if (out / "log.jsonl").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    man = manifest(); rows = man["rows"]; tr = man["train"][:a.limit] if a.limit else man["train"]; va = man["val"][:max(4, a.limit // 4)] if a.limit else man["val"]
    wjson(out / "args.json", {**vars(a), "student_init": str(sel(a.student)), "teacher": str(sel(a.teacher)), "split_sha256": man["split_sha256"]})
    torch.manual_seed(a.seed); random.seed(a.seed)
    student = load_model(adapter=sel(a.student), trainable=True, device=dev); teacher = load_model(adapter=sel(a.teacher), merge=True, device=dev)
    for p in teacher.parameters():
        p.requires_grad_(False)
    eos = tokenizer().convert_tokens_to_ids("<|im_end|>")
    params = [p for p in student.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = math.ceil(len(tr) / a.accum) * a.epochs; warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    log = open(out / "log.jsonl", "a"); t0 = time.time(); step = 0; run = []
    r0 = {"epoch": 0, **val_greedy(student, va, rows, f"{a.out}_e0_greedy"), "s": round(time.time() - t0)}; log.write(json.dumps(r0) + "\n"); log.flush(); print(r0, flush=True)
    sampler = load_model(device=dev); refresh_sampler(sampler, student)  # merged copy of the CURRENT student, refreshed after every step
    for ep in range(1, a.epochs + 1):
        order = tr[:]; random.shuffle(order)
        for g0 in range(0, len(order), a.accum):
            group = order[g0:g0 + a.accum]
            GS = generate_multi(sampler, [encode(k, PROMPT) for k in group], n=a.k, sample=True, temperature=1.0)  # on-policy samples
            for k, G in zip(group, GS):
                resp = [g["ids"][:159] + [eos] for g in G]
                with torch.no_grad():
                    it, Rt, mt = batch_with_responses(k, teacher_prompt(rows[k]), resp, eos); lt = resp_logits(teacher, it, Rt, dev)
                inp, R, m = batch_with_responses(k, PROMPT, resp, eos); assert R == Rt
                ls = resp_logits(student, inp, R, dev); md = m.to(dev); kl = rev_kl(ls, lt, md); loss = kl.sum() / md.sum()
                (loss / len(group)).backward()
                with torch.no_grad():
                    tl = torch.log_softmax(lt, -1).gather(-1, it["input_ids"][:, -R:].to(dev)[..., None])[..., 0]
                    ent = -(torch.softmax(ls, -1) * torch.log_softmax(ls, -1)).sum(-1)
                run.append([float(loss), float((tl * md).sum() / md.sum()), float(np.mean([parse(g["raw"])["strict"] for g in G])),
                            float(m.sum(1).float().mean()), float((ent * md).sum() / md.sum())])
                del ls, lt, kl
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            refresh_sampler(sampler, student)
            if step % 5 == 0:
                r = {"epoch": ep, "step": step, "rev_kl": float(np.mean([x[0] for x in run])), "teacher_logp_sampled": float(np.mean([x[1] for x in run])),
                     "sample_strict": float(np.mean([x[2] for x in run])), "resp_len": float(np.mean([x[3] for x in run])),
                     "student_entropy": float(np.mean([x[4] for x in run])), "lr": sched.get_last_lr()[0],
                     "s": round(time.time() - t0), "mem_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
                run = []; log.write(json.dumps(r) + "\n"); log.flush(); print(r, flush=True)
        student.save_pretrained(out / f"epoch{ep}")
        if ep % a.val_every_epoch == 0 or ep == a.epochs:
            r = {"epoch": ep, **val_greedy(student, va, rows, f"{a.out}_e{ep}_greedy"), "s": round(time.time() - t0)}; log.write(json.dumps(r) + "\n"); log.flush(); print(r, flush=True)
    wjson(out / "selection.json", {"rule": "pre-registered: final epoch", "selected": f"epoch{a.epochs}"})


if __name__ == "__main__":
    main()
