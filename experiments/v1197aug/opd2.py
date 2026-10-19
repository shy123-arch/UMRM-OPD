"""(v1197aug: run with --k 8 --epochs 1 --half-ckpt as OPD v3; mental_reasoning = human or generated, see common.manifest)
OPD v2: privileged-information on-policy self-distillation (qwen env, 1 GPU). Pre-registered in README §2.
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


def span_mask(resp_ids, R, span):
    if span == "re":
        return loss_mask(resp_ids, R)
    m = torch.zeros(len(resp_ids), R, dtype=torch.bool)  # every response token incl. the end token
    for i, r in enumerate(resp_ids):
        m[i, :len(r)] = True
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
    man = manifest(); rows = man["rows"]
    zk = {"mr_obs": lambda r: r.get("mr_obs"), "mr": lambda r: r.get("mental_reasoning") and r.get("mr_src") != "generated"}.get(ZMODE, lambda r: r.get("mental_reasoning"))
    keys = [k for k in man["val"] if zk(rows[k])]
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
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--gate", action="store_true"); ap.add_argument("--half-ckpt", action="store_true")
    ap.add_argument("--fwd-chunk", type=int, default=0); ap.add_argument("--resume", action="store_true"); ap.add_argument("--skip-e0", action="store_true"); ap.add_argument("--save-every", type=int, default=25); ap.add_argument("--gen-videos", type=int, default=0); ap.add_argument("--mr-filter", choices=["all", "human", "obs", "none", "chain"], default="all"); ap.add_argument("--teacher-base", default=None)
    ap.add_argument("--kl-span", choices=["re", "all"], default="re", help="re = <reaction>+<exec> lines (v2-v5); all = every response token incl. <plan> and end (v1, v6)")
    a = ap.parse_args(); dev = "cuda:0"
    if a.gate:
        return gate(a, dev)
    # optional data parallel: one process per GPU (each sees its GPU as cuda:0), LoRA grads summed over ranks with gloo.
    # Each optimiser step still covers the same `accum` videos (split across ranks) with loss / len(group) -> the
    # gradient equals the single-process one; step count and LR schedule are unchanged.
    ws, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    if ws > 1:
        import datetime, torch.distributed as dist
        dist.init_process_group("gloo", timeout=datetime.timedelta(minutes=90))
    out = RUNS / a.out
    if rank == 0 and (out / "log.jsonl").exists() and not a.resume:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True); man = manifest(); rows = man["rows"]
    barrier = (lambda: dist.barrier()) if ws > 1 else (lambda: None)
    tr = [k for k in man["train"] if rows[k].get("mental_reasoning")]; va = man["val"]
    if a.mr_filter == "human":  # OPD v4: only items whose inner reasoning is human-written (real + their mirrors)
        tr = [k for k in tr if rows[k].get("mr_src", "human") != "generated" and not str(rows[k].get("mr_src", "")).startswith("generated")]
    if a.mr_filter == "chain":  # OPD v8 (README §8): items with an extractive R-chain (perception + inference, decision removed)
        tr = [k for k in man["train"] if rows[k].get("mr_chain")]
    if a.mr_filter == "none":  # OPD v6: no privileged info -> every TRAIN item
        tr = list(man["train"])
    if a.mr_filter == "obs":  # OPD v5 (mr_obs): items that have an extractive perceptual reasoning
        tr = [k for k in man["train"] if rows[k].get("mr_obs")]
    tz = rjson(sel(a.teacher).parent / "args.json").get("zmode", "kin_mr")
    assert tz == ZMODE, f"teacher was trained with zmode={tz} but VHO_ZMODE={ZMODE}"
    if a.limit:
        tr, va = tr[:a.limit], va[:max(4, a.limit // 4)]
    if rank == 0:
        wjson(out / "args.json", {**vars(a), "student_init": str(sel(a.student)), "teacher": str(sel(a.teacher)), "n_train": len(tr),
                                  "split_sha256": man["split_sha256"], "world_size": ws, "zmode": ZMODE})
    torch.manual_seed(a.seed + 1000 * rank); random.seed(a.seed)
    student = load_model(adapter=sel(a.student), trainable=True, device=dev); teacher = load_model(adapter=sel(a.teacher), merge=True, device=dev, base=a.teacher_base)
    for p in teacher.parameters():
        p.requires_grad_(False)
    sampler = load_model(device=dev); refresh_sampler(sampler, student)
    student.train()  # gradient checkpointing + LoRA dropout need train mode (v2 got it from val_greedy's student.train())
    eos = tokenizer().convert_tokens_to_ids("<|im_end|>")
    params = [p for p in student.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = math.ceil(len(tr) / a.accum) * a.epochs; warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    log = open(out / ("log.jsonl" if rank == 0 else f"log_rank{rank}.jsonl"), "a"); t0 = time.time(); step = 0; run = []
    rs_path = out / "resume.pt"; start = (1, 0)
    if a.resume and rs_path.exists():  # exact continuation: LoRA weights, AdamW state, LR schedule, RNG, position in the epoch order
        st = torch.load(rs_path, map_location="cpu", weights_only=False)
        named = dict(student.named_parameters())
        with torch.no_grad():
            for n_, t_ in st["lora"].items():
                named[n_].copy_(t_.to(named[n_].device))
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); step = st["step"]; start = (st["ep"], st["g0"])
        torch.set_rng_state(st["rng_cpu"]); torch.cuda.set_rng_state(st["rng_cuda"]); refresh_sampler(sampler, student)
        log.write(json.dumps({"resumed_at_step": step, "ep": start[0], "g0": start[1], "s": 0}) + "\n"); log.flush(); print("resumed", step, start, flush=True)
    elif not a.skip_e0 and rank == 0:
        r0 = {"epoch": 0, **val_greedy(student, va, rows, f"{a.out}_e0_greedy"), "s": round(time.time() - t0)}; log.write(json.dumps(r0) + "\n"); log.flush(); print(r0, flush=True)
    random.seed(a.seed)
    for ep in range(1, a.epochs + 1):
        order = tr[:]; random.shuffle(order)  # python RNG is only used here -> identical order on resume
        if ep < start[0]:
            continue
        for g0 in range(0, len(order), a.accum):
            if (ep, g0) < start:
                continue
            grp = order[g0:g0 + a.accum]; mine = grp[rank::ws]
            GV = a.gen_videos or len(mine); GS = []
            for v0 in range(0, len(mine), GV):  # on-policy samples, GV videos x K per generate call (memory)
                GS += generate_multi(sampler, [encode(k, PROMPT) for k in mine[v0:v0 + GV]], n=a.k, sample=True, temperature=1.0)
                torch.cuda.empty_cache()
            for k, G in zip(mine, GS):
                resp = [g["ids"][:159] + [eos] for g in G]
                R = max(len(r_) for r_ in resp); m_all = span_mask(resp, R, a.kl_span); n = m_all.sum().clamp(min=1).to(dev)
                C = a.fwd_chunk or len(resp); acc = np.zeros(4)
                for c0 in range(0, len(resp), C):  # chunked forward/backward over the K samples; identical gradient (sum / total tokens)
                    rc = resp[c0:c0 + C]
                    with torch.no_grad():
                        it, Rc = batch_with_responses(k, teacher_prompt(rows[k]), rc); lt = resp_logits(teacher, it, Rc, dev)
                    inp, R2 = batch_with_responses(k, PROMPT, rc); assert R2 == Rc
                    m = span_mask(rc, Rc, a.kl_span).to(dev); ls = resp_logits(student, inp, Rc, dev)
                    rev, fwd = kls(ls, lt)
                    loss = (((1 - a.lam_fwd) * rev + a.lam_fwd * fwd) * m).sum() / n
                    (loss / len(grp)).backward()
                    with torch.no_grad():
                        ent = -(torch.softmax(ls, -1) * torch.log_softmax(ls, -1)).sum(-1)
                        acc += [float(loss), float((rev * m).sum() / n), float((fwd * m).sum() / n), float((ent * m).sum() / n)]
                    del ls, lt, rev, fwd, loss
                run.append(acc.tolist() + [float(m_all.sum(1).float().mean()), float(np.mean([parse(g["raw"])["strict"] for g in G]))])
            if ws > 1:  # sum LoRA grads over ranks (CPU gloo; ~33M params)
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
            refresh_sampler(sampler, student)
            if rank == 0 and step % a.save_every == 0:
                st = {"lora": {n_: p_.detach().cpu() for n_, p_ in student.named_parameters() if p_.requires_grad}, "opt": opt.state_dict(),
                      "sched": sched.state_dict(), "step": step, "ep": ep, "g0": g0 + a.accum, "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()}
                torch.save(st, str(rs_path) + ".tmp"); os.replace(str(rs_path) + ".tmp", rs_path)
            if rank == 0 and a.half_ckpt and step == total // 2:  # exploratory mid-training checkpoint (README: not the pre-registered one)
                student.save_pretrained(out / "half")
            if step % 5 == 0:
                r = {"epoch": ep, "step": step, **dict(zip(["loss", "rev_kl", "fwd_kl", "student_entropy", "distilled_tokens", "sample_strict"],
                                                         np.mean(run, 0).tolist())), "lr": sched.get_last_lr()[0], "s": round(time.time() - t0),
                     "mem_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
                run = []; log.write(json.dumps(r) + "\n"); log.flush(); print(r, flush=True)
        if ws > 1:  # weights must be identical on all ranks
            cs = torch.tensor([float(sum(p_.detach().float().sum() for p_ in params))]); cs_all = [torch.zeros(1) for _ in range(ws)]
            dist.all_gather(cs_all, cs); log.write(json.dumps({"epoch_end_lora_checksums": [float(x) for x in cs_all]}) + "\n"); log.flush()
        if rank == 0:
            student.save_pretrained(out / f"epoch{ep}")
            r = {"epoch": ep, **val_greedy(student, va, rows, f"{a.out}_e{ep}_greedy"), "s": round(time.time() - t0)}; log.write(json.dumps(r) + "\n"); log.flush(); print(r, flush=True)
        barrier()
    if rank == 0:
        wjson(out / "selection.json", {"rule": "pre-registered: final epoch", "selected": f"epoch{a.epochs}"})
    barrier()


if __name__ == "__main__":
    main()
