"""Qwen3-VL-4B I/O (qwen35_532 env: transformers 5.3, peft 0.19, flash-attn 2.8). Video tensors are pre-processed once per
clip (prep_video_cache.py) and prompts are tokenised separately: prefix(video block) + prompt + end + answer. This was
verified token-for-token against joint processor output for the student prompt, the teacher prompt and a full answer
(dev/probe_processor.py)."""
import torch
from common import *

LORA_RE = r"^model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$"
N_LLM_LAYERS = 36
_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(QWEN)
    return _TOK


def load_model(adapter=None, trainable=False, device="cuda:0", r=16, merge=False, base=None):
    """base: base model path (default QWEN = Qwen3-VL-4B); e.g. the Qwen3-VL-8B teacher of OPD v6."""
    from transformers import Qwen3VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, PeftModel
    model = Qwen3VLForConditionalGeneration.from_pretrained(base or QWEN, dtype=torch.bfloat16, device_map=device,
                                                            attn_implementation="flash_attention_2")
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
        if merge and not trainable:
            model = model.merge_and_unload()
    elif trainable:
        model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules=LORA_RE, task_type="CAUSAL_LM"))
    if trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads(); model.config.use_cache = False
        for n, p in model.named_parameters():
            p.requires_grad_("lora_" in n)
        n_lora = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
        assert n_lora == N_LLM_LAYERS * 7, n_lora
    else:
        model.eval()
    return model


def video_path(k):
    return CACHE / "video" / f"{k}.pt"


_VC = {}


def video(k):
    if k not in _VC:
        _VC[k] = torch.load(video_path(k), map_location="cpu", weights_only=True)
        if len(_VC) > 256:
            _VC.pop(next(iter(_VC)))
    return _VC[k]


END = "<|im_end|>\n<|im_start|>assistant\n"


def per_frame_grid(g):
    """transformers 5.3 Qwen3-VL get_rope_index consumes one grid row per contiguous video-token run, but the processor
    splits each video into T runs separated by timestamp text -> StopIteration. Expand [T,h,w] to T x [1,h,w] (what 4.57
    did inside get_rope_index). Vision features are identical (checked in dev/smoke_vt.py)."""
    g = g.repeat_interleave(g[:, 0], dim=0).clone(); g[:, 0] = 1
    return g


def encode(k, prompt, answer=None):
    """-> model inputs (batch of 1). labels only on answer tokens (incl. the closing <|im_end|>)."""
    tok = tokenizer(); v = video(k)
    pid = v["prefix_ids"].tolist() + tok(prompt, add_special_tokens=False)["input_ids"] + tok(END, add_special_tokens=False)["input_ids"]
    aid = tok(answer + "<|im_end|>\n", add_special_tokens=False)["input_ids"][:-1] if answer is not None else []  # drop trailing "\n"
    ids = torch.tensor([pid + aid])
    mm = torch.zeros_like(ids); mm[0, :len(v["prefix_mm"])] = v["prefix_mm"]
    b = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "mm_token_type_ids": mm,
         "pixel_values_videos": v["pixel_values_videos"], "video_grid_thw": per_frame_grid(v["video_grid_thw"])}
    if answer is not None:
        lab = ids.clone(); lab[0, :len(pid)] = -100; b["labels"] = lab
    b["n_prompt"] = len(pid)
    return b


def to_dev(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items() if k not in ("labels", "n_prompt")}


def seq_logp(model, b, dev="cuda:0"):
    """sum log p(answer tokens), n answer tokens (teacher-forced)."""
    lab = b["labels"].to(dev); n_ans = int((lab != -100).sum())
    logits = model(**to_dev(b, dev), use_cache=False, logits_to_keep=n_ans + 1).logits[:, :-1].float()
    tgt = lab[:, -n_ans:]
    lp = torch.log_softmax(logits, -1).gather(-1, tgt[..., None])[..., 0]
    return lp.sum(), torch.tensor(n_ans)


GEN_GREEDY = dict(do_sample=False, temperature=None, top_p=None, top_k=None)


@torch.no_grad()
def generate(model, b, n=1, sample=False, temperature=1.0, top_p=1.0, max_new_tokens=160, dev="cuda:0"):
    inp = to_dev(b, dev)
    if n > 1:
        inp = {k: (v.repeat(n, 1) if k in ("input_ids", "attention_mask", "mm_token_type_ids") else v) for k, v in inp.items()}
        inp["pixel_values_videos"] = b["pixel_values_videos"].to(dev).repeat(n, 1); inp["video_grid_thw"] = b["video_grid_thw"].to(dev).repeat(n, 1)
    kw = dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=0) if sample else GEN_GREEDY
    out = model.generate(**inp, max_new_tokens=max_new_tokens, repetition_penalty=1.0, use_cache=True, **kw)
    gen = out[:, inp["input_ids"].shape[1]:]
    tok = tokenizer(); eos = tok.convert_tokens_to_ids("<|im_end|>")
    res = []
    for g in gen.tolist():
        g = g[:g.index(eos)] if eos in g else g
        res.append({"ids": g, "raw": tok.decode(g, skip_special_tokens=True)})
    return res


def collate_left(bs):
    """Left-pad several encode() outputs (no labels) into one generation batch (pixel/grid concatenated in order)."""
    L = max(b["input_ids"].shape[1] for b in bs); pad = tokenizer().pad_token_id; n = len(bs)
    ids = torch.full((n, L), pad, dtype=torch.long); am = torch.zeros_like(ids); mm = torch.zeros_like(ids)
    for i, b in enumerate(bs):
        l = b["input_ids"].shape[1]; ids[i, L - l:] = b["input_ids"][0]; am[i, L - l:] = 1; mm[i, L - l:] = b["mm_token_type_ids"][0]
    return {"input_ids": ids, "attention_mask": am, "mm_token_type_ids": mm,
            "pixel_values_videos": torch.cat([b["pixel_values_videos"] for b in bs]), "video_grid_thw": torch.cat([b["video_grid_thw"] for b in bs])}


@torch.no_grad()
def generate_multi(model, bs, n=1, sample=False, temperature=1.0, top_p=1.0, max_new_tokens=160, dev="cuda:0"):
    """Several videos x n samples in ONE generate call (amortises per-step overhead). Returns list (per video) of lists."""
    rep = [b for b in bs for _ in range(n)]; inp = {k: v.to(dev) for k, v in collate_left(rep).items()}
    kw = dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=0) if sample else GEN_GREEDY
    out = model.generate(**inp, max_new_tokens=max_new_tokens, repetition_penalty=1.0, use_cache=True, **kw)
    gen = out[:, inp["input_ids"].shape[1]:].tolist(); tok = tokenizer(); eos = tok.convert_tokens_to_ids("<|im_end|>"); pad = tok.pad_token_id
    res = []
    for g in gen:
        g = g[:g.index(eos)] if eos in g else [t for t in g if t != pad]
        res.append({"ids": g, "raw": tok.decode(g, skip_special_tokens=True)})
    return [res[i * n:(i + 1) * n] for i in range(len(bs))]


@torch.no_grad()
def refresh_sampler(sampler, student):
    """sampler (plain model) <- base + LoRA delta of the PEFT student (same as merge_and_unload, without touching student)."""
    sm = dict(sampler.named_modules()); n = 0
    for name, mod in student.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            tgt = sm[name.replace("base_model.model.", "", 1)]
            delta = (mod.lora_B["default"].weight.float() @ mod.lora_A["default"].weight.float()) * mod.scaling["default"]
            tgt.weight.copy_((mod.base_layer.weight.float() + delta).to(tgt.weight.dtype)); n += 1
    assert n == N_LLM_LAYERS * 7, n
