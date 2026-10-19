"""B1c (vlm-r1 env, 1 GPU, Qwen3-8B greedy): repair of B1b strict rewrites that DROPPED a locomotion verb stated in the
source (walk/run/jump/turn/step/squat families). One regeneration with STRICT + 'keep every verb'. Accepted only if it passes
both the B1b addition gate and the omission check; otherwise the B1b output is kept and flagged. Output: runs/b1c_omission.json"""
import re, sys, torch
from common import *
from b1b_fidelity import STRICT, fails, QWEN3
STEM = {"run": r"\bra?n\b|runn|\bruns\b|sprint|dash|rush", "walk": r"\bwalk", "jump": r"\bjump|leap|hop", "turn": r"\bturn|spin|rotate",
        "step": r"\bstep", "squat": r"\bsquat|crouch|kneel|duck"}


def omits(src, rw):
    return [w for w, p in STEM.items() if re.search(p, src.lower()) and not re.search(p, rw.lower())]


def main():
    out = RUNS / "b1c_omission.json"
    if out.exists():
        raise FileExistsError(out)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    b = rjson(RUNS / "b1b_fidelity.json"); inv = rjson(RUNS / "inventory.json")["rows"]
    src = {k: b["translations"].get(k) or inv[k]["reaction"] for k in b["gt3p_final"]}
    redo = sorted(k for k, r in b["gt3p_final"].items() if omits(src[k], r))
    P = STRICT.replace("Output only", "Keep every verb of the original (for example walk, run, rush, step, turn, jump) in third person. Output only")
    tok = AutoTokenizer.from_pretrained(QWEN3); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(QWEN3, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    texts = [tok.apply_chat_template([{"role": "user", "content": P + src[k]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for k in redo]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        o = model.generate(**enc, max_new_tokens=100, do_sample=False)
    gen = [tok.decode(s, skip_special_tokens=True).strip().strip('"').splitlines()[0].strip() for s in o[:, enc["input_ids"].shape[1]:]]
    final, flag = dict(b["gt3p_final"]), {}
    for k, g in zip(redo, gen):
        if not fails(src[k], g) and not omits(src[k], g):
            final[k] = g
        else:
            flag[k] = {"candidate": g, "omits": omits(src[k], g), "adds": fails(src[k], g)}
    save_json(out, {"prompt": P, "regenerated": redo, "outputs": dict(zip(redo, gen)), "gt3p_final": final, "still_flagged": flag,
                    "omission_rate_after": round(sum(bool(omits(src[k], r)) for k, r in final.items()) / len(final), 3)})
    print("regenerated", len(redo), "still flagged", len(flag))
    for k, g in zip(redo, gen):
        print(k, "|", src[k], "=>", g)


if __name__ == "__main__":
    main()
