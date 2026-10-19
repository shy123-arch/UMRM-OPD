"""B1 (vlm-r1 env, 1 GPU, local Qwen3-8B, greedy, non-thinking) = route-6 s0_llm.py jobs 2a/2 with the SAME prompts:
 (a) translate Chinese reactions into one English first-person sentence (new items without an English original + the v860
     desk5124 items whose 'English original' was Chinese), (b) third-person gt3p rewrite (harness_v1 'rewrite_third_person'
     prompt) for every TRAIN/VAL/SEALED item without a gt3p. Categories come from the frozen v1197 categories.csv (no LLM).
Output: runs/b1_llm.json"""
import re, sys, torch
from common import *
sys.path.insert(0, str(R6))
QWEN3 = "/mnt/workspace/models/Qwen3-8B"
REWRITE = ("Rewrite the following first-person description as a motion prompt for a text-to-motion model. "
           "Start with 'A person', use third person present tense, keep every body movement, direction and "
           "number of steps, do not add anything new. Output only the rewritten sentence.\n\n")
TR = ("Translate this Chinese first-person description of my reaction into ONE natural English sentence. Keep the exact "
      "meaning, first person (\"I ...\"), past tense, no extra details.\nChinese: {cn}\nReturn only the English sentence.")


def main():
    out = RUNS / "b1_llm.json"
    if out.exists():
        raise FileExistsError(out)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    inv = rjson(RUNS / "inventory.json")["rows"]
    tok = AutoTokenizer.from_pretrained(QWEN3); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(QWEN3, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()

    def run(prompts, max_new):
        res = []
        for i in range(0, len(prompts), 16):
            texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) for p in prompts[i:i + 16]]
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
            res += [tok.decode(s, skip_special_tokens=True).strip() for s in o[:, enc["input_ids"].shape[1]:]]
            print(len(res), "/", len(prompts), flush=True)
        return res
    tr_keys = sorted(k for k, r in inv.items() if r.get("to_translate"))
    trans = {k: t.strip().strip('"').splitlines()[0].strip() for k, t in zip(tr_keys, run([TR.format(cn=inv[k]["to_translate"]) for k in tr_keys], 80))}
    rw_keys = sorted(k for k, r in inv.items() if r["group"] in ("train", "val", "sealed") and not r.get("exec"))
    src = {k: trans.get(k) or inv[k]["reaction"] for k in rw_keys}
    rw = {k: t.strip().strip('"').splitlines()[0].strip() for k, t in zip(rw_keys, run([REWRITE + src[k] for k in rw_keys], 120))}
    bad_rw = [k for k in rw_keys if not rw[k].startswith("A person")]
    bad_tr = [k for k in tr_keys if re.search(r"[一-鿿]", trans[k]) or not trans[k]]
    save_json(out, {"model": QWEN3, "decoding": "greedy, enable_thinking=False", "translate_prompt": TR, "rewrite_prompt": REWRITE,
                    "translations": trans, "gt3p_rewrites": rw, "rewrites_not_starting_with_A_person": bad_rw, "translations_with_cjk_or_empty": bad_tr})
    print("translated", len(trans), "rewritten", len(rw), "bad rewrites", len(bad_rw), "bad translations", len(bad_tr))


if __name__ == "__main__":
    main()
