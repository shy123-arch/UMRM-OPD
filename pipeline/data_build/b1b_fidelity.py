"""B1b (vlm-r1 env, 1 GPU, Qwen3-8B greedy): fidelity repair of the B1 gt3p rewrites (NEW rewrites only; inherited v860
gt3p untouched). Gate (fixed before regeneration): a rewrite FAILS if len(words) > 2 x source, or it states a step/repeat
count absent from the source ("two steps", "three times", ...). Failures are regenerated once with a stricter prompt
(route-6 prompt + explicit no-addition constraints); the stricter output is kept if it passes the gate, otherwise the
shorter of the two (flagged). Also: 3 translations kept the Chinese word 羽毛球 -> deterministic replacement 'shuttlecock'
(their rewrites are regenerated). Output: runs/b1b_fidelity.json"""
import re, sys, torch
from common import *
QWEN3 = "/mnt/workspace/models/Qwen3-8B"
STRICT = ("Rewrite the following first-person description as a motion prompt for a text-to-motion model. "
          "Start with 'A person', use third person present tense. Keep exactly the movements that are stated and nothing else: "
          "do NOT add steps, walking, turning, directions, repetitions, counts, speeds or body parts that the original does not "
          "mention. If the original does not mention moving the feet, do not mention steps. Keep it about as short as the "
          "original. Output only the rewritten sentence.\n\n")
COUNT = re.compile(r"\b(one|two|three|four|five|six|\d+)\s+(?:\w+\s+)?(steps?|times|paces|strides)\b|\btwice\b|\bthrice\b", re.I)


def fails(src, rw):
    return len(rw.split()) > 2 * len(src.split()) or (bool(COUNT.search(rw)) and not COUNT.search(src)) or not rw.startswith("A person")


def main():
    out = RUNS / "b1b_fidelity.json"
    if out.exists():
        raise FileExistsError(out)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    b1 = rjson(RUNS / "b1_llm.json"); inv = rjson(RUNS / "inventory.json")["rows"]
    trans = {k: v.replace("羽毛球", " shuttlecock").replace("the  shuttlecock", "the shuttlecock") for k, v in b1["translations"].items()}
    fixed_tr = sorted(k for k in trans if trans[k] != b1["translations"][k])
    src = {k: trans.get(k) or inv[k]["reaction"] for k in b1["gt3p_rewrites"]}
    redo = sorted(k for k, r in b1["gt3p_rewrites"].items() if fails(src[k], r) or k in fixed_tr)
    tok = AutoTokenizer.from_pretrained(QWEN3); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(QWEN3, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    gen = []
    for i in range(0, len(redo), 16):
        texts = [tok.apply_chat_template([{"role": "user", "content": STRICT + src[k]}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for k in redo[i:i + 16]]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=100, do_sample=False)
        gen += [tok.decode(s, skip_special_tokens=True).strip().strip('"').splitlines()[0].strip() for s in o[:, enc["input_ids"].shape[1]:]]
        print(len(gen), "/", len(redo), flush=True)
    final, flag = dict(b1["gt3p_rewrites"]), {}
    for k, g in zip(redo, gen):
        old = b1["gt3p_rewrites"][k]
        if not fails(src[k], g):
            final[k] = g
        else:
            final[k] = min([x for x in (g, old) if x.startswith("A person")] or [g], key=lambda x: len(x.split())); flag[k] = "still_fails_gate"
    save_json(out, {"model": QWEN3, "strict_prompt": STRICT, "gate": "len>2x source OR new count phrase OR not 'A person'",
                    "translations": trans, "translations_fixed": fixed_tr, "regenerated": redo, "strict_outputs": dict(zip(redo, gen)),
                    "gt3p_final": final, "still_failing": flag,
                    "fail_rate_before": round(sum(fails(src[k], r) for k, r in b1["gt3p_rewrites"].items()) / len(src), 3),
                    "fail_rate_after": round(sum(fails(src[k], r) for k, r in final.items()) / len(src), 3)})
    print("regenerated", len(redo), "still failing", len(flag))


if __name__ == "__main__":
    main()
