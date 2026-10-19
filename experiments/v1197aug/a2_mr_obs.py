"""A2 (vlm-r1 env, 1 GPU, local Qwen3-8B, greedy, non-thinking): perceptual part of the HUMAN inner reasoning
(mental_reasoning of real formal-636 TRAIN + VAL items). Keeps only what the camera wearer saw / heard / noticed in the scene
(= in principle inferable from the pre-reaction clip); removes intentions, decisions, goals, plans and any description of
the wearer's own upcoming movement. Mirrors 'k_M' get the left/right-swapped extract of k (dropped if ambiguous).
Leakage audit: content-word overlap with the reaction text (full reasoning vs extract) and a regex check for first-person
action / intention verbs. Output: runs/mr_obs.json {"final": {id: text}, ...}"""
import json, re, sys, torch
from common import *

QWEN3 = "/mnt/workspace/models/Qwen3-8B"
P = """Below is a first-person note written by someone about the moment just before they reacted to a situation.
Rewrite it keeping ONLY what they perceived in the scene: what they saw, heard or noticed (people, objects, events, sounds, positions).
Remove every intention, decision, goal, plan, feeling about what to do, and anything about their own next action or movement.
Keep the first person ("I see ...", "I hear ..."). Do not add anything that is not in the note. If nothing perceptual remains, output NONE.
Note: {mr}
Output only the rewritten text."""
STOP = set("i the a an to and my so of it that is in on me for this with as be was am at they their them i'm are there".split())
INTENT = re.compile(r"\b(decide|decided|want|wanted|need to|have to|must|plan|going to|will|should|try to|realize i|i'll|intend)\b", re.I)


def cw(s):
    return set(re.findall(r"[a-z]+", (s or "").lower())) - STOP


def main():
    out = RUNS / "mr_obs.json"
    if out.exists():
        raise FileExistsError(out)
    src = (HERE / "a0_mirror.py").read_text(); ns = {}
    exec("import re\n" + src[src.index("LAT = {"):src.index("def mirror_joints")], ns); mirror_text = ns["mirror_text"]
    man = manifest(); rows = man["rows"]
    keys = [k for k in man["train"] + man["val"] if not rows[k].get("mirror_of") and rows[k].get("mental_reasoning") and rows[k].get("mr_src") != "generated"]
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(QWEN3); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(QWEN3, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    res = {}
    for i in range(0, len(keys), 16):
        ks = keys[i:i + 16]
        texts = [tok.apply_chat_template([{"role": "user", "content": P.format(mr=rows[k]["mental_reasoning"])}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) for k in ks]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=120, do_sample=False)
        for k, g in zip(ks, o[:, enc["input_ids"].shape[1]:]):
            t = tok.decode(g, skip_special_tokens=True).strip().strip('"').replace("\n", " ").strip()
            res[k] = None if t.upper().startswith("NONE") or not t else t
        print(len(res), "/", len(keys), flush=True)
    final = {k: v for k, v in res.items() if v}
    dropped = []
    for k in man["train"]:
        b = rows[k].get("mirror_of")
        if b and final.get(b):
            mt, amb = mirror_text(final[b])
            if amb:
                dropped.append(k)
            else:
                final[k] = mt
    real = [k for k in keys if res.get(k)]
    ov = lambda a, b: len(cw(a) & cw(b)) / max(1, len(cw(b)))
    audit = {"n_real": len(keys), "n_none": sum(1 for k in keys if not res[k]), "n_mirrored": len(final) - len(real), "mirror_dropped_ambiguous": len(dropped),
             "reaction_overlap_full_mr": sum(ov(rows[k]["mental_reasoning"], rows[k]["reaction"]) for k in real) / len(real),
             "reaction_overlap_obs": sum(ov(final[k], rows[k]["reaction"]) for k in real) / len(real),
             "intent_regex_full_mr": sum(bool(INTENT.search(rows[k]["mental_reasoning"])) for k in real) / len(real),
             "intent_regex_obs": sum(bool(INTENT.search(final[k])) for k in real) / len(real),
             "mean_words_full_mr": sum(len(rows[k]["mental_reasoning"].split()) for k in real) / len(real),
             "mean_words_obs": sum(len(final[k].split()) for k in real) / len(real)}
    save_json(out, {"model": QWEN3, "prompt": P, "final": final, "raw": res, "audit": audit,
                    "examples": [{"reaction": rows[k]["reaction"], "full": rows[k]["mental_reasoning"], "obs": final[k]} for k in real[:12]]})
    print(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
