"""A1 (qwen env, 1 GPU): generated 'what/why' inner reasoning (mental_reasoning) for items that lack the human field,
with local Qwen3-VL-8B-Instruct, greedy. Input = pre-reaction clip (same 16-frame cache as the student) + the TRUE reaction
+ 4 fixed style examples (human mental_reasoning of formal TRAIN items, text only). Training-time teacher field only.
Targets: supplement TRAIN items (real) + supplement VAL items (teacher diagnostic only) + 40 formal TRAIN items that HAVE
the human field (calibration only; never used). Mirrors: mental_reasoning of 'k_M' = left/right-swapped text of k's
(dropped if ambiguous). Output: runs/mr_generated.json {"final": {id: text}, "calibration": {...}, ...}"""
import importlib.util, json, random, re, time
import torch
from common import *
from vt_io import encode, generate_multi


FEW = 4
TEMPLATE = """This first-person video ends right before I (the camera wearer) react. My actual reaction was: "{reaction}"
Write my inner reasoning at that moment in one or two first-person sentences: what I noticed in the scene and why I chose to react this way (my intention). Do not describe my body movement step by step, and do not mention the video or the camera.
Examples of the style from other situations:
{examples}
Now write the reasoning for my reaction "{reaction}". Output only the reasoning."""


def mirror_text(t):
    # same rule as a0_mirror.mirror_text (a0 imports hero-env modules at top level, so the function is re-implemented here
    # by exec of its source block)
    return _MT(t)


def main():
    out = RUNS / "mr_generated.json"
    if out.exists():
        raise FileExistsError(out)
    src = (HERE / "a0_mirror.py").read_text(); ns = {}
    block = src[src.index("LAT = {"):src.index("def mirror_joints")]
    exec("import re\n" + block, ns); global _MT; _MT = ns["mirror_text"]
    man = manifest(); rows = man["rows"]; real_tr = man["train_real"]
    fewk = sorted([k for k in real_tr if rows[k].get("mental_reasoning") and rows[k]["cohort"] == "formal636"], key=h)
    few = fewk[:FEW]; calib = fewk[FEW:FEW + 40]
    ex = "\n".join(f'Reaction: "{rows[k]["reaction"]}" -> Reasoning: "{rows[k]["mental_reasoning"]}"' for k in few)
    tgt = [k for k in real_tr + man["val"] if not rows[k].get("mental_reasoning")]
    jobs = tgt + calib
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(QWEN8B, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="flash_attention_2").eval()
    res, t0 = {}, time.time()
    for i in range(0, len(jobs), 8):
        ks = jobs[i:i + 8]
        G = generate_multi(model, [encode(k, TEMPLATE.format(reaction=rows[k]["reaction"], examples=ex)) for k in ks], n=1, max_new_tokens=110)
        for k, g in zip(ks, G):
            t = g[0]["raw"].strip().strip('"').replace("\n", " ").strip()
            res[k] = re.sub(r"^(Reasoning:\s*)", "", t).strip().strip('"')
        print(len(res), "/", len(jobs), round(time.time() - t0), "s", flush=True)
    final = {k: res[k] for k in tgt}
    mirrored, dropped = {}, []
    for k in man["train"]:
        if k.endswith("_M") and not rows[k].get("mental_reasoning"):
            base = k[:-2]; txt = final.get(base) or rows[base].get("mental_reasoning")
            if txt:
                mt, amb = _MT(txt)
                if amb:
                    dropped.append(k)
                else:
                    mirrored[k] = mt
    final.update(mirrored)
    rlow = lambda s: set(re.findall(r"[a-z]+", s.lower()))
    overlap = {k: len(rlow(rows[k]["reaction"]) & rlow(final[k])) / max(1, len(rlow(rows[k]["reaction"]))) for k in tgt}
    save_json(out, {"model": QWEN8B, "decoding": "greedy", "template": TEMPLATE, "few_shot_ids": few, "few_shot_text": ex,
                    "final": final, "n_generated": len(tgt), "n_mirrored": len(mirrored), "mirrored_dropped_ambiguous": dropped,
                    "calibration": {k: {"generated": res[k], "human": rows[k]["mental_reasoning"], "reaction": rows[k]["reaction"]} for k in calib},
                    "reaction_word_overlap_mean": sum(overlap.values()) / len(overlap), "elapsed_s": round(time.time() - t0)})
    print("generated", len(tgt), "mirrored", len(mirrored), "dropped", len(dropped))


if __name__ == "__main__":
    main()
