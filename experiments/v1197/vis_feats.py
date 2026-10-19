"""Visual-semantic features for head branch B (qwen env, 1 GPU). Encoder = the FROZEN SFT student (selected LoRA merged),
input = video + PROMPT + answer (teacher-forced). Kept per item, at LLM hidden_states (9, 18, 27, 36):
  (a) video tokens mean-pooled per temporal group (contiguous video-token runs; 8 for 16 frames) -> type 0
  (b) tokens of the <exec> line content (after the tag, up to the newline)                       -> type 1
--source gt   : answer = manifest target (TRAIN+VAL)   -> corpus 'gt_trainval'   (training + VAL-oracle)
--source pred : answer = raw greedy output in runs/<name>.shard*.jsonl (VAL) -> corpus 'pred_<name>'
Output: CACHE/visfeat/<corpus>.{feats.npy, meta.json}"""
import argparse, time
import numpy as np, torch
from common import *
from vt_io import load_model, encode, to_dev, tokenizer

LAYERS = (9, 18, 27, 36)


def exec_token_mask(answer, ans_ids):
    tok = tokenizer(); enc = tok(answer, add_special_tokens=False, return_offsets_mapping=True)
    assert enc["input_ids"] == ans_ids[:len(enc["input_ids"])], "answer tokenisation mismatch"
    i0 = answer.find("<exec>")
    if i0 < 0:
        m = re.search(r"(?im)^a person", answer); i0, j0 = (m.start(), m.start()) if m else (0, 0)
    else:
        j0 = i0 + len("<exec>")
    i1 = answer.find("\n", j0); i1 = len(answer) if i1 < 0 else i1
    sel = [t for t, (a, b) in enumerate(enc["offset_mapping"]) if a >= j0 and a < i1 and b > a]
    return sel or list(range(len(enc["input_ids"])))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--source", choices=["gt", "pred"], required=True)
    ap.add_argument("--name"); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    corpus = "gt_trainval" if a.source == "gt" else f"pred_{a.name}"; d = CACHE / "visfeat"; d.mkdir(parents=True, exist_ok=True)
    if (d / f"{corpus}.meta.json").exists():
        print("exists", corpus); return
    man = manifest(); rows = man["rows"]
    if a.source == "gt":
        keys = man["train"] + man["val"]; ans = {k: rows[k]["target"] for k in keys}
    else:
        R = load_jsonl_shards(a.name, RUNS); keys = man["val"]; assert set(keys) <= set(R); ans = {k: R[k]["samples"][0]["raw"] for k in keys}
    keys = keys[:a.limit] if a.limit else keys
    model = load_model(adapter=a.adapter if Path(a.adapter).is_absolute() else HERE / a.adapter, merge=True); t0 = time.time()
    feats, lens, nvid = [], [], []
    for i, k in enumerate(keys):
        b = encode(k, PROMPT, ans[k]); P = b["n_prompt"]; ids = b["input_ids"][0].tolist()
        with torch.no_grad():
            hs = model(**to_dev(b, "cuda:0"), output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states
        H = torch.stack([hs[l][0] for l in LAYERS], 1)  # (seq, nL, d)
        mm = b["mm_token_type_ids"][0].tolist(); runs, s_ = [], None
        for j, t in enumerate(mm + [0]):
            if t == 2 and s_ is None:
                s_ = j
            elif t != 2 and s_ is not None:
                runs.append((s_, j)); s_ = None
        V = torch.stack([H[s:e].float().mean(0) for s, e in runs])
        sel = exec_token_mask(ans[k], ids[P:]); X = H[[P + j for j in sel]].float()
        f = torch.cat([V, X]).to(torch.float16).cpu().numpy(); feats.append(f); lens.append(len(f)); nvid.append(len(V))
        if (i + 1) % 100 == 0:
            print(i + 1, "/", len(keys), round(time.time() - t0), "s", flush=True)
    off = np.concatenate([[0], np.cumsum(lens)[:-1]]).tolist()
    np.save(d / f"{corpus}.feats.npy", np.concatenate(feats))
    wjson(d / f"{corpus}.meta.json", {"layers": LAYERS, "ids": keys, "offsets": off, "lengths": lens, "n_video": nvid, "adapter": str(a.adapter),
                                      "source": a.source, "name": a.name, "elapsed_s": round(time.time() - t0)})
    print("saved", corpus, len(keys), "median len", int(np.median(lens)), "n_video", sorted(set(nvid)))


if __name__ == "__main__":
    main()
