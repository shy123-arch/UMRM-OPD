"""Interface-A text encoder for the T->M head: FROZEN Qwen3-VL-4B base (no LoRA, text only, LLM part only).
Text is encoded as "<exec>" + " " + caption (the same tag the V->T model writes before its motion caption), and the hidden
states of the caption tokens (max 48) are kept at LLM layers LAYERS (hidden_states index; 36 = last, after final norm).
Corpora -> CACHE/textfeat/<name>.{feats.npy (fp16 [Ntok, nL, 2560]), meta.json (texts, offsets, lengths, ...)}:
  hml3d_{train,val,test} : every HumanML3D caption (incl. #from#to sub-segments, MoMask length rules 40 <= len < 200)
  gt3p_trainval           : route-6 v860 <exec> targets (gt3p) of TRAIN+VAL rows (TEST never read)
Library use: encode(texts) -> list of [L_i, nL, 2560] fp16 arrays (for VLM-predicted <exec> texts at generation time).
Usage: textenc.py --corpus hml3d|v860 [--bs 128]"""
import argparse, time
import numpy as np, torch
from common import *

LAYERS = (9, 18, 27, 36)
MAXTOK = 48
_M = None


def _model(dev="cuda:0"):
    global _M
    if _M is None:
        from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer
        m = Qwen3VLForConditionalGeneration.from_pretrained(QWEN, dtype=torch.bfloat16, device_map=dev, attn_implementation="flash_attention_2").eval()
        _M = (m.model.language_model, AutoTokenizer.from_pretrained(QWEN))
    return _M


def norm_caption(t):
    t = " ".join(t.strip().split())
    return t[:1].upper() + t[1:] if t else t


@torch.no_grad()
def encode(texts, bs=128, dev="cuda:0"):
    lm, tok = _model(dev); pre = tok("<exec>", add_special_tokens=False)["input_ids"]; out = []
    for s in range(0, len(texts), bs):
        ch = texts[s:s + bs]
        cap = [tok(" " + norm_caption(t), add_special_tokens=False)["input_ids"][:MAXTOK] for t in ch]
        L = max(len(pre) + len(c) for c in cap)
        ids = torch.full((len(ch), L), tok.pad_token_id, dtype=torch.long); am = torch.zeros_like(ids)
        for i, c in enumerate(cap):
            ids[i, :len(pre) + len(c)] = torch.tensor(pre + c); am[i, :len(pre) + len(c)] = 1
        hs = lm(input_ids=ids.to(dev), attention_mask=am.to(dev), output_hidden_states=True, use_cache=False).hidden_states
        H = torch.stack([hs[l] for l in LAYERS], 2)  # (b, L, nL, d)
        for i, c in enumerate(cap):
            out.append(H[i, len(pre):len(pre) + len(c)].to(torch.float16).cpu().numpy())
    return out


def hml3d_corpus(split):
    names = [l.strip() for l in (HML3D / f"{split}.txt").read_text().splitlines() if l.strip()]
    items = []
    for n in names:
        p = HML3D / "new_joint_vecs" / f"{n}.npy"; tp = HML3D / "texts" / f"{n}.txt"
        if not p.exists() or not tp.exists():
            continue
        T = np.load(p, mmap_mode="r").shape[0]
        for line in tp.read_text().splitlines():
            parts = line.strip().split("#")
            if len(parts) < 4 or not parts[0].strip():
                continue
            f, t = float(parts[2]) if parts[2] else 0.0, float(parts[3]) if parts[3] else 0.0
            f, t = (0.0, 0.0) if (np.isnan(f) or np.isnan(t)) else (f, t)
            a, b = (0, T) if (f == 0.0 and t == 0.0) else (int(f * 20), min(T, int(t * 20)))
            if 40 <= b - a < 200:
                items.append({"motion": n, "text": parts[0].strip(), "start": a, "end": b})
    return items


def write(name, items, bs):
    d = CACHE / "textfeat"; d.mkdir(parents=True, exist_ok=True)
    if (d / f"{name}.meta.json").exists():
        print("exists", name); return
    t0 = time.time(); texts = [x["text"] for x in items]; uniq = sorted(set(texts)); feats = encode(uniq, bs)
    lens = np.array([len(f) for f in feats]); off = np.concatenate([[0], np.cumsum(lens)[:-1]])
    mm = np.lib.format.open_memmap(d / f"{name}.feats.npy", mode="w+", dtype=np.float16, shape=(int(lens.sum()), len(LAYERS), feats[0].shape[-1]))
    for o, f in zip(off, feats):
        mm[o:o + len(f)] = f
    mm.flush(); idx = {t: i for i, t in enumerate(uniq)}
    for x in items:
        x["tid"] = idx[x["text"]]
    wjson(d / f"{name}.meta.json", {"layers": LAYERS, "maxtok": MAXTOK, "template": "<exec> + ' ' + caption (first letter upper)",
                                    "qwen": QWEN, "texts": uniq, "offsets": off.tolist(), "lengths": lens.tolist(), "items": items,
                                    "elapsed_s": round(time.time() - t0)})
    print(name, "items", len(items), "unique texts", len(uniq), "tokens", int(lens.sum()), "s", round(time.time() - t0), flush=True)


DUR_FRAMES = {"1.5": 30, "2.5": 50, "3.5": 70, "4.4": 88}
FALLBACK = "A person stands still."


def pred_items(name):
    """VAL items from VLM predictions (route-6 jsonl format): text = first sample's <exec> (fallback if missing),
    nframes = predicted <plan> duration bin (median train duration 3.1 s -> 62 if missing/invalid)."""
    R = load_jsonl_shards(name, RUNS); keys = manifest()[os.environ.get("VHO_SPLIT", "val")]; assert set(keys) <= set(R), "missing predictions"
    return [{"id": k, "text": R[k]["samples"][0]["exec"] or FALLBACK, "nframes": DUR_FRAMES.get(R[k]["samples"][0]["plan"].get("duration"), 62)} for k in keys]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--corpus", choices=["hml3d", "gt3p", "pred", "oracle"], required=True); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--name"); a = ap.parse_args()
    if a.corpus == "pred":
        write(f"pred_{a.name}", pred_items(a.name), a.bs); return
    if a.corpus == "oracle":  # gt3p text + GT plan-label duration (= route-6 O_TPH_lab text/length inputs)
        m = manifest(); write("oracle_gt3p_val", [{"id": k, "text": m["rows"][k]["exec"], "nframes": DUR_FRAMES[m["rows"][k]["plan"]["duration"]]} for k in m["val"]], a.bs); return
    if a.corpus == "gt3p":
        m = manifest(); write("gt3p_trainval", [{"id": k, "text": m["rows"][k]["exec"], "group": m["rows"][k]["group"]} for k in m["train"] + m["val"]], a.bs)
    else:
        for s in ("train", "val", "test"):
            write(f"hml3d_{s}", hml3d_corpus(s), a.bs)


if __name__ == "__main__":
    main()
