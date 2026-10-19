"""README §9 (CPU): video tensor cache for runs/ego_manifest.json items, same recipe as prep_video_cache.py (Qwen3-VL-4B processor,
16 frames, max_pixels 360*640) but reading a time WINDOW [start, end] directly from the source video (qwen_vl_utils video_start /
video_end); frame indices are shifted to the window start so the timestamps in the prompt start at ~0 s like our own clips.
Output: $VHO_CACHE/video/<id>.pt (run with VHO_CACHE=/tmp/vho_ego_cache). Usage: ego_cache.py [--workers N] [--ids a,b] [--part train|probe|all]"""
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from common import *

_P = None


def one(it):
    global _P
    p = CACHE / "video" / f"{it['id']}.pt"
    if p.exists():
        return it["id"], "exists", 0.0
    t0 = time.time()
    try:
        from transformers import AutoProcessor
        from qwen_vl_utils import process_vision_info
        if _P is None:
            _P = AutoProcessor.from_pretrained(QWEN)
        ele = {"type": "video", "video": it["path"], "nframes": NFRAMES, "max_pixels": MAX_PIXELS}
        if it.get("start") is not None:
            ele["video_start"], ele["video_end"] = float(it["start"]), float(it["end"])
        import decord
        if not getattr(decord, "_vho_patched", False):  # decord's threaded decoder fails on many Ego4D files ("Error sending packet");
            _VR = decord.VideoReader                       # num_threads=1 works and avoids the slow full-window torchvision fallback
            decord.VideoReader = lambda path, *x, **kw: _VR(path, *x, **{"num_threads": 1, **kw}); decord._vho_patched = True
        vr = decord.VideoReader(it["path"]); fps = vr.get_avg_fps(); tot = len(vr)
        s_f = int(np.ceil(float(it["start"]) * fps)) if it.get("start") is not None else 0
        e_f = min(int(np.floor(float(it["end"]) * fps)), tot - 1) if it.get("end") is not None else tot - 1
        n = e_f - s_f + 1; del vr
        if n < NFRAMES:  # short window: largest even frame count (same rule as our clips)
            ele["nframes"] = max(2, n - n % 2)
        msgs = [{"role": "user", "content": [ele, {"type": "text", "text": ""}]}]
        text = _P.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        _, vid, vk = process_vision_info(msgs, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
        vids, metas = zip(*vid); metas = [dict(m) for m in metas]
        for m in metas:  # timestamps relative to the window start (decord gives absolute indices; torchvision window-relative ones)
            fi = [int(i) for i in m["frames_indices"]]
            m["frames_indices"] = [i - s_f for i in fi] if s_f > 0 and min(fi) >= s_f else fi
        b = _P(text=[text], videos=list(vids), video_metadata=metas, do_resize=False, return_tensors="pt", **vk)
        end = _P.tokenizer("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        ids = b["input_ids"][0]; assert ids[-len(end):].tolist() == end
        out = {"prefix_ids": ids[:-len(end)].clone(), "prefix_mm": b["mm_token_type_ids"][0][:-len(end)].clone(),
               "pixel_values_videos": b["pixel_values_videos"].to(torch.bfloat16), "video_grid_thw": b["video_grid_thw"], "nframes": ele["nframes"],
               "window": [it.get("start"), it.get("end")], "src_fps": fps,
               "backend": metas[0].get("video_backend")}
        tmp = p.with_suffix(".tmp"); torch.save(out, tmp); os.replace(tmp, p)
        return it["id"], "ok", time.time() - t0
    except Exception as e:  # noqa: BLE001  (logged; item is skipped by training/probes if its cache file is missing)
        return it["id"], f"ERR {type(e).__name__}: {str(e)[:160]}", time.time() - t0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--ids", default="")
    ap.add_argument("--part", choices=["train", "probe", "all"], default="all"); a = ap.parse_args()
    (CACHE / "video").mkdir(parents=True, exist_ok=True)
    m = rjson(RUNS / "ego_manifest.json"); its = (m["probe"] if a.part != "train" else []) + (m["train"] if a.part != "probe" else [])
    if a.ids:
        want = set(a.ids.split(",")); its = [x for x in its if x["id"] in want]
    its.sort(key=lambda x: (x["path"], x.get("start") or 0))  # same source video consecutively (page cache)
    t0 = time.time(); n = 0; err = []; log = open(CACHE / f"cache_log_{a.part}.jsonl", "a")
    with ProcessPoolExecutor(a.workers) as ex:
        for k, s, dt in ex.map(one, its, chunksize=4):
            n += 1; log.write(json.dumps({"id": k, "status": s, "s": round(dt, 2)}) + "\n")
            if s.startswith("ERR"):
                err.append(k)
            if n % 200 == 0:
                log.flush(); print(n, "/", len(its), round(time.time() - t0), "s; errors", len(err), flush=True)
    print("done", n, round(time.time() - t0), "s; errors", len(err))


if __name__ == "__main__":
    main()
