"""Pre-process every TRAIN+VAL clip once with the Qwen3-VL-4B processor (16 frames, max_pixels 360*640; same budget as
route 6; clips with < 16 frames use the largest even count, route-6 clip_frames.json) -> CACHE/video/<id>.pt
{prefix_ids, prefix_mm, pixel_values_videos (bf16; the vision tower runs in bf16 anyway), video_grid_thw}. CPU only.
Usage: prep_video_cache.py [--workers N]"""
import argparse, time
from concurrent.futures import ProcessPoolExecutor
import torch
from common import *

_P = None


def nframes(k, fr):
    n = fr.get(k, NFRAMES)
    return NFRAMES if n >= NFRAMES else n - n % 2


def one(args):
    global _P
    k, clip, nf = args
    if video_path_(k).exists():
        return k, "exists"
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    if _P is None:
        _P = AutoProcessor.from_pretrained(QWEN)
    msgs = [{"role": "user", "content": [{"type": "video", "video": clip, "nframes": nf, "max_pixels": MAX_PIXELS}, {"type": "text", "text": ""}]}]
    text = _P.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    _, vid, vk = process_vision_info(msgs, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
    vids, metas = zip(*vid)
    b = _P(text=[text], videos=list(vids), video_metadata=list(metas), do_resize=False, return_tensors="pt", **vk)
    end = _P.tokenizer("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    ids = b["input_ids"][0]; assert ids[-len(end):].tolist() == end
    out = {"prefix_ids": ids[:-len(end)].clone(), "prefix_mm": b["mm_token_type_ids"][0][:-len(end)].clone(),
           "pixel_values_videos": b["pixel_values_videos"].to(torch.bfloat16), "video_grid_thw": b["video_grid_thw"], "nframes": nf}
    p = video_path_(k); tmp = p.with_suffix(".tmp"); torch.save(out, tmp); os.replace(tmp, p)
    return k, "ok"


def video_path_(k):
    return CACHE / "video" / f"{k}.pt"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); a = ap.parse_args()
    (CACHE / "video").mkdir(parents=True, exist_ok=True)
    m = manifest(); fr = {k: v["frames"] for k, v in rjson(R6RUNS / "clip_frames.json")["clips"].items()}
    jobs = [(k, m["rows"][k]["clip"], nframes(k, fr)) for k in m["train"] + m["val"]]
    t0 = time.time(); n = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for k, s in ex.map(one, jobs, chunksize=4):
            n += 1
            if n % 50 == 0:
                print(n, "/", len(jobs), round(time.time() - t0), "s", flush=True)
    wjson(CACHE / "video_manifest.json", {"n": len(jobs), "ids": [j[0] for j in jobs], "nframes": {j[0]: j[2] for j in jobs},
                                          "qwen": QWEN, "max_pixels": MAX_PIXELS, "elapsed_s": round(time.time() - t0)})
    print("done", len(jobs), round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
