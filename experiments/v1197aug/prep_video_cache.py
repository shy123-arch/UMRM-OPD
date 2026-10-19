"""Pre-process every TRAIN+VAL clip once with the Qwen3-VL-4B processor (16 frames, max_pixels 360*640; same budget as
route 6; clips with < 16 frames use the largest even count, route-6 clip_frames.json) -> CACHE/video/<id>.pt
{prefix_ids, prefix_mm, pixel_values_videos (bf16; the vision tower runs in bf16 anyway), video_grid_thw}. CPU only.
Usage: prep_video_cache.py [--workers N]"""
import argparse, time
from concurrent.futures import ProcessPoolExecutor
import torch
from common import *

_P = None


def nframes(k, fr, clip=None):
    """route-6 rule: 16 frames; shorter clips use the largest even count. Frame count from route-6 clip_frames.json, else decord."""
    n = fr.get(k)
    if n is None:
        import decord
        n = len(decord.VideoReader(clip))
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
    if k.endswith("_M"):  # mirror augmentation: horizontal flip of every frame (T, C, H, W) before patching
        vids = tuple(v.flip(-1) for v in vids)
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
    real = Path("/tmp/vho1197_cache/video")
    for k in m["train"] + m["val"]:  # real items: byte-identical v1197 cache (symlink; only for the default 16-frame cache)
        if NFRAMES == 16 and not k.endswith("_M") and not video_path_(k).exists():
            video_path_(k).symlink_to(real / f"{k}.pt")
    fr.update({k + "_M": v for k, v in fr.items()})
    jobs = [(k, m["rows"][k]["clip"], nframes(k, fr, m["rows"][k]["clip"])) for k in m["train"] + m["val"] + m.get("sealed", [])]
    t0 = time.time(); n = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for k, s in ex.map(one, jobs, chunksize=4):
            n += 1
            if n % 50 == 0:
                print(n, "/", len(jobs), round(time.time() - t0), "s", flush=True)
    wjson(CACHE / ("video_manifest_sealed.json" if m.get("sealed") else "video_manifest.json"), {"n": len(jobs), "ids": [j[0] for j in jobs], "nframes": {j[0]: j[2] for j in jobs},
                                          "qwen": QWEN, "max_pixels": MAX_PIXELS, "elapsed_s": round(time.time() - t0)})
    print("done", len(jobs), round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
