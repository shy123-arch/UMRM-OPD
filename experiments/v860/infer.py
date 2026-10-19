"""Generate <reaction>/<exec>/<plan> for a split (qwen env, 1 GPU). Output RUNS/<name>.shard<i>.jsonl, one record per
item: {"id", "samples": [parse(raw) for each sample]} -- the route-6 format. --prompt teacher uses the privileged hint
(diagnostic only: VAL hint = GT motion info). TEST is not reachable (manifest() drops it).
Usage: infer.py --adapter runs/sft_student/epoch2 --name sft_student_greedy [--k 8 --temp 1.0] [--shard 0/1]"""
import argparse, time
import torch
from common import *
from vt_io import load_model, encode, generate


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter"); ap.add_argument("--name", required=True); ap.add_argument("--split", default="val")
    ap.add_argument("--prompt", default="student", choices=["student", "teacher"]); ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--temp", type=float, default=1.0); ap.add_argument("--top-p", type=float, default=1.0); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); si, sn = map(int, a.shard.split("/"))
    man = manifest(); keys = man[a.split][si::sn]
    keys = keys[:a.limit] if a.limit else keys
    out = RUNS / f"{a.name}.shard{si}.jsonl"; done = set(load_jsonl_shards(a.name, RUNS)) if out.exists() else set()
    model = load_model(adapter=(HERE / a.adapter) if a.adapter and not Path(a.adapter).is_absolute() else a.adapter, merge=True)
    torch.manual_seed(1234 + si); t0 = time.time(); f = open(out, "a")
    for i, k in enumerate(keys):
        if k in done:
            continue
        r = man["rows"][k]; b = encode(k, teacher_prompt(r) if a.prompt == "teacher" else PROMPT)
        G = generate(model, b, n=a.k, sample=a.k > 1, temperature=a.temp, top_p=a.top_p)
        f.write(json.dumps({"id": k, "samples": [parse(g["raw"]) for g in G]}, ensure_ascii=False) + "\n"); f.flush()
        if (i + 1) % 20 == 0:
            print(i + 1, "/", len(keys), round(time.time() - t0), "s", flush=True)
    wjson(RUNS / f"{a.name}.args.json", vars(a)); print("done", len(keys), round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
