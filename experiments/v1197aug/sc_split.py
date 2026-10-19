"""G-T7a (README §7): split an N-sample VAL prediction file RUNS/<name>.shard0.jsonl into N single-sample prediction files
RUNS/<name>_kKK.shard0.jsonl (route-6 format), so every sample can be read out through the A36 head on its own. CPU only.
Usage: sc_split.py sc16_stu [16]"""
import json, sys
from common import *


def main(name, n=16):
    rows = [json.loads(l) for l in open(RUNS / f"{name}.shard0.jsonl")]
    val = set(manifest()["val"]); ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)) == len(val) and set(ids) == val, "need exactly the VAL items once"
    assert all(len(r["samples"]) == n for r in rows), "every item needs n samples"
    for k in range(n):
        with open(RUNS / f"{name}_k{k:02d}.shard0.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "samples": [r["samples"][k]]}, ensure_ascii=False) + "\n")
    print(name, "->", n, "files; strict rate", round(sum(s["strict"] for r in rows for s in r["samples"]) / (n * len(rows)), 3))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 16)
