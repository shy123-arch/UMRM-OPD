"""Build an anonymized, complete 2026-10-15 v1197 source-data release.

The frozen source is read only. The output is a new dated directory, and the
builder refuses to overwrite either a completed or an interrupted release.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "umrm_model/outputs/data_v1197_20261015"
BASELINE = ROOT / "benchmark/judge_exploration/evaluations/astra_motion_loop/baseline636_20260915/pico_baseline_636.json"
PICO_FIELDS = ("body_joint_names", "raw_body_poses", "raw_motion_timestamp_ns")
TEXT_FIELDS = ("current_state_en", "current_state_cn", "mental_reasoning_en",
               "mental_reasoning_cn", "reaction_en", "reaction_cn", "motion_prompt_en",
               "motion_prompt_cn")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def records(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    partial = out.with_name(out.name + ".partial")
    if out.exists() or partial.exists():
        raise FileExistsError(f"release path exists: {out} or {partial}")

    frozen = json.loads((SOURCE / "FROZEN.json").read_text())
    for name in ("items.jsonl", "split.json", "data.npz", "clips.json", "provenance_new.jsonl"):
        if digest(SOURCE / name) != frozen["outputs"][name]:
            raise ValueError(f"frozen source hash mismatch: {name}")
    items = records(SOURCE / "items.jsonl")
    by_id = {r["qid"]: r for r in items}
    clips = {r["id"]: Path(r["clip_path"]) for r in json.loads((SOURCE / "clips.json").read_text())["items"]}
    base = {r["id"]: Path(r["primary_recording"]) for r in json.loads(BASELINE.read_text())["items"]}
    supplement = {r["id"]: Path(r["pico_file"]) for r in records(SOURCE / "provenance_new.jsonl")}
    pico = {**base, **supplement}
    split = json.loads((SOURCE / "split.json").read_text())
    ids = set(by_id)
    if len(items) != len(ids) or len(ids) != 1197 or ids != set(clips) or ids != set(pico) or ids != set(split):
        raise ValueError("v1197 ID sets or count differ")
    for path in (*clips.values(), *pico.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    z = np.load(SOURCE / "data.npz", allow_pickle=False)
    if set(map(str, z["ids"])) != ids:
        raise ValueError("target ID set differs")
    if args.check_only:
        print(f"preflight OK: {len(ids)} IDs; {len(base)} formal + {len(supplement)} supplemental PICO files")
        return

    partial.mkdir(parents=True)
    try:
        for subdir in ("clips", "pico", "motion"):
            (partial / subdir).mkdir()
        with (partial / "items.jsonl").open("w", encoding="utf-8") as stream:
            for item_id in sorted(ids):
                source = by_id[item_id]
                row = {"id": item_id, "cohort": source["cohort"], "category": source["category"],
                       "split_original": split[item_id], "clip": f"clips/{item_id}.mp4",
                       "pico": f"pico/{item_id}.npz", **{k: source[k] for k in TEXT_FIELDS if source.get(k)}}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        (partial / "split_original.json").write_text(json.dumps(split, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        shutil.copyfile(SOURCE / "data.npz", partial / "motion/targets.npz")
        for index, item_id in enumerate(sorted(ids), 1):
            # Stream-copy the original video/audio while stripping container and stream metadata.
            subprocess.run([args.ffmpeg, "-nostdin", "-v", "error", "-i", str(clips[item_id]),
                            "-map", "0", "-map_metadata", "-1", "-map_metadata:s", "-1", "-map_chapters", "-1",
                            "-c", "copy", str(partial / "clips" / f"{item_id}.mp4")], check=True)
            with np.load(pico[item_id], allow_pickle=False) as original:
                clean = {key: original[key] for key in PICO_FIELDS}
            clean["raw_motion_timestamp_ns"] = clean["raw_motion_timestamp_ns"] - clean["raw_motion_timestamp_ns"][0]
            np.savez_compressed(partial / "pico" / f"{item_id}.npz", **clean)
            if index % 100 == 0:
                print(f"processed {index}/{len(ids)}", flush=True)
        (partial / "README.md").write_text(
            "# UMRM v1197 · 2026-10-20\n\n"
            "1197 motion backed samples from the frozen 2026-10-15 experimental dataset: "
            "formal636=636, mocap925_224=224, mocap1014_337=337. "
            "This release does not change the formal 636 benchmark or its split.\n\n"
            "`items.jsonl` contains an allowlisted training view with relative media paths. "
            "`split_original.json` preserves the source split labels for traceability. "
            "`motion/targets.npz` contains the frozen 88-frame motion targets and masks. "
            "`pico/` keeps only body joint names, poses, and relative timestamps; "
            "`clips/` has video container metadata removed. Annotator identities, original paths, "
            "source titles, URLs, provenance, session IDs, and capture timestamps are excluded.\n\n"
            "Private access for noncommercial research only. Third-party video rights remain with their owners. "
            "Do not redistribute the clips.\n")
        sums = {}
        for path in sorted(p for p in partial.rglob("*") if p.is_file()):
            sums[str(path.relative_to(partial))] = digest(path)
        (partial / "SHA256SUMS.json").write_text(json.dumps(sums, sort_keys=True, indent=2) + "\n")
        partial.rename(out)
        print(f"built {out}: {len(ids)} items, {len(sums)} hashed files")
    except Exception:
        print(f"incomplete build left at {partial}")
        raise


if __name__ == "__main__":
    main()
