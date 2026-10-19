"""Assemble the release packages from the original server's experiment artefacts (read-only on all inputs).
  data : umrm_vho_data_v1197aug/   -> ModelScope DATASET repo (private)
  ckpt : umrm_vho_ckpt/            -> ModelScope MODEL repo (private): head_pre.pt, vq_v860.pt, HERO ViMo weights (MIT)
The v860 TEST split (182 items) is withheld entirely (no text, clip, mocap or targets). VAL + the 50-item new-source sealed
block are included (sealed stays locked behind VHO_UNSEAL_SEALED=1 in the code).
Usage: python tools/build_release.py --out /mnt/workspace/Wilson/ck/umrm/release [--part data|ckpt|all]"""
import argparse, csv, hashlib, json, os, shutil, sys
from pathlib import Path
import numpy as np

U = Path("/mnt/workspace/Wilson/ck/umrm"); X = U / "umrm_model/experiments"; O = U / "umrm_model/outputs"
AUG = X / "vt_head_opd_v1197aug/runs"; V1197 = X / "vt_head_opd_v1197/runs"; DATA = O / "data_v1197_20261015"
KEEP = ("train", "val", "sealed")
FIELDS = ("id", "group", "cohort", "category", "category_src", "category_confidence", "source", "staged_film", "prev_group", "in_v860_val",
          "eval_exclude", "mirror_of", "reaction", "reaction_src", "exec", "exec_src", "plan", "plan_raw", "mental_reasoning", "mr_src", "target")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text(json.dumps(o, ensure_ascii=False, indent=1))


def build_data(out):
    m = json.loads((AUG / "data_manifest.json").read_text()); mr = json.loads((AUG / "mr_generated.json").read_text())["final"]
    rows = {}
    for k, r in m["rows"].items():
        if r["group"] not in KEEP:
            continue
        r = dict(r)
        if not r.get("mental_reasoning") and k in mr:
            r["mental_reasoning"], r["mr_src"] = mr[k], "generated:qwen3-vl-8b"
        elif r.get("mental_reasoning") and not r.get("mr_src"):
            r["mr_src"] = "human" if not r.get("mirror_of") else "human+mirror"
        base = r.get("mirror_of") or k
        rows[k] = {**{f: r[f] for f in FIELDS if f in r}, "clip": f"clips/{base}.mp4"}
    real = sorted(k for k, r in rows.items() if not r.get("mirror_of"))
    src_clip = {k: m["rows"][k]["clip"] for k in real}
    (out / "clips").mkdir(parents=True, exist_ok=True)
    for k in real:
        d = out / "clips" / f"{k}.mp4"
        if not d.exists():
            shutil.copy2(src_clip[k], d)
    BASE = U / "benchmark/judge_exploration/evaluations/astra_motion_loop/baseline636_20260915"
    refs = {r["id"]: r["primary_recording"] for r in json.loads((BASE / "pico_baseline_636.json").read_text())["items"]}
    refs.update({json.loads(l)["id"]: json.loads(l)["pico_file"] for l in (DATA / "provenance_new.jsonl").read_text().splitlines()})
    (out / "pico").mkdir(exist_ok=True)
    for k in real:
        d = out / "pico" / f"{k}.npz"
        if not d.exists():
            shutil.copy2(refs[k], d)
    (out / "motion").mkdir(exist_ok=True)
    for f in ("humanml263_full.npz", "humanml263_capped.npz"):
        shutil.copy2(AUG / "motion" / f, out / "motion" / f)
    z = np.load(DATA / "data.npz"); ids = [str(x) for x in z["ids"]]; pos = {k: i for i, k in enumerate(ids)}
    ii = [pos[k] for k in real]
    np.savez_compressed(out / "motion/targets.npz", ids=np.array(real), targets=z["targets"][ii], masks=z["masks"][ii], durations=z["durations"][ii])
    keys = {g: [k for k in m[g] if k in rows] for g in KEEP}
    split = {"data_version": "v1197aug-release", "item_group": {k: rows[k]["group"] for k in rows}, "withheld": {"test_v860": len(m["test"])}}
    dump(out / "split.json", split)
    dump(out / "data_manifest.json", {"data_version": "v1197aug-release", "split_sha256": sha(out / "split.json"), **keys, "rows": rows,
                                      "withheld": {"test_v860": len(m["test"]), "note": "sealed TEST of the v860 split; not released"}})
    shutil.copy2(AUG / "compiler_stats.json", out / "compiler_stats.json")
    for f in ("mr_generated.json", "mirror_report.json", "mr_calibration.json"):
        (out / "meta").mkdir(exist_ok=True); shutil.copy2(AUG / f, out / "meta" / f)
    # evaluator assets (our own artefacts, restricted to released items): LLM2Vec text cache, SOMA fit, PICO bone lengths
    ea = out / "eval_assets"; ea.mkdir(exist_ok=True)
    need = {r["exec"] for r in rows.values() if r.get("exec")} | {"A person stands still."}
    T = {}
    for p in (O / "kimodo_ft_v1_v860_20260927/texts.npz", X / "route6_v860_20261014/runs/v860/texts_cache.npz", AUG / "texts_cache_llm2vec.npz",
              V1197 / "texts_cache_llm2vec.npz"):
        if p.exists():
            zz = np.load(p); T.update({str(t): e for t, e in zip(zz["texts"], zz["emb"]) if str(t) in need})
    tk = sorted(T); np.savez(ea / "texts.npz", texts=np.array(tk), emb=np.stack([T[t] for t in tk]).astype(np.float32))
    f = np.load(O / "kimodo_ft_v1_v860_20260927/fit/soma_fit.npz"); fi = [i for i, k in enumerate(f["ids"]) if str(k) in set(real)]
    (ea / "fit").mkdir(exist_ok=True)
    np.savez(ea / "fit/soma_fit.npz", **{k: (f[k][fi] if f[k].ndim and f[k].shape[0] == len(f["ids"]) else f[k]) for k in f.files})
    rep = {"n_rows": len(rows), "groups": {g: len(v) for g, v in keys.items()}, "n_real": len(real), "n_mirror": len(rows) - len(real),
           "texts_cached": len(tk), "texts_missing": len(need - set(tk)), "soma_fit_items": len(fi)}
    dump(out / "meta/release_report.json", rep); print(json.dumps(rep, indent=1))


def build_ckpt(out):
    (out / "hero_vimo").mkdir(parents=True, exist_ok=True)
    shutil.copy2(X / "vt_head_opd_v860/runs/head_pre/best.pt", out / "head_pre.pt")
    shutil.copy2(X / "hero_ft_v860_20261015/runs/v860/vq/best.pt", out / "vq_v860.pt")
    for d in ("rvq_bs256_finetune_ep10", "mtrans_memo_cross_l6_bs64_ep200", "rtrans_memo_cross_l6_bs64_ep200"):
        shutil.copytree(U / "HERO/checkpoints/vimo" / d, out / "hero_vimo" / d, dirs_exist_ok=True)
    shutil.copy2(U / "HERO/LICENSE", out / "hero_vimo/LICENSE_HERO_MIT")
    print("ckpt done", sorted(p.name for p in out.iterdir()))


def build_trained(out):
    """trained artefacts of v1197aug (for inference without retraining): LoRA adapters + heads + selection files."""
    A = AUG
    for src, dst in (("sft_student/epoch1", "sft_student"), ("sft_teacher_v3/epoch1", "sft_teacher"), ("opd_v3/epoch1", "opd")):
        shutil.copytree(A / src, out / "trained_v1197aug" / dst, dirs_exist_ok=True)
    for h in ("head_A_full", "head_AB_full"):
        (out / "trained_v1197aug" / h).mkdir(parents=True, exist_ok=True); shutil.copy2(A / h / "best.pt", out / "trained_v1197aug" / h / "best.pt")
        shutil.copy2(A / h / "result.json", out / "trained_v1197aug" / h / "result.json")
    dump(out / "trained_v1197aug/README.json", {"sft_student": "student SFT epoch1 (lowest VAL NLL among completed epochs)", "sft_teacher": "teacher v3 SFT epoch1",
          "opd": "OPD v3 final (G4-v3 not passed; see docs/results/v1197aug.md)", "heads": "PICO fine-tuned action heads (A text interface; AB + visual branch)"})
    print("trained done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--part", default="all"); a = ap.parse_args()
    o = Path(a.out)
    if a.part in ("data", "all"):
        build_data(o / "umrm_vho_data_v1197aug")
    if a.part in ("ckpt", "all"):
        build_ckpt(o / "umrm_vho_ckpt")
    if a.part in ("trained", "all"):
        build_trained(o / "umrm_vho_ckpt")
