"""README §9 (CPU, deterministic): build runs/ego_manifest.json = egocentric causal pre-training mix PT-C + G-E0 probe items.
Videos are read-only references: local Ego4D v2 full-scale videos (eckes) or Shawn's pre-cut Ego4D windows (EgoRe). Nothing is
rewritten by an LLM. Every video_uid used by a probe (EgoNormia verified items' videos + all EgoToM videos) is excluded from training."""
import csv, json, os, random, re
from pathlib import Path

HERE = Path(__file__).resolve().parent; RUNS = HERE / "runs"
FS = Path("/mnt/data/oss/eckes/EgoData/Ego4d/v2_packed/full_scale"); ANN = Path("/mnt/data/oss/eckes/EgoData/Ego4d/v2_packed/annotations")
X = Path("/mnt/workspace/Wilson/ck/umrm/external_data/ego_causal"); S = Path("/mnt/workspace/Shawn/projects/RynnScale/stagea_data/complete/cog_egoc")
N_GS, N_ER, SEED = 10000, 8000, 0
EGO_PROMPT = ("This first-person video ends right before the camera wearer (me) does the next thing.\n"
              "Infer why I act as I do and predict my next action. Output exactly two lines and nothing else:\n"
              "<reason> one or two first-person sentences: my goal or the situation, and why the next action makes sense\n"
              "<next> one sentence describing my next action")
PROBE_NEXT_PROMPT = ("This first-person video ends right before the camera wearer (me) does the next thing.\n"
                     "What should I do next? Output one line:\n<next> one sentence describing my next action")
cap = lambda s: (s.strip()[:1].upper() + s.strip()[1:]).rstrip(".") + "."
low1 = lambda s: s.strip()[:1].lower() + s.strip()[1:]


def main():
    out = RUNS / "ego_manifest.json"
    if out.exists():
        raise FileExistsError(out)
    have = {f[:-4] for f in os.listdir(FS) if f.endswith(".mp4")}
    N = json.load(open(X / "egonormia/repo/src/final_dataset/final_data.json"))
    ver = json.load(open(X / "egonormia/repo/src/final_dataset/verified_split.json"))["split"]
    uid = lambda k: k.rsplit("_", 1)[0]; tm = lambda k: float(k.rsplit("_", 1)[1].replace("-", "."))
    tom = {q: list(csv.DictReader(open(X / f"egotom/repo/egotom/egotom_{q}_shuffled.csv"))) for q in ("actions", "belief", "goal")}
    tom_prompts = json.load(open(X / "egotom/repo/egotom/all_prompts.json"))
    evalv = {uid(k) for k in ver} | {r["vuid"] for q in tom for r in tom[q]}
    items, probes = [], []
    # --- probes ---------------------------------------------------------------------------------------------------
    for k in ver:
        v = N[k]; opts = [(i, b) for i, b in enumerate(v["behaviors"]) if b.strip()]
        if not v["behaviors"][v["correct"]].strip() or uid(k) not in have:
            continue
        probes.append({"id": "pn_" + k, "set": "egonormia_verified", "video_uid": uid(k), "path": str(FS / f"{uid(k)}.mp4"),
                       "start": max(0.0, tm(k) - 5.0), "end": tm(k), "prompt": PROBE_NEXT_PROMPT, "prefix": "<next> ",
                       "options": [b.strip() for _, b in opts], "gold": [i for i, _ in opts].index(v["correct"])})
    for q, R in tom.items():
        for r in R:
            if r["vuid"] not in have:
                continue
            u = tom_prompts[q][r["cuid"]]["user"]; m = re.search(r"Question 1:\s*(.*?)\n", u); qtext = m.group(1).strip()
            ch = [r[f"{q}_choice_{c}"] for c in "abcd" if f"{q}_choice_{c}" in r]  # goal questions have 3 options
            probes.append({"id": f"pt_{q}_" + re.sub(r"[^A-Za-z0-9-]", "_", r["cuid"]), "set": f"egotom_{q}", "video_uid": r["vuid"],
                           "path": str(FS / f"{r['vuid']}.mp4"), "start": float(r["clip_start_time"]), "end": float(r["clip_end_time"]),
                           "prompt": f"This first-person video shows the camera wearer C.\nQuestion: {qtext}\nAnswer with one sentence.",
                           "prefix": "", "options": ch, "gold": ch.index(r[f"gt_{q}"])})
    # --- training mix PT-C -------------------------------------------------------------------------------------------
    rng = random.Random(SEED)
    G = json.load(open(ANN / "goalstep_train.json"))["videos"]; pairs = []
    for v in G:
        if v["video_uid"] in evalv or v["video_uid"] not in have:
            continue
        leaves = []
        for s in v["segments"]:
            subs = s.get("segments") or []
            leaves += [(float(u["start_time"]), float(u["end_time"]), u["step_description"].strip()) for u in subs] if subs else \
                      [(float(s["start_time"]), float(s["end_time"]), s["step_description"].strip())]
        leaves.sort()
        for a, b in zip(leaves, leaves[1:]):
            if a[2] and b[2] and b[0] - a[1] <= 10 and b[0] >= 8 and a[2].lower() != b[2].lower():
                pairs.append((v["video_uid"], v["goal_description"].strip(), a, b))
    for j, (vu, goal, a, b) in enumerate(rng.sample(pairs, N_GS)):
        items.append({"id": f"gs_{j:05d}", "src": "goalstep", "video_uid": vu, "path": str(FS / f"{vu}.mp4"), "start": b[0] - 8.0, "end": b[0],
                      "prompt": EGO_PROMPT, "target": f"<reason> I am working on: {low1(goal).rstrip('.')}. I have just done: {low1(a[2]).rstrip('.')}.\n<next> {cap(b[2])}"})
    for k in sorted(N):
        v = N[k]; c = v["correct"]
        if uid(k) in evalv or uid(k) not in have or not v["behaviors"][c].strip() or not v["justifications"][c].strip():
            continue
        items.append({"id": "en_" + k, "src": "egonormia", "video_uid": uid(k), "path": str(FS / f"{uid(k)}.mp4"), "start": max(0.0, tm(k) - 5.0),
                      "end": tm(k), "prompt": EGO_PROMPT, "target": f"<reason> {cap(v['justifications'][c])}\n<next> {cap(v['behaviors'][c])}"})
    L = json.load(open(S / "egothinker/EgoRe_clips/ego4d_long.json"))
    why = [r for r in L if re.match(r"\s*why\b", r["conversations"][0]["value"].replace("<video>", "").strip(), re.I) and r["src_video"] not in evalv
           and (S / "sources/ego4d-clips" / r["video"]).exists()]
    for j, r in enumerate(rng.sample(why, N_ER)):
        q = r["conversations"][0]["value"].replace("<video>", "").strip(); ans = r["conversations"][1]["value"].strip()
        items.append({"id": f"er_{j:05d}", "src": "egore_why", "video_uid": r["src_video"], "path": str(S / "sources/ego4d-clips" / r["video"]),
                      "start": None, "end": None, "prompt": q, "target": ans})
    assert not ({x["video_uid"] for x in items} & evalv)
    assert len({x["id"] for x in items + probes}) == len(items + probes) and not any(x["id"].endswith("_M") for x in items + probes)
    stats = {"n_train": len(items), "by_src": {s: sum(x["src"] == s for x in items) for s in ("goalstep", "egonormia", "egore_why")},
             "n_probe": len(probes), "probe_by_set": {s: sum(p["set"] == s for p in probes) for s in sorted({p["set"] for p in probes})},
             "goalstep_pairs_available": len(pairs), "egore_why_available": len(why), "eval_video_uids": len(evalv),
             "train_video_uids": len({x["video_uid"] for x in items})}
    json.dump({"rule": "README §9 (ego_build.py)", "stats": stats, "train": items, "probe": probes, "eval_video_uids": sorted(evalv)},
              open(out, "w"), ensure_ascii=False)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
