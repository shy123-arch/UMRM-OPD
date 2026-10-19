"""UMRM V->T->M pipeline: Qwen3-VL-4B (video -> <reaction>/<exec>/<plan>, LoRA SFT) + own T->M action head
(HumanML3D-pretrained RVQ + masked/residual transformers, Qwen text semantics [interface A] + optional zero-gated visual
semantics [interface B]) + privileged-information on-policy self-distillation (OPD, reverse+forward KL).
All locations come from environment variables (see configs/paths.env.example); defaults are the original server.
Guide: docs/GUIDE.md. Every training/eval script imports this module."""
import hashlib, json, os, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # pipeline/
REPO = HERE.parent
THIRD = REPO / "third_party" / "umrm_eval"      # vendored helper modules (plan extractor, HERO glue, route-6 evaluator)
EXP = THIRD                                     # legacy name: scripts address helpers as EXP / "<module dir>/<file>"
R6 = THIRD / "route6_v860_20261014"
_E = os.environ.get
DATA = Path(_E("VHO_DATA", "/mnt/workspace/Wilson/ck/umrm/release/umrm_vho_data_v1197aug"))   # released data package
CKPT = Path(_E("VHO_CKPT", "/mnt/workspace/Wilson/ck/umrm/release/umrm_vho_ckpt"))            # released checkpoints
RUNS = Path(_E("VHO_RUNS", str(REPO / "runs")))                                               # all outputs
CACHE = Path(_E("VHO_CACHE", "/tmp/vho_repo_cache"))                                          # regenerable caches
QWEN = _E("VHO_QWEN", "/mnt/workspace/Wilson/ck/umrm/models/Qwen3-VL-4B-Instruct")
QWEN8B = _E("VHO_QWEN8B", "/mnt/workspace/Wilson/MemoGen/models/Qwen3-VL-8B-Instruct")
HML3D = Path(_E("VHO_HML3D", "/mnt/workspace/Wilson/ck/umrm/LaMP/dataset/HumanML3D"))
SALT = "umrm-vho-v1197aug-20261016"
LIMIT = int(_E("VHO_LIMIT", "0"))  # smoke tests: keep only the first N TRAIN and N//4 VAL items

NFRAMES, MAX_PIXELS = int(_E("VHO_NFRAMES", "16")), 360 * 640  # G-T7b: VHO_NFRAMES=32 together with its own VHO_CACHE
PLAN_FIELDS = ("locomotion", "direction", "turn", "posture", "duration")
# groups any script may load. 'sealed' (50 new-source items) only with VHO_UNSEAL_SEALED=1, which the pre-registered
# confirmation C-sealed sets (README §2); v860 TEST stays sealed for everything.
EVAL_GROUPS = ("train", "val") + (("sealed",) if os.environ.get("VHO_UNSEAL_SEALED") == "1" else ())

PROMPT = """This first-person video ends right before the camera wearer (me) reacts to what is happening.
Predict what I do next in the following 1-4 seconds. Output exactly three lines and nothing else:
<reaction> one first-person sentence describing my reaction
<exec> one third-person caption for a motion generator, starting with "A person", at most 30 words, body movements only
<plan> locomotion=stay|step|walk|run; direction=none|forward|forward-right|right|back-right|back|back-left|left|forward-left; turn=none|left|right|around; posture=upright|crouch|jump; duration=1.5|2.5|3.5|4.4"""


def hint_text(row):
    """v1 privileged hint (identical wording to vt_head_opd_v860): kinematics of the recorded PICO reaction."""
    p, l = row["plan_raw"], row["plan"]
    yaw = p["yaw_deg"]; turn = "no clear turn" if abs(yaw) < 30 else f"a {abs(yaw):.0f}-degree turn to the {'left' if yaw > 0 else 'right'}"
    return (f"Privileged hint (measured from the motion capture of my real reaction; use it only to make the prediction "
            f"physically consistent, and still ground the reaction in what the video shows): the reaction lasts about "
            f"{p['duration_s']:.1f} s; my body travels {p['distance_m']:.2f} m ({p['forward_m']:+.2f} m forward, "
            f"{p['right_m']:+.2f} m to the right) with {turn}; locomotion={l['locomotion']}, direction={l['direction']}, "
            f"posture={l['posture']}.")


ZMODE = os.environ.get("VHO_ZMODE", "kin_mr")  # privileged-z mode of the TEACHER prompt (README §5)
#   kin_mr : kinematic hint + inner reasoning (human or generated)   [v2-v4 teachers; default, unchanged]
#   mr     : human inner reasoning only (no kinematics, no generated text)
#   mr_obs : only the perceptual part of the human inner reasoning (what I saw/heard/noticed; Qwen3-8B extraction)
REASON_HDR = ("Privileged reason (my own inner reasoning at that moment, written down afterwards; it tells what I "
              "noticed and why I reacted; the reaction must still be something visible in my body movement): ")
OBS_HDR = ("Privileged observation (what I noticed in the scene just before reacting; it contains no decision or plan; "
           "ground the reaction in it and in the video): ")


CHAIN_HDR = ("Privileged reasoning (what I noticed and how I interpreted the situation just before reacting; the decision itself is "
             "removed; the reaction must still be grounded in the video and visible in my body movement): ")
# README §8: reasoning as the STUDENT's output
PROMPT_COT = PROMPT.replace("Output exactly three lines and nothing else:\n",
                            "Output exactly four lines and nothing else:\n<reason> my inner reasoning in one or two first-person sentences: "
                            "what I notice and why I react\n")
PROMPT_OBS = PROMPT.replace("Output exactly three lines and nothing else:\n",
                            "Output exactly four lines and nothing else:\n<observe> what I notice in the scene just before reacting, "
                            "first person, no decision\n")
PROMPT_REASON = ("This first-person video ends right before the camera wearer (me) reacts to what is happening.\n"
                 "Write my inner reasoning at that moment in one or two first-person sentences: what I notice and why I react. "
                 "Output only the reasoning.")
OBS_NONE = "nothing specific."


def task_io(row, task):
    """README §8 SFT tasks -> (prompt, target)."""
    if task == "answer":
        return PROMPT, row["target"]
    if task == "cot":
        return PROMPT_COT, "<reason> " + " ".join(row["mental_reasoning"].split()) + "\n" + row["target"]
    if task == "obs":
        return PROMPT_OBS, "<observe> " + " ".join((row.get("mr_obs_all") or OBS_NONE).split()) + "\n" + row["target"]
    if task == "reason":
        return PROMPT_REASON, " ".join(row["mental_reasoning"].split())
    raise ValueError(task)


INFER_PROMPT = {"answer": PROMPT, "cot": PROMPT_COT, "obs": PROMPT_OBS}


def teacher_prompt(row, version=2):
    if ZMODE == "none":  # OPD v6: capability teacher, NO privileged information (same prompt as the student)
        return PROMPT
    if ZMODE == "mr":
        t = row.get("mental_reasoning") if row.get("mr_src") != "generated" else None
        return PROMPT + ("\n\n" + REASON_HDR + t if t else "")
    if ZMODE == "mr_chain":  # README §8: perception + inference chain, decision removed
        t = row.get("mr_chain")
        return PROMPT + ("\n\n" + CHAIN_HDR + t if t else "")
    if ZMODE == "mr_obs":
        t = row.get("mr_obs")
        return PROMPT + ("\n\n" + OBS_HDR + t if t else "")
    """v1 = kinematic hint; v2 = kinematic hint + my recorded inner reasoning ('what/why'; formal-636 items only).
    Training-time teacher only; never shown to the student."""
    t = PROMPT + "\n\n" + hint_text(row)
    if version >= 2 and row.get("mental_reasoning"):
        t += ("\nPrivileged reason (my own inner reasoning at that moment, written down afterwards; it tells what I "
              "noticed and why I reacted; the reaction must still be something visible in my body movement): "
              + row["mental_reasoning"])
    return t


def wjson(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp"); tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1)); os.replace(tmp, p)


def save_json(p, obj):
    if Path(p).exists():
        raise FileExistsError(p)
    wjson(p, obj)


def rjson(p):
    return json.loads(Path(p).read_text())


def sha(p):
    hh = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            hh.update(b)
    return hh.hexdigest()


def h(k, salt=SALT):
    return hashlib.sha256((salt + str(k)).encode()).hexdigest()


_MAN = None


def manifest():
    """Experimental manifest (DATA/data_manifest.json). Only TRAIN and VAL rows are kept in memory ('sealed' only with
    VHO_UNSEAL_SEALED=1); the v860 TEST rows are not part of the released package at all. Relative clip paths are
    resolved against DATA. Mirror items ('<id>_M') reuse the clip of '<id>' and are horizontally flipped at caching time."""
    global _MAN
    if _MAN is None:
        m = rjson(DATA / "data_manifest.json")
        rows = {k: dict(r) for k, r in m["rows"].items() if r["group"] in EVAL_GROUPS}
        for r in rows.values():
            if r.get("clip") and not Path(r["clip"]).is_absolute():
                r["clip"] = str(DATA / r["clip"])
        op = RUNS / "mr_obs.json"  # perceptual part of the human inner reasoning (teacher-only field, README §5)
        if op.exists():
            for k, t in rjson(op)["final"].items():
                if k in rows and t:
                    rows[k]["mr_obs"] = t
        for fn, fld in (("mr_chain.json", "mr_chain"), ("mr_obs_all.json", "mr_obs_all")):  # README §8 (data_build/a2c_mr_chain.py)
            if (RUNS / fn).exists():
                for k, t in rjson(RUNS / fn)["final"].items():
                    if k in rows and t:
                        rows[k][fld] = t
        gp = RUNS / "mr_generated.json"  # optional override of generated inner reasoning (teacher-only field)
        if gp.exists():
            for k, t in rjson(gp)["final"].items():
                if k in rows and not rows[k].get("mental_reasoning"):
                    rows[k]["mental_reasoning"] = t; rows[k]["mr_src"] = "generated"
        tr, va = list(m["train"]), list(m["val"])
        if LIMIT:
            tr, va = tr[:LIMIT], va[:max(4, LIMIT // 4)]
        _MAN = {"train": tr, "val": va, "sealed": list(m.get("sealed", [])) if "sealed" in EVAL_GROUPS else [],
                "rows": rows, "split_sha256": m["split_sha256"], "val_v860": [k for k in va if rows[k].get("in_v860_val")],
                "train_real": [k for k in tr if not rows[k].get("mirror_of")]}
    return _MAN


def resolve(p):
    """artefact path: absolute | relative to cwd if it exists | relative to RUNS (e.g. 'sft_student/epoch1')."""
    p = Path(p)
    return p if p.is_absolute() or p.exists() else RUNS / p


def load_jsonl_shards(stem, d):
    recs = {}
    for f in sorted(Path(d).glob(f"{stem}.shard*.jsonl")):
        for l in f.read_text().splitlines():
            if l.strip():
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                recs.setdefault(r["id"], r)
    return recs


def pick_gpus(n=1, min_free_gb=30, exclude=()):
    """n most idle GPUs: lowest utilisation bucket (25% steps) first, then most free memory, then index."""
    import subprocess
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    g = [(int(i), (int(t) - int(u)) / 1024, int(ut)) for i, u, t, ut in (l.split(", ") for l in q.strip().splitlines())]
    g = [x for x in g if x[0] not in exclude and x[1] >= min_free_gb]
    return [x[0] for x in sorted(g, key=lambda x: (x[2] // 25, -round(x[1]), x[0]))[:n]]


def target_text(reaction, exec_, plan):
    pl = "; ".join(f"{k}={plan[k]}" for k in PLAN_FIELDS)
    return f"<reaction> {reaction}\n<exec> {exec_}\n<plan> {pl}"


def parse(raw):
    """Identical to route-6 r6_common.parse."""
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    tag = lambda t: next((l for l in lines if l.lower().startswith(f"<{t}>")), None)
    r, e, p = tag("reaction"), tag("exec"), tag("plan")
    out = {"reaction": r[10:].strip() if r else None, "exec": None, "plan": {}, "strict": bool(r and e and p), "raw": raw}
    if e:
        out["exec"] = e[6:].strip().strip('"')
    else:
        c = next((l for l in lines if re.match(r"^<?a person>?", l, re.I)), None)
        out["exec"] = re.sub(r"^<a person>", "A person", c, flags=re.I) if c else None
    if out["exec"]:
        out["exec"] = re.sub(r"^a person", "A person", out["exec"], flags=re.I)
        if not out["exec"].startswith("A person"):
            out["exec"] = None
    pl = p or next((l for l in lines if "=" in l), "")
    for k, v in re.findall(r"(\w+)\s*=\s*([\w.\-]+)", pl):
        out["plan"][k.lower()] = v.lower()
    return out


def plan_acc(plan, lab):
    return sum(plan.get(f) == lab[f] for f in PLAN_FIELDS) / len(PLAN_FIELDS)
