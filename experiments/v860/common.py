"""vt_head_opd_v860: Qwen3-VL-4B (V->T SFT) + own T->M head (HumanML3D-pretrained, HERO/MoMask-style RVQ tokens)
+ privileged-information on-policy self-distillation (reverse KL). Route doc:
experiments/route_discussion_20261015_head_opd/ROUTE.md. Data = route-6 frozen v860 split (train 582 / val 96 /
test 182 SEALED, never read). All upstream artifacts (route6, hero_ft_v860, HERO/, LaMP/) are read-only."""
import hashlib, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
UMRM_MODEL = EXP.parent
UMRM = UMRM_MODEL.parent
R6 = EXP / "route6_v860_20261014"                 # read-only
R6RUNS = R6 / "runs" / "v860"                    # read-only (manifest, split, compiler stats, evals)
HVRUNS = EXP / "hero_ft_v860_20261015/runs/v860"  # read-only (HumanML263 of v860, v860-tuned RVQ)
RUNS = HERE / "runs"
QWEN_SRC = UMRM / "models/Qwen3-VL-4B-Instruct"
QWEN_LOCAL = Path("/tmp/vho_models/Qwen3-VL-4B-Instruct")   # byte-identical local copy (CPFS FUSE small reads are slow)
CACHE = Path(os.environ.get("VHO_CACHE", "/tmp/vho_cache"))  # regenerable local caches (video tensors)
HML3D = UMRM / "LaMP/dataset/HumanML3D"                       # read-only
ENV = {"qwen": "/mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python",
       "kimodo": "/mnt/workspace/Wilson/miniconda3/envs/kimodo/bin/python",
       "hero": "/mnt/workspace/Wilson/miniconda3/envs/hero/bin/python"}


def _local_ok(src, dst):
    s, d = Path(src), Path(dst)
    return d.is_dir() and all((d / f.name).exists() and (d / f.name).stat().st_size == f.stat().st_size
                              for f in s.iterdir() if f.is_file())


QWEN = str(QWEN_LOCAL if _local_ok(QWEN_SRC, QWEN_LOCAL) else QWEN_SRC)

NFRAMES, MAX_PIXELS = 16, 360 * 640          # same video budget as route 6
PLAN_FIELDS = ("locomotion", "direction", "turn", "posture", "duration")

# student prompt: IDENTICAL to route 6 (r6_common.PROMPT) so 4B SFT/OPD are directly comparable to the 7B runs
PROMPT = """This first-person video ends right before the camera wearer (me) reacts to what is happening.
Predict what I do next in the following 1-4 seconds. Output exactly three lines and nothing else:
<reaction> one first-person sentence describing my reaction
<exec> one third-person caption for a motion generator, starting with "A person", at most 30 words, body movements only
<plan> locomotion=stay|step|walk|run; direction=none|forward|forward-right|right|back-right|back|back-left|left|forward-left; turn=none|left|right|around; posture=upright|crouch|jump; duration=1.5|2.5|3.5|4.4"""


def hint_text(row):
    """Privileged information z (TRAINING-TIME teacher only): what the recorded PICO motion of my actual reaction did.
    Derived from motion capture (plan_raw + plan labels), never from the reaction text. yaw > 0 = turn left."""
    p, l = row["plan_raw"], row["plan"]
    yaw = p["yaw_deg"]; turn = "no clear turn" if abs(yaw) < 30 else f"a {abs(yaw):.0f}-degree turn to the {'left' if yaw > 0 else 'right'}"
    return (f"Privileged hint (measured from the motion capture of my real reaction; use it only to make the prediction "
            f"physically consistent, and still ground the reaction in what the video shows): the reaction lasts about "
            f"{p['duration_s']:.1f} s; my body travels {p['distance_m']:.2f} m ({p['forward_m']:+.2f} m forward, "
            f"{p['right_m']:+.2f} m to the right) with {turn}; locomotion={l['locomotion']}, direction={l['direction']}, "
            f"posture={l['posture']}.")


def teacher_prompt(row):
    return PROMPT + "\n\n" + hint_text(row)


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
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


_MAN = None


def manifest():
    """route-6 v860 manifest (read-only). TEST rows are removed here so no stage can read them by accident."""
    global _MAN
    if _MAN is None:
        m = rjson(R6RUNS / "data_manifest.json"); inv = rjson(R6RUNS / "inventory.json")["rows"]
        test = set(m["test"])  # m["test_sealed"] is a flag (True)
        rows = {}
        for k, r in m["rows"].items():
            if k in test or r["group"] == "test":
                continue
            rows[k] = {**r, "plan_raw": inv[k]["plan_raw"]}
        _MAN = {"train": list(m["train"]), "val": list(m["val"]), "rows": rows, "split_sha256": m["split_sha256"]}
        assert (len(_MAN["train"]), len(_MAN["val"])) == (582, 96)
    return _MAN


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
    """Indices of the n most idle GPUs: lowest utilisation bucket (25% steps) first, then most free memory, then index."""
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
    """Identical to route-6 r6_common.parse (lenient; 'strict' = exact three-tag format)."""
    import re
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
