import os
"""Route-6 mainline, first version on the v860 cohort (2026-10-14).
Report: umrm_model/reports/generation_routes_report_20261014/UMRM_动作生成路线调研与建议_20261014.md (route 6, §5.2-5.4).
Builds on the v1 run (experiments/react_sft_dpo_20261013, read-only): same model I/O, Kimodo engine, independent metrics.
New here: frozen source-isolated / category-stratified split, human-English targets for the 224 supplement, revised reward
(within-group standardized, sem/dec/plan dominated, capped s_hum), free + fixed-reaction DPO pairs, MBR equal-budget control.
Everything is parametrized by DATA_VERSION so the same scripts re-run on the ~1.5k cohort."""
import hashlib, json, os, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
UMRM_MODEL = EXP.parent
UMRM = UMRM_MODEL.parent
OUT = Path(os.environ.get("UMRM_OUTPUTS", "/mnt/workspace/Wilson/ck/umrm/umrm_model/outputs"))
DATA_VERSION = os.environ.get("R6_DATA", "v860")
RUNS = HERE / "runs" / DATA_VERSION
V1 = EXP / "react_sft_dpo_20261013"          # read-only predecessor
PILOT = EXP / "design_pilot_20261013"         # read-only (Kimodo/TMR engine)
FT = Path(os.environ.get("VHO_EVAL_FT", str(OUT / "kimodo_ft_v1_v860_20260927")))       # read-only (gt3p, LLM2Vec cache, SOMA fit)
QWEN_VL_SRC = "/mnt/workspace/models/Qwen/Qwen2___5-VL-7B-Instruct"
QWEN_VL_LOCAL = "/tmp/r6_models/Qwen2.5-VL-7B-Instruct"  # byte-identical local copy (CPFS FUSE reads are slow)


def _local_ok(src, dst):
    s, d = Path(src), Path(dst)
    if not d.is_dir():
        return False
    return all((d / f.name).exists() and (d / f.name).stat().st_size == f.stat().st_size for f in s.iterdir() if f.is_file())


QWEN_VL = QWEN_VL_LOCAL if _local_ok(QWEN_VL_SRC, QWEN_VL_LOCAL) else QWEN_VL_SRC
QWEN3 = "/mnt/workspace/models/Qwen3-8B"
SALT = "umrm-route6-v860-20261014"
sys.path.insert(0, str(EXP / "harness_v1"))

NFRAMES, MAX_PIXELS = 16, 360 * 640
DIRS = ["forward", "forward-right", "right", "back-right", "back", "back-left", "left", "forward-left"]
DIR_ANG = {d: 45 * i for i, d in enumerate(DIRS)}
DURS = [1.5, 2.5, 3.5, 4.4]
PLAN_FIELDS = ("locomotion", "direction", "turn", "posture", "duration")
CATS = {"A": "户外出行 outdoor travel / commuting", "B": "服务工作 service work / occupational tasks",
        "C": "社交人际 social / interpersonal interaction", "D": "运动竞技 sports / competition",
        "E": "突发失误 sudden incidents / accidents / mistakes"}

# identical to v1 (keeps SFT/DPO comparable to the v1 run)
PROMPT = """This first-person video ends right before the camera wearer (me) reacts to what is happening.
Predict what I do next in the following 1-4 seconds. Output exactly three lines and nothing else:
<reaction> one first-person sentence describing my reaction
<exec> one third-person caption for a motion generator, starting with "A person", at most 30 words, body movements only
<plan> locomotion=stay|step|walk|run; direction=none|forward|forward-right|right|back-right|back|back-left|left|forward-left; turn=none|left|right|around; posture=upright|crouch|jump; duration=1.5|2.5|3.5|4.4"""


def h(k, salt=SALT):
    return hashlib.sha256((salt + str(k)).encode()).hexdigest()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wjson(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp"); tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1)); os.replace(tmp, p)


def save_json(p, obj):
    if Path(p).exists():
        raise FileExistsError(p)
    wjson(p, obj)


def rjson(p):
    return json.loads(Path(p).read_text())


def load_jsonl_shards(stem, d=None):
    recs = {}
    for f in sorted((d or RUNS).glob(f"{stem}.shard*.jsonl")):
        for l in f.read_text().splitlines():
            if l.strip():
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:  # partially written last line of a file still being appended to
                    continue
                recs.setdefault(r["id"], r)
    return recs


def manifest():
    return rjson(RUNS / "data_manifest.json")


def plan_labels(p):
    d = min(DURS, key=lambda x: abs(x - p["duration_s"]))
    return {"locomotion": p["locomotion"], "direction": p["direction"].replace("_", "-"), "turn": p["turn"],
            "posture": p["posture"], "duration": f"{d}"}


def target_text(reaction, exec_, plan):
    pl = "; ".join(f"{k}={plan[k]}" for k in PLAN_FIELDS)
    return f"<reaction> {reaction}\n<exec> {exec_}\n<plan> {pl}"


def parse(raw):
    """Lenient parse (same as v1); 'strict' = exact three-tag format."""
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
    return float(np.mean([plan.get(f) == lab[f] for f in PLAN_FIELDS]))


class Compiler:
    """Rule compiler (identical to v1): plan -> (n_frames @30 fps, Root2D path, terminal heading). Train-only stats."""
    def __init__(self, stats):
        self.s = stats

    def nframes(self, plan):
        try:
            d = float(plan.get("duration", self.s["dur_median"]))
        except ValueError:
            d = self.s["dur_median"]
        return int(np.clip(round(d * 30), 45, 131))

    def constraints(self, plan, n, theta0=None, heading=True):
        import plan as PL
        loco = plan.get("locomotion", "stay"); dr = plan.get("direction", "none")
        if loco not in self.s["dist"]:
            return []
        dist = 0.0 if (loco == "stay" or dr not in DIR_ANG) else self.s["dist"][loco]
        a = np.radians(DIR_ANG.get(dr, 0))
        p = {"right_m": dist * np.sin(a), "forward_m": dist * np.cos(a),
             "onset_s": self.s["onset_frac"] * n / 30, "settle_s": self.s["settle_frac"] * n / 30}
        fr = np.arange(0, n, 6); path = PL.coarse_root_path(p, n, 30)
        xz = np.stack([-path[:, 0], -path[:, 1]], -1)
        out = [(fr, xz[fr], None)]
        t = plan.get("turn", "none")
        if heading and theta0 is not None and t in self.s["yaw"]:
            yaw = np.radians(self.s["yaw"][t]) * (1 if t in ("left", "around") else -1)
            ends = np.array([0, n - 1])
            hd = np.stack([[np.cos(theta0), np.sin(theta0)], [np.cos(theta0 + yaw), np.sin(theta0 + yaw)]])
            out.append((ends, xz[ends], hd))
        return out
