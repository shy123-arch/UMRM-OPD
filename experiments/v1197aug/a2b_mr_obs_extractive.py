"""A2b (CPU, deterministic): EXTRACTIVE perceptual part of the human inner reasoning. Replaces the Qwen3-8B rewrite
(runs/mr_obs_llm_REJECTED.json: it added unseen details, mean 79 words vs 34 of the source).
Rule: split into sentences; in each sentence cut at the first intention / decision / plan marker and keep the part before it
(trailing connectors removed); a leading 'Since/Because/As X, I decide...' keeps clause X. Only verbatim spans of the source
survive (asserted), so nothing can be added. Mirrors: left/right-swapped extract of the source (dropped if ambiguous)."""
import json, re
from common import *

MARK = re.compile(r"\b(i\s+)?(realize|realise|realized|decide|decided|decides|want|wants|wanted|need|needs|needed|have to|has to|had to|must|"
                  r"plan|planning|going to|will|should|try to|tries to|trying to|intend|hope|in order to|so that|let me|my goal|my priority|"
                  r"prepare|preparing|ready to|i can|i could|i would|i'd|i'll|i'm going)\b|\bi know\b|\bi think\b|\bi feel\b|\bi understand\b|\bi'm\b", re.I)
LEAD = re.compile(r"^(since|because|as|seeing that|now that)\s+", re.I)
TRAIL = re.compile(r"(,|\band\b|\bso\b|\bbut\b|\bthen\b|\bwhich\b|\bthat\b|\bthus\b|\btherefore\b|\bto\b|\s)+$", re.I)


def extract(mr):
    keep = []
    for s in re.split(r"(?<=[.!?])\s+", mr.strip()):
        s = s.strip()
        if not s:
            continue
        m0 = LEAD.match(s)
        if m0 and "," in s:  # 'Since X, I decide ...' -> X
            clause = s[m0.end():s.index(",")]
            if not MARK.search(clause) and len(clause.split()) >= 3:
                keep.append(clause[0].upper() + clause[1:] + "."); continue
        m = MARK.search(s)
        part = s if m is None else s[:m.start()]
        part = TRAIL.sub("", part.strip()).strip()
        if len(part.split()) >= 3:
            keep.append(part if part.endswith((".", "!", "?")) else part + ".")
    return " ".join(keep) if keep else None


def main():
    out = RUNS / "mr_obs.json"
    if out.exists():
        raise FileExistsError(out)
    src = (HERE / "a0_mirror.py").read_text(); ns = {}
    exec("import re\n" + src[src.index("LAT = {"):src.index("def mirror_joints")], ns); mirror_text = ns["mirror_text"]
    man = manifest(); rows = man["rows"]
    keys = [k for k in man["train"] + man["val"] if not rows[k].get("mirror_of") and rows[k].get("mental_reasoning") and rows[k].get("mr_src") != "generated"]
    res = {k: extract(rows[k]["mental_reasoning"]) for k in keys}
    norm = lambda t: re.sub(r"[^a-z0-9 ]", "", t.lower())
    for k, t in res.items():  # verbatim guarantee: every kept word sequence comes from the source
        if t:
            for piece in re.split(r"(?<=[.!?])\s+", t):
                assert norm(piece.rstrip(".!?"))[:40] in norm(rows[k]["mental_reasoning"]), (k, piece)
    final = {k: v for k, v in res.items() if v}; dropped = []
    for k in man["train"]:
        b = rows[k].get("mirror_of")
        if b and final.get(b):
            mt, amb = mirror_text(final[b])
            (dropped.append(k) if amb else final.__setitem__(k, mt))
    STOP = set("i the a an to and my so of it that is in on me for this with as be was am at they their them i'm are there".split())
    cw = lambda s: set(re.findall(r"[a-z]+", (s or "").lower())) - STOP
    ov = lambda a, b: len(cw(a) & cw(b)) / max(1, len(cw(b)))
    real = [k for k in keys if res[k]]
    audit = {"n_real": len(keys), "n_none": len(keys) - len(real), "n_none_by_group": {g: sum(1 for k in keys if not res[k] and rows[k]["group"] == g) for g in ("train", "val")},
             "n_mirrored": len(final) - len(real), "mirror_dropped_ambiguous": len(dropped),
             "reaction_overlap_full_mr": sum(ov(rows[k]["mental_reasoning"], rows[k]["reaction"]) for k in real) / len(real),
             "reaction_overlap_obs": sum(ov(final[k], rows[k]["reaction"]) for k in real) / len(real),
             "intent_marker_rate_full_mr": sum(bool(MARK.search(rows[k]["mental_reasoning"])) for k in real) / len(real),
             "intent_marker_rate_obs": sum(bool(MARK.search(final[k])) for k in real) / len(real),
             "mean_words_full_mr": sum(len(rows[k]["mental_reasoning"].split()) for k in real) / len(real),
             "mean_words_obs": sum(len(final[k].split()) for k in real) / len(real)}
    save_json(out, {"method": "extractive rule (a2b_mr_obs_extractive.py)", "final": final, "raw": res, "audit": audit,
                    "examples": [{"reaction": rows[k]["reaction"], "full": rows[k]["mental_reasoning"], "obs": res[k]} for k in keys[:15]]})
    print(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
