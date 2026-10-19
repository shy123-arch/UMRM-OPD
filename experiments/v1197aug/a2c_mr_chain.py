"""A2c (CPU, deterministic; README §8). Two EXTRACTIVE views of the inner reasoning (human + generated, every TRAIN/VAL item;
mirrors carry their own left/right-swapped reasoning, so the rule is applied to each item's own text):
  runs/mr_chain.json   R-chain = perception + inference chain, DECISION removed. Per sentence, cut at the earliest of
                       (a) a decision / intention marker, (b) a consequence connector 'so/therefore/thus/hence/then + I/we',
                       (c) 'I/we + <verb>' whose verb is not a perception / cognition / state verb (an action), unless the
                       clause is subordinate ('if/when/unless/whether/before/after/while/that/once/though/although/because/since/
                       as/until' + I/we ...); decision markers inside an indirect question ('asks what I want') are not cuts.
                       A sentence-initial 'By <verb>ing / To <verb>' clause (means / purpose = the decision) drops the sentence.
                       Trailing connectors and a dangling 'I know / I think (that)' are removed; parts < 3 words dropped.
  runs/mr_obs_all.json R-obs for every item: the human items keep the v5 extract (runs/mr_obs.json final); generated
                       reasoning gets the SAME v5 rule (a2b_mr_obs_extractive.extract).
Only verbatim spans of the source survive (asserted)."""
import json, re
from common import *
from a2b_mr_obs_extractive import extract as extract_obs

DEC = re.compile(r"\b(decide|decided|decides|deciding|want|wants|wanted|need|needs|needed|have to|has to|had to|must|plan|plans|planned|"
                 r"planning|going to|gonna|will|should|try to|tries to|tried to|trying to|intend|intends|intended|hope|hoped|hoping|"
                 r"in order to|so that|let me|let's|my goal|my priority|prepare|prepares|prepared|preparing|ready to|i can|i could|"
                 r"i would|i'd|i'll|we'll|i'm going|about to|time to|choose|chose|opt|opted|better to)\b", re.I)
CONSEQ = re.compile(r"\b(so|therefore|thus|hence|then)\s+(i|we|my|it's time)\b", re.I)
KEEPV = ("see|saw|seen|sees|notice|noticed|notices|noticing|hear|heard|hears|realize|realized|realise|realised|realizes|understand|"
         "understood|understands|know|knew|knows|think|thought|thinks|feel|felt|feels|observe|observed|spot|spotted|recognize|recognized|"
         "recognise|recognised|remember|remembered|am|was|were|are|find|found|sense|sensed|guess|guessed|suspect|suspected|believe|"
         "believed|expect|expected|wonder|wondered|assume|assumed|figure|figured|infer|inferred|judge|judged|can't|cannot|don't|didn't|"
         "do|did|have|had|seem|seemed|may|might|still|also|just|already|now|clearly")
ACT = re.compile(rf"\b(i|we)\s+(?!(?:{KEEPV})\b)[a-z']+", re.I)
HYPO = re.compile(r"\b(if|when|unless|whether|before|after|while|that|once|though|although|because|since|as|until|whenever)\s*$", re.I)
INDQ = re.compile(r"\b(what|whether|if|how|where|why|which|who)\s+(i|we)\s+$", re.I)  # indirect question: 'asks what I want
TRAIL = re.compile(r"(,|;|:|\band\b|\bso\b|\bbut\b|\bthen\b|\bwhich\b|\bthat\b|\bthus\b|\btherefore\b|\bto\b|\s)+$", re.I)
TRAIL_COG = re.compile(r"(\b(i|we)\s+(know|knew|think|thought|feel|felt|realize|realized|understand|understood|see|saw)(\s+that)?[\s,]*)+$", re.I)
PRON = re.compile(r"\b(i|we|you|he|she|they)\s+$", re.I)
LEADCUT = re.compile(r"^(by|to|in order to)\s+[a-z]+", re.I)  # sentence-initial means / purpose clause = the decision ('By moving left ...')


def cut_pos(s):
    if LEADCUT.match(s):
        return 0
    cands = [m.start() for m in DEC.finditer(s) if not INDQ.search(s[:m.start()])][:1] + [m.start() for m in CONSEQ.finditer(s)][:1]
    for m in ACT.finditer(s):
        if not HYPO.search(s[:m.start()]) and not re.search(r"\b(what|whether|how|where|why|which|who)\s*$", s[:m.start()], re.I):
            cands.append(m.start()); break
    if not cands:
        return None
    c = min(cands); p = PRON.search(s[:c])  # 'I should ...' -> cut before the pronoun
    return p.start() if p else c


def extract_chain(mr):
    keep = []
    for s in re.split(r"(?<=[.!?])\s+", mr.strip()):
        s = s.strip()
        if not s:
            continue
        c = cut_pos(s); part = s if c is None else s[:c]
        for _ in range(3):
            part = TRAIL_COG.sub("", TRAIL.sub("", part.strip())).strip()
        if len(part.split()) >= 3:
            keep.append(part if part.endswith((".", "!", "?", '"')) else part + ".")
    return " ".join(keep) if keep else None


def main():
    for f in ("mr_chain.json", "mr_obs_all.json"):
        if (RUNS / f).exists():
            raise FileExistsError(RUNS / f)
    man = manifest(); rows = man["rows"]; keys = [k for k in man["train"] + man["val"] if rows[k].get("mental_reasoning")]
    norm = lambda t: re.sub(r"[^a-z0-9 ]", "", t.lower())
    chain = {k: extract_chain(rows[k]["mental_reasoning"]) for k in keys}
    obs_h = rjson(RUNS / "mr_obs.json")["final"]
    obs = {k: (obs_h.get(k) if rows[k].get("mr_src") != "generated" else extract_obs(rows[k]["mental_reasoning"])) for k in keys}
    for D in (chain, obs):
        for k, t in D.items():
            if t and not (D is obs and rows[k].get("mr_src") != "generated"):
                for piece in re.split(r"(?<=[.!?])\s+", t):
                    assert norm(piece.rstrip('.!?"'))[:40] in norm(rows[k]["mental_reasoning"]), (k, piece)
    STOP = set("i the a an to and my so of it that is in on me for this with as be was am at they their them i'm are there".split())
    cw = lambda s: set(re.findall(r"[a-z]+", (s or "").lower())) - STOP
    ov = lambda a, b: len(cw(a) & cw(b)) / max(1, len(cw(b)))
    def audit(D):
        out = {}
        for src in ("human", "generated"):
            ks = [k for k in keys if (rows[k].get("mr_src") == "generated") == (src == "generated")]; nz = [k for k in ks if D[k]]
            out[src] = {"n": len(ks), "n_nonempty": len(nz),
                        "reaction_overlap_full": round(sum(ov(rows[k]["mental_reasoning"], rows[k]["reaction"]) for k in nz) / max(1, len(nz)), 3),
                        "reaction_overlap_extract": round(sum(ov(D[k], rows[k]["reaction"]) for k in nz) / max(1, len(nz)), 3),
                        "decision_marker_rate_extract": round(sum(bool(DEC.search(D[k])) for k in nz) / max(1, len(nz)), 3),
                        "mean_words_full": round(sum(len(rows[k]["mental_reasoning"].split()) for k in nz) / max(1, len(nz)), 1),
                        "mean_words_extract": round(sum(len(D[k].split()) for k in nz) / max(1, len(nz)), 1)}
        return out
    A = {"chain": audit(chain), "obs": audit(obs)}
    ex = lambda D: [{"id": k, "src": rows[k].get("mr_src") or "human", "reaction": rows[k]["reaction"], "full": rows[k]["mental_reasoning"], "extract": D[k]}
                    for k in keys[:12] + [k for k in keys if rows[k].get("mr_src") == "generated"][:12]]
    save_json(RUNS / "mr_chain.json", {"method": "extractive rule (a2c_mr_chain.py, R-chain)", "final": {k: v for k, v in chain.items() if v}, "audit": A["chain"], "examples": ex(chain)})
    save_json(RUNS / "mr_obs_all.json", {"method": "human: runs/mr_obs.json final (v5); generated: same rule a2b extract()", "final": {k: v for k, v in obs.items() if v},
                                         "audit": A["obs"], "examples": ex(obs)})
    print(json.dumps(A, indent=1))


if __name__ == "__main__":
    main()
