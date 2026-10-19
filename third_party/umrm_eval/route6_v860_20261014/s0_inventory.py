"""S0-a (CPU, read-only on all inputs): inventory of the cohort -> runs/<ver>/inventory.json
Per item: clip, PICO plan labels, human English reaction (+provenance), text to rewrite into gt3p, existing gt3p,
category (+provenance), source identity (union-find over src_id / normalized title / BV id), historical exposure.
Supplement (224): English reaction = uploader's original from the review-desk DB (reaction_cn there is an LLM translation
of it); 8 legacy auto-pipeline items -> their annotation.json. Nothing is written outside runs/."""
import csv, collections, re, sqlite3
import numpy as np
from r6_common import *

DESK_DB = UMRM / "benchmark/judge_exploration/annotation_review_20260914/review.sqlite3"
EXPORT = OUT / "data_umrm925_20260927/umrm925_approved_20260927_132408/metadata"


# conservative series-level merges (same creator series / near-identical titles found in the near-duplicate audit)
SERIES = [("VLOG｜第一视角记录小县城宠物店的日常", "VLOG｜第一视角体验小县城宠物店的一天"),
          ("在学校挑战找陌生同学蹭饭", "挑战找陌生同学蹭饭并跟她玩一天")]


def norm_title(t):
    t = (t or "").strip()
    if not t:
        return ""
    t = re.sub(r"\.(f\d+)?(\.mp4)?$", "", re.sub(r"\.mp4$", "", t, flags=re.I))
    t = re.sub(r"(\.f\d+)?_merged$", "", t); t = re.sub(r"^\d+_(?=BV)", "", t)
    t = re.sub(r"^(LSF|WZH|CK|ZJZ)_\d+_", "", t, flags=re.I); t = re.sub(r"^BV[0-9A-Za-z]{10}_", "", t)
    t = re.sub(r"[^0-9a-z一-鿿]+", "", t.lower())
    for grp in SERIES:
        if t in {re.sub(r"[^0-9a-z一-鿿]+", "", g.lower()) for g in grp}:
            return "series:" + re.sub(r"[^0-9a-z一-鿿]+", "", grp[0].lower())
    return t


class UF:
    def __init__(self):
        self.p = {}

    def f(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x

    def u(self, a, b):
        self.p[self.f(a)] = self.f(b)


def main():
    import plan as PL
    D = OUT / "data_v860_20260927"
    items = {json.loads(l)["qid"]: json.loads(l) for l in (D / "items.jsonl").read_text().splitlines()}
    clips = {c["id"]: c["clip_path"] for c in rjson(D / "clips.json")["items"]}
    hist = rjson(OUT / "data_v860h_20260927/split.json")
    prompts = rjson(FT / "prompts.json")
    z = np.load(D / "data.npz"); pos = {k: i for i, k in enumerate(z["ids"])}
    src_index = {r["id"]: r for r in csv.DictReader(open(EXPORT / "source_index.csv", encoding="utf-8-sig"))}
    con = sqlite3.connect(f"file:{DESK_DB}?mode=ro", uri=True)
    desk = {i: json.loads(p) for i, p in con.execute("select id, payload from items")}
    rows, uf, notes = {}, UF(), collections.Counter()
    for k, it in items.items():
        assert k in pos and k in clips and Path(clips[k]).exists(), k
        r = {"id": k, "clip": clips[k], "hist_group": hist[k], "annotator": it.get("annotator", "")}
        p = PL.extract(z["targets"][pos[k]], z["masks"][pos[k]]); r["plan"] = plan_labels(p)
        r["plan_raw"] = {x: float(p[x]) for x in ("duration_s", "distance_m", "yaw_deg", "forward_m", "right_m")}
        keys = []
        if it["scene"] != "N":  # formal 636
            r["cohort"] = "formal636"
            r["reaction"] = it["reaction_en"].strip(); r["reaction_src"] = "formal.reaction_en"
            r["rewrite_src_text"] = (it.get("motion_prompt_en") or "").strip() or r["reaction"]
            r["gt3p_existing"] = prompts["gt3p"].get(k)
            r["category"], r["category_src"] = it["scene"], "formal.scene"
            if it.get("src_id"):
                keys.append("id:" + it["src_id"])
            t = it.get("src_title", "")
        else:  # approved supplement 224
            r["cohort"] = "supplement224"
            si = src_index[k]; r["supp_source"] = si["source"]
            r["gt3p_existing"] = prompts["gt3p"].get(k)  # Qwen3.6 rewrite of reaction_cn (kept for reference only)
            if si["source_id"] in desk:
                d = desk[si["source_id"]]
                r["reaction"] = (d.get("reaction_original") or "").strip(); r["reaction_src"] = "desk.reaction_original"
                if d.get("reaction_cn", "").strip() != it["reaction_cn"].strip():
                    notes["supp_reaction_cn_differs_from_desk"] += 1
                t = d.get("title", "")
                r["category"], r["category_src"] = None, "pending_llm"
            else:
                a = rjson(si["source_artifact"])
                r["reaction"] = (a.get("reaction_en") or "").strip(); r["reaction_src"] = "auto_annotation.reaction_en(legacy)"
                if a.get("reaction_cn", "").strip() != it["reaction_cn"].strip():
                    notes["legacy_reaction_cn_differs"] += 1
                t = a.get("src_title", "")
                if a.get("src_id"):
                    keys.append("id:" + a["src_id"])
                r["category"], r["category_src"] = a.get("scene"), "auto_annotation.scene(unverified)"
            r["rewrite_src_text"] = r["reaction"]
            r["reaction_cn"] = it["reaction_cn"]
            # v860 finding: 33 desk5124 'reaction_original' fields are Chinese. From the next cohort on (R6_DATA != v860)
            # such items are flagged and translated to English by s0_llm.py before the gt3p rewrite.
            if DATA_VERSION != "v860" and re.search(r"[一-鿿]", r["reaction"]):
                r["needs_translation"] = True; r["reaction_src"] += "(chinese->translate)"
            if not r["reaction"]:
                notes["supp_missing_english"] += 1
        r["src_title"] = t; nt = norm_title(t)
        if nt:
            keys.append("t:" + nt)
        bv = re.findall(r"BV[0-9A-Za-z]{10}", (t or "") + " " + it.get("src_id", ""))
        keys += ["id:" + b for b in bv]
        if not keys:
            keys = ["single:" + k]; notes["no_source_identity"] += 1
        r["_keys"] = sorted(set(keys))
        for x in r["_keys"]:
            uf.u(x, r["_keys"][0])
        rows[k] = r
    # canonical source id = salted hash of the union-find root's member set
    comp = collections.defaultdict(set)
    for r in rows.values():
        comp[uf.f(r["_keys"][0])].update(r["_keys"])
    for r in rows.values():
        members = sorted(comp[uf.f(r["_keys"][0])]); r["source"] = "S" + h("|".join(members))[:10]; r["source_keys"] = members
        del r["_keys"]
    by = collections.defaultdict(list)
    for r in rows.values():
        by[r["source"]].append(r["id"])
    cross = {s: sorted({rows[k]["cohort"] for k in ks}) for s, ks in by.items()}
    stats = {"n": len(rows), "notes": dict(notes), "n_sources": len(by),
             "sources_spanning_both_cohorts": sum(len(v) > 1 for v in cross.values()),
             "items_in_cross_cohort_sources": sum(len(by[s]) for s, v in cross.items() if len(v) > 1),
             "hist_groups_per_source_mixed": sum(len({rows[k]["hist_group"] for k in ks}) > 1 for ks in by.values()),
             "source_size_top": sorted([len(v) for v in by.values()], reverse=True)[:12],
             "cohort": dict(collections.Counter(r["cohort"] for r in rows.values())),
             "category": dict(collections.Counter(str(r["category"]) for r in rows.values())),
             "reaction_src": dict(collections.Counter(r["reaction_src"] for r in rows.values())),
             "missing_reaction": sum(not r["reaction"] for r in rows.values())}
    save_json(RUNS / "inventory.json", {"data_version": DATA_VERSION, "stats": stats, "rows": rows,
                                         "inputs": {"items": str(D / "items.jsonl"), "items_sha256": sha(D / "items.jsonl"),
                                                    "desk_db": str(DESK_DB), "export_source_index": str(EXPORT / "source_index.csv")}})
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
