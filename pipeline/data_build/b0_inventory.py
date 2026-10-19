"""B0 (CPU, read-only on all inputs): v1197 inventory + inherited split (rules fixed before any model output exists).
Inputs: outputs/data_v1197_20261015 (FROZEN), route6 v860 manifest/inventory/split, 1014 batch source_index, review-desk DB.
Per item: clip, PICO plan labels (route-6 extractor on the frozen 88-frame targets), English reaction (+provenance; Chinese
-> flagged for Qwen3-8B translation), existing gt3p, v1197 category, policy flag, mental_reasoning (formal 636), source id.
Split (inherits route6 v860 = route6 s0_split R6_INHERIT semantics):
  * every v860 item keeps its v860 group; v860 TEST stays sealed (no text of TEST items is copied here);
  * union-find source identity over route-6 source keys + new items' normalised source title / BV id;
  * new items in a source whose old items are all in group g -> g (TEST -> 'test_extra', never trained on);
    new links merging several old groups -> new items 'excluded_conflict';
  * brand-new sources (salted-hash order): first a 'sealed' block of SEALED_FRAC of new-source items (sources <= 20 items),
    then VAL by category quota VAL_FRAC x n(category) counting inherited VAL items, rest TRAIN;
  * staged-film items (policy_flag, 24, all from the v860 supplement): decision 2026-10-15 = keep where v860 put them for
    training (9 TRAIN); exclude from every evaluation set (the 15 in sealed TEST get eval_exclude='staged_film').
Output: runs/inventory.json"""
import csv, collections, re, sqlite3, sys
import numpy as np
from common import *
sys.path.insert(0, str(R6))
import r6_common as R6C  # noqa: E402  (adds harness_v1 to sys.path)
from s0_inventory import norm_title, UF  # noqa: E402  (route6, read-only)
import plan as PL  # noqa: E402

CJK = re.compile(r"[一-鿿]")
SEALED_FRAC, VAL_FRAC, MAX_EVAL_SRC = 0.15, 0.10, 20
DESK_DB = UMRM / "benchmark/judge_exploration/annotation_review_20260914/review.sqlite3"
B1014 = UMRM_MODEL / "outputs/data_umrm1014_20261015/umrm1014_approved_20261015_204248/metadata"


def main():
    out = RUNS / "inventory.json"
    if out.exists():
        raise FileExistsError(out)
    m6 = rjson(R6RUNS / "data_manifest.json"); inv6 = rjson(R6RUNS / "inventory.json")["rows"]; prev = rjson(R6RUNS / "split.json")["item_group"]
    items = {json.loads(l)["qid"]: json.loads(l) for l in (DATA / "items.jsonl").read_text().splitlines()}
    cats = {r["id"]: r for r in csv.DictReader(open(DATA / "categories.csv", encoding="utf-8-sig"))}
    clips = {c["id"]: c["clip_path"] for c in rjson(DATA / "clips.json")["items"]}
    z = np.load(DATA / "data.npz"); ids = [str(x) for x in z["ids"]]; pos = {k: i for i, k in enumerate(ids)}
    si = {r["id"]: r for r in csv.DictReader(open(B1014 / "source_index.csv", encoding="utf-8-sig"))}
    con = sqlite3.connect(f"file:{DESK_DB}?mode=ro", uri=True); desk = {i: json.loads(p) for i, p in con.execute("select id, payload from items")}
    rows, uf, notes = {}, UF(), collections.Counter()
    for k in ids:
        it, c = items[k], cats[k]; p = PL.extract(z["targets"][pos[k]], z["masks"][pos[k]])
        r = {"id": k, "cohort": c["cohort"], "category": c["category"], "category_src": c["category_src"], "category_confidence": c["confidence"],
             "staged_film": bool(c["policy_flag"]), "clip": clips[k], "plan": R6C.plan_labels(p),
             "plan_raw": {x: float(p[x]) for x in ("duration_s", "distance_m", "yaw_deg", "forward_m", "right_m")}}
        if k in prev:  # v860 item
            assert m6["rows"][k]["plan"] == r["plan"], k
            r["prev_group"] = prev[k]; r["in_v860_val"] = prev[k] == "val"; keys = list(inv6[k]["source_keys"])
            if prev[k] != "test":
                r6 = m6["rows"][k]; r["mental_reasoning"] = (it.get("mental_reasoning_en") or "").strip() if c["cohort"] == "formal636" else ""
                r["src_title"] = inv6[k].get("src_title", "")
                if CJK.search(r6["reaction"]):  # v860 known issue: desk5124 'English original' was Chinese -> translate + re-rewrite
                    r.update(reaction=None, to_translate=r6["reaction"], reaction_src=r6["reaction_src"] + "->qwen3_8b_translation", exec=None)
                    notes["v860_chinese_reaction_retranslated"] += 1
                else:
                    r.update(reaction=r6["reaction"], reaction_src=r6["reaction_src"], exec=r6["exec"], exec_src=r6["exec_src"])
        else:  # new mocap1014_337
            s = si[k]; d = desk[s["source_id"]]; orig = (d.get("reaction_original") or "").strip()
            r.update(prev_group=None, in_v860_val=False, mental_reasoning="", src_title=s["title"], annotator=s["annotator"],
                     reaction_cn=items[k]["reaction_cn"], exec=None)
            if not orig or CJK.search(orig):
                r.update(reaction=None, to_translate=orig or items[k]["reaction_cn"], reaction_src="desk.reaction_original(chinese)->qwen3_8b_translation")
                notes["new_translate"] += 1
            else:
                r.update(reaction=orig, reaction_src="desk.reaction_original(english)"); notes["new_english_original"] += 1
            nt = norm_title(s["title"]); keys = (["t:" + nt] if nt else []) + ["id:" + b for b in re.findall(r"BV[0-9A-Za-z]{10}", s["title"])]
            keys = keys or ["single:" + k]
        r["_keys"] = sorted(set(keys))
        for x in r["_keys"]:
            uf.u(x, r["_keys"][0])
        rows[k] = r
    comp = collections.defaultdict(list)
    for r in rows.values():
        comp[uf.f(r["_keys"][0])].append(r["id"])
    old_src = {k: inv6[k]["source"] for k in prev}
    group, src_of, rep = {}, {}, collections.Counter()
    new_components = []
    for root, ks in comp.items():
        olds = [k for k in ks if k in prev]
        if not olds:
            sid = "N" + h("|".join(sorted({x for k in ks for x in rows[k]["_keys"]})))[:10]; new_components.append((sid, ks))
            for k in ks:
                src_of[k] = sid
            continue
        g_old = {prev[k] for k in olds}
        for k in olds:
            group[k] = prev[k]; src_of[k] = old_src[k]
        news = [k for k in ks if k not in prev]
        if len(g_old) > 1:
            for k in news:
                group[k] = "excluded_conflict"; src_of[k] = "conflict:" + root; rep["new_items_excluded_conflict"] += 1
        else:
            g = g_old.pop(); s_old = collections.Counter(old_src[k] for k in olds).most_common(1)[0][0]
            for k in news:
                group[k] = "test_extra" if g == "test" else g; src_of[k] = s_old; rep[f"new_items_into_{group[k]}"] += 1
    # brand-new sources
    new_components.sort(key=lambda x: h(x[0]))
    n_new = sum(len(ks) for _, ks in new_components); quota, got = SEALED_FRAC * n_new, 0
    for sid, ks in new_components:
        if got >= quota:
            break
        if len(ks) <= MAX_EVAL_SRC:
            for k in ks:
                group[k] = "sealed"
            got += len(ks)
    n_cat = collections.Counter(r["category"] for r in rows.values())
    scat = {sid: collections.Counter(rows[k]["category"] for k in ks).most_common(1)[0][0] for sid, ks in new_components}
    for c in sorted(n_cat):
        q = VAL_FRAC * n_cat[c]; g_ = sum(1 for k, r in rows.items() if group.get(k) == "val" and r["category"] == c)
        for sid, ks in new_components:
            if scat[sid] != c or any(k in group for k in ks) or len(ks) > MAX_EVAL_SRC:
                continue
            if g_ >= q:
                break
            if g_ + len(ks) > 1.25 * q + 2:
                continue
            for k in ks:
                group[k] = "val"
            g_ += len(ks)
    for sid, ks in new_components:
        for k in ks:
            group.setdefault(k, "train")
    for k, r in rows.items():
        r["group"] = group[k]; r["source"] = src_of[k]; del r["_keys"]
        if r["staged_film"]:
            assert r["group"] in ("train", "test"), k
            if r["group"] == "test":
                r["eval_exclude"] = "staged_film"
        if r["group"] in ("test", "test_extra", "excluded_conflict"):  # never carry text of items nobody may train/evaluate on now
            for f in ("reaction", "to_translate", "exec", "mental_reasoning", "reaction_cn"):
                r.pop(f, None)
    S = lambda *gs: {r["source"] for r in rows.values() if r["group"] in gs}
    for g in ("val", "test", "test_extra", "sealed"):
        assert not (S(g) & S("train")), g
    assert not (S("val") & S("test", "test_extra", "sealed")) and not (S("sealed") & S("test", "test_extra"))
    stats = {"n": len(rows), "notes": dict(notes), "inherit": dict(rep), "n_new_source_items": n_new, "sealed_quota": quota,
             "group": dict(collections.Counter(r["group"] for r in rows.values())),
             "group_by_cohort": {g: dict(collections.Counter(r["cohort"] for r in rows.values() if r["group"] == g)) for g in set(group.values())},
             "group_by_category": {g: dict(sorted(collections.Counter(r["category"] for r in rows.values() if r["group"] == g).items())) for g in set(group.values())},
             "n_sources": {g: len({r["source"] for r in rows.values() if r["group"] == g}) for g in set(group.values())},
             "staged_film": dict(collections.Counter(r["group"] for r in rows.values() if r["staged_film"])),
             "mental_reasoning_available": {g: sum(bool(r.get("mental_reasoning")) for r in rows.values() if r["group"] == g) for g in ("train", "val")},
             "params": {"SEALED_FRAC": SEALED_FRAC, "VAL_FRAC": VAL_FRAC, "MAX_EVAL_SRC": MAX_EVAL_SRC, "salt": SALT}}
    save_json(out, {"data": str(DATA), "frozen_sha": sha(DATA / "FROZEN.json"), "inherit_from": str(R6RUNS / "split.json"), "rules": __doc__,
                    "stats": stats, "rows": rows})
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
