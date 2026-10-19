"""README §9 G-E0 probe (1 GPU): multiple-choice accuracy by option log-likelihood on the ego probe items (EgoNormia verified 200,
EgoToM actions/belief/goal). For each item all options are scored in one batch: mean log-probability of the option tokens
(after the shared answer prefix, e.g. '<next>'), argmax = prediction. Same prompts for every model. Output runs/probe_<name>.json.
Run with VHO_CACHE=/tmp/vho_ego_cache. Usage: ego_probe.py --name base | --adapter runs/sft_student/epoch1 --name sft"""
import argparse, json, time
import torch
from common import *
from vt_io import load_model, tokenizer
from opd2 import batch_with_responses, resp_logits


@torch.no_grad()
def score(model, it, dev):
    tok = tokenizer(); pre = it["prefix"].strip()
    ids_pre = tok(pre, add_special_tokens=False)["input_ids"] if pre else []
    resp, spans = [], []
    for o in it["options"]:
        full = tok(pre + (" " if pre else "") + o.strip(), add_special_tokens=False)["input_ids"]
        assert full[:len(ids_pre)] == ids_pre, (pre, o)
        resp.append(full); spans.append((len(ids_pre), len(full)))
    inp, R = batch_with_responses(it["id"], it["prompt"], resp)
    lp = torch.log_softmax(resp_logits(model, inp, R, dev), -1)
    out = []
    for i, (r, (a, b)) in enumerate(zip(resp, spans)):
        t = torch.tensor(r[a:b], device=lp.device); out.append(float(lp[i, a:b].gather(-1, t[:, None]).mean()))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", default=None); ap.add_argument("--name", required=True); a = ap.parse_args()
    dev = "cuda:0"; out = RUNS / f"probe_{a.name}.json"
    m = rjson(RUNS / "ego_manifest.json"); its = [p for p in m["probe"] if (CACHE / "video" / f"{p['id']}.pt").exists()]
    ad = (HERE / a.adapter) if a.adapter and not Path(a.adapter).is_absolute() else a.adapter
    model = load_model(adapter=ad, merge=True, device=dev); t0 = time.time(); res = {}
    for j, it in enumerate(its):
        s = score(model, it, dev); res[it["id"]] = {"set": it["set"], "video_uid": it["video_uid"], "scores": s, "gold": it["gold"],
                                                    "pred": int(max(range(len(s)), key=lambda i: s[i]))}
        if (j + 1) % 100 == 0:
            print(j + 1, "/", len(its), round(time.time() - t0), "s", flush=True)
    acc = {}
    for st in sorted({v["set"] for v in res.values()}):
        R = [v for v in res.values() if v["set"] == st]; acc[st] = {"n": len(R), "acc": round(sum(v["pred"] == v["gold"] for v in R) / len(R), 4),
                                                                     "chance": round(sum(1 / len(v["scores"]) for v in R) / len(R), 4)}
    wjson(out, {"name": a.name, "adapter": str(ad) if ad else None, "n_missing_cache": len(m["probe"]) - len(its), "acc": acc, "items": res,
                "elapsed_s": round(time.time() - t0)})
    print(json.dumps(acc, indent=1))


if __name__ == "__main__":
    main()
