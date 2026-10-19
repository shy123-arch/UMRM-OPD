#!/bin/bash
# README §7 addendum: stop sft_f32 once epoch 2 is saved and its VAL NLL is logged; select among completed epochs (1, 2).
cd "$(dirname "$0")"; PID=$1
until grep -q '"epoch": 2, "val_nll_per_tok"' runs/sft_f32/log.jsonl 2>/dev/null; do kill -0 $PID 2>/dev/null || exit 1; sleep 60; done
/usr/bin/python3 - <<'P'
import json
L = [json.loads(l) for l in open("runs/sft_f32/log.jsonl")]
ev = {r["epoch"]: r["val_nll_per_tok"] for r in L if "val_nll_per_tok" in r and r["epoch"] > 0}
assert set(ev) == {1, 2}, ev
json.dump({"rule": "lowest val NLL per token AMONG COMPLETED EPOCHS 1-2 (pre-registered addendum README §7: stopped after epoch 2, "
           "same completed epochs as the 16-frame reference sft_student)", "val_nll": ev, "selected": f"epoch{min(ev, key=ev.get)}",
           "epochs_completed": [1, 2]}, open("runs/sft_f32/selection.json", "w"), indent=1)
P
kill $PID; echo "$(date) stopped sft_f32 after epoch 2: $(cat runs/sft_f32/selection.json | tr -d '\n ')"
