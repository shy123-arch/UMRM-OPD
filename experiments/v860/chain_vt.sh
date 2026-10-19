#!/bin/bash
# V->T chain: wait for both SFTs -> pick the most idle GPU -> VAL greedy (student; teacher-with-z diagnostic) -> G-T gate
# -> OPD (only if G-T passes) -> Kimodo eval of SFT greedy in parallel. Logs: logs/chain_vt.log
cd "$(dirname "$0")"; Q=/mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python; K=/mnt/workspace/Wilson/miniconda3/envs/kimodo/bin/python
pick() { python3 -c "import common as C; g=C.pick_gpus(1, $1); print(g[0] if g else -1)"; }
waitgpu() { while true; do g=$(pick $1); [ "$g" != "-1" ] && { echo $g; return; }; sleep 60; done; }
while [ ! -f runs/sft_student/selection.json ] || [ ! -f runs/sft_teacher/selection.json ]; do sleep 60; done
echo "$(date) SFT done: student $(cat runs/sft_student/selection.json | tr -d '\n ') teacher $(cat runs/sft_teacher/selection.json | tr -d '\n ')"
G=$(waitgpu 45); echo "$(date) V->T GPU = $G"; echo "vt_after_sft=$G $(date)" >> runs/gpus_picked.txt
SS=runs/sft_student/$(python3 -c "import json;print(json.load(open('runs/sft_student/selection.json'))['selected'])")
ST=runs/sft_teacher/$(python3 -c "import json;print(json.load(open('runs/sft_teacher/selection.json'))['selected'])")
CUDA_VISIBLE_DEVICES=$G $Q -u infer.py --adapter $SS --name sft4b_student_greedy > logs/infer_sft4b_student_greedy.log 2>&1
CUDA_VISIBLE_DEVICES=$G $Q -u infer.py --adapter $ST --prompt teacher --name sft4b_teacher_zgreedy > logs/infer_sft4b_teacher_zgreedy.log 2>&1
echo "$(date) greedy VAL done"
(CUDA_VISIBLE_DEVICES=$G $K -u eval_kimodo.py --name sft4b_student_greedy --stage encode > logs/evalk_sft4b_student_greedy.log 2>&1 && \
 CUDA_VISIBLE_DEVICES=$G $K -u eval_kimodo.py --name sft4b_student_greedy >> logs/evalk_sft4b_student_greedy.log 2>&1; echo "$(date) kimodo eval SFT done") &
CUDA_VISIBLE_DEVICES=$G $Q -u opd.py --out opd_v1 --gate --k 4 > logs/gate_opd_v1.log 2>&1
PASS=$(python3 - <<'PY'
import json, common as C
g = json.load(open("runs/gate_GT_opd_v1.json")); R = lambda n: C.load_jsonl_shards(n, C.RUNS); m = C.manifest()
acc = lambda n: sum(C.plan_acc(R(n)[k]["samples"][0]["plan"], m["rows"][k]["plan"]) for k in m["val"]) / len(m["val"])
s, t = acc("sft4b_student_greedy"), acc("sft4b_teacher_zgreedy"); ok = g["kl_with_z"] >= 1.5 * g["kl_without_z"] and t - s >= 0.10
C.wjson(C.RUNS / "gate_GT_decision.json", {**g, "student_plan_acc": s, "teacher_z_plan_acc": t, "pass": ok,
        "rule": "kl_with_z >= 1.5 * kl_without_z AND teacher_z_plan_acc - student_plan_acc >= 0.10 (pre-registered)"})
print("PASS" if ok else "FAIL")
PY
)
echo "$(date) G-T gate: $PASS ($(tr -d '\n ' < runs/gate_GT_decision.json))"
if [ "$PASS" = PASS ]; then
  CUDA_VISIBLE_DEVICES=$G $Q -u opd.py --out opd_v1 --epochs 2 --k 4 --lr 5e-5 --accum 4 > logs/opd_v1.log 2>&1; echo "$(date) OPD done"
  for e in 1 2; do [ -f runs/opd_v1_e${e}_greedy.shard0.jsonl ] && (CUDA_VISIBLE_DEVICES=$G $K -u eval_kimodo.py --name opd_v1_e${e}_greedy --stage encode && \
    CUDA_VISIBLE_DEVICES=$G $K -u eval_kimodo.py --name opd_v1_e${e}_greedy) > logs/evalk_opd_v1_e${e}.log 2>&1; done
  echo "$(date) kimodo eval OPD done"
fi
wait; touch logs/chain_vt.done; echo "$(date) chain_vt finished"
