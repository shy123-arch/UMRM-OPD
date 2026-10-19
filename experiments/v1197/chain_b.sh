#!/bin/bash
# Chain B: teacher v2 -> teacher greedy VAL (with z; diagnostic) -> G-T2 (KL part on GPU 1; text part on eval GPU) ->
# OPD v2 (runs regardless, per user directive) -> OPD epoch evals (Kimodo, A_full, AB_full).
cd "$(dirname "$0")"; Q=/mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python; K=/mnt/workspace/Wilson/miniconda3/envs/kimodo/bin/python
GV=1
pickE() { echo 1; }  # 2026-10-16 01:10: shared GPUs other than 1 OOM-ed/saturated -> evals on GPU 1
log() { echo "$(date) $*"; }
sel() { python3 -c "import json;print(json.load(open('runs/$1/selection.json'))['selected'])"; }
while [ ! -f runs/sft_teacher_v2/selection.json ] || [ ! -f runs/sft_student/selection.json ]; do sleep 60; done
ST=runs/sft_teacher_v2/$(sel sft_teacher_v2); log "teacher v2 selected $ST"
CUDA_VISIBLE_DEVICES=$GV $Q -u infer.py --adapter $ST --prompt teacher --name sft_teacher_v2_zgreedy > logs/infer_teacher_v2_zgreedy.log 2>&1; log "teacher greedy done"
CUDA_VISIBLE_DEVICES=$GV $Q -u opd2.py --out opd_v2 --gate --k 4 > logs/gate_GT2.log 2>&1; log "G-T2 KL: $(tr -d '\n ' < runs/gate_GT2_opd_v2.json)"
while [ ! -f runs/sft_student_greedy.shard0.jsonl ] || [ $(wc -l < runs/sft_student_greedy.shard0.jsonl) -lt 125 ]; do sleep 60; done
GE=$(pickE); echo "chain_b_eval=$GE $(date)" >> runs/gpus_picked.txt
CUDA_VISIBLE_DEVICES=$GE $K -u gate_text.py --a sft_teacher_v2_zgreedy --b sft_student_greedy --out gate_GT2_text > logs/gate_GT2_text.log 2>&1; log "G-T2 text: $(tr -d '\n ' < runs/gate_GT2_text.json)"
CUDA_VISIBLE_DEVICES=$GV $Q -u opd2.py --out opd_v2 --epochs 2 --k 4 --lr 5e-5 --accum 4 --lam_fwd 0.5 > logs/opd_v2.log 2>&1; log "OPD v2 done"
while [ ! -f logs/chain_a.done ]; do sleep 60; done
GE=$(pickE); echo "chain_b_eval2=$GE $(date)" >> runs/gpus_picked.txt
for e in 2 1; do
  bash run_val_motion.sh opd_v2_e${e}_greedy runs/head_A_full/best.pt $GE kimodo > logs/val_opd_e${e}_A_full.log 2>&1
  bash run_val_motion.sh opd_v2_e${e}_greedy runs/head_AB_full/best.pt $GE skip_kimodo > logs/val_opd_e${e}_AB_full.log 2>&1; log "OPD e$e evals done"
done
bash run_val_motion.sh sft_teacher_v2_zgreedy runs/head_A_full/best.pt $GE kimodo > logs/val_teacher_A_full.log 2>&1; log "teacher-z diag evals done"
touch logs/chain_b.done; log "chain B finished"
