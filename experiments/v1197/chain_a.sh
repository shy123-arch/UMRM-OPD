#!/bin/bash
# Chain A: oracle evals -> SFT student greedy VAL -> visual feats (GT) -> head AB_full training -> SFT evals (Kimodo, A_cap,
# A_full, AB_full, AB_full with shifted visual features) -> oracle via AB_full. V->T GPU = 1; eval GPU picked when needed.
cd "$(dirname "$0")"; Q=/mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python
H=/mnt/workspace/Wilson/miniconda3/envs/hero/bin/python; HL=/mnt/workspace/Wilson/miniconda3/envs/hero/lib/python3.10/site-packages/torch/lib
GV=1; PRE=$(realpath ../vt_head_opd_v860/runs/head_pre/best.pt)
pickE() { echo 1; }  # 2026-10-16 01:10: shared GPUs other than 1 OOM-ed/saturated -> evals on GPU 1
log() { echo "$(date) $*"; }
sel() { python3 -c "import json;print(json.load(open('runs/$1/selection.json'))['selected'])"; }
while [ ! -f runs/head_A_cap/result.json ] || [ ! -f runs/head_A_full/result.json ]; do sleep 60; done
GE=$(pickE); log "A: eval GPU $GE"; echo "chain_a_eval=$GE $(date)" >> runs/gpus_picked.txt
bash run_val_motion.sh oracle_gt3p runs/head_A_cap/best.pt $GE kimodo > logs/val_oracle_A_cap.log 2>&1
bash run_val_motion.sh oracle_gt3p runs/head_A_full/best.pt $GE skip_kimodo > logs/val_oracle_A_full.log 2>&1; log "oracle evals (Kimodo, A_cap, A_full) done"
while [ ! -f runs/sft_student/selection.json ]; do sleep 60; done; SS=runs/sft_student/$(sel sft_student); log "SFT student selected $SS"
CUDA_VISIBLE_DEVICES=$GV $Q -u infer.py --adapter $SS --name sft_student_greedy > logs/infer_sft_student_greedy.log 2>&1; log "student greedy VAL done"
CUDA_VISIBLE_DEVICES=$GV $Q -u vis_feats.py --adapter $SS --source gt > logs/visfeat_gt.log 2>&1; log "vis feats GT done"
CUDA_VISIBLE_DEVICES=$GV LD_LIBRARY_PATH=$HL:/usr/local/cuda/lib64 $H -u head.py finetune --init $PRE --out head_AB_full --steps 3000 --eval-every 100 \
  --lr 1e-4 --warmup 200 --full-event --vis gt_trainval > logs/head_AB_full.log 2>&1 &
AB=$!
bash run_val_motion.sh sft_student_greedy runs/head_A_full/best.pt $GE kimodo > logs/val_sft_A_full.log 2>&1
bash run_val_motion.sh sft_student_greedy runs/head_A_cap/best.pt $GE skip_kimodo > logs/val_sft_A_cap.log 2>&1; log "SFT evals (Kimodo, A_full, A_cap) done"
wait $AB; log "head_AB_full done $(tr -d '\n ' < runs/head_AB_full/result.json)"
bash run_val_motion.sh sft_student_greedy runs/head_AB_full/best.pt $GE skip_kimodo > logs/val_sft_AB_full.log 2>&1
bash run_val_motion.sh sft_student_greedy runs/head_AB_full/best.pt $GE skip_kimodo 1 > logs/val_sft_AB_full_shift.log 2>&1
bash run_val_motion.sh oracle_gt3p runs/head_AB_full/best.pt $GE skip_kimodo > logs/val_oracle_AB_full.log 2>&1; log "AB evals done"
touch logs/chain_a.done; log "chain A finished"
