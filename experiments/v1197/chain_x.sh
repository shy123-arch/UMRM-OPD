#!/bin/bash
# EXPLORATORY (not pre-registered): frozen-base visual branch head_ABfrz_full on SFT predictions (+ mismatched-video control) and oracle.
cd "$(dirname "$0")"
while [ ! -f runs/eval_head_sft_student_greedy__head_AB_full_best_visshift1.json ]; do sleep 60; done   # reuse chain A's pred vis feats
bash run_val_motion.sh sft_student_greedy runs/head_ABfrz_full/best.pt 1 skip_kimodo > logs/val_sft_ABfrz.log 2>&1
bash run_val_motion.sh sft_student_greedy runs/head_ABfrz_full/best.pt 1 skip_kimodo 1 > logs/val_sft_ABfrz_shift.log 2>&1
bash run_val_motion.sh oracle_gt3p runs/head_ABfrz_full/best.pt 1 skip_kimodo > logs/val_oracle_ABfrz.log 2>&1
echo "$(date) chain_x done"
