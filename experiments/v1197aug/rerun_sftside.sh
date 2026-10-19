#!/bin/bash
# re-run of chain_v3's SFT-side evals that failed at 17:20 (HERO env + PYTORCH_CUDA_ALLOC_CONF); same commands
cd "$(dirname "$0")"; G=$1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash run_val_motion.sh sft_student_greedy runs/head_AB_full/best.pt $G skip_kimodo > logs/val_sft_AB.log 2>&1
bash run_val_motion.sh sft_student_greedy runs/head_AB_full/best.pt $G skip_kimodo 1 > logs/val_sft_AB_shift.log 2>&1
bash run_val_motion.sh sft_student_greedy runs/head_A_full/best.pt $G kimodo > logs/val_sft_A.log 2>&1
bash run_val_motion.sh oracle_gt3p runs/head_AB_full/best.pt $G skip_kimodo > logs/val_oracle_AB.log 2>&1
bash run_val_motion.sh sft_teacher_v3_zgreedy runs/head_AB_full/best.pt $G kimodo > logs/val_teacher_AB.log 2>&1
echo "$(date) SFT-side evals re-run done"; touch logs/sftside.done
