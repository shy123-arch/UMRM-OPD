#!/bin/bash
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/mnt/workspace/Wilson/miniconda3/envs/hero/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64 /mnt/workspace/Wilson/miniconda3/envs/hero/bin/python -u head.py finetune --full-event --init runs/head_pre/best.pt --out head_ft_e1 --steps 6000 --eval-every 250 --lr 1e-4 --warmup 200 > logs/head_ft_e1.log 2>&1
echo "$(date) head_ft_e1 done $(tr -d '\n ' < runs/head_ft_e1/result.json)"
bash run_val_motion.sh oracle runs/head_ft_e1/best.pt 1 > logs/val_oracle_head_ft_e1.log 2>&1; echo "$(date) oracle e1 done"
bash run_val_motion.sh sft4b_student_greedy runs/head_ft_e1/best.pt 1 skip_kimodo > logs/val_sft4b_head_ft_e1.log 2>&1; echo "$(date) sft4b e1 done"
