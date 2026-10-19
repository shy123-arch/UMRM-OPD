#!/bin/bash
cd "$(dirname "$0")"
for c in gt3p oracle hml3d; do CUDA_VISIBLE_DEVICES=2 /mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python -u textenc.py --corpus $c --adapter /mnt/workspace/Wilson/ck/umrm/umrm_model/experiments/vt_head_opd_v1197aug/runs/sft_student/epoch1 --prefix stu_ > logs/textenc_stu_$c.log 2>&1; echo "$(date) stu_$c done"; done
CUDA_VISIBLE_DEVICES=2 /mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python -u textenc.py --corpus pred --name sft_student_greedy --adapter /mnt/workspace/Wilson/ck/umrm/umrm_model/experiments/vt_head_opd_v1197aug/runs/sft_student/epoch1 --prefix stu_ > logs/textenc_stu_pred.log 2>&1; echo "$(date) stu_pred done"
touch logs/cache_stu.done
