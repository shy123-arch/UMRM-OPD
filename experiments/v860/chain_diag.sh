#!/bin/bash
# diagnostics (not pre-registered comparisons): privileged teacher (VAL hint = GT motion info) through all executors;
# then OPD epoch-1 VAL predictions through Kimodo + all own heads as soon as they are complete.
cd "$(dirname "$0")"; G=1
bash run_val_motion.sh sft4b_teacher_zgreedy runs/head_pre/best.pt $G > logs/val_teacherz_head_pre.log 2>&1; echo "$(date) teacher-z: Kimodo + head_pre done"
for h in head_ft head_ft_e1; do bash run_val_motion.sh sft4b_teacher_zgreedy runs/$h/best.pt $G skip_kimodo > logs/val_teacherz_$h.log 2>&1; done; echo "$(date) teacher-z: head_ft/e1 done"
while [ $(wc -l < runs/opd_v1_e1_greedy.shard0.jsonl) -lt 96 ]; do sleep 60; done
bash run_val_motion.sh opd_v1_e1_greedy runs/head_pre/best.pt $G > logs/val_opd_e1_head_pre.log 2>&1; echo "$(date) opd e1: Kimodo + head_pre done"
bash run_val_motion.sh opd_v1_e1_greedy runs/head_ft_e1/best.pt $G skip_kimodo > logs/val_opd_e1_head_ft_e1.log 2>&1; echo "$(date) opd e1: head_ft_e1 done"
