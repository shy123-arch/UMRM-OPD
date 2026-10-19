#!/bin/bash
cd "$(dirname "$0")"; G=1
while [ ! -f runs/opd_v1_e2_greedy.shard0.jsonl ] || [ $(wc -l < runs/opd_v1_e2_greedy.shard0.jsonl) -lt 96 ]; do sleep 60; done
for h in head_pre head_ft_e1; do bash run_val_motion.sh opd_v1_e2_greedy runs/$h/best.pt $G skip_kimodo > logs/val_opd_e2_$h.log 2>&1; echo "$(date) opd e2 $h done"; done
