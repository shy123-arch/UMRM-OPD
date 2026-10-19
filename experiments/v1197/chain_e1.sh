#!/bin/bash
# OPD v2 epoch-1 (exploratory read-out) evals while epoch 2 trains; chain_b skips files that already exist.
cd "$(dirname "$0")"
while [ ! -f runs/opd_v2_e1_greedy.shard0.jsonl ] || [ $(wc -l < runs/opd_v2_e1_greedy.shard0.jsonl) -lt 125 ]; do sleep 60; done
sleep 30
bash run_val_motion.sh opd_v2_e1_greedy runs/head_A_full/best.pt 1 kimodo > logs/val_opd_e1_A_full.log 2>&1
bash run_val_motion.sh opd_v2_e1_greedy runs/head_AB_full/best.pt 1 skip_kimodo > logs/val_opd_e1_AB_full.log 2>&1
echo "$(date) e1 evals done"
