#!/bin/bash
# After OPD v3 really finished (selection.json + e1 greedy complete): OPD evals (skips existing), half-ckpt evals,
# gate decision, summary. Eval GPU = most idle with >= 14 GB (excluding the OPD GPUs while they are busy).
cd "$(dirname "$0")"; Q=/mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python
log() { echo "$(date) $*"; }
while [ ! -f runs/opd_v3/selection.json ] || [ ! -f runs/opd_v3_e1_greedy.shard0.jsonl ] || [ $(wc -l < runs/opd_v3_e1_greedy.shard0.jsonl) -lt 125 ]; do sleep 120; done
while [ ! -f logs/v3.done ] && ps -eo cmd | grep -v grep | grep -q "chain_v3.sh"; do sleep 120; done   # let chain_v3 finish its own steps first
G=$(python3 -c "import common as C; print(C.pick_gpus(1, 20, exclude=(1,7))[0])"); log "final evals on GPU $G"; echo "chain_final eval=$G $(date)" >> runs/gpus_picked.txt
n=opd_v3_e1_greedy
bash run_val_motion.sh $n runs/head_AB_full/best.pt $G skip_kimodo > logs/final_${n}_AB.log 2>&1
bash run_val_motion.sh $n runs/head_AB_full/best.pt $G skip_kimodo 1 > logs/final_${n}_AB_shift.log 2>&1
bash run_val_motion.sh $n runs/head_A_full/best.pt $G kimodo > logs/final_${n}_A.log 2>&1; log "OPD v3 evals done"
if [ -f runs/opd_v3/half/adapter_config.json ]; then
  [ -f runs/opd_v3_half_greedy.shard0.jsonl ] && [ $(wc -l < runs/opd_v3_half_greedy.shard0.jsonl) -ge 125 ] || CUDA_VISIBLE_DEVICES=$G $Q -u infer.py --adapter runs/opd_v3/half --name opd_v3_half_greedy > logs/final_infer_half.log 2>&1
  bash run_val_motion.sh opd_v3_half_greedy runs/head_AB_full/best.pt $G kimodo > logs/final_half_AB.log 2>&1; log "half evals done"
fi
/usr/bin/python3 decide_aug.py > logs/final_decide.log 2>&1; log "gates written: runs/gates_aug.json"; touch logs/final.done
