#!/bin/bash
# OPD v6 (README §6): 8B capability teacher, no privileged info, pure reverse KL.
cd "$(dirname "$0")"; Q=/mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python; K=/mnt/workspace/Wilson/miniconda3/envs/kimodo/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VHO_ZMODE=none OPD_MINFREE=48
B8=/tmp/vho_models/Qwen3-VL-8B-Instruct; LL=../vt_lastlayer_v1197aug
log() { echo "$(date) $*"; }
inuse() { cat runs/opd_inuse/* 2>/dev/null | cut -d: -f2 | tr ' ' '\n' | grep -E "^[0-9]+$" | tr '\n' ',' | sed 's/,$//'; }
pick1() { while true; do g=$(python3 -c "import common as C; g=C.pick_gpus(1, $1, exclude=tuple(int(x) for x in '$(inuse)'.split(',') if x)); print(g[0] if g else -1)"); [ "$g" != "-1" ] && { echo $g; return; }; sleep 120; done; }
sel() { python3 -c "import json;print(json.load(open('runs/$1/selection.json'))['selected'])"; }
a36() { n=$1; [ -e $LL/runs/$n.shard0.jsonl ] || ln -s ../../vt_head_opd_v1197aug/runs/$n.shard0.jsonl $LL/runs/$n.shard0.jsonl; bash $LL/chain_a36.sh $2 $n > $LL/logs/chain_a36_$n.log 2>&1; }
COMMON="--student sft_student --teacher sft_teacher_8b --teacher-base $B8 --epochs 1 --k 8 --lr 5e-5 --accum 4 --lam_fwd 0.0 --mr-filter none --fwd-chunk 4 --gen-videos 2 --skip-e0"
while [ ! -f runs/sft_teacher_8b/selection.json ]; do sleep 120; done; TS=runs/sft_teacher_8b/$(sel sft_teacher_8b); log "8B teacher selected $TS"
bash opd_runner.sh opd_v6 none --out opd_v6 --kl-span all $COMMON > logs/runner_opd_v6.log 2>&1 &
R1=$!; sleep 600
bash opd_runner.sh opd_v6_re none --out opd_v6_re --kl-span re $COMMON > logs/runner_opd_v6_re.log 2>&1 &
R2=$!
G=$(pick1 26); log "teacher diagnostics on GPU $G"; echo "v6 teacher diag=$G $(date)" >> runs/gpus_picked.txt
[ -f runs/sft_teacher_8b_greedy.shard0.jsonl ] && [ $(wc -l < runs/sft_teacher_8b_greedy.shard0.jsonl) -ge 125 ] || CUDA_VISIBLE_DEVICES=$G $Q -u infer.py --adapter $TS --base $B8 --name sft_teacher_8b_greedy > logs/infer_teacher_8b.log 2>&1
a36 sft_teacher_8b_greedy $G; bash run_val_motion.sh sft_teacher_8b_greedy runs/head_AB_full/best.pt $G skip_kimodo > logs/val_teacher8b_AB.log 2>&1
bash run_val_motion.sh sft_teacher_8b_greedy runs/head_A_full/best.pt $G kimodo > logs/val_teacher8b_A.log 2>&1; log "teacher diagnostics done"
evalarm() { n=$1; while [ ! -f runs/${n}_e1_greedy.shard0.jsonl ] || [ $(wc -l < runs/${n}_e1_greedy.shard0.jsonl) -lt 125 ]; do sleep 120; done
  G=$(pick1 22); log "evals $n on GPU $G"; echo "v6 eval $n=$G $(date)" >> runs/gpus_picked.txt
  a36 ${n}_e1_greedy $G
  bash run_val_motion.sh ${n}_e1_greedy runs/head_AB_full/best.pt $G skip_kimodo > logs/val_${n}_AB.log 2>&1
  bash run_val_motion.sh ${n}_e1_greedy runs/head_A_full/best.pt $G kimodo > logs/val_${n}_A.log 2>&1; log "evals $n done"; }
wait $R1; evalarm opd_v6
wait $R2; evalarm opd_v6_re
/usr/bin/python3 decide_v6.py > logs/decide_v6.log 2>&1; log "decisions: runs/gates_v6.json"; touch logs/v6.done
