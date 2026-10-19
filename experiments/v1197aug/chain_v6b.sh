#!/bin/bash
# Replaces chain_v6.sh after the teacher diagnostics / OPD training (17:50): chain_v6.sh's a36() overwrote evalarm's $n
# (no 'local'), so the AB / A+Kimodo evals of opd_v6 looked for a non-existent file. Same eval commands, fixed scoping.
cd "$(dirname "$0")"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VHO_ZMODE=none; LL=../vt_lastlayer_v1197aug
log() { echo "$(date) $*"; }
inuse() { cat runs/opd_inuse/* 2>/dev/null | cut -d: -f2 | tr ' ' '\n' | grep -E "^[0-9]+$" | tr '\n' ',' | sed 's/,$//'; }
pick1() { local g; while true; do g=$(python3 -c "import common as C; g=C.pick_gpus(1, $1, exclude=tuple(int(x) for x in '$(inuse)'.split(',') if x)); print(g[0] if g else -1)"); [ "$g" != "-1" ] && { echo $g; return; }; sleep 120; done; }
a36() { local p=$1 g=$2; [ -e $LL/runs/$p.shard0.jsonl ] || ln -s ../../vt_head_opd_v1197aug/runs/$p.shard0.jsonl $LL/runs/$p.shard0.jsonl; bash $LL/chain_a36.sh $g $p > $LL/logs/chain_a36_$p.log 2>&1; }
evalarm() { local arm=$1 p=$1_e1_greedy g
  while [ ! -f runs/$p.shard0.jsonl ] || [ $(wc -l < runs/$p.shard0.jsonl) -lt 125 ]; do sleep 120; done
  g=$(pick1 22); log "evals $arm on GPU $g"; echo "v6 eval $arm=$g $(date)" >> runs/gpus_picked.txt
  a36 $p $g
  bash run_val_motion.sh $p runs/head_AB_full/best.pt $g skip_kimodo > logs/val_${arm}_AB.log 2>&1
  bash run_val_motion.sh $p runs/head_A_full/best.pt $g kimodo > logs/val_${arm}_A.log 2>&1; log "evals $arm done"; }
evalarm opd_v6
while [ ! -f runs/opd_v6_re/selection.json ]; do sleep 120; done; evalarm opd_v6_re
/usr/bin/python3 decide_v6.py > logs/decide_v6.log 2>&1; log "decisions: runs/gates_v6.json"; touch logs/v6.done
