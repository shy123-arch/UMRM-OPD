#!/bin/bash
# Takes over from chain_t7.sh (20:40): the 8B sampling runs on a SECOND GPU in parallel with the (still running, orphaned) student
# sampling, and the two readout series run on their own GPUs. Design unchanged (README §7). Usage: chain_t7b.sh STU_INFER_PID BRANCH_B_PID
cd "$(dirname "$0")"; Q=/mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python; LL=../vt_lastlayer_v1197aug
Q8=/tmp/vho_models/Qwen3-VL-8B-Instruct; PS=$1; PB=$2
log() { echo "$(date) $*"; }
pick() {
  local ex g
  while true; do
    ex=$(cat runs/opd_inuse/* 2>/dev/null | grep -v "^$1:" | cut -d: -f2 | tr ' ' '\n' | grep -E "^[0-9]+$" | tr '\n' ','); ex=${ex%,}
    g=$($Q -c "import common as C; print(*C.pick_gpus(1, $2, exclude=tuple(int(x) for x in '$ex'.split(',') if x)))")
    [ -n "$g" ] && break; log "$1: no GPU with >= $2 GB free (excluding $ex); wait" >&2; sleep 300
  done
  echo "$1:$g" > runs/opd_inuse/$1; echo "$1 GPU=$g $(date)" >> runs/gpus_picked.txt; echo $g; }
link() { for n in "$@"; do [ -e $LL/runs/$n.shard0.jsonl ] || ln -s ../../vt_head_opd_v1197aug/runs/$n.shard0.jsonl $LL/runs/$n.shard0.jsonl; done; }
readout() { local m=$1 g=$2 names=""; /usr/bin/python3 sc_split.py sc16_$m >> logs/t7_split.log 2>&1
  for k in $(seq -w 0 15); do names="$names sc16_${m}_k$k"; done; link $names; bash $LL/chain_a36.sh $g $names > logs/t7_a36_$m.log 2>&1; }

stu() { local g; g=$(cut -d: -f2 runs/opd_inuse/t7a)
  while kill -0 $PS 2>/dev/null; do sleep 60; done
  [ -f runs/sc16_stu.args.json ] || { log "student sampling did not finish"; return 1; }
  log "student sampled; readouts on GPU $g"; readout stu $g; rm -f runs/opd_inuse/t7a; log "student readouts done"; }
t8b() { local g; g=$(pick t7a8 30); log "8B sampling on GPU $g"
  [ -f runs/sc16_t8b.args.json ] || CUDA_VISIBLE_DEVICES=$g $Q -u infer.py --adapter runs/sft_teacher_8b/epoch1 --base $Q8 --name sc16_t8b --k 16 --temp 1.0 > logs/t7_sc16_t8b.log 2>&1
  log "8B sampled"; readout t8b $g; rm -f runs/opd_inuse/t7a8; log "8B readouts done"; }

stu > logs/t7_stu.log 2>&1 & P1=$!
t8b > logs/t7_t8b.log 2>&1 & P2=$!
wait $P1 $P2
while kill -0 $PB 2>/dev/null; do sleep 60; done; log "branch B finished"
/usr/bin/python3 decide_t7.py > logs/decide_t7.log 2>&1; log "decide_t7 rc=$?"; touch logs/t7.done
