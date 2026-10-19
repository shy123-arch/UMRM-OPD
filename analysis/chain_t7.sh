#!/bin/bash
# README §7 teacher gates, two branches in parallel (each picks the most idle GPU at its launch, excluding GPUs listed in runs/opd_inuse/*):
#   A (G-T7a): 16 temperature-1 samples per VAL item from the SFT student and the 8B teacher -> per-sample A36 readouts
#   B (G-T7b): 32-frame video cache (CPU) -> 4B SFT (same recipe) -> VAL greedy -> A36 readout
# then decide_t7.py -> runs/gates_t7.json, logs/t7.done.
cd "$(dirname "$0")"; Q=/mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python; LL=../vt_lastlayer_v1197aug
Q8=/tmp/vho_models/Qwen3-VL-8B-Instruct; C32=/tmp/vho1197aug_f32_cache
log() { echo "$(date) $*"; }
pick() {  # pick NAME MIN_FREE_GB -> GPU index (registered in runs/opd_inuse/NAME)
  local ex g
  while true; do
    ex=$(cat runs/opd_inuse/* 2>/dev/null | grep -v "^$1:" | cut -d: -f2 | tr ' ' '\n' | grep -E "^[0-9]+$" | tr '\n' ','); ex=${ex%,}
    g=$($Q -c "import common as C; print(*C.pick_gpus(1, $2, exclude=tuple(int(x) for x in '$ex'.split(',') if x)))")
    [ -n "$g" ] && break; log "$1: no GPU with >= $2 GB free (excluding $ex); wait" >&2; sleep 300
  done
  echo "$1:$g" > runs/opd_inuse/$1; echo "$1 GPU=$g $(date)" >> runs/gpus_picked.txt; echo $g; }
link() { for n in "$@"; do [ -e $LL/runs/$n.shard0.jsonl ] || ln -s ../../vt_head_opd_v1197aug/runs/$n.shard0.jsonl $LL/runs/$n.shard0.jsonl; done; }

branchA() {
  local g; g=$(pick t7a 30); log "A on GPU $g"
  [ -f runs/sc16_stu.args.json ] || CUDA_VISIBLE_DEVICES=$g $Q -u infer.py --adapter runs/sft_student/epoch1 --name sc16_stu --k 16 --temp 1.0 > logs/t7_sc16_stu.log 2>&1
  log "A student sampled"
  [ -f runs/sc16_t8b.args.json ] || CUDA_VISIBLE_DEVICES=$g $Q -u infer.py --adapter runs/sft_teacher_8b/epoch1 --base $Q8 --name sc16_t8b --k 16 --temp 1.0 > logs/t7_sc16_t8b.log 2>&1
  log "A 8B sampled"
  local names=""
  for m in stu t8b; do /usr/bin/python3 sc_split.py sc16_$m >> logs/t7_split.log 2>&1; for k in $(seq -w 0 15); do names="$names sc16_${m}_k$k"; done; done
  link $names; bash $LL/chain_a36.sh $g $names > logs/t7_a36_A.log 2>&1
  rm -f runs/opd_inuse/t7a; log "A done"; }

branchB() {
  [ -f $C32/video_manifest.json ] || VHO_NFRAMES=32 VHO_CACHE=$C32 $Q -u prep_video_cache.py --workers 8 > logs/t7_f32_cache.log 2>&1
  log "B cache done $(tail -1 logs/t7_f32_cache.log)"
  local g; g=$(pick t7b 30); log "B on GPU $g"
  [ -f runs/sft_f32/selection.json ] || VHO_NFRAMES=32 VHO_CACHE=$C32 CUDA_VISIBLE_DEVICES=$g $Q -u sft.py --mode student --out sft_f32 > logs/t7_sft_f32.log 2>&1
  local sel; sel=$(/usr/bin/python3 -c "import json;print(json.load(open('runs/sft_f32/selection.json'))['selected'])"); log "B SFT done, selected $sel"
  [ -f runs/sft_f32_greedy.args.json ] || VHO_NFRAMES=32 VHO_CACHE=$C32 CUDA_VISIBLE_DEVICES=$g $Q -u infer.py --adapter runs/sft_f32/$sel --name sft_f32_greedy > logs/t7_infer_f32.log 2>&1
  link sft_f32_greedy; bash $LL/chain_a36.sh $g sft_f32_greedy > logs/t7_a36_B.log 2>&1
  rm -f runs/opd_inuse/t7b; log "B done"; }

branchA > logs/t7_branchA.log 2>&1 &
PA=$!; sleep 30
branchB > logs/t7_branchB.log 2>&1 &
PB=$!
wait $PA $PB
/usr/bin/python3 decide_t7.py > logs/decide_t7.log 2>&1; log "decide_t7 rc=$?"; touch logs/t7.done
