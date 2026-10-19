#!/bin/bash
# README §9 orchestration. <= CAP GPUs of mine at any time (runs/opd_inuse/*, flock). Waits for the ego video cache.
cd "$(dirname "$0")"; Q=/mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python; LL=../vt_lastlayer_v1197aug; CAP=${CAP:-3}; EC=/tmp/vho_ego_cache
log() { echo "$(date) $*"; }
used() { cat runs/opd_inuse/* 2>/dev/null | cut -d: -f2 | tr ' ' '\n' | grep -cE "^[0-9]+$"; }
acquire() {  # acquire NAME N MIN_FREE_GB -> N GPU indices
  local g ex
  while true; do
    g=$( (flock 9
          if [ $(( $(used) + $2 )) -le $CAP ]; then
            ex=$(cat runs/opd_inuse/* 2>/dev/null | cut -d: -f2 | tr ' ' '\n' | grep -E "^[0-9]+$" | tr '\n' ','); ex=${ex%,}
            x=$($Q -c "import common as C; r=C.pick_gpus($2, $3, exclude=tuple(int(x) for x in '$ex'.split(',') if x)); print(*r if len(r)==$2 else [])")
            [ -n "$x" ] && { echo "$1:$x" > runs/opd_inuse/$1; echo "$1 GPU=$x $(date)" >> runs/gpus_picked.txt; echo $x; }
          fi ) 9>runs/.acquire.lock )
    [ -n "$g" ] && { echo $g; return; }; sleep 120
  done; }
release() { rm -f runs/opd_inuse/$1; }
link() { for n in "$@"; do [ -e $LL/runs/$n.shard0.jsonl ] || ln -s ../../vt_head_opd_v1197aug/runs/$n.shard0.jsonl $LL/runs/$n.shard0.jsonl; done; }
decide() { (flock 8; /usr/bin/python3 decide_e.py > logs/decide_e.log 2>&1) 8>runs/.decide.lock; }
probe() {  # probe NAME [ADAPTER]
  local g; g=$(acquire probe_$1 1 20); log "probe $1 on GPU $g"
  [ -f runs/probe_$1.json ] || VHO_CACHE=$EC CUDA_VISIBLE_DEVICES=$g $Q -u ego_probe.py --name $1 ${2:+--adapter $2} > logs/e_probe_$1.log 2>&1
  release probe_$1; decide; log "probe $1 done"; }
sel() { /usr/bin/python3 -c "import json;print(json.load(open('runs/$1/selection.json'))['selected'])"; }

# 1) probes of the reference models as soon as the probe cache is complete
( until grep -q "^done" logs/ego_cache_probe.log 2>/dev/null; do sleep 120; done
  probe base & probe sft runs/sft_student/epoch1 & wait ) > logs/q_e_probes_ref.log 2>&1 &
PREF=$!
# 2) stage A on 3 GPUs once the training cache is complete
until [ -f logs/ego_cache.done ]; do sleep 120; done; log "ego cache complete: $(ls $EC/video | wc -l) files"
wait $PREF
n=0
while [ ! -f runs/pt_c/selection.json ] && [ $n -lt 6 ]; do
  G=$(acquire pt_c 3 24); RES=""; [ -f runs/pt_c/resume.pt ] && RES="--resume"
  if [ -z "$RES" ] && [ -d runs/pt_c ]; then mv runs/pt_c runs/_failed_pt_c_$(date +%H%M%S); fi
  log "pt_c attempt $n on GPUs $G $RES"
  bash run_ego_pt.sh "$G" $((29800 + n)) --out pt_c --epochs 1 --lr 1e-4 --accum 8 --save-every 50 $RES
  for r in 0 1 2; do [ -f logs/pt_c_rank$r.log ] && cp logs/pt_c_rank$r.log logs/pt_c_attempt${n}_rank$r.log; done
  release pt_c; n=$((n+1)); [ -f runs/pt_c/selection.json ] || sleep 120
done
[ -f runs/pt_c/selection.json ] || { log "pt_c failed"; exit 1; }
log "pt_c trained"; decide
# 3) stage B (SFT from pt_c) || probe pt_c
probe pt_c runs/pt_c/epoch1 > logs/q_e_probe_ptc.log 2>&1 &
PP=$!
g=$(acquire sft_ptc 1 20); log "sft_ptc on GPU $g"; try=0
while [ ! -f runs/sft_ptc/selection.json ] && [ $try -lt 3 ]; do
  [ -d runs/sft_ptc ] && mv runs/sft_ptc runs/_failed_sft_ptc_$(date +%H%M%S)
  CUDA_VISIBLE_DEVICES=$g $Q -u sft.py --mode student --task answer --init-adapter runs/pt_c/epoch1 --out sft_ptc --stop-after 2 > logs/e_sft_ptc.try$try.log 2>&1; try=$((try+1))
done
[ -f runs/sft_ptc/selection.json ] || { log "sft_ptc failed 3x"; release sft_ptc; exit 1; }
S=$(sel sft_ptc); log "sft_ptc trained, selected $S"
[ -f runs/sft_ptc_greedy.args.json ] || CUDA_VISIBLE_DEVICES=$g $Q -u infer.py --adapter runs/sft_ptc/$S --name sft_ptc_greedy > logs/e_infer_sft_ptc.log 2>&1
link sft_ptc_greedy; bash $LL/chain_a36.sh $g sft_ptc_greedy > logs/e_a36_sft_ptc.log 2>&1
release sft_ptc; decide; log "sft_ptc readout done"
wait $PP
probe sft_ptc runs/sft_ptc/$S
decide; log "queue_e done"; touch logs/e.done
