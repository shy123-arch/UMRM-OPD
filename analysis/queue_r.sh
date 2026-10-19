#!/bin/bash
# README §8 job queue. At most CAP GPUs of mine at any time, counted from runs/opd_inuse/* (which also holds G-T7b's GPU);
# GPU acquisition is serialised with flock so two jobs never take the same GPU. Each job: SFT (stop after epoch 2) -> VAL
# greedy -> A36 readout -> decide_r.py. The R-chain teacher job runs the G-T8 gate and, only if it passes, OPD v8 (2 GPUs).
cd "$(dirname "$0")"; Q=/mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python; LL=../vt_lastlayer_v1197aug; CAP=${CAP:-3}
log() { echo "$(date) $*"; }
used() { cat runs/opd_inuse/* 2>/dev/null | cut -d: -f2 | tr ' ' '\n' | grep -cE "^[0-9]+$"; }
acquire() {  # acquire NAME N MIN_FREE_GB -> prints N GPU indices (registered as NAME)
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
link() { for n in "$@"; do [ -e $LL/runs/$n.shard0.jsonl ] || ln -s ../../vt_head_opd_v1197aug/runs/$n.shard0.jsonl $LL/runs/$n.shard0.jsonl; done; }
decide() { (flock 8; /usr/bin/python3 decide_r.py > logs/decide_r.log 2>&1) 8>runs/.decide.lock; }

sftjob() {  # sftjob NAME ZMODE "SFT args" PRED "infer args"
  local name=$1 zm=$2 sargs=$3 pn=$4 iargs=$5 g try=0 sel
  g=$(acquire $name 1 20); log "$name on GPU $g"
  while [ ! -f runs/$name/selection.json ] && [ $try -lt 3 ]; do
    [ -d runs/$name ] && mv runs/$name runs/_failed_${name}_$(date +%H%M%S)   # sft.py has no resume: restart clean
    VHO_ZMODE=$zm CUDA_VISIBLE_DEVICES=$g $Q -u sft.py $sargs --out $name --stop-after 2 > logs/r_$name.try$try.log 2>&1; try=$((try+1))
  done
  [ -f runs/$name/selection.json ] || { log "$name SFT failed 3x"; rm -f runs/opd_inuse/$name; return 1; }
  sel=$(/usr/bin/python3 -c "import json;print(json.load(open('runs/$name/selection.json'))['selected'])"); log "$name trained, selected $sel"
  [ -f runs/$pn.args.json ] || VHO_ZMODE=$zm CUDA_VISIBLE_DEVICES=$g $Q -u infer.py --adapter runs/$name/$sel --name $pn $iargs > logs/r_infer_$pn.log 2>&1
  link $pn; bash $LL/chain_a36.sh $g $pn > logs/r_a36_$pn.log 2>&1
  rm -f runs/opd_inuse/$name; decide; log "$name readout done"; }

opdjob() {  # OPD v8: pure reverse KL, <reaction>/<exec> span, R-chain teacher (README §8); 2 GPUs
  local G n=0 RES
  while [ ! -f runs/opd_v8/selection.json ] && [ $n -lt 6 ]; do
    G=$(acquire opd_v8 2 36); RES=""; [ -f runs/opd_v8/resume.pt ] && RES="--resume"
    if [ -z "$RES" ] && [ -d runs/opd_v8 ]; then mv runs/opd_v8 runs/_failed_opd_v8_$(date +%H%M%S); fi
    log "opd_v8 attempt $n on GPUs $G $RES"
    VHO_ZMODE=mr_chain bash run_opd_ddp.sh "$G" $((29700 + n)) --out opd_v8 --kl-span re --student sft_student --teacher sft_teacher_chain \
      --epochs 1 --k 8 --lr 5e-5 --accum 4 --lam_fwd 0.0 --mr-filter chain --fwd-chunk 4 --gen-videos 2 --skip-e0 --save-every 10 $RES
    for r in 0 1; do [ -f logs/opd_v8_rank$r.log ] && cp logs/opd_v8_rank$r.log logs/opd_v8_attempt${n}_rank$r.log; done
    n=$((n+1)); [ -f runs/opd_v8/selection.json ] || { rm -f runs/opd_inuse/opd_v8; sleep 120; }
  done
  [ -f runs/opd_v8/selection.json ] || { log "opd_v8 failed"; rm -f runs/opd_inuse/opd_v8; return 1; }
  link opd_v8_e1_greedy; bash $LL/chain_a36.sh $(echo $G | cut -d' ' -f1) opd_v8_e1_greedy > logs/r_a36_opd_v8.log 2>&1
  rm -f runs/opd_inuse/opd_v8; decide; log "opd_v8 readout done"; }

chainjob() {
  sftjob sft_teacher_chain mr_chain "--mode teacher" sft_teacher_chain_zgreedy "--prompt teacher" || return 1
  if /usr/bin/python3 -c "import json,sys; sys.exit(0 if json.load(open('runs/gates_r.json')).get('G-T8',{}).get('pass') else 1)"; then
    log "G-T8 PASSED -> OPD v8"; opdjob
  else log "G-T8 not passed -> no OPD v8 (README §8)"; fi; }

sftjob sft_rcot kin_mr "--mode student --task cot" sft_rcot_greedy "--task cot --max-new-tokens 288" > logs/q_rcot.log 2>&1 &
sleep 90; chainjob > logs/q_chain.log 2>&1 &
sleep 90; sftjob sft_rmt kin_mr "--mode student --task mt" sft_rmt_greedy "--task answer" > logs/q_rmt.log 2>&1 &
sleep 90; sftjob sft_robs kin_mr "--mode student --task obs" sft_robs_greedy "--task obs --max-new-tokens 288" > logs/q_robs.log 2>&1 &
wait; decide; log "queue done"; touch logs/r.done
