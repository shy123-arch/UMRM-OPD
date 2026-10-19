#!/bin/bash
# ~20-minute smoke test of every stage on 16 TRAIN / 4 VAL items (VHO_LIMIT), tiny step counts. One GPU, ~30 GB.
# Usage: GPU=0 [SMOKE_FROM=prep|sft|infer|vis|head|opd|eval] [SMOKE_KIMODO=kimodo] bash pipeline/smoke.sh
# Outputs in ${VHO_RUNS_SMOKE:-/tmp/vho_smoke/runs}; caches in ${VHO_CACHE_SMOKE:-/tmp/vho_smoke/cache}.
set -e; cd "$(dirname "$0")"; source ./_env.sh
export VHO_LIMIT=16 VHO_RUNS=${VHO_RUNS_SMOKE:-/tmp/vho_smoke/runs} VHO_CACHE=${VHO_CACHE_SMOKE:-/tmp/vho_smoke/cache}
source ./_env.sh   # re-derive RUNSDIR / C for the smoke locations
G=${GPU:-0}; run() { CUDA_VISIBLE_DEVICES=$G $Q -u "$@"; }; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STAGES=(prep sft infer vis head opd eval); FROM=${SMOKE_FROM:-prep}; on=0
want() { [ "$1" = "$FROM" ] && on=1; [ $on = 1 ]; }
log() { echo "$(date '+%T') SMOKE $*"; }
if want prep; then log prep; $Q -u prep_video_cache.py --workers 4; run textenc.py --corpus gt3p; fi
if want sft; then log sft; run sft.py --mode student --out sft_student --epochs 1 --accum 4; run sft.py --mode teacher --out sft_teacher --epochs 1 --accum 4; fi
if want infer; then log infer; run infer.py --adapter sft_student/epoch1 --name sft_student_greedy; fi
if want vis; then log vis; run vis_feats.py --adapter $RUNSDIR/sft_student/epoch1 --source gt; fi
if want head; then log head
  for v in A AB; do VIS=""; [ $v = AB ] && VIS="--vis gt_trainval"
    hero $G head.py finetune --init $VHO_CKPT/head_pre.pt --out head_${v}_full --steps 20 --eval-every 10 --lr 1e-4 --warmup 5 --full-event --batch-size 8 $VIS; done; fi
if want opd; then log opd; run opd.py --out opd --student sft_student --teacher sft_teacher --epochs 1 --k 8 --accum 4 --fwd-chunk 2 --gen-videos 1 --limit 8 --skip-e0 --save-every 1; fi
if want eval; then log eval; export VHO_VIS_ENCODER=$RUNSDIR/sft_student/epoch1
  bash run_val_motion.sh sft_student_greedy $RUNSDIR/head_AB_full/best.pt $G skip_kimodo
  bash run_val_motion.sh opd_e1_greedy $RUNSDIR/head_A_full/best.pt $G ${SMOKE_KIMODO:-skip_kimodo}; fi
log "OK - stages from '$FROM' ran"; ls $RUNSDIR
