#!/bin/bash
# OPD v4 (README §4): train (2 GPUs, watchdog-style unlimited resume) -> evals on an idle eval GPU -> gates.
cd "$(dirname "$0")"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date) $*"; }
OPDARGS="--out opd_v4 --student sft_student --teacher sft_teacher_v3 --epochs 1 --k 8 --lr 5e-5 --accum 4 --lam_fwd 0.5 --half-ckpt --fwd-chunk 4 --gen-videos 2 --skip-e0 --mr-filter human"
GO=$1; n=0; first=1
while [ ! -f runs/opd_v4/selection.json ]; do
  RES=""; [ -f runs/opd_v4/resume.pt ] && RES="--resume"
  if [ $first = 0 ]; then n=$((n+1)); sleep 120; for r in 0 1; do [ -f logs/opd_v4_rank$r.log ] && cp logs/opd_v4_rank$r.log logs/opd_v4_attempt${n}_rank$r.log; done
    GO=$(python3 -c "import common as C; print(' '.join(map(str, C.pick_gpus(2, 36))))"); [ $(echo $GO | wc -w) -lt 2 ] && { log "no 2 GPUs free"; sleep 300; continue; }; fi
  first=0; log "OPD v4 attempt $n on GPUs $GO $RES"; echo "opd_v4 attempt $n GPUs=$GO $(date)" >> runs/gpus_picked.txt
  bash run_opd_ddp.sh "$GO" $((29800+n)) $OPDARGS $RES
done
log "OPD v4 trained"
while [ ! -f runs/opd_v4_e1_greedy.shard0.jsonl ] || [ $(wc -l < runs/opd_v4_e1_greedy.shard0.jsonl) -lt 125 ]; do sleep 60; done
G=$(python3 -c "import common as C; print(C.pick_gpus(1, 20)[0])"); log "evals on GPU $G"; echo "opd_v4 eval=$G $(date)" >> runs/gpus_picked.txt
bash run_val_motion.sh opd_v4_e1_greedy runs/head_AB_full/best.pt $G skip_kimodo > logs/val_opd_v4_AB.log 2>&1
bash run_val_motion.sh opd_v4_e1_greedy runs/head_AB_full/best.pt $G skip_kimodo 1 > logs/val_opd_v4_AB_shift.log 2>&1
bash run_val_motion.sh opd_v4_e1_greedy runs/head_A_full/best.pt $G kimodo > logs/val_opd_v4_A.log 2>&1; log "OPD v4 evals done"; touch logs/v4.done
