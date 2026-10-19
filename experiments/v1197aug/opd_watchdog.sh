#!/bin/bash
# Keeps OPD v3 alive until runs/opd_v3/selection.json exists: whenever no opd_v3 training process is running (and the
# original chain's retry loop has given up or is between attempts), archive the rank logs and resume on the 2 GPUs with
# the most free memory (state is global, so changing GPUs does not change the run). Unlimited attempts, 3-min back-off.
cd "$(dirname "$0")"; n=0
OPDARGS="--out opd_v3 --student sft_student --teacher sft_teacher_v3 --epochs 1 --k 8 --lr 5e-5 --accum 4 --lam_fwd 0.5 --half-ckpt --fwd-chunk 4 --gen-videos 2 --skip-e0"
running() { ps -eo cmd | grep -v grep | grep -q "opd2.py --out opd_v3 --student sft_student --teacher sft_teacher_v3 --epochs 1"; }
while [ ! -f runs/opd_v3/selection.json ]; do
  if running; then sleep 120; continue; fi
  sleep 180; running && continue; [ -f runs/opd_v3/selection.json ] && break
  n=$((n+1)); for r in 0 1; do [ -f logs/opd_v3_rank$r.log ] && cp logs/opd_v3_rank$r.log logs/opd_v3_attempt$(date +%H%M)_rank$r.log; done
  G=$(python3 -c "import common as C; print(' '.join(map(str, C.pick_gpus(2, 36))))"); [ $(echo $G | wc -w) -lt 2 ] && { echo "$(date) no 2 GPUs with 36 GB free; wait"; sleep 300; continue; }
  echo "$(date) watchdog resume #$n on GPUs $G (step $(python3 -c "import torch;print(torch.load('runs/opd_v3/resume.pt',map_location='cpu',weights_only=False)['step'])" 2>/dev/null))"
  echo "opd_v3 watchdog resume #$n GPUs=$G $(date)" >> runs/gpus_picked.txt
  bash run_opd_ddp.sh "$G" $((29700+n)) $OPDARGS --resume
done
echo "$(date) watchdog: OPD v3 finished"
