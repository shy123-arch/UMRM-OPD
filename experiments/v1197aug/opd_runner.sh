#!/bin/bash
# opd_runner.sh NAME ZMODE <opd2.py args>  -- train until runs/NAME/selection.json exists. Each attempt picks the 2 most idle
# GPUs (>= 36 GB free) not used by another of my OPD runners (runs/opd_inuse/*), resumes from runs/NAME/resume.pt, archives logs.
cd "$(dirname "$0")"; NAME=$1; export VHO_ZMODE=$2; shift 2; n=0
while [ ! -f runs/$NAME/selection.json ]; do
  EX=$(cat runs/opd_inuse/* 2>/dev/null | grep -v "^$NAME:" | cut -d: -f2 | tr ' ' '\n' | grep -E "^[0-9]+$" | tr '\n' ',' ); EX=${EX%,}
  G=$(python3 -c "import common as C; print(' '.join(map(str, C.pick_gpus(2, ${OPD_MINFREE:-36}, exclude=tuple(int(x) for x in '$EX'.split(',') if x)))))")
  if [ $(echo $G | wc -w) -lt 2 ]; then echo "$(date) $NAME: no 2 free GPUs (excluding $EX); wait"; sleep 300; continue; fi
  echo "$NAME:$G" > runs/opd_inuse/$NAME; RES=""; [ -f runs/$NAME/resume.pt ] && RES="--resume"
  echo "$(date) $NAME attempt $n on GPUs $G $RES"; echo "$NAME attempt $n GPUs=$G zmode=$VHO_ZMODE $(date)" >> runs/gpus_picked.txt
  bash run_opd_ddp.sh "$G" $((29900 + 10 * $(echo $NAME | cksum | cut -c1-2) % 900 + n)) "$@" $RES
  for r in 0 1; do [ -f logs/${NAME}_rank$r.log ] && cp logs/${NAME}_rank$r.log logs/${NAME}_attempt${n}_rank$r.log; done
  n=$((n+1)); [ -f runs/$NAME/selection.json ] || sleep 120
done
rm -f runs/opd_inuse/$NAME; echo "$(date) $NAME trained"
