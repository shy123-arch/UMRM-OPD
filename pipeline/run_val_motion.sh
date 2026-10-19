#!/bin/bash
# Evaluate one V->T prediction file ($RUNSDIR/<NAME>.shard*.jsonl; NAME=oracle uses gt3p + GT plan durations) through
#   the own action head HEAD_CKPT (text feats -> [visual feats if the head has branch B] -> head.py gen -> eval_head.py)
#   and, unless SKIP=skip_kimodo, the frozen-Kimodo executor (eval_kimodo.py encode + eval, arms T / TPH).
# Usage: run_val_motion.sh NAME HEAD_CKPT GPU [kimodo|skip_kimodo] [VIS_SHIFT]   (VHO_SPLIT=val|sealed selects the split)
set -e
cd "$(dirname "$0")"; source ./_env.sh
NAME=$1; CKPT=$2; G=$3; KM=${4:-kimodo}; SHIFT=${5:-0}
TAG=$(basename $(dirname $CKPT))_$(basename $CKPT .pt)
sel() { python3 -c "import json;print(json.load(open('$RUNSDIR/$1/selection.json'))['selected'])"; }
if [ "$NAME" = oracle ]; then CORPUS=oracle_gt3p_val; VC=gt_trainval
  [ -f $C/textfeat/$CORPUS.meta.json ] || CUDA_VISIBLE_DEVICES=$G $Q -u textenc.py --corpus oracle
else CORPUS=pred_$NAME; VC=pred_$NAME
  [ -f $C/textfeat/$CORPUS.meta.json ] || CUDA_VISIBLE_DEVICES=$G $Q -u textenc.py --corpus pred --name $NAME; fi
VISARG=""; SUF=""
if [[ "$TAG" == *AB* ]]; then   # branch B needs the frozen SFT student's hidden states of these predictions
  SS=${VHO_VIS_ENCODER:-$RUNSDIR/sft_student/$(sel sft_student)}
  [ -f $C/visfeat/$VC.meta.json ] || CUDA_VISIBLE_DEVICES=$G $Q -u vis_feats.py --adapter $SS --source pred --name $NAME
  VISARG="--vis-corpus $VC --vis-shift $SHIFT"; [ "$SHIFT" != 0 ] && SUF="_visshift$SHIFT"
fi
OUT=gen/${NAME}__${TAG}${SUF}.npz
[ -f $RUNSDIR/$OUT ] || hero $G head.py gen --ckpt $CKPT --corpus $CORPUS --out $OUT --samples 3 $VISARG
[ -f $RUNSDIR/eval_head_${NAME}__${TAG}${SUF}.json ] || CUDA_VISIBLE_DEVICES=$G $K -u eval_head.py --npz $RUNSDIR/$OUT --name ${NAME}__${TAG}${SUF}
if [ "$NAME" != oracle ] && [ "$KM" = kimodo ] && [ ! -f $RUNSDIR/eval_kimodo_$NAME.json ]; then
  CUDA_VISIBLE_DEVICES=$G $K -u eval_kimodo.py --name $NAME --stage encode; CUDA_VISIBLE_DEVICES=$G $K -u eval_kimodo.py --name $NAME; fi
echo "done $NAME $TAG$SUF"
