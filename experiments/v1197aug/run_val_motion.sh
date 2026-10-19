#!/bin/bash
# VAL motion evaluation of one V->T prediction file (or NAME=oracle: gt3p + GT plan-label durations) through
#   own head HEAD_CKPT (text feats -> [vis feats if the head has branch B] -> head.py gen -> eval_head.py) and, unless
#   skip_kimodo, frozen Kimodo (eval_kimodo.py encode/eval, arms T/TPH).
# Usage: run_val_motion.sh NAME HEAD_CKPT GPU [skip_kimodo|kimodo] [VIS_SHIFT]
set -e
cd "$(dirname "$0")"; NAME=$1; CKPT=$2; G=$3; KM=${4:-kimodo}; SHIFT=${5:-0}
Q=/mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python; K=/mnt/workspace/Wilson/miniconda3/envs/kimodo/bin/python
H=/mnt/workspace/Wilson/miniconda3/envs/hero/bin/python; HL=/mnt/workspace/Wilson/miniconda3/envs/hero/lib/python3.10/site-packages/torch/lib
TAG=$(basename $(dirname $CKPT))_$(basename $CKPT .pt); C=${VHO_CACHE:-/tmp/vho1197aug_cache}
if [ "$NAME" = oracle ]; then CORPUS=oracle_gt3p_val; VC=gt_trainval; [ -f $C/textfeat/$CORPUS.meta.json ] || CUDA_VISIBLE_DEVICES=$G $Q -u textenc.py --corpus oracle
else CORPUS=pred_$NAME; VC=pred_$NAME; [ -f $C/textfeat/$CORPUS.meta.json ] || CUDA_VISIBLE_DEVICES=$G $Q -u textenc.py --corpus pred --name $NAME; fi
VISARG=""; SUF=""
if [[ "$TAG" == *AB* ]]; then
  SS=runs/sft_student/$(python3 -c "import json;print(json.load(open('runs/sft_student/selection.json'))['selected'])")
  [ -f $C/visfeat/$VC.meta.json ] || CUDA_VISIBLE_DEVICES=$G $Q -u vis_feats.py --adapter $SS --source pred --name $NAME
  VISARG="--vis-corpus $VC --vis-shift $SHIFT"; [ "$SHIFT" != 0 ] && SUF="_visshift$SHIFT"
fi
OUT=gen/${NAME}__${TAG}${SUF}.npz
[ -f runs/$OUT ] || env -u PYTORCH_CUDA_ALLOC_CONF CUDA_VISIBLE_DEVICES=$G LD_LIBRARY_PATH=$HL:/usr/local/cuda/lib64 $H -u head.py gen --ckpt $CKPT --corpus $CORPUS --out $OUT --samples 3 $VISARG
[ -f runs/eval_head_${NAME}__${TAG}${SUF}.json ] || CUDA_VISIBLE_DEVICES=$G $K -u eval_head.py --npz runs/$OUT --name ${NAME}__${TAG}${SUF}
if [ "$NAME" != oracle ] && [ "$KM" = kimodo ] && [ ! -f runs/eval_kimodo_$NAME.json ]; then
  CUDA_VISIBLE_DEVICES=$G $K -u eval_kimodo.py --name $NAME --stage encode; CUDA_VISIBLE_DEVICES=$G $K -u eval_kimodo.py --name $NAME; fi
echo "done $NAME $TAG$SUF"
