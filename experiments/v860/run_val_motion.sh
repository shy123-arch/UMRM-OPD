#!/bin/bash
# VAL motion evaluation of one V->T prediction file (or the gt3p oracle) through BOTH executors:
#   own head (textenc pred -> head.py gen -> eval_head.py)  and  frozen Kimodo (eval_kimodo.py encode/eval, arms T/TPH).
# Usage: run_val_motion.sh NAME HEAD_CKPT GPU [skip_kimodo]      NAME=oracle uses gt3p + GT plan-label durations.
set -e
cd "$(dirname "$0")"; NAME=$1; CKPT=$2; G=$3
Q=/mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python; K=/mnt/workspace/Wilson/miniconda3/envs/kimodo/bin/python
H=/mnt/workspace/Wilson/miniconda3/envs/hero/bin/python; HL=/mnt/workspace/Wilson/miniconda3/envs/hero/lib/python3.10/site-packages/torch/lib
TAG=$(basename $(dirname $CKPT))_$(basename $CKPT .pt)
if [ "$NAME" = oracle ]; then CORPUS=oracle_gt3p_val; [ -f /tmp/vho_cache/textfeat/$CORPUS.meta.json ] || CUDA_VISIBLE_DEVICES=$G $Q -u textenc.py --corpus oracle
else CORPUS=pred_$NAME; [ -f /tmp/vho_cache/textfeat/$CORPUS.meta.json ] || CUDA_VISIBLE_DEVICES=$G $Q -u textenc.py --corpus pred --name $NAME; fi
OUT=gen/${NAME}__${TAG}.npz
[ -f runs/$OUT ] || CUDA_VISIBLE_DEVICES=$G LD_LIBRARY_PATH=$HL:/usr/local/cuda/lib64 $H -u head.py gen --ckpt $CKPT --corpus $CORPUS --out $OUT --samples 3
[ -f runs/eval_head_${NAME}__${TAG}.json ] || CUDA_VISIBLE_DEVICES=$G $K -u eval_head.py --npz runs/$OUT --name ${NAME}__${TAG}
if [ "$NAME" != oracle ] && [ -z "$4" ] && [ ! -f runs/eval_kimodo_$NAME.json ]; then
  CUDA_VISIBLE_DEVICES=$G $K -u eval_kimodo.py --name $NAME --stage encode; CUDA_VISIBLE_DEVICES=$G $K -u eval_kimodo.py --name $NAME; fi
echo "done $NAME $TAG"
