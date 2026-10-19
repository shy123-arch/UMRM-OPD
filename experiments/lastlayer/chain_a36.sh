#!/bin/bash
# Exploratory (README §6 of vt_head_opd_v1197aug): every existing OPD / SFT VAL prediction through the A36 head.
cd "$(dirname "$0")"; G=$1; Q=/mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python; K=/mnt/workspace/Wilson/miniconda3/envs/kimodo/bin/python
H=/mnt/workspace/Wilson/miniconda3/envs/hero/bin/python; HL=/mnt/workspace/Wilson/miniconda3/envs/hero/lib/python3.10/site-packages/torch/lib
a36() { n=$1
  [ -f /tmp/vho1197aug_cache/textfeat/pred_$n.meta.json ] || CUDA_VISIBLE_DEVICES=$G $Q -u textenc.py --corpus pred --name $n > logs/a36_textenc_$n.log 2>&1
  [ -f runs/gen/${n}__A36.npz ] || env -u PYTORCH_CUDA_ALLOC_CONF CUDA_VISIBLE_DEVICES=$G LD_LIBRARY_PATH=$HL:/usr/local/cuda/lib64 $H -u head.py gen --ckpt runs/head_A36/best.pt --corpus pred_$n --out gen/${n}__A36.npz --samples 3 > logs/a36_gen_$n.log 2>&1
  [ -f runs/eval_head_${n}__A36.json ] || CUDA_VISIBLE_DEVICES=$G $K -u eval_head.py --npz $PWD/runs/gen/${n}__A36.npz --name ${n}__A36 > logs/a36_eval_$n.log 2>&1
  echo "$(date) $n done"; }
for n in ${@:2}; do a36 $n; done; touch logs/a36_${3:-x}.done
