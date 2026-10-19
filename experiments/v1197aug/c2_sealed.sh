#!/bin/bash
# Pre-registered confirmation C2 (README §2.2): ORIGINAL v1197 models (SFT student e2, OPD v2 e2) through the ORIGINAL v1197
# AB_full head (+ A_full, Kimodo as secondary) on the 50-item new-source SEALED block, unsealed here for the first time.
set -e
cd "$(dirname "$0")"; G=$1
export VHO_UNSEAL_SEALED=1 VHO_SPLIT=sealed
Q=/mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python; K=/mnt/workspace/Wilson/miniconda3/envs/kimodo/bin/python
H=/mnt/workspace/Wilson/miniconda3/envs/hero/bin/python; HL=/mnt/workspace/Wilson/miniconda3/envs/hero/lib/python3.10/site-packages/torch/lib
V=$(realpath ../vt_head_opd_v1197/runs); SS=$V/sft_student/epoch2; OPD=$V/opd_v2/epoch2
log() { echo "$(date) $*"; }
$Q -u prep_video_cache.py --workers 6 > logs/c2_prep_video.log 2>&1; log "sealed video cache done"
CUDA_VISIBLE_DEVICES=$G $Q -u infer.py --adapter $SS --name c2_sft > logs/c2_infer_sft.log 2>&1
CUDA_VISIBLE_DEVICES=$G $Q -u infer.py --adapter $OPD --name c2_opd > logs/c2_infer_opd.log 2>&1; log "sealed greedy done"
for n in c2_sft c2_opd; do
  CUDA_VISIBLE_DEVICES=$G $Q -u textenc.py --corpus pred --name $n > logs/c2_textenc_$n.log 2>&1
  CUDA_VISIBLE_DEVICES=$G $Q -u vis_feats.py --adapter $SS --source pred --name $n > logs/c2_vis_$n.log 2>&1
  for h in AB_full A_full; do
    VIS=""; [ $h = AB_full ] && VIS="--vis-corpus pred_$n"
    CUDA_VISIBLE_DEVICES=$G LD_LIBRARY_PATH=$HL:/usr/local/cuda/lib64 $H -u head.py gen --ckpt $V/head_$h/best.pt --corpus pred_$n $VIS --out gen/${n}__head_${h}_best.npz --samples 3 > logs/c2_gen_${n}_$h.log 2>&1
    CUDA_VISIBLE_DEVICES=$G $K -u eval_head.py --npz runs/gen/${n}__head_${h}_best.npz --name ${n}__head_${h}_best > logs/c2_evalh_${n}_$h.log 2>&1
  done; log "$n own-head evals done"
done
for n in c2_sft c2_opd; do CUDA_VISIBLE_DEVICES=$G $K -u eval_kimodo.py --name $n --stage encode > logs/c2_evalk_$n.log 2>&1; CUDA_VISIBLE_DEVICES=$G $K -u eval_kimodo.py --name $n >> logs/c2_evalk_$n.log 2>&1; done
log "Kimodo evals done"; touch logs/c2.done
