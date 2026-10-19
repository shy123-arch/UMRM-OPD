#!/bin/bash
# Download the release from ModelScope into DEST and place the HERO ViMo weights where the action head expects them.
# Usage: NS=<namespace> DEST=/data/umrm_vho bash tools/fetch.sh     (needs `modelscope login` for the private repos)
set -e; NS=${NS:?set NS}; DEST=${DEST:?set DEST}; PY=${VHO_PY_QWEN:-python}
$PY - <<PY
from modelscope import dataset_snapshot_download, snapshot_download
dataset_snapshot_download("$NS/umrm-vho-data-v1197aug", local_dir="$DEST/umrm_vho_data_v1197aug")
snapshot_download("$NS/umrm-vho-ckpt", local_dir="$DEST/umrm_vho_ckpt")
snapshot_download("Qwen/Qwen3-VL-4B-Instruct", local_dir="$DEST/Qwen3-VL-4B-Instruct")
PY
echo "VHO_DATA=$DEST/umrm_vho_data_v1197aug  VHO_CKPT=$DEST/umrm_vho_ckpt  VHO_QWEN=$DEST/Qwen3-VL-4B-Instruct"
if [ -n "$VHO_HERO" ]; then mkdir -p $VHO_HERO/checkpoints/vimo && cp -rn $DEST/umrm_vho_ckpt/hero_vimo/* $VHO_HERO/checkpoints/vimo/ && echo "HERO ViMo weights -> $VHO_HERO/checkpoints/vimo"; fi
