#!/bin/bash
# usage: run_ego_pt.sh "GPU_A GPU_B GPU_C" PORT <ego_pt.py args>   (one process per GPU; logs in logs/<out>_rank*.log)
cd "$(dirname "$0")"; GPUS=($1); PORT=$2; shift 2
Q=/mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python; OUT=$(echo "$@" | sed -E 's/.*--out ([^ ]+).*/\1/')
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT WORLD_SIZE=${#GPUS[@]} VHO_CACHE=/tmp/vho_ego_cache
pids=()
for r in "${!GPUS[@]}"; do CUDA_VISIBLE_DEVICES=${GPUS[$r]} RANK=$r $Q -u ego_pt.py "$@" > logs/${OUT}_rank$r.log 2>&1 & pids+=($!); done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; exit $rc
