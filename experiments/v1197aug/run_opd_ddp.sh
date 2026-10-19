#!/bin/bash
# usage: run_opd_ddp.sh "GPU_A GPU_B" PORT <opd2.py args...>   (one process per GPU; rank logs in logs/<out>_rank*.log)
cd "$(dirname "$0")"; GPUS=($1); PORT=$2; shift 2
Q=/mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python; OUT=$(echo "$@" | sed -E 's/.*--out ([^ ]+).*/\1/')
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT WORLD_SIZE=${#GPUS[@]}
pids=()
for r in "${!GPUS[@]}"; do CUDA_VISIBLE_DEVICES=${GPUS[$r]} RANK=$r $Q -u opd2.py "$@" > logs/${OUT}_rank$r.log 2>&1 & pids+=($!); done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; exit $rc
