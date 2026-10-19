#!/bin/bash
# README §9: build the full ego video cache (CPU). probe part with 2 workers and train part with 6 workers, in parallel.
cd "$(dirname "$0")"; Q=/mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python
export VHO_CACHE=/tmp/vho_ego_cache CUDA_VISIBLE_DEVICES=""
$Q -u ego_cache.py --part probe --workers 2 > logs/ego_cache_probe.log 2>&1 &
P1=$!
$Q -u ego_cache.py --part train --workers 6 > logs/ego_cache_train.log 2>&1
wait $P1; echo "$(date) cache done" ; touch logs/ego_cache.done
