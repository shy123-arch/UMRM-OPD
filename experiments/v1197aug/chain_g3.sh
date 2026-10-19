#!/bin/bash
# GPU 3: teacher v3 SFT (after generated MR exists)  ||  C1 = OPD v2 seed 1 in the v1197 experiment (identical config) -> its evals.
cd "$(dirname "$0")"; G=3; Q=/mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python
log() { echo "$(date) $*"; }
while [ ! -f runs/mr_generated.json ]; do sleep 60; done; log "MR generated"
CUDA_VISIBLE_DEVICES=$G nohup $Q -u sft.py --mode teacher --out sft_teacher_v3 > logs/sft_teacher_v3.log 2>&1 &
TP=$!
( cd ../vt_head_opd_v1197 && CUDA_VISIBLE_DEVICES=$G $Q -u opd2.py --out opd_v2_s1 --epochs 2 --k 4 --lr 5e-5 --accum 4 --lam_fwd 0.5 --seed 1 > logs/opd_v2_s1.log 2>&1 ); log "C1 OPD v2 seed1 done"
( cd ../vt_head_opd_v1197 && for h in head_AB_full head_A_full; do KM=skip_kimodo; [ $h = head_A_full ] && KM=kimodo
    bash run_val_motion.sh opd_v2_s1_e2_greedy runs/$h/best.pt $G $KM > logs/val_opd_s1_e2_$h.log 2>&1; done ); log "C1 evals done"
wait $TP; log "teacher v3 SFT done"; touch logs/g3.done
