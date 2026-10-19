#!/bin/bash
# v1197aug main chain (resumed 2026-10-16 ~14:40 after the SFT crash; deviations in README §3).
#   track OPD : OPD v3 (pre-registered config) data-parallel on GPUS_OPD, auto-resume on failure (<=5 retries)
#   track PREP: on GP: student greedy -> gt3p text feats -> GT vis feats -> heads A_full / AB_full -> teacher z-greedy -> G-T3
#               -> SFT-side evals (AB [primary], AB mismatched-video, A + Kimodo, oracle, teacher, train mean)
#   then      : OPD v3 final (+ exploratory half) evals on GP.
cd "$(dirname "$0")"; GPUS_OPD="1 7"; GP=0
Q=/mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python; K=/mnt/workspace/Wilson/miniconda3/envs/kimodo/bin/python
H=/mnt/workspace/Wilson/miniconda3/envs/hero/bin/python; HL=/mnt/workspace/Wilson/miniconda3/envs/hero/lib/python3.10/site-packages/torch/lib
PRE=$(realpath ../vt_head_opd_v860/runs/head_pre/best.pt); export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date) $*"; }
sel() { python3 -c "import json;print(json.load(open('runs/$1/selection.json'))['selected'])"; }
SS=runs/sft_student/$(sel sft_student); ST=runs/sft_teacher_v3/$(sel sft_teacher_v3); log "student $SS teacher $ST"
echo "v3 chain: OPD=$GPUS_OPD prep/eval=$GP $(date)" >> runs/gpus_picked.txt
OPDARGS="--out opd_v3 --student sft_student --teacher sft_teacher_v3 --epochs 1 --k 8 --lr 5e-5 --accum 4 --lam_fwd 0.5 --half-ckpt --fwd-chunk 4 --gen-videos 2 --skip-e0"
( n=0; bash run_opd_ddp.sh "$GPUS_OPD" 29641 $OPDARGS; rc=$?
  while [ ! -f runs/opd_v3/selection.json ] && [ $n -lt 5 ]; do n=$((n+1)); echo "$(date) OPD failed (rc=$rc), resume #$n"; sleep 60
    bash run_opd_ddp.sh "$GPUS_OPD" $((29641+n)) $OPDARGS --resume; rc=$?; done; echo "$(date) OPD track end" ) > logs/opd_v3_track.log 2>&1 &
OPDT=$!
CUDA_VISIBLE_DEVICES=$GP $Q -u infer.py --adapter $SS --name sft_student_greedy > logs/infer_sft_student_greedy.log 2>&1; log "student greedy done"
CUDA_VISIBLE_DEVICES=$GP $Q -u textenc.py --corpus gt3p > logs/textenc_gt3p.log 2>&1
CUDA_VISIBLE_DEVICES=$GP $Q -u vis_feats.py --adapter $SS --source gt > logs/visfeat_gt.log 2>&1; log "text + vis feats done"
for v in A AB; do VIS=""; [ $v = AB ] && VIS="--vis gt_trainval"
  CUDA_VISIBLE_DEVICES=$GP LD_LIBRARY_PATH=$HL:/usr/local/cuda/lib64 $H -u head.py finetune --init $PRE --out head_${v}_full --steps 3000 --eval-every 100 \
    --lr 1e-4 --warmup 200 --full-event $VIS > logs/head_${v}_full.log 2>&1 &
done
CUDA_VISIBLE_DEVICES=$GP $Q -u infer.py --adapter $ST --prompt teacher --name sft_teacher_v3_zgreedy > logs/infer_teacher_zgreedy.log 2>&1; log "teacher greedy done"
CUDA_VISIBLE_DEVICES=$GP $Q -u opd2.py --out opd_v3 --student sft_student --teacher sft_teacher_v3 --gate --k 4 > logs/gate_GT3.log 2>&1
CUDA_VISIBLE_DEVICES=$GP $K -u gate_text.py --a sft_teacher_v3_zgreedy --b sft_student_greedy --out gate_GT3_text > logs/gate_GT3_text.log 2>&1; log "G-T3 done"
while [ ! -f runs/head_AB_full/result.json ] || [ ! -f runs/head_A_full/result.json ]; do sleep 60; done; log "heads done"
bash run_val_motion.sh sft_student_greedy runs/head_AB_full/best.pt $GP skip_kimodo > logs/val_sft_AB.log 2>&1
bash run_val_motion.sh sft_student_greedy runs/head_AB_full/best.pt $GP skip_kimodo 1 > logs/val_sft_AB_shift.log 2>&1
bash run_val_motion.sh sft_student_greedy runs/head_A_full/best.pt $GP kimodo > logs/val_sft_A.log 2>&1
bash run_val_motion.sh oracle_gt3p runs/head_AB_full/best.pt $GP skip_kimodo > logs/val_oracle_AB.log 2>&1
bash run_val_motion.sh sft_teacher_v3_zgreedy runs/head_AB_full/best.pt $GP kimodo > logs/val_teacher_AB.log 2>&1
CUDA_VISIBLE_DEVICES=$GP $K -u baseline_mean.py > logs/baseline_mean.log 2>&1; log "SFT-side evals done"
while [ ! -f runs/opd_v3/half/adapter_config.json ]; do sleep 120; done
CUDA_VISIBLE_DEVICES=$GP $Q -u infer.py --adapter runs/opd_v3/half --name opd_v3_half_greedy > logs/infer_opd_half.log 2>&1
bash run_val_motion.sh opd_v3_half_greedy runs/head_AB_full/best.pt $GP kimodo > logs/val_opd_half_AB.log 2>&1; log "OPD half (exploratory) evals done"
wait $OPDT; log "OPD track finished: $(tail -1 logs/opd_v3_track.log)"
n=opd_v3_e1_greedy
bash run_val_motion.sh $n runs/head_AB_full/best.pt $GP skip_kimodo > logs/val_${n}_AB.log 2>&1
bash run_val_motion.sh $n runs/head_AB_full/best.pt $GP skip_kimodo 1 > logs/val_${n}_AB_shift.log 2>&1
bash run_val_motion.sh $n runs/head_A_full/best.pt $GP kimodo > logs/val_${n}_A.log 2>&1; log "OPD v3 evals done"; touch logs/v3.done
