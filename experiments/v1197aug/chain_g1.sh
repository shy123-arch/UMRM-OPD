#!/bin/bash
# GPU 1: student SFT -> VAL greedy -> gt3p text feats + GT vis feats -> heads A_full / AB_full -> (teacher ready) teacher z-greedy,
# G-T3 gates -> OPD v3 (K=8, 1 epoch) -> evals (SFT and OPD v3 through AB_full [primary], A_full, Kimodo; mismatched-video
# control; oracle; teacher diagnostic; train mean).
cd "$(dirname "$0")"; G=1; Q=/mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python; K=/mnt/workspace/Wilson/miniconda3/envs/kimodo/bin/python
H=/mnt/workspace/Wilson/miniconda3/envs/hero/bin/python; HL=/mnt/workspace/Wilson/miniconda3/envs/hero/lib/python3.10/site-packages/torch/lib
PRE=$(realpath ../vt_head_opd_v860/runs/head_pre/best.pt)
log() { echo "$(date) $*"; }
sel() { python3 -c "import json;print(json.load(open('runs/$1/selection.json'))['selected'])"; }
while [ ! -f runs/sft_student/selection.json ]; do sleep 60; done; SS=runs/sft_student/$(sel sft_student); log "student selected $SS"
CUDA_VISIBLE_DEVICES=$G $Q -u infer.py --adapter $SS --name sft_student_greedy > logs/infer_sft_student_greedy.log 2>&1; log "student greedy done"
CUDA_VISIBLE_DEVICES=$G $Q -u textenc.py --corpus gt3p > logs/textenc_gt3p.log 2>&1
CUDA_VISIBLE_DEVICES=$G $Q -u vis_feats.py --adapter $SS --source gt > logs/visfeat_gt.log 2>&1; log "text + vis feats done"
for v in A AB; do VIS=""; [ $v = AB ] && VIS="--vis gt_trainval"
  CUDA_VISIBLE_DEVICES=$G LD_LIBRARY_PATH=$HL:/usr/local/cuda/lib64 nohup $H -u head.py finetune --init $PRE --out head_${v}_full --steps 3000 --eval-every 100 \
    --lr 1e-4 --warmup 200 --full-event $VIS > logs/head_${v}_full.log 2>&1 &
done
while [ ! -f runs/sft_teacher_v3/selection.json ]; do sleep 60; done; ST=runs/sft_teacher_v3/$(sel sft_teacher_v3); log "teacher selected $ST"
CUDA_VISIBLE_DEVICES=$G $Q -u infer.py --adapter $ST --prompt teacher --name sft_teacher_v3_zgreedy > logs/infer_teacher_zgreedy.log 2>&1; log "teacher greedy done"
CUDA_VISIBLE_DEVICES=$G $Q -u opd2.py --out opd_v3 --student sft_student --teacher sft_teacher_v3 --gate --k 4 > logs/gate_GT3.log 2>&1
CUDA_VISIBLE_DEVICES=$G $K -u gate_text.py --a sft_teacher_v3_zgreedy --b sft_student_greedy --out gate_GT3_text > logs/gate_GT3_text.log 2>&1; log "G-T3 done"
CUDA_VISIBLE_DEVICES=$G $Q -u opd2.py --out opd_v3 --student sft_student --teacher sft_teacher_v3 --epochs 1 --k 8 --lr 5e-5 --accum 4 --lam_fwd 0.5 --half-ckpt > logs/opd_v3.log 2>&1 &
OP=$!
while [ ! -f runs/head_AB_full/result.json ] || [ ! -f runs/head_A_full/result.json ]; do sleep 60; done; log "heads done"
bash run_val_motion.sh sft_student_greedy runs/head_AB_full/best.pt $G skip_kimodo > logs/val_sft_AB.log 2>&1
bash run_val_motion.sh sft_student_greedy runs/head_AB_full/best.pt $G skip_kimodo 1 > logs/val_sft_AB_shift.log 2>&1
bash run_val_motion.sh sft_student_greedy runs/head_A_full/best.pt $G kimodo > logs/val_sft_A.log 2>&1
bash run_val_motion.sh oracle_gt3p runs/head_AB_full/best.pt $G skip_kimodo > logs/val_oracle_AB.log 2>&1
bash run_val_motion.sh sft_teacher_v3_zgreedy runs/head_AB_full/best.pt $G skip_kimodo > logs/val_teacher_AB.log 2>&1
CUDA_VISIBLE_DEVICES=$G $K -u baseline_mean.py > logs/baseline_mean.log 2>&1; log "SFT-side evals done"
wait $OP; log "OPD v3 done"
for n in opd_v3_e1_greedy; do
  bash run_val_motion.sh $n runs/head_AB_full/best.pt $G skip_kimodo > logs/val_${n}_AB.log 2>&1
  bash run_val_motion.sh $n runs/head_AB_full/best.pt $G skip_kimodo 1 > logs/val_${n}_AB_shift.log 2>&1
  bash run_val_motion.sh $n runs/head_A_full/best.pt $G kimodo > logs/val_${n}_A.log 2>&1
done; log "OPD v3 evals done"; touch logs/g1.done
