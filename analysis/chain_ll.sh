#!/bin/bash
# last-layer experiment chain (README §2). GPUs: GA=2 (stu cache -> S36 pretrain), GB=4 (A36 pretrain); later phases on both.
cd "$(dirname "$0")"; GA=2; GB=4
H=/mnt/workspace/Wilson/miniconda3/envs/hero/bin/python; HL=/mnt/workspace/Wilson/miniconda3/envs/hero/lib/python3.10/site-packages/torch/lib
K=/mnt/workspace/Wilson/miniconda3/envs/kimodo/bin/python; PRE4=$(realpath ../vt_head_opd_v860/runs/head_pre/best.pt)
hero() { g=$1; shift; env -u PYTORCH_CUDA_ALLOC_CONF CUDA_VISIBLE_DEVICES=$g LD_LIBRARY_PATH=$HL:/usr/local/cuda/lib64 $H -u head.py "$@"; }
log() { echo "$(date) $*"; }
FT="--full-event --steps 3000 --eval-every 100 --lr 1e-4 --warmup 200"
while [ ! -f logs/cache_stu.done ]; do sleep 60; done; log "student caches done"
[ -f runs/head_pre_S36/result.json ] || hero $GA pretrain --out head_pre_S36 --layers 36 --text-prefix stu_ --steps 60000 --eval-every 2000 > logs/head_pre_S36.log 2>&1 &
PS=$!; echo "head_pre_S36=$GA $(date)" >> runs/gpus_picked.txt
# A4 fine-tune can start now (its pretraining already exists) on GB alongside A36 pretraining
[ -f runs/head_A4/result.json ] || hero $GB finetune --init $PRE4 --out head_A4 --layers 9,18,27,36 $FT > logs/head_A4.log 2>&1; log "A4 finetune done $(tr -d '\n ' < runs/head_A4/result.json | cut -c 1-90)"
while [ ! -f runs/head_pre_A36/result.json ]; do sleep 120; done; log "A36 pretrain done $(tr -d '\n ' < runs/head_pre_A36/result.json | cut -c 1-90)"
[ -f runs/head_A36/result.json ] || hero $GB finetune --init runs/head_pre_A36/best.pt --out head_A36 --layers 36 $FT > logs/head_A36.log 2>&1; log "A36 finetune done"
wait $PS; log "S36 pretrain done $(tr -d '\n ' < runs/head_pre_S36/result.json | cut -c 1-90)"
[ -f runs/head_S36/result.json ] || hero $GA finetune --init runs/head_pre_S36/best.pt --out head_S36 --layers 36 --text-prefix stu_ $FT > logs/head_S36.log 2>&1 &
F1=$!
[ -f runs/head_V36/result.json ] || hero $GB finetune --init runs/head_pre_S36/best.pt --out head_V36 --layers 36 --cond-vis gt_trainval $FT > logs/head_V36.log 2>&1
wait $F1; log "S36 / V36 finetune done"
# HumanML3D self-check (same 1024-item test subset, same code) for the three pretrained heads
for x in A4:$PRE4 A36:runs/head_pre_A36/best.pt S36:runs/head_pre_S36/best.pt; do n=${x%%:*}; c=${x#*:}
  [ -f runs/hml3d_gen_$n.npz ] || hero $GA hml3d_gen --ckpt $c --out hml3d_gen_$n.npz --n 1024 > logs/hml3d_gen_$n.log 2>&1
  [ -f runs/hml3d_rprec_$n.json ] || CUDA_VISIBLE_DEVICES=$GA $K -u hml3d_rprec.py --npz runs/hml3d_gen_$n.npz --name $n > logs/hml3d_rprec_$n.log 2>&1
done; log "HumanML3D self-check done"
# generation (VAL, 3 samples) + independent evaluator; SFT predictions (pre-registered) and oracle inputs (exploratory)
gen() { n=$1; ck=$2; shift 2; [ -f runs/gen/$n.npz ] || hero $GB gen --ckpt runs/$ck/best.pt --out gen/$n.npz --samples 3 "$@" > logs/gen_$n.log 2>&1
        [ -f runs/eval_head_$n.json ] || CUDA_VISIBLE_DEVICES=$GB $K -u eval_head.py --npz $PWD/runs/gen/$n.npz --name $n > logs/eval_$n.log 2>&1; log "eval $n done"; }
gen sft__A4  head_A4  --corpus pred_sft_student_greedy
gen sft__A36 head_A36 --corpus pred_sft_student_greedy
gen sft__S36 head_S36 --corpus stu_pred_sft_student_greedy
gen sft__V36 head_V36 --corpus pred_sft_student_greedy --cond-vis-corpus pred_sft_student_greedy
gen oracle__A4  head_A4  --corpus pred_oracle_gt3p
gen oracle__A36 head_A36 --corpus pred_oracle_gt3p
gen oracle__S36 head_S36 --corpus stu_oracle_gt3p_val
gen oracle__V36 head_V36 --corpus pred_oracle_gt3p --cond-vis-corpus pred_oracle_gt3p
/usr/bin/python3 decide_ll.py > logs/decide_ll.log 2>&1; log "decisions written: runs/gates_ll.json"; touch logs/ll.done
