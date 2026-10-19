#!/bin/bash
# End-to-end v1197aug pipeline from the released data package (docs/GUIDE.md explains every stage).
# Usage:  GPU=0 [OPD_GPUS="0 1"] [PRETRAIN_HEAD=1] [EVAL_KIMODO=1] bash pipeline/train.sh [prep] [sft] [head] [opd] [opd8b] [eval]
#   prep : video tensor cache (16 frames, mirrors flipped) + frozen-Qwen text features of the gt3p captions
#   sft  : LoRA SFT of the student (video only) and of the privileged teacher (video + kinematic hint + inner reasoning)
#   head : action head = HumanML3D-pretrained (released head_pre.pt, or PRETRAIN_HEAD=1 to pretrain) -> PICO fine-tune,
#          A_full (text interface A) and AB_full (A + zero-gated visual branch B; needs the student's hidden states)
#   opd  : on-policy self-distillation (K=8, 1 epoch, 0.5 reverse + 0.5 forward KL on <reaction>/<exec> tokens)
#   opd8b: OPD v6 = NO privileged info, Qwen3-VL-8B capability teacher (LoRA SFT with the student prompt), pure reverse
#          KL on every response token (opt-in stage; needs ~2x45 GB with 2 GPUs). docs/prereg/v1197aug.md §6
#   eval : VAL greedy of SFT and OPD -> own heads (and frozen Kimodo if EVAL_KIMODO=1) -> independent evaluator
# Outputs: $VHO_RUNS. Every stage skips work whose outputs already exist (rerun = resume).
set -e; cd "$(dirname "$0")"; source ./_env.sh
G=${GPU:-0}; OG=${OPD_GPUS:-$G}; ST=" ${*:-prep sft head opd eval} "; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
has() { [[ "$ST" == *" $1 "* ]]; }; log() { echo "$(date '+%F %T') $*"; }
sel() { python3 -c "import json;print(json.load(open('$RUNSDIR/$1/selection.json'))['selected'])"; }
run() { CUDA_VISIBLE_DEVICES=$G $Q -u "$@"; }
if has prep; then
  log prep; $Q -u prep_video_cache.py --workers ${WORKERS:-6}
  [ -f $C/textfeat/gt3p_trainval.meta.json ] || run textenc.py --corpus gt3p
fi
if has sft; then
  [ -f $RUNSDIR/sft_student/selection.json ] || { log "SFT student"; run sft.py --mode student --out sft_student; }
  [ -f $RUNSDIR/sft_teacher/selection.json ] || { log "SFT teacher"; run sft.py --mode teacher --out sft_teacher; }
fi
SS=$RUNSDIR/sft_student/$(sel sft_student 2>/dev/null || echo epoch1)
if has head; then
  HEAD_PRE=${HEAD_PRE:-$VHO_CKPT/head_pre.pt}
  if [ -n "$PRETRAIN_HEAD" ]; then
    $Q -u pack_hml3d.py; [ -f $C/textfeat/hml3d_test.meta.json ] || run textenc.py --corpus hml3d
    [ -f $RUNSDIR/head_pre/result.json ] || { log "head pretrain (HumanML3D, ~2 GPU-h)"; hero $G head.py pretrain --out head_pre --steps 60000 --eval-every 2000; }
    HEAD_PRE=$RUNSDIR/head_pre/best.pt
  fi
  [ -f $C/visfeat/gt_trainval.meta.json ] || run vis_feats.py --adapter $SS --source gt
  for v in A AB; do VIS=""; [ $v = AB ] && VIS="--vis gt_trainval"
    [ -f $RUNSDIR/head_${v}_full/result.json ] || { log "head $v"; hero $G head.py finetune --init $HEAD_PRE --out head_${v}_full --steps 3000 \
      --eval-every 100 --lr 1e-4 --warmup 200 --full-event $VIS; }
  done
fi
OPDARGS="--out opd --student sft_student --teacher sft_teacher --epochs 1 --k 8 --lr 5e-5 --accum 4 --lam_fwd 0.5 --half-ckpt --fwd-chunk 4 --gen-videos 2 --skip-e0"
if has opd && [ ! -f $RUNSDIR/opd/selection.json ]; then
  log "OPD on GPUs $OG"; RES=""; [ -f $RUNSDIR/opd/resume.pt ] && RES="--resume"
  if [ $(echo $OG | wc -w) -gt 1 ]; then bash run_opd_ddp.sh "$OG" ${PORT:-29651} $OPDARGS $RES
  else CUDA_VISIBLE_DEVICES=$OG $Q -u opd.py $OPDARGS $RES; fi
fi
if has opd8b; then
  [ -f $RUNSDIR/sft_teacher_8b/selection.json ] || { log "SFT 8B capability teacher (same prompt as the student)"; VHO_ZMODE=none run sft.py --mode student --base 8b --out sft_teacher_8b; }
  [ -f $RUNSDIR/sft_teacher_8b_greedy.shard0.jsonl ] || VHO_ZMODE=none run infer.py --adapter $RUNSDIR/sft_teacher_8b/$(sel sft_teacher_8b) --base 8b --name sft_teacher_8b_greedy
  O8="--out opd_v6 --student sft_student --teacher sft_teacher_8b --teacher-base 8b --epochs 1 --k 8 --lr 5e-5 --accum 4 --lam_fwd 0.0 --kl-span all --mr-filter none --fwd-chunk 4 --gen-videos 2 --skip-e0"
  if [ ! -f $RUNSDIR/opd_v6/selection.json ]; then
    log "OPD v6 (8B teacher, no z, pure reverse KL) on GPUs $OG"; RES=""; [ -f $RUNSDIR/opd_v6/resume.pt ] && RES="--resume"
    if [ $(echo $OG | wc -w) -gt 1 ]; then VHO_ZMODE=none bash run_opd_ddp.sh "$OG" ${PORT:-29661} $O8 $RES
    else VHO_ZMODE=none CUDA_VISIBLE_DEVICES=$OG $Q -u opd.py $O8 $RES; fi
  fi
fi
if has eval; then
  KM=skip_kimodo; [ -n "$EVAL_KIMODO" ] && KM=kimodo
  [ -f $RUNSDIR/sft_student_greedy.shard0.jsonl ] || run infer.py --adapter $SS --name sft_student_greedy
  for n in sft_student_greedy opd_e1_greedy sft_teacher_8b_greedy opd_v6_e1_greedy; do
    [ -f $RUNSDIR/$n.shard0.jsonl ] || continue
    bash run_val_motion.sh $n $RUNSDIR/head_AB_full/best.pt $G skip_kimodo
    bash run_val_motion.sh $n $RUNSDIR/head_A_full/best.pt $G $KM
  done
  log "eval done; compare with: $Q compare.py \$VHO_RUNS/eval_head_opd_e1_greedy__head_AB_full_best.json:H \$VHO_RUNS/eval_head_sft_student_greedy__head_AB_full_best.json:H"
fi
