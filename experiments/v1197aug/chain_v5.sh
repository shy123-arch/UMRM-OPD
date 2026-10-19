#!/bin/bash
# OPD v5 (README §5): teachers R-full / R-obs -> diagnostics -> 3 OPD arms -> evals -> decisions.
cd "$(dirname "$0")"; Q=/mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python; K=/mnt/workspace/Wilson/miniconda3/envs/kimodo/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date) $*"; }
inuse() { cat runs/opd_inuse/* 2>/dev/null | cut -d: -f2 | tr ' ' '\n' | grep -E "^[0-9]+$" | tr '\n' ',' | sed 's/,$//'; }
pick1() { python3 -c "import common as C; print(C.pick_gpus(1, $1, exclude=tuple(int(x) for x in '$(inuse)'.split(',') if x))[0])"; }
sel() { python3 -c "import json;print(json.load(open('runs/$1/selection.json'))['selected'])"; }
COMMON="--student sft_student --epochs 1 --k 8 --lr 5e-5 --accum 4 --fwd-chunk 4 --gen-videos 2 --skip-e0"
# 1. teachers in parallel
G1=$(python3 -c "import common as C; print(' '.join(map(str, C.pick_gpus(2, 24))))"); GA=${G1%% *}; GB=${G1##* }; log "teacher SFT GPUs $GA $GB"
echo "v5 teacher SFT r=$GA o=$GB $(date)" >> runs/gpus_picked.txt
[ -f runs/sft_teacher_r/selection.json ] || VHO_ZMODE=mr CUDA_VISIBLE_DEVICES=$GA $Q -u sft.py --mode teacher --out sft_teacher_r > logs/sft_teacher_r.log 2>&1 &
P1=$!
[ -f runs/sft_teacher_o/selection.json ] || VHO_ZMODE=mr_obs CUDA_VISIBLE_DEVICES=$GB $Q -u sft.py --mode teacher --out sft_teacher_o > logs/sft_teacher_o.log 2>&1 &
P2=$!; wait $P1 $P2; log "teachers done r=$(sel sft_teacher_r) o=$(sel sft_teacher_o)"
# 2. OPD arms (primary first; second starts once the first has its GPUs)
bash opd_runner.sh opd_v5o mr_obs --out opd_v5o --teacher sft_teacher_o --lam_fwd 0.0 --mr-filter obs $COMMON > logs/runner_opd_v5o.log 2>&1 &
R1=$!; sleep 600
bash opd_runner.sh opd_v5r mr --out opd_v5r --teacher sft_teacher_r --lam_fwd 0.0 --mr-filter human $COMMON > logs/runner_opd_v5r.log 2>&1 &
R2=$!
# 3. teacher diagnostics on a free GPU meanwhile
G=$(pick1 24); log "teacher diagnostics on GPU $G"
for t in r:mr o:mr_obs; do n=${t%%:*}; z=${t#*:}
  [ -f runs/sft_teacher_${n}_zgreedy.shard0.jsonl ] && [ $(wc -l < runs/sft_teacher_${n}_zgreedy.shard0.jsonl) -ge 125 ] || \
    VHO_ZMODE=$z CUDA_VISIBLE_DEVICES=$G $Q -u infer.py --adapter runs/sft_teacher_$n/$(sel sft_teacher_$n) --prompt teacher --name sft_teacher_${n}_zgreedy > logs/infer_teacher_$n.log 2>&1
  [ -f runs/gate_GT2_opd_v5$n.json ] || VHO_ZMODE=$z CUDA_VISIBLE_DEVICES=$G $Q -u opd2.py --out opd_v5$n --teacher sft_teacher_$n --gate --k 4 > logs/gate_GT5$n.log 2>&1
  [ -f runs/gate_GT5${n}_text.json ] || CUDA_VISIBLE_DEVICES=$G $K -u gate_text.py --a sft_teacher_${n}_zgreedy --b sft_student_greedy --out gate_GT5${n}_text > logs/gate_GT5${n}_text.log 2>&1
done; log "teacher diagnostics done"
bash run_val_motion.sh sft_teacher_o_zgreedy runs/head_AB_full/best.pt $G skip_kimodo > logs/val_teacher_o_AB.log 2>&1; log "teacher-o AB eval done"
# 4. third arm after the primary finishes
wait $R1; log "opd_v5o trained"
bash opd_runner.sh opd_v5o_mix mr_obs --out opd_v5o_mix --teacher sft_teacher_o --lam_fwd 0.5 --mr-filter obs $COMMON > logs/runner_opd_v5o_mix.log 2>&1 &
R3=$!
evalarm() { n=$1; while [ ! -f runs/${n}_e1_greedy.shard0.jsonl ] || [ $(wc -l < runs/${n}_e1_greedy.shard0.jsonl) -lt 125 ]; do sleep 60; done
  G=$(pick1 20); log "evals $n on GPU $G"; echo "v5 eval $n=$G $(date)" >> runs/gpus_picked.txt
  bash run_val_motion.sh ${n}_e1_greedy runs/head_AB_full/best.pt $G skip_kimodo > logs/val_${n}_AB.log 2>&1
  bash run_val_motion.sh ${n}_e1_greedy runs/head_A_full/best.pt $G kimodo > logs/val_${n}_A.log 2>&1; log "evals $n done"; }
evalarm opd_v5o
wait $R2; evalarm opd_v5r
wait $R3; evalarm opd_v5o_mix
/usr/bin/python3 decide_v5.py > logs/decide_v5.log 2>&1; log "decisions: runs/gates_v5.json"; touch logs/v5.done
