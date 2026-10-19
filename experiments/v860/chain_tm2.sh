#!/bin/bash
# T->M chain (GPU of head_pre): wait head_pre -> G-B HumanML3D self-check -> v860 finetune -> gt3p oracle via own head
# -> own-head eval of every V->T prediction file that appears. Logs: logs/chain_tm.log
cd "$(dirname "$0")"; K=/mnt/workspace/Wilson/miniconda3/envs/kimodo/bin/python
H=/mnt/workspace/Wilson/miniconda3/envs/hero/bin/python; HL=/mnt/workspace/Wilson/miniconda3/envs/hero/lib/python3.10/site-packages/torch/lib
G=$(sed -n 1p runs/gpus_picked.txt | cut -d, -f2)
hero() { CUDA_VISIBLE_DEVICES=$G LD_LIBRARY_PATH=$HL:/usr/local/cuda/lib64 $H -u head.py "$@"; }
echo "$(date) chain_tm2 start; head_pre $(tr -d "\n " < runs/head_pre/result.json)"
hero hml3d_gen --ckpt runs/head_pre/best.pt --out hml3d_gen_head_pre.npz --n 1024 > logs/hml3d_gen_pre.log 2>&1
CUDA_VISIBLE_DEVICES=$G $K -u hml3d_rprec.py --npz runs/hml3d_gen_head_pre.npz --name head_pre > logs/hml3d_rprec_pre.log 2>&1; echo "$(date) G-B: $(grep -E 'R@1/2/3|ratio' logs/hml3d_rprec_pre.log | tr -d '\n ')"
hero finetune --init runs/head_pre/best.pt --out head_ft --steps 6000 --eval-every 250 --lr 1e-4 --warmup 200 > logs/head_ft.log 2>&1; echo "$(date) head_ft done $(tr -d '\n ' < runs/head_ft/result.json)"
bash run_val_motion.sh oracle runs/head_ft/best.pt $G > logs/val_oracle_head_ft.log 2>&1; echo "$(date) oracle via own head done"
bash run_val_motion.sh oracle runs/head_pre/best.pt $G > logs/val_oracle_head_pre.log 2>&1; echo "$(date) oracle via pretrain-only head done"
for n in sft4b_student_greedy opd_v1_e1_greedy opd_v1_e2_greedy; do
  while [ ! -f runs/$n.shard0.jsonl ] || [ $(wc -l < runs/$n.shard0.jsonl) -lt 96 ]; do sleep 120; [ -f logs/chain_vt.done ] && break; done
  [ -f runs/$n.shard0.jsonl ] && bash run_val_motion.sh $n runs/head_ft/best.pt $G skip_kimodo > logs/val_${n}_head_ft.log 2>&1 && echo "$(date) own-head eval $n done"
done
echo "$(date) chain_tm finished"
