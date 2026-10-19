# 冒烟测试记录（2026-10-16，原服务器，从全新 `git clone` 出发）

配置：`configs/server_wilson.env`，`VHO_LIMIT=16`（16 条 TRAIN / 4 条 VAL）。GPU 都与他人共享。

| 阶段 | 用时 | 产物 |
|---|---|---|
| prep（视频缓存 + gt3p 文本特征） | 12 s（缓存已存在）/ 约 1 min | `$VHO_CACHE/video`、`textfeat/gt3p_trainval` |
| sft（学生 + 教师，各 1 个 epoch） | 约 5 min | `sft_student/epoch1`、`sft_teacher/epoch1` |
| infer（VAL greedy） | 约 2 min | `sft_student_greedy.shard0.jsonl` |
| vis（学生隐状态，接口 B） | 约 30 s | `visfeat/gt_trainval` |
| head（A_full、AB_full 各 20 步） | 约 3 min | `head_A_full/best.pt`、`head_AB_full/best.pt` |
| opd（K=8，2 步，断点每步保存） | 约 5 min | `opd/epoch1`、`opd_e1_greedy.shard0.jsonl` |
| eval（自有头 + Kimodo + TMR） | 约 7 min | `eval_head_*.json`、`eval_kimodo_opd_e1_greedy.json` |

发现并修复的问题：
1. HERO 环境的 torch 1.12 不认识 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` → `_env.sh` 的 `hero()` 在调用前去掉该变量。
2. 在别人占约 42 GB 的卡上，冒烟版 OPD OOM → 冒烟版改为每次生成 1 个视频 × 8 个样本、前向每块 2 条（峰值约 28 GB）。

另外：仓库内评测器 + 发布包资产，复现 v1197 的 `SFT → 自有头 A` 结果，8 项指标差值全部为 0。
