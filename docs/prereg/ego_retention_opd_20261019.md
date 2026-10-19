# Ego 能力保留与固定教师匹配 OPD（ER19）

登记日期：2026-10-19。用户授权：先检查 SFT 能力保留，再做固定教师、同一配方的匹配 OPD。此登记早于 ER19 的任何 GPU 运行。既有 E 实验结果已知，本实验属于后续机制验证；不回写 G-E1 或改变正式 636 benchmark。

## 已有检查与待检验问题

代码核验：阶段 B 使用 `PeftModel.from_pretrained(..., is_trainable=True)` 延续 PT-C，语言侧 252 个线性模块的 LoRA 可训练，视觉塔冻结。阶段 B 新建 AdamW；lr=1e-4，3 epoch 余弦日程，完成 2 epoch 后按验证 NLL 选择 epoch1。没有 ego 回放。现有中间产物只有 epoch1/2，无法定位第一个 epoch 内何时回落，也不能将回落唯一归因于学习率。

已知行为：PT-C→SFT 的 EgoNormia 67→61.5%；actions 30.51→24.29%，后者配对 CI 为 [−9.35,−3.14] 个百分点。最终四项相对旧 SFT 的区间均跨零。不能称内部表征全部遗忘。

本轮只改变 SFT 学习率，增加中途诊断；不混入回放或新增教师，以保持可解释性。匹配 OPD 用来检验在同一 SFT 配方下，ego 初始化是否改变蒸馏收益。

## 固定输入与范围

- 4B 基座：`/tmp/vho_models/Qwen3-VL-4B-Instruct`；预训练初始化：原实验 `runs/pt_c/epoch1`。
- 数据：v1197aug，TRAIN 1,652，VAL 125；TEST 与 sealed 不读取。正式 benchmark 仍为 636。
- 所有视频采用已有 16 帧缓存：UMRM `/tmp/vho1197aug_cache`；ego `/tmp/vho_ego_cache`。复用现有缓存，不下载、搬运或重建视频。
- 外部探针固定 EgoNormia 200 + EgoToM 1,035；相同提示与选项评分。
- 结果以 `er19_` 为前缀写入原运行目录的新文件；旧检查点、原登记门和旧输出只读。控制记录位于 `runs/er19_control/`，独立日志 `logs/er19/`。

## 第一步：SFT 能力保留

| 臂 | 初始化 | SFT |
|---|---|---|
| A：`er19_sft_base` | 原版 4B，新建 r16 LoRA | lr=2e-5，accum8，seed0，3 epoch 余弦日程，完成 epoch1 停止 |
| C：`er19_sft_ego` | PT-C epoch1，延续同一个 LoRA | 与 A 完全相同 |

两臂都用三行 answer 目标、TRAIN 全量与相同顺序；视觉塔冻结、LoRA dropout0.05、weight_decay0、梯度裁剪1。保存第104次更新（约半 epoch）及 epoch1（207次更新）。第104步验证保存并恢复 RNG，避免诊断改变后续训练。第104步仅诊断；两臂均固定 epoch1 进入 OPD，不按探针挑检查点。

完整探针在两臂 step104 与 epoch1 上各跑一次；同时记录 UMRM VAL NLL。epoch1 均经过同一个 A36 头读数，三次动作采样。已有旧 SFT/PT-C/PT-C→旧 SFT 仅作历史描述性参照，不能将新旧差异全部归因于 ego 预训练。

保留分析报告 C−PT-C、C−旧 PT-C→SFT、C−A，按视频 UID 配对 bootstrap；同时报告各探针的正确数、修正数和退化数，不将单项提升替代任务质量。降低学习率能否保留能力是待测结果。未证明保留时，后续 OPD 仍按登记运行，但不能据其负结果否定“保留充分时更易蒸馏”的条件假设。

## 第二步：同一固定教师的匹配 OPD

| 初始化 | 无 OPD | 有 OPD |
|---|---|---|
| 原版→低学习率 SFT | A | B：`er19_opd_base` |
| PT-C→低学习率 SFT | C | D：`er19_opd_ego` |

- 固定教师：原实验 `sft_teacher_r/epoch1`，Qwen3-VL-4B + 已有 R-full 教师 LoRA。**不训练或更换教师**。`VHO_ZMODE=mr`，人工完整 mental_reasoning，无运动学特权、无生成 reason。
- TRAIN：相同 832 条带人工 reason 的真实及镜像样本；`--mr-filter human`。每个分支顺序/seed相同；on-policy 响应随学生而不同，这是方法定义。
- 两臂：1 epoch、208 次更新、K8、temperature1、top_p1、全词表纯反向 KL（`--lam_fwd 0`）；只蒸馏 reaction/exec，排除 plan/end（`--kl-span re`）。lr5e-5，accum4，5% warm-up+余弦，seed0。
- 每臂单卡，`--gen-videos 1 --fwd-chunk 2`，避免两卡/单卡随机数差异；每25步保存恢复状态。固定最后 epoch1，不按结果挑步数。与历史 v5r 的并行方式不同，因此重新训练 B，不直接复用历史 OPD 结果。
- 教师输入提示、权重与数据哈希在 preflight 固定。教师已有 VAL greedy 输出用 A36 补读数，报告其相对 A/C 的优势及人工 reason VAL 子集；教师优势不足会限制结论，但不事后改教师。
- B/D 均运行全部外部探针和相同 A36 VAL 评测。报告文本/计划准确率、方向/时长 TV、教师一致性、训练 KL、熵、格式率；KL 下降不能单独判为有效。

## 主要比较与统计

主终点：A36 R@10，先在每个样本内部平均三次生成，再按 VAL 的34个来源配对 bootstrap 4,000次。主比较为 `(D−C)−(B−A)`；同时报告 D−C、B−A、D−B 及四臂绝对值。只有交互差和 D−C 的95% CI下界均大于0，才支持“此配置下预训练增加了有用的 OPD 收益”。单训练 seed 的结果仍需跨 seed 复核。

外部探针按视频 UID 配对聚类。其他字段、子组与阶段性诊断为探索性，未多重校正。VAL 已用于开发；不把它视为全新测试集。

## 资源、后台与失败语义

入口：`tools/queue_retention_opd.py`。默认仅 preflight/dry-run；`--execute` 才运行 GPU。输出 exact command plan 和输入 SHA256，之后按阶段顺序执行。两个臂并行，每臂最多一张卡；共享原 `runs/.acquire.lock` 和 `runs/opd_inuse/`，连同已有登记作业总计不超过3张卡。失败即写失败状态并停止后续阶段，不记录虚假 done，不终止其他作业。

估算新增约20–30 GPU小时；共享机器下约10–18小时墙钟时间，排队另计。SFT 不支持中断续训：失败后保留输出，停止并报告，不自动覆盖或重复消耗预算。OPD 可从已有 `resume.pt` 恢复。缓存、既有预测或必需资产缺失时 preflight 失败，不自动下载。

2026-10-19 启动状态：GPU 查询需 sandbox escalation；自动审批服务返回403（Fast价格配置错误），审批未完成、查询未执行。因此当前登记和代码准备不表示已启动。恢复审批通道并通过GPU小样本检查后再正式执行。

### 可执行入口

仅检查（已完成，不使用 GPU）：

```bash
python /mnt/workspace/Wilson/ck/umrm/umrm-vho/tools/queue_retention_opd.py
```

正式后台入口（待审批通道恢复；会先执行16条SFT、4条K8 OPD的GPU小样本检查，再顺序执行上述两阶段）：

```bash
nohup /mnt/workspace/Wilson/miniconda3/envs/qwen35_532/bin/python -u \
  /mnt/workspace/Wilson/ck/umrm/umrm-vho/tools/queue_retention_opd.py --execute \
  > /mnt/workspace/Wilson/ck/umrm/umrm_model/experiments/vt_head_opd_v1197aug/logs/er19/queue.log 2>&1 < /dev/null &
```

本地状态：`runs/er19_control/status.json`；完成标志：`runs/er19_control/done.json`；阶段性统计：`runs/er19_control/analysis/REPORT.md`。没有 `done.json` 时不能称流程完成。

已通过：语法检查、依赖导入、全部输入与缓存存在性检查、release 与原运行目录的 split/target/人工 reason 一致性、CPU小模型的诊断前后权重逐位一致性与检查点选择隔离测试、聚类 bootstrap 与直接重采样实现的一致性测试。尚未执行：GPU小样本检查、训练、探针推理及A36读数。

## 发布时运行状态补记

2026-10-19 12:26审批通道恢复，原服务器后台队列已启动；12:33小样本SFT/OPD验证通过并进入两臂正式SFT。以上“审批阻塞”保留作历史记录，实验方法与门未改动。GitHub快照不包含实时状态；未宣布最终实验完成。当前运行路径仍是原`umrm-vho`。
