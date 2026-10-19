# vt_head_opd_v1197aug — ≥1.5k 训练样本（镜像增广）+ 补充数据的"做什么"信息 + 主执行器 A+B 头 + OPD v3（K=8）

承接 `../vt_head_opd_v1197/RESULT.md`。用户 2026-10-16 的目标：
1. 用 1.5k 以上的数据重跑，并给补充数据配上"做什么"的信息；
2. 把"自有头 + 视觉分支"作为事先登记的主执行器，独立确认 +0.077 能否复现；
3. 加大 OPD 训练量（K 从 4 提到 8，或增加 epoch）。

## 1. 数据现状与做法（开始前已核实）

- **真实配对数据仍是 1197 条**。1014 回传包 656 条中，337 条有动捕，251 条采集时被驳回、68 条待采，都没有 PICO 文件。审核台 1025 条中，通过且未进 v1197 的只有 11 条，而且也没有动捕。目前没有更新的回传包，所以 1.5k 条真实配对数据还不存在。
- **用左右镜像增广把训练样本扩到 1.5k 以上**（`a0_mirror.py`，HumanML3D 本身也用这种做法）：
  - TRAIN 中每条真实样本生成一条镜像样本：视频水平翻转；PICO 动作取 x → −x 并交换左右关节；文本中的方位词 left↔right、顺时针↔逆时针互换。
  - 镜像后从动作重新提取的计划标签，与"直接左右互换标签"在 5 个字段上 **100% 一致**，说明几何镜像正确。
  - 有 28 条因为文本里的 left/right 不能确定是方位词（如"I left the room""right away"）而不做镜像。
  - 得到 **TRAIN 1652 条（真实 840 + 镜像 812）**，全部样本 2009 条。VAL 125、封存块 50、v860 TEST 182 都不变，也都不做镜像。
  - 视频张量翻转已核验：两份的图块集合相同，排列不同。
- **补充数据的"做什么"信息**（`a1_mr_gen.py`）：
  - 用本地 Qwen3-VL-8B，输入反应前视频（16 帧）+ 真实反应文本 + 正式数据中人工 mental_reasoning 的示例（只用 TRAIN 里的文本），生成第一人称的"我注意到了什么、为什么这样反应"，1–2 句，不复述身体动作。
  - 生成范围：补充数据的 TRAIN 410 条 + VAL 49 条（VAL 的只用于教师诊断，学生永远看不到）。
  - 镜像样本的 mental_reasoning = 源样本 mental_reasoning 的方位互换版；如果出现歧义，这条镜像样本就不带 mental_reasoning。
  - 校准：在 40 条有人工 mental_reasoning 的正式 TRAIN 条目上同样生成一遍，报告与人工文本的相似度（只用于描述，不作为门）。
  - 这个字段只给训练期的教师看。

## 2. 事先登记（2026-10-16 11:10 CST 写入；早于本版任何训练，也早于封存块解封）

### 2.1 设置（除注明外，与 v1197 版完全相同）
- **SFT**（学生 / 教师 v3）：Qwen3-VL-4B LoRA r16，lr 1e-4，3 个 epoch，accum 8，按 VAL NLL 选 epoch；在 TRAIN 1652 条上训练。教师 v3 的 prompt = PROMPT + 运动学提示 + mental_reasoning（正式数据用人工的，补充数据用生成的）。
- **动作头**：从 v860 的 head_pre 初始化，完整事件目标，3000 步，每 100 步在 VAL 上看 CE，取最低点。
  - `A_full`：只有文本接口 A；
  - `AB_full`：文本接口 A + 视觉分支 B，零初始化门。B 的编码器是本版 SFT 学生。
- **主执行器 = `AB_full`**（用户指定）；Kimodo-TPH 和 `A_full` 作为第二读数。
- **OPD v3**：
  - 教师 v3 冻结，学生 = 本版 SFT 学生。
  - 训练集 = 所有带 mental_reasoning 的 TRAIN 条目（真实 + 镜像）。
  - **K=8**（v2 为 4），温度 1，lr 5e-5，accum 4，**1 个 epoch**，取最后的检查点。
  - 损失 = 0.5·反向 KL + 0.5·前向 KL，只在 `<reaction>` 和 `<exec>` 两行计算。
  - 训练量：约 1650 条 × 8 个样本 ≈ 13,200 条 rollout，是 v2（430 × 4 × 2 = 3,440）的约 3.8 倍；优化步数约 410（v2 为 216）。
  - 训练到半个 epoch 时另存一个检查点，只作探索性读数。

### 2.2 确认性检验："+0.077 能否复现"
+0.077 的原始结果是：v1197 版中，OPD v2 第 2 个 epoch 相对 SFT，经 `AB_full` 头，VAL 125 条上 R@10 +0.077，CI [+0.024, +0.145]，当时属于探索性比较。
- **C1（独立训练种子，同数据同 VAL）**：用完全相同的配置在 v1197 上重训 OPD v2，只把 seed 从 0 改为 1，取第 2 个 epoch，经 v1197 的 `AB_full` 头，与 v1197 SFT 比较。通过 = VAL 125 上 R@10 差值的 CI 下界 > 0。
- **C2（独立条目：解封 50 条新来源封存块）**：原始 OPD v2 e2 与 v1197 SFT，经 v1197 的 `AB_full` 头，在封存块上比较（检索池 = 这 50 条，随机 R@10 = 0.2）。通过 = R@10 差值的 CI 下界 > 0。封存块只有 7 个来源、以 B 类为主，统计功效很低，结果会照实报告。
- **判定**：C1 和 C2 都通过 = 复现；只有一个通过 = 部分复现；都不通过 = 未复现。
- v860 TEST 182 条仍然封存，不参与本次任何检验。

### 2.3 本版的门（VAL 125；按来源 cluster bootstrap 95% CI）
- **G4-v3（主）**：OPD v3 相对 SFT-aug，经 `AB_full`-aug 头，R@10 的 CI 下界 > 0；同时 direction 和 duration 计划分布的 TV 距离都不比 SFT 恶化超过 0.1（与 v2 的判定口径相同）。第二读数：Kimodo-TPH、`A_full`。
- **G-T3（生成的"做什么"信息是否有用，诊断）**：在 VAL 中 49 条生成 mental_reasoning 的补充条目上，教师 v3（带 z）的 greedy 相对学生的 exec–gt3p 文本余弦高 ≥ 0.05。同时报告 76 条人工 mental_reasoning 条目上的同一指标。
- **探索性（不是门）**：
  - SFT-aug 相对 v1197 SFT（镜像增广的效果）；
  - `AB_full` 相对 `A_full`（视觉分支在更多数据下是否开始起作用）；
  - 错配视频对照。

## 3. 执行偏差（事后记录，2026-10-16 14:10 CST；早于任何 aug 版生成结果）

1. **SFT 中途崩溃**：约 12:55，学生和教师 v3 的 DataLoader 子进程在传递张量时连接失败（`multiprocessing/resource_sharer` 的 unix socket 报 `FileNotFoundError`），两个任务同时退出。
   - 学生：epoch 1 和 epoch 2 已完成并做了验证（VAL NLL 0.700 / 0.790）；第 3 个 epoch 在第 615/620 步时崩溃，没有保存。
   - 教师 v3：只有 epoch 1 完整（VAL NLL 0.489），第 2 个 epoch 在第 380 步时崩溃。
   - **处理**：仍按"VAL NLL 最低"的规则，在已完成的 epoch 中选择，学生和教师都取 epoch 1（学生 epoch 2 已经比 epoch 1 差，训练损失 0.12，明显过拟合）。这偏离了"3 个 epoch 中取最优"的登记。之后 `sft.py` 的 DataLoader 改为 num_workers=0。
2. **OPD v3 的执行方式**（优化目标、超参、步数都没变）：
   - **两卡数据并行**：每个优化步仍是同样的 4 个视频，两张卡各算 2 个；LoRA 梯度用 gloo 求和后除以 4，与单卡梯度相同，只差浮点求和顺序。冒烟测试中，2 步后两张卡的 LoRA 校验和完全一致。
   - **分块计算**：K=8 拆成每次生成"2 个视频 × 8 个样本"，前向/反向每块 4 条，梯度与一次算完相同。
   - **跳过 e0 评测**：e0 的 VAL greedy 就是 SFT 学生本身，已单独生成。
   - **显式调用 `student.train()`**：v2 是通过 e0 评测后的 `student.train()` 间接进入训练模式的，跳过 e0 后必须显式调用，否则梯度检查点和 LoRA dropout 都不生效。
   - **断点续训**：每 25 步保存一次 LoRA、AdamW、调度器、随机数状态和进度位置。
   - 原因：单卡实测约 130 秒/步，1 个 epoch 约 15 小时；所有卡都被他人占着算力。
3. **C2 已在 11:13 解封并完成**，结果见 `runs/cmp_C2_*.json`，判定见 `runs/gates_aug.json`。
4. **动作头首次启动崩溃（15:22）**：`chain_v3.sh` 全局导出了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，HERO 环境的 torch 1.12 不认识这个选项，两个动作头一启动就退出。16:15 去掉该变量后重新训练，配置不变。日志另存为 `logs/head_*_full_crash_allocconf.log`。
5. **OPD v3 中途崩溃与恢复**：约 16:41（第 150 步之后），两卡并行的 OPD 退出（推测是 GPU 7 上他人进程的显存涨到 41 GB 导致 OOM；原日志已被重试覆盖）。原编排的自动重试在 16:42、16:45、16:47 各失败一次，原因未知，日志同样被覆盖；第 3 次重试从第 150 步的断点正常继续。16:55 加了 `opd_watchdog.sh`：不限次数续训、每次另存日志、每次重新挑选显存最多的两张卡。断点包含 LoRA、AdamW、调度器、随机数状态和训练顺序中的位置，所以续训后的优化轨迹与不中断时相同，只差每一步的采样随机数。
6. **SFT 侧评测第一次失败（17:20）**：原因同第 4 条，aug 版 `run_val_motion.sh` 里 HERO 环境的生成步骤继承了 `PYTORCH_CUDA_ALLOC_CONF`。修复为 `env -u` 后用 `rerun_sftside.sh` 重跑，命令不变。

## 4. 事先登记：OPD v4（2026-10-16 21:05 CST，早于 v4 的任何训练）

**动机**（来自 v1197aug RESULT §1 的事后分析）：OPD v3 在人工版"做什么"信息的 VAL 条目上显著变差（R@10 −0.079），在生成版条目上方向为正。假设：生成版信息复述了答案，并稀释了人工版信息的占比，导致 v3 失败。

**设置**：与 OPD v3 完全相同，只有一处不同：训练条目只取 mental_reasoning 为人工撰写的（`--mr-filter human`：真实 430 + 镜像 402 = 832 条）。
- 学生 = SFT-aug 学生 epoch1，教师 = 教师 v3 epoch1，均不重训；
- K=8，温度 1，lr 5e-5，accum 4，1 个 epoch（208 步），取最后的检查点；
- 损失 = 0.5·反向 KL + 0.5·前向 KL，只算 reaction 和 exec 两行；
- 两卡数据并行，断点续训用看门狗。

**门**（VAL 125，按来源 cluster bootstrap 95% CI）：
- **G4-v4（主）**：OPD v4 相对 SFT-aug，经 AB_full-aug 头，R@10 的 CI 下界 > 0；同时 direction 和 duration 计划分布的 TV 都不比 SFT 恶化超过 0.1。第二读数：Kimodo-TPH、A_full。
- **H-v4（检验上面的假设）**：在 76 条人工版 VAL 条目上，经 AB 头，OPD v4 相对 OPD v3 的 R@10 差值 CI 下界 > 0 → 支持假设；CI 跨 0 → 不支持。

## 5. 事先登记：OPD v5 —— 只用 reason 作特权信息，并且只保留"能从视频推断"的部分（2026-10-17 00:45 CST，早于 v5 的任何训练）

**背景**：OPD v4 显著变差（经 AB 头，R@10 −0.093，CI [−0.172, −0.024]），H-v4 不成立。v2–v4 的教师都带运动学提示（动捕测得的时长、位移、转角），这部分不可能从反应前视频推断。理论上，z 不可推断时，反向 KL 的最优学生是各教师分布的几何平均，会塌向"安全答案"。用户 00:30 的意见：z 只应包含能从视频得出的信息，用 reason 当特权信息。

**特权信息 z 的两种形式**（都去掉运动学提示，也都不用生成版 reason）：
- **R-full**（`VHO_ZMODE=mr`）：完整的人工 mental_reasoning，含"看到了什么"和"打算做什么"。
- **R-obs**（`VHO_ZMODE=mr_obs`）：只保留感知部分。`a2b_mr_obs_extractive.py` 按句子和子句切分，在第一个意图、决策、计划词处截断，纯抽取，只保留原文片段（有断言保证）。
  - 原本用 Qwen3-8B 改写，但它编造了视频里不存在的细节（平均 79 词，原文 34 词），已拒用，产物另存为 `runs/mr_obs_llm_REJECTED.json`。
  - 审计：506 条真实条目中 19 条无感知部分（不带 z）；意图词比例 91% → 0%；与反应原文的内容词重合率 0.40 → 0.19；平均 19 词；镜像 388 条。

**教师**：Qwen3-VL-4B LoRA SFT，配置与之前完全相同（r16，lr 1e-4，3 个 epoch，accum 8，按 VAL NLL 选 epoch），在 TRAIN 1652 上训练。有 z 的条目在 prompt 末尾附上 z，其余用普通 prompt。
- `sft_teacher_r`（R-full）
- `sft_teacher_o`（R-obs）

**学生**：沿用 SFT-aug 学生 epoch1，不重训。

**OPD 的三个臂**：都是 K=8、温度 1、lr 5e-5、accum 4、1 个 epoch、取最后的检查点、只在 reaction 和 exec 两行算 KL、两卡数据并行加断点续训、不调用生成器。
- **opd_v5o（主）**：教师 R-obs，**纯反向 KL（λ=0）**，训练条目 = 有 R-obs 的 TRAIN（803 条）。
- **opd_v5r**：教师 R-full，纯反向 KL，训练条目 = 有人工 reason 的 TRAIN（832 条）。
- **opd_v5o_mix（次）**：教师 R-obs，λ=0.5（与 v2–v4 相同的混合 KL）。

**门**（VAL 125，按来源 cluster bootstrap 95% CI）：
- **G-T5（诊断，不影响是否训练）**：每个教师在有 z 的 VAL 条目上，带 z 的 greedy 相对学生的 exec–gt3p 文本余弦提升；以及 KL(带 z) / KL(不带 z)。
- **G4-v5（主）**：opd_v5o 相对 SFT-aug，经 AB_full 头，R@10 的 CI 下界 > 0；同时 direction 和 duration 计划分布的 TV 都不比 SFT 恶化超过 0.1。第二读数：Kimodo-TPH、A_full。
- **H-v5a（"z 能从视频推断时，反向 KL 不塌缩"）**：opd_v5o 的 direction TV 恶化 ≤ 0.1，且方向分布中最大类别的占比不比 SFT 高出 0.1 以上 → 成立。
- **H-v5b（去掉运动学和意图有帮助）**：在 76 条人工 reason 的 VAL 条目上，经 AB 头，opd_v5o 相对 opd_v4 的 R@10 差值 CI 下界 > 0 → 成立。
- **次要比较**：opd_v5r 相对 SFT；opd_v5o_mix 相对 opd_v5o；以上各臂的 Kimodo-TPH 和 A_full 读数。

GPU：每次启动前挑当时最空闲的卡（先比利用率，再比剩余显存），记录在 `runs/gpus_picked.txt`。

## 6. 事先登记：OPD v6 —— 不给特权，换成能力更强的教师（2026-10-17，早于 v6 的任何训练）

**动机**：v1–v5 的教师都靠特权信息 z 才比学生强，而 z 要么从视频推不出来，要么带来的优势太小。文献里证据最多的做法是"教师和学生输入相同，教师只是能力更强"，这时纯反向 KL 的最优解就是教师本身（MiniLLM、Thinking Machines、Qwen3）。用户指示：教师不给特权，换成更强的模型，仍用 OPD + 反向 KL。

**教师**：Qwen3-VL-8B-Instruct（与 4B 学生的分词器逐字节相同，词表都是 151,936，可以在整个词表上精确计算 KL）。
- LoRA r16，加在 LLM 36 层 × 7 个线性层上；视觉塔冻结。
- **prompt 与学生完全相同**（`VHO_ZMODE=none`，不带任何 z）；目标是同样的三行答案。
- 在 TRAIN 1652 上做 SFT，配方与学生相同：lr 1e-4，3 个 epoch，accum 8，按 VAL NLL 选 epoch，DataLoader num_workers=0。
- 产物：`sft_teacher_8b`。

**学生**：SFT-aug 学生的 epoch1（4B，与 v3–v5 相同）。

**OPD 的两个臂**：都是 K=8、温度 1、lr 5e-5、accum 4、1 个 epoch，在全部 1652 条 TRAIN 上训练（`--mr-filter none`），取最后的检查点；两卡数据并行加断点续训，训练期间不调用生成器。
- **opd_v6（主）**：**纯反向 KL（λ=0）**，在**全部回答 token** 上计算（含计划行和结束符，`--kl-span all`），与标准 OPD 的做法一致。
- **opd_v6_re（次）**：纯反向 KL，只算 reaction 和 exec 两行。

**执行器**：
- 主读数：**A36 头**（`vt_lastlayer_v1197aug/runs/head_A36`：冻结原版 4B 只读文字、只用最后一层）。它在 last-layer 实验中最好，对"部分对的描述"更敏感。
- 次读数：A+B 头、A 头、Kimodo-TPH。

**门**（VAL 125，按来源 cluster bootstrap 95% CI）：
- **G-T6（教师是否真的更强；只用于诊断，不影响是否训练 OPD）**：8B 教师相对 4B 学生的 VAL NLL；两者 greedy 预测经 A36 头和 Kimodo-TPH 的 R@10、exec 余弦、计划准确率。
- **G4-v6（主）**：opd_v6 相对 SFT-aug，经 A36 头，R@10 的 CI 下界 > 0；并且 direction 和 duration 计划分布的 TV 都不比 SFT 恶化超过 0.1。
- **H-v6（不给特权时，即使计划行也算 KL，纯反向 KL 也不塌缩）**：opd_v6 的 direction TV 增加 ≤ 0.1，且最大方向类别占比增加 ≤ 0.1 → 成立。对照：v1（特权 + 纯反向 + 计划行也算）为 +0.18 / +0.25。
- **描述性**：迁移比例 =（OPD − SFT）/（教师 − SFT），按执行器分别报告；opd_v6_re 相对 SFT；opd_v6 相对 opd_v6_re。
- **零成本补充（探索性）**：已有的 SFT 和 v2–v5 各版 OPD 预测，都过一遍 A36 头。
7. **opd_v6_re 启动失败（13:2x–14:0x）**：第一次尝试时，rank1 所在的卡上他人进程占用增加，导致 OOM。之后的重试因为残留目录里已有 `log.jsonl`、却还没有 `resume.pt`（崩在第一个检查点之前），触发了 `FileExistsError`。14:10 改用 `opd_runner2.sh` 重启：遇到这种情况先把残留目录归档为 `runs/_failed_*`，再从头开始；同时把 `--save-every` 从 25 降到 10，只影响检查点频率，不影响训练。
8. **v6 的评测编排 bug（17:41）**：`chain_v6.sh` 里的 `a36()` 没有把变量 `n` 声明为 local，覆盖了 `evalarm` 的 `n`，导致 opd_v6 的 A+B 头和 Kimodo 评测去找一个不存在的预测文件，失败了。A36 头的读数不受影响。17:50 换成 `chain_v6b.sh`（变量都声明为 local），评测命令不变，并接管了 opd_v6_re 的评测和判定。

## 7. 事先登记：教师门 G-T7 —— 自洽共识教师与更多帧教师（2026-10-17 20:05 CST，早于本节的任何采样和训练）

**动机**（2026-10-17 与用户的讨论）：标签是标注者在多种合理反应里选出的"最可能的一种"。如果 z 来自标签，前向 KL 的最优学生约等于 p(y|x)（与 SFT 相同），反向 KL 的最优学生是几何平均（塌缩）。因此 z 只能是**视频的函数**，教师的优势只能来自更多的计算或更多的输入视角。本节只做教师门，**不训练 OPD**；门通过后，另行登记 OPD v7。用户已批准本节（"可以安排在 v6_re 之后或同时进行"）。

### G-T7a：自洽共识（MBR）

- **采样**：SFT-aug 学生（epoch1）和 8B 教师 `sft_teacher_8b`（epoch1，同一 prompt）各对 VAL 125 条采样 N = 16 次。温度 1.0，top_p 1.0（与 OPD rollout 相同），`infer.py --k 16`，种子 1234。产物：`sc16_stu`、`sc16_t8b`。
- **读数**：第 k 个样本（k = 0..15）单独组成一份预测，经 A36 头读数（每条文字生成 3 段动作，与以往相同），得到"条目 × 样本"的逐条指标。
- **选择规则**（事先固定）：
  - **MBR-exec（主）**：
    - 候选：严格解析合格的样本；全部不合格时取样本 0。
    - 嵌入：冻结原版 4B 第 36 层在 caption 词上的隐状态取平均（即 A36 头使用的文本特征）。减去该模型在 VAL 全部候选上的均值，再做 L2 归一化。
    - 选 Σ_{j≠i} cos(e_i, e_j) 最大的样本；并列时取编号最小的。
  - **MBR-dur（次）**：先对时长做多数投票（并列时取并列者中 MBR 分数最高的样本所用的时长），只在时长等于多数票的候选里做 MBR-exec。
  - **mean@16**：16 个样本逐条指标的平均，代表"单个温度 1 样本"的期望水平。
  - **oracle@16**（描述性上界）：
    - 朴素版：16 个样本里逐条 R@10 最高的那个，存在"挑噪声"的乐观偏差；
    - **交叉拟合版**：用第 1 段动作的结果挑样本，用第 2、3 段动作评分，去掉动作生成噪声带来的选择偏差。
- **比较**：A36 头，按来源做 cluster bootstrap 求 95% CI，配 Wilcoxon。
  - **G-T7a（主）**：学生 MBR-exec − 学生 greedy（`vt_lastlayer_v1197aug/runs/eval_head_sft__A36.json`），R@10 的 CI 下界 > 0 → 通过，"共识教师"值得做 OPD v7。
  - 次要：8B MBR-exec − 学生 greedy；8B MBR-exec − 8B greedy；学生 MBR-dur − 学生 greedy；学生 mean@16 − 学生 greedy。
  - 描述性：oracle@16（两版）− greedy。解读规则：若交叉拟合 oracle − greedy ≥ 0.15 而 MBR 不通过，结论是"正确答案在分布里，但这个选择器找不到"。

### G-T7b：更多帧（32 帧）

- **缓存**：新的视频缓存，32 帧，max_pixels 不变（360×640）。不足 32 帧的片段取最大偶数帧；镜像样本做水平翻转。缓存目录为 `/tmp/vho1197aug_f32_cache`，由 `VHO_NFRAMES=32` 控制。16 帧缓存不动。
- **SFT**：4B 学生，配方与 SFT-aug 完全相同（lr 1e-4，3 个 epoch，accum 8，r16，在已完成的 epoch 里按 VAL NLL 选）→ `sft_f32`。
- **读数**：VAL greedy → A36 头。
- **G-T7b**：sft_f32 greedy − 16 帧 SFT greedy，A36 R@10 的 CI 下界 > 0 → 通过。描述性：VAL NLL、计划准确率（直接由预测与标签计算）。
- **说明**：32 帧是测试时也能拿到的输入，不是特权。如果通过，最直接的用法是学生本身改用 32 帧；作为 OPD 教师是第二种用法。

### 通用

- 只用 VAL，TEST 和封存块都不读。
- GPU：每次启动前挑当时最空闲的卡，排除 opd_v6_re 占用的卡，记录在 `runs/gpus_picked.txt`。
- 判定由 `decide_t7.py` 机械执行，写入 `runs/gates_t7.json`。

**§7 执行记录**：
- 20:19 启动 `chain_t7.sh`。启动前用一条样本验证了 32 帧缓存：16 组 × 220 = 3,520 个视频 token。
- 20:36 发现所有卡都被其他租户占到约 100%，学生采样约 45 秒/条，串行编排要到凌晨才能结束。于是按 PID 停掉 `chain_t7.sh` 的主外壳和支线 A 的外壳；正在采样的学生进程和支线 B（32 帧 SFT）都保留。
- 改由 `chain_t7b.sh` 接手：8B 采样挪到另挑的一张卡上（GPU 7）并行运行，学生和 8B 的逐样本读数各在自己的卡上做。采样参数、选择规则和门都没有改动。
- **G-T7b 补登记（21:00，早于 sft_f32 任何 epoch 的结果；此时只有 epoch 0 的 VAL NLL）**：整台机器都很挤，各卡实测只能拿到 33–42 TFLOPs。sft_f32 在 GPU 2 上约 81 秒/步，跑满 3 个 epoch 约需 14 小时。因此改为**完成 epoch 2 后停止**，在已完成的 epoch（1、2）中按 VAL NLL 选。这与 16 帧参照 `sft_student` 完全一致：参照也只完成了 epoch 1–2，选的是 epoch1。学习率日程不变（仍按 3 个 epoch 的余弦），所以两者在同一 epoch 上的日程位置相同。由 `t7b_stop.sh` 执行。

## 8. 事先登记：reason 当输出（R-CoT / R-obs / R-MT）与 R-chain 特权教师（2026-10-18 00:24 CST，早于本节的任何训练）

**动机**（2026-10-17 与用户的讨论）：特权信息分两种。**信息型**是视频里没有的（标注者选了哪个反应、动捕测量值），任何方法都学不到，而且标签本身已包含它。**计算型**是视频里有、但难以提取的（感知要点、推理链），这才是蒸馏能传递的。一条 mental reasoning 由"感知 + 推理链 + 决定"组成，其中"决定"基本是信息型。
- 当作**学生的输出目标**（MLE，相当于前向 KL）时，信息型部分与标签无异，无害；
- 当作**教师的输入**配反向 KL 时，信息型部分会导致几何平均和塌缩。
- v5r（完整 reason）和 v5o（只有感知）都没有测过"感知 + 推理链、去掉决定"。

用户已批准（目标）：两部分都做；reason 用人工 + 生成；挑空闲的卡，**同时最多占用 3 张**（包括仍在运行的 G-T7b）。

**抽取**（`a2c_mr_chain.py`，确定性规则，不用大模型改写，只保留原文中逐字存在的片段）：
- **R-chain**：每句在最早出现的"决定/意图标记""so/therefore/thus/hence/then + I/we""I/we + 非感知/认知动词（动作）"处截断。从句（if/when/though/because/since/as …）和间接疑问（asks what I want）除外；句首的"By …ing / To …"整句去掉；补上代词回退和结尾清理。
  - 覆盖：人工 864/908、生成 868/869。
  - 与反应原文的内容词重合：人工 0.405 → 0.169，生成 0.631 → 0.346。
- **R-obs-all**：人工部分沿用 v5 的 `mr_obs.json`；生成部分用同一条 v5 规则。覆盖：人工 875、生成 867。
- 抽取前看过的输出只有规则调试用的 12 条示例和上述统计，没有任何模型结果。

### 第一部分：reason 当学生的输出（三个 SFT 臂，4B）

| 臂 | 训练样本（prompt → 目标） | 推理 |
|---|---|---|
| **R-CoT（主）** | `PROMPT_COT` → `<reason> 完整 reason` + 原三行 | 先写 reason，再写三行 |
| R-obs | `PROMPT_OBS` → `<observe> R-obs-all`（为空时写 `<observe> nothing specific.`）+ 原三行 | 先写 observe，再写三行 |
| R-MT | 每条样本出现两次：`PROMPT` → 原三行；`PROMPT_REASON` → 完整 reason | `PROMPT` → 三行，格式与 SFT 完全相同 |

- **配方**：与 SFT-aug 相同（LoRA r16，lr 1e-4，accum 8，按 3 个 epoch 的余弦日程），**完成 epoch 2 后停止**，在 epoch 1–2 中按 VAL NLL 选：
  - R-CoT、R-obs 用完整目标的 NLL；
  - R-MT 只用"原三行"任务的 NLL；
  - 这与 16 帧参照 `sft_student`（完成 epoch 1–2，选 epoch1）以及 G-T7b 一致。
- **推理**：VAL greedy，`max_new_tokens` = 288（三行版保持 160）。三行照常解析，多出的首行被忽略。
- **读数**：A36 头（每条 3 段动作）。
- **门**（按来源做 cluster bootstrap 求 95% CI）：
  - **G-R1（主）**：R-CoT greedy − SFT-aug greedy（`vt_lastlayer_v1197aug/runs/eval_head_sft__A36.json`），A36 R@10 的 CI 下界 > 0 → "会推理的学生更强"。
  - **G-R2 / G-R3（次）**：R-obs、R-MT 各自相对 SFT-aug，用同样的标准；不做多重比较校正，结论以 G-R1 为主。
  - 描述性：计划准确率、t2m、同类 R@3。对 R-CoT 和 R-obs 另报一致性：自己的 reason 与自己的 reaction 的内容词重合，减去与随机另一条 reaction 的重合。
  - 任一臂通过时，它就是"计算型优势的教师"候选；用它做 OPD 需要另行登记，不在本节范围内。

### 第二部分：R-chain 作为特权信息

- **教师**：4B，`VHO_ZMODE=mr_chain`，prompt = PROMPT + "Privileged reasoning (…the decision itself is removed…): " + R-chain。没有 R-chain 的条目用普通 PROMPT。SFT 配方同上（完成 epoch 2 后停止，按 VAL NLL 选）→ `sft_teacher_chain`。
- **G-T8 教师门**：教师在 VAL 上带 z 做 greedy，与 SFT-aug 学生 greedy 比较。**两条都满足才算通过**：
  1. A36 R@10 差的 CI 下界 > 0（教师显著更强）；
  2. 计划准确率**不上升**：差的点估计 ≤ +0.02，且 cluster bootstrap CI 下界 ≤ 0。上升即视为决定泄漏进了 R-chain。
  - 描述性：在人工 reason 的 VAL 子集（76 条）上分开报告。
- **只有 G-T8 通过时才做 OPD v8**：
  - 教师 `sft_teacher_chain`，学生 SFT-aug epoch1；
  - **纯反向 KL（λ=0）**，KL 只算 reaction 和 exec 两行（`--kl-span re`；时长主要是表演噪声，不蒸馏）；
  - K=8，lr 5e-5，accum 4，1 个 epoch，只用有 R-chain 的 TRAIN 条目（`--mr-filter chain`）；两卡数据并行，取最后的检查点。
  - **G4-v8**：opd_v8 greedy − SFT-aug greedy，A36 R@10 的 CI 下界 > 0；且预测计划的方向、时长分布相对真值的 TV，都不比 SFT 恶化超过 0.1（由预测直接计算）。
- 判定由 `decide_r.py` 机械执行，写入 `runs/gates_r.json`。只用 VAL；TEST 和封存块都不读。

## 9. 事先登记：第一人称因果预训练（视频→文本）再做 SFT（2026-10-18 15:47 CST，早于本节的任何数据构造和训练）

**动机**（用户路线，2026-10-18）：先用带因果解释的第一人称公开数据，把 VLM 的"视频→文本"推理能力训出来（情境 → 为什么 → 下一步），再在我们的数据上 SFT，动作头不变。之前所有尝试都只在 840 条真实样本内部做，"第一人称因果推理数据的数量和多样性"这个变量从没动过。
**用户批准**：可以使用本地 Ego4D；下载 EgoNormia 和 EgoToM 的标注；"开始吧"。同时最多占 **3 张卡**（沿用此前的上限）。

**数据**（`ego_build.py`，确定性构造，不用大模型改写；视频一律来自本地 Ego4D v2 或 Shawn 已经切好的 Ego4D 片段，只读引用）：
- **评测侧视频**：EgoNormia verified 200 题所在的视频，加上全部 EgoToM 视频，共 795 个 video_uid。**这些视频的任何片段都不进训练。**
- **训练混合 PT-C（约 1.9 万条）**：
  1. **Goal-Step 下一步**（`goalstep_train.json`）：
     - 每个视频取最细一层的片段（有子步骤用子步骤，否则用步骤），按时间排序，取相邻对 (a, b)；要求间隔 ≤ 10 s、b 的开始时刻 ≥ 8 s、两段描述不同；可用 19,200 对，种子 0 随机抽 **10,000** 对。
     - 视频窗口 = [b 开始前 8 s, b 开始]。
     - 目标：`<reason> I am working on: {goal}. I have just done: {a}.` + `<next> {b}.`
  2. **EgoNormia**：非评测视频上、正确项不为空的 **1,252** 题。
     - 窗口 = [t − 5 s, t]，t 是 ID 中的动作时刻；与官方 `video_prev` 一致。
     - 目标：`<reason> {正确项的理由}` + `<next> {正确行为}`。
  3. **EgoRe Why**（EgoThinker，CC BY 4.0）：`EgoRe_clips/ego4d_long.json` 中首轮以 "Why" 开头、且不在评测视频上的条目（19,171 条），种子 0 随机抽 **8,000** 条。
     - 视频 = 对应的本地片段（10 fps / 360p）。
     - prompt = 原问题，目标 = 原回答。
- 前两类用同一个 prompt `EGO_PROMPT`："视频停在我做下一件事之前；推断我为什么这样做，并预测下一步；严格输出 `<reason>` 和 `<next>` 两行"。
- 视频处理与我们的数据相同：16 帧，max_pixels 360×640，时间戳从窗口起点开始计。缓存放在 `/tmp/vho_ego_cache`。

**训练**：
- **阶段 A（PT-C）**：4B，新建 LoRA r16（与 SFT 相同的模块），lr 1e-4，余弦，3% warm-up，全局每 8 条更新一次，**1 个 epoch**，取最后的检查点。最多 3 张卡数据并行（gloo 汇总 LoRA 梯度，与单卡梯度相同），支持断点续训。→ `pt_c`
- **阶段 B**：从 `pt_c` 的 LoRA 继续，在 TRAIN 1652 上 SFT，配方与 SFT-aug **完全相同**（三行目标，lr 1e-4，按 3 个 epoch 的余弦日程，完成 epoch 2 后停止，在 epoch 1–2 中按 VAL NLL 选）。→ `sft_ptc`
- **读数**：VAL greedy → A36 头（每条 3 段动作）。

**门**（VAL 125，按来源做 cluster bootstrap 求 95% CI）：
- **G-E1（主）**：sft_ptc greedy − SFT-aug greedy（0.291，`vt_lastlayer_v1197aug/runs/eval_head_sft__A36.json`），A36 R@10 的 CI 下界 > 0。
- **描述性**：t2m、同类 R@3、计划准确率、VAL NLL（与 SFT-aug 的 0.700 比较）。
- **G-E0（诊断：阶段 A 有没有学到第一人称因果推理；不影响是否做阶段 B）**：对每个选项计算平均 token 对数概率，选最高的那个；按 video_uid 做 bootstrap 求 95% CI。
  - EgoNormia verified 200 题：窗口 [t−5, t]，prompt 只问下一步，对 `<next>` 之后的行为文本打分。
  - EgoToM（本地有视频的 1,035 题，三类分开报告）：窗口为官方上下文片段，问题为官方问题，对选项文本打分。
  - 比较对象：原版 4B、SFT-aug 学生、pt_c、sft_ptc。主比较是 pt_c 相对原版 4B。
- **条件性追加**：只有 G-E1 通过时，才另行登记对照臂 PT-N（同样的视频和下一步，但去掉 `<reason>` 解释），用来区分"有因果解释"和"只是多看了第一人称视频"。
- 判定由 `decide_e.py` 机械执行，写入 `runs/gates_e.json`。只用 VAL；TEST 和封存块都不读。EgoToM 只用于评测（许可如此）。
