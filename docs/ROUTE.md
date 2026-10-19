# 路线讨论：Qwen3-VL-4B（V→T SFT）+ 自有动作头（T→M 预训练）+ 特权信息 on-policy 自蒸馏（反向 KL）

> 版本说明：本文保留当时的路线与解释。最新结论、ego数据使用和ER19状态见 [实验时间线](EXPERIMENTS.md) 与 [接管记录](TAKEOVER_20261019.md)；“同尺寸必须有特权”“学生已掌握全部可观测信息”等历史判断不作为已证实前提。

日期：2026-10-15。性质：路线讨论草案，**未实施、未训练、未下载任何东西**。
证据标记：[本地] = 本工作区文件或实测；[常识] = 广为人知的文献或做法；[需核实] = 印象中存在，定稿前要查原文和编号；[估算] = 未实测的粗估。
前作：`route_discussion_20261013/ROUTE.md`、`reports/generation_routes_report_20261014/…调研与建议_20261014.md`、`route6_v860_20261014/RESULT.md`。

---

## 0. 结论（一页版）

1. **把用户的路线翻译成模块**：Qwen3-VL-4B + LoRA 学 V→T（视频 → `<reaction>/<exec>/<plan>`）；自有动作头学 T→M（先在 HumanML3D 上预训练，再在 PICO 上微调）；两者在"隐状态接口"处拼接；最后用**同尺寸、带特权信息的 teacher**，在 student 自己的采样上做逐 token 反向 KL（on-policy distillation，OPD）。
2. **1–2 张 A100-80G 足够**：核心实验约 2–3 GPU·天，加消融约 4–6 GPU·天 [估算]，两张卡并行约 1.5–2 周墙钟。前提是 GPU 不被抢。
3. **三个决定成败的设计点**：
   - **接口**：用哪一层、哪些 token、LoRA 开还是关。默认推荐"生成文本 → 关掉 LoRA 重编码 → 中间层多层混合 → 交叉注意力"。原因：这样预训练得到的 T→M 能力可以原样迁移，后续 OPD 改 LoRA 也不会破坏动作头。
   - **同尺寸自蒸馏必须有信息不对称**：teacher 和 student 输入相同时，初始 KL ≡ 0，什么也学不到。teacher 必须在训练期看到特权信息 z（PICO 计划标签、mental_reasoning、反应发生后的画面……），student 只看反应前视频。
   - **反向 KL 在"z 从视频推不出来"时有理论陷阱**：最优 student 是各 teacher 条件分布的**几何平均**，倾向于输出"在所有情境下都说得过去"的泛化反应。这和 MPJPE 偏爱平均动作是同一类病，必须监控。
4. **主瓶颈仍是"决策"**（路线 6：可部署最好 R@10≈0.25，gt3p oracle 0.44）[本地]。换动作头、做蒸馏都不会凭空增加视频里的信息。合理预期：OPD 有望追平或略超"按奖励选样"的诊断上限（R@10 0.288）[本地]，而不是逼近 oracle。
5. **推荐的执行方式**：两条卡各跑一条线。GPU0 跑 V→T SFT → OPD，**先接冻结 Kimodo 评测**，可以和路线 6 的 DPO 在同一个事先登记的 G4 下直接比较；GPU1 跑 T→M 动作头预训练和微调。第 2 周用 2×2 设计（{SFT, SFT+OPD} × {Kimodo, 自有头}）合流，把"VLM 的收益"和"生成器的收益"分开。
6. **开工前的阻塞项**：
   - 本地**没有 Qwen3-VL-4B 权重**：`/mnt/workspace/Tao/lingbot-vla-v2/lingbot-vla/Qwen3-VL-4B-Instruct` 只有 config/tokenizer，共 11 MB。需要下载约 9 GB，按工作区规则需要用户批准。Qwen3-VL-8B 本地完整（`/mnt/workspace/eckes/models/Qwen3-VL-8B-Instruct`），且与 4B 的 tokenizer.json 逐字节相同 [本地]。
   - 现有 `vlm-r1` 环境的 transformers 是 4.51.3，不支持 Qwen3-VL（需要 ≥4.57）。`qwen36_vllm` 环境有 vLLM 0.25.1 / transformers 5.13.1，可以试，但能否加载 Qwen3-VL + LoRA 未验证 [本地]。

---

## 1. 路线的模块化表达

```text
                     训练期 teacher：同一个 4B 基座 + 冻结 LoRA_T + 特权信息 z
                                     │  逐 token 反向 KL（在 student 自己的采样 y 上）
                                     ▼
反应前视频 x ──► Qwen3-VL-4B + LoRA_S ──► y = <reaction> / <exec> / <plan>            (V→T)
                                               │
                           <exec> 文本（关 LoRA 重编码，取第 ℓ 层 / 多层混合的隐状态）
                                               ▼
                                 projector ──► 动作头 ──► HumanML263 动作（→ PICO / G1）   (T→M)
                                               ▲
                     预训练：HumanML3D caption 走同一个模板、同一个冻结编码器
```

Qwen3-VL-4B 的关键参数 [本地 config]：
- 文本部分：hidden 2560、36 层、GQA 32/8、vocab 151,936、`tie_word_embeddings=True`；
- 视觉部分：24 层，deepstack 从视觉第 5、11、17 层注入 LLM 前几层；patch 16，时间 patch 2，空间 merge 2。

---

## 2. 分阶段详解

### 2.1 阶段 1：V→T SFT

- **直接复用路线 6 的格式和数据管线**（`route6_v860_20261014/s1_sft.py`、`r6_vlio.py`）。目标是三段文本：
  - `<reaction>`：人工反应；
  - `<exec>`：gt3p 第三人称动作描述；
  - `<plan>`：PICO 自动提取的计划标签。

  只需把骨干从 Qwen2.5-VL-7B 换成 Qwen3-VL-4B。
- **参照线** [本地]：7B SFT 在 VAL 的 TPH 臂上 R@10 0.253、计划准确率 0.592、格式合格率 100%。4B 若显著低于 7B，要考虑改用本地现成的 8B。
- **设置**：LoRA r16–64 只加在 LLM 上，视觉塔冻结。1.5k 条、每条约 1k token、3 epoch，单卡 < 2 GPU·h，显存约 12–18 GB [估算；7B 实测峰值 20.7 GiB]。
- **可选的 V→T 预训练**（Ego4D 叙述、Nymeria 叙述）这一个月内不建议做：数据下载和处理量大，回报不确定。

### 2.2 阶段 2：T→M 动作头预训练（"重定向"怎么做）

**方向反过来：不要把 HumanML3D 重定向到 PICO，而是把 PICO 转成 HumanML263。**
- `experiments/hero_finetune/pico_to_humanml.py` 已实现 PICO22 → HumanML263：round-trip 误差 3.6e-5 m，骨长比 0.998 [本地，memory 88]；
- HumanML3D 本地可用：`LaMP/dataset/HumanML3D`，包含 29,228 条 new_joint_vecs，train/val/test 列表行数为 24,844/1,460/4,384 [本地]；
- Motion-X、AMASS 本地也有，可以作为扩充 [本地]。

**口径**：HumanML263 是 22 关节，不含双手末端。按工作区规则只能称为 HumanML22 track，**不能包装成原生 24**。服务、社交类反应依赖手部端点，原生 24 版本可以作为第二步（AMASS SMPL24 → 与 PICO 24 对齐）。

**动作头选型：**

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **A. 离散 RVQ + 自回归 Transformer**（T2M-GPT / MoMask 系）[常识] | 复用 HERO 已在 PICO 上微调的 RVQ（`outputs/hero_ft_vq_20260926/best.pt`，val 重建 0.182→0.090）或 HumanML3D 版 VQ [本地] | 解码器保证骨架结构（修掉"骨头伸缩"）；反应约 3 s ≈ 60 帧 ≈ 15 个 token，便宜；**每个 token 都有分布，可以直接做 KL 蒸馏** | VQ 有量化误差；只做 AR 的话多样性一般 |
| B. 连续 flow / 扩散头（π0 系）[常识] | 直接在 263 维特征上做 flow matching | 结构简单，连续空间精细 | 反向 KL 无法逐 token 计算；本地已有一次负结果（见下） |

**必须正视的本地负结果** [本地，报告 §3.4]：冻结 Qwen2.5-VL-3B + flow 头直接回归 88 帧 × 24 关节位置：
- test 误差 0.964 m，比训练均值 0.914 m 还差；
- 骨长变异系数 0.19（真人 0.014）。

新方案通过"大数据先验 + 结构化表示 + 更好的接口"回应这三处失败：
- 无动作先验 → 在 HumanML3D 上预训练；
- 回归原始关节位置 → 改用 VQ 或 263 维表示；
- 对所有 token（含数百个视觉 token）的最后一层取平均 → 改用 exec span、中间层、交叉注意力。

**"最后一层的输出"作为语义条件：建议改为多层。**
- 4B 是 tie-embedding 模型。最后一层隐状态直接与词嵌入点乘得到 logits，所以它更像"下一个词是什么"，而不是"当前语境的含义"。
- 表征研究与 VLA 实践都提示中间层更适合做条件：GR00T N1 取 LLM 中间层（第 12 层）[常识]；Layer-by-Layer（Skean et al. 2025）[需核实]。
- 中间层没有经过最终 RMSNorm，并且存在 massive activations（Sun et al. 2024）[常识]。projector 前要先做归一化，并丢掉 system/BOS 这类 sink token。
- **做法**：一次性缓存第 12/18/24/36 层，学一个 softmax 标量混合（ELMo 式）。消融"仅最后一层 / 仅中间层 / 混合"，成本几乎为零。
- **用哪些 token**：用 `<exec>` span 的逐 token 序列做交叉注意力，不做全局平均池化。现有的 `umrm_model/qwen3vl.py::Qwen3VLConditioner` 正是"最后一层 + 全 token 均值"，也就是负结果的那种配置。

**预训练流程：**
1. 把 HumanML3D caption 套进与推理时完全相同的 `<exec>…</exec>` 模板，用**冻结、关 LoRA** 的 4B 做纯文本前向，缓存多层隐状态（约 4.5 万条 caption，分钟级）；
2. 训练动作头，约 0.5–1 GPU·天 [估算]；
3. 自检：用 HumanML3D 标准评测器（FID / R-precision）和本地 T2M-GPT / MoMask 系权重比较。差得多就先修接口，不进入下一步；
4. 在 UMRM train 上微调（PICO→263 + gt3p 文本）；
5. 可选：把 `<plan>` 编译成根路径，作为额外条件。路线 6 的 TPH 臂证明根路径条件收益很大 [本地]。

### 2.3 阶段 3：V→M 拼接（接口选择）

| 接口 | 怎么接 | 预训练能否原样迁移 | OPD 改 LoRA 后动作头是否失配 | 能否带上视觉细节 |
|---|---|---|---|---|
| **A 文本瓶颈**（推荐默认） | VLM（开 LoRA）生成 `<exec>`；再用**关 LoRA 的基座**纯文本重编码，取隐状态 | 能，接口与预训练完全一致 | 不失配 | 不能（只经过文本） |
| B 视觉语境隐状态 | 直接取生成时（开 LoRA、带视频语境）`<exec>` token 的隐状态 | 部分：有视频语境，分布偏移 | 失配，需要联合重调 | 能 |
| C 可学习动作 query（MetaQuery 式）[需核实] | 在序列尾部追加 N 个 query token | 不能：预训练时没有视频 | 失配 | 能 |

- **建议**：以 A 为主；B 作为**零初始化门控残差**叠加在 A 上，训练起点与 A 完全等价，再与"零修正"对照。这与 10-14 报告 §3.3 的"零初始化残差桥"思路一致。
- **动作损失的梯度要不要回传 VLM**：默认 stop-grad，也就是 knowledge insulation（π0.5 / KI-VLA，Driess et al. 2025 [需核实编号]）。在 1.5k 条数据上，动作回归梯度进入 LoRA 很容易破坏语言和世界知识，而开放类别泛化恰恰依赖这些知识。
- **诚实的定位**：接口 A 本质上是"以 Qwen3-VL-4B 为冻结文本编码器的 T2M 模型"，相当于用自有生成器替换 Kimodo。
  - 相对 Kimodo 的好处：可以在 PICO 域微调；能叠加 B 残差；动作 token 有似然，可以做蒸馏和打分；部署时是一个栈。
  - 坏处：先验只有 HumanML3D 约 28 小时，Kimodo 的先验更大；而且 Kimodo-FT 已经试过"在 PICO 上微调生成器"，数据加多后没有稳定收益 [本地，memory 91]。
  - 所以"自有头胜过 Kimodo"是一个**要用实验回答**的问题，不能默认成立。

### 2.4 "LoRA 可选"——实际上在 1–2 卡上近乎必选

| 选项 | 评价 |
|---|---|
| VLM 完全冻结 | 零样本格式合格率只有 49% [本地]，V→T 不可用；只能走接口 B/C，即负结果那种配置。不建议 |
| **LoRA（LLM 部分）** | 推荐。OPD 时 teacher 和 student **共享一个基座**，各挂一个 adapter，显存只多一个 LoRA |
| 全参微调 4B | bf16 权重 8.9 GB + 梯度 8.9 + fp32 Adam 35 + fp32 主权重 17.7 ≈ 71 GB，还没算激活 → 需要两卡 ZeRO；1.5k 条上过拟合和遗忘风险高。不建议 |
| 动作头 | 全量训练（1–3 亿参数量级） |

---

## 3. 阶段 4：OPD 自蒸馏（反向 KL）

### 3.1 是什么

student 在自己的分布下采样 y ~ π_S(·|x)，然后在每个位置计算它与 teacher 的**整词表** KL：

```text
L(θ) = E_x E_{y~π_S(·|x)} Σ_t KL( π_S(·|x, y<t) ‖ π_T(·|x, z, y<t) )        （采样路径 stop-grad，GKD 式）
```

- 同尺寸、同 tokenizer，可以直接算整词表的精确 KL，方差比单样本估计低。
- 等价的 RL 写法：把 r_t = log π_T(y_t) − log π_S(y_t) 当作逐 token 优势（Thinking Machines "On-Policy Distillation" 博客）[常识]。
- 相关工作：GKD（Agarwal et al., arXiv 2306.13649）[常识]；MiniLLM 的反向 KL（arXiv 2306.08543）[常识]；Qwen3 技术报告中的 on-policy 强到弱蒸馏 [常识]。
- 显存：整词表 logits 为 151,936 × 256 token × 2 B ≈ 78 MB/序列/模型，可以接受。

### 3.2 同尺寸自蒸馏必须有特权信息

teacher 和 student 同权重、同输入时，KL ≡ 0。同尺寸蒸馏也没有压缩收益。所以这条路的唯一意义是**训练期特权信息 → 测试期只看视频**，属于 LUPI / context distillation（Snell et al. 2022）[常识]。

| 特权信息 z | 覆盖率 [本地] | 评价 |
|---|---|---|
| **PICO 计划标签 / plan_raw**（方向、距离、偏航、时长、姿态） | 860/860 | **最推荐**。直接命中已知瓶颈（根轨迹、方向、时长）；只是"提示"，不是答案 |
| mental_reasoning + current_state | 仅正式 636；补充 224 和新上传批次都没有 | 推荐作第二种。"为什么"是部分可从视频推断的信息 |
| GT reaction / gt3p | 全覆盖 | 等于把答案给 teacher → OPD 退化为 on-policy SFT（DAgger 式），只用作上限或消融 |
| 反应发生后的源视频画面（hindsight） | annotation 里有 `src_title`、`clip_start/end`，但**源视频是否在盘上未确认**（抽查 1 条没找到） | 叙事最强（"把事后所见蒸馏成事前预判"），先确认数据 |

### 3.3 为什么用反向 KL，以及四个陷阱

**好处：**
- mode-seeking：输出一个连贯的反应，不会把"后退"和"举手"混成一句；
- 在 student 自己的状态上训练，修正 exposure bias；
- 信号稠密，不需要奖励模型，**训练期不调用 Kimodo**（路线 6 的 DPO 用了 13,956 次生成，约 10.5 GPU·h）[本地]。

**陷阱 1（理论）：几何平均。**
z 从 x 推不出来时，最小化 E_z KL(S ‖ T_z) 的解是 S*(y|x) ∝ exp(E_z log T_z(y|x))，即各 teacher 条件分布的**几何平均**。它只把概率放在"每种 z 下都不算错"的 y 上 → 泛化、保守的反应（"我环顾四周"）。前向 KL 的解则是混合分布 E_z T_z。由于每段视频只有一个 z，实际表现是 train 上背答案、test 上变泛。
- 监控：输出多样性、计划分布 TV、静止率、反应长度。
- 对照：前向 KL、GKD 的广义 JSD（β 可调）。

**陷阱 2：模仿鸿沟。**
teacher 会用 student 看不到的信息写出具体细节，student 学会的是"编细节"。
- 对策：teacher prompt 要求"只用 cutoff 前可见的线索"；z 只给提示，不给答案；再混入小比例 SFT / 前向 KL 项。

**陷阱 3：teacher 不会用 z。**
SFT 时 prompt 里从来没有 z，teacher 可能忽略它。
- 对策："teacher-SFT"：同一个 LoRA 用混合 prompt 训练（一半带 z、一半不带）。带 z 时当 teacher，不带 z 时作为 student 的初始化。
- **teacher 必须冻结**（或用很慢的 EMA）。否则存在平凡解：teacher 学会忽略 z，KL 归零，什么都没学到。

**陷阱 4：单一真值惩罚其他合理反应。**
带答案的 teacher 很尖锐，同样合理的其他反应会被压掉。这与"每段视频只有一条动捕"是同一个问题，数据量解决不了。

### 3.4 动作头在 OPD 里的位置

- 接口 A 下，给定相同文本时 teacher 和 student 的动作头输入完全相同 → **只做文本级 OPD**，动作质量通过更好的文本间接提升。最干净，推荐第一版这样做。
- 接口 B + 离散头下，可以再加动作 token 级 KL：teacher 的隐状态带有 z 语境。但梯度会从动作头进入 student 的 LoRA，与 knowledge insulation 冲突，列为加分项。
- KL 只在 `<reaction>/<exec>/<plan>` span 上计算。`<plan>` token 可以加权，因为它对应决策瓶颈。

### 3.5 进入 OPD 前的门（G-T）

1. 带 z 的 teacher 在 VAL 上的 greedy 输出（经 Kimodo 或自有头），在**独立指标**（原版 TMR R@10、计划准确率）上必须明显好于 student。否则 OPD 没有可学的东西。
2. 初始 teacher-student KL 要有量级；去掉 z 后 KL 应显著下降，以证明 teacher 确实在用 z。

### 3.6 与路线 6 DPO 的关系（为什么 OPD 可能补上 DPO 的缺口）

- 路线 6 的 DPO 没通过 G4：语义 t2m sim −0.028 [−0.053, −0.005]，时长分布偏移 [本地]。
- 诊断结论是"奖励可迁移，候选池太差"：温度 1.0 的样本平均不如 greedy [本地]。
- OPD **不依赖候选池里恰好有好答案**：teacher 对 student 自己写的每个 token 都给出方向正确的修正。
- 可以复用同一个事先登记的 G4、同一个 VAL、同一个 cluster bootstrap，得到干净的"SFT vs DPO vs OPD"比较。

### 3.7 新颖性边界

- 方法本身已有先例：LLM 领域的 on-policy 蒸馏 [常识]，以及 2026 年带特权语境的 on-policy *自*蒸馏（例如 SDFT "Self-Distillation Enables Continual Learning"、OPSD）[需核实]。
- 所以贡献只能落在：
  - 以真人动捕计划 / 事后画面为特权信息，服务于**前瞻式**具身反应；
  - 动作落地；
  - 留一类泛化。
- SDFT 一类工作声称"自蒸馏比 SFT 遗忘更少"[需核实]。这正好可以在 UMRM 上检验：留一类（LOCO）表现 + 通用 VQA 保留率。

---

## 4. 算力（1–2 张 A100-80G）[估算]

| 步骤 | 显存 | GPU 时间 |
|---|---|---|
| 阶段 1 SFT（LoRA） | 12–18 GB | 1–2 h |
| teacher-SFT（混合 prompt） | 同上 | 1–2 h |
| HumanML3D caption 隐状态缓存（多层） | 约 10 GB | < 0.5 h；4 层 fp16 缓存约 20 GB 磁盘 |
| 动作头预训练 | 10–30 GB | 12–24 h |
| 动作头 PICO 微调 | 同上 | 2–4 h |
| OPD（一个基座 + 2 个 LoRA；每轮 1.5k × K=4 次 rollout，约 100 token） | 25–40 GB | 每 epoch 2–4 h，共 2–3 epoch |
| VAL 评测（Kimodo 臂，576 次生成） | 约 4 GB | 约 20 min/次 [本地实测] |

- 合计：核心约 2–3 GPU·天，含消融约 4–6 GPU·天。
- 工程注意：
  - CPFS 小文件很慢 → 沿用路线 6 的做法，把环境和权重复制到 `/tmp` [本地]；
  - K 次采样在 bs≥2 时，预填充会算全长 logits 导致 OOM → 用 `logits_to_keep` 或分块 [本地]。

---

## 5. 执行顺序与门（按约 4 周）

| 周 | GPU0（V→T / OPD） | GPU1（T→M） | 门 |
|---|---|---|---|
| W1 | 下载 4B（待批准）、新环境、4B SFT → Kimodo 评测 | 缓存 HumanML3D 多层隐状态 → 动作头预训练 → HumanML3D 自检 | **G-A**：4B SFT 不显著劣于 7B SFT（否则换 8B）；**G-B**：动作头在 HumanML3D 上达到 T2M-GPT 量级 |
| W2 | teacher-SFT → G-T → OPD 第 1 轮 → G4（对 SFT / DPO / MBR@8） | PICO 微调；gt3p oracle 文本下比较自有头和 Kimodo | **G-C**：自有头在 VAL 独立指标上不劣于 Kimodo，否则自有头降为消融 |
| W3 | 2×2 合流：{SFT, OPD} × {Kimodo, 自有头}；LOCO；消融（层、接口 A/B、z 的种类、反向 / 前向 KL / SFT） | | |
| W4 | 冻结方法，TEST 只跑一次，写作 | | |

---

## 6. 评测注意

- **评价器偏心**：原版 TMR 在 HumanML3D 上训练，而自有头也在 HumanML3D 上预训练，会占到域内便宜。这与已观察到的 TMR-SOMA 偏爱 Kimodo 是同类问题 [本地]。要同时报告 PICO 空间指标，最好再加一组小规模人评。
- 主指标仍用检索类：R@10、中位排名、t2m sim、同类 R@3。MPJPE 只作辅助，因为它偏爱平均动作 [本地]。
- 正式 636 的成员和 split 不改；实验划分沿用 v860 的冻结划分，或用继承方式扩到 1.5k [本地]。

---

## 7. 需要用户拍板

1. 下载 Qwen3-VL-4B-Instruct（约 9 GB）？还是先用本地完整的 8B / 2B 起步？
2. 特权信息 z 选哪种？是否值得先确认源视频在盘上，以便做"事后画面"teacher？
3. 主张落在哪里：自有动作头替换 Kimodo，还是 OPD 作为主贡献、动作头作为组件？
4. 数据版本：v860，还是等 1.5k 继承划分？
5. 用哪两张卡、在什么时间窗口。查看时 8 张卡都空闲，但这是共享环境。
6. 第一版接受 HumanML22 口径，还是必须原生 24？

---

## 8. 用户补充（2026-10-15）：VPP2 的事件级做法能否用到反应动作长短上

**原文要点**（arXiv 2610.10270，已读 HTML 版 §3.1、§4）：
- **事件切分**：用速度极小值、夹爪开合等启发式找候选边界，再由 VLM 合并和调整，得到 1–12 s 的完整子任务。
- **Stage 1（事件级）**：每个事件均匀取 N=48 帧作为目标（加 o₀ 共 49 帧），配详细 caption（操作过程、哪个执行器、目标物体），在 Wan2.1-I2V-14B 上继续预训练 3 万步。
- **Stage 2（后训练）**：固定帧数采样不同长度的事件，会使帧间隔不一致、难以和动作对齐，因此再后训练为固定时长的 chunk（人手 2 s / 机器人 8 s），并做一致性蒸馏。
- **消融**：直接训固定时长的 VPP2-Fixed-Step 指令跟随更差。作者把收益归因于"完整事件 + 详细 caption"建立了稳定的语义→轨迹映射。

**本地诊断**（`vt_head_opd_v860/runs/diag_event_level.json`、`diag_truncation.json`）：
1. **v860 动作目标有截断**：train 19.9%（116/582）、val 8.3% 的反应在构建时被 88 帧（4.4 s）上限截断，原始录制中位 5.44 s、最长 10.3 s。其中 61% 被截掉的尾段里有实质动作（尾段中位 1.0 s、活动帧占 85%、速度为前段的 72%）。例如"跑过去扶起后座的朋友……再回去扶前座的朋友"只保留了前 4.4 s。这正是 VPP2 说的"只看到抽屉开了一半"：文本描述完整事件，目标只有半段。路线 6 的 SFT/DPO、HERO-FT 和 Kimodo 臂都受影响。
2. **时长跨度窄**：PICO 反应 p5–p95 为 1.8–4.4 s（跨度 2.4 倍），HumanML3D 为 2.2–10 s（4.5 倍），VPP2 为 1–12 s。动捕录制本身基本已是"事件"（前导静止中位为 0）。
3. **节奏错位不是当前主要误差**：固定 88 帧的 HERO-FT 做事件归一化（两边都重采样到 48 帧）后，MPJPE 反而从 0.648 升到 0.749，根相对误差从 0.149 升到 0.157。误差主要来自"做什么"。

**判断**：
- **值得做、成本低（E1）**：用完整事件（不截断，≤196 帧 / 9.8 s）作为动作头的微调目标；`<plan>` 的 duration 档位扩到 > 4.4 s。评测时 4.4 s 口径保留作为主表（与 route6 可比），再加"完整事件"臂。
- **不建议照搬固定 N 帧重采样**：T2M 领域本来就是"一条 caption ↔ 整段动作 + 显式长度"的事件级目标（MoMask/T2M-GPT 生成变长序列，长度由估计器或我们的 `<plan>` duration 给出）。VPP2 需要 Stage 1→Stage 2 两步，是因为视频模型只能生成固定帧数。我们直接按实时变长生成，已经同时满足"完整事件"和"固定时间尺度"，不需要再绕一圈。如果做固定 N 帧归一化（E2），按 VPP2 的经验必须配 Stage-2 式回到实时尺度的后训练。另外反应的节奏本身有语义（惊跳快、环顾慢），归一化后需要靠 duration 把节奏补回来，而 duration 从视频很难预测（route6 G2b 线性探针未超过多数类）。
- **详细 caption 这一半可以借鉴（E3）**：在 `<exec>` 中写明主导肢体和左右、方向、幅度、先后阶段。其中方向、距离、转角、时长可以直接从 PICO 动捕自动提取（与 OPD 的特权提示 z 同源）。路线 6 的 G1 显示"保持原意的改写"对 Kimodo 没有帮助；对从 HumanML3D 预训练、可在 PICO 上微调的自有头是否有效，需要实测。
