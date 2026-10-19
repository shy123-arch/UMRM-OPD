# 第一人称因果数据调研：EgoPlan-IT、EgoTaskQA、EgoCOT、EgoNormia、EgoToM（2026-10-18）

**用途**：给 VLM 做"第一人称视频 → 因果解释 + 下一步"的预训练（视频→文本），再接现有的 SFT 和动作头。
**来源**：
- arXiv 原文摘要（arXiv API）；
- 官方 GitHub README（GitHub API）；
- HuggingFace 数据集卡片和文件列表（HF API，只看了元数据，没有下载任何数据）；
- EgoTaskQA 论文 PDF 的数据说明部分；
- Ego4D 官方文档。

**本次未下载任何数据集文件。**

## 一、总表

| | EgoPlan-IT | EgoTaskQA | EgoCOT | EgoNormia | EgoToM |
|---|---|---|---|---|---|
| 论文 | 2312.06722（IJCV） | 2210.03929（NeurIPS D&B 2022） | 2305.15021（NeurIPS 2023，EmbodiedGPT） | 2502.20490（ACL Findings 2025） | 2503.22152 |
| 规模 | 训练集 **50K**；另有 Val 3,355 / Test 1,584（Test 答案不公开） | **40K** 问答（平衡前 368K），**2K** 个视频片段；按 3:1:1 分 train/val/test | HF 卡片标注 10 万–100 万条（未核实确切数） | **1,853** 题，来自 1,077 个视频，97 个场景 | **1,039** 题：动作 354 / 信念 334 / 目标 351 |
| 因果形式 | 任务目标 + 已完成步骤 → **下一步动作** | 模板生成：描述、预测（what will）、**解释（what caused）**、**反事实（what if）**；涉及动作依赖与效果、意图与目标、对他人信念的推断 | 片段描述 + **分步计划**（子目标分解） | **动作前 5 秒片段** + 5 个候选行为，每个都附**理由**，标明正确项和"合理项" | 用因果心智模型生成：佩戴者的**目标、当下信念、下一步动作** |
| 答案形式 | 短动作短语（如 "grab onion"），有负例 | 短答案（物体、动作、状态、是/否） | 文本计划 | 自然语言行为 + 理由 | 四选一 |
| 有没有自然语言的"为什么" | 无（只有目标） | 无（因果是结构化的） | 弱（计划步骤） | **有** | 有（选项本身包含信念和意图） |
| 视频来源 | **EPIC-Kitchens-100 + Ego4D** | LEMMA（第一人称家务、多人协作） | Ego4D | Ego4D | Ego4D |
| 标注许可 | 代码 BSD-3；数据未单独声明 | **CC BY-NC-SA** | 仓库写明"仅限研究用途"；HF 镜像标为 apache-2.0 | HF 卡片 **CC BY-SA 4.0**；代码 MIT | **CC BY-NC，并明确写"仅用于评测（benchmarking purposes only）"**；含 GPT-4 Turbo 输出，受 OpenAI 条款约束 |
| 视频需要 Ego4D 协议吗 | Ego4D 那部分需要；EPIC 部分不需要 | **不需要**（视频在作者网站上） | 需要 | 视频派生自 Ego4D，按规定应受 Ego4D 协议约束（HF 上虽然不设门槛，可以直接下） | 需要（仓库只有标注，片段要用脚本从 Ego4D 原视频裁出） |
| 下载方式 | JSON 在 **Google Drive**；视频要分别下载 EPIC-100（官方脚本）和 Ego4D（CLI） | 问答和视频在作者网站（**Google Drive**）；HF 有第三方的问答 JSON 镜像 `lotsoliu/EgoTaskQA`（60 MB，无视频） | 基础版在 **Google Drive**；完整版在 HF `wofmanaf/ego4d-video`（第三方，52 个分卷，约 **558 GB**，含视频） | HF `open-social-world/EgoNormia`（不设门槛；parquet 2.2 MB + 每题一个视频目录） | GitHub `facebookresearch/EgoToM`（CSV 和 prompt）；HF 有标注镜像 `kfkas/egotom-benchmark` |
| 大致存储 | 标注很小；视频取决于用到的 EPIC 和 Ego4D 子集。参考：lmms-lab 打包的 EgoPlan **Val 3,355 条加视频约 122 GB** | 问答约 60 MB；LEMMA 视频大小未核实 | 约 558 GB（第三方完整版） | 按 5 条抽样粗估：全部约 150 GB，只要 `video_prev.mp4` 约 30 GB | 标注约 10 MB；视频取决于涉及多少条 Ego4D 原视频（未统计） |

## 二、获取上的硬约束

1. **Google Drive 在这台服务器上访问不了**（实测超时）。EgoPlan-IT 的 JSON、EgoTaskQA 的视频和标注、EgoCOT 基础版都只放在 Drive 上，需要在能访问的机器上下载后再传到服务器。
2. **Ego4D 协议**：
   - 在 ego4d.dev 签署，约 48 小时后邮件发来 AWS 凭证；
   - 代表整个机构签署需要主管级签字人；
   - 凭证 **14 天后过期**，要求下载到本地，过期可以续签；
   - CLI：`ego4d --output_directory=... --datasets {annotations|clips|full_scale|video_540ss} --video_uids/--video_uid_file ...`，可以按 UID 只下载用到的视频；
   - 体量：完整原始视频约 7 TB，benchmark 片段约 1 TB，标注约 2 GB，只要旁白约 350 MB，另有缩到 540p 的版本。
3. **EPIC-Kitchens-100**：官方脚本 `epic_downloader.py --videos/--rgb-frames --participants ... --specific-videos ...` 从 data.bris.ac.uk 下载（服务器可访问）。许可一般是 CC BY-NC 4.0（未逐字核实）。
4. **第三方镜像**（`wofmanaf/ego4d-video`、`lmms-lab/EgoPlan` 等）把 Ego4D 视频重新分发，许可标注不可靠。建议只用作者官方的标注，视频凭我们自己的 Ego4D 授权去下载。
5. **全部是非商业或仅限研究**；EgoToM 明确写了仅用于评测。

## 三、和"有因有果"需求的匹配

- **最贴近**：EgoNormia。动作前 5 秒的片段，加上"该做什么 + 为什么"，格式几乎就是我们的任务（第一人称、截在动作之前）。但只有 1,853 条。
- **因果结构强、但没有自然语言解释**：EgoTaskQA（what caused / what if / what will）、EgoPlan-IT（目标 → 下一步）。可以用规则模板把它们改写成"<reason> 为了 X，已经做了 A、B → <next> 做 C"，不需要大模型改写。
- **规模大、但因果偏弱**：EgoCOT（片段描述 + 分步计划）。
- **只适合评测**：EgoToM（许可如此）。可以和 EgoNormia 的一部分留出题一起，作为"第一人称因果推理学到了没有"的探针。
- **共同的局限**：全部是日常任务、家务或社交规范，不是突发事件下的身体反应；都没有音频线索。

## 四、建议的组合

| 用途 | 数据 | 规模 | 所需许可 |
|---|---|---|---|
| 预训练主体（因果 → 下一步） | EgoPlan-IT（先只用 EPIC 部分，不依赖 Ego4D） + EgoTaskQA 的预测、解释、反事实题 | 约 5 万 + 约 2.4 万条训练题 | EPIC CC BY-NC；EgoTaskQA CC BY-NC-SA |
| 贴近反应和社交 | EgoNormia（留 20% 作评测） | 约 1,500 条 | CC BY-SA 加 Ego4D 协议 |
| 扩大规模（可选） | EgoCOT 标注 + 用 Ego4D 视频按 UID 下载 | 10 万条以上 | Ego4D 协议 |
| 评测探针 | EgoToM 全部 + EgoNormia 留出部分 + 我们 benchmark 的 D2 因果题（只用 VAL） | 约 1,400 题 | Ego4D 协议 |

**不依赖 Ego4D 就能先启动的部分**：EgoPlan-IT 的 EPIC-Kitchens 部分，加 EgoTaskQA。两者都需要先有人从 Google Drive 下好标注（EgoTaskQA 还包括视频），再传到服务器。

## 五、本地和 OSS 现有数据盘点（2026-10-18 下午；只读查看，没有改动任何人的目录）

**查找范围**：OSS 挂载点 `/mnt/data/oss`（ossfs2），以及共享盘 `/mnt/workspace`（CPFS）。按名字搜索深度 ≤ 6，然后逐个打开相关目录核实。

| 数据 | 位置 | 内容 / 规模 | 视频本地有没有 | 能不能直接用 |
|---|---|---|---|---|
| **Ego4D v2（完整）** | `/mnt/data/oss/eckes/EgoData/Ego4d/v2_packed/` | `full_scale/` 有 9,811 个完整视频（另有 tar 分卷，每卷约 52 GB），`annotations/` 是全套标注 | 有（抽查 EgoNormia 用到的 4 个视频，全都在） | 能（需要确认我们的 Ego4D 授权，见下） |
| ├ 旁白 | `annotations/narration.json` | 约 500 万条带时间戳的旁白（两轮标注） | 有 | 能构造"截在下一个动作之前 → 预测下一个动作" |
| ├ Goal-Step | `goalstep_{train,val}.json` | 训练集 583 个视频，242.9 小时；13,342 个步骤、18,224 个子步骤；带目标描述和步骤描述 | 有 | **能构造"目标 + 已完成步骤 → 下一步"，相当于 Ego4D 版的 EgoPlan** |
| ├ 长期 / 短期动作预测 | `fho_lta_*`、`fho_sta_*` | 长期预测训练集 63,956 个片段，短期预测 98,276 条 | 有 | 有"下一个动作"标签，没有"为什么" |
| └ 社交（AV） | `av_*` | 训练集 153 个视频（谁在对我说话、谁在看我） | 有 | 社交线索 |
| **Ego4D 片段（已切好）** | `/mnt/workspace/Shawn/projects/RynnScale/stagea_data/complete/cog_egoc/sources/ego4d-clips` | 180,051 段，10 fps，短边 360，共 291 GB | — | 能（EgoRe 直接引用这些片段） |
| **EgoThinker / EgoRe**（CC BY 4.0 标注） | `…/cog_egoc/egothinker/EgoRe{,_clips}` | `ego4d_cot` **51,385 条**推理链（回答中位数 118 词）；`ego4d_long` 504,450 条，其中以"Why"开头的 **84,057 条**，粗略计入意图和下一步类约 168K；`ego4d_short` 118,239 条 | 有（51,383 / 504,408 条已对应到本地片段） | **能，是目前最大的第一人称"为什么"类数据**；但标注由大模型生成，片段多半已经包含结果，要截短才能变成预测式样本 |
| EgoPlan 训练集 | `…/egothinker/Ego-Related/egoplan_train.json` | 50,285 条，**全部来自 EPIC-Kitchens** | **没有**：EPIC 在 `eckes/EgoData/EPIC-KITCHENS-100/raw` 只有一个 4.9 GB 的分卷 `.tar.gz.00`，不完整 | 暂时不能（EPIC 视频可以从 data.bris 下载，需要批准） |
| EgoTaskQA | `…/egothinker/Ego-Related/egotaskqa.json`（23,734 条），`…/cog_egoc/egotaskqa`（40,323 条） | 标注齐全 | **没有**（LEMMA 视频在 Google Drive，这台服务器访问不了） | 暂时不能 |
| EgoTimeQA、QaEgo4D | `…/egothinker/Ego-Related/egotimeqa.json`、`…/cog_egoc/qaego4d` | 时间定位和记忆类问答，约 126 万行 | 部分有 | 和因果关系不大 |
| EgoLife | `eckes/EgoData/Egolife` | 42 段整天视频（6 人 × 7 天，每段约 13 GB） | 有 | 没有问答标注，不能直接用 |
| Charades-Ego | `…/cog_ocr/sources/charades-ego` | 7,860 段 480p 视频 + 动作标签 | 有 | 没有因果解释 |
| EgoNormia、EgoToM、EgoCOT | 没找到 | — | 它们的源视频都在本地 Ego4D 里 | 只缺标注：EgoNormia 的 parquet 2.2 MB（HF，不设门槛），EgoToM 的 CSV（GitHub），EgoCOT 的 JSON（Drive 访问不了，HF 上有第三方副本）。下载需要批准 |
| 其他（ego-1K、Egocentric-10K、EgoDex、Gen-EgoData、Unihand） | `eckes/EgoData/*` | 多视角三维重建、工厂视频、灵巧操作 | 有 | 和因果推理无关 |

**注意事项**：
- Ego4D、EgoThinker、RynnScale 的数据都在**别人的目录**（eckes、Shawn）里，只能只读引用，不能改动。
- Ego4D 的授权按个人或机构签署。本地有这份副本，不等于我们这个项目已经被授权使用，需要用户确认。
- EgoThinker 的 `egoplan_train.json` 里视频路径指向别的机器（`/mnt/inspurfs/...`），要用的话需要重新映射。

## 六、已下载（2026-10-18 15:10，用户批准）

位置：`/mnt/workspace/Wilson/ck/umrm/external_data/ego_causal/`（64 MB，只读，含 README 和 SHA256SUMS）。

| 数据 | 版本 | 规模 | 本地视频覆盖 | 用途 |
|---|---|---|---|---|
| EgoNormia | GitHub release 09d8a7c + HF parquet | 1,853 题 / 1,075 个视频（剔除正确项为空的 21 题后剩 1,832 题）；verified 200 题 | **100%** | verified 200 题作评测；其余是否进训练，在事先登记时决定 |
| EgoToM | GitHub ed6abe7 | 1,039 题 / 914 个视频 | 1,035 题 | **仅评测**（许可如此） |

EgoNormia 的片段可以从本地 Ego4D 按 [动作时刻 − 5 s, 动作时刻] 裁出，与官方的 `video_prev` 一致。
