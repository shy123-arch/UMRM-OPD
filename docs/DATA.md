# 数据卡：umrm-vho-data-v1197aug（ModelScope 私有数据集）

## 完整 v1197 动捕归档

2026-10-20 的完整 v1197 归档另由 `tools/build_v1197_release.py` 从 2026-10-15 冻结源构建，包含 1197 条真实样本（正式 636 + 补充 224 + 补充 337），不含镜像样本。归档逐条提供反应前片段、去除采集会话信息的 PICO 动捕、任务文本及冻结的 88 帧动作目标。构建器仅保留字段白名单，从视频容器去除元数据，不带标注者、上传者、原始路径、源标题或采集 session。`split_original.json` 只是冻结源的历史划分；它不更改正式 636 benchmark。完整片段仅可存于私有数据集，不适用现有公开许可。

这个归档与下文用于 v1197aug 训练的包各有用途；v1197aug 包采用独立的来源隔离实验划分，并不包含 v860 的 182 条封存 TEST。

获准访问私有数据集后，可用 ModelScope 的快照下载接口取得完整文件树（`MsDataset.load` 适合表格型数据，不会自动整理本包的 MP4 与 NPZ 资产）：

```python
from pathlib import Path
from modelscope import dataset_snapshot_download

root = Path(dataset_snapshot_download(
    "ma3re111/UMRM",
    allow_patterns=["releases/2026-10-20/v1197/**"],
))
data = root / "releases/2026-10-20/v1197"
# 条目索引：data / "items.jsonl"；视频：data / "clips"；动捕：data / "pico"
```

已用同一命令的 `allow_patterns=["releases/2026-10-20/v1197/README.md"]` 小文件子集验证目录解析；下载完整包约 2.3 GiB。

**用途与许可**：仅限非商业学术研究，申请制获取，不得二次分发。视频片段来自 YouTube、Bilibili 等平台的第一人称视频，按平台条款和合理使用原则采集。承载这些数据的仓库必须保持私有；本GitHub仓库仅发布代码与汇总结果。

## 组成

| 组 | 条数 | 说明 |
|---|---|---|
| train | 1652 | 真实 840（正式 636 中的 430 + 0927 回传 152 + 1015 回传 258）+ 左右镜像 812 |
| val | 125 | 其中 96 条就是 v860 的 VAL；用于选模和全部登记过的比较 |
| sealed | 50 | 1015 回传中的新来源（7 个来源，以 B 类为主），只用于确认实验 C2 |
| （未发布）test | 182 | v860 的 TEST，从未读取，不包含在发布包里 |

划分按来源隔离：用 src_id、规范化标题和 BV 号做并查集，同一来源只出现在一个组；数据扩充时继承 v860 的划分。

## 文件

| 路径 | 内容 |
|---|---|
| `data_manifest.json` | 每条一行，字段：`reaction`（第一人称）、`exec`（gt3p，第三人称动作描述）、`plan`（从动捕提取的 5 个离散字段）、`plan_raw`（时长/位移/转角）、`mental_reasoning` + `mr_src`（human / human+mirror / generated:qwen3-vl-8b；**只给教师用**）、`target`（SFT 的三行目标）、`category`、`source`、`cohort`、`mirror_of`、`staged_film`、`eval_exclude`、`clip` |
| `clips/<id>.mp4` | 1015 段反应前视频；镜像样本复用源视频，缓存时再做水平翻转 |
| `pico/<id>.npz` | 原始 PICO 动捕（24 关节，约 80 帧/秒，带时间戳） |
| `motion/humanml263_full.npz` | 完整事件的 HumanML263 特征（≤196 帧 = 9.8 秒，20 帧/秒），含镜像 |
| `motion/humanml263_capped.npz` | 截断在 88 帧（4.4 秒）的版本 |
| `motion/targets.npz` | 规范化后的 24 关节目标（88 帧 + 有效帧掩码），供评测器使用 |
| `compiler_stats.json` | 计划编译器的统计量（只用 TRAIN 计算） |
| `eval_assets/` | LLM2Vec 文本缓存、SOMA 拟合、原评测器用的骨长中位数 |
| `meta/` | 镜像报告、"做什么"生成记录与校准、发布报告 |

## 文本来源和已知问题

- **reaction**：正式数据是人工英文；补充数据中有上传者英文原文的直接用，只有中文的由 Qwen3-8B 翻译（254 条）。
- **gt3p**：正式数据是 Qwen3.6 改写版；补充数据由 Qwen3-8B 改写。初次改写有 46% 超长或凭空添加步数，用严格提示词重写，再修复漏掉的动词，最终 0 条不过门。v860 补充 224 条的旧改写没有重做（约 37% 超长）。
- **mental_reasoning（生成版）**：由 Qwen3-VL-8B 根据视频和真实反应写出。在 40 条正式数据上与人工版本的内容词重合率为 0.20（随机配对 0.05），与反应原文的重合率偏高（0.63，人工版为 0.41）。
- **镜像**：重新提取的计划标签与"直接左右互换标签"100% 一致；有 28 条因 left/right 不能确定是方位词而不做镜像。
- **类别**：补充数据的类别为会话判断，未经人工审核。24 条真人影视短片：9 条在 train（保留），15 条在未发布的 test。
