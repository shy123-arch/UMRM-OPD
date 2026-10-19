# 原实验源码快照

本目录保存周四以来四个实验目录的顶层 `.py`、`.sh`、`README.md`、`RESULT.md`。不包含其`runs/`、日志、数据或权重。

| 目录 | 原实验目录 |
|---|---|
| `v860/` | `umrm_model/experiments/vt_head_opd_v860/` |
| `v1197/` | `umrm_model/experiments/vt_head_opd_v1197/` |
| `v1197aug/` | `umrm_model/experiments/vt_head_opd_v1197aug/` |
| `lastlayer/` | `umrm_model/experiments/vt_lastlayer_v1197aug/` |

这些脚本依赖原有兄弟目录和绝对路径，留作历史复现依据。**不要在本归档目录直接运行训练/守护脚本。** 新开发以`pipeline/`为主；原始路径对应关系与每个文件的导入前哈希记录在`docs/SOURCE_SNAPSHOT.json`。

快照包含早期试验和失败方案，文件存在不代表它是当前推荐配置。比如`a2_mr_obs.py`是曾尝试的LLM改写方案；后续采用抽取式`a2b_mr_obs_extractive.py`。实验日期和结论见[时间线](../docs/EXPERIMENTS.md)。
