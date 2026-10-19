# 数据、权重与存储

本仓库只管理源码、文档和汇总结果。所需资产独立获取，按各自访问条件与许可使用。

| 资产 | 用途 | 当前管理方式 |
|---|---|---|
| Qwen3-VL-4B-Instruct；部分历史实验8B | VLM、文本编码与教师 | 官方模型来源；本地SSD缓存 |
| UMRM v1197aug数据包 | 视频/反应/mental reasoning/PICO、冻结split | 申请制私有发布包；见数据卡 |
| HumanML3D | 文本—动作预训练 | 按项目官方流程取得 |
| HERO/ViMo、Kimodo、TMR/LLM2Vec | 动作先验、对照与评测 | 外部工程和权重；见NOTICE |
| Ego4D Goal-Step源视频及标注 | ego预训练 | 原服务器OSS挂载`/mnt/data/oss/.../Ego4d/v2_packed` |
| EgoNormia、EgoToM | 训练/外部探针 | 独立许可与版本记录；见ego数据调研 |
| EgoRe Why片段 | ego预训练 | 目前部分引用另一工作区，本次未迁移 |
| 视频张量缓存、LoRA、动作头权重 | 高速训练与推理 | 本地缓存/实验产物；Git忽略 |

大视频主库已经在OSS；“全部数据已经迁至OSS”不成立。本次代码整理不下载、上传或迁移动捕/视频资产。

## 数据边界

- 正式636保持不变；实验v1197aug的划分独立冻结。
- 182条TEST不进入本代码发布的数据包；sealed50只按已登记实验处理。
- 公共Git仓库不收录逐样本manifest、reason、视频、动捕、探针选项答案、预测明细或`cases.json`。
- 文档可保留少量方法说明和历史定性示例，不构成数据集发布。汇总JSON不提供完整训练样本。
- `tools/fetch.sh`和`tools/modelscope_upload.py`是已有私有资产管理辅助工具；公开代码不意味着有数据仓库访问权限。上传工具默认dry-run。

## 外部链接

- [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [HumanML3D](https://github.com/EricGuo5513/HumanML3D)
- [HERO](https://github.com/JackYu6/HERO_release)
- [Kimodo](https://github.com/nv-tlabs/kimodo)
- [Ego4D](https://ego4d-data.org/)
- [本项目ego数据调研](EGO_CAUSAL_DATASETS.md)
