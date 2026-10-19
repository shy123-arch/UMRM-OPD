# 分析与历史编排

- `analyze_ego_pretraining.py`：已有PT-C/SFT结果的CPU离线分析，可用参数指定路径。
- `analyze_retention_opd.py`：ER19保留分析和OPD增益之差，可用参数指定路径；缺失臂标为pending。
- `test_retention_opd.py`：CPU小模型验证半程诊断不改变训练权重、诊断检查点不参与选模，以及聚类bootstrap一致性。
- 其他`decide_*.py`、`chain_*.sh`、`queue_*.sh`等是原服务器的历史分析/编排；部分依赖对应实验目录的`common.py`，不能假定在本目录直接执行。完整上下文保存在[experiments](../experiments/README.md)。

公开结果放`docs/results/`；运行输出放Git忽略的目录。保存聚类单位、比较方向、单seed限制和数据版本，不用旧评测输出覆盖新登记结果。
