# 实验时间线与代码索引

截至2026-10-19。本页提供导航；数值与具体比较以对应登记、结果和汇总JSON为准。历史结果中的机制解释不自动成为已证实事实。

| 日期 | 问题 / 实验 | 代码与登记 | 结果 |
|---|---|---|---|
| 10-15 周四 | 4B V→T、自有T→M头、OPD v1；完整事件初探 | [v860快照](../experiments/v860/)、[登记](prereg/v860.md) | [v860结果](results/v860.md)：分段链路可用，OPD主门未通过，方向多数类偏移 |
| 10-16 周五 | 扩到v1197，人工reason教师、OPD v2、视觉分支 | [v1197快照](../experiments/v1197/)、[登记](prereg/v1197.md) | [结果](results/v1197.md)：教师语义优势增加，自有头部分正信号，主门未通过 |
| 10-16至17 | 镜像TRAIN1652、生成reason、v2复现、OPD v3/v4 | [v1197aug快照](../experiments/v1197aug/)、[登记§2–4](prereg/v1197aug.md) | [结果](results/v1197aug.md)：部分复现；扩展后没有稳定提升 |
| 10-17 周六 | R-full / R-obs，纯反向与混合KL，OPD v5 | [opd.py](../pipeline/opd.py)、[登记§5](prereg/v1197aug.md) | [gates_v5](results/gates_v5.json)：部分配置缓解偏移，收益未建立 |
| 10-17 | A4/A36/S36/V36语义接口 | [lastlayer快照](../experiments/lastlayer/)、[登记](prereg/lastlayer.md) | [结果](results/lastlayer.md)：A36为当前参照0.2907 |
| 10-17 | 无特权8B教师→4B，OPD v6及reaction/exec限定版 | [opd.py](../pipeline/opd.py)、[登记§6](prereg/v1197aug.md) | [gates_v6](results/gates_v6.json)：能向教师靠拢，任务收益未建立 |
| 10-17至18 | 共识选择、随机候选对照、32帧、reason输出/多任务、R-chain教师 | [v1197aug代码](../experiments/v1197aug/)、[登记§7–8](prereg/v1197aug.md) | [gates_t7](results/gates_t7.json)、[gates_r](results/gates_r.json)：没有稳定增强 |
| 10-18至19 | PT-C：外部ego预训练→UMRM SFT→固定A36与探针 | [ego_pt.py](../pipeline/ego_pt.py)、[登记§9](prereg/v1197aug.md) | [完整分析](results/ego_pretrain_20261019/REPORT.md)：部分探针改善，SFT后减弱，G-E1未通过 |
| 10-19起 | ER19：匹配低学习率SFT→同一reason教师纯反向OPD | [队列](../tools/queue_retention_opd.py)、[登记](prereg/ego_retention_opd_20261019.md) | 发布时在运行，尚无最终结论 |

## 贯穿路线的当前假设

1. ego监督改善视频情境到反应文本的推理能力。
2. 这份能力需要经过任务SFT保留到进入OPD的学生上。
3. 更好的学生基础可能直接改善动作，也可能增加从mental reasoning教师获得的OPD收益。

三项分别验证。外部选择题不等于完整的视觉因果推理；未显著不等于已证明等效；KL下降不等于动作质量提高。ER19主比较为 `(D−C)−(B−A)`，同时要求检查D−C的实际收益。

## 数字口径

- 正式benchmark：636；642仅历史溯源。
- v860：TRAIN582 / VAL96 / TEST182。
- v1197：TRAIN840 / VAL125 / 新来源封存50 / TEST182。
- v1197aug：TRAIN1652（840真实+812镜像），VAL125。50条封存块曾用于登记的C2；182条TEST保持封存。
- R@10是TMR动作—文本检索，不是人工反应正确率。不同VAL规模、动作头或教师配置之间不能直接比较绝对分数。
