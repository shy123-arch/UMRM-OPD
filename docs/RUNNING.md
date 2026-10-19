# 运行与复现

## 1. 三类入口

| 入口 | 用途 | 依赖 |
|---|---|---|
| `pipeline/` | 可配置路径的统一实现 | 授权数据包、模型与三套环境 |
| `experiments/`、多数旧`analysis/`脚本 | 原始实验快照、具体版本追溯 | 原服务器目录结构；不能在归档目录直接启动 |
| `tools/queue_retention_opd.py` | 当前ER19的原服务器后台编排 | 现有PT-C、R-full教师、A36、两个视频缓存及原实验目录 |

此代码快照不会接管或重启原服务器已有进程。正在运行的ER19使用原 `umrm-vho` 路径。队列固定输入/代码哈希；更换代码位置后不要直接对已有任务重新执行。

## 2. 环境和路径

按[环境清单](../envs/README.md)安装Qwen、HERO、Kimodo三个环境。获取[资产](ASSETS.md)后：

```bash
cp configs/paths.env.example configs/paths.env
# 编辑本机路径，然后将配置导入当前shell。
set -a
source configs/paths.env
set +a
```

直接执行Python脚本时需要导出`VHO_*`；shell入口通过`pipeline/_env.sh`读取本机配置。`configs/paths.env`已忽略，不提交认证或本机配置。

## 3. 验证顺序

```bash
python tools/check_repository.py
python analysis/test_retention_opd.py  # 需要torch/numpy，CPU小模型
```

完整GPU smoke为`GPU=0 bash pipeline/smoke.sh`，需要外部资产，且会训练/生成少量样本；应先确认资源配置。历史[smoke记录](results/smoke.md)描述的是原环境验证，不保证在任意新机器直接运行。

## 4. 显式选择配方

- `pipeline/sft.py`：任务SFT；`--init-adapter`延续已有LoRA；`--checkpoint-steps`新增仅诊断检查点。
- `pipeline/ego_pt.py`：ego预训练；混合数据先由`pipeline/data_build/ego_build.py`构建。构建脚本目前仍需原数据布局。
- `pipeline/opd.py`：on-policy token分布蒸馏；纯反向用`--lam_fwd 0`；`--kl-span re`排除plan；教师提示由`VHO_ZMODE`决定，必须与教师训练一致。
- `pipeline/head.py --layers 36`：A36头训练的层选择。文本缓存保留四层，动作头选择36层；应使用对应A36权重。
- `pipeline/train.sh`：历史A/AB及混合KL编排，不代表当前A36+ego+纯反向KL配方。

## 5. ER19

在**原服务器目录布局和资产具备**时，先执行无GPU preflight：

```bash
python tools/queue_retention_opd.py
```

具体训练命令、资源范围、后台入口和停止规则见[登记](prereg/ego_retention_opd_20261019.md)。该工具用`REPO.parent`定位工作区，并包含原服务器环境路径；它不是通用云训练启动器。正式入口`--execute`首先做GPU小样本检查，任何阶段失败都会阻止进入后续阶段。重新执行前检查原队列、预约文件和状态；不自动清理失败SFT。

## 6. 离线分析

两份新增分析脚本均只读取已有产物：

```bash
python analysis/analyze_ego_pretraining.py --help
python analysis/analyze_retention_opd.py --help
```

按参数指定`--experiment`、`--lastlayer`、`--data`、`--out`。输出默认包含本地逐样本诊断，不能直接整目录推送公共GitHub。公开发布只选汇总统计，保留输入哈希和口径。
