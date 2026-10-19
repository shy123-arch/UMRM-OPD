# vt_head_opd_v860 — Qwen3-VL-4B (V→T) + 自有动作头（T→M）+ 特权信息 on-policy 自蒸馏（反向 KL）

路线文档：`../route_discussion_20261015_head_opd/ROUTE.md`。用户 2026-10-15 批准：下载 Qwen3-VL-4B；OPD 与自有动作头两条主张都试；第一版用 v860；启动前挑相对空闲的两张卡；动作统一转成 HumanML3D 口径（HumanML263 / 22 关节；评测时手部按前臂比例补全，沿用 route6 eval_hero 的做法）。

## 数据与边界
- 划分沿用 route6 冻结的 v860：train 582 / val 96 / **test 182 封存**（`common.manifest()` 在加载时就删掉了 test 行，任何脚本都读不到）。
- 只读引用：route6（清单、划分、计划编译器、Kimodo/TMR 评测器）、hero_ft_v860（v860 的 HumanML263、v860 微调 VQ）、HERO/、LaMP/dataset/HumanML3D。正式 636 的成员和 split 未改动。
- 本地缓存（可再生，`/tmp/vho_cache`）：视频张量、Qwen 文本隐状态、打包后的 HumanML3D。模型副本：`/tmp/vho_models`。

## 组件
| 步骤 | 脚本 | 环境 | 说明 |
|---|---|---|---|
| 视频缓存 | `prep_video_cache.py` | qwen35_532，CPU | 16 帧、max_pixels 360×640（与 route6 相同） |
| V→T SFT | `sft.py --mode student/teacher` | qwen35_532，1 GPU | LoRA r16（36 层×7），lr 1e-4，3 epoch，accum 8；按 VAL NLL 选 epoch。teacher 的 prompt 带特权提示 z |
| 推理 | `infer.py` | 同上 | 输出格式与 route6 相同 |
| 文本编码器 | `textenc.py` | 同上 | 冻结 4B 基座（无 LoRA），`<exec> caption`，第 9/18/27/36 层 |
| T→M 动作头 | `head.py` | hero，1 GPU | HERO RVQ + 掩码/残差 Transformer（ViMo 初始化）+ QwenAdapter；修正 CFG 空条件 |
| OPD | `opd.py` | qwen35_532，1 GPU | 学生采样 K=4（T=1，合并权重采样器，每步刷新），整词表逐 token 反向 KL；教师冻结 |
| 评测 | `eval_kimodo.py`、`eval_head.py`、`run_val_motion.sh` | kimodo | route6 独立评测器（原版 HumanML3D TMR + 几何），写入本目录 |

特权信息 z（第一版）：PICO 动捕测得的真实反应的时长、位移（前/右）、转向角，以及 locomotion/direction/posture 标签。不含反应文本。仅训练期教师可见。

## 已核验的实现细节
- prompt 分段分词与处理器整体处理逐 token 一致（`dev/probe_processor.py`）。
- transformers 5.3 中 Qwen3-VL 的 `get_rope_index` 每段视频 token 消耗一行 grid，而视频被时间戳切成 T 段。因此把 grid 展开为 T×[1,h,w]，视觉特征（含 deepstack）差值为 0（`dev/check_grid.py`）。
- 左填充批量前向与单样本前向的教师强制 NLL 一致（`dev/check_leftpad.py`）；采样器刷新后与 `merge_and_unload` 的 greedy 输出一致（`dev/check_multi.py`）。
- `eval_head.py` 复现 route6 的 eval_hero_ft_B，8 项指标逐位一致（`runs/eval_head__check_hero_ft_B.json`）。

## 事先登记的选择规则与门（2026-10-15 17:00 CST 写入，早于任何 VAL 生成结果）
- **选模**：SFT 取 VAL NLL 最低的 epoch；动作头预训练取 HumanML3D val（mask CE + res CE）最低；动作头微调取 v860 VAL（mask CE + res CE）最低；OPD 取最后一个 epoch（共 2 个）。
- **统计**：VAL 96 条；Kimodo 臂 3 个种子，自有头 3 个样本；按来源做 cluster bootstrap 95% CI + Wilcoxon（复用 route6 的做法）；未做多重比较校正。
- **主语义指标**：R@10、t2m sim、同类 R@3、中位排名。MPJPE 和动作-动作余弦只作辅助，因为它们偏爱平均动作。
- **G-A（4B 是否够用）**：4B SFT / Kimodo-TPH 相对 route6 7B SFT / TPH，在 R@10、t2m sim 上不显著更差（CI 上界 ≥ 0）。否则考虑换 8B。
- **G-T（教师是否真的用到 z）**：在 VAL 前 32 条上，学生样本对"带 z 教师"的 KL ≥ 1.5 × 对"不带 z 教师"的 KL；且教师 greedy（VAL z，仅诊断用）的计划准确率比学生高 ≥ 0.10。
- **G-B（动作头自检）**：HumanML3D test 子集上，生成动作的 TMR R-precision@3 ≥ 0.6 × 同一 VQ 重建的 GT；v860 VAL 上 gt3p oracle 文本经自有头的 R@10 显著高于随机（0.104）。
- **G-C（自有头 vs Kimodo）**：同一 gt3p oracle 文本下，自有头相对 Kimodo-T 臂的 R@10、t2m sim 不显著更差，才作为主执行器；否则降为消融。
- **G4-OPD（OPD 是否有效）**：OPD-final 相对 SFT，在 Kimodo-TPH 主臂上 R@10 或 t2m sim 的 CI 下界 > 0；duration/direction 计划分布的 TV 距离恶化不超过 0.1。自有头臂作为第二读数。同时与 route6 DPO 并列报告。CI 跨 0 但方向一致，记为"未证实"。

## 事件级（VPP2 启发，2610.10270）的后续消融（未实施）
诊断见 `runs/diag_event_level.json`：
- v860 train 有 19.9% 的反应在 88 帧（4.4 s）处被截断，原始时长中位数 5.44 s；
- PICO 时长跨度窄（p95/p5 = 2.4）；
- 事件归一化不能降低当前误差，误差主要来自决策。
计划：E1 用未截断的完整事件目标微调动作头，duration 档位扩展到 > 4.4 s；E2 若做 VPP2 式固定 N 归一化，需要配 Stage-2 式的"回到实时尺度"后训练。
