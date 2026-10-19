# 运行环境（3 个）

| 环境 | 用途 | 关键版本 | 清单 |
|---|---|---|---|
| `vho-qwen` | 视频预处理、SFT、推理、文本/视觉特征、OPD | Python 3.10，torch 2.6.0+cu124，transformers **5.3.0**，peft 0.19.1，flash-attn 2.8.3，qwen-vl-utils 0.0.14，decord 0.6.0 | `vho-qwen.requirements.txt` |
| `vho-hero` | 动作头（HERO 的 RVQ / 掩码 Transformer）、PICO→HumanML263 转换 | Python 3.10，**torch 1.12.0+cu113**，numpy 1.23.5 | `vho-hero.requirements.txt` |
| `vho-kimodo` | 独立评测器（官方 HumanML3D TMR + 冻结的 Kimodo 生成） | Python 3.12；基于 NGC PyTorch 25.03 容器（torch 2.7.0a0 nv25.3），Kimodo `pip install -e` @54257dd，transformers 5.3.0 | `vho-kimodo.requirements.txt`（容器自带的 wheel 已滤掉） |

建议（conda）：

```bash
conda create -n vho-qwen python=3.10 -y && conda activate vho-qwen
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r envs/vho-qwen.requirements.txt          # flash-attn 如编译失败，用对应 cu124/torch2.6 的预编译 wheel
conda create -n vho-hero python=3.10 -y && conda activate vho-hero
pip install torch==1.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 && pip install -r envs/vho-hero.requirements.txt
# vho-kimodo：在 nvcr.io/nvidia/pytorch:25.03-py3 容器内 git clone nv-tlabs/kimodo && pip install -e kimodo，再装清单里其余包
```

注意事项：
- **transformers 5.3 与 Qwen3-VL**：`get_rope_index` 会为每一段连续的视频 token 消耗一行 `video_grid_thw`，而处理器用时间戳把视频切成了 T 段。`vt_io.per_frame_grid` 已把 grid 展开为 T×[1,h,w]，展开前后视觉特征完全一致（差值为 0）。升级 transformers 后请重新核验。
- **vho-hero 的 torch 动态库**：系统里如果另有 torch，要把 `VHO_HERO_TORCHLIB` 指向本环境的 `torch/lib`，`_env.sh` 里的 `hero()` 会把它放到 `LD_LIBRARY_PATH` 最前面。
- **CPFS/FUSE 等慢盘**：把 Qwen 权重和 `VHO_CACHE` 放在本地 SSD 上。原服务器上，环境导入从 30 分钟降到 1 分钟。
