# 来源与使用边界

UMRM路线代码整理自本地`umrm-vho`及四个实验目录。初始来源版本、文件对应关系、导入前SHA256在`docs/SOURCE_SNAPSHOT.json`。工作树中新加入的ego结果分析、ER19队列和SFT半程诊断一并纳入；历史Git对象没有上传。

`third_party/umrm_eval/`保存项目内对已有UMRM评测器、PICO转换、HERO/路线6的适配胶水代码，不包含完整HERO、Kimodo或Qwen模型仓库。外部工程和权重遵守其各自许可：

- HERO/ViMo：代码来源`JackYu6/HERO_release`；既有环境记录的HERO版本为`475774d`。
- Kimodo：来源`nv-tlabs/kimodo`；既有环境记录版本为`54257dd`。
- HumanML3D、Qwen、TMR、LLM2Vec、Ego4D及派生数据分别按官方条件获取。

本次整理未为整个项目擅自指定MIT、Apache等统一许可证，也不授予第三方数据或权重的再分发权限。公开可见性不替代许可；需要再分发或商业使用时应核实各部分许可及项目授权。
