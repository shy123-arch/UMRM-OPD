# ego数据反应标注台

独立的 ego 视频反应切片工具。保留源数据已有 reaction/reason，预填中文译文，让标注者主要调整反应前窗口。与现有5131标注台、5130审核台、正式636和ER19训练分开。

## 首批数据

2026-10-19导入20条候选：EgoNormia12条、GoalStep8条，20个不同视频UID；排除现有探针795个视频UID。视频来源为已有OSS挂载，只读引用。每条生成最多30秒的H.264播放窗口，供精细定位；不复制完整源视频。

候选来自PT-C训练manifest，因此标记`pretraining_seen=true`，不作为当前模型未见过的独立测试数据。仅做文本初筛，未将原标注时间戳自动视为合格动作起点，20条初始状态均为pending。EgoReWhy缺少直接可用的前瞻性反应描述，首批不导入。

原始反应和理由逐字保留。中文为Codex预填译文，并保留翻译来源；没有调用额外LLM服务。数据、源标注、候选清单、数据库和页面截图不提交公共Git仓库。

## 操作

1. 填写标注者缩写，选择一个候选。中文反应和英文原文已预填。
2. 播放短定位窗口，通过慢放、±0.1秒、逐帧按钮定位。
3. 用`[`/`]`或按钮设置起点、截断点；输入框允许毫秒精度，实际帧精度受源视频限制。
4. 用“查看截断前后原始帧”核对：从源视频直接解码，并显示实际源时间戳，不只看播放代理。右图只作边界诊断。
5. 确认动作准备尚未开始、描述与前情相符，点击“确认并生成切片”。切片范围1–15秒；源reaction不合适可标“暂不采用”。
6. 可存草稿、修改后重切、播放生成结果；每次修改增加revision，保留操作者和历史。并发冲突拒绝覆盖。本浏览器暂存未保存输入。
7. “导出已切片数据”只导出ready行。导出仍为`ego_candidate_pool`，待独立审核/动捕，不自动合入正式benchmark或现有审核台。

键盘：空格播放/暂停，J/L ±0.1秒，Shift+J/L ±1秒，左右方向键切换，S保存。输入框内不触发快捷键。

## 数据格式与兼容性

保留现有标注上传表的必要列名：

`视频名称 / 开始时间(秒) / 结束时间(秒) / reaction`

另有`reaction_ch / 时长(秒) / 标注者 / 备注 / 标注来源ID`，以及源数据集、视频UID、源绝对时间、State、原reaction/reason和分类。

导出ZIP包含：

```text
reaction_clips.xlsx     现有上传表列名；reaction为英语，reaction_ch为中文
videos.zip             仅包含最终反应前切片，不含cutoff之后的定位画面
translations_ego.json   工作簿哈希、逐行中文和翻译来源；适配既有上传读取器
manifest.json          ego_reaction_candidates/v1，完整来源/版本/时间/审核身份
```

**时间口径**：表里的“视频名称”对应`videos.zip`内最终切片，因此开始时间为0、结束时间为实际切片时长。原始源视频绝对起止时间保存在独立列和manifest中。不得拿源视频绝对时间去再次裁剪导出的短片。

此工具建立独立的候选格式，不修改正式636 schema。源reason不能自动充当已审核mental_reasoning，空State不虚构；需要State才能确定反应的样本应补写并进入独立审核。UI的确认表示切片操作者判断，不表示正式benchmark准入或已有动捕。

精确切片从原始OSS视频读取，视频无音频；代理/输出采用H.264，不改变原始文件。一个worker顺序执行FFmpeg，每次最多2线程；无GPU、无模型调用。

## 本地运行与部署

依赖：Python、Flask、openpyxl、FFmpeg/FFprobe；浏览器验收另需Playwright及Chromium。

```bash
python tools/ego_reaction_desk/server.py \
  --data-dir /mnt/workspace/Wilson/ck/umrm/ego_reaction_annotation \
  --host 127.0.0.1 --port 5132
```

命令默认仅监听本机。2026-10-19 经用户明确授权，已使用[supervisor配置](deploy/ego_reaction_desk.conf)部署到 `0.0.0.0:5132`，服务名为 `ego_reaction_desk`。服务由 supervisord 常驻管理，脱离当前终端，异常退出自动重启；只新增此服务，未重启现有服务。首次部署曾因缺少网络范围和持久部署授权被自动审批拒绝，用户补充授权后完成部署。

访问方式：沿用原标注台的服务器地址，将端口改为 `5132`。通过平台代理访问时，使用平台的 5132 端口入口。本机健康检查为 `http://127.0.0.1:5132/health`；外部网络可达性取决于平台端口映射。

该工具按现有平台内部访问模式设计，不含独立账户认证；标注者缩写用于记录归属，不能充当身份验证。对外访问应通过受控平台入口或已授权网络。

运行数据在仓库外：

```text
ego_reaction_annotation/
  ego_reaction.sqlite3   标注、版本和操作历史（含WAL时使用SQLite backup）
  pilot_selection.json   小批候选ID和预填译文
  import_lock.json       输入哈希，防止重跑混入不同清单
  batch_summary.json     导入计数和来源范围
  media/                短定位视频、来源元数据
  clips/                按候选ID与revision存放的精确切片
  exports/              不可变导出包
  logs/                 服务日志
```

导入先dry-run，再显式执行：

```bash
python tools/ego_reaction_desk/import_batch.py \
  --manifest /path/to/ego_manifest.json --selection /path/to/selection.json
# 核对计划后，使用同一命令加 --execute。
```

每批最多30条；输入必须来自manifest训练部分，并与全部probe视频隔离。selection格式是对象列表，字段`id, reaction_cn, note(可选)`。重跑只核对并复用同一批，不重置已保存标注。

## 验证

```bash
python tools/ego_reaction_desk/test_desk.py -v
node --check tools/ego_reaction_desk/web/app.js
python tools/ego_reaction_desk/browser_smoke.py
python tools/check_repository.py
```

功能测试用临时SQLite和合成视频：非法/非有限时间、必需确认、并发版本冲突、源标注不可变、旧worker不能覆盖新revision、HTTP Range、跨站写入限制、源帧时间、真实FFmpeg裁片、XLSX/ZIP时间与哈希。

浏览器测试仅启动127.0.0.1临时端口，用合成数据验证预填、视频加载、草稿恢复、原始帧、保存、切片、下载和窄屏布局；不向实际候选库写测试标注。

2026-10-19验收：6项功能测试通过；上述浏览器流程通过；首批真实候选只读验收通过（20条列表、EgoNormia/GoalStep播放、来源筛选、手机宽度无横向溢出、无JavaScript错误），20条标注均保持pending/version0。截图与结果保留在运行目录的`validation/`，不发布原视频画面。播放副本合计约167 MiB，完整源视频仍在OSS。

部署后再次从本机检查实际 5132 服务：supervisord 状态 RUNNING、健康接口20条、浏览器视频播放通过、媒体 Range 返回206，检查前后标注版本及状态一致；原有5130和5131页面均返回200。尚未从用户外部网络验证平台入口。
