# 开发记录

每完成一个任务记一笔：做了什么、实测数字、已知问题。最新的写在最下面。

## 2026-10-07 · 设计与准备

- 与老师确认设计（`docs/superpowers/specs/2026-10-07-teaching-template-design.md`）和实施计划（`docs/superpowers/plans/2026-10-07-teaching-template.md`）。
- 开发环境：Linux 云端容器，4 核 CPU，Python 3.11 虚拟环境；gradio 6.29.1、sherpa-onnx 1.13.8。
- build_spec 6.3 的最小代码在本环境复测：`0-four-speakers-zh.wav`（57 秒）端点检测 7 段 0.43 秒，识别 6.05 秒，说话人分离（设 4 人）7.56 秒、分出 4 人；`zh.wav` 显示模式输出"……早上9点至下午5点。"，测评模式输出"……早上九点至下午五点"。

## Task 1 · 仓库整理、依赖、配置

- 原交接包移到 `handoff/`；从开工包复制 `data/`、`docs/build_spec.md`、`docs/handoff_slim.md`、`docs/script_spec.md`；新增 `.gitignore`、`requirements*.txt`、`config.yaml`、`pyproject.toml`，更新 `CLAUDE.md`。

## Task 2–5 · 基础模块（并行开发，每个任务独立审查通过）

- Task 2 `pipeline/schema.py`、`pipeline/data.py`：统一中间格式（中文表头、UTF-8 带 BOM 的 CSV、JSON 带通知语和"由人工智能技术自动生成"）；剧本、台词、类别、热词、录音计划读取。核对：24 个剧本、1055 句台词、7 个类别（5 个会标疑似）、121 个热词、8 个变体、39 个虚构名称、72 个计划录音。
- Task 3 `pipeline/config.py`、`pipeline/methods.py`：读 `config.yaml`（相对路径换成绝对路径）；做法登记（8 个槽位，`register`/`get_method`/`available`/`resolve`/`load_all`）；导入时不加载重依赖（测试在子进程里验证）。
- Task 4 `pipeline/audio.py`、`pipeline/step1_ingest.py`：ffmpeg 查找（环境变量 → PATH → imageio-ffmpeg）、格式统一、SHA-256、只读保存原始文件、五项质检。一段 14 秒 44.1kHz 立体声 m4a（文件名含中文和空格）入池用时 0.15 秒。
- Task 5 `models/download_models.py`、`pipeline/models.py`：必需 5 项 + 可选 5 项模型清单，已下载的跳过；10 个下载地址当天都能访问。
- 集成时的修正：pytest 配置加 `pythonpath`；ruff 也检查 `models/download_models.py`；便携包构建脚本一开始就设 UTF-8 输出；"威胁消费"显示为"消费施压"（`labels.json` 加 `display`，`data.FLAG_OUTPUTS`、`data.DISPLAY_NAMES`）；`audio.remove_tree` 能删除只读文件（Windows）；读不出采样率时给出正确提示；schema 的几处小问题。
- 测试：101 个全部通过（Linux）。
