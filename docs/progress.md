# 开发记录

每完成一个任务记一笔：做了什么、实测数字、已知问题。最新的写在最下面。

## 2026-10-07 · 设计与准备

- 与老师确认设计（`docs/superpowers/specs/2026-10-07-teaching-template-design.md`）和实施计划（`docs/superpowers/plans/2026-10-07-teaching-template.md`）。
- 开发环境：Linux 云端容器，4 核 CPU，Python 3.11 虚拟环境；gradio 6.29.1、sherpa-onnx 1.13.8。
- build_spec 6.3 的最小代码在本环境复测：`0-four-speakers-zh.wav`（57 秒）端点检测 7 段 0.43 秒，识别 6.05 秒，说话人分离（设 4 人）7.56 秒、分出 4 人；`zh.wav` 显示模式输出"……早上9点至下午5点。"，测评模式输出"……早上九点至下午五点"。

## Task 1 · 仓库整理、依赖、配置

- 原交接包移到 `handoff/`；从开工包复制 `data/`、`docs/build_spec.md`、`docs/handoff_slim.md`、`docs/script_spec.md`；新增 `.gitignore`、`requirements*.txt`、`config.yaml`、`pyproject.toml`，更新 `CLAUDE.md`。
