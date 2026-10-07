# CLAUDE.md · 旅游纠纷录音材料整理（教学原型）

## 这是什么

高校《语音识别技术及应用》实训课的全班项目（31 名专科生，8 组：第 2 组 3 人，其余 7 组各 4 人；10 周）。本仓库是**教师模板**：一个本地运行、用浏览器访问的工具，把游客提交的投诉录音或视频整理成供投诉处理人员人工复核的"核查初稿"（带时间戳、区分说话人的文字稿，标为"疑似·费用"等的片段，提取的金额、电话、证号，人工复核栏）。工具只提示、不判定。

八个处理步骤各是一个小模块；每组在自己的文件 `pipeline/groups/gN_*.py` 里为某个"槽位"写"改进做法"，基线做法永远保留，用 `config.yaml` 或界面切换（见 `docs/adr/0001-groups-plug-in-as-methods.md`）。代码要简单、好读、每步可单独运行和测评。

## 先读这些

1. `docs/superpowers/specs/2026-10-07-teaching-template-design.md` —— 设计文档：课程组织、数据工作、Windows 运行、协作方式。与其他文档冲突时以它为准。
2. `docs/build_spec.md` —— 代码层面的开发规格：八步、统一中间格式、界面、Word 初稿、测评、已实测的模型和代码（第 6 节）。
3. `docs/superpowers/plans/2026-10-07-teaching-template.md` —— 实施计划（任务、接口、测试）。
4. `GLOSSARY.md` —— 术语表（核查初稿、疑似片段、槽位、做法、数据池版本……），写代码和文档都用这些词。
5. `docs/adr/` —— 技术决定记录；`docs/handoff_slim.md`（项目背景、红线）；`data/README.md`（数据说明）。

## 红线（优先于其他一切要求）

1. **只用虚构材料**。仓库里只能有 `data/` 的虚构剧本和 AI 生成的虚构句子。开发和云端测试只用模型自带的公开测试音频（可以拼接成长音频）和 ffmpeg 生成的纯音、静音；这些测试音频只用于自动测试和便携包自检，不放进演示界面。学生按剧本录的录音只在本地（机房、老师电脑）处理，不上传到云端开发环境或公网演示，也不进仓库。不下载、不处理、不提交任何真实投诉录音、网上的导游视频或真实个人信息。学生真实姓名、学号不进仓库。
2. **不做声音合成和声音克隆**，也不要用语音合成把剧本"念"成测试录音。
3. **数据不出本机**。识别、分离、分类全部本地运行；处理录音时不调用任何云 API 或公共大模型；Gradio 永远不设 `share=True`；设 `GRADIO_ANALYTICS_ENABLED=False`。
4. **只提示、不判定**。被标出的片段一律写成"疑似·类别"（只有 `data/labels.json` 里 `flag=true` 的 5 类会被标出，正常讲解和其他不标）；界面、初稿和导出文件都醒目写明"识别可能有误，所有标注均为疑似、待核查，必须人工复核"；不出现"违规""违法""执法级准确率""可作为法律证据"等说法；不鉴定录音真伪。
5. **只做商业纠纷**。类别只有 `data/labels.json` 里的 7 个；不做政治、民族、宗教、地域、辱骂类言论检测。对外名称用"旅游纠纷录音材料整理"，界面不用"举报""曝光""证据"等字眼，片段叫"疑似片段"。
6. **标明是机器生成**。Word 初稿、JSON 和导出的 ZIP 都写明由人工智能技术自动生成，并写进 Word 文档属性。
7. **如实报告**。剧本数据上的测评结果不代表真实场景；不对准确率和速度做未实测的承诺；在云端开发环境里不报告任何字错率数字。

## 关键技术决定

- 识别：sherpa-onnx + SenseVoice int8 **2024-07-17 版**（不要用 2025-09-09 版，那是粤语模型）。`use_itn=True` 给界面显示（写 `text`），`use_itn=False` 给算字错率（写 `text_raw`，与参考文本口径一致）。
- 端点检测：Silero VAD；说话人分离：pyannote 分割 3.0 + CAM++ 声纹 + 聚类，知道人数时一定要设人数。
- 热词：sherpa-onnx 里 SenseVoice 不支持解码时热词，基线用识别后的拼音相似度纠错（同时作用于 `text` 和 `text_raw`），真正的热词方案留给第 5 组。
- 话术分类：关键词规则基线；TensorFlow 是可选依赖，有训练好的模型且装了 TensorFlow 才用，否则退回规则。分类前的文字归一只用标准库（`pipeline/text_norm.py`），不调用步骤 5。
- 界面只用 Gradio 6（主题、CSS 传给 `launch()`）；Word 初稿用 python-docx。不需要 PyTorch。
- **Windows 是主要运行环境**：新文件和目录名只用 ASCII；临时文件放项目内 `tmp/`；ffmpeg 依次找 `FFMPEG_BINARY`、PATH、`imageio-ffmpeg`；主线是 GitHub Actions 构建的免安装便携包（`packaging/windows/`）。
- **轻量导入**：`import pipeline`、`pipeline.text_norm`、`pipeline.methods`、`pipeline.groups` 不得导入 sherpa_onnx、gradio、tensorflow（重依赖在函数内导入），这样第 6、7 组能用机房自带的 TensorFlow 跑训练脚本。

## 怎么干活

- 按实施计划的任务顺序做，先写失败的测试再实现。每完成一个任务：跑 `pytest -q` 和 `ruff check .`、提交、在 `docs/progress.md` 记一笔（做了什么、实测数字、已知问题）。
- 遇到设计文档第 11 节和 build_spec 第 13 节列的事项（例如是否公网部署、用谁的账号）停下来问老师，不要自行决定；不要自己注册或登录任何托管平台账号。
- `models/`、录音、`data_pool/`、`outputs/`、`tmp/` 都不进 git，提交前检查 `git status`。
- 每个模块开头写清"基线做法 / 可改进方向 / 测评指标"，注释用中文。基线要简单可用，不要替学生把专项改进做完。
- 版本号、Gradio 参数名以安装的版本和官方文档为准，不确定就先查（`help()`），不要凭记忆写。

## 常用命令

```bash
# 开发环境（Linux/Mac）
python -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
python models/download_models.py           # 下载必需模型到 models/
pytest -q                                  # 冒烟测试在模型已下载时运行
python app.py                              # http://0.0.0.0:7860
python tools/evaluate.py --help            # 测评
```

```bat
:: Windows 便携包（解压到纯英文路径，如 D:\asr\tiandu）
start.bat                                  :: 启动网页工具
python\python.exe tools\selfcheck.py       :: 自检
python\python.exe tools\evaluate.py --help :: 测评
```
