# 教学模板实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在本仓库做出可运行的"旅游纠纷录音材料整理"教学模板（八步流程、Gradio 界面、数据池与测评工具、Windows 便携包、CI）和全部中文课程文档（README、8 份组任务书、指南、老师文档）。

**Architecture:** `pipeline/` 是一个深模块：对外只有 `run_pipeline()`、各步公开函数和"做法登记"（槽位 → 做法）。基线做法在 `pipeline/stepN_*.py`，各组改进做法在 `pipeline/groups/gN_*.py`，用 `config.yaml` 或界面切换。数据池、测评、对比都是 `pipeline/` 里的纯函数模块，`tools/*.py` 和 `app.py` 只是薄薄的命令行和界面外壳。

**Tech Stack:** Python 3.11（Windows 目标）；sherpa-onnx 1.13.8（Silero VAD、SenseVoice int8 2024-07-17、pyannote 分割 3.0、CAM++）；Gradio 6.29；numpy、soundfile、PyYAML、python-docx、pypinyin、cn2an、rapidfuzz、noisereduce、matplotlib、imageio-ffmpeg；可选 TensorFlow ≥ 2.10；pytest、ruff；GitHub Actions（ubuntu-latest、windows-latest）。

**Spec:** `docs/superpowers/specs/2026-10-07-teaching-template-design.md`（与 `docs/build_spec.md` 一起读；冲突时以 spec 为准）。

## Global Constraints

- 红线全部照 `CLAUDE.md`：只用虚构材料；不做语音合成；处理录音不联网、不调云 API；永远不设 `share=True`；`GRADIO_ANALYTICS_ENABLED=False`；标签只能是 `data/labels.json` 中 flag=true 的 5 个"疑似·类别"或空字符串；界面、初稿、导出文件不出现"违规""违法""执法级准确率""可作为法律证据"；Word/JSON/ZIP 写明由人工智能技术自动生成。
- 通知语固定为：`识别可能有误；所有标注均为疑似、待核查，必须人工复核`（`pipeline.schema.NOTICE`）。
- 界面顶部横幅固定为：`仅限虚构演示材料，请勿上传真实投诉录音或含个人信息的录音｜识别可能有误，所有标注均为"疑似、待核查"，必须人工复核｜不作为任何定性依据`。
- 新文件和目录名只用 ASCII；文档、界面、注释用中文；代码标识符用英文。
- 运行时依赖不能含 PyTorch；TensorFlow 只在 `requirements-tf.txt`，任何运行时模块在模块顶层都不得 `import tensorflow`。
- `import pipeline`、`import pipeline.text_norm`、`import pipeline.methods`、`import pipeline.groups` 不得导入 sherpa_onnx、gradio、tensorflow（重依赖只能在函数内部导入）。
- `data/` 下的 CSV 用 `encoding="utf-8-sig"` 读写。导出 CSV 用 UTF-8 带 BOM。
- 采样率固定 16000，单声道，16 位。
- 识别模型目录：`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17`（不得用 2025-09-09 版）。
- 测试音频（模型自带）只用于自动测试和便携包自检，不放进演示界面。
- 每个模块文件开头用中文写"基线做法 / 可改进方向 / 测评指标"。
- 每个任务结束：`.venv/bin/pytest -q` 全绿（冒烟测试在模型存在时也要过）、`.venv/bin/ruff check .` 无错误，再提交。提交信息用中文，结尾带两行署名（见会话说明）。
- 开发环境：`.venv`（Python 3.11）已装好依赖；模型已下载到 `models/`。`models/`、`data_pool/`、`outputs/`、`tmp/` 和所有音视频文件不得提交。

## Review Focus

1. **Windows 中文路径**：便携包或上传文件路径里有中文（如 `C:\Users\张三\...`）时，模型能加载、临时文件不落到中文路径 → Task 4 测 `has_non_ascii`；Task 13 让 Gradio 临时目录用项目内 `tmp/`；Task 14 自检提示。
2. **很短或无人声的录音**（1 秒、全静音、纯音乐）：流程不应崩溃，应返回 0 段并在摘要里说明 → Task 12 用 ffmpeg 生成的 2 秒静音跑 `run_pipeline`，断言返回空列表且 meta 有提示。
3. **学生交来的文件名不规范**（`g1-s1-q.WAV`、`G1-S1-Q (1).m4a`、`G9-S1-Q.wav`）：明确报错、提示怎么改，不猜 → Task 15 测试。
4. **用户在表格里改了说话人、文字、复核结论后再导出**：导出必须用修改后的内容 → Task 11/13 测 `rows_to_segments` 往返。
5. **没装 TensorFlow 或模型没训练时选了 g6/g7 做法**：退回关键词规则并给出中文说明，不崩溃 → Task 10 测试。

---

## 阶段 A：骨架与数据

### Task 1: 仓库整理、依赖、配置

**Files:**
- Move: 根目录 `01_对话记录/ 02_课程设计/ 03_测评剧本/ 04_项目介绍视频/ HANDOFF.md 文件总清单.txt README.md` → `handoff/`（`git mv`；原 README 改名为 `handoff/README.md`）
- Create（从 zip 开工包复制，内容不改）：`data/`（全部）、`docs/build_spec.md`、`docs/handoff_slim.md`、`docs/script_spec.md`；zip 路径：`$SCRATCH/handoff_zip/tiandu-voice-to-text/`
- Create: `.gitignore`（zip 里的内容 + `reports/*/outputs/`、`*.h5` 不忽略）、`requirements.txt`、`requirements-tf.txt`、`requirements-dev.txt`、`config.yaml`、`pyproject.toml`（只放 ruff 和 pytest 配置）、`CLAUDE.md`（zip 版 + 本计划的更新）、`docs/progress.md`
- Test: `tests/test_repo_layout.py`

**Interfaces:**
- Produces: `config.yaml` 键（后续任务都按这些名字读）：
  ```yaml
  paths: {models: models, data_pool: data_pool, outputs: outputs, tmp: tmp}
  methods: {denoise: baseline, enhance: baseline, vad: baseline, hotword: baseline,
            diarize: baseline, normalize: baseline, classify: baseline, clips: baseline}
  vad: {threshold: 0.5, min_silence_duration: 0.3, min_speech_duration: 0.25, max_speech_duration: 15}
  asr: {num_threads: 4, batch_size: 20, language: zh}
  hotword: {enabled: false, min_len: 3, max_syllable_mismatch: 0}
  diarize: {num_speakers: -1, threshold: 0.5, min_duration_on: 0.3, min_duration_off: 0.5}
  qc: {min_rms_dbfs: -40.0, clip_ratio: 0.001, long_silence_seconds: 10.0, silence_dbfs: -50.0,
       duration_tolerance: 0.4, min_sample_rate: 16000}
  clips: {padding: 1.0}
  script_demo: {chars_per_minute: 220}
  app: {host: 0.0.0.0, port: 7860, max_file_size: 500mb, cleanup_hours: 24}
  ```
- requirements.txt（兼容范围，不锁死）：`gradio>=5,<7`、`sherpa-onnx>=1.12,<2`、`numpy`、`soundfile`、`PyYAML`、`python-docx>=1.1`、`pypinyin`、`cn2an`、`rapidfuzz`、`noisereduce`、`matplotlib`、`imageio-ffmpeg`；requirements-tf.txt：`tensorflow-cpu>=2.10`；requirements-dev.txt：`pytest`、`ruff`。

- [ ] **Step 1: 写失败的测试** `tests/test_repo_layout.py`
  - `test_handoff_moved`：`handoff/HANDOFF.md`、`handoff/04_项目介绍视频/` 存在；根目录不再有 `HANDOFF.md`。
  - `test_data_present`：`data/scripts/group_1.json`…`group_8.json`、`data/lines.csv`、`data/labels.json`、`data/hotwords.txt` 存在。
  - `test_gitignore_blocks_media`：`.gitignore` 含 `models/*`、`!models/download_models.py`、`data_pool/`、`*.wav`、`.venv/`。
  - `test_config_keys`：`yaml.safe_load(open("config.yaml"))` 含上面全部一级键，`methods` 有 8 个槽位且都是 `baseline`。
- [ ] **Step 2: 运行，确认失败** `.venv/bin/pytest tests/test_repo_layout.py -q` → FAIL
- [ ] **Step 3: 做上面的移动、复制、新建**。`CLAUDE.md` 在 zip 版基础上改：人数 34→31、8 组（第 2 组 3 人）、"做法登记 + 每组一个文件"、ASCII 路径、便携包、前 6 周不用 GitHub；常用命令换成 Windows 和 Linux 两种写法。`docs/progress.md` 写第一条记录。
- [ ] **Step 4: 运行测试通过**；`git status` 确认没有 models/ 和 .venv/ 被跟踪。
- [ ] **Step 5: 提交** `git add -A && git commit`（提交前 `git status --short | grep -E "models/|\.venv|\.wav"` 必须为空）。

### Task 2: 统一中间格式与数据读取

**Files:**
- Create: `pipeline/__init__.py`（此任务只放 `__version__ = "0.1.0"`，`run_pipeline` 在 Task 12）、`pipeline/schema.py`、`pipeline/data.py`
- Test: `tests/test_schema.py`、`tests/test_data.py`

**Interfaces:**
- Produces (`pipeline.schema`):
  - `FIELDS = ["start", "end", "speaker", "text", "label"]`
  - `HEADERS: dict[str, str]`：start→`开始时间（秒）`、end→`结束时间（秒）`、speaker→`说话人`、text→`文字内容`、label→`标签`、text_raw→`识别原文（汉字读法）`、speaker_id→`说话人编号`、category→`类别`、score→`置信度`、numbers→`数字`、entities→`名称`、corrections→`热词纠错`、review→`复核结论`、review_note→`复核意见`、clip→`片段文件`、source→`来源`
  - `OPTIONAL_FIELDS`：上面除前 5 个以外的键，按上面顺序
  - `NOTICE`（见 Global Constraints）、`REVIEW_CHOICES = ["未复核", "确认", "修改", "驳回"]`
  - `new_segment(start: float, end: float, **extra) -> dict`：默认 `speaker="未知"`、`text=""`、`label=""`、`review="未复核"`，时间保留 2 位小数
  - `write_json(path, segments: list[dict], meta: dict) -> None`、`read_json(path) -> tuple[list[dict], dict]`（`meta["notice"]` 缺失时写入 NOTICE）
  - `write_csv(path, segments: list[dict]) -> None`、`read_csv(path) -> list[dict]`：中文表头，前 5 列固定顺序，可选列只写出现过的；`numbers`/`entities` 用"；"连接；`corrections` 写成 `错→对` 用"；"连接，读回 `[{"from","to"}]`；数字列读回 float
- Produces (`pipeline.data`)，`DATA_DIR = <repo>/data`：
  - `load_scripts() -> list[dict]`（24 个，每个带 `group:int`）、`get_script(script_id: str) -> dict`（不存在抛 `KeyError`）
  - `load_lines() -> list[dict]`：`numbers`、`hotwords` 拆成去重保序的列表，`group`、`line_no`、`effective_chars` 为 int
  - `script_reference_text(script_id: str) -> str`：该剧本全部台词 `text` 按顺序用 `\n` 连接（不含 `direction`）
  - `LABEL_NAMES: list[str]`（7 个，`labels.json` 顺序）、`FLAG_LABELS: list[str]`（5 个）、`label_output(category: str) -> str`（flag 类返回 `疑似·类别`，否则 `""`；未知类别抛 `ValueError`）
  - `load_hotwords() -> list[str]`、`load_hotword_variants() -> list[dict]`、`load_fictional_names() -> list[dict]`、`load_recording_plan() -> list[dict]`（72 行，`expected_minutes` 为 float）

- [ ] **Step 1: 写失败的测试**
  - `test_json_roundtrip`：两个段（含 numbers、corrections、review）写入再读回完全相等，meta 含 NOTICE。
  - `test_csv_headers_and_order`：写出的 CSV 首行前 5 列为 `开始时间（秒）,结束时间（秒）,说话人,文字内容,标签`；文件以 BOM 开头；读回与写入相等。
  - `test_new_segment_defaults`：`new_segment(1.234, 2.0)` → start 1.23、speaker 未知、review 未复核、label ""。
  - `test_load_scripts`：24 个，id 从 G1-S1 到 G8-S3；`len(load_lines()) == 1055`。
  - `test_label_output`：`label_output("费用") == "疑似·费用"`；`label_output("正常讲解") == ""`；`label_output("违规")` 抛 ValueError。
  - `test_reference_text_excludes_direction`：G1-S1 参考文本不含 `（用车载扩音器）`，行数 53。
  - `test_recording_plan`：72 行，文件名都匹配 `^G[1-8]-S[1-3]-[QNF]\.wav$`。
- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现 `pipeline/schema.py`、`pipeline/data.py`**（只用标准库 + PyYAML 不需要；`data.py` 用 `functools.lru_cache` 缓存读取）
- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交**

### Task 3: 配置读取与做法登记

**Files:**
- Create: `pipeline/config.py`、`pipeline/methods.py`
- Test: `tests/test_methods.py`

**Interfaces:**
- Produces (`pipeline.config`)：`ROOT: Path`（仓库根）、`load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict`（读 `config.yaml`，深度合并 overrides；`paths` 下相对路径转成相对 ROOT 的绝对路径字符串）
- Produces (`pipeline.methods`)：
  - `SLOTS = ("denoise", "enhance", "vad", "hotword", "diarize", "normalize", "classify", "clips")`
  - `SLOT_TITLES: dict[str, str]`：denoise→`降噪`、enhance→`远距离增强`、vad→`端点检测`、hotword→`热词纠错`、diarize→`说话人分离`、normalize→`数字规范化`、classify→`话术分类`、clips→`疑似片段`
  - `register(slot: str, name: str) -> Callable[[F], F]`：装饰器；未知槽位、重复名字抛 `ValueError`（中文信息）
  - `get_method(slot: str, name: str) -> Callable`：找不到抛 `KeyError`，信息里列出可用做法
  - `available(slot: str) -> list[str]`：`baseline` 在前，其余按名字排序
  - `load_all() -> None`：导入 `pipeline.step2_vad`…`pipeline.step7_clips` 和 `pipeline.groups`，使全部做法登记；可重复调用
  - `resolve(cfg: dict) -> dict[str, Callable]`：按 `cfg["methods"]` 取出每个槽位的函数（先 `load_all()`）

- [ ] **Step 1: 写失败的测试**
  - `test_register_and_get`：在测试里用一个临时槽位名会失败 → `register("unknown", "x")` 抛 ValueError；对 `denoise` 注册 `"_test"` 后 `get_method` 拿到同一函数，`available("denoise")[0] == "baseline"`（本任务里先手动注册一个 baseline 占位，测试结束后清理）。
  - `test_duplicate_name_rejected`。
  - `test_get_unknown_lists_available`：`KeyError` 信息含 `baseline`。
  - `test_load_config_overrides`：`load_config(overrides={"vad": {"threshold": 0.3}})["vad"]` threshold 为 0.3、其余键保留；`paths.models` 是绝对路径。
  - `test_light_imports`：子进程运行 `import pipeline, pipeline.methods, pipeline.config; import sys; assert "sherpa_onnx" not in sys.modules and "gradio" not in sys.modules and "tensorflow" not in sys.modules`。
- [ ] **Step 2–5:** 失败 → 实现 → 通过 → 提交（`load_all` 此时导入尚不存在的模块要用 `importlib` 并对 `ModuleNotFoundError` 只忽略 `pipeline.stepN`/`pipeline.groups` 自身缺失，等 Task 12 完成后改成严格导入并删除这段容错）

### Task 4: 音频工具与步骤 1（上传与格式统一）

**Files:**
- Create: `pipeline/audio.py`、`pipeline/step1_ingest.py`
- Test: `tests/test_step1_ingest.py`、`tests/conftest.py`（fixture `make_audio(tmp_path, kind, ext, seconds, sr, channels)` 用 ffmpeg 生成 `sine=frequency=440`、`anullsrc` 或"1 秒正弦 + 12 秒静音 + 1 秒正弦"）

**Interfaces:**
- Produces (`pipeline.audio`)：`SR = 16000`；`class FfmpegNotFound(RuntimeError)`；`find_ffmpeg() -> str`（顺序：环境变量 `FFMPEG_BINARY` → `shutil.which("ffmpeg")` → `imageio_ffmpeg.get_ffmpeg_exe()`）；`probe(path) -> dict`（`duration`、`sample_rate`、`channels`，解析 `ffmpeg -i` 的 stderr）；`convert_to_wav(src, dst) -> None`（`-y -i src -vn -ac 1 -ar 16000 -sample_fmt s16`，失败抛 `RuntimeError` 带 stderr 末 20 行）；`read_wav(path) -> np.ndarray`（float32 单声道，采样率不是 16000 抛 ValueError）；`write_wav(path, samples, sr=SR)`；`sha256_file(path) -> str`；`has_non_ascii(path) -> bool`
- Produces (`pipeline.step1_ingest`)：`SUPPORTED_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".mp4", ".mov", ".mkv"}`；`quality_check(samples, original_sr: int, cfg: dict, expected_seconds: float | None = None) -> list[str]`（返回中文问题列表，空列表=合格；五项：时长偏差超过 `duration_tolerance`、整体音量低于 `min_rms_dbfs`、削波比例超过 `clip_ratio`、最长静音超过 `long_silence_seconds`（按 0.1 秒帧、低于 `silence_dbfs` 算静音）、原始采样率低于 `min_sample_rate`）；`ingest(src_path, work_dir, cfg, expected_seconds=None) -> dict`（返回 `file`、`original`、`wav`、`sha256`、`duration`、`original_sample_rate`、`qc`；原始文件复制到 `work_dir/original/` 并设只读；不支持的扩展名抛 ValueError，信息提到"微信语音（.amr/.silk）请先导出为 m4a 或 mp3"）

- [ ] **Step 1: 写失败的测试**
  - `test_convert_formats`（参数化 mp3/m4a/mp4/wav，44.1kHz 立体声 3 秒）：输出 `soundfile.info` 为 16000、1 声道、`PCM_16`；原始文件 SHA-256 不变、为只读。
  - `test_sha256_matches_hashlib`。
  - `test_qc_reports_long_silence`：1+12+1 秒文件 → `qc` 有一项含 `静音`。
  - `test_qc_low_sample_rate`：8000Hz 源 → 含 `采样率`。
  - `test_qc_ok_for_normal_tone`：-20dBFS 正弦 5 秒、expected 5 秒 → `[]`。
  - `test_unsupported_ext`：`.amr` 抛 ValueError，信息含 `微信`。
  - `test_has_non_ascii`：`"D:/asr/x"` False，`"C:/Users/张三/x"` True。
  - `test_find_ffmpeg_env_override`：`monkeypatch.setenv("FFMPEG_BINARY", <真实路径>)` 返回该路径。
- [ ] **Step 2–5:** 失败 → 实现 → 通过 → 提交

### Task 5: 模型下载与模型路径

**Files:**
- Create: `models/download_models.py`、`pipeline/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces (`pipeline.models`)：`MODEL_SPECS: list[dict]`，每项 `name`、`url`、`archive`（下载文件名）、`check`（解压后必须存在的相对路径列表）、`size_mb`、`required: bool`；必需 4 项（sense_voice、silero_vad、pyannote、campplus）+ `test_wav`（必需，给自检和冒烟测试）；可选：`eres2net`、`itn_fst`、`paraformer_small`、`punct`、`conformer_hotword`（URL 全部照 build_spec 6.2）。`model_path(cfg, key) -> Path`：key ∈ `sense_voice_model`、`sense_voice_tokens`、`silero_vad`、`pyannote`、`campplus`、`test_wav`、`sense_voice_test_zh`；`missing_models(cfg) -> list[str]`（缺的 name）
- `models/download_models.py`：`python models/download_models.py [--optional NAME ...] [--list] [--models-dir DIR]`；已存在且 check 通过就跳过；下载用 `urllib.request`，重试 3 次，打印进度和中文失败提示（"国内访问 GitHub 慢，可以先在别处下载好再拷进 models/"）；`.tar.bz2` 解压后删除压缩包；结束时打印每个模型 OK/缺失；有必需模型缺失时退出码 1。脚本开头把仓库根加入 `sys.path`。

- [ ] **Step 1: 写失败的测试**
  - `test_specs_complete`：必需项 5 个；所有 URL 以 `https://github.com/k2-fsa/sherpa-onnx/releases/download/` 开头；sense_voice 的 URL 含 `2024-07-17`，不含 `2025-09-09`。
  - `test_missing_models_empty_dir`：`missing_models` 对空目录返回 5 个名字。
  - `test_download_skips_existing`：在临时目录放好 check 要求的空文件，运行 `main(["--models-dir", tmp])`（mock 掉下载函数）不调用下载、返回 0。
  - `test_models_present`（有模型时运行）：仓库 `models/` 下 `missing_models` 为空。
- [ ] **Step 2–5:** 失败 → 实现 → 通过（在本环境实际运行一次 `python models/download_models.py`，应全部跳过）→ 提交

## 阶段 B：基线流程

### Task 6: 步骤 2（降噪、远距离增强、端点检测）

**Files:**
- Create: `pipeline/step2_vad.py`
- Test: `tests/test_step2_vad.py`

**Interfaces:**
- Consumes: `pipeline.methods.register`、`pipeline.models.model_path`、`pipeline.schema.new_segment`
- Produces：登记 `denoise/baseline`（原样返回）、`denoise/noisereduce`（`noisereduce.reduce_noise(y=samples, sr=sr)`，float32）、`enhance/baseline`（原样返回）、`vad/baseline`（Silero VAD，按 `cfg["vad"]` 参数，检测器按配置缓存，返回 `[(start_s, end_s)]` 保留 2 位小数）；`detect_speech(samples, sr, cfg, methods: dict | None = None) -> tuple[np.ndarray, list[dict]]`（依次 denoise → enhance → vad，返回处理后的采样和 `new_segment` 列表）

- [ ] **Step 1: 写失败的测试**
  - `test_baseline_denoise_identity`、`test_enhance_identity`（输入输出 `np.array_equal`）。
  - `test_vad_silence_returns_empty`：3 秒全零 → `[]`。
  - `test_vad_on_four_speakers`（有模型时）：`0-four-speakers-zh.wav` 段数 ≥ 5，每段 end > start，全在 0–57 秒内，按时间递增。
  - `test_detect_speech_uses_methods`：传入 `methods={"denoise": 自定义函数(把采样乘 0)}` → 返回空段列表。
- [ ] **Step 2–5:** 失败 → 实现 → 通过 → 提交

### Task 7: 步骤 3（识别、热词纠错基线）

**Files:**
- Create: `pipeline/step3_asr.py`、`pipeline/hotwords.py`
- Test: `tests/test_step3_asr.py`、`tests/test_hotwords.py`

**Interfaces:**
- Produces (`pipeline.step3_asr`)：`recognize(samples, sr, segments, cfg, mode: str = "display", progress=None) -> list[dict]`（mode ∈ `display`/`eval`/`both`；display 写 `text`（`use_itn=True`），eval 写 `text_raw`（`use_itn=False`），both 两个都写；每种模式的识别器按 `use_itn` 缓存；按 `cfg["asr"]["batch_size"]` 分批 `decode_streams`；`progress(fraction, desc)` 每批调用；空段列表直接返回）
- Produces (`pipeline.hotwords`)：`fuzzy_pinyin(text: str) -> list[str]`（`pypinyin.lazy_pinyin` 无声调，再做模糊归并：声母 zh→z、ch→c、sh→s、开头 l→n；韵母 ing→in、eng→en、ang→an）；`correct_text(text: str, hotwords: list[str], min_len: int = 3, max_mismatch: int = 0) -> tuple[str, list[dict]]`（只在连续汉字串里滑窗；窗口长度 = 热词长度；窗口与热词字面不同但模糊拼音逐音节不一致数 ≤ max_mismatch 时替换；长热词优先；返回替换记录 `{"from", "to"}`）；登记 `hotword/baseline`：`cfg["hotword"]["enabled"]` 为假时原样返回；否则对每段的 `text` 和 `text_raw` 都纠错，记录写入 `corrections`（含 `start`）

- [ ] **Step 1: 写失败的测试**
  - `test_correct_text_fixes_homophone`：`correct_text("我们报的是雾影行走旅行社", ["雾隐行舟旅行社"])` → 文字含 `雾隐行舟旅行社`，记录 1 条 `from="雾影行走旅行社"`。
  - `test_correct_text_keeps_variant`：`"松风行舟"` 不会被改成 `松风晚渡旅行社`（长度不同）；`"雾隐晚渡"` 不被改成 `雾隐行舟`（`hotwords=["雾隐行舟"]`，拼音 wan du ≠ xing zhou）。
  - `test_short_hotwords_ignored`：`"周导"`（2 字）不参与。
  - `test_hotword_disabled_noop`：enabled=False 时段落原样返回、无 corrections 键。
  - `test_recognize_zh_modes`（有模型时）：SenseVoice `test_wavs/zh.wav` 经 `detect_speech` + `recognize(mode="both")` → 第一段 `text` 含阿拉伯数字 `9`，`text_raw` 含 `九` 且不含阿拉伯数字。
  - `test_recognize_empty`：空段列表返回 `[]` 且不加载模型（mock `get_recognizer` 断言未调用）。
- [ ] **Step 2–5:** 失败 → 实现 → 通过 → 提交

### Task 8: 步骤 4（说话人分离）

**Files:**
- Create: `pipeline/step4_diarize.py`
- Test: `tests/test_step4_diarize.py`

**Interfaces:**
- Produces：`diarize_turns(samples, sr, cfg) -> list[tuple[float, float, int]]`（sherpa-onnx 离线分离，`num_clusters=cfg["diarize"]["num_speakers"]`，返回 0 起的说话人编号；音频短于 2 秒返回 `[(0, 时长, 0)]`）；`attach_speakers(segments, turns) -> list[dict]`（按重叠时长最大配对，写 `speaker_id`（1 起）和 `speaker="说话人N"`，无重叠写 `未知`、`speaker_id=None`）；登记 `diarize/baseline`；`apply_speaker_map(segments, mapping: dict[str, str]) -> list[dict]`（只改 `speaker`，保留 `speaker_id`）；`speaker_durations(segments) -> dict[str, float]`；`SPEAKER_ROLES = ["导游", "游客", "店员", "司机", "经理", "未知"]`

- [ ] **Step 1: 写失败的测试**
  - `test_attach_by_overlap`：turns `[(0,5,0),(5,10,1)]`，段 `[1,4]`→说话人1，`[4.5,9]`→说话人2，`[11,12]`→未知。
  - `test_apply_speaker_map`：`{"说话人1": "导游"}` 后 speaker 为导游、speaker_id 仍为 1。
  - `test_speaker_durations`。
  - `test_four_speakers`（有模型时）：`0-four-speakers-zh.wav`，`num_speakers=4` → turns 里不同编号恰为 4 个。
- [ ] **Step 2–5:** 失败 → 实现 → 通过 → 提交

### Task 9: 步骤 5（数字、证号、金额规范化与提取）

**Files:**
- Create: `pipeline/step5_normalize.py`
- Test: `tests/test_step5_normalize.py`

**Interfaces:**
- Produces：
  - `NUMBER_TYPES = ["金额", "电话", "证号", "合同号", "订单号", "时刻", "日期", "数量", "时长", "其他"]`、`MAIN_TYPES = NUMBER_TYPES[:7]`
  - `number_type(value: str) -> str`：判断顺序与规则（写进模块说明）：含`元`→金额；`^\d{3,4}-\d{4}-\d{4}$`→电话；`^[A-Z]{2}-\d{4}-\d{4}$`→证号；`^HT-`→合同号；`^DD-`→订单号；`^\d{1,2}:\d{2}$`→时刻；含`年`/`月`/`日`且以数字开头（不含`日游`）→日期；以`分钟`/`小时`/`天`/`晚`/`年`/`个月`/`个工作日`结尾或含`小时`→时长；以量词结尾（人、家、盒、克、张、饼、个、泡、样、笔、排、页、条、号门、房间、折、%、毫米、公里）或以`第`开头→数量；其余→其他
  - `spoken_to_digits(text: str) -> str`：汉字读法转阿拉伯数字。基线：先按规则处理逐位读的串（由空格分隔的、全是`零一二三四五六七八九幺`的字组成的号码段，以及前面带两个大写字母的证号，转成 `0871-0000-6688`、`YN-0000-3721` 这样的连字符格式；合同号 `HT`、订单号 `DD` 同理），再保护`一下`、`一般`、`一点`、`一个是`、`一起`、`一直`、`一样`、`统一`、`万一`等词，再用 `cn2an.transform(text, "cn2an")`，再把`块`/`块钱`紧跟数字的改成`元`，`X点Y`/`X点Y分`（时间语境：前面有`上午`/`下午`/`早上`/`晚上`/`中午`/`点钟`）转成 `HH:MM`（下午、晚上加 12）
  - `extract_numbers(text: str) -> list[str]`：从阿拉伯数字文本提取规范写法（金额 `2800元`、`2800多元`、`5.5元`、`20元/克`；电话；证号；合同号；订单号；时刻；日期；带量词的数量和时长），去重保序
  - `extract_entities(text: str) -> list[str]`：匹配 `data/fictional_names.csv` 里类型为旅行社、店铺、酒店、品牌、演出与设施的名称（长名优先）
  - 登记 `normalize/baseline`：`(segment, mode, cfg) -> segment`；mode=`spoken`（输入是汉字读法：剧本演示、测评模式）时，若无 `text_raw` 先把原文存进 `text_raw`，再 `text = spoken_to_digits(text)`；mode=`display` 不改 `text`；两种模式都写 `numbers`、`entities`
  - `numbers_accuracy(lines: list[dict]) -> dict`：对每行 `spoken_to_digits` + `extract_numbers`，与 `numbers` 金标准按行集合比较，返回 `{"by_type": {类型: {"gold", "hit", "pred", "precision", "recall"}}, "main": {...}, "secondary": {...}}`

- [ ] **Step 1: 写失败的测试**
  - `test_number_type`（参数化）：`2800元`→金额、`0871-0000-6688`→电话、`YN-0000-3721`→证号、`HT-20260000-118`→合同号、`DD-0000-5566`→订单号、`15:40`→时刻、`10月5日`→日期、`40分钟`→时长、`3盒`→数量、`第2天`→数量、`99999`→其他。
  - `test_spoken_phone`：`"零八七一 零零零零 六六八八"` → 含 `0871-0000-6688`。
  - `test_spoken_cert`：`"YN 零零零零 三七二一"` → 含 `YN-0000-3721`。
  - `test_spoken_money`：`"这个手镯两千八百块"` → 含 `2800元`；`extract_numbers` 得 `["2800元"]`。
  - `test_protected_words`：`"大家等一下"` 不含 `1下`；`"一般来说"` 不含 `1般`。
  - `test_time`：`"下午三点四十集合"` → 含 `15:40`。
  - `test_script_accuracy_report`：`numbers_accuracy(load_lines())` 的 `main` 召回率 ≥ 0.6（打印各类型数字）；已知失败（`两个半小时`、`三四十块`、`九点五十`）各写一个 `pytest.mark.xfail(reason=...)` 测试。
  - `test_extract_entities`：`"我们是雾隐行舟旅行社的"` → `["雾隐行舟旅行社"]`。
- [ ] **Step 2–5:** 失败 → 实现 → 通过 → 提交（把剧本上的正确率写进 `docs/progress.md`）

### Task 10: 文字归一与步骤 6（话术分类）

**Files:**
- Create: `pipeline/text_norm.py`、`pipeline/step6_classify.py`、`pipeline/rules_keywords.json`、`pipeline/tf_classifier.py`
- Test: `tests/test_text_norm.py`、`tests/test_step6_classify.py`

**Interfaces:**
- Produces (`pipeline.text_norm`，只用标准库)：`normalize_for_cer(text) -> str`（NFKC 全角转半角；只保留汉字、ASCII 字母、数字；字母转大写）；`normalize_for_classification(text) -> str`（NFKC；字母大写；把连续 ASCII 数字（含其间的 `.:/-`）替换为 `#`；把两个及以上连续的 `零〇一二两三四五六七八九十百千万亿` 替换为 `#`；去掉标点和空白）
- Produces (`pipeline.step6_classify`)：`load_keywords() -> dict[str, list[str]]`（`rules_keywords.json`：`{"_说明": "...", "priority": [...], "keywords": {类别: [词...]}}`，7 个类别都有键，`其他`为空表）；登记 `classify/baseline`：`(texts, cfg) -> list[str]`（对归一后的文字数每类命中次数，最多者胜，平局按 priority；全不命中→`其他`）；登记 `classify/tf_model`：用 `models/classifier/`，不可用时退回 baseline 并 `warnings.warn` 中文说明；`classify_segments(segments, cfg, method: Callable | None = None) -> list[dict]`（写 `category` 和 `label = label_output(category)`）
- Produces (`pipeline.tf_classifier`)：`is_tf_available() -> bool`；`load_classifier(model_dir) -> Callable[[list[str]], list[str]] | None`（目录里要有 `model.h5`、`vocab.json`、`meta.json`；TF 没装或文件缺失返回 None；TensorFlow 只在函数内导入）；`save_classifier(model, vocab: dict[str, int], meta: dict, model_dir) -> None`；`encode(texts, vocab, max_len) -> list[list[int]]`（先 `normalize_for_classification`，按字查表，未知字为 1，补 0）

- [ ] **Step 1: 写失败的测试**
  - `test_cer_norm`：`normalize_for_cer("ＡＢ，你好 12！")` == `"AB你好12"`。
  - `test_classification_norm_consistent`：`normalize_for_classification("两千八百块")` 与 `normalize_for_classification("2800块")` 相等（都是 `#块`）；`"等一下"` 不变成 `#`。
  - `test_labels_only_legal`（参数化 lines.csv 前 200 句）：`classify_segments` 输出的 label 都在 `{""} ∪ {"疑似·"+x for x in FLAG_LABELS}`。
  - `test_keywords_not_copied_from_lines`：每个关键词长度 ≤ 6，且不等于任何一句台词去标点后的全文。
  - `test_rules_baseline_report`：在 1055 句上算各类召回率并打印，不设下限；`正常讲解`的误报率（被标成任何疑似类的比例）打印出来。
  - `test_tf_model_fallback`：`cfg` 指向空目录时选 `tf_model` → 结果与 baseline 相同且发出 warning。
- [ ] **Step 2–5:** 失败 → 实现 → 通过 → 提交（基线召回率、误报率写进 progress.md）

### Task 11: 步骤 7（疑似片段）与步骤 8（核查初稿）

**Files:**
- Create: `pipeline/step7_clips.py`、`pipeline/step8_report.py`、`pipeline/table.py`
- Test: `tests/test_step7_clips.py`、`tests/test_step8_report.py`、`tests/test_table.py`

**Interfaces:**
- Produces (`pipeline.step7_clips`)：登记 `clips/baseline`：`(segments, samples, sr, cfg) -> list[tuple[float, float, int]]`（label 非空的段，前后各留 `cfg["clips"]["padding"]` 秒，裁到 [0, 时长]）；`clip_filename(n: int, start: float, end: float) -> str` → `f"clip_{n:03d}_{start:08.1f}-{end:08.1f}.wav"`；`export_clips(segments, samples, sr, out_dir, cfg, method=None) -> list[dict]`（按采样点切，写 WAV 和 `clips_index.csv`（序号、开始、结束、说话人、文字、标签、文件），在对应段写 `clip`）
- Produces (`pipeline.step8_report`)：`TITLE = "旅游纠纷录音材料核查初稿（疑似、待核查）"`；`DISCLAIMER = "本初稿由语音识别等人工智能技术自动生成，识别和分类都可能出错。所有标注均为“疑似、待核查”，不代表任何定性结论，必须由工作人员对照原始录音逐条复核。本工具不鉴定录音的真伪。"`；`AUTHOR = "旅游纠纷录音材料整理工具（教学原型）"`；`FORBIDDEN_WORDS = ["违规", "违法", "执法级准确率", "可作为法律证据"]`；`fmt_mmss(seconds) -> str`（`02:05.4` 格式，分:秒）；`build_docx(segments, meta, path) -> None`（内容顺序照 build_spec 5.8 的 7 项；文档属性 title=TITLE、author=AUTHOR、comments=`由人工智能技术自动生成`）；`export_bundle(segments, meta, out_dir, clips_dir: str | None = None) -> str`（生成 `review_draft.docx`、`segments.csv`、`segments.json`、`clips/`、`说明.txt`，打包成 `out_dir/<原文件名去扩展名>_review.zip`（ASCII 安全：非 ASCII 字符替换成 `_`），返回 zip 路径）
- Produces (`pipeline.table`)：`TABLE_HEADERS = ["序号", "开始", "结束", "说话人", "文字", "标签", "数字", "复核结论", "复核意见"]`；`segments_to_rows(segments) -> list[list]`；`rows_to_segments(rows, segments) -> list[dict]`（用表格里的说话人、文字、标签、复核结论、复核意见覆盖对应段；标签不合法时清空并记一条 `review_note` 提示；复核结论不在 `REVIEW_CHOICES` 时设为 `未复核`）

- [ ] **Step 1: 写失败的测试**
  - `test_clip_padding_and_bounds`：段 `[0.5, 2.0]` 标费用、音频 3 秒、padding 1 → `(0.0, 3.0, 0)`；无标签段不出片段。
  - `test_clip_filename`：`clip_filename(3, 12.4, 18.9) == "clip_003_000012.4-000018.9.wav"`。
  - `test_export_clips_exact_samples`：导出 WAV 的采样数 = (end-start)*16000（±1）。
  - `test_docx_contents`：生成后用 python-docx 读回：段落文字含 TITLE 和 DISCLAIMER；`core_properties.author == AUTHOR`；全文不含 FORBIDDEN_WORDS 任何一个。
  - `test_bundle_zip`：zip 内有 5 类文件，`说明.txt` 含 NOTICE，`segments.json` 的 `meta.notice` 为 NOTICE。
  - `test_rows_roundtrip_with_edits`：改第 2 行说话人为 `导游`、复核结论为 `确认`、复核意见 `已听原声` 后 `rows_to_segments` 反映修改；把标签改成 `违规` → 标签被清空、`review_note` 有提示。
- [ ] **Step 2–5:** 失败 → 实现 → 通过 → 提交

### Task 12: 串起整条流程、剧本文本演示、各组文件

**Files:**
- Modify: `pipeline/__init__.py`（加 `run_pipeline`）、`pipeline/methods.py`（`load_all` 改严格导入）
- Create: `pipeline/script_demo.py`、`pipeline/groups/__init__.py`、`pipeline/groups/g1_denoise.py`、`g2_far_field.py`、`g3_diarize.py`、`g4_numbers.py`、`g5_hotwords.py`、`g6_classifier_a.py`、`g7_classifier_b.py`、`g8_clips.py`
- Test: `tests/test_pipeline.py`、`tests/test_groups.py`、`tests/test_script_demo.py`

**Interfaces:**
- Produces：`run_pipeline(path, options: dict | None = None, progress=None, cfg: dict | None = None) -> tuple[list[dict], dict]`。options 键：`num_speakers`（-1 自动、2–5）、`hotword_fix: bool`、`hotwords: list[str] | None`（None 时用 `data/hotwords.txt`）、`methods: dict`（覆盖 config）、`asr_mode`（默认 `display`）、`out_dir`（默认 `outputs/<YYYYmmdd-HHMMSS>_<ASCII 化文件名>`）。依次：step1 ingest → step2 detect_speech → step3 recognize → hotword → diarize → normalize(mode=`display`，`asr_mode="eval"` 时用 `spoken`) → classify_segments → export_clips → 不导出 Word（导出由界面或调用方用 `export_bundle`）。返回 meta：`file`、`duration`、`sha256`、`processed_at`（ISO 8601 带时区）、`models`（`{"asr": "sense-voice-int8-2024-07-17", "vad": "silero_vad", "diarization": "pyannote-3.0 + campplus"}`）、`options`、`methods`（实际用的做法名）、`notice`、`timings`（每步秒数）、`rtf`（总耗时/时长）、`work_dir`、`wav`、`qc`、`message`（0 段时为 `没有检测到人声，请检查录音`）。每步结束 `logging.info` 打印耗时和实时率。
- Produces (`pipeline.script_demo`)：`script_segments(script_id, chars_per_minute=220) -> list[dict]`（每句一段：说话人=剧本角色，`text` 为台词，时间按有效字数累计估算，`source="script_demo"`，`gold_label`、`gold_numbers` 来自剧本）；`run_script_demo(script_id, cfg=None, methods=None) -> tuple[list[dict], dict, dict]`（步骤 5 用 spoken 模式、步骤 6 分类；第三项是对照统计：`total`、`flagged`、`correct_flags`、`false_positives`（gold 为正常讲解却被标疑似）、`missed`、`fp_rate`；meta 带 `demo_notice = "演示模式：直接使用剧本文字，没有经过语音识别；时间为估算"`）
- Produces (`pipeline.groups`)：每个文件用中文写清楚"你们组负责什么、在哪个函数里改、怎么测、注意什么"，并登记一个名为 `gN` 的做法，函数体调用基线（第 1 组：denoise、vad；第 2 组：enhance、vad；第 3 组：diarize；第 4 组：normalize；第 5 组：hotword；第 6 组：classify（用 `models/classifier_g6/`，并定义 `build_model(vocab_size: int, num_classes: int, max_len: int)` 返回与基线相同的卷积模型，TensorFlow 在函数内导入）；第 7 组：classify（`models/classifier_g7/`，`build_model` 先与基线相同，注释说明改成 LSTM）；第 8 组：clips）

- [ ] **Step 1: 写失败的测试**
  - `test_groups_registered`：`load_all()` 后每个槽位 `available()` 含 baseline 和对应组名（denoise 含 g1；vad 含 g1、g2；enhance 含 g2；diarize 含 g3；normalize 含 g4；hotword 含 g5；classify 含 g6、g7、tf_model；clips 含 g8）。
  - `test_group_stubs_equal_baseline`：对 3 秒正弦、示例段落、示例文字，每个组的初始做法输出与基线相同。
  - `test_pipeline_silence`：2 秒静音 → `segments == []`，`meta["message"]` 为上面的提示，不抛异常。
  - `test_pipeline_four_speakers`（有模型时）：`0-four-speakers-zh.wav`，`num_speakers=4` → 段数 ≥ 5；每段有 start/end/speaker/text/label；不同 `speaker_id` 恰为 4；`meta["rtf"] > 0`；`meta["timings"]` 有 6 个以上步骤；标签全部合法。
  - `test_script_demo_g8`：`run_script_demo("G8-S1")` → 段数等于台词行数；返回的对照统计有 `fp_rate`；用 `export_bundle` 生成的 docx 含 DISCLAIMER。打印 G8 三个剧本的误报率。
- [ ] **Step 2–5:** 失败 → 实现 → 通过 → 提交；在 `docs/progress.md` 记录四人测试音频上各步耗时和实时率。

## 阶段 C：界面

### Task 13: Gradio 界面（整理录音、剧本文本演示、使用说明）

**Files:**
- Create: `app.py`、`pipeline/ui_text.py`（界面上的长段中文：横幅、使用说明、各选项说明）
- Test: `tests/test_app.py`、`tests/test_static_checks.py`

**Interfaces:**
- Consumes: `run_pipeline`、`run_script_demo`、`export_bundle`、`segments_to_rows`、`rows_to_segments`、`apply_speaker_map`、`speaker_durations`、`methods.available`、`SLOT_TITLES`
- Produces：`build_app(cfg: dict | None = None) -> gr.Blocks`（不启动）；`main(argv=None)`：参数 `--host`、`--port`、`--inbrowser`；启动前设 `os.environ["GRADIO_ANALYTICS_ENABLED"]="False"`、`GRADIO_TEMP_DIR=<ROOT>/tmp`；路径含非 ASCII 字符时打印中文警告；`DEMO_USERNAME`/`DEMO_PASSWORD` 都设置时传 `auth`；`launch(server_name, server_port, max_file_size, inbrowser, theme=gr.themes.Soft(font=["Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "sans-serif"]))`，队列 `default_concurrency_limit=1`；启动时删除 `outputs/`、`tmp/` 中超过 `cleanup_hours` 的内容。
- 处理函数（模块级，可单独测试）：`process_file(file_path, num_speakers, hotword_on, hotword_text, *method_names, progress=gr.Progress()) -> tuple[str, list[list], dict]`（摘要 Markdown、表格行、状态）；`play_row(state, evt: gr.SelectData) -> tuple[int, np.ndarray] | None`；`apply_mapping(state, rows, mapping_text) -> tuple[list[list], dict]`（mapping_text 每行 `说话人1=导游`）；`export_files(state, rows) -> list[str]`（返回 zip、docx、csv、json 路径）；`run_demo(script_id) -> tuple[str, list[list], str]`（摘要、表格、对照说明）
- 布局照 spec 5.4：横幅；标签页"整理录音"（`gr.File` 上传、说话人数下拉 自动/2/3/4/5、降噪 开/关（开=denoise 用 noisereduce）、热词纠错 开/关 + 可编辑热词文本框、折叠的"高级设置"每个槽位一个下拉、开始按钮、摘要、可编辑 `gr.Dataframe`（复核结论列提示可选值）、说话人映射文本框 + 应用按钮（下方显示各说话人时长和"说话时间最长的可能是导游，请人工确认"）、`gr.Audio` 播放选中行、导出按钮 + `gr.File` 下载）；标签页"剧本文本演示"（剧本下拉 24 项 `G1-S1 石林路上推特产`、运行按钮、演示提示、表格、对照统计）；标签页"使用说明"（Markdown）。"数据校对""录音质检"标签页在 Task 17 加。

- [ ] **Step 1: 写失败的测试**
  - `test_build_app`：`build_app()` 返回 `gr.Blocks`，不启动服务。
  - `test_run_demo_function`：`run_demo("G8-S1")` 表格行数 = 台词数，摘要含 `演示模式`。
  - `test_apply_mapping_and_export`（不需要模型）：用剧本演示得到的状态，`apply_mapping` 后导出，zip 里 CSV 的说话人列含 `导游`。
  - `test_process_file_four_speakers`（有模型时）：返回的行数 ≥ 5，摘要含时长和疑似片段数。
  - `test_server_starts`（有模型时）：子进程 `python app.py --port 7899`，30 秒内 `GET http://127.0.0.1:7899/` 返回 200，然后结束进程。
  - `tests/test_static_checks.py`：全仓库 `*.py` 里没有 `share=True`；`pipeline/` 下没有 `requests`、`urllib.request`、`http.client`、`socket` 的导入（`models/download_models.py` 不在 `pipeline/` 下，不受限）；没有形如 `(api_key|secret|token)\s*=\s*["'][A-Za-z0-9]{16,}` 的字符串；`pipeline/`、`app.py` 里的中文字符串不含 FORBIDDEN_WORDS（`step8_report.py` 里定义 FORBIDDEN_WORDS 的那一行除外）。
- [ ] **Step 2–5:** 失败 → 实现 → 通过 → 用 Playwright（`executablePath: /opt/pw-browsers/chromium`）打开页面截图，确认横幅、三个标签页可见、上传四人测试音频后表格出现（截图存到草稿目录，不提交）→ 提交

## 阶段 D：便携包与 CI（尽早交付"课前测试版"）

### Task 14: 自检脚本、Windows 便携包构建、CI

**Files:**
- Create: `tools/selfcheck.py`、`packaging/windows/build_portable.ps1`、`packaging/windows/start.bat`、`packaging/windows/selfcheck.bat`、`packaging/windows/README.txt`（UTF-8 带 BOM，中文使用说明）、`.github/workflows/ci.yml`、`.github/workflows/build-windows-portable.yml`、`scripts/windows/install.bat`、`scripts/windows/start.bat`
- Test: `tests/test_selfcheck.py`

**Interfaces:**
- `tools/selfcheck.py`：`main(argv=None) -> int`；依次检查并打印中文"通过/不通过"：Python 版本 ≥ 3.10、项目路径是否含中文（只警告）、ffmpeg 可用、必需模型齐全、`run_pipeline(models/0-four-speakers-zh.wav, {"num_speakers": 4})` 成功且段数 ≥ 5、`export_bundle` 生成 zip；`--quick` 跳过识别；全部通过返回 0。
- `build_portable.ps1` 参数：`-PythonVersion 3.11.9 -OutDir dist -Version <tag 或 dev>`；步骤：下载 `https://www.python.org/ftp/python/<ver>/python-<ver>-embed-amd64.zip` 解压到 `dist/tiandu/python/`；把 `python311._pth` 改为 `python311.zip`、`.`、`Lib\site-packages`、`..`、`import site` 五行；`python -m pip install --target dist/tiandu/python/Lib/site-packages -r requirements.txt`（runner 上 setup-python 3.11）；复制 `app.py config.yaml pipeline tools models/download_models.py data docs reports README.md GLOSSARY.md` 和 `packaging/windows/*.bat`、`README.txt` 到 `dist/tiandu/`；在 `dist/tiandu` 下用便携 Python 运行 `models/download_models.py`；运行 `python\python.exe tools\selfcheck.py`，失败则退出码非 0；`Compress-Archive` 成 `dist/tiandu-portable-win64-<Version>.zip`。
- `start.bat`：`@echo off`、`chcp 65001 >nul`、`cd /d "%~dp0"`、`set GRADIO_ANALYTICS_ENABLED=False`、`set PYTHONUTF8=1`、`python\python.exe app.py --inbrowser`、`pause`（只有 ASCII 命令；中文提示由 Python 打印）。
- `ci.yml`：`on: [push, pull_request]`；矩阵 `ubuntu-latest`、`windows-latest`，Python 3.11；缓存 `models/`（key 含 `pipeline/models.py` 的哈希）；`pip install -r requirements.txt -r requirements-dev.txt`；`python models/download_models.py`；`ruff check .`；`pytest -q`。
- `build-windows-portable.yml`：`on: workflow_dispatch`、`push: tags: ['v*']`、`push: branches: ['**'] paths: ['packaging/**', 'requirements.txt', '.github/workflows/build-windows-portable.yml']`；`windows-latest`；运行 `build_portable.ps1`；`actions/upload-artifact` 上传 zip（保留 30 天）；打标签时用 `softprops/action-gh-release@v2` 发布为 Release（标签含 `-` 时设为预发布）。

- [ ] **Step 1: 写失败的测试** `test_selfcheck_quick`：`main(["--quick"]) == 0`（本环境有模型）。
- [ ] **Step 2–3:** 失败 → 实现 selfcheck 和全部打包、CI 文件。
- [ ] **Step 4: 本地验证**：`pytest -q`、`ruff check .` 通过；`.venv/bin/python tools/selfcheck.py` 全部通过。
- [ ] **Step 5: 提交并推送**；用 GitHub MCP 的 `actions_list`/`get_job_logs` 看 `ci.yml`（两个系统）和 `build-windows-portable.yml` 的结果，失败就按日志修到通过。
- [ ] **Step 6: 发布课前测试版**：推送标签 `v0.1.0-pre`，等 Release 生成，确认 zip 可从公开链接下载（`curl -I` 返回 200/302），把链接给老师。

## 阶段 E：数据工具（课程重点）

### Task 15: 数据池入池与参考文本

**Files:**
- Create: `pipeline/pool.py`、`tools/ingest_pool.py`、`tools/export_references.py`
- Test: `tests/test_pool.py`

**Interfaces:**
- Produces (`pipeline.pool`)：`FILENAME_RE = re.compile(r"^G([1-8])-S([1-3])-([QNF])\.([A-Za-z0-9]+)$")`；`CONDITIONS = {"Q": "安静", "N": "嘈杂教室", "F": "口袋或远距离"}`；`parse_recording_name(name: str) -> dict`（返回 `script_id`、`group`、`condition`、`stem`；不符合时抛 `ValueError`，中文信息给出正确示例 `G1-S1-Q.m4a` 和常见错误原因：小写、多余空格或"(1)"、组号超出 1–8）；`pool_paths(root) -> dict[str, Path]`（raw、normalized、references、annotations_speakers、annotations_clips、digits、versions，以及 manifest、qc_report、proofread_log、acceptance、acceptance_summary 五个 CSV）；`init_pool(root) -> None`（建目录）；`ingest_pool(root, cfg) -> dict`（处理 `raw/` 里尚未入池（按 SHA-256 判断）的文件：名字不合格的列进 `errors` 不处理；合格的转成 `normalized/<stem>.wav`、质检（期望时长来自 recording_plan）、追加 manifest 和 qc_report、原始文件设只读；返回 `{"added": [...], "skipped": [...], "errors": [{"file", "reason"}]}`）；`export_references(root, overwrite: bool = False) -> list[Path]`（为 recording_plan 的 72 个录音各写 `references/<stem>.txt`，内容为剧本参考文本；已存在且 overwrite=False 时不覆盖）；`log_proofread(root, stem, proofreader, note="") -> None`
- CLI：`python tools/ingest_pool.py [--pool DIR]`（打印中文汇总，有 errors 时退出码 1）；`python tools/export_references.py [--pool DIR] [--overwrite]`。manifest 列：`文件编号,剧本编号,录音条件,原始文件,转换后文件,时长（秒）,原始采样率,质检结果,上传时间,SHA-256`。

- [ ] **Step 1: 写失败的测试**
  - `test_parse_names`（参数化）：`G1-S1-Q.wav` 通过；`g1-s1-q.wav`、`G1-S1-Q (1).m4a`、`G9-S1-Q.wav`、`G1-S4-Q.wav`、`G1-S1-X.wav`、`G1_S1_Q.wav` 都抛 ValueError 且信息含 `G1-S1-Q`。
  - `test_ingest_pool`：在 `raw/` 放 `G1-S1-Q.m4a`（ffmpeg 正弦）和 `bad name.wav` → added 1、errors 1；`normalized/G1-S1-Q.wav` 存在；manifest 一行，表头为中文；再运行一次 → skipped 1、added 0。
  - `test_export_references`：生成 72 个文件；`G1-S1-Q.txt` 与 `script_reference_text("G1-S1")` 相同；改写其中一个后再运行不覆盖。
  - `test_log_proofread`：两次不同校对人 → proofread_log 两行。
- [ ] **Step 2–5:** 失败 → 实现 → 通过 → 提交

### Task 16: 验收表、冻结版本、Audacity 标签转换

**Files:**
- Create: `pipeline/annotations.py`、`tools/check_pool.py`、`tools/freeze_pool.py`、`tools/labels_to_csv.py`；Modify: `pipeline/pool.py`
- Test: `tests/test_acceptance.py`、`tests/test_annotations.py`

**Interfaces:**
- Produces (`pipeline.annotations`)：`read_audacity_labels(path) -> list[tuple[float, float, str]]`（制表符分隔，跳过以 `\` 开头的频率行，去掉标签两侧空白）；`write_turns_csv(path, rows)`、`read_turns_csv(path) -> list[tuple[float, float, str]]`（列 `start,end,speaker`）；`annotated_seconds(rows) -> float`
- Produces (`pipeline.pool`)：`acceptance(root) -> list[dict]`（每个计划录音一行：`文件`、`组`、`条件`、`已交`、`命名`、`质检`（合格/问题文字/未入池）、`校对遍数`（不同校对人数）、`说话人标注（秒）`、`片段标注`、`通过`（已交 且 质检合格 且 校对遍数 ≥ 2））；`acceptance_summary(rows) -> list[dict]`（每组：已交/应交、通过数、说话人标注总分钟、是否达到 10 分钟）；`write_acceptance(root) -> tuple[Path, Path]`；`freeze(root, version: str) -> Path`（复制 manifest、qc_report、proofread_log、acceptance、references/、annotations/ 到 `versions/<version>/`，写 `checksums.json`（每个文件的 SHA-256，含 normalized WAV 的哈希但不复制 WAV）；版本已存在抛 `FileExistsError`）；`current_version(root) -> str | None`
- CLI：`check_pool.py [--pool DIR]` 打印每组汇总；`freeze_pool.py --version v1 [--pool DIR]`；`labels_to_csv.py 输入标签.txt 输出.csv`（或 `--pool DIR --file G3-S1-Q --kind speakers|clips` 直接写到 annotations 下）

- [ ] **Step 1: 写失败的测试**
  - `test_audacity_parse`：含频率行的标签文件 → 2 条，时间为 float。
  - `test_acceptance_flow`：空池 → 72 行全部未交；入池一个文件、两次校对、写一个 600 秒的说话人标注 → 该行 `通过` 为 True；第 1 组汇总 `说话人标注` = 10.0 分钟且达到。
  - `test_freeze`：`freeze(root, "v1")` 生成目录和 checksums.json；再次 `freeze("v1")` 抛 FileExistsError；`current_version` 返回 `v1`。
- [ ] **Step 2–5:** 失败 → 实现 → 通过 → 提交

### Task 17: 数据校对与录音质检标签页、语谱图工具

**Files:**
- Create: `pipeline/align.py`、`pipeline/features.py`、`tools/show_spectrogram.py`；Modify: `app.py`、`pipeline/ui_text.py`
- Test: `tests/test_align.py`、`tests/test_features.py`；Modify: `tests/test_app.py`

**Interfaces:**
- Produces (`pipeline.align`)：`cer_details(reference: str, hypothesis: str) -> dict`（两边先 `normalize_for_cer`；用 `rapidfuzz.distance.Levenshtein.editops`；返回 `cer`、`sub`、`dele`、`ins`、`n_ref`；参考为空时 cer 为 0.0（假设也空）或 1.0）；`align_segments(segment_texts: list[str], reference: str) -> list[dict]`（把各段识别文字拼接后与参考文本逐字对齐，返回每段 `hyp`、`ref`（参考文本中对应的原文片段，保留标点）、`diff: bool`、`cer`）
- Produces (`pipeline.features`，只用 numpy + matplotlib)：`spectrogram_db(samples, sr, n_fft=400, hop=160) -> tuple[np.ndarray, np.ndarray, np.ndarray]`（times、freqs、dB）；`mfcc(samples, sr, n_mfcc=13, n_mels=40, n_fft=400, hop=160) -> np.ndarray`（形状 `[帧数, n_mfcc]`；预加重 0.97、汉明窗、梅尔滤波、对数、DCT-II）；`plot_recording(samples, sr, path=None, title="")`（三行：波形、语谱图、MFCC；返回 `matplotlib.figure.Figure`；中文字体候选 `Microsoft YaHei`、`SimHei`、`Noto Sans CJK SC`）
- app 新增：标签页"数据校对"：数据池路径（默认 config）、文件下拉（`normalized/` 下的 wav）、校对人输入框、"生成对照"按钮（测评模式识别，结果缓存到 `data_pool/asr_cache/<stem>.json`）、对照表（序号、开始、结束、识别结果、参考文本、是否不同）、点行播放、参考文本编辑框、"保存参考文本"（写回 txt 并 `log_proofread`）、整体字错率显示；标签页"录音质检"：上传或选池内文件 → 显示 `plot_recording` 图、质检问题列表、`ui_text.QC_EXPLAIN`（每项中文解释和怎么办）
- `tools/show_spectrogram.py 录音.wav [--out 图.png]`：任何格式先经 `convert_to_wav` 到临时文件。

- [ ] **Step 1: 写失败的测试**
  - `test_cer_details`：`cer_details("雾隐行舟旅行社", "雾影行走旅行社")` → sub 2、cer = 2/7（与视频例子一致）；`cer_details("你好", "你好啊")` → ins 1。
  - `test_align_segments`：参考 `"今天去石林。下午回昆明。"`、段 `["今天去石林", "下午回昆名"]` → 第 1 段 diff False、第 2 段 diff True 且 `ref` 为 `下午回昆明。`。
  - `test_mfcc_shape`：1 秒 16k 正弦 → `mfcc` 形状 `(98±2, 13)`；`spectrogram_db` 在 440Hz 附近有最大能量。
  - `test_plot_recording_writes_png`。
  - `test_app_has_five_tabs`：`build_app()` 的配置里有"整理录音""剧本文本演示""数据校对""录音质检""使用说明"五个标签页名字。
- [ ] **Step 2–5:** 失败 → 实现 → 通过 → Playwright 截图检查两个新标签页 → 提交

## 阶段 F：测评、分类模型、补充数据

### Task 18: 测评库与 cer / hotwords / numbers 子命令、对比工具

**Files:**
- Create: `pipeline/evaluation.py`、`tools/evaluate.py`、`tools/compare.py`、`reports/g1/README.md`…`reports/g8/README.md`（报告模板）
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Produces (`pipeline.evaluation`)：`pool_items(root, files: list[str] | None = None) -> list[dict]`（manifest 中的录音：`stem`、`wav`、`reference`（txt 路径）、`group`、`condition`）；`eval_cer(root, cfg, methods=None, files=None) -> tuple[list[dict], dict]`（每个文件：识别（`mode="eval"`）→ hotword → `cer_details(参考, 拼接的 text_raw)`；汇总按条件、按组、全部；每行带 `stem`、`group`、`condition`、`cer`、`sub`、`dele`、`ins`、`n_ref`、`seconds`）；`eval_hotwords(root, cfg, methods=None, files=None) -> tuple[list[dict], dict]`（专名正确率：参考文本中出现的 hotwords 在识别结果中出现的比例；过度纠正：参考文本中出现的 `hotword_variants` 的 spoken_variant 在结果中被替换成 correct_name 的次数）；`eval_numbers_lines(cfg, methods=None) -> dict`（剧本文字上的 `numbers_accuracy`，可指定 normalize 做法）；`write_report(out_dir, name, rows, summary, notes: list[str]) -> tuple[Path, Path]`（CSV + Markdown；Markdown 开头写数据池版本、做法、参数、生成时间和固定局限说明 `剧本数据上的测评结果不代表真实场景的效果`）
- CLI：`python tools/evaluate.py {cer,hotwords,numbers,speakers,classify,clips} [--pool DIR] [--method 槽位=做法 ...] [--files G1-S1-Q ...] [--out reports/gN]`；`python tools/compare.py --slot denoise --method g1 --metric cer [--pool DIR] [--out reports/g1]`（同一数据跑基线和指定做法，写 `compare_<metric>.md/.csv`，表格列：分组项、基线、改进、差值）。
- 报告模板 `reports/gN/README.md`：本组专项、数据池版本、基线结果、改进做法说明、改进后结果、对比表、结论与局限、分工。

- [ ] **Step 1: 写失败的测试**（用 ffmpeg 生成的音频 + 测试音频改名建一个临时池，不报任何字错率数字到文档）
  - `test_eval_cer_runs_on_renamed_test_audio`（有模型时）：把 `0-four-speakers-zh.wav` 复制成 `raw/G1-S1-Q.wav`，入池、导出参考文本 → `eval_cer` 返回 1 行，有 `cer` 键且 0 ≤ cer。
  - `test_eval_numbers_lines`：返回 `main` 和 `by_type`。
  - `test_compare_writes_report`：用 `--metric numbers --slot normalize --method g4` 生成 md，表里基线与改进相等（g4 初始等于基线），差值 0。
  - `test_report_header`：Markdown 含 `剧本数据上的测评结果不代表真实场景的效果`。
- [ ] **Step 2–5:** 失败 → 实现 → 通过 → 提交

### Task 19: speakers / classify / clips 子命令

**Files:**
- Modify: `pipeline/evaluation.py`、`tools/evaluate.py`
- Create: `pipeline/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces (`pipeline.metrics`)：`speaker_error_rate(ref_turns, hyp_turns, step=0.01) -> float`（按 10 毫秒网格，在两边都有说话的时间上，用最优一一对应（匈牙利算法用 `scipy.optimize.linear_sum_assignment`）后标错的时长比例）；`confusion_matrix(gold, pred, labels) -> list[list[int]]`；`per_class_pr(gold, pred, labels) -> dict[str, dict]`；`false_positive_rate(gold, pred) -> float`（gold 为`正常讲解`的里被预测为 5 个 flag 类之一的比例）；`clip_boundary_error(ref_clips, hyp_clips) -> dict`（按最大重叠配对，返回起点、终点平均绝对误差和未配对数）
- Produces (`pipeline.evaluation`)：`eval_speakers(root, cfg, methods=None, files=None)`（只评有 `annotations/speakers/<stem>.csv` 的文件）；`eval_classify_rules(cfg, method="baseline") -> dict`（1055 句：混淆矩阵、各类 P/R、全体误报率、第 8 组误报率）；`eval_clips(root, cfg, methods=None, files=None)`（只评有 `annotations/clips/<stem>.csv` 的文件）

- [ ] **Step 1: 写失败的测试**
  - `test_speaker_error_perfect_and_swapped`：完全相同 → 0；标签互换（1↔2）→ 0；一半时间标错 → 0.5±0.01。
  - `test_pr_and_confusion`：小例子手算对照。
  - `test_false_positive_rate`：gold `[正常讲解, 正常讲解, 费用]`、pred `[费用, 正常讲解, 费用]` → 0.5。
  - `test_clip_boundary_error`。
  - `test_eval_classify_rules`：返回 7×7 混淆矩阵，`fp_rate_g8` 存在。
- [ ] **Step 2–5:** 失败 → 实现 → 通过 → 提交

### Task 20: 话术分类补充句子与 TensorFlow 训练脚本

**Files:**
- Create: `data/classification_extra.csv`、`tools/check_classification_data.py`、`tools/train_classifier.py`；Modify: `data/README.md`（加一节说明补充句子）
- Test: `tests/test_classification_data.py`、`tests/test_train_classifier.py`（TF 没装时跳过）

**Interfaces:**
- `data/classification_extra.csv` 列：`id,text,label,source,reviewed,reviewer,note`；`id` 为 `X0001` 起；`source` 全为 `AI生成`；`reviewed` 全为 `未审核`。目标数量：威胁消费 200、服务态度 150、行程变更 150、正常讲解 200、购物安排 110、费用 110、其他 80（合计约 1000）。生成规则：照 `docs/script_spec.md` 的内容红线和写法；虚构名称只用 `data/fictional_names.csv`；数字约八成用汉字读法、两成用阿拉伯数字；每句 6–60 字；不得与 `lines.csv` 任何一句去标点后相同；同一类里不得重复。
- `tools/check_classification_data.py [文件]`：检查列、类别合法、长度、重复（类内、与剧本）、红线词（`RED_LINE_WORDS`：违法、违规、警察、报警、开光、民族、宗教、政府、外地人、乡下人、骂、打人、滚 等，写在脚本里并注明来源 script_spec），打印每类数量；有问题退出码 1。
- `tools/train_classifier.py`：`--model baseline|g6|g7`（从 `pipeline.groups` 取 `build_model`，baseline 为 Embedding(词表,64) → Conv1D(128, 3, relu) → GlobalMaxPooling1D → Dense(7, softmax)）、`--extra/--no-extra`、`--eval logo|random|none`（logo：按组留一 8 折，汇总一个混淆矩阵；random：分层 8:2）、`--epochs 15`、`--out DIR`（默认 `models/classifier_<model>`）；最后用全部数据训练并 `save_classifier`；报告写到 `reports/g6` 或 `reports/g7`（`--report-dir`）；只依赖 tensorflow、numpy 和 `pipeline.text_norm`、`pipeline.data`、`pipeline.metrics`、`pipeline.tf_classifier`（这几个模块不导入 sherpa_onnx、gradio）。

- [ ] **Step 1: 写失败的测试** `test_extra_csv_valid`：`check_classification_data.main([...]) == 0`；每类数量 ≥ 目标的 90%；`test_train_light_imports`：子进程导入 `tools.train_classifier` 的依赖模块后 `sherpa_onnx`、`gradio` 不在 `sys.modules`；`test_train_smoke`（TF 已装时）：`--epochs 1 --eval none` 生成 `model.h5`、`vocab.json`、`meta.json`，`load_classifier` 能预测出合法类别。
- [ ] **Step 2: 生成句子**：按类别分给并行子任务生成，每个子任务读 `docs/script_spec.md`、`data/labels.json`、`data/fictional_names.csv` 和该类在 `lines.csv` 的例句，只输出 CSV 行；合并后跑检查脚本，再由一个独立审核子任务逐类抽查标签和红线，修正后再检查。
- [ ] **Step 3–5:** 实现脚本 → 在临时虚拟环境装 `tensorflow-cpu` 跑 `--eval logo` 一次，把"用 / 不用补充句子"的按组留一结果写进 `docs/progress.md` → 提交

### Task 21: 原理课与选做数字实验（tf_lab）

**Files:**
- Create: `tools/tf_lab/README.md`、`tools/tf_lab/mfcc_compare.py`、`tools/tf_lab/split_digits.py`、`tools/tf_lab/train_digits.py`
- Test: `tests/test_tf_lab.py`

**Interfaces:**
- `mfcc_compare.py 录音.wav`：用 `pipeline.features.mfcc` 和（装了 TF 时）`tf.signal` 各算一遍，画在一张图上，打印两者的相关系数。
- `split_digits.py 录音.wav --speaker 学号后四位 --out data_pool/digits`：Silero VAD（`min_speech_duration=0.1`、`min_silence_duration=0.4`）切段；段数必须是 50（"零到九"念 5 遍），不是 50 时打印每段时长并提示"念慢一点、每个字之间停顿约 1 秒，重录"，退出码 1；是 50 时按顺序存成 `<数字>/<speaker>_<遍>.wav`。
- `train_digits.py --data data_pool/digits`：MFCC（`pipeline.features.mfcc`，固定 1 秒补零）→ 小 CNN（Conv2D 16 → MaxPool → Conv2D 32 → GlobalAveragePooling → Dense 10）；按说话人留出 20% 做测试；打印准确率和混淆矩阵。

- [ ] **Step 1: 写失败的测试** `test_split_digits_wrong_count`：用 ffmpeg 生成 3 段"正弦+静音"的文件 → 退出码 1、不写文件（Silero 对正弦不一定检出，断言改为"段数 ≠ 50 时退出码 1"）；`test_mfcc_compare_without_tf`：没装 TF 时只画 numpy 版、退出码 0。
- [ ] **Step 2–5:** 失败 → 实现 → 通过 → 提交

### Task 22: 长录音测试

**Files:**
- Create: `tools/long_audio_test.py`
- Modify: `docs/progress.md`

**Interfaces:**
- `python tools/long_audio_test.py --minutes 60`：用 ffmpeg concat 把 `0-four-speakers-zh.wav` 重复拼到指定时长（放 `tmp/`），跑 `run_pipeline`（`num_speakers=4`），打印每步耗时、实时率、最大内存（`resource.getrusage`；Windows 上用 `psutil` 若可用，否则跳过），结果追加到 `docs/progress.md`。不进默认 pytest。

- [ ] **Step 1: 实现并运行 `--minutes 10`，再运行 `--minutes 60`**（后台运行），把数字写进 progress.md
- [ ] **Step 2: 提交**

## 阶段 G：文档（全部中文）

### Task 23: README 首页

**Files:**
- Create: `README.md`（覆盖 Task 1 移走后的位置）

章节和内容照 spec 5.9 的 15 节。必须包含：视频链接 `handoff/04_项目介绍视频/旅游投诉录音证据智能整理助手_项目介绍视频.mp4` 和要点；八步流程表（步骤、做什么、文件、槽位、负责组）；31 人分组表（spec 第 6 节）与组内四角色、客串安排（第 2 组每剧本缺 1 人、第 3 组和第 8 组每剧本缺 1 人）；10 周总表（spec 第 7 节）和 8/12 周对照；**8 组 × 10 周总表**（每格一句话，与任务书一致）；第 1 周便携包使用步骤（下载 Release → 解压到 `D:\asr\tiandu` 或 U 盘 → 双击 `start.bat` → 浏览器自动打开 → 下课前拷走文件）；联网安装步骤；录音、交录音、校对、标注、测评、对比的命令示例（Windows 写法 `python\python.exe tools\evaluate.py ...` 和开发环境写法）；提交代码（组长 GitHub Desktop、老师代交）；红线；建议考核方案（个人 40%：第 1–6 周任务、录音与校对、个人字错率作业、语谱图质检作业；小组 60%：数据质量 15、改进对比数据 20、代码与提交 10、报告 5、答辩 10；标明"建议，老师可改"）；目录结构；常见问题（至少 8 条：中文路径、电脑还原、端口被占用、浏览器没自动打开、上传后没反应、ffmpeg 找不到、识别很慢、TensorFlow 报错）；给老师的链接。

- [ ] **Step 1:** 写完后自查：每条命令都在本环境实际跑过一次（或对应测试覆盖）；表格与任务书、spec 一致；没有英文段落（命令、路径、代码除外）。
- [ ] **Step 2: 提交**

### Task 24: 8 份组任务书

**Files:**
- Create: `docs/groups/g1.md` … `docs/groups/g8.md`

每份按 spec 5.9 的 11 节结构；"第 1—10 周每周任务"每周写：课内、课外、具体步骤和命令、本周交什么、下课前拷走什么。第 1—6 周包含本组分管的数据质量项（spec 5.5 表）；第 7—8 周的改进方向至少 2 个、由易到难，引用本组文件里的函数名；测评命令与 `tools/evaluate.py`/`tools/compare.py` 的真实参数一致。第 2 组写明 3 人怎么合并角色；第 2、3、8 组写明需要客串的角色。

- [ ] **Step 1:** 8 份并行撰写（每份一个子任务，给它 spec、本组剧本概况、本组文件内容、测评命令帮助输出）。
- [ ] **Step 2:** 一致性检查：周次与 README 总表一致；命令可运行；文件名、函数名存在。
- [ ] **Step 3: 提交**

### Task 25: 指南与老师文档、协作配套

**Files:**
- Create: `docs/guides/install_windows.md`、`recording.md`、`consent_form.md`、`proofreading.md`、`annotation.md`、`evaluation.md`、`submit_code.md`、`ai_tools.md`；`docs/teacher/week0_checklist.md`、`portable_package.md`、`data_pool.md`、`github_setup.md`、`merging.md`、`week9_integration.md`、`gaps.md`；`.github/pull_request_template.md`、`.github/ISSUE_TEMPLATE/question.md`、`.github/ISSUE_TEMPLATE/weekly_report.md`；`CONTRIBUTING.md`；`Dockerfile`；`docs/adr/` 按需补充
- 要点：`consent_form.md` 写明用途、保存期限、是否公开、撤回方式、声纹单独同意、不采集 14 岁以下、录音只在本地处理；`gaps.md` 列 spec 第 11 节 + 机房实测、录音设备、存储、U 盘、学生 GitHub 账号、仓库是否设私有（原始对话记录在公开仓库里）、视频配音非商用许可、合作单位；`github_setup.md` 写加协作者、分支保护（必须 PR、必须 CI 通过、必须老师批准）的点击步骤；`submit_code.md` 写 GitHub Desktop 从安装到发 PR 的每一步，和"登不上时交哪几个文件给老师"。

- [ ] **Step 1:** 并行撰写。
- [ ] **Step 2:** 交叉检查与 README、任务书一致。
- [ ] **Step 3: 提交**

## 阶段 H：验收

### Task 26: 全面验收与正式预发布

- [ ] **Step 1:** `pytest -q`、`ruff check .` 全绿；`python models/download_models.py` 全部跳过；`python app.py` 启动后 Playwright 走一遍：上传四人测试音频 → 表格 → 点行播放 → 改说话人和复核结论 → 导出 zip；剧本文本演示跑 G1-S1 和 G8-S2；数据校对和录音质检页能打开。
- [ ] **Step 2:** build_spec 第 9 节验收清单逐条核对，写进 `docs/progress.md`。
- [ ] **Step 3:** 推送；CI 在 Windows 和 Linux 都通过；便携包构建和自检通过。
- [ ] **Step 4:** 独立审查：派一个子任务按 spec 和红线审查整个分支（代码 + 文档），修复发现的问题。
- [ ] **Step 5:** 打标签 `v0.2.0-pre`，Release 附便携包；把下载链接和使用说明发给老师。
