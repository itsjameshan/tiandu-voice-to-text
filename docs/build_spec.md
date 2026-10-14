# 开发规格：旅游纠纷录音材料整理（教学原型）

> **2026-10-14 变更：测评录音改为 AI 合成，学生不再录音。** 72 段测评录音（24 个剧本 × Q、N、F）全部由 AI 按剧本离线合成（sherpa-onnx + Kokoro v1.1，不做声音克隆），每段开头念 AI 合成声明，文件名带 `-TTS`。详见 `docs/adr/0006-ai-synthesized-recordings.md` 和本文第 4.1 节。本文里凡是和"学生录音""不用语音合成生成录音"冲突的说法，以 ADR 0006 为准；数据来源、测试音频规矩、测评说明已按它改过。

本文件是给开发者（Claude Code 或老师）看的详细规格。项目背景、决策理由和课程安排见 `docs/handoff_slim.md`；最重要的规则摘要在仓库根目录的 `CLAUDE.md`。

> 第 6 节里的模型、版本和代码，2026-10-07 在一台 Linux 云端容器（2 核 CPU、Python 3.13、sherpa-onnx 1.13.8）上实测过。其他版本号和平台说明是当天查到的，用之前请再核实一遍。

---

## 1. 要做出来的东西

一个用浏览器访问的本地网页工具（Gradio），输入一段游客提交的投诉录音或视频，输出一份供投诉处理人员核查的"核查初稿"：

1. 带时间戳、区分说话人的文字稿；
2. 疑似问题片段：按中性类别标为"疑似·购物安排""疑似·费用"等，标出在第几分第几秒，可以点开听原声；
3. 提取出来的金额、时间、电话、证号、合同号、旅行社名、店名；
4. 人工复核栏（确认、修改、驳回、意见），复核结果随初稿一起导出；
5. 导出 Word 初稿、表格（CSV/JSON）和疑似片段音频（ZIP）。

它同时是课程的"教师模板"：八个处理步骤各是一个独立的小模块，学生分 8 组，每组只替换或改进自己那一步，模块之间用统一中间格式传数据。所以代码要**简单、好读、每一步可单独运行和测评**，不要做成学生看不懂的大框架。

## 2. 八步流程与模块

| 步骤 | 模块文件 | 本仓库要做到的基线 | 学生改进（对应组） | 测评指标 |
|---|---|---|---|---|
| 1 上传与格式统一 | `pipeline/step1_ingest.py` | ffmpeg 转 16kHz 单声道 16 位 WAV；原始文件只读保存；SHA-256；清单与质检报告（只报告不修改） | 教师模板，学生不改 | — |
| 2 降噪与端点检测 | `pipeline/step2_vad.py` | Silero VAD 切段；降噪函数留接口，默认不处理，可选 noisereduce | 第 1 组降噪与切段；第 2 组远距离、口袋录音 | 加噪前后、降噪前后的 CER；漏切人声；Q/N/F 三种条件的 CER 差距 |
| 3 语音识别与热词 | `pipeline/step3_asr.py` | SenseVoice 识别每段；热词纠错基线（拼音相似度后处理，默认关闭） | 第 5 组热词 | CER；开关热词前后的专名正确率 |
| 4 说话人分离 | `pipeline/step4_diarize.py` | pyannote 分割 + CAM++ 声纹 + 聚类；按重叠时长给每段配说话人；界面里把"说话人1"对应成导游、游客等 | 第 3 组 | 说话人标错的时长比例 |
| 5 数字、证号、金额规范化 | `pipeline/step5_normalize.py` | 格式规范 + 提取（正则）；汉字数字转换基线（cn2an + 少量规则） | 第 4 组 | 数字、证号、金额的正确率 |
| 6 话术分类 | `pipeline/step6_classify.py` | 关键词规则基线（可解释）；能加载 TensorFlow 模型，没有模型时退回规则 | 第 6 组模型 A、第 7 组模型 B | 各类别准确率、召回率；正常讲解被误标的比例 |
| 7 时间戳与疑似片段 | `pipeline/step7_clips.py` | 按起止时间截取疑似片段（前后各留 1 秒，可调），打包 ZIP，附索引表 | 第 8 组（兼测误报率） | 片段起止误差；误报率 |
| 8 核查初稿与人工复核 | `pipeline/step8_report.py` + 界面 | Word 初稿、CSV/JSON；复核栏；声明 | 投诉处理人员（人工，必须） | — |

`pipeline/__init__.py` 里写 `run_pipeline(path, options, progress=None)`，按顺序调用八步，返回 `(segments, meta)`。每一步都能单独调用，输入输出都是下面的统一中间格式。

**原则**：每一步先做最简单、能跑通的基线，在模块开头的说明里写清"基线做法 / 可以改进的方向 / 用什么指标测"。不要替学生把专项改进做完（深度降噪、解码器热词、模型 A/B 对比等），这些留给各组；但整条流程必须端到端可用，演示时结果要像样。

## 3. 统一中间格式

每段文字一条记录。课程规定的五个字段（导出表格时用中文表头，顺序固定）：

| 中文表头 | 代码里的键 | 类型 | 说明 |
|---|---|---|---|
| 开始时间（秒） | `start` | float，保留 2 位小数 | 该段在录音中的起点 |
| 结束时间（秒） | `end` | float | 该段在录音中的终点 |
| 说话人 | `speaker` | str | 导游、游客、未知（可扩展店员、司机等）；映射之前是"说话人1""说话人2" |
| 文字内容 | `text` | str | 识别并规范化后的文字（带标点、数字已规范） |
| 标签 | `label` | str | "疑似·购物安排"等 5 类之一；正常讲解和其他留空字符串 |

可选字段（可以加，但不能改上面五个字段的含义）：`text_raw`（汉字读法、无标点的识别结果，算 CER 用）、`speaker_id`（映射前的编号）、`category`（7 类原始预测）、`score`（置信度）、`numbers`（规范写法列表，格式同剧本的 numbers 字段，如 `["2800元", "15:40", "0871-0000-6688", "YN-0000-3721"]`）、`entities`（旅行社、店名等）、`corrections`（热词纠错记录）、`review`（未复核/确认/修改/驳回）、`review_note`、`clip`（片段文件名）、`source`（`asr` 或 `script_demo`）。

JSON 导出格式：

```json
{
  "meta": {"file": "G1-S1-Q.wav", "duration": 268.4, "sha256": "…", "processed_at": "2026-10-07T15:20:00+08:00",
           "models": {"asr": "sense-voice-int8-2024-07-17", "vad": "silero_vad", "diarization": "pyannote-3.0 + campplus"},
           "options": {"num_speakers": 2, "denoise": false, "hotword_fix": false},
           "notice": "识别可能有误；所有标注均为疑似、待核查，必须人工复核"},
  "segments": [
    {"start": 12.4, "end": 18.9, "speaker": "导游", "text": "这个手镯今天优惠价2800元。", "label": "疑似·费用",
     "numbers": ["2800元"], "review": "未复核"}
  ]
}
```

CSV 导出用 UTF-8 带 BOM（`utf-8-sig`），Excel 直接双击打开不乱码。界面里的表格可以用简短表头（开始、结束、说话人、文字、标签……），导出文件一律用上表的中文表头和顺序，可选字段排在后面。`data/lines.csv` 里的 `numbers` 是用"；"分隔的字符串，代码里拆成列表（同一行里重复的值去重后再比对）。

## 4. 数据（全部虚构）

见 `data/README.md`。要点：

- `data/scripts/group_1.json`—`group_8.json` 是 24 个剧本的源数据，`data/lines.csv` 是同样内容的扁平表（1055 行），多了有效字数和标注备注。
- 参考文本（算 CER 用）= 一个剧本所有台词的 `text` 按顺序拼起来；不含说话人、不含 `direction`（动作提示，不念）。AI 录音的参考文本第一行另加开头那句声明（见 4.1）。
- 台词里的数字都是汉字读法（"两千八百块"），`numbers` 字段是规范写法（"2800元"）。
- `data/hotwords.txt` 只含正确名称；第 5 组剧本里故意让人物说错或简称的变体在 `data/hotword_variants.csv`，用来检验热词会不会"过度纠正"——参考文本以人物实际说的为准。
- 7 个类别的定义和是否标"疑似"见 `data/labels.json`。类别分布很不均：威胁消费只有 41 行，其中 35 行在第 6 组；第 4、7 组没有"购物安排"；第 8 组（正常讲解对照组）也有 39 行标为费用、行程变更或购物安排。测评设计要考虑这些（见 5.6）。
- `numbers` 里有不少是数量和时长（如"1盒""40分钟"），测评时按类型分开统计（见 5.5）。
- 测评录音：72 段，2026-10-14 起由 AI 按剧本合成（见 4.1），学生不录音。

**测试用音频的规矩**（与 CLAUDE.md 红线一致）：自动测试和便携包自检只用模型自带的公开测试音频（可以拼接成长音频测速度和内存）和 ffmpeg 生成的纯音、静音；这些测试音频不放进演示界面。自动测试不用语音合成的"人说话"音频（合成模块的测试用假的合成函数）。测评录音由 AI 按虚构剧本合成，发布在 GitHub Releases，不进 git；课堂演示、答辩、核查初稿样例都用 AI 录音（开头的声明保留）。万一有真人录音（例如老师自己试的），只在本地处理，不上传到云端开发环境或公网演示，也不进仓库。

### 4.1 AI 合成测评录音（2026-10-14 起，见 ADR 0006）

学生不再按剧本录音。72 段测评录音（24 个剧本 × Q 安静、N 嘈杂教室、F 口袋或远距离）全部由 AI 按剧本离线合成，为什么这样定见 `docs/adr/0006-ai-synthesized-recordings.md`。

- **工具**：`tools/synthesize_recordings.py`（合成规则在 `pipeline/tts_dataset.py`，只用 numpy、scipy，合成函数由工具加载模型后传进去，测试时传假的合成函数）。老师一般不用自己跑：GitHub Actions 的"合成 AI 测评录音"工作流（`.github/workflows/build-ai-recordings.yml`）生成并发布。自己跑时：
  - `python models/download_models.py --optional kokoro_tts` 先下载语音合成模型；
  - `python tools/synthesize_recordings.py` 合成全部 72 段，放在项目里的 `ai_recordings/`（不进 git）；`--scripts G1-S1 --conditions Q` 只合成一段；
  - `--digits` 生成选做数字实验的录音；`--make-voice-plan` 重新给每个角色挑声音（改写 `data/tts_voice_plan.csv`）。
- **模型**：sherpa-onnx + Kokoro v1.1 中文多说话人模型（`kokoro-multi-lang-v1_1`，Apache-2.0 许可），自带 100 个中文合成声音：3—57 号女声、58—102 号男声。全程离线，不联网、不调用云服务。选模型时用 SenseVoice 回听比较过 vits-aishell3，它念错得多，不用。
- **不做声音克隆**：只用模型自带的合成声音，不用任何真人（学生、老师、网上的人）的声音做参考。网页工具本身不做声音合成。
- **声音分配表** `data/tts_voice_plan.csv`（Excel 能打开；列：script_id、role、sid、gender、speed、note；`*` 那一行是开头声明用的声音）：每个剧本的角色用不同声音，改 `sid`、`speed` 后重新合成。按剧本挑（`pipeline/tts_dataset.py` 的 `VOICE_RULES`）：第 3 组三个剧本里"声线相近"的游客甲、乙故意选相近的声音；从称呼看得出的性别（大姐、大哥、阿姨、老公、闺女）、孩子（G7-S3 游客乙，音调最高、稍快）、老人（G7-S2 游客丙，稍慢）都按剧本挑；开头的声明用一个单独的声音，不给任何角色用。
- **舞台说明做成声音效果**：台词的 `direction` 字段（不念）变成与上一句部分重叠、打断、停顿几秒、车载扩音器（带一点失真和回声）、电话外放（电话音质）、离手机远、靠近手机小声说、提高音量、隧道回声、提示音、扩音器尖啸、发动机声等。车上的场景一直有发动机声。
- **三种录音条件**，都用同一份合成台词渲染（每个条件的停顿、重叠时间不同）：
  - Q：安静房间，一点混响和底噪；
  - N：叠加别的剧本的合成人声当嘈杂声（模拟教室，信噪比约 8—12 dB），偶尔有桌椅碰撞声；
  - F：组号 + 剧本号是单数的剧本模拟"手机在口袋里"（2800 Hz 以上被滤掉、声音发闷、偶尔有衣服摩擦声），双数的模拟"手机在 2—3 米外"（音量小、混响大）。
- **标明是 AI 合成**：每段录音开头念声明"以下为人工智能合成的虚构教学录音，不是真实录音。"（`pipeline.data.AI_DISCLAIMER`）；文件名带 `-TTS`（如 `G1-S1-Q-TTS.m4a`）；文件信息（标题、注释、作者）写明 AI 合成。不得剪掉声明，不得当作真实录音使用或传播。网页整理 AI 录音时，导出的初稿、说明.txt、JSON 也写明录音来源，疑似片段文件名带 `-TTS`（见 5.7、5.8）。
- **输出格式**：每段约 4—6 分钟（72 段共约 6 小时），24000 Hz 单声道 AAC（m4a）；入池时照常转成 16kHz 单声道 16 位 WAV。
- **入池**：`tools/ingest_pool.py` 认 `-TTS` 后缀，`G1-S1-Q-TTS.m4a` 入池后编号仍是 `G1-S1-Q`（`normalized/G1-S1-Q.wav`、`references/G1-S1-Q.txt`、标注都用这个编号）；同一段录音既有带 `-TTS` 又有不带的文件会报错，不猜用哪个。
- **参考文本**：`config.yaml` 里 `recordings.ai_synthesized: true`，所以 `tools/export_references.py` 生成的初稿第一行是那句声明，后面是剧本台词；"数据校对"页没有参考文本文件时也一样先照抄（含声明）。学生按"实际说了什么"校对：合成念成别的字、漏念、多念的，按录音改；多音字念错但字没变的，保留剧本的字（都记进 AI 录音问题清单；其他约定见 ADR 0006 "校对的几条约定"）。
- **说话人标注**：开头的声明标说话人"声明"（`pipeline.data.AI_DISCLAIMER_SPEAKER`），`tools/labels_to_csv.py` 不会把它当成写错的角色名。`tools/evaluate.py speakers --speakers ref` 按标注里的人数设人数，声明算一个人（它用单独的合成声音）。
- **标准答案**：合成时同时得到，放在 `answer_key/`：`references/`（参考文本，第一行是声明）、`annotations/speakers/`（start,end,speaker）、`annotations/clips/`（start,end,label，连续几句同一个疑似类别的台词连成一个片段）、`audacity_labels/`（`G1-S1-Q.speakers.txt`、`G1-S1-Q.clips.txt`，Audacity "文件 → 导入 → 标签"能直接导入）、`timeline/`（每句台词：序号、说话人、开始、结束、类别、声音编号、舞台说明、台词）、`summary.csv`（每段：时长、计划时长、F 条件是口袋还是 2—3 米外、句数、说话人数（含声明）、片段数、声音编号）、`voice_plan.csv`。标准答案的参考文本是照剧本写的，不知道合成念错的地方；**数据池里用的是学生校对、标注的结果**，标准答案只在学生交了以后（第 6 周）拿来对照。
- **发布**：GitHub Releases 标签 `ai-recordings-v1`（预发布，不顶替便携包；便携包仍在原来 `v` 开头的 Release 里），三个压缩包：`tiandu-ai-recordings-v1.zip`（`ai_recordings/README.txt` + `ai_recordings/recordings/` 下 72 段）、`tiandu-ai-answer-key-v1.zip`（`ai_recordings/answer_key/`）、`tiandu-ai-digits-v1.zip`（`ai_recordings/digits/<数字>/tts<声音编号>_<第几遍>.wav`，100 个合成声音 × 零到九 × 3 遍，16000 Hz 单声道 16 位；`python tools/tf_lab/train_digits.py --data 解压后的digits文件夹` 直接训练，文件名"_"前面算一个人）。录音、标准答案都不进 git。
- **测评提醒**（报告里必须写）：合成语音比真人干净、发音标准，嘈杂和远距离是模拟出来的，字错率、说话人分离结果会偏乐观，只能当作方法练习，不代表工具在真实录音上的效果；报告写明"测评数据是 AI 合成的"。

## 5. 各步骤要点

### 5.1 步骤 1：上传与格式统一（教师模板）

- 接收 wav、mp3、m4a、aac、flac、ogg 和视频 mp4、mov、mkv（只取音轨）。提示用户不要用微信语音消息格式。界面上传用 `gr.File`（`gr.Audio` 不收视频，而且可能先转换格式，转换后算出的 SHA-256 就不是原始文件的了）。
- ffmpeg 不一定预装：云端环境用 `apt-get install -y ffmpeg`；Windows 离线包要带 ffmpeg 静态版（也可以用 pip 包 `imageio-ffmpeg` 提供的 ffmpeg 程序）。启动时检查 ffmpeg 是否可用，不可用就给出明确提示。
- 转换：`ffmpeg -y -i 输入 -vn -ac 1 -ar 16000 -sample_fmt s16 输出.wav`。原始文件原样保存并设为只读，转换结果另存。
- 对原始文件算 SHA-256，写进清单和初稿（证明处理的是哪一份文件、文件没被改过）。
- 质检只报告、不修改：时长与剧本预计时长差太多、音量太小、削波、大段无声、原始采样率低于 16kHz。**这一步不要降噪、不要调音量**，否则分不清第 1、2 组的改进是谁带来的。阈值做成配置项，先给保守的默认值。
- 批量模式（给课程数据池用）：`tools/ingest_pool.py` 处理 `data_pool/raw/` 里的文件，检查文件名是否符合"剧本编号-条件代码.wav"（如 `G1-S1-Q.wav`，条件代码 Q 安静、N 嘈杂教室、F 口袋或远距离；AI 录音多一个 `-TTS` 后缀，如 `G1-S1-Q-TTS.m4a`，编号仍是 `G1-S1-Q`，见 4.1），不符合就报错提示改名，不要自动猜。输出 `data_pool/normalized/`、`data_pool/manifest.csv`（文件编号、剧本编号、录音条件、原始文件、转换后文件、时长、原始采样率、质检结果、上传时间、SHA-256）、`data_pool/qc_report.csv`。这些目录对应交接文档里的"数据池/原始、标准化、参考文本、清单.csv、质检报告.csv"。
- `tools/export_references.py`：从 `data/scripts/*.json` 导出 `data_pool/references/G1-S1.txt` 等 24 个参考文本。（设计文档第 5.10 节改成按录音导出 72 份，如 `G1-S1-Q.txt`；`recordings.ai_synthesized: true` 时第一行是 AI 合成声明，见 4.1。）

### 5.2 步骤 2：端点检测与降噪

- 基线用 Silero VAD（第 6 节代码）。参数做成配置项：threshold 0.5、min_silence_duration 0.3、min_speech_duration 0.25、max_speech_duration 15。
- 长录音（30—60 分钟）：模型只加载一次（模块级缓存）；音频按块喂给 VAD；识别分批解码（例如每批 20 段），每批更新一次进度条；说话人分离需要整段音频在内存里（1 小时 16kHz 单声道 float32 约 230 MB，可以接受）。速度和内存要用拼接出来的 60 分钟测试音频实测（见第 9 节），实测之前不要对外说处理时长。
- `denoise(wav_path) -> wav_path`：默认原样返回；选项开启时用 noisereduce。降噪是否真有帮助要用 CER 验证，不要默认开启。

### 5.3 步骤 3：语音识别与热词

- 基线模型：SenseVoice（sherpa-onnx，int8，**2024-07-17 版**，见第 6 节的坑）。
- 两种识别模式，做成选项：
  - 显示模式（界面默认）：`use_itn=True`，输出带标点、数字已转阿拉伯数字的文字，写入 `text`；
  - 测评模式：`use_itn=False`，输出汉字读法、无标点，写入 `text_raw`，与剧本参考文本口径一致，用来算 CER。
  - 需要两种都要时就各识别一遍（识别时间翻倍）。算 CER 优先用测评模式；如果只能拿到显示模式的结果，参考文本和识别结果必须先用同一个数字归一函数处理再比，并在报告里注明。
- 热词：sherpa-onnx 里 SenseVoice 和 Paraformer 都不支持解码时热词；支持的是 transducer 模型（`hotwords_file`），以及 1.13.8 里新增的 `from_funasr_nano`、`from_qwen3_asr` 接口（有 `hotwords` 参数，模型较大，未实测）。本仓库的基线是**识别后的拼音相似度纠错**（pypinyin）：拿 `data/hotwords.txt` 里 3 个字以上的名称，在识别结果里找拼音相近、字不同的片段替换，把每次替换记在 `corrections` 里，界面可以开关，默认关闭。纠错要同时作用于 `text` 和 `text_raw`（第 5 组用 `text_raw` 算专名正确率，只改 `text` 的话测不出效果）。第 5 组可对比的真正热词方案（需要他们自己实测）：
  1. sherpa-onnx transducer 模型 + `hotwords_file`（文档示例用 `sherpa-onnx-conformer-zh-stateless2-2023-05-23`，`modeling_unit="cjkchar"`，`decoding_method="modified_beam_search"`）；
  2. FunASR 的 SeACo-Paraformer 热词（pip 安装 funasr 需要 PyTorch；`funasr-onnx` 包里也有 `SeacoParaformer` 类，但要另外准备导出好的 ONNX 模型）；
  3. sherpa-onnx 的同音字替换（homophone replacer，参数 `hr_lexicon`、`hr_rule_fsts`，规则要用 pynini 编译）。
- 识别时间：57 秒测试音频上实测 SenseVoice int8 在 2 核容器上实时率约 0.13（1 分钟录音约 8 秒）；长录音和办公电脑上要另测。

### 5.4 步骤 4：说话人分离

- 基线：sherpa-onnx 离线说话人分离（第 6 节代码）。界面让用户选说话人数（自动、2、3、4、5、6），知道人数时一定要填（AI 录音开头的声明也算一个人：第 3、8 组剧本 5 个角色 + 声明，共 6 个声音，选 6）：实测 4 人的测试音频在"自动"下被分成了 5 类，填 4 后正确。
- 按重叠时长给每个识别段配说话人。已知问题：一段 VAD 切出来的话里如果换了人（中间没停顿），整段只会算给一个人（实测出现过）。按说话人边界再切分是第 3 组的改进方向。
- 映射：分离结果只是"说话人1、说话人2…"，由界面上的人把它们对应成导游、游客、店员、司机、未知。可以提示"说话时间最长的可能是导游"，但不要自动定。

### 5.5 步骤 5：数字、证号、金额规范化与提取

- 显示模式下（SenseVoice 已做 ITN）：只做格式规范和提取，用正则把金额统一成"2800元"、电话"0871-0000-6688"、导游证号"YN-0000-3721"、合同号"HT-20260000-118"、订单号"DD-0000-5566"、时间"15:40"，结果放进 `numbers`；旅行社名、店名按 `data/hotwords.txt` 和 `data/fictional_names.csv` 匹配，放进 `entities`。
- 测评模式或剧本文本演示（输入是汉字读法）：先做汉字数字转换。基线可以用 cn2an（`cn2an.transform(text, "cn2an")`），但实测有这些问题，要用规则先保护或后修正，这正是第 4 组的题目：
  - "一下""一般""一点""一个是…"被转成"1下""1般""1点""1个是"；
  - "两个半小时"变成"2个0.5小时"；
  - "三四十块""九点五十"转换失败（会报警告，原样保留）；
  - 电话、证号是逐位读的（"零八七一 零零零零 六六八八""YN 零零零零 三七二一"），要单独按逐位规则处理。
- sherpa-onnx 自带的 `itn_zh_number.fst`（通过 `rule_fsts` 参数）也能用，但实测会把"一个"转成"1个"，不建议当默认。
- 测评：拿剧本每行的 `numbers` 字段当标准答案，比对提取结果（同一行内去重后按集合算正确率和召回率）。`numbers` 里约四成是数量和时长（"1盒""40分钟"），所以按类型分开报告：
  - 主指标（与纠纷核查直接相关）：金额（以"元"结尾或含"元/"）、电话（如 0871-0000-6688）、证号（如 YN-0000-3721）、合同号（HT-…）、订单号（DD-…）、时刻（如 15:40）、日期；
  - 次要指标：数量、时长、其他。
  类型用正则判断，判断规则写在代码里并在报告中说明。

### 5.6 步骤 6：话术分类

- 7 个类别：购物安排、费用、行程变更、服务态度、威胁消费、正常讲解、其他（定义见 `data/labels.json`、`docs/script_spec.md`）。前 5 类输出"疑似·类别"，正常讲解和其他标签留空。
- 基线：关键词规则。词表写在一个单独的数据文件里（例如 `pipeline/rules_keywords.json`），短、可解释，每个词都能说清为什么属于这一类。不要把剧本原句直接抄成关键词（会让测评结果虚高）。
- 输入口径要统一：训练和预测都先经过同一个 `normalize_for_classification(text)`——汉字数字先用步骤 5 的转换函数转成阿拉伯数字，再把所有数字串替换成同一个占位符（如 `#`），并去掉标点。否则模型训练时见的是"两千八百块"，界面上收到的却是"2800元"。
- TensorFlow：写 `tools/train_classifier.py`，最小可用的字级模型（Embedding + Conv1D + 全局最大池化 + Dense），用 `data/lines.csv` 训练，保存到 `models/classifier/`。`step6_classify.py` 发现有训练好的模型就加载，否则用规则。TensorFlow 是可选依赖（单独的 `requirements-tf.txt`），没装也要能跑。
- 测评要如实：
  - 主结果用"按组留一"交叉验证（用 7 个组的台词训练、测剩下 1 个组，轮 8 次），对应"用其他同学写的句子测试"。因为各组类别分布差别很大（见第 4 节），不要报每一折的指标，把 8 折的预测**汇总成一个混淆矩阵**再算各类准确率和召回率；
  - 同时报一次随机划分的结果，让学生看到随机划分会虚高；
  - 必报：各类别准确率和召回率、混淆矩阵、误报率。误报率 = 标准答案为"正常讲解"的句子中被标成任一疑似类别的比例（全部 24 个剧本算一次，第 8 组单独再算一次；第 8 组里也有标为费用、行程变更、购物安排的句子，它们不算误报）；
  - 在说明里写明：剧本数据上的结果不代表真实录音上的效果，换个说法就可能失效。

### 5.7 步骤 7：疑似片段导出

- 对每个有标签的段，按 `start`、`end` 从标准化后的 WAV 截取（前后各留 1 秒，不超出文件边界），文件名用 ASCII，例如 `clip_003_000012.4-000018.9.wav`；原录音是 AI 合成的测评录音时，片段文件名末尾也加 `-TTS`，如 `clip_003_000012.4-000018.9-TTS.wav`（片段里听不到开头的声明）。另附 `clips_index.csv`（序号、起止、说话人、文字、标签）。全部打包成 ZIP。
- 截取可以直接用 soundfile/numpy 按采样点切，保证起止精确。

### 5.8 步骤 8：核查初稿（Word）

用 python-docx 生成，内容依次为：

1. 标题："旅游纠纷录音材料核查初稿（疑似、待核查）"；
2. 醒目的声明："本初稿由语音识别等人工智能技术自动生成，识别和分类都可能出错。所有标注均为'疑似、待核查'，不代表任何定性结论，必须由工作人员对照原始录音逐条复核。本工具不鉴定录音的真伪。"；
3. 文件信息：原始文件名、时长、SHA-256、处理时间、模型名称和版本、处理参数；原录音是 AI 合成的测评录音时，另加"录音来源"一行（见本节最后）；
4. 摘要：各类疑似片段的数量、复核状态（已复核 N 条 / 共 M 条）；
5. 疑似片段表：序号、起止时间（分:秒）、说话人、文字、标签、提取的数字和名称、复核结论（未复核/确认/修改/驳回）、复核意见；
6. 全文时间轴转写；
7. 复核人、复核日期签字栏。

文档属性（隐式标识）：标题、作者写"旅游纠纷录音材料整理工具（教学原型）"，备注写"由人工智能技术自动生成"。文中不出现"违规""违法""执法级准确率""可作为法律证据"等说法。

导出时把 Word 初稿、CSV、JSON 和疑似片段打成一个 ZIP，ZIP 里附一个"说明.txt"，写明"由人工智能技术自动生成、识别可能有误、所有标注均为疑似、必须人工复核"；JSON 的 `meta.notice` 也写这句话。

**AI 合成的测评录音**（见 4.1）：文件名带 `-TTS` 的录音，以及 `config.yaml` 里 `recordings.ai_synthesized: true` 时文件名是录音编号的录音（如从数据池 `normalized` 里复制出来的 `G1-S1-Q.wav`），整理时都在元信息里写上录音来源（`pipeline.schema.AI_SOURCE_NOTE`）：Word 文件信息加"录音来源"一行，说明.txt 和 JSON 的 `meta.source_note` 写明"原录音是 AI 按虚构剧本合成的教学录音，不是真实录音；疑似片段音频也是合成的声音"，疑似片段文件名带 `-TTS`（见 5.7）。文件名改成别的名字就认不出是 AI 录音了，所以上传时最好直接用 AI 录音包里的 `G?-S?-?-TTS.m4a`；从数据池复制 WAV 时不要改名。

## 6. 已实测的技术选型（2026-10-07）

### 6.1 包版本（当天 PyPI 最新）

gradio 6.29.1、sherpa-onnx 1.13.8、tensorflow-cpu 2.21.0、python-docx 1.2.0、jiwer 4.0.0、noisereduce 3.0.3、pypinyin 0.55.0、cn2an 0.5.24、soundfile 0.14.0、openpyxl 3.1.5、funasr 1.4.16（需 PyTorch，可选）、funasr-onnx 0.4.3（可选）。建议 Python 3.11 或 3.12（以后要做 Windows 离线包，3.11 最稳）。`requirements.txt` 里写兼容范围，不要把版本锁得太死（例如 `gradio>=6,<7`：代码用了 Gradio 6 才有的写法，下限要写 6）；Gradio 6 改过一些用法（例如主题、CSS 不再建议传给 `Blocks()`），参数名以安装的版本为准，用之前先查文档或 `help()`，不要凭记忆写。托管平台支持的 Gradio 版本可能比最新版旧，部署前先确认。

### 6.2 模型（都从 sherpa-onnx 的 GitHub Releases 下载，链接当天都能打开）

| 用途 | 文件 | 大小 | 下载地址 |
|---|---|---|---|
| 识别（必需） | `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2` | 约 163 MB | https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2 |
| 端点检测（必需） | `silero_vad.onnx` | 0.6 MB | https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx |
| 说话人分割（必需） | `sherpa-onnx-pyannote-segmentation-3-0.tar.bz2` | 7 MB | https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 |
| 声纹（必需） | `3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx` | 28 MB | https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx （地址里的 recongition 拼写就是这样） |
| 冒烟测试音频 | `0-four-speakers-zh.wav`（4 人，57 秒） | 1.8 MB | https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav |
| 可选：声纹备选 | `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx` | 40 MB | 同一发布页 speaker-recongition-models |
| 可选：数字规则 | `itn_zh_number.fst` | 26 KB | https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst |
| 可选：对照识别模型 | `sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2` | 78 MB | asr-models 发布页（不带标点，要配标点模型） |
| 可选：标点模型 | `sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2` | 279 MB | punctuation-models 发布页 |
| 可选：热词实验（第 5 组） | `sherpa-onnx-conformer-zh-stateless2-2023-05-23.tar.bz2` | 456 MB | asr-models 发布页（transducer，支持热词） |

2026-10-14 加了可选模型 `kokoro_tts`（Kokoro v1.1 语音合成，只用来生成 AI 测评录音，`python models/download_models.py --optional kokoro_tts`，见 4.1）；网页工具和便携包不用它。

`models/download_models.py`：下载必需模型到 `models/`，已存在就跳过，解压后检查关键文件存在；可选模型用参数开启。`models/` 不进 git。下载失败时打印清楚的提示（国内网络访问 GitHub 可能慢，可以先在别处下载好再拷进 `models/`）。

### 6.3 已实测的最小代码

下面这段代码当天实测可运行（VAD → 识别 → 说话人分离 → 配说话人）。正式代码按模块拆开，但调用方式照这个来：

```python
import numpy as np
import sherpa_onnx
import soundfile as sf

SR = 16000
M = "models"
SV = f"{M}/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"


def load_wav(path):
    samples, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == SR and samples.ndim == 1, "先用 ffmpeg 转成 16kHz 单声道"
    return samples


def vad_segments(samples):
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = f"{M}/silero_vad.onnx"
    cfg.silero_vad.threshold = 0.5
    cfg.silero_vad.min_silence_duration = 0.3
    cfg.silero_vad.min_speech_duration = 0.25
    cfg.silero_vad.max_speech_duration = 15
    cfg.sample_rate = SR
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
    win = cfg.silero_vad.window_size
    out = []

    def drain():
        while not vad.empty():
            seg = vad.front
            out.append((seg.start / SR, np.array(seg.samples, dtype=np.float32)))
            vad.pop()

    for i in range(0, len(samples), win):
        vad.accept_waveform(samples[i:i + win])
        drain()
    vad.flush()
    drain()
    return out


def recognize(segs, use_itn=True):
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=f"{SV}/model.int8.onnx", tokens=f"{SV}/tokens.txt",
        num_threads=4, language="zh", use_itn=use_itn)
    streams = []
    for _, s in segs:
        st = rec.create_stream()
        st.accept_waveform(SR, s)
        streams.append(st)
    rec.decode_streams(streams)
    return [{"start": round(t0, 2), "end": round(t0 + len(s) / SR, 2), "text": st.result.text}
            for (t0, s), st in zip(segs, streams)]


def diarize(samples, num_speakers=-1):
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=f"{M}/sherpa-onnx-pyannote-segmentation-3-0/model.onnx")),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=f"{M}/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers, threshold=0.5),
        min_duration_on=0.3, min_duration_off=0.5)
    assert cfg.validate()
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    return [(t.start, t.end, t.speaker) for t in sd.process(samples).sort_by_start_time()]


def attach_speakers(rows, turns):
    for r in rows:
        best, ov = None, 0.0
        for s, e, spk in turns:
            o = min(r["end"], e) - max(r["start"], s)
            if o > ov:
                best, ov = spk, o
        r["speaker"] = f"说话人{best + 1}" if best is not None else "未知"
    return rows
```

实测结果（57 秒 4 人测试音频，2 核容器）：识别 7.5 秒（实时率 0.13），说话人分离 5.9 秒（实时率 0.10），加载模型约 1.2 秒；`use_itn=True` 输出带标点和阿拉伯数字（模型自带测试音频 zh.wav 输出"……早上9点至下午5点。"），`use_itn=False` 输出汉字读法、无标点（"……早上九点至下午五点"）。按这个速度推算，1 小时录音在 2 核容器上大约要 15 分钟，但这是从 57 秒的测试音频推算的，要用拼接的 60 分钟测试音频实测后才能对外说；办公电脑另测。

这段代码为了好读，每次调用都重新加载模型、一次解码全部段落；正式代码里模型只加载一次，识别分批解码并更新进度。

### 6.4 已踩过的坑

1. **不要用 `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09`**：名字像新版，实际是粤语微调模型（来自 WSYue-ASR），普通话也按粤语输出、没有标点、没有 ITN。用 2024-07-17 版。
2. sherpa-onnx 里 Paraformer、SenseVoice 不支持解码时热词（见 5.3）。
3. `itn_zh_number.fst` 会把"一个"转成"1个"；cn2an 的问题见 5.5。
4. 说话人数设"自动"时容易多分；同一段里换人只会标成一个人。
5. `data/` 下的 CSV 是 UTF-8 带 BOM，Python 读要用 `encoding="utf-8-sig"`。
6. 剧本数字是汉字读法，SenseVoice 显示模式输出阿拉伯数字，两者直接比 CER 会把数字全算错，所以 CER 用测评模式（`use_itn=False`）的结果（例外情况见 5.3）。
7. 云端环境和机房电脑不一定装了 ffmpeg（见 5.1）。

## 7. 界面（Gradio）

只用 Gradio，界面和后台在同一个 Python 程序里；不做 React、不做数据库、不做账号系统（云端演示时用 Gradio 自带的简单密码）。

- 顶部标题"旅游纠纷录音材料整理（教学原型）"，下面一条醒目提示：**"仅限虚构演示材料，请勿上传真实投诉录音或含个人信息的录音｜识别可能有误，所有标注均为'疑似、待核查'，必须人工复核｜不作为任何定性依据"**。
- 标签页"整理录音"：
  - 上传文件（音频或视频，用 `gr.File`）；选项：说话人数（自动/2/3/4/5/6）、降噪（关/开）、热词纠错（关/开，可编辑热词表）；"开始整理"按钮；分步骤进度条。
  - 结果：摘要（时长、处理用时、各类疑似片段数）；可编辑表格（序号、开始、结束、说话人、文字、标签、数字、复核结论、复核意见），复核结论用下拉或固定取值；说话人映射（说话人1 → 导游等），点"应用"后更新表格。
  - 点表格某一行，播放这一段原声。
  - "导出核查初稿"：用当前表格（含人工修改和复核结果）生成 Word、CSV/JSON 和片段 ZIP，提供下载。
- 标签页"剧本文本演示"：下拉选择 24 个剧本之一，直接用剧本台词当输入（说话人用剧本角色，时间按每分钟 220 字估算），跑步骤 5—8，显示结果并和剧本标注对照（哪些句子标对了、哪些是误报）。页面顶部写明"演示模式：直接使用剧本文字，没有经过语音识别；时间为估算"。这让没有录音时也能演示分类、规范化和初稿导出。
- 标签页"使用说明"：怎么用、处理流程、红线和局限。
- 设置：`server_name="0.0.0.0"`、端口 7860；**永远不设 `share=True`**；环境变量 `GRADIO_ANALYTICS_ENABLED=False`；限制上传大小；同一时间只处理一个任务（队列）；上传文件和输出定时清理；如果设置了环境变量 `DEMO_USERNAME`、`DEMO_PASSWORD` 就启用登录密码（云端演示用），没设就不要求登录（机房和内网用）。

## 8. 测评工具 `tools/evaluate.py`

- CER：只保留汉字、字母、数字（去标点和空格），全角转半角，字母统一大写；用测评模式结果对比参考文本（两边都是汉字读法，不需要再做数字归一）；按文件、按录音条件（Q/N/F）、按组汇总。可用 jiwer 的 `cer`，或自己写编辑距离。
- 测评录音（AI 合成，见 4.1）只在本地（机房、老师电脑）跑测评。在云端开发环境里只用测试音频改名（如复制成 `G1-S1-Q.wav`）检验批量流程能跑通，不报告任何 CER 数字（包括 AI 录音上测的）。
- 专名正确率：参考文本里出现的 `data/hotwords.txt` 名称，在识别结果里有多少被正确识别；同时统计"过度纠正"（参考文本里是说错的变体，结果被改成了正确名称）。
- 数字正确率：剧本文本演示时按行比对 `numbers`；录音按文件比对。
- 说话人标错的时长比例：需要人工标注"谁在什么时间说话"（每组约 10 分钟，CSV：start, end, speaker），按最优对应计算。
- 分类：见 5.6。
- 片段起止误差：需要人工标注的疑似片段起止时间。
- 输出 CSV 和一份简短的 Markdown 报告；报告里写明数据来源（虚构剧本、哪种录音条件）和局限，并写明"测评数据是 AI 合成的"：合成语音偏干净，嘈杂和远距离是模拟的，结果偏乐观，不代表真人录音。

## 9. 测试与验收

`pytest` 测试（必须能在没有 GPU、没有真实录音的环境里跑）：

- 中间格式：读写 JSON/CSV 往返一致，中文表头和顺序正确。
- 步骤 1：用 ffmpeg 生成几秒的正弦波或静音（44.1kHz 立体声 mp3、m4a、mp4），检查转出来是 16kHz 单声道 16 位、原始文件未被修改、SHA-256 正确、质检能报出"大段无声"。**不要用语音合成生成"人说话"的测试音频**（AI 合成录音只用于测评，见 4.1；合成模块的测试用假的合成函数）。
- 步骤 5：用剧本里有 `numbers` 的句子测提取，报告正确率（不要求 100%，已知失败的写成 xfail 并注明原因）。
- 步骤 6：输出只可能是 5 个"疑似·类别"或空字符串。
- 剧本文本演示：G8（正常讲解对照组）跑完整流程，生成 Word 初稿，检查声明文字在、所有标签合法，并打印误报率。
- 冒烟测试（模型已下载时运行，否则跳过；测试音频只用于自动测试，不放进演示界面）：
  - P1：SenseVoice 自带的 `test_wavs/zh.wav` 和 `0-four-speakers-zh.wav` 走完 VAD 和识别，文字非空，`use_itn=True` 时 zh.wav 的结果里有阿拉伯数字；
  - P2：`0-four-speakers-zh.wav` 说话人数设 4 时得到 4 个说话人。
- 长录音测试（较慢，单独运行，不放进默认的 pytest）：把 `0-four-speakers-zh.wav` 重复拼接成约 60 分钟（ffmpeg concat），跑完整流程，记录每一步耗时、实时率和最大内存，写进 `docs/progress.md`。
- 静态检查：代码里没有 `share=True`；`pipeline/` 里没有任何联网调用；没有写死的密钥。

验收清单：

1. `python models/download_models.py` 能下载并校验必需模型，重复运行会跳过已下载的；
2. `pytest` 全部通过（冒烟测试在有模型时也通过）；
3. `python app.py` 启动后，在浏览器上传一段录音能得到结果表格、能点行播放、能导出 Word 初稿、CSV/JSON 和片段 ZIP；
4. "剧本文本演示"能跑 24 个剧本中的任意一个；
5. 处理日志里打印每一步耗时和实时率；
6. 在 `README.md` 里补充安装、下载模型、运行、局域网访问、离线安装、云端演示部署（保留现有的项目简介、内容和注意事项）。

## 10. 部署

### 10.1 课堂和内网（主线）

- 安装：`python -m venv .venv`，`pip install -r requirements.txt`，`python models/download_models.py`，`python app.py`。机房其他电脑在浏览器打开 `http://这台电脑的IP:7860`。
- 离线安装（试用阶段要装到合作单位一台不联网的电脑上）：在联网电脑上按目标系统和 Python 版本 `pip download -r requirements.txt -d wheels/`，连同 `models/`、ffmpeg 静态版和代码一起拷过去，`pip install --no-index --find-links wheels -r requirements.txt`。Windows 离线包要尽早在一台没装过 Python 的普通 Windows 电脑上实测（交接文档第 13.3 节）。这一项可以留到基线完成后单独做。
- 不需要登录，因为只在内网能访问；正式使用的账号、权限、操作记录不在课程范围。

### 10.2 云端演示（只放虚构材料）

- 写 `Dockerfile`：Python 3.11/3.12 slim 镜像，装 ffmpeg 和依赖，构建时下载必需模型，暴露 7860，设 `GRADIO_ANALYTICS_ENABLED=False`，启动 `python app.py`。云端演示必须设置 `DEMO_USERNAME`、`DEMO_PASSWORD`，页面横幅写明只能上传虚构材料。
- 托管平台（需要老师自己的账号，账号和令牌只从环境变量读，不写进代码、不提交到仓库）：
  - 魔搭社区（ModelScope）创空间：国内可以直接访问，支持 Gradio，有免费 CPU 资源，用 git 推送 `app.py`、`requirements.txt`、`README.md`。平台支持的 Gradio 版本和资源规格以平台当前说明为准；模型从 GitHub 下载可能慢，可以把模型文件放进创空间仓库或另建模型仓库；ModelScope 上如果有同样的 sherpa-onnx 模型，也可以作为国内下载源（需核实是同一个文件）。
  - Hugging Face Spaces：国外平台，国内访问不稳定，不推荐给国内用户演示。
  - 国内云服务器 + Docker：绑定域名需要 ICP 备案；可以先用"IP:端口 + 密码"访问。
- 云端演示只证明"能跑"，真实投诉录音永远不能传到公网。

### 10.3 在 Claude Code 云端环境里

- 云端环境可以安装依赖、下载模型、跑测试、启动服务并用 curl 或 Playwright 自测，但不会给外部提供访问地址；要别人能打开，需要 10.2 的托管平台。
- 默认网络是"受信任"白名单（包括 GitHub 和 PyPI）。如果模型下载失败，请用户在环境设置里放宽网络，或者由用户把模型文件放好。

## 11. 目录结构（建议）

仓库根目录应有 `.gitignore`（网页上传时隐藏文件可能被漏掉，P0 先检查）。至少忽略：`models/` 下除下载脚本以外的文件、`data_pool/`、`outputs/`、`tmp/`、`wheels/`、`.gradio/`、各种音视频文件、`.venv/`、`__pycache__/`、`.env`。

```
app.py                      # Gradio 界面，只做界面，调用 pipeline
config.yaml                 # 参数：VAD 阈值、质检阈值、片段前后留白、路径等
pipeline/
  __init__.py               # run_pipeline()
  schema.py                 # 中间格式、中英文字段对照、JSON/CSV 读写
  step1_ingest.py … step8_report.py
  rules_keywords.json       # 步骤 6 规则基线的关键词表
models/download_models.py   # 模型下载（models/ 下其他内容不进 git）
tools/
  ingest_pool.py            # 数据池批量处理
  export_references.py      # 导出参考文本
  evaluate.py               # 测评
  train_classifier.py       # TensorFlow 话术分类训练（可选）
data/                       # 虚构剧本与表格（已提供）
docs/                       # 交接、规格、进度记录
tests/
requirements.txt  requirements-tf.txt  Dockerfile  README.md
```

## 12. 分阶段计划

每个阶段结束时：跑测试、提交、在 `docs/progress.md` 里记一笔（做了什么、实测数字、已知问题），再向用户简短汇报。先把 P0、P1 做通再往下。

| 阶段 | 内容 | 完成标志 |
|---|---|---|
| P0 骨架 | 目录、依赖、配置、模型下载脚本、数据读取、参考文本导出；检查 ffmpeg 和 `.gitignore` | 模型能下载；数据能读；中间格式测试通过 |
| P1 基线流程 | 步骤 1、2、3 + 最简界面（上传 → 表格 → 导出 CSV/JSON） | 用测试音频上传能出带时间戳的文字；P1 冒烟测试通过；记录实时率 |
| P2 说话人 | 步骤 4 + 说话人数选项 + 映射 | P2 冒烟测试通过（填 4 人时分出 4 人） |
| P3 规范化 | 步骤 5（两种模式）+ 名称提取 | 剧本数字提取正确率报告 |
| P4 分类 | 步骤 6 规则基线 + TensorFlow 训练脚本（可选加载） | 按组留一的测评报告，含误报率 |
| P5 初稿 | 步骤 7、8 + 复核栏 + 点行播放 + 剧本文本演示页 | Word 初稿内容符合 5.8；演示页可用 |
| P6 测评 | `tools/evaluate.py` 与数据池批量处理；长录音测试 | 批量流程能跑通（云端只用改名的测试音频，不报 CER 数字）；60 分钟测试的耗时和内存已记录 |
| P7 部署 | Dockerfile、README（内网、离线、云端演示）、安全设置检查 | 验收清单全部通过 |

## 13. 需要老师确认、不要自行决定的事

- 剧本里 3 处待教师过目的地方（交接文档 9.4 节：景点常识、G6-S3 的 5 句标注、虚构名称重名）。
- 是否部署到公网演示、用哪个平台、用谁的账号。
- 是否接入火山引擎等云 API 做对照（只能处理虚构录音，密钥由老师统一管理）。
- 改动八步流程、统一中间格式或 7 个类别。
- 类别名"威胁消费"在界面和初稿上是否换成更中性的显示名（例如"消费施压"）；数据里的标签不变。
