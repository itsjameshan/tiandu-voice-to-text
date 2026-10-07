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

## Task 6–8 · 步骤 2、3、4（并行开发，独立审查通过）

- Task 6 `pipeline/step2_vad.py`：做法 denoise/baseline（不处理）、denoise/noisereduce（谱门限降噪，备选）、enhance/baseline、vad/baseline（Silero VAD）；`detect_speech`。检测器有状态，每次调用新建（约 13 毫秒）。四人测试音频（56.9 秒）切出 7 段，用时约 0.5 秒（实时率约 0.008）。`max_speech_duration` 不是严格上限（设 3 秒时最长段约 8.9 秒）。
- Task 7 `pipeline/step3_asr.py`、`pipeline/hotwords.py`：SenseVoice 显示模式和测评模式（各缓存一个识别器，两个一起加载约 4.8 秒）；按批识别并更新进度。zh.wav 显示模式"开饭时间早上9点至下午5点。"、测评模式"开饭时间早上九点至下午五点"。四人测试音频 7 段识别 4.2 秒（实时率 0.073）。热词基线：模糊拼音（zh/z、ch/c、sh/s、l/n、in/ing、en/eng、an/ang）逐音节比较，默认关闭。
- Task 8 `pipeline/step4_diarize.py`：pyannote 分割 + CAM++ + 聚类；按重叠时长配说话人；说话人映射只按表格里当前显示的名字改（不会悄悄覆盖人工修改）。设 4 人时四人测试音频分出 4 人。
- 集成小修：热词纠错记录按次数去重（text 和 text_raw 的同一处只记一次）；缺少模型文件时报出具体缺哪个；降噪说明更正为"谱门限"。

## Task 9–11 · 步骤 5、6、7、8（并行开发，独立审查通过）

- Task 9 `pipeline/step5_normalize.py`：汉字读法转数字（逐位读的号码、带时间词的时刻、保护词与热词名称、cn2an、块→元），按优先级提取规范写法，按类型分组。剧本 1055 句上（按行集合比较）：主指标 正确率 0.92 / 召回率 0.64（237/372），次要指标 0.72 / 0.84；金额 0.90/0.90，时刻 1.00/0.14（没有时间词的时刻基线不转，留给第 4 组），日期、证号、合同号、订单号 1.00/1.00，电话 1.00/0.50，数量 0.64/0.95，时长 0.85/0.87。已知失败（xfail）：两个半小时、三四十块、九点五十。
- Task 10 `pipeline/text_norm.py`、`pipeline/step6_classify.py`、`pipeline/rules_keywords.json`、`pipeline/tf_classifier.py`：字错率和分类用的文字归一（只用标准库）；关键词规则基线（7 类、可解释的短词，按命中次数和优先级）；TensorFlow 模型可选加载，不可用时退回规则并给出中文提示。剧本 1055 句上：7 类总体正确率 0.56；召回率 购物安排 0.46、费用 0.46、行程变更 0.34、服务态度 0.34、威胁消费 0.51、正常讲解 0.57、其他 0.92；正常讲解误报率 2/155 = 0.01（第 8 组剧本 0/49）。注意关键词是参照剧本写的，这些数字偏乐观，"按组留一"的模型测评才是主结果。
- Task 11 `pipeline/step7_clips.py`、`pipeline/step8_report.py`、`pipeline/table.py`：按采样点截取疑似片段（前后各留 1 秒）和索引表；Word 初稿（标题、红色声明、文件信息与 SHA-256、摘要、疑似片段表、全文时间轴、签字栏；文档属性写明由人工智能技术自动生成；中文字体设好）；导出 ZIP（Word、CSV、JSON、片段、说明.txt）；界面表格与段落互转（非法标签清空并提示）。合成的 60 分钟、1000 段数据：截片段 0.76 秒，导出 1.09 秒。
- 集成小修：小数"一点五元"不再被保护词拦住；带"月""米"等单位的数不当金额；热词表为空时不出错；关键词"贵"改成"太贵/好贵/贵了"，去掉易误标的"提前""投诉"；表格序号异常值不崩溃；规则文件允许带 BOM。
- 全部测试：585 通过，1 跳过（没装 TensorFlow），3 个已知失败（xfail）。

## Task 12 · 串起整条流程、剧本文本演示、各组文件

- `run_pipeline`（`pipeline/__init__.py`）：按 config 和 options 选好各槽位做法，依次执行步骤 1—7；识别用降噪/增强后的声音，说话人分离和截片段用原始录音；分类模型不可用时的原因写进 `meta["warnings"]`；没有人声时返回空列表和提示。
- 实测（本环境 4 核，四人测试音频 56.9 秒，设 4 人）：统一格式 0.23 秒，端点检测 0.41 秒，识别 7.32 秒，说话人分离 6.57 秒，其余不到 0.01 秒；合计 14.5 秒，实时率 0.26。`tools/selfcheck.py` 全部通过。
- `pipeline/script_demo.py`：剧本台词按每分钟 220 字估算时间，跑步骤 5、6 并和剧本标注对照（标对、误报、漏标、误报率）。第 8 组三个剧本误报率都是 0。
- `pipeline/groups/g1`—`g8`：每组一个文件，初始调用基线，开头写明负责什么、改哪里、怎么测、注意什么、改进方向；第 6、7 组有 `build_model`（初始为模板的字级卷积网络，`pipeline/tf_classifier.build_baseline_model`）。`load_all` 改为严格导入。
- `data/classification_extra.csv`：964 句 AI 生成的补充句子（11 个写作子任务 + 7 个对抗式审查子任务 + 合并检查）。

## Task 15、Task 17（非界面部分）· 数据池入池与参考文本、对齐与语谱图

- `pipeline/pool.py`、`tools/ingest_pool.py`、`tools/export_references.py`：文件名检查（小写、"(1)"、下划线、前导零、全角字母、少条件代码、组号超出等都给出具体提示，不猜）；转成 16k WAV（先写临时文件再替换，失败不留半成品）、质检、只读、清单最后写（写进去才算入池），按 SHA-256 跳过已入池文件；复录时替换旧行、删掉旧识别缓存；72 份参考文本只补不覆盖；校对记录追加写（带 BOM，可被 Excel 打开）。
- `pipeline/align.py`：字错率明细（错字、漏字、多字）和"识别结果逐段对齐到参考文本"（给"数据校对"页用）；视频里的例子"雾隐行舟旅行社"→"雾影行走旅行社"算出 2/7。
- `pipeline/features.py`、`tools/show_spectrogram.py`：只用 numpy 算语谱图和 MFCC（预加重、汉明窗、梅尔滤波、对数、DCT），画波形、语谱图、MFCC 三联图（30 分钟录音约 4.4 秒）；默认存到 `outputs/spectrograms/`。
- 测试：数据池 54 个、对齐与特征 24 个，全部通过。

## Task 16、Task 18 · 验收表与冻结、标注转换、测评与对比（cer、hotwords、numbers）

- `pipeline/annotations.py`、`tools/labels_to_csv.py`：读 Audacity 标签（UTF-8、带 BOM、GBK、UTF-16 都能读，跳过频率行），转成说话人表（start,end,speaker）或片段表（start,end,label，"消费施压"存成"威胁消费"）；有错一条也不写，并说明第几个标签什么问题。
- `pipeline/pool.py`（扩展）、`tools/check_pool.py`、`tools/freeze_pool.py`：验收表（已交、命名、质检、入池后的不同校对人数、说话人标注覆盖秒数、片段标注、是否通过）和各组汇总；冻结版本前重算验收表，原子地写进 `versions/<版本>/`，录音只记指纹不复制；版本按自然顺序排序（v10 在 v9 后面）。
- `pipeline/evaluation.py`、`tools/evaluate.py`、`tools/compare.py`：字错率（错字、漏字、多字，按录音条件和组汇总）、专名正确率与过度纠正、数字提取（按类型）；识别结果按"录音 + 上游做法 + 参数"缓存，只换下游做法时不重新识别；`--method` 可写多次，`--hotword on/off`、`--files`、`--no-cache`；报告开头写明数据池版本、做法、参数和局限说明。云端只用改名的测试音频检验流程，没有报告任何字错率数字。
- `reports/g1`—`g8/README.md`：各组报告模板。
- 集成小修：10 分钟达标用未取整的秒数判断；文档补上测热词要加 `--hotword on`、过度纠正的计数规则、复录与验收表说明。

## Task 13、19（指标部分）、20 · 网页界面、测评指标、分类训练脚本

- `app.py`、`pipeline/ui_text.py`：Gradio 6 界面——"整理录音"（上传、说话人数、降噪、热词纠错与可编辑热词表、高级设置里每个槽位的做法下拉框、分步进度、摘要、可编辑表格、说话人映射与各人时长、点行播放原声、导出 Word/CSV/JSON/ZIP）、"剧本文本演示"、"使用说明"。不开公网分享、关闭使用统计、临时文件放项目内 tmp/、启动时清理旧结果、设置了 DEMO_USERNAME/DEMO_PASSWORD 才要求登录。浏览器实测：四人测试音频 56.9 秒处理 12.0 秒（实时率 0.21），导出可下载。
- 集成修正（安全）：导出文件复制到 `tmp/exports/<随机编号>/` 再给浏览器下载，不再用 allowed_paths 对网页开放整个 outputs 文件夹（审查发现可以按路径下载别人的原始录音）；不打印 Gradio 自带的"怎么开公网分享"英文提示；整理失败时不清空说话人时长；高级设置里选了 baseline 以外的降噪做法时以高级设置为准。
- 集成修正（机房 Python）：`pipeline/groups/__init__.py` 不再导入全部组文件（改由 `methods.load_all` 按 `GROUP_MODULES` 导入），第 6、7 组训练时只加载自己的文件；模拟"只有 numpy"的环境验证 `tools/train_classifier.py` 能取到 baseline、g6、g7 的 `build_model`。
- 集成修正：noisereduce 遇到全静音会算出 NaN，导致误判有人声——换成 0（加了测试）。
- `pipeline/metrics.py`：说话人标错比例（10 毫秒网格 + 匈牙利算法最优对应）、混淆矩阵、各类准确率召回率、误报率（只接受类别名）、片段起止误差。
- `tools/check_classification_data.py`、`tools/train_classifier.py`：补充句子检查；字级卷积基线（Embedding 64 → Conv1D 128×3 → 全局最大池化 → Dense 7），按组留一 / 随机划分，报告写到 reports/。实测（TensorFlow 2.21 CPU，15 轮，按组留一，测试集只有人写的 1055 句剧本台词）：
  - 用补充句子：7 类正确率 0.59（620/1055）；误报率 0.17（27/155），第 8 组 0.29（14/49）；威胁消费 准确率 0.28 / 召回率 0.54。
  - 不用补充句子：7 类正确率 0.56（588/1055）；误报率 0.20（31/155），第 8 组 0.33（16/49）；威胁消费 准确率 0.25 / 召回率 0.12。
  - 随机划分（只作对照，会虚高）：正确率 0.67—0.68。
  - 结论：补充句子主要改善了样本少的"威胁消费"召回率；模型的误报率明显高于关键词规则（规则是参照剧本写的，那个数字偏乐观），这正是第 6、7、8 组要研究的。
- 长录音测试（`tools/long_audio_test.py`）：10.4 分钟录音（四人测试音频重复拼接）用时 193 秒，实时率 0.31，最大内存约 1.5 GB（同时有其他任务在跑，偏慢）；识别 105 秒、说话人分离 81 秒。60 分钟测试放在最后机器空闲时做。
- 全部测试：783 通过，8 跳过（没装 TensorFlow 等），3 个已知失败（xfail）。

## Task 14 · Windows 便携包（GitHub Actions，windows-latest）

- 第一次完整构建（提交 9b6f5af）：Python 3.11.9 嵌入版 + `pip install --target` 装依赖 + 下载必需模型 + 用便携 Python 跑 `tools/selfcheck.py`：Python 版本、安装路径、ffmpeg（imageio-ffmpeg 自带的 ffmpeg 7.1）、模型都通过；四人测试音频 56.9 秒完整流程用时 14.0 秒（实时率 0.24），导出核查初稿通过。压缩包 415.5 MB，整个构建约 3 分钟。
- 构建脚本加了第 6b 步：用便携 Python 启动网页工具（`app.py --port 7861`），确认首页返回 200 再打包。

## Task 14 · Windows 便携包（GitHub Actions，windows-latest）

- 第一次完整构建（提交 9b6f5af）：Python 3.11.9 嵌入版 + `pip install --target` 装依赖 + 下载必需模型 + 用便携 Python 跑 `tools/selfcheck.py`：Python 版本、安装路径、ffmpeg（imageio-ffmpeg 自带的 ffmpeg 7.1）、模型都通过；四人测试音频 56.9 秒完整流程用时 14.0 秒（实时率 0.24），导出核查初稿通过。压缩包 415.5 MB，整个构建约 3 分钟。
- 构建脚本加了第 6b 步：用便携 Python 启动网页工具（`app.py --port 7861`），确认首页返回 200 再打包。

- 发布便携包预览版 v0.1.0-pre（2026-10-07）：开发环境不能推送标签，改为在"构建 Windows 便携包"的手动触发里填"发布版本号"，由 GitHub Actions 建标签并发布（运行 37646306941，提交 b1c9289）。
  这次构建在 GitHub 的 Windows 机器上自检全部通过（测试音频 56.9 秒，用时 13.9 秒，实时率 0.24；这是 GitHub 机器的数字，机房要另测），网页工具能正常打开；
  zip 415.6 MB，公开下载地址：https://github.com/itsjameshan/tiandu-voice-to-text/releases/download/v0.1.0-pre/tiandu-portable-win64-v0.1.0-pre.zip（未登录也能下载）。
  这一版还不含"数据校对""录音质检"两页和说话人、分类、片段三项测评（正在做，做完发 v0.2.0-pre）。

- 任务 21 完成（第 3 周选做的数字小实验 `tools/tf_lab/`：split_digits 切数字、mfcc_compare 对照 numpy 和 tf.signal 的 MFCC、train_digits 小卷积网络按说话人留出测试）。
  测试：开发环境（没有 TensorFlow）29 通过、4 跳过；临时 TensorFlow 2.21 环境 29 通过、4 跳过（跳过的是要 sherpa-onnx 的切分测试），两边合起来 33 个测试都跑过。
  合成信号（440 Hz 纯音 + 静音，不是语音）上 numpy 版和 TensorFlow 版 MFCC 的平均相关系数 1.0000。
  已知问题：split_digits 的端点检测参数（threshold 0.5、最短语音 0.1 秒、最短静音 0.4 秒）没在真人数字录音上验证过；用测试音频剪出的数字长度片段只检出 40/50。
  已写进第 0 周清单，请老师课前用一份真实录音试一次。机房 Python 没有 matplotlib 时 mfcc_compare 只出相关系数、不画图。
  审查后的小改动：`pipeline.ascii_name`、`pipeline.features.use_chinese_font` 改成公开函数（工具脚本要用）；TensorFlow 装坏时打印原因；split_digits 读配置失败时用中文提示。

- 任务 17（界面部分）完成：网页新增"数据校对""录音质检"两页（共五页：整理录音、剧本文本演示、数据校对、录音质检、使用说明）。
  数据校对：选池内录音 → 生成对照（测评模式识别，和测评工具共用 `asr_cache/<编号>.eval-<钥匙>.json` 缓存，录音、识别设置或做法代码变了才重新识别）→ 对照表、点行听原声 → 改参考文本 → 填学号后四位保存（写 `references/<编号>.txt`，记 `proofread_log.csv`）。
  录音质检：上传任意格式或选池内录音 → 波形/频谱/音量三张图 + 五项质检和每项的中文解释。数据池不存在或是空的时只显示中文提示；下载仍只走 tmp/exports 的副本，没有开放 outputs/ 和数据池。
  实测（开发环境、模型自带四人测试音频改名为 G1-S1-Q 入池）：第一次生成对照 6.0 秒（含加载模型），第二次用缓存约 0.1 秒。Playwright 逐页点过。
  审查后的小改动：找不到 ffmpeg 时提示怎么装（不再说"文件可能已损坏"）；校对人接受全角数字；刷新录音列表时保留原来选的录音；字错率说明写清"数字写法不同算错"；摘要不再重复对齐。
- 任务 19（后半）完成：`tools/evaluate.py speakers / classify / clips` 和 `tools/compare.py` 对应的指标。
  speakers：只评有说话人标注的录音，`--speakers auto|ref|N`（ref = 每段录音按标注里的人数，即"设对人数"），报告比对的时长、标注人数和分出人数；和整个流程一样先做热词纠错再分说话人。
  classify：在 1055 句剧本台词上测，混淆矩阵、各类准确率/召回率、误报率（全体和第 8 组剧本）；第 6、7 组模型用不了时退回规则，报告开头写明。
  clips：只评有片段标注的录音，起止误差、没配上的片段、和标注都不重叠的工具片段（片段层面的误报）、类别不同的工具片段。
  关键词规则基线在剧本台词上：7 类准确率 588/1055 = 0.557；误报率全体 2/155、第 8 组剧本 0/49（关键词参照剧本写的，偏乐观）。说话人、片段指标没有报告数字（测试用的标注是手写的占位数据）；没有报告任何字错率。
  审查后的小改动：`pipeline.metrics.speaker_error_details` 公开（测评不再用 metrics 的私有函数）；Python 里直接调用 eval_speakers 也接受 "auto"/"4"；第 7 组退回规则也有测试；测评指南第四、五、六、七节补上新选项和"模型成绩只看按组留一报告"。

- 任务 22 完成：60 分钟长录音测试（`python tools/long_audio_test.py --minutes 60`，开发环境 Linux 4 核，机器基本空闲，识别线程数 4）：
  录音 59.7 分钟（四人测试音频重复拼接），397 段；总用时 712 秒（11.9 分钟），实时率 0.20；最大内存约 2.0 GB。
  各步用时（秒）：统一格式 8.6，降噪与端点检测 21.9，识别与热词 227，说话人分离 454，其余不到 0.1。说话人分离占了六成多时间，用时和录音长度大致成正比（10 分钟时 81 秒）。
  这只是开发环境的数字：机房电脑要用 `python\python.exe tools\long_audio_test.py --minutes 10` 另测（第 0 周清单里有），实测之前不要对外说处理时长。内存约 2 GB，8 GB 内存的电脑够用；4 GB 内存的电脑处理一小时录音前要先实测。
