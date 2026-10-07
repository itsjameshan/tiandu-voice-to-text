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
