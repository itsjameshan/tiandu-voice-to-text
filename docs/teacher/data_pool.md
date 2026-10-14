# 数据池管理（老师）

数据池是全班 72 段 AI 测评录音、参考文本、标注和清单的统一存放处。放在老师电脑或 U 盘上，**不进代码仓库、不上传网络**。

> 2026-10-14 起，72 段测评录音全部由 AI 按剧本合成（24 个剧本 × Q 安静、N 嘈杂教室、F 口袋或远距离），同学不再录音（见 [ADR 0006](../adr/0006-ai-synthesized-recordings.md)）。录音从哪来、三种条件怎么模拟的，见 [AI 测评录音说明](../guides/recording.md)。

## 一、建数据池（第 0 周）

1. 在老师电脑上建一个纯英文路径的文件夹，例如 `D:\data_pool`。
2. 把便携包（或仓库）里 `config.yaml` 的 `paths.data_pool` 改成这个路径；也可以每条命令都加 `--pool D:\data_pool`。同时确认 `config.yaml` 里 `recordings.ai_synthesized` 是 `true`（默认就是）。
3. 生成 72 份待校对的参考文本（第一行是录音开头念的声明"以下为人工智能合成的虚构教学录音，不是真实录音。"，后面照抄剧本台词）：
   ```bat
   python\python.exe tools\export_references.py --pool D:\data_pool
   ```
   已存在的文件不会被覆盖（校对过的不会丢）。

文件夹结构（工具自动建）：

```
D:\data_pool\
  raw\                  AI 测评录音，如 G1-S1-Q-TTS.m4a（入池后设为只读）
  normalized\           统一格式后的 16kHz 单声道 WAV，如 G1-S1-Q.wav（编号不带 -TTS）
  references\           72 份参考文本，如 G1-S1-Q.txt
  annotations\speakers\ 说话人时间标注
  annotations\clips\    疑似片段起止标注
  asr_cache\            识别结果缓存（"数据校对"页和测评工具共用，可以删，删了只是下次要重新识别）
  digits\               选做数字实验的旧位置；现在用 tiandu-ai-digits-v1.zip 里的合成录音，这里一般空着
  manifest.csv          清单（文件、剧本、条件、时长、采样率、质检、上传时间、SHA-256 指纹）
  qc_report.csv         质检报告
  proofread_log.csv     校对记录（谁、什么时候校对了哪段）
  acceptance.csv        验收表（每段录音一行）
  acceptance_summary.csv 各组汇总
  versions\v1\          冻结的版本
```

## 二、入池（第 2 周，老师和各组数据负责人一起做）

1. 从 [ai-recordings-v1 发布页](https://github.com/itsjameshan/tiandu-voice-to-text/releases/tag/ai-recordings-v1)下载的 `tiandu-ai-recordings-v1.zip`（第 0 周已拷进 U 盘）解压到纯英文路径，如 `D:\asr`，得到 `D:\asr\ai_recordings\recordings\`，里面是 72 段 `G1-S1-Q-TTS.m4a` ……
2. 把这 72 段 m4a 拷进 `D:\data_pool\raw\`。**不要改名、不要去掉 `-TTS`**。各组数据负责人核对本组 9 段都在。
3. 运行：
   ```bat
   python\python.exe tools\ingest_pool.py --pool D:\data_pool
   python\python.exe tools\export_references.py --pool D:\data_pool
   ```
   第二条在第 0 周运行过的话，已有的参考文本不会被覆盖，可以放心再运行一次。
4. 看输出：
   - 合格的会转成 `normalized\G1-S1-Q.wav`，写进 `manifest.csv`，原始文件设为只读。**`G1-S1-Q-TTS.m4a` 入池后编号是 `G1-S1-Q`**：参考文本 `references\G1-S1-Q.txt`、标注 `annotations\speakers\G1-S1-Q.csv` 都用这个编号；
   - **文件名不合格**的会列出来并说明原因（小写、多了"(1)"、组号不对……），不会被处理。AI 录音原来的文件名都是合格的，出现这种情况一般是拷贝时被改了名（如拷了两次变成 `G1-S1-Q-TTS (1).m4a`）：删掉多余的或改回原名，再运行一次；
   - **同一段录音既有带 `-TTS` 的、又有不带的文件**（如 `G1-S1-Q-TTS.m4a` 和 `G1-S1-Q.m4a`）会报错：只留带 `-TTS` 的那个；
   - **质检有问题**的会写出问题（时长和预计时长差太多、音量太小、削波、大段无声、采样率低）。质检只报告、不修改录音。质检提示里的"重录"是给真人录音写的，**AI 录音不用重录**：先听一听，记进本组的"AI 录音问题清单"，告诉老师（需要时老师改声音分配表、重新合成，见第八节）；质检结果同时给第 1 组试定阈值用（见第 6 条）。
5. 已经入池的文件（按 SHA-256 指纹判断）再运行时会跳过，可以放心重复运行。
6. 质检阈值在 `config.yaml` 的 `qc` 一节。第 1 组用这 72 段的质检结果试定阈值，老师确认后改。预计时长来自 `data/recording_plan.csv`，是写剧本时估的，AI 录音的实际时长会有出入。
7. **换成新的 AI 录音**（例如改用重新合成的 `ai-recordings-v2`，见第八节）：先把 `raw\` 里的旧文件移走（它是只读的，Windows 会提示，确认即可），把新录音用同样的文件名放进 `raw\`，再运行入池。工具会替换 `manifest.csv` 和 `qc_report.csv` 里的旧行、删掉旧的识别缓存；**参考文本要按新录音重新校对**（验收时只数入池之后的校对）。所以最好在第 4 周校对开始前换好。
8. 入池中途提示"没有权限写入"（例如 `qc_report.csv` 正用 Excel 打开着）：关掉 Excel 再运行一次即可，没处理完的文件会重新处理。
9. `manifest.csv` 和 `qc_report.csv` 由工具维护，每次入池会按固定的列重写。可以用 Excel 打开看，但**不要往里加列或另存**（要做笔记请另建表格）。

AI 录音开头的声明不得剪掉，不得把 AI 录音当作真实录音使用或传播。

## 三、校对和标注（第 4—6 周）

- 学生在网页"数据校对"页校对参考文本，保存时自动写 `proofread_log.csv`。参考文本按录音"实际说了什么"改：**第一行的声明保留**；合成念错了字（念成别的字、漏念、多念）的按录音改；多音字读错了音、字没变的，保留剧本的字；这两种都记进本组的"AI 录音问题清单"。重叠、打断的台词听不清时保留剧本原文，备注写"重叠"；N 条件背景里别的剧本的台词不写进参考文本（见 [../guides/proofreading.md](../guides/proofreading.md)）。"数据校对"页里某段录音还没有参考文本文件时，也是先照抄剧本（含声明）再改。
- **存着数据池的电脑只给本机用**：网页工具默认 `host: 0.0.0.0`（局域网里别的电脑也能打开），这样别的电脑也能打开这台电脑的"数据校对"页、读写数据池。在这台电脑上把 `config.yaml` 里 `app:` 下的 `host` 改成 `127.0.0.1`，或者用命令 `python\python.exe app.py --host 127.0.0.1` 启动。学生在各自电脑上用便携包校对时不受影响。网页工具的"数据池路径"要指向同一个 `D:\data_pool`（共享文件夹也可以，如 `\\老师电脑\data_pool`）。
- 如果学生在自己电脑上校对（拷了一份数据池），收回来时只需要把 `references\` 里改过的 txt 和 `proofread_log.csv` 里新增的行合并回来。
- 标注用 Audacity 导出标签，再用 `tools\labels_to_csv.py` 放进 `annotations\`（见 [../guides/annotation.md](../guides/annotation.md)）。说话人标注里，开头的声明标说话人"**声明**"（`labels_to_csv.py` 不会把"声明"当成写错的角色名）。测评时 `tools\evaluate.py speakers --speakers ref` 按标注里的人数设人数，声明算一个人（它用单独的合成声音）。
- AI 录音里没有同学的声音，**任何一段都可以做说话人标注**，不用再核对同意书。每组说话人标注至少 10 分钟、要有片段标注，这些验收要求不变（见第四节）。

### 标准答案怎么用（老师）

合成时同时生成了标准答案，在 `tiandu-ai-answer-key-v1.zip` 里，解压后是 `ai_recordings\answer_key\`：

| 文件 | 内容 |
|---|---|
| `README.txt` | 说明 |
| `references\G1-S1-Q.txt` | 参考文本：第一行是声明，后面是合成时照着念的剧本台词 |
| `annotations\speakers\G1-S1-Q.csv` | 说话人时间（`start,end,speaker`），开头的声明标"声明" |
| `annotations\clips\G1-S1-Q.csv` | 疑似片段起止（`start,end,label`） |
| `audacity_labels\G1-S1-Q.speakers.txt`、`G1-S1-Q.clips.txt` | 同样的说话人和片段，Audacity 标签格式，"文件 → 导入 → 标签"能直接导入 |
| `timeline\G1-S1-Q.csv` | 每句台词：序号、说话人、开始、结束、类别、声音编号、舞台说明、台词 |
| `summary.csv` | 每段录音一行：时长、计划时长、F 条件是口袋还是 2—3 米外、句数、说话人数（含声明）、片段数、声音编号 |
| `voice_plan.csv` | 每个剧本每个角色用的合成声音（`data/tts_voice_plan.csv` 的副本） |

怎么用：

1. **老师自己留着**，第 1—5 周不发给任何组。第 2 组检查 F 条件用 [AI 测评录音说明](../guides/recording.md#24-个剧本的-f-条件) 第五节的表（组号 + 剧本号单数是口袋，双数是 2—3 米外），再自己听、看语谱图核对，不需要 `summary.csv`。
2. 老师自己检查时：用 `summary.csv` 看每段的时长和计划时长、F 条件的类型，和质检结果对一对；用 `timeline\` 查某句台词在第几秒，抽查同学的校对和标注更快。
3. **第 6 周，同学交了校对和标注以后**，再把标准答案发给各组对照：
   - 标注：在 Audacity 里打开 `normalized\G3-S1-Q.wav`，"文件 → 导入 → 标签"先导入自己标的标签，再导入 `audacity_labels\G3-S1-Q.speakers.txt`（片段用 `.clips.txt`），两条标签轨上下对照，把差别记进对照记录（角色标错、漏标、起止拖得太宽或太窄……）。标准答案的说话人标签是一句台词一个标签，和同学"同一个人连续说的话标成一段"个数对不上是正常的；起止差一点点也是正常的。详细做法和对照记录的模板见 [../guides/annotation.md](../guides/annotation.md) 第四节。第 3、8 组汇总全班和标准答案的偏差。
   - 参考文本：和 `references\G1-S1-Q.txt` 对照着看。标准答案的参考文本是"合成时照着念的剧本台词"，**不知道合成哪里念错了**；同学按录音改过的地方和它不一样是对的（问题清单里应该有这一条）。
4. **数据池里用的是同学校对、标注的结果**。不要把标准答案拷进数据池代替同学的校对和标注。对照后同学发现自己标错、校错的，可以重新听录音改正（参考文本在"数据校对"页改，标注在 Audacity 里改好、重新导出再转换），**不能照抄标准答案的时间**；改了什么写进对照记录。改动要在冻结第 1 版之前完成。
5. **标准答案在公开的 Release 页面上，同学也能下载**。所以考核看过程，不只看结果：老师课上抽查，例如看校对记录里谁在什么时候校对的、本组的问题清单、Audacity 工程，随机挑一段让同学边听边讲为什么这样标。

## 四、验收（第 6 周末）

```bat
python\python.exe tools\check_pool.py --pool D:\data_pool
```

生成 `acceptance.csv`（每段录音一行）和 `acceptance_summary.csv`（每组一行），并在屏幕上打印各组汇总：

| 列 | 含义 |
|---|---|
| 已交 | `raw\` 里有这段录音的文件（文件名不合格也算交了，"命名"一列会写出原因），或已经入池。AI 录音由老师和数据负责人统一放进 `raw\`，正常情况下 72 段都是"是" |
| 命名 | 文件名是否合格 |
| 质检 | 合格 / 具体问题 / 未入池 |
| 校对遍数 | 入池（上传时间）之后有几个不同的人校对过（换成新录音后要重新校对） |
| 说话人标注（秒） | 这段录音标了多少秒 |
| 片段标注 | 有没有片段标注 |
| 通过 | 已交 且 质检合格 且 校对遍数 ≥ 2 |

各组汇总（`acceptance_summary.csv`，列：组、已交、应交、通过数、说话人标注总分钟、是否达到 10 分钟）：每组应交 9 段；说话人标注按标签覆盖的时长算，要求 ≥ 10 分钟。表格里的是/否方便在 Excel 里筛选。标注表格有错（时间倒了、类别名不对）时工具会报出是哪个文件第几行，改好后再运行。

## 五、冻结版本（验收后）

```bat
python\python.exe tools\freeze_pool.py --pool D:\data_pool --version v1
```

先重新生成验收表，再把清单、质检报告、参考文本、标注、校对记录、验收表和各组汇总复制到 `versions\v1\`，并记下每个文件（包括录音）的 SHA-256 指纹。标注表格有错、或者清单里的录音文件不见了时，工具会拒绝冻结并说明原因。之后所有测评报告都会写明"数据池版本 v1"；冻结以后有人又改了清单、参考文本、标注或录音（例如在"数据校对"页又保存了一次），报告里会写成"v1（冻结后有 N 个文件改动，和 v1 不一致：……）"，提醒你冻结新版本。

- 冻结后发现参考文本还有错：改好后冻结一个新版本 `v2`，并通知各组用 v2 重新测基线和改进（**基线和改进必须用同一个版本**）。
- 同一个版本名不能覆盖，工具会拒绝。
- 数据池版本（v1、v2）和 AI 录音的发布版本（`ai-recordings-v1`、`ai-recordings-v2`）是两回事：换了 AI 录音，数据池要重新校对、验收，再冻结一个新的数据池版本。

## 六、分发给各组测评（第 7 周起）

各组跑测评需要 `normalized\`、`references\`、`annotations\`、`manifest.csv`。可以：

- 共享文件夹：各组直接用 `--pool \\老师电脑\data_pool`。**最好给各组写入权限**：测评和"数据校对"页会把识别结果缓存在 `asr_cache\` 里（同一段录音、同样的做法只识别一次，8 组共用）；只读也能测，但每次都要把录音全部重新识别，72 段要几十分钟。测评不会改别的文件；
- U 盘：把整个数据池（约 1 GB 以内，主要是 `normalized\` 里的 WAV）拷给每组一份。

`normalized\` 里的 WAV 文件名就是录音编号（如 `G1-S1-Q.wav`）。`config.yaml` 里 `recordings.ai_synthesized` 是 `true`（默认）时，工具靠这个编号认出它是 AI 录音，在导出的 Word 文件信息、ZIP 里的说明.txt 和 JSON 里写上录音来源。所以拷出来用时**不要改名**（改成别的名字就认不出是 AI 录音了），也不要剪掉开头的声明、不要当作真实录音传给别人（AI 录音本来就在公开的 Release 页面上，带回家练习没问题）。在网页上整理、演示时，最好直接用 AI 录音包里的 `G1-S1-Q-TTS.m4a`（文件名带 `-TTS`，最清楚）。

提醒各组：测评数据是 AI 按剧本合成的，字错率、说话人分离结果会偏乐观，报告里要写明"测评数据是 AI 合成的"，不代表工具在真人录音上的效果。

## 七、课程结束后

- AI 录音里没有同学的声音，数据池可以留给下一轮课程用（开头的声明、文件名的 `-TTS` 保持不变）。数据池仍然不进代码仓库、不上传网络。
- 收回或删掉各组 U 盘、机房电脑里的副本（`raw\`、`normalized\`、`asr_cache\` 和解压出来的 `ai_recordings\`），以免被剪掉声明、当作真实录音传播。
- 参考文本、标注、统计结果可以保留用于教学。

## 八、重新合成 AI 录音（一般不用）

72 段 AI 录音、标准答案和数字实验录音已经发布在 `ai-recordings-v1`，老师一般不用自己合成。需要重新合成的情况，例如改了某个角色的声音（用 Excel 改 `data/tts_voice_plan.csv` 里的 `sid`、`speed`）：

- **用 GitHub Actions（推荐）**：仓库上方 **Actions** → 左边"**合成 AI 测评录音**" → 右边 **Run workflow** → 分支选 `main` → "发布版本号"填一个新的版本号，如 `ai-recordings-v2`（必须以 `ai-recordings-` 开头）→ 绿色按钮 **Run workflow**。合成完会发布一个新的预发布 Release（不会顶替便携包成为"最新版本"），三个压缩包在它的 Assets 里。"发布版本号"留空就只合成不发布：压缩包在这次运行页面最下面的 **Artifacts** 里（保留 30 天，需要登录才能下载）。
- **在本机合成**（用联网安装了项目的电脑，见 README 第 7.2 节；要联网下载模型；全部 72 段在 4 核电脑上约 1 小时）：
  ```bat
  python models\download_models.py --optional kokoro_tts
  python tools\synthesize_recordings.py
  ```
  合成好的放在项目里的 `ai_recordings\` 文件夹（不进 git）。只想先试一段：`python tools\synthesize_recordings.py --scripts G1-S1 --conditions Q`；`--digits` 生成数字实验录音；`--make-voice-plan` 重新给每个角色挑声音（会改写 `data/tts_voice_plan.csv`）。

重新合成以后：按第二节第 7 条换录音、重新校对，标准答案也要换成新的版本。新录音开头照样念声明，文件名照样带 `-TTS`。
