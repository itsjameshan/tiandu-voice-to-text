# 第 0 周（开课前）准备清单

按顺序做，大约需要半天到一天。做完每一项打勾。更完整的"还差什么"见 [gaps.md](gaps.md)。

> 2026-10-14 起，72 段测评录音全部由 AI 按剧本合成，**同学不再录音**（不录剧本，也不录第 1 周的朗读和选做的"零到九"），所以不用准备同意书和录音设备。为什么这样改、第 0—6 周怎么安排，见 [ADR 0006](../adr/0006-ai-synthesized-recordings.md)。

## 1. 下载便携包和 AI 测评录音（需要联网，下载时间看网速）

- [ ] 打开 https://github.com/itsjameshan/tiandu-voice-to-text/releases ，在 `v` 开头的版本下面下载 `tiandu-portable-win64-*.zip`（约 400 MB）。
- [ ] 打开 AI 测评录音的发布页 https://github.com/itsjameshan/tiandu-voice-to-text/releases/tag/ai-recordings-v1 （标着"预发布"，和便携包不在同一个 Release 里），下载三个压缩包：
  - [ ] `tiandu-ai-recordings-v1.zip`：72 段 AI 测评录音（`ai_recordings\recordings\G1-S1-Q-TTS.m4a` ……）和说明 `ai_recordings\README.txt`；
  - [ ] `tiandu-ai-answer-key-v1.zip`：标准答案（参考文本、说话人时间、疑似片段起止、Audacity 标签、`summary.csv` 等）。**老师自己保管**，第 6 周同学交了校对和标注以后再拿出来对照（见 [data_pool.md](data_pool.md) 第三节）；
  - [ ] `tiandu-ai-digits-v1.zip`：选做的 TensorFlow 数字实验用的合成录音（100 个合成声音 × 零到九 × 3 遍）。
- [ ] 另外下载（放进同一个 U 盘，给机房没网时用）：
  - [ ] GitHub Desktop 安装包（https://desktop.github.com ，给组长第 7 周用）；
  - [ ] Audacity 安装包（https://www.audacityteam.org ，免费，标注说话人时间、和标准答案对照都用它）。
- [ ] 全部拷进 U 盘。详见 [portable_package.md](portable_package.md)。
- [ ] （建议）把 `ai_recordings\recordings\` 里的 72 段按组分到 8 个文件夹（`G1`……`G8`，每组 9 段），第 1 周各组从 U 盘或共享文件夹拷本组的。**不要改文件名、不要去掉 `-TTS`**。

## 2. 机房实测（一台电脑，约 1 小时）

- [ ] 把便携包解压到 `D:\asr`（纯英文路径；压缩包里自带一层 `tiandu` 文件夹，解压后就是 `D:\asr\tiandu`）。
- [ ] 把 `tiandu-ai-recordings-v1.zip` 也解压到 `D:\asr`，解压后是 `D:\asr\ai_recordings\`。
- [ ] 双击 `selfcheck.bat`，等约 1 分钟，最后一行应是"全部通过"。记下"完整流程"那一行的用时和实时率：________
- [ ] （可选，约 5—10 分钟）测长录音：在便携包文件夹打开命令行，运行 `python\python.exe tools\long_audio_test.py --minutes 10`，记下总用时、实时率和最大内存：________。实时率 × 录音分钟数 ≈ 处理要几分钟（开发电脑上一小时录音约 12 分钟、内存约 2 GB，机房电脑以这里测的为准）。Windows 上没装 psutil 时显示"最大内存未测"，可以同时开任务管理器看。
- [ ] **放一段听听开头的声明**：用播放器打开 `D:\asr\ai_recordings\recordings\G8-S1-Q-TTS.m4a`，开头应先听到"以下为人工智能合成的虚构教学录音，不是真实录音。"再顺便听听机房电脑能不能出声、耳机能不能用。
- [ ] 双击 `start.bat`，浏览器自动打开（没打开就手动输入 `http://127.0.0.1:7860`）。
- [ ] 在网页"整理录音"页上传同一段 AI 录音（`G8-S1-Q-TTS.m4a`），开始整理。结果表格第一句应是开头的声明（识别可能有个别错字），后面是剧本台词。记下处理用时：________
- [ ] **入池试一段**（用一个临时的试验数据池，不要用正式的 `D:\data_pool`）：
  1. 建文件夹 `D:\pool_test\raw`，把 `G8-S1-Q-TTS.m4a` 拷进去（不要改名）。
  2. 在便携包文件夹打开命令行，运行：
     ```bat
     python\python.exe tools\ingest_pool.py --pool D:\pool_test
     python\python.exe tools\export_references.py --pool D:\pool_test
     ```
  3. 看到 `D:\pool_test\normalized\G8-S1-Q.wav`（入池后编号是 `G8-S1-Q`，不带 `-TTS`）和质检结果；打开 `D:\pool_test\references\G8-S1-Q.txt`，第一行应是那句声明，后面是剧本台词。
  4. 试完可以删掉 `D:\pool_test`（`raw` 里的录音入池后是只读的，Windows 会提示，确认即可）。正式入池在第 2 周，和各组数据负责人一起做（见 [data_pool.md](data_pool.md) 第二节）。
- [ ] 点"剧本文本演示"页，选 G8-S1，运行，能看到结果表格。
- [ ] 重启电脑，看 `D:\asr\tiandu` 还在不在：☐ 在（D 盘不还原，以后放 D 盘）　☐ 不在（以后放 U 盘或每次课前从共享文件夹拷）
- [ ] 在另一台机房电脑的浏览器打开 `http://这台电脑的IP:7860`（IP 用 `ipconfig` 查），能不能访问：☐ 能　☐ 不能（防火墙拦了，那就每人用自己的电脑开）
- [ ] 机房自带 TensorFlow 版本：命令行运行 `python -c "import tensorflow as tf; print(tf.__version__)"`，结果：________（给第 6、7 组）
- [ ] （只在做第 3 周选做的数字小实验时需要）机房 Python 有没有 matplotlib：`python -c "import matplotlib"` 不报错就有。没有时 `tools/tf_lab/mfcc_compare.py` 照样算出相关系数，只是不画图。
- [ ] （同上）把 `tiandu-ai-digits-v1.zip` 解压到 `D:\asr`，解压后是 `D:\asr\ai_recordings\digits\`（里面是 `0`—`9` 十个文件夹，文件名如 `tts23_1.wav`）。在便携包文件夹用**机房自带的 Python**（装了 TensorFlow 的那个，不是 `python\python.exe`）运行 `python tools\tf_lab\train_digits.py --data D:\asr\ai_recordings\digits`，看能不能训练起来。同学不用再录"零到九"，也不用 `split_digits.py` 切分；`train_digits.py` 按文件名"_"前面的部分分人（`tts23` 算一个人）。
- [ ] U 盘能不能用：☐ 能　☐ 不能；老师电脑的共享文件夹能不能访问：☐ 能　☐ 不能

## 3. 便携包分发给 31 台电脑

- 共享文件夹能用：把 zip 放在共享文件夹，第 1 周上课时每人自己拷、解压（约 2—5 分钟）。本组的 AI 录音也放在共享文件夹里拷。
- 共享文件夹不能用：准备几个 U 盘轮流拷。
- D 盘会还原：每次课前都要重新解压，或者让学生直接从自己的 U 盘运行（U 盘建议 USB 3.0、8 GB 以上）。

## 4. 数据池

- [ ] 在老师电脑上建文件夹 `D:\data_pool`（或 U 盘），把 `config.yaml` 里 `paths.data_pool` 改成这个路径。确认 `config.yaml` 里 `recordings.ai_synthesized` 是 `true`（默认就是）。
- [ ] 运行 `python\python.exe tools\export_references.py --pool D:\data_pool`，生成 72 份待校对的参考文本（第一行自动是那句声明，后面是剧本台词）。
- 72 段 AI 录音第 2 周和各组数据负责人一起入池。详见 [data_pool.md](data_pool.md)。

## 5. 纸质材料

- [ ] 分组名单和角色表（线下保存，不进仓库）：8 组，第 2 组 3 人，其余各 4 人；每组定组长、数据、算法、测评（数据负责人管理本组 9 段 AI 录音）。
- [ ] 每组的任务书（`docs/groups/g1.md`…`g8.md`）可以打印发给各组，或者让学生在便携包的 `docs\groups\` 里看。

## 6. 听录音用的耳机

- [ ] 第 2 周每组要把本组 9 段 AI 录音听一遍、记下合成的问题，第 4 周校对、第 5 周标注也要反复听。确认机房电脑能出声；请同学自带耳机，或者准备几副公用的。

## 7. 课程内容

- [ ] 看一遍项目介绍视频（`handoff/04_项目介绍视频/`，第 1 周课上播放）。视频里说同学按剧本录音、签同意书，现在改了，见 README 第 2 节"和视频不一样的地方"。
- [ ] 第 1 周讲红线时加上：**AI 录音不得剪掉开头的声明，不得当作真实录音使用或传播**（见 [AI 测评录音说明](../guides/recording.md) 第三节）。
- [ ] 过一遍剧本里需要老师过目的 3 处（`data/README.md` 末尾）。
- [ ] 读一遍 README 的"每组每周做什么"总表，按实际周数（8/10/12 周）调整。
