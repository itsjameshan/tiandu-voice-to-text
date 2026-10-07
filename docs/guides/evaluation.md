# 测评指南：每个指标怎么算、怎么看

## 总原则

1. **同一份数据比**：基线和改进都在**冻结的数据池版本**（如 v1）上跑，否则数字不可比。报告开头会自动写明版本；冻结以后清单、参考文本、标注或录音又改过的，版本后面会注明"冻结后有 N 个文件改动，和 v1 不一致"——这时的数字不能和别人用 v1 测的比，请老师冻结新版本，基线和改进都重测。
2. **改一个地方测一次**：一次只换一个槽位的做法，才知道变化是谁带来的。
3. **按录音条件分开看**：安静（Q）、嘈杂（N）、口袋/远距离（F）分开报告，很多改进只在某一种条件下有效。
4. **如实写**：变差了也写；剧本数据上的结果不代表真实场景。

命令都在便携包文件夹里运行（`python\python.exe tools\...`），`--pool` 填数据池路径。加 `--out reports\gN` 把结果存进本组报告文件夹。
**基线和改进做法的结果放不同的文件夹**：基线放 `reports\gN`，本组做法（带 `--method`）放 `reports\gN\after`，放在同一个文件夹会互相覆盖；`compare.py` 写的是 `compare_*.md`，可以和基线放在一起。有几种基线或几种做法的组按本组任务书和报告模板分文件夹：第 1 组降噪做法放 `reports\g1\after`、端点检测做法放 `reports\g1\after_vad`；第 3 组人数自动、设对两次基线放 `reports\g3\auto`、`reports\g3\ref`；第 5 组热词纠错关、开两次基线放 `reports\g5\off`、`reports\g5\on`；第 6、7 组关键词规则基线放 `reports\gN\baseline`，`--method classify=gN` 只是加载检查，放 `reports\gN\check`（模型成绩看训练报告）。

## 一、字错率（CER）—— 全员；第 1、2 组主指标

```bat
python\python.exe tools\evaluate.py cer --pool D:\data_pool --out reports\g1
```

- **怎么算**：把识别结果和参考文本都去掉标点和空格、全角转半角、字母转大写，然后逐字对齐，数出
  - 错字（替换）：参考"隐"识别成"影"；
  - 漏字（删除）：参考有、识别没有；
  - 多字（插入）：参考没有、识别多出来；

  字错率 = （错字 + 漏字 + 多字）÷ 参考文本字数。例："雾隐行舟旅行社"→"雾影行走旅行社"，错 2 字，2 ÷ 7 ≈ 28.6%。
- **用的是测评模式**：识别时不把数字转成阿拉伯数字（输出"两千八百"），和参考文本的汉字读法口径一致。
- **怎么看**：越低越好。看全体、看每种条件（Q/N/F）、看每组。字错率可能超过 100%（多字很多时）。
- 输出：`cer.csv`（每段录音一行：错字、漏字、多字、字数、字错率）和 `cer.md`（汇总表）。

## 二、专名正确率、过度纠正 —— 第 5 组

```bat
python\python.exe tools\evaluate.py hotwords --pool D:\data_pool --hotword off --out reports\g5\off
python\python.exe tools\evaluate.py hotwords --pool D:\data_pool --hotword on --out reports\g5\on
python\python.exe tools\evaluate.py hotwords --pool D:\data_pool --method hotword=g5 --hotword on --out reports\g5\after
```

- **专名正确率**：参考文本里出现的热词（`data/hotwords.txt` 里的虚构旅行社、店名、地名、行话），在识别结果里也出现了的比例。
- **过度纠正**：参考文本里人物**故意说错或简称**的名字（`data/hotword_variants.csv`，如把"松风晚渡"说成"松风行舟"），本来识别对了，却被热词纠错改掉了（改成正确名称、或者改成正确名称的简称）的次数。这是错误——工具应该忠实记录人说了什么。精确的计数规则写在每份 `hotwords.md` 报告的"说明"一节里。
- **热词纠错默认是关着的**：测热词时一定要加 `--hotword on`，否则测出来的是"没纠错"的结果。开（`--hotword on`）、关（`--hotword off`）各跑一次，比较两个指标。
- 识别结果会缓存在数据池的 `asr_cache` 里，只换热词做法时不用重新识别，很快；想全部重新识别就加 `--no-cache`。数据池文件夹不能写入（只读的共享文件夹）时不留缓存，每次都全部重新识别。

## 三、数字提取正确率 —— 第 4 组

```bat
python\python.exe tools\evaluate.py numbers --out reports\g4
python\python.exe tools\evaluate.py numbers --method normalize=g4 --out reports\g4\after
```

- **在剧本文字上测**：拿每句台词（汉字读法）做转换和提取，和剧本每行的 `numbers`（标准写法，如"2800元；15:40；0871-0000-6688"）比较。不需要录音，随时能测。
- 按类型分开报告：**主指标**是金额、电话、证号、合同号、订单号、时刻、日期（和纠纷核查直接相关）；**次要指标**是数量、时长、其他。
- **正确率（精确率）**：提取出来的里面有多少是对的；**召回率**：标准答案里有多少被提取出来了。两个都要看。

## 四、说话人标错的时长比例 —— 第 3 组

```bat
:: 人数设自动（工具自己判断有几个人）
python\python.exe tools\evaluate.py speakers --pool D:\data_pool --speakers auto --out reports\g3\auto
:: 人数设对（每段录音按标注里有几个人来设）
python\python.exe tools\evaluate.py speakers --pool D:\data_pool --speakers ref --out reports\g3\ref
:: 换成本组的做法
python\python.exe tools\evaluate.py speakers --pool D:\data_pool --method diarize=g3 --speakers ref --out reports\g3\after
```

- 只评有人工标注（`data_pool/annotations/speakers/`）的录音。
- **怎么算**：把时间切成 10 毫秒一格；工具输出的"说话人1、说话人2"和人工标的"导游、游客甲"先找最佳对应关系（谁和谁重叠最多）；然后在两边都有人说话的时间里，数被标错的格数占多少。
- **`--speakers`**：`auto` 工具自己判断人数；`ref` 每段录音按它的标注里有几个人来设（"设对人数"，每段录音可能不同）；也可以写一个数，如 `4`。不写就按 `config.yaml` 的 `diarize.num_speakers`。比较"人数设自动"和"人数设对"两种情况。
- 越低越好。同时看报告里的**比对的时长**：只有两边都有人说话的时间才算，工具漏掉的、多出来的说话时间不算在内。比对的时长很短时，标错比例再低也说明不了什么。
- 报告里还列出每段录音标注了几个人、工具分出了几个人。

## 五、话术分类：准确率、召回率、误报率 —— 第 6、7、8 组

```bat
:: 关键词规则基线（在剧本台词文字上测，不需要录音和数据池）
python\python.exe tools\evaluate.py classify --out reports\g6\baseline
:: 训练并按组留一测评 TensorFlow 模型（用机房自带的 Python；第 7 组把 g6 换成 g7）
python tools\train_classifier.py --model g6 --eval logo
```

- **准确率（精确率）**：被分到某一类的句子里，真属于这一类的比例。
- **召回率**：真属于某一类的句子里，被分对的比例。
- **混淆矩阵**：一张 7×7 的表，行是正确类别、列是预测类别，能看出哪两类最容易混。
- **误报率**：正确答案是"正常讲解"的句子里，被标成任何一个"疑似"类别的比例。全部 24 个剧本算一次，第 8 组（正常讲解对照组）的剧本再单独算一次。误报会冤枉人，越低越好。
- **按组留一**（主结果）：用 7 个组的台词训练、测剩下 1 个组，轮 8 次，把 8 次的预测汇总成一个混淆矩阵——相当于"用别的同学写的句子来测"。**随机划分**的结果会虚高（训练集和测试集里有同一个剧本的相似句子），只作对照。
- AI 生成的补充句子只进训练集；报告里写"用 / 不用补充句子"两种结果。
- **模型的成绩只看 `train_classifier.py` 的报告**（按组留一，用 / 不用补充句子两份）。`evaluate.py classify --method classify=g6` 只用来确认模型能加载、能跑：训练脚本最后是用全部剧本台词训练模型的，再拿同样的台词测等于"考原题"，数字会明显虚高。
- 便携包里没有 TensorFlow，在便携包里运行时 `g6`、`g7`、`tf_model` 都会退回关键词规则，报告开头的"运行时的提示"会写明。
- 关键词规则的关键词是参照剧本写的，在剧本上测的数字也偏乐观。

## 六、片段起止误差 —— 第 8 组

```bat
python\python.exe tools\evaluate.py clips --pool D:\data_pool --out reports\g8
python\python.exe tools\evaluate.py clips --pool D:\data_pool --method clips=g8 --out reports\g8\after
```

- 只评有片段标注（`data_pool/annotations/clips/`）的录音。
- 把工具截出的片段和人工标的片段按重叠最多配对，报告**起点平均误差、终点平均误差**（秒）和**没配上的片段数**（工具漏掉的 + 多出来的）。
- 基线一句话出一个片段，而人工标的一个片段往往包含好几句话，只有一个工具片段能和它配上，所以"没配上的工具片段"多是正常的——合并相邻片段正是第 8 组的改进方向。
- **和标注都不重叠的工具片段**：标注人认为没有纠纷的地方被剪成了疑似片段（片段层面的误报），越少越好。句子层面的误报率看第五节的 `classify`。
- **类别不同的工具片段**：时间对上了、类别不一样，主要由话术分类（第 6、7 组）决定。
- 为了快，测片段时不做说话人分离（片段里没有说话人），报告里写明了。

## 七、一条命令出对比表

```bat
python\python.exe tools\compare.py --slot denoise --method g1 --metric cer --pool D:\data_pool --out reports\g1\denoise
```

在同一份数据上先用基线跑、再用你们的做法跑，生成 `compare_cer.md` 和 `.csv`：每行一个分组项（全体、Q、N、F……），列出基线、改进、差值。这张表直接放进报告的"对比数据"一节。

第 3 组比较时加 `--speakers ref`（或 `auto`）：`python\python.exe tools\compare.py --slot diarize --method g3 --metric speakers --pool D:\data_pool --speakers ref --out reports\g3`。

第 5 组比较热词做法时加 `--hotword on`：`python\python.exe tools\compare.py --slot hotword --method g5 --metric hotwords --pool D:\data_pool --hotword on --out reports\g5`。

所有测评命令都可以加 `--files G1-S1-Q G1-S1-N` 只测几段录音（先试一下、很快），确认没问题再全部跑。全部 72 段录音识别一遍，在普通电脑上大约要几十分钟。

| 组 | `--slot` | `--metric` |
|---|---|---|
| 第 1 组 | `denoise`（或 `vad`） | `cer` |
| 第 2 组 | `enhance`（或 `vad`） | `cer` |
| 第 3 组 | `diarize` | `speakers` |
| 第 4 组 | `normalize` | `numbers` |
| 第 5 组 | `hotword` | `hotwords` |
| 第 6、7 组 | `classify` | `classify` |
| 第 8 组 | `clips` | `clips` |

## 八、在云端开发环境里

开发这个模板时没有任何学生录音，只用模型自带的测试音频改名检验流程能跑通，**没有报告任何字错率数字**。所有真实的测评数字都由同学们在本地数据池上测出来。
