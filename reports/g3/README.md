# 第 3 组报告：说话人分离

> **填写说明**：把"（填写）"换成你们自己的内容，表格里的数字从本文件夹里的测评报告复制：`auto\speakers.md`（人数自动）、`ref\speakers.md`（人数设对）、`after\speakers.md`（改进后）、`compare_speakers.md`（对比表）。
> 只写角色和学号后四位，**不写真实姓名、学号、手机号**。变差了也要如实写。
> 剧本数据上的测评结果不代表真实场景的效果。
>
> 下面的命令都在便携包文件夹里运行；联网安装的电脑把 `python\python.exe` 换成 `python`。`D:\data_pool` 换成老师给的数据池路径。

## 1. 本组专项

- 负责的步骤：第 4 步"说话人分离"；槽位 **diarize（说话人分离）**
- 本组文件：`pipeline/groups/g3_diarize.py`（基线在 `pipeline/step4_diarize.py`）
- 主要指标：**说话人标错的时长比例**（只在有人工说话人标注的录音上算）
- 我们要解决的问题（一两句话）：（填写，例如"几个人快速抢话、声音相近时被分成同一个人"）

## 2. 数据池版本

- 数据池版本：（填写，测评报告开头"数据池版本"那一行，如 v1；基线和改进必须用同一个版本）
- 有说话人标注的录音：（填写：哪几段、一共多少分钟；标注放在 `data_pool\annotations\speakers\`）
- 标注规范：（填写：本组制定的规范要点，如重叠说话怎么标、短回应标不标）

## 3. 基线结果

分别在"人数设自动"和"人数设对"两种情况下各测一次。`--speakers auto` 让工具自己判断人数；
`--speakers ref` 表示每段录音按它的标注里有几个人来设人数（不同录音人数不同，所以不要全部写成同一个数）。
不写 `--speakers` 时按 `config.yaml` 的 `diarize.num_speakers`（默认 -1，自动）。

```bat
python\python.exe tools\evaluate.py speakers --pool D:\data_pool --speakers auto --out reports\g3\auto
python\python.exe tools\evaluate.py speakers --pool D:\data_pool --speakers ref --out reports\g3\ref
```

数字从两份 `speakers.md` 的"明细"表复制（"标注的人数"就是实际人数）。**同时看"比对的时长"**：
只在工具和标注两边都有人说话的时间里比，比对的时长很短时，标错比例再低也说明不了什么。

| 录音 | 标注的人数 | 分出的人数（自动） | 标错比例（人数自动） | 标错比例（人数设对） | 比对的时长（秒，人数设对） |
|---|---|---|---|---|---|
| （填写，如 G3-S1-Q） |  |  |  |  |  |
| 全体 |  |  |  |  |  |

观察：（填写：错在哪里？快速换人、重叠说话、电话外放、声音相近的两个人……挑一两处举例）

## 4. 改进做法说明

- 思路：（填写：调阈值、按说话人边界再切分、换声纹模型……为什么选它）
- 改了哪些函数：（填写：`diarize_g3`）
- 关键参数和取值：（填写：`diarize.threshold`、`min_duration_on`、`min_duration_off` 等）
- 试过但没用上的做法：（填写，也算成果）

## 5. 改进后结果

```bat
python\python.exe tools\evaluate.py speakers --pool D:\data_pool --method diarize=g3 --speakers ref --out reports\g3\after
```

和基线用同样的人数设置比（这里都用"人数设对"，基线的数字取第 3 节 `ref\speakers.md`）：

| 录音 | 标错比例（基线） | 标错比例（改进） |
|---|---|---|
| （填写） |  |  |
| 全体 |  |  |

## 6. 对比表

```bat
python\python.exe tools\compare.py --slot diarize --method g3 --metric speakers --pool D:\data_pool --speakers ref --out reports\g3
```

把 `compare_speakers.md` 里的对比表复制到这里（差值 = 改进 − 基线，标错比例的差值为负表示变好）。
分组项有全体、各录音条件（Q 安静 / N 嘈杂教室 / F 口袋或远距离）和各组，只列有标注的：

| 分组项 | 基线 | 改进 | 差值 |
|---|---|---|---|
| 全体 |  |  |  |
| Q（安静） |  |  |  |
| N（嘈杂教室） |  |  |  |
| F（口袋或远距离） |  |  |  |

## 7. 结论与局限

- 结论：（填写：标错比例变了多少；"人数设对"有多重要）
- 局限：剧本数据上的测评结果不代表真实场景的效果；（填写：标注只有约 10 分钟、扮演者是同学、重叠说话标注本身有误差……）
- 和别的组的关系：（填写：第 1、2 组的降噪和增强会不会影响分离效果？第 9 周串联测试时看）

## 8. 分工（只写角色和学号后四位）

| 角色 | 学号后四位 | 做了什么 |
|---|---|---|
| 组长 |  |  |
| 数据负责人 |  |  |
| 算法负责人 |  |  |
| 测评负责人 |  |  |

## 9. 用了哪些 AI 帮助

- 用了什么工具：（填写）
- 哪些部分用了 AI 帮助：（填写）
- 我们自己检查、修改了什么：（填写）
- 确认：没有把任何录音、真实个人信息粘贴或上传给 AI 工具（见 `docs/guides/ai_tools.md`）。
