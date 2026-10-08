# 第 5 组报告：热词

> **填写说明**：把"（填写）"换成你们自己的内容，表格里的数字从本文件夹里的测评报告（`hotwords.md`、`compare_hotwords.md`）复制。
> 只写角色和学号后四位，**不写真实姓名、学号、手机号**。变差了也要如实写。
> 剧本数据上的测评结果不代表真实场景的效果。
>
> 下面的命令都在便携包文件夹里运行；联网安装的电脑把 `python\python.exe` 换成 `python`。`D:\data_pool` 换成老师给的数据池路径。
> `--hotword on` / `--hotword off` 临时打开、关闭热词纠错（不填就按 `config.yaml` 的 `hotword.enabled`，默认关闭）。

## 1. 本组专项

- 负责的步骤：第 3 步"语音识别与热词"；槽位 **hotword（热词纠错）**
- 本组文件：`pipeline/groups/g5_hotwords.py`（基线在 `pipeline/hotwords.py`：识别后按拼音相似度替换）
- 主要指标：**专名正确率**（越高越好）和**过度纠正次数**（越少越好），两个一起看
- 我们要解决的问题（一两句话）：（填写，例如"虚构旅行社名几乎都被识别成同音别字，但人物故意说错的名字不能被改对"）

## 2. 数据池版本

- 数据池版本：（填写，测评报告开头"数据池版本"那一行，如 v1；基线和改进必须用同一个版本）
- 用了哪些录音：（填写）
- 说错、简称的名称：`data/hotword_variants.csv`（填写：数据阶段抽查参考文本时，又发现了哪些实际说错的地方）

## 3. 基线结果

```bat
:: 不纠错
python\python.exe tools\evaluate.py hotwords --pool D:\data_pool --hotword off --out reports\g5\off
:: 基线纠错（拼音相似度）
python\python.exe tools\evaluate.py hotwords --pool D:\data_pool --hotword on --out reports\g5\on
```

| 分组项 | 专名正确率（不纠错） | 专名正确率（基线纠错） | 过度纠正（不纠错） | 过度纠正（基线纠错） |
|---|---|---|---|---|
| 全体 |  |  |  |  |
| Q（安静） |  |  |  |  |
| N（嘈杂教室） |  |  |  |  |
| F（口袋或远距离） |  |  |  |  |

观察：（填写：`hotwords.md` 里"各专名的命中情况"表，哪些名称最常错？错成了什么？在"数据校对"页找一两个例子）

## 4. 改进做法说明

- 思路：（填写：调 `min_len`、`max_syllable_mismatch`，保护名单，看上下文，或者换成解码时热词……为什么选它）
- 改了哪些函数：（填写：`hotword_g5`）
- 怎么避免过度纠正：（填写）
- 试过但没用上的做法：（填写，也算成果）

## 5. 改进后结果

```bat
python\python.exe tools\evaluate.py hotwords --pool D:\data_pool --method hotword=g5 --hotword on --out reports\g5\after
```

| 分组项 | 专名正确率 | 过度纠正次数 | 热词纠错次数 |
|---|---|---|---|
| 全体 |  |  |  |
| Q（安静） |  |  |  |
| N（嘈杂教室） |  |  |  |
| F（口袋或远距离） |  |  |  |

## 6. 对比表

```bat
python\python.exe tools\compare.py --slot hotword --method g5 --metric hotwords --pool D:\data_pool --hotword on --out reports\g5
:: 热词纠错也会改字错率，顺便看一下：
python\python.exe tools\compare.py --slot hotword --method g5 --metric cer --pool D:\data_pool --hotword on --out reports\g5
```

识别结果有缓存，换热词做法时不用重新识别，几秒钟就出结果。把 `compare_hotwords.md` 里的对比表复制到这里
（差值 = 改进 − 基线；专名正确率的差值为正、过度纠正的差值为负表示变好）：

| 分组项 | 基线 | 改进 | 差值 |
|---|---|---|---|
| 全体·专名正确率 |  |  |  |
| 全体·过度纠正次数 |  |  |  |

## 7. 结论与局限

- 结论：（填写：专名正确率提高了多少，过度纠正有没有增加；字错率有没有变化）
- 局限：剧本数据上的测评结果不代表真实场景的效果；过度纠正是按次数估算的（见报告里的说明），没有逐字对齐；（填写：热词表只有 121 个虚构名称……）
- 和别的组的关系：（填写：纠错后的名称会进入第 4 组的名称提取和第 6、7 组的分类，第 9 周串联测试时看）

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
