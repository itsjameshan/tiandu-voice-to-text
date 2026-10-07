# 第 9 周：串联测试（老师主导）

目标：把 8 个组的改进做法都接进完整流程，跑一遍全量测试，生成几份核查初稿样例，答辩时演示。

## 一、合并各组代码（周初）

1. 确认 8 个组的 PR 都已合并进 `main`（或代交的文件都已提交），见 [merging.md](merging.md)。
2. 第 6、7 组的训练好的模型拷进老师便携包的 `models\classifier_g6\`、`models\classifier_g7\`。
3. 发布一个测试版便携包（Releases → 新建 `v0.9.0-week9`，见 [portable_package.md](portable_package.md) 第五节），约 15 分钟后下载；或者老师电脑用联网安装的方式直接 `git pull`。

## 二、在完整流程里切换各组做法

因为每组的改进都是"做法"，**不用改代码**，只需要切换：

- **网页里**："整理录音"页 → 展开"高级设置" → 每个槽位选 `baseline` 或组名（g1、g2……），然后开始整理。
- **配置文件里**：改 `config.yaml` 的 `methods` 一节，例如：
  ```yaml
  methods:
    denoise: g1
    enhance: g2
    vad: g1
    hotword: g5
    diarize: g3
    normalize: g4
    classify: g6
    clips: g8
  ```
  改完重启网页工具。

## 三、全量测试（课上，各组配合）

建议顺序（每一步都和"全部基线"比较）：

1. **全部基线**跑一次全体测评，作为参照：
   ```bat
   python\python.exe tools\evaluate.py cer --pool D:\data_pool --out reports\integration\baseline
   ```
2. **只换一个组的做法**，各组确认自己的改进在完整流程里依然有效（各组自己的 `compare.py` 结果应该能复现）。
3. **全部换成各组做法**再跑一次，看整体效果：
   ```bat
   python\python.exe tools\evaluate.py cer --pool D:\data_pool --method denoise=g1 --method enhance=g2 --method hotword=g5 --out reports\integration\all
   ```
   各组做法放在一起时可能互相影响（例如第 1 组的降噪让第 3 组的说话人分离变差），这是很好的讨论题，写进各组报告的"局限"一节。
4. 第 6、7 组的模型 A、B 在同一份数据上对比；第 8 组在全部做法下测一次误报率。

## 四、生成核查初稿样例（答辩演示用）

1. 选 2—3 段录音（建议：一段第 1 组的大巴录音、一段第 4 组的费用纠纷、一段第 8 组的正常讲解），用全部最佳做法在网页上整理。
2. 演示人工复核：确认几条、修改一条、驳回一条（尤其第 8 组录音里的误报），再导出 Word 初稿和片段 ZIP。
3. 打开 Word 初稿检查：标题、声明、文件信息和 SHA-256、摘要、疑似片段表、全文时间轴、签字栏都在；标签都是"疑似·……"。

## 五、修问题

- 某组做法在完整流程里报错：让该组当场修（只改本组文件），重新提交。
- 自动测试失败：看 GitHub Actions 里的错误信息。
- 本周末各组报告定稿，放进 `reports\gN\`。

## 六、第 10 周发布

答辩后：合并最后的修改 → Releases 新建 `v1.0.0`（不带后缀，是正式版）→ 约 15 分钟后便携包自动附在 Release 上。这就是"别人下载就能用"的版本。
