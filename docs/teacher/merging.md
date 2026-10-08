# 合并学生代码与代交（老师）

## 一、正常情况：组长发 PR

按 [github_setup.md](github_setup.md) 第三节审核、合并即可。审核要点：

1. **只改了本组文件**（`pipeline/groups/gN_*.py`、`reports/gN/`、`tests/test_gN_*.py`）。
2. **自动测试全绿**（Windows 和 Linux 两项）。
3. **对比数据可信**：报告开头写了数据池版本（如 v1）、用的做法名；基线和改进用的是同一份数据。
4. **红线**：没有联网调用、没有真实姓名学号、没有把录音提交进来（`.wav`、`.m4a` 等不应该出现在改动里）。
5. 代码组员讲得清楚（答辩时再问）。

## 二、组长登不上 GitHub：老师代交

组长会交来一个文件夹，如 `g1-submit-1105`，里面是本组文件（结构见 [../guides/submit_code.md](../guides/submit_code.md) 方式二）。

**用 GitHub Desktop 代交**（老师电脑已登录 GitHub Desktop、已 clone 仓库）：

1. GitHub Desktop：Fetch origin → Current branch 选 `main` → Pull origin。
2. Current branch → New branch，名字用该组的分支名（如 `g1-denoise`；已经有了就直接选中）。
3. 把组长交来的文件夹里的文件，按相同的相对路径复制进仓库文件夹，覆盖旧文件。
4. 左边 Changes 里应该只有本组文件。Summary 写 `第 1 组：……（老师代交）`，Description 写组员学号后四位。Commit → Push origin → Create Pull Request。
5. 按第一节审核，没问题就合并。

**用命令行代交**（熟悉 git 的话）：

```bash
git switch main && git pull
git switch -c g1-denoise        # 已有分支就 git switch g1-denoise
# 复制文件……
git add pipeline/groups/g1_denoise.py reports/g1 tests/test_g1_*.py
git commit -m "第 1 组：……（老师代交）"
git push -u origin g1-denoise
```

然后在网页上建 PR、审核、合并。

## 三、两组改了同一个文件

正常情况下不会发生：每组只有自己的文件。第 1、2 组都和端点检测（vad 槽位）有关，但做法分别写在 `g1_denoise.py` 和 `g2_far_field.py` 里，登记的名字是 `g1`、`g2`，互不影响。第 6、7 组同理（`g6_classifier_a.py`、`g7_classifier_b.py`）。

如果某组确实需要改基线代码或公共文件（比如发现了基线的 bug），让组长在 PR 里单独说明，老师判断后自己改，或者单独开一个 PR。

## 四、训练好的模型文件

第 6、7 组训练的模型（`models/classifier_g6/`、`models/classifier_g7/`）不进仓库（`.gitignore` 已经忽略 `models/`）。组长用 U 盘交给老师，老师放进自己便携包的 `models\` 下；第 9 周串联时网页工具就能选 g6、g7 做法。
