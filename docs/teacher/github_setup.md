# GitHub 设置（老师，第 6 周前做一次）

目标：8 个组长能往仓库推送本组分支、发 Pull Request（PR）；**只有老师能批准和合并**；PR 必须通过自动测试才能合并。

> GitHub 网页的菜单名称偶尔会改版，下面写的是 2026 年 10 月的叫法；找不到时在设置页搜索框里搜关键词。

## 一、把组长加为协作者

1. 打开仓库页面 https://github.com/itsjameshan/tiandu-voice-to-text → 上方 **Settings**。
2. 左侧 **Collaborators**（或 Collaborators and teams）→ **Add people** → 输入组长的 GitHub 用户名 → 选中 → Add。
3. 组长邮箱会收到邀请，接受后才生效（[../guides/submit_code.md](../guides/submit_code.md) 第一次准备第 1 步）。
4. 8 个组长都加完后，在页面上确认 8 个人都是 "Collaborator"、不是 "Pending"。

> 协作者对仓库有写权限，理论上也能直接改主分支，所以一定要做第二步的保护。

## 二、保护主分支（main）

1. **Settings** → 左侧 **Rules** → **Rulesets** → **New ruleset** → **New branch ruleset**。
2. Ruleset Name 填 `保护主分支`；Enforcement status 选 **Active**。
3. **Bypass list**：点 Add bypass，把 **Repository admin**（就是老师自己）加进去，这样老师在紧急情况下仍能直接修复。
4. **Target branches** → Add target → **Include default branch**。
5. 勾选下面几项：
   - **Restrict deletions**（不能删主分支）
   - **Require a pull request before merging**，展开后：
     - Required approvals 填 **1**
     - 勾 **Require review from Code Owners**（仓库里的 `.github/CODEOWNERS` 写了 `* @itsjameshan`，所以只有老师的批准算数，组长之间互相批准无效）
     - 勾 **Dismiss stale pull request approvals when new commits are pushed**（批准后又改了代码要重新批准）
   - **Require status checks to pass**，点 Add checks，搜 `test` 选中自动测试的两项（`test (ubuntu-latest)`、`test (windows-latest)`）。这两项要等仓库里至少跑过一次自动测试后才能搜到。
   - **Block force pushes**
6. 点 **Create** 保存。

## 三、审核和合并 PR

1. 组长发 PR 后，老师会收到通知（GitHub 网页右上角铃铛和邮箱）。
2. 打开 PR → **Files changed** 标签：**先看改动的文件是不是只有本组文件**（`pipeline/groups/gN_*.py`、`reports/gN/`、`tests/test_gN_*.py`）。改了别的文件就请组长撤回那部分。
3. 看 PR 下方的自动测试是否都是绿色对勾。红色叉就请组长按错误信息修改。
4. 看代码和报告：是否遵守红线（只用虚构材料、没有联网调用、没有真实姓名）；对比数据是不是用冻结的数据池版本测的。
5. 没问题：右上角 **Review changes** → 选 **Approve** → Submit review → 回到 Conversation 标签点 **Merge pull request**（建议选 **Squash and merge**，主分支历史更清楚）。
6. 有问题：**Review changes** → 选 **Request changes**，写清楚要改什么。

组长登不上 GitHub、把文件交给老师代交时，见 [merging.md](merging.md)。

## 四、本仓库现在的分支

- 开发阶段的代码在分支 `claude/voice-recognition-deployment-rnnh1u` 上。老师确认后，在仓库页面点 **Compare & pull request**（或 Pull requests → New pull request，base 选 `main`，compare 选这个分支）建一个 PR，检查后合并到 `main`。**组长开始提交之前（第 7 周前）要先把它合并到 main**，组长都从 main 开始建自己的分支。
