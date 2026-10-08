# 组长提交代码指南（第 7 周起）

只有**组长**需要做这件事。其他同学不需要 GitHub 账号。

每组只改自己的这几个文件（"本组文件"）：

| 组 | 本组文件 |
|---|---|
| 第 1 组 | `pipeline/groups/g1_denoise.py`、`reports/g1/`、`tests/test_g1_*.py`（可选） |
| 第 2 组 | `pipeline/groups/g2_far_field.py`、`reports/g2/`、`tests/test_g2_*.py`（可选） |
| 第 3 组 | `pipeline/groups/g3_diarize.py`、`reports/g3/`、`tests/test_g3_*.py`（可选） |
| 第 4 组 | `pipeline/groups/g4_numbers.py`、`reports/g4/`、`tests/test_g4_*.py`（可选） |
| 第 5 组 | `pipeline/groups/g5_hotwords.py`、`reports/g5/`、`tests/test_g5_*.py`（可选） |
| 第 6 组 | `pipeline/groups/g6_classifier_a.py`、`reports/g6/`、`tests/test_g6_*.py`（可选） |
| 第 7 组 | `pipeline/groups/g7_classifier_b.py`、`reports/g7/`、`tests/test_g7_*.py`（可选） |
| 第 8 组 | `pipeline/groups/g8_clips.py`、`reports/g8/`、`tests/test_g8_*.py`（可选） |

> 训练好的模型文件（`models/classifier_g6/` 等）很大，**不要提交**，交给老师用 U 盘拷。

有两种提交方式，**能用 GitHub 就用方式一，登不上就用方式二**，两种都算数。

---

## 方式一：GitHub Desktop（图形界面，不用命令行）

### 第一次准备（只做一次，约 20 分钟，需要联网）

1. **注册 GitHub 账号**：浏览器打开 https://github.com ，点 Sign up，用邮箱注册。用户名用英文，**不要用真实姓名**（例如 `tiandu-g1-lead`）。把用户名告诉老师，老师会把你加为仓库协作者；你的邮箱里会收到邀请邮件，点里面的 **Accept invitation** 接受。
2. **安装 GitHub Desktop**：老师 U 盘里有安装包（`GitHubDesktopSetup-x64.exe`），或从 https://desktop.github.com 下载。双击安装。
3. **登录**：打开 GitHub Desktop → File（文件）→ Options（选项）→ Accounts（账户）→ Sign in（登录），浏览器里确认授权。
   > **仓库是公开的，提交记录谁都能看到**：每次提交会记下作者名字和邮箱。先在 GitHub 网页右上角头像 → Settings → Emails 里勾选 **Keep my email addresses private**，记下下面显示的 `……@users.noreply.github.com` 邮箱；再在 GitHub Desktop → File → Options → Git 里，Name 填你的英文用户名（不要填真实姓名），Email 选这个 noreply 邮箱。
4. **把仓库下载到电脑（clone）**：File → Clone repository → 选 **URL** 标签 → 填 `https://github.com/itsjameshan/tiandu-voice-to-text` → Local path 选一个**纯英文路径**，如 `D:\asr\repo` → Clone。
   > 机房电脑会还原的话，Local path 选 U 盘（如 `E:\asr\repo`）。

### 每次提交（约 5 分钟）

1. **先同步最新内容**：GitHub Desktop 顶部点 **Fetch origin**，如果出现 **Pull origin** 就再点一下。
2. **切到本组分支**（第一次要新建）：点顶部 **Current branch** → **New branch** → 名字按下表填 → Create branch。以后每次都先在 Current branch 里选中它。

   | 组 | 分支名 |
   |---|---|
   | 第 1 组 | `g1-denoise` |
   | 第 2 组 | `g2-far-field` |
   | 第 3 组 | `g3-diarize` |
   | 第 4 组 | `g4-numbers` |
   | 第 5 组 | `g5-hotwords` |
   | 第 6 组 | `g6-classifier-a` |
   | 第 7 组 | `g7-classifier-b` |
   | 第 8 组 | `g8-clips` |

3. **把本组文件复制进来**：大家平时在便携包文件夹（如 `D:\asr\tiandu`）里改代码、跑测评。提交时，把便携包里的**本组文件**（上表）复制到仓库文件夹（如 `D:\asr\repo`）里**相同的位置**，覆盖旧文件。只复制本组文件，别的文件不要动。
4. **检查改动**：回到 GitHub Desktop，左边 Changes 列表里应该**只有本组文件**。如果出现了别的文件，右键 → Discard changes（放弃修改）。
5. **提交（commit）**：左下角 Summary 填一句中文说明，例如"第 1 组：加入谱减法降噪，对比报告见 reports/g1"；Description 里写组员的 GitHub 用户名或学号后四位（**不写真实姓名**）。点 **Commit to g1-denoise**。
6. **上传（push）**：点顶部 **Push origin**（第一次是 **Publish branch**）。
7. **发 Pull Request**：点 **Create Pull Request**（或顶部 Branch → Create pull request），浏览器会打开 GitHub 页面。
   - 标题：`第 1 组：……`
   - 内容按模板填：做了什么、对比数据（链接到 `reports/g1/` 里的报告）、自己测过没有、是否只改了本组文件、是否遵守红线。
   - 点 **Create pull request**。
8. **等自动测试和老师审核**：PR 页面下方会显示"自动测试"在 Windows 和 Linux 上运行，约 10—20 分钟。
   - 绿色对勾：通过，等老师审核；
   - 红色叉：点 Details 看哪里出错，改好后重复第 3—6 步（同一个 PR 会自动更新，不用重新发）；
   - 老师留言要求修改：同样改好后重复第 3—6 步。
9. 老师点 Merge 后，本组的改进就进入主分支了。

### 常见问题

- **Push 时提示没有权限**：确认已经接受了老师的协作邀请（邮箱里的 Accept invitation）。
- **登录页面打不开、一直转圈**：网络访问 GitHub 不稳定，换个时间或换个网络再试；实在不行用方式二。
- **Changes 里出现几百个文件**：多半是复制了整个便携包。点 Repository → Discard all changes，重新只复制本组文件。

---

## 方式二：交给老师代交

登不上 GitHub、或者网络不行时用这种方式，效果完全一样。

1. 把**本组文件**（上表）连同文件夹结构一起复制到 U 盘的一个文件夹里，文件夹名用 `g1-submit-日期`，例如 `g1-submit-1105`：

   ```
   g1-submit-1105\
     pipeline\groups\g1_denoise.py
     reports\g1\（整个文件夹）
     tests\test_g1_denoise.py（如果有）
     说明.txt   ← 写：做了什么、对比结果在哪、组员的学号后四位
   ```

2. 交给老师。老师会替你们提交，提交说明里写上"第 1 组"和组员学号后四位。
3. 老师审核有意见时会告诉组长，改好后再交一次。
