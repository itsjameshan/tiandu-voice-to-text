# 怎么交给 Claude Code 开工

## 方法一：GitHub 仓库 + 网页版 Claude Code（在云端搭建，推荐）

1. 把这些资料放进一个 GitHub 仓库（已经在仓库里的话跳过这一步）：网页上打开仓库 → "Add file → Upload files"，把文件和文件夹拖进去 → 提交。注意 `.gitignore` 是隐藏文件，Mac 的访达默认看不到，可能没传上去；没传上去也没关系，Claude Code 开工时会检查并补上。
2. 打开 claude.ai/code，选这个仓库，新建任务，粘贴下面的"开工指令"。私有仓库需要先给 Claude 的 GitHub 应用授权访问这个仓库；公开仓库可以直接用。
3. Claude Code 会在一个新分支上提交代码。做完后在网页上创建合并请求（Pull Request），确认后合并到 main。

## 方法二：在自己电脑上搭

Claude 桌面应用 → Code → 选择这个文件夹 → 本地会话，然后粘贴开工指令。代码直接生成在这个文件夹里。

## 开工指令（复制粘贴）

> 请先完整阅读 CLAUDE.md、docs/build_spec.md、docs/handoff_slim.md 第 5、7、11、13 节和 data/README.md。然后按 docs/build_spec.md 第 12 节的阶段顺序开工，从 P0 做起：每完成一个阶段就跑测试、提交、在 docs/progress.md 记录并简短汇报，然后继续下一阶段。全程遵守 CLAUDE.md 的红线：只用仓库里的虚构剧本和模型自带测试音频，不做语音合成，不调用云端识别接口，不开 share=True。遇到 build_spec 第 13 节列的事项（例如是否公网部署、用谁的账号）停下来问我。

## 搭好以后

- 在 Claude Code 的云端环境里能装好、跑通测试、启动服务自测，但别人打不开它；要给别人演示，需要部署到托管平台（见 `docs/build_spec.md` 第 10.2 节，公网上只能放虚构材料）。
- 在机房或单位内网使用：把仓库下载到那台电脑，按 README 安装运行，同一网络里的电脑用浏览器打开 `http://那台电脑的IP:7860`。
