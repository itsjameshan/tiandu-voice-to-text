# Windows 便携包（老师）

## 一、是什么

一个 zip 压缩包（约 400 MB，解压后约 600 MB），解压就能用：

- Python 3.11 嵌入版（不用安装、不写注册表、不需要管理员权限）；
- 全部依赖（Gradio 网页界面、sherpa-onnx 识别引擎等），**不含 TensorFlow**；
- 必需的模型（识别、端点检测、说话人分离）和模型自带的测试音频（只给自检用）；
- ffmpeg（格式转换）；
- 项目代码、数据（虚构剧本）、文档；
- `start.bat`（启动网页工具）、`selfcheck.bat`（自检）、`README.txt`（中文使用说明）。

便携包是 GitHub 在一台真实的 Windows 电脑上自动构建的，构建完会用测试音频跑一遍完整流程，**自检通过才会生成 zip**。

## 二、去哪里下载

- **正式发布的版本**：仓库页面右侧 **Releases** → 选最新版本 → 下面的 Assets 里点 `tiandu-portable-win64-版本号.zip`。不需要登录。
  地址：https://github.com/itsjameshan/tiandu-voice-to-text/releases
- **每次改动后的测试版**：仓库上方 **Actions** → 左边"构建 Windows 便携包" → 点最新一次运行 → 页面最下面 Artifacts 里的 `tiandu-portable-win64`（需要登录 GitHub，保留 30 天）。

国内下载 GitHub 文件有时很慢，可以用手机热点或换个时间；下载一次后拷进 U 盘，在机房里传播就不需要网络了。

## 三、怎么用（学生也看 README.txt）

1. 解压到**纯英文路径**，如 `D:\asr\tiandu`、`E:\asr\tiandu`（U 盘）。不要放在桌面或"文档"里（路径里有中文用户名）。
2. 第一次先双击 `selfcheck.bat`，看到"全部通过"。
3. 双击 `start.bat`，浏览器自动打开 `http://127.0.0.1:7860`。
4. 关闭黑色窗口就退出。

命令行用法（在解压出来的文件夹里打开命令行）：

```bat
python\python.exe tools\selfcheck.py
python\python.exe tools\evaluate.py --help
python\python.exe tools\ingest_pool.py --pool D:\data_pool
```

## 四、需要 TensorFlow 时（第 6、7 组训练分类模型）

便携包不带 TensorFlow（太大，约 1 GB）。按顺序试：

1. **用机房自带的 Python**（老师说机房已装 TensorFlow）。训练脚本只需要 TensorFlow 和 numpy：在便携包文件夹里运行
   ```bat
   python tools\train_classifier.py --model g6 --eval logo
   ```
   注意这里是机房自带的 `python`，不是 `python\python.exe`。训练出的模型在 `models\classifier_g6\`，便携包的网页工具会自动使用它——但网页工具要用到 TensorFlow 才能加载模型；没有 TensorFlow 时会退回关键词规则，界面上有提示。
2. **在有网的电脑上给便携包装 TensorFlow**：在一台联网的 Windows 电脑上装好 Python 3.11，然后在便携包文件夹里运行
   ```bat
   py -3.11 -m pip install --target python\Lib\site-packages tensorflow-cpu
   ```
   装完后整个文件夹约 1.6 GB，再拷到机房。

## 五、怎么发布新版本（课程中、课程结束后）

各组的改进合并进 `main` 以后（第 9 周），想让全班用上新代码：

1. 仓库页面 → 右侧 **Releases** → **Draft a new release**；
2. **Choose a tag** 填一个新版本号，如 `v0.3.0`（课程中的测试版加后缀，如 `v0.3.0-week9`，会标为"预发布"）；Target 选 `main`；
3. 点 **Publish release**。
4. 等约 15 分钟，"构建 Windows 便携包"自动运行完，zip 会自动出现在这个 Release 的 Assets 里。

课程结束、答辩后发布 `v1.0.0`，就是"交给别人用"的版本。

也可以不发 Release，只手动构建一次测试版：**Actions** → "构建 Windows 便携包" → **Run workflow**。
