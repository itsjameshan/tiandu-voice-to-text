## 旅游纠纷录音材料整理（教学原型）· Windows 便携包

下载下面的 `tiandu-portable-win64-*.zip`（约 400 MB），拷进 U 盘，解压到**纯英文路径**（目标文件夹填 `D:\asr`，压缩包里自带一层 `tiandu` 文件夹，解压后就是 `D:\asr\tiandu`），双击 `selfcheck.bat` 自检，再双击 `start.bat` 启动，浏览器会自动打开。不需要联网、不需要安装 Python、不需要管理员权限。

- 包含：Python 3.11 嵌入版、全部依赖、识别（SenseVoice）、端点检测（Silero VAD）和说话人分离（pyannote + CAM++）模型、ffmpeg、项目代码和中文文档；不含 TensorFlow（第 6、7 组训练模型用机房自带的 Python）。
- 网页工具五个页面：整理录音（上传 → 时间轴文字稿、说话人、疑似片段 → 人工复核 → 导出 Word 初稿、表格、片段 ZIP）、剧本文本演示、数据校对（识别结果和参考文本逐段对照、点行听原声、保存校对）、录音质检（三张图 + 五项检查）、使用说明。
- 命令行工具：数据池入池、导出参考文本、验收、冻结版本（`tools\ingest_pool.py` 等），测评（`tools\evaluate.py`：字错率、专名、数字、说话人、话术分类、疑似片段）和对比表（`tools\compare.py`）。
- 本包在 GitHub 的 Windows 机器上构建，并用模型自带的测试音频跑完整流程自检通过；机房电脑请老师在开课前实测一次（见 `docs\teacher\week0_checklist.md`）。
- 只能处理按虚构剧本录的录音；识别可能有误，所有标注均为疑似、待核查，必须人工复核。

详细说明见仓库首页 README。
