# 识别用 sherpa-onnx + SenseVoice（2024-07-17 版），不依赖 PyTorch

全流程（端点检测、识别、说话人分离）用 sherpa-onnx 离线运行 ONNX 模型，识别模型用 SenseVoice int8 的 2024-07-17 版。这样不需要 PyTorch，在机房普通 Windows 电脑的 CPU 上能跑，数据不出本机；2026-10-07 已在云端容器实测可用。2025-09-09 版名字像新版，实际是粤语模型，不要用。sherpa-onnx 里 SenseVoice 不支持解码时热词，所以热词基线用识别后的拼音相似度纠错，真正的热词方案留给第 5 组对比。TensorFlow 只用在第 6、7 组的话术分类（和选做的数字识别小实验）。
