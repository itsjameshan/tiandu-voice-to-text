"""各组的改进做法：每组一个文件，导入时把自己的做法登记到对应槽位。

    g1_denoise.py        第 1 组  降噪、端点检测（denoise、vad）
    g2_far_field.py      第 2 组  远距离增强、端点检测（enhance、vad）
    g3_diarize.py        第 3 组  说话人分离（diarize）
    g4_numbers.py        第 4 组  数字规范化（normalize）
    g5_hotwords.py       第 5 组  热词纠错（hotword）
    g6_classifier_a.py   第 6 组  话术分类模型 A（classify）
    g7_classifier_b.py   第 7 组  话术分类模型 B（classify）
    g8_clips.py          第 8 组  疑似片段（clips）

每组只改自己的文件。这些文件在导入时不能加载 sherpa-onnx、gradio、tensorflow 等重依赖
（需要时在函数里面导入），这样只装了 TensorFlow 的电脑也能运行训练脚本。

这个包的 __init__ 不导入各组文件：导入哪个组文件，就只加载那个组需要的东西
（例如第 6 组训练模型时只需要 TensorFlow 和 numpy，不需要第 4 组用的 cn2an）。
各组的做法由 pipeline.methods.load_all() 按 GROUP_MODULES 逐个导入登记。
"""

GROUP_MODULES = (
    "pipeline.groups.g1_denoise",
    "pipeline.groups.g2_far_field",
    "pipeline.groups.g3_diarize",
    "pipeline.groups.g4_numbers",
    "pipeline.groups.g5_hotwords",
    "pipeline.groups.g6_classifier_a",
    "pipeline.groups.g7_classifier_b",
    "pipeline.groups.g8_clips",
)
