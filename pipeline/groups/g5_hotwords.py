"""第 5 组：热词（槽位 hotword）

你们组负责什么
    识别模型不认识我们虚构的旅行社名、店名和云南的地名、行话，常常听成同音的别的字
    （"雾隐行舟旅行社"→"雾影行走旅行社"）。你们要提高这些"专名"的正确率，
    同时不能"过度纠正"：剧本里人物故意说错或简称的名字（见 data/hotword_variants.csv），
    工具应该照实记录，不能改成正确名称。
    基线做法在 pipeline/hotwords.py：识别后按拼音相似度替换（只用 3 个字以上的热词，默认关闭）。

在哪里改
    只改这个文件。下面的函数现在直接调用基线；把函数体换成你们的改进做法即可，
    函数名、参数、返回值的格式不要改：返回新的段落列表，text 和 text_raw 都要改，
    每次替换记进段落的 corrections（[{"from": 原来的字, "to": 热词, "start": 段落开始时间}]）。

怎么测（数据池路径换成老师给的）
    python tools/evaluate.py hotwords --pool 数据池路径 --out reports/g5
    python tools/compare.py --slot hotword --method g5 --metric hotwords --pool 数据池路径 --out reports/g5
    注意：基线只有在热词纠错打开时才工作（config.yaml 的 hotword.enabled，或网页上的"热词纠错"开关）。

注意
    - 热词表在 data/hotwords.txt（121 个），网页上可以临时编辑。
    - 专名正确率和过度纠正次数要一起报告，只看一个会误导。
    - 不联网，不用云端识别的热词功能。

可以试的方向（由易到难）
    1. 打开纠错，调 config.yaml 里 hotword 的 min_len、max_syllable_mismatch，看两个指标怎么变。
    2. 给常用词加"保护名单"，或者要求前后文也对得上才替换，减少过度纠正。
    3. 真正的解码时热词：python models/download_models.py --optional conformer_hotword，
       用 sherpa-onnx 的 transducer 模型和 hotwords_file 识别，和基线比较（见 docs/build_spec.md 5.3）。
"""
from pipeline.hotwords import hotword_baseline
from pipeline.methods import register


@register("hotword", "g5")
def hotword_g5(segments: list[dict], hotwords: list[str] | None, cfg: dict) -> list[dict]:
    """第 5 组的热词做法。hotwords 为 None 时用 data/hotwords.txt。"""
    # TODO 第 5 组：在这里写你们的改进。现在先直接用基线。
    return hotword_baseline(segments, hotwords, cfg)
