"""第 4 组：数字、证号、金额规范化（槽位 normalize）

你们组负责什么
    流水线第 5 步：把口语里的数字写规范（"两千八百块"→"2800元"、"零八七一 零零零零 六六八八"→"0871-0000-6688"），
    并提取出金额、电话、导游证号、合同号、订单号、时刻、日期，放进段落的 numbers；
    旅行社名、店名放进 entities。
    基线做法在 pipeline/step5_normalize.py（正则 + cn2an + 少量规则），模块开头列了它已知的问题。

在哪里改
    只改这个文件。下面的函数现在直接调用基线；把函数体换成你们的改进做法即可，
    函数名、参数、返回值的格式不要改：返回新的段落（不要改传进来的那个），
    mode="spoken" 时输入是汉字读法（剧本、测评模式），要先转数字；mode="display" 时识别已经转好了，只需提取。

怎么测（不需要录音，用剧本每句的标准答案 numbers 字段）
    python tools/evaluate.py numbers --out reports/g4                             （基线）
    python tools/evaluate.py numbers --method normalize=g4 --out reports/g4/after （你们的做法，放子文件夹，不盖掉基线）
    python tools/compare.py --slot normalize --method g4 --metric numbers --out reports/g4
    网页"剧本文本演示"页也能直接看每句提取出了什么。

注意
    - 主指标是金额、电话、证号、合同号、订单号、时刻、日期；数量、时长是次要指标，分开报告。
    - 正确率（提取出来的有多少对）和召回率（标准答案有多少被提取出来）都要看。
    - 不联网。

可以试的方向（编号不代表难度：最容易上手的是方向 2 的固定说法和方向 1 里补时间词那一步；基线在剧本上时刻的召回率只有 0.14，方向 1 提升空间最大）
    1. 没有"上午/下午"等时间词的时刻（"九点五十""十一点二十"）：根据前后文判断要不要转成时刻、是上午还是下午。
    2. "两个半小时""半个小时""三四十块"这类说法。
    3. 识别结果没有空格时，连续 7 个以上逐位读的数字（零一二……幺）很可能是电话或编号。
    4. 把房间号、足银"九九九"这类不是金额的数排除掉。
"""
from pipeline.methods import register
from pipeline.step5_normalize import normalize_baseline


@register("normalize", "g4")
def normalize_g4(segment: dict, mode: str, cfg: dict) -> dict:
    """第 4 组的规范化做法。mode 是 "spoken"（汉字读法）或 "display"（识别已转好数字）。"""
    # TODO 第 4 组：在这里写你们的改进。现在先直接用基线。
    return normalize_baseline(segment, mode, cfg)
