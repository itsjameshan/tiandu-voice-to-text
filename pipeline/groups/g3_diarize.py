"""第 3 组：说话人分离（槽位 diarize）

你们组负责什么
    流水线第 4 步：区分"谁在说话"，给每一段话配上说话人（说话人1、说话人2……），
    之后由使用者在界面上把它们对应成导游、游客等。
    基线做法在 pipeline/step4_diarize.py：pyannote 分割 + CAM++ 声纹 + 聚类，按重叠时长给每段配说话人。

在哪里改
    只改这个文件。下面的函数现在直接调用基线；把函数体换成你们的改进做法即可，
    函数名、参数、返回值的格式不要改：返回的每一段都要有 speaker_id（1 起的整数或 None）
    和 speaker（"说话人N"或"未知"）。

怎么测（需要人工标注的说话人时间，见 docs/guides/annotation.md）
    python tools/evaluate.py speakers --pool 数据池路径 --speakers auto --out reports/g3/auto   （人数设自动）
    python tools/evaluate.py speakers --pool 数据池路径 --speakers ref --out reports/g3/ref     （人数设对）
    python tools/compare.py --slot diarize --method g3 --metric speakers --pool 数据池路径 --speakers ref --out reports/g3

注意
    - 知道人数时一定要设人数（config.yaml 的 diarize.num_speakers，或网页上的"说话人数"）。
      实测四人测试音频在"自动"下会多分出人来。
    - 已知问题：端点检测切出来的一段话里如果中间换了人（没有停顿），整段只会算给一个人。
    - 不联网，不建声纹库，不识别"是谁"，只区分"不是同一个人"。

可以试的方向（由易到难）
    1. 比较"自动"和"设对人数"的说话人标错比例；调 config.yaml 里 diarize 的 threshold、min_duration_on/off。
    2. 按说话人边界再切分：如果一段话和两个说话人的时间都重叠很多，就在边界处切成两段。
    3. 换声纹模型：python models/download_models.py --optional eres2net，比较 CAM++ 和 ERes2Net。
"""
import numpy as np

from pipeline.methods import register
from pipeline.step4_diarize import diarize_baseline


@register("diarize", "g3")
def diarize_g3(samples: np.ndarray, sr: int, segments: list[dict], cfg: dict) -> list[dict]:
    """第 3 组的说话人分离做法：给每一段填 speaker_id 和 speaker，返回新的段落列表。"""
    # TODO 第 3 组：在这里写你们的改进。现在先直接用基线。
    return diarize_baseline(samples, sr, segments, cfg)
