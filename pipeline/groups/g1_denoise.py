"""第 1 组：降噪与端点检测（槽位 denoise、vad）

你们组负责什么
    流水线第 2 步：先把录音里的噪声降下来（denoise），再切出有人声的段落（vad）。
    基线做法在 pipeline/step2_vad.py：降噪什么都不做；端点检测用 Silero VAD。

在哪里改
    只改这个文件。下面两个函数现在直接调用基线，所以全流程从第一天就能跑通。
    把函数体换成你们的改进做法即可；函数名、参数、返回值的格式不要改。

怎么测（数据池路径换成老师给的，如 D:\\data_pool）
    python tools/compare.py --slot denoise --method g1 --metric cer --pool 数据池路径 --out reports/g1/denoise
    python tools/compare.py --slot vad --method g1 --metric cer --pool 数据池路径 --out reports/g1/vad
    （两个对比放不同的文件夹，否则后一个会盖掉前一个的 compare_cer.md）
    网页"整理录音"页的"高级设置"里把"降噪"或"端点检测"选成 g1，也能直接看效果。

注意
    - 降噪后采样点个数不能变（时间戳要和原录音对得上），程序会检查。
    - 降噪不一定让识别变好，一定要用字错率验证，安静（Q）、嘈杂（N）、口袋/远距离（F）分开看。
    - 不联网，不用语音合成。

可以试的方向（由易到难）
    1. 调 config.yaml 里 vad 一节的参数（threshold、min_silence_duration……），看会不会漏切人声或多切。
    2. 用模板自带的 noisereduce 做法（--method denoise=noisereduce），再在这里调它的参数
       （例如 noisereduce.reduce_noise 的 prop_decrease、stationary）。
    3. 自己写谱减法：用录音开头一小段没人说话的声音估计噪声频谱，再从每一帧里减掉。
    4. 先估计信噪比，只对噪声大的录音降噪。
"""
import numpy as np

from pipeline.methods import register
from pipeline.step2_vad import denoise_baseline, vad_baseline


@register("denoise", "g1")
def denoise_g1(samples: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """第 1 组的降噪做法。输入、输出都是 16000Hz 单声道 float32 采样，长度必须相同。"""
    # TODO 第 1 组：在这里写你们的降噪方法。现在先直接用基线（不降噪）。
    return denoise_baseline(samples, sr, cfg)


@register("vad", "g1")
def vad_g1(samples: np.ndarray, sr: int, cfg: dict) -> list[tuple[float, float]]:
    """第 1 组的端点检测做法。返回 [(开始秒, 结束秒), ...]，按时间先后排列。"""
    # TODO 第 1 组：在这里改端点检测。现在先直接用基线（Silero VAD）。
    return vad_baseline(samples, sr, cfg)
