"""第 2 组：远距离、口袋录音（槽位 enhance、vad）

你们组负责什么
    手机放在口袋里或离说话人 2—3 米远时，声音变小、变闷（高频损失），识别会明显变差。
    你们先测出"安静（Q）、嘈杂（N）、口袋/远距离（F）"三种条件的字错率差距，再想办法缩小 F 条件的差距。
    基线做法在 pipeline/step2_vad.py：远距离增强什么都不做；端点检测用 Silero VAD。

在哪里改
    只改这个文件。下面两个函数现在直接调用基线；把函数体换成你们的改进做法即可，
    函数名、参数、返回值的格式不要改。

怎么测（数据池路径换成老师给的）
    python tools/evaluate.py cer --pool 数据池路径 --out reports/g2          （看三种条件的差距）
    python tools/compare.py --slot enhance --method g2 --metric cer --pool 数据池路径 --out reports/g2
    python tools/compare.py --slot vad --method g2 --metric cer --pool 数据池路径 --out reports/g2/vad
    （vad 的对比放子文件夹，否则会盖掉 enhance 对比的 compare_cer.md）

注意
    - 增强后采样点个数不能变（时间戳要和原录音对得上），程序会检查。
    - 改进可能让 F 变好、却让 Q 变差，三种条件都要报告。
    - 不联网，不用语音合成。

可以试的方向（由易到难）
    1. 音量归一化：把整段录音的音量调到差不多 -20 dBFS（注意不要削波）。
    2. 提升高频（预加重或简单的均衡），弥补口袋录音"发闷"。
    3. 远距离录音人声能量低，试试调低端点检测的 threshold，减少漏切。
    4. 动态范围压缩：把小声的部分放大、大声的部分压一压。
"""
import numpy as np

from pipeline.methods import register
from pipeline.step2_vad import enhance_baseline, vad_baseline


@register("enhance", "g2")
def enhance_g2(samples: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """第 2 组的远距离增强做法。输入、输出都是 16000Hz 单声道 float32 采样，长度必须相同。"""
    # TODO 第 2 组：在这里写你们的增强方法。现在先直接用基线（不处理）。
    return enhance_baseline(samples, sr, cfg)


@register("vad", "g2")
def vad_g2(samples: np.ndarray, sr: int, cfg: dict) -> list[tuple[float, float]]:
    """第 2 组的端点检测做法。返回 [(开始秒, 结束秒), ...]，按时间先后排列。"""
    # TODO 第 2 组：在这里改端点检测（例如为远距离录音调参数）。现在先直接用基线。
    return vad_baseline(samples, sr, cfg)
