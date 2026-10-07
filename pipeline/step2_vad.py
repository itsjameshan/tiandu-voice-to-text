"""步骤 2：降噪、远距离增强、端点检测。

这一步有三个槽位，按顺序执行：
  1. 降噪（denoise）：把背景噪声压低一些；
  2. 远距离增强（enhance）：给手机放口袋里、离得远的录音加大音量、调均衡；
  3. 端点检测（vad）：找出录音里"哪几段有人在说话"，只把这些段落交给下一步识别。
detect_speech() 把三步串起来，返回处理后的采样和段落列表（只有开始、结束时间，文字在步骤 3 填）。

基线做法：
    - 降噪 baseline：原样返回，不做任何处理（降噪不一定让识别更准，要用字错率验证后才能默认开启）；
    - 降噪 noisereduce：用 noisereduce 库的 reduce_noise 做谱门限（spectral gating）降噪（备选做法，默认不用）；
    - 远距离增强 baseline：原样返回；
    - 端点检测 baseline：Silero VAD（sherpa-onnx 提供），参数在 config.yaml 的 vad 下面：
        threshold             判断"有人声"的把握（0 到 1），越大越严格、切得越碎，默认 0.5；
        min_silence_duration  停顿超过多少秒才算一段话结束，越大段落越长、段数越少，默认 0.3；
        min_speech_duration   连续有人声超过多少秒才开始算一段话，默认 0.25；
        max_speech_duration   一段话超过多少秒后，就在较短的停顿处切开，默认 15
                              （不是严格的上限：超过这个时长后会在更短的停顿处切开，但没有停顿时段落可能明显更长，实测设 3 秒时最长约 8.9 秒）。
      录音按 512 个采样（0.032 秒）一小块喂给检测器，每检测完一段话就取出来。
可改进方向：
    第 1 组（pipeline/groups/g1_denoise.py）：换更好的降噪做法，调端点检测参数，减少漏切人声；
    第 2 组（pipeline/groups/g2_far_field.py）：给口袋、远距离录音做增益、均衡等增强。
    注意：降噪和增强必须返回和输入一样长的采样，否则时间轴会和原录音对不上。
测评指标：
    降噪、增强前后的字错率（按安静 Q、嘈杂 N、口袋/远距离 F 三种录音条件分开比较）；
    端点检测漏切人声（该切出来的话没切出来）的多少。
"""
import logging
import time
from functools import lru_cache

import numpy as np

from pipeline.audio import SR, has_non_ascii
from pipeline.methods import get_method, load_all, register
from pipeline.models import model_path
from pipeline.schema import new_segment

logger = logging.getLogger(__name__)

# 本步骤的三个槽位，按执行顺序排列
STEP_SLOTS = ("denoise", "enhance", "vad")

# 检测器里能暂存多少秒的声音（照 sherpa-onnx 的示例代码写 60 秒）
VAD_BUFFER_SECONDS = 60


# ---------- 降噪 ----------

@register("denoise", "baseline")
def denoise_baseline(samples: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """基线降噪：原样返回，不做处理。"""
    return samples


@register("denoise", "noisereduce")
def denoise_noisereduce(samples: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """用 noisereduce 库降噪（谱门限）：先估计每个频带的噪声水平，再把低于这个门限的时间—频率点调小。"""
    import noisereduce

    cleaned = noisereduce.reduce_noise(y=samples, sr=sr)
    # 全静音时 noisereduce 会除以 0 算出 NaN，NaN 会让端点检测误以为有人声，所以换成 0
    return np.nan_to_num(np.asarray(cleaned, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# ---------- 远距离增强 ----------

@register("enhance", "baseline")
def enhance_baseline(samples: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """基线增强：原样返回，不做处理。"""
    return samples


# ---------- 端点检测 ----------

@lru_cache(maxsize=None)
def _vad_config(model: str, threshold: float, min_silence_duration: float,
                min_speech_duration: float, max_speech_duration: float):
    """做好一份 Silero VAD 的配置。同样的参数只做一次，以后直接用记下来的那份。

    注意：这里只记住"配置"，不记住检测器本身。检测器会记住之前听过的声音，
    如果两次调用共用一个检测器，第二次的时间会接着第一次往后算，所以每次都要新建检测器。
    """
    import sherpa_onnx

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = model
    config.silero_vad.threshold = threshold
    config.silero_vad.min_silence_duration = min_silence_duration
    config.silero_vad.min_speech_duration = min_speech_duration
    config.silero_vad.max_speech_duration = max_speech_duration
    config.sample_rate = SR
    return config


def _take_finished(detector, sr: int) -> list[tuple[float, float]]:
    """把检测器里已经检测完的话全部取出来，返回 [(开始秒, 结束秒)]。"""
    spans = []
    while not detector.empty():
        speech = detector.front  # speech.start 是第几个采样，speech.samples 是这段话的采样
        start = speech.start / sr
        end = (speech.start + len(speech.samples)) / sr
        spans.append((round(start, 2), round(end, 2)))
        detector.pop()
    return spans


@register("vad", "baseline")
def vad_baseline(samples: np.ndarray, sr: int, cfg: dict) -> list[tuple[float, float]]:
    """基线端点检测：Silero VAD。返回每段话的 [(开始秒, 结束秒)]，保留 2 位小数，按时间先后排列。"""
    import sherpa_onnx

    if sr != SR:
        raise ValueError(f"端点检测需要 {SR} Hz 的录音，现在是 {sr} Hz，请先用步骤 1 转换格式")
    model = model_path(cfg, "silero_vad")
    if not model.is_file():
        raise FileNotFoundError(f"找不到端点检测模型：{model}。请先运行 python models/download_models.py 下载模型")
    if has_non_ascii(model):
        logger.warning("模型路径里有中文等非英文字符，Windows 上可能加载失败，建议把程序放到纯英文路径：%s", model)

    params = cfg["vad"]
    config = _vad_config(
        str(model),
        float(params["threshold"]),
        float(params["min_silence_duration"]),
        float(params["min_speech_duration"]),
        float(params["max_speech_duration"]),
    )
    detector = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=VAD_BUFFER_SECONDS)

    samples = np.asarray(samples, dtype=np.float32)
    window = config.silero_vad.window_size  # 每次喂 512 个采样
    spans = []
    for i in range(0, len(samples), window):
        detector.accept_waveform(samples[i:i + window])
        spans.extend(_take_finished(detector, sr))
    detector.flush()  # 录音结束：最后一段话即使没有停顿也要取出来
    spans.extend(_take_finished(detector, sr))
    return spans


# ---------- 把三步串起来 ----------

def detect_speech(samples: np.ndarray, sr: int, cfg: dict,
                  methods: dict | None = None) -> tuple[np.ndarray, list[dict]]:
    """步骤 2：依次 降噪 → 远距离增强 → 端点检测。

    methods：{槽位: 函数}，可以只给其中几个槽位（测评对比、测试时用）；
             没给的槽位按 cfg["methods"] 选做法，那里也没写就用 baseline。
    返回 (处理后的采样, 段落列表)。段落只有开始、结束时间，说话人"未知"，文字留空。
    """
    methods = methods or {}
    load_all()  # 让各组文件里的做法都登记进来
    chosen = cfg.get("methods") or {}
    funcs = {}
    for slot in STEP_SLOTS:
        if slot in methods:
            funcs[slot] = methods[slot]
        else:
            funcs[slot] = get_method(slot, chosen.get(slot) or "baseline")

    started = time.perf_counter()
    samples = np.asarray(samples, dtype=np.float32)
    length = len(samples)
    for slot in ("denoise", "enhance"):
        samples = np.asarray(funcs[slot](samples, sr, cfg), dtype=np.float32)
        if len(samples) != length:
            raise ValueError(
                f"槽位“{slot}”的做法把录音长度从 {length} 个采样改成了 {len(samples)} 个。"
                f"降噪和增强必须返回同样长度的采样，否则时间轴会和原录音对不上"
            )

    spans = funcs["vad"](samples, sr, cfg)
    segments = [new_segment(start, end) for start, end in spans]

    seconds = time.perf_counter() - started
    duration = length / sr if sr else 0.0
    rtf = seconds / duration if duration else 0.0
    logger.info("步骤 2 完成：录音 %.1f 秒，找到 %d 段话，用时 %.2f 秒（实时率 %.3f）",
                duration, len(segments), seconds, rtf)
    return samples, segments
