"""AI 合成测评录音：把剧本台词合成成"录音"，并写出标准答案（见 docs/adr/0006-ai-synthesized-recordings.md）。

老师 2026-10-14 决定：72 段测评录音（24 个剧本 × 安静 Q、嘈杂教室 N、口袋或远距离 F）全部由 AI 合成。
怎么合成：
    1. 每个角色一个合成声音（data/tts_voice_plan.csv，可以改）；
    2. 每句台词的舞台说明（direction，如"与上一句部分重叠""用车载扩音器""离手机较远"）变成声音效果；
    3. 按 Q / N / F 渲染整段录音，开头加一句清楚的声明 AI_DISCLAIMER（"以下为人工智能合成的虚构教学录音……"）；
    4. 同时得到标准答案：参考文本、说话人时间、疑似片段起止（连续几句同一个疑似类别的台词连成一个片段）。
本文件只用 numpy、scipy（在函数里导入），不加载语音合成模型：合成函数 tts(text, sid, speed) -> (采样, 采样率)
由 tools/synthesize_recordings.py 加载 Kokoro 模型后传进来；测试时传一个假的合成函数。

基线做法：
    关键词 → 效果参数的简单规则；嘈杂声用别的剧本的合成台词叠成"人声嘈杂"；口袋用低通滤波，远距离用人工混响。
可改进方向（第 1、2 组可以试）：
    更真实的嘈杂声和房间混响、更多说话风格（情绪、口音）、按剧本场景换背景声。
测评指标：
    用 SenseVoice 回听的字错率检查合成得清不清楚（tools/synthesize_recordings.py --check）。
    合成录音上的测评结果不代表真人录音的效果，报告里要写明。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np

from pipeline.data import AI_DISCLAIMER, AI_DISCLAIMER_SPEAKER, FLAG_LABELS, script_reference_text

SR = 24000  # 输出采样率（Kokoro 的原生采样率），入池时会转成 16000 Hz

# ---------------- 舞台说明 → 效果 ----------------

OVERLAP_WORDS = ("与上一句部分重叠", "与上一句末尾部分重叠", "打断", "几个人同时说", "与周围游客的问好声部分重叠")
QUIET_WORDS = ("小声", "压低", "声音很轻", "低声", "嘀咕", "声音不大", "含糊")
LOUD_WORDS = ("提高音量", "提高声音", "声音稍大", "没压住", "大声")
NEAR_WORDS = ("靠近手机", "离手机很近", "离手机较近", "离录音手机较近", "凑近手机")
FAR_WORDS = ("离手机较远", "离录音手机较远", "声音较远", "站在门口", "背对手机", "房间另一头", "展柜前", "在车头",
             "在驾驶座", "从手机旁经过", "远处", "走到一旁打电话", "探出头")
MEGAPHONE_WORDS = ("扩音器", "话筒", "小喇叭")
MEGAPHONE_OFF = ("放下扩音器", "未用扩音器")
PHONE_WORDS = ("电话外放", "免提", "从免提里传出", "发语音")
ECHO_WORDS = ("隧道", "回声")
FAST_WORDS = ("语速较快", "语速快", "快步", "有点喘")
SLOW_WORDS = ("语速较慢", "语速放慢", "语气放慢")
SCENE_JUMP_WORDS = ("过了一会儿", "又过了一段时间", "分钟后", "一小时后", "半分钟后")
# 一次性的环境声：关键词 → 声音名（见 make_event）
EVENT_WORDS = {"提示音": "beep", "尖啸": "squeal", "咔哒": "click", "敲": "clink", "颠簸": "thump",
               "减速带": "thump", "关车门": "thump", "鸟叫": "chirp"}

_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_PAUSE_RE = re.compile(r"(?:停顿|安静)约([一二两三四五六七八九十\d]+)秒")
MAX_PAUSE = 10.0  # 秒；"车内安静约十秒"以上的也只停 10 秒
SCENE_JUMP_PAUSE = 3.0  # "过了一会儿""约十分钟后"这类时间跳跃：停 3 秒


def _number(text: str) -> float:
    if text.isdigit():
        return float(text)
    if text.startswith("十"):
        return 10 + _CN_NUM.get(text[1:], 0)
    if "十" in text:
        tens, _, ones = text.partition("十")
        return _CN_NUM[tens] * 10 + _CN_NUM.get(ones, 0)
    return float(_CN_NUM.get(text, 2))


def line_effects(direction: str) -> dict:
    """把一句台词的舞台说明变成效果参数。

    返回 {"overlap": 和上一句部分重叠, "pause": 这句之前停几秒（None 表示正常停顿）, "speed": 语速倍数,
          "gain_db": 音量增减（分贝）, "near": 靠近手机, "far": 离手机远, "megaphone": 扩音器, "phone": 电话外放,
          "echo": 回声大, "events": 这句开始时的环境声列表}。
    """
    d = direction or ""
    found = _PAUSE_RE.search(d)
    pause = min(_number(found.group(1)), MAX_PAUSE) if found else None
    if pause is None and any(w in d for w in SCENE_JUMP_WORDS):
        pause = SCENE_JUMP_PAUSE
    gain = 0.0
    if any(w in d for w in QUIET_WORDS):
        gain -= 8.0
    if any(w in d for w in LOUD_WORDS):
        gain += 4.0
    near = any(w in d for w in NEAR_WORDS)
    if near:
        gain += 3.0
    speed = 1.15 if any(w in d for w in FAST_WORDS) else (0.9 if any(w in d for w in SLOW_WORDS) else 1.0)
    return {
        "overlap": any(w in d for w in OVERLAP_WORDS),
        "pause": pause,
        "speed": speed,
        "gain_db": gain,
        "near": near,
        "far": (not near) and any(w in d for w in FAR_WORDS),
        "megaphone": any(w in d for w in MEGAPHONE_WORDS) and not any(w in d for w in MEGAPHONE_OFF),
        "phone": any(w in d for w in PHONE_WORDS),
        "echo": any(w in d for w in ECHO_WORDS),
        "events": [name for word, name in EVENT_WORDS.items() if word in d],
    }


def clean_tts_text(text: str) -> str:
    """给合成模型念的文字：去掉书名号，省略号、破折号、冒号换成逗号（模型念不了这些符号，或者会把冒号念成英文 colon）；
    连着的大写字母（合同编号 HT、订单号 DD）中间加逗号，模型才会一个字母一个字母念清楚。"""
    text = text.replace("《", "").replace("》", "")
    text = re.sub(r"[…]+|—{2,}|——|[：:]", "，", text)
    text = re.sub(r"[A-Z]{2,}", lambda m: ", ".join(m.group()), text)
    return re.sub(r"，+", "，", text).strip("，") or text


def has_latin(text: str) -> bool:
    """台词里有没有英文字母（如"A区""HT"）：有的要用中英混读的模型念，只用中文的模型会把字母吞掉。"""
    return re.search(r"[A-Za-z]", text) is not None


# ---------------- 声音处理的小工具 ----------------

def db(value: float) -> float:
    """分贝 → 倍数。"""
    return float(10 ** (value / 20))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def trim_silence(x: np.ndarray, sr: int, threshold_db: float = -40.0, margin: float = 0.03) -> np.ndarray:
    """去掉合成结果首尾的静音（比峰值低 threshold_db 分贝算静音），两头各留 margin 秒。"""
    if not len(x):
        return x
    level = np.abs(x)
    loud = np.flatnonzero(level > level.max() * db(threshold_db))
    if not len(loud):
        return x[:0]
    pad = int(margin * sr)
    return x[max(0, loud[0] - pad): loud[-1] + pad + 1]


def bandpass(x: np.ndarray, sr: int, low: float | None, high: float | None, order: int = 4) -> np.ndarray:
    """带通（low、high 都给）、低通（只给 high）或高通（只给 low）滤波。"""
    from scipy.signal import butter, sosfilt

    if low and high:
        sos = butter(order, [low, high], btype="bandpass", fs=sr, output="sos")
    elif high:
        sos = butter(order, high, btype="lowpass", fs=sr, output="sos")
    else:
        sos = butter(order, low, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def reverb(x: np.ndarray, sr: int, rt60: float, wet: float, rng: np.random.Generator) -> np.ndarray:
    """人工混响：和一段按 rt60 秒衰减的随机噪声（房间冲激响应）做卷积。输出比输入长（多出混响的尾巴）。"""
    from scipy.signal import fftconvolve

    n = max(1, int(rt60 * sr))
    t = np.arange(n) / sr
    rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    rir[0] = 0.0
    rir /= np.sqrt(np.sum(rir ** 2)) + 1e-9
    tail = fftconvolve(x, rir)
    dry = np.concatenate([x, np.zeros(len(tail) - len(x), dtype=np.float32)])
    return ((1 - wet) * dry + wet * tail).astype(np.float32)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """粉红噪声（低频多、高频少，像房间里的底噪），均方根为 1。"""
    from scipy.fft import irfft, next_fast_len, rfft

    m = next_fast_len(max(n, 1), real=True)  # 长度凑成好算的数，否则很长的录音做 FFT 会非常慢
    spectrum = rfft(rng.standard_normal(m))
    spectrum[1:] /= np.sqrt(np.arange(1, len(spectrum)))
    x = irfft(spectrum, m)[:n]
    return (x / (rms(x) + 1e-12)).astype(np.float32)


def make_event(name: str, sr: int, rng: np.random.Generator) -> np.ndarray:
    """一次性的环境声：beep 提示音、squeal 扩音器尖啸、click 开关声、clink 敲银器、thump 颠簸/关门、chirp 鸟叫。"""
    def tone(freq, seconds, decay):
        t = np.arange(int(seconds * sr)) / sr
        return np.sin(2 * np.pi * freq * t) * np.exp(-t * decay)

    if name == "beep":
        one = tone(1000, 0.25, 2)
        return np.concatenate([one, np.zeros(int(0.15 * sr)), one]).astype(np.float32) * 0.3
    if name == "squeal":
        t = np.arange(int(0.5 * sr)) / sr
        return (np.sin(2 * np.pi * (2400 + 300 * t) * t) * np.minimum(1, t * 8) * 0.35).astype(np.float32)
    if name == "click":
        return (rng.standard_normal(int(0.02 * sr)) * np.exp(-np.arange(int(0.02 * sr)) / (0.003 * sr)) * 0.5).astype(np.float32)
    if name == "clink":
        hits = [tone(f, 0.4, 12) for f in (3100, 4200, 5300)]
        return (sum(hits) * 0.12).astype(np.float32)
    if name == "thump":
        t = np.arange(int(0.3 * sr)) / sr
        return (np.sin(2 * np.pi * 55 * t) * np.exp(-t * 18) * 0.6).astype(np.float32)
    if name == "chirp":
        t = np.arange(int(0.15 * sr)) / sr
        one = np.sin(2 * np.pi * (3500 + 4000 * t) * t) * np.hanning(len(t)) * 0.12
        return np.concatenate([one, np.zeros(int(0.1 * sr)), one, np.zeros(int(0.1 * sr)), one]).astype(np.float32)
    raise ValueError(f"不认识的环境声：{name}")


def render_line(x: np.ndarray, sr: int, fx: dict, rng: np.random.Generator) -> np.ndarray:
    """给一句干净的合成台词加上舞台说明里的效果（音量、扩音器、电话外放、离手机远、回声）。"""
    y = x.astype(np.float32) * db(fx["gain_db"])
    if fx["megaphone"]:
        y = bandpass(y, sr, 400, 3800)
        peak = np.max(np.abs(y)) + 1e-9
        y = (np.tanh(2.5 * y / peak) * peak).astype(np.float32) * db(3)
        y = reverb(y, sr, 0.35, 0.25, rng)
    if fx["phone"]:
        y = bandpass(y, sr, 300, 3400) * db(-4)
    if fx["far"]:
        y = reverb(y, sr, 0.55, 0.55, rng) * db(-9)
    if fx["echo"]:
        y = reverb(y, sr, 1.1, 0.6, rng)
    return y.astype(np.float32)


# ---------------- 一段录音：排时间、混音、标准答案 ----------------

def plan_timeline(lengths: list[float], effects: list[dict], rng: np.random.Generator, start: float = 0.0,
                  gap: tuple[float, float] = (0.25, 0.8)) -> list[float]:
    """每句台词从第几秒开始。lengths 是每句的时长（秒），effects 是 line_effects 的结果。

    正常两句之间停 gap 秒（随机）；"与上一句部分重叠""打断"的句子提前 0.3—0.8 秒开始（最多提前上一句的一半）；
    "停顿约 N 秒"停 N 秒。
    """
    starts = []
    prev_start, prev_end = start, start
    for i, (length, fx) in enumerate(zip(lengths, effects)):
        if i == 0:
            begin = start
        elif fx["overlap"]:
            back = min(rng.uniform(0.3, 0.8), 0.5 * (prev_end - prev_start))
            begin = max(prev_start + 0.3, prev_end - back)
        elif fx["pause"] is not None:
            begin = prev_end + fx["pause"]
        else:
            begin = prev_end + rng.uniform(*gap)
        starts.append(round(begin, 3))
        prev_start, prev_end = begin, begin + length
    return starts


def clips_from_lines(timeline: list[dict]) -> list[tuple[float, float, str]]:
    """疑似片段的标准答案：连续几句同一个疑似类别（5 个 flag 类别）的台词连成一个片段 (开始秒, 结束秒, 类别名)。"""
    clips = []
    for item in timeline:
        label = item["label"] if item["label"] in FLAG_LABELS else None
        if label and clips and clips[-1][2] == label and clips[-1][3] == item["index"] - 1:
            start, end, _, _ = clips[-1]
            clips[-1] = (start, max(end, item["end"]), label, item["index"])
        elif label:
            clips.append((item["start"], item["end"], label, item["index"]))
    return [(round(s, 3), round(e, 3), label) for s, e, label, _ in clips]


def synthesize(tts: Callable, text: str, sid: int, speed: float = 1.0, cache: dict | None = None) -> np.ndarray:
    """合成一句话：调用 tts(text, sid, speed)，转成 SR 采样率，去掉首尾静音。cache 里有就不再合成。"""
    key = (text, sid, round(speed, 3))
    if cache is not None and key in cache:
        return cache[key]
    samples, sr = tts(*key)
    samples = np.asarray(samples, dtype=np.float32)
    if sr != SR:
        from scipy.signal import resample_poly

        samples = resample_poly(samples, SR, sr).astype(np.float32)
    samples = trim_silence(samples, SR)
    if cache is not None:
        cache[key] = samples
    return samples


def mix_dialog(script: dict, voices: dict, tts: Callable, rng: np.random.Generator, cache: dict | None = None,
               lead: float = 0.5) -> tuple[np.ndarray, list[dict]]:
    """把一个剧本的全部台词合成、加效果、按时间排好混在一起（还没有加录音条件和开头的声明）。

    voices：{角色: {"sid": 声音编号, "speed": 语速倍数}}（来自 data/tts_voice_plan.csv）；
    tts(text, sid, speed) -> (采样, 采样率)；cache 用来存每句的合成结果，同一句在 Q/N/F 里只合成一次。
    返回 (混好的采样（SR 采样率）, 每句的时间线 [{index, speaker, start, end, label, text, direction}, ...])。
    """
    cache = {} if cache is None else cache
    dry, effects = [], []
    for line in script["lines"]:
        voice = voices[line["speaker"]]
        fx = line_effects(line.get("direction", ""))
        speed = voice.get("speed", 1.0) * fx["speed"]
        dry.append(synthesize(tts, clean_tts_text(line["text"]), voice["sid"], speed, cache))
        effects.append(fx)

    starts = plan_timeline([len(x) / SR for x in dry], effects, rng, start=lead)
    rendered = [render_line(x, SR, fx, rng) for x, fx in zip(dry, effects)]
    total = max(int(s * SR) + len(y) for s, y in zip(starts, rendered)) + SR
    out = np.zeros(total, dtype=np.float32)
    timeline = []
    for i, (line, start, x, y, fx) in enumerate(zip(script["lines"], starts, dry, rendered, effects)):
        at = int(start * SR)
        out[at: at + len(y)] += y
        for name in fx["events"]:
            event = make_event(name, SR, rng)
            out[at: at + len(event)] += event[: len(out) - at]
        timeline.append({"index": i, "speaker": line["speaker"], "start": round(start, 3),
                         "end": round(start + len(x) / SR, 3), "label": line.get("label", ""),
                         "text": line["text"], "direction": line.get("direction", ""),
                         "sid": voices[line["speaker"]]["sid"]})
    if any(w in script.get("setting", "") for w in ("大巴", "车上", "车载", "车里")):
        out += engine_hum(len(out), SR, rng) * db(-38)  # 车上的场景：一直有发动机声
    return out, timeline


def engine_hum(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """发动机的低频嗡嗡声（几个低频正弦 + 低通噪声），均方根约为 1。"""
    t = np.arange(n) / sr
    hum = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) / k for k, f in enumerate((38, 76, 114), start=1))
    noise = bandpass(rng.standard_normal(n).astype(np.float32), sr, None, 250)
    x = hum + 3 * noise
    return (x / (rms(x) + 1e-12)).astype(np.float32)


def babble(n: int, sr: int, pool: list[np.ndarray], rng: np.random.Generator, talkers: int = 6) -> np.ndarray:
    """教室里别人说话的嘈杂声：从别的剧本的合成台词里随机取，几路人声叠在一起、加混响，均方根约为 1。"""
    out = np.zeros(n, dtype=np.float32)
    for _ in range(talkers):
        pos = -int(rng.uniform(0, 2) * sr)  # 从别人一句话的中间开始：录音一开始就有嘈杂声
        while pos < n:
            piece = pool[int(rng.integers(len(pool)))] * rng.uniform(0.5, 1.0)
            begin, end = max(pos, 0), min(n, pos + len(piece))
            if end > begin:
                out[begin:end] += piece[begin - pos: end - pos]
            pos += len(piece) + int(rng.uniform(0.1, 0.6) * sr)
    out = bandpass(out, sr, 150, 5000)
    out = reverb(out, sr, 0.6, 0.7, rng)[:n]
    return (out / (rms(out) + 1e-12)).astype(np.float32)


def speech_rms(x: np.ndarray, sr: int) -> float:
    """说话部分的均方根：只算最响的 50% 的 100 毫秒小段，不让大段停顿把音量拉低。"""
    hop = int(0.1 * sr)
    blocks = [rms(x[i:i + hop]) for i in range(0, len(x) - hop, hop)]
    if not blocks:
        return rms(x)
    blocks.sort()
    loud = blocks[len(blocks) // 2:]
    return float(np.sqrt(np.mean(np.square(loud))))


# 三种录音条件的渲染参数（F 单数剧本 = 口袋，双数剧本 = 远距离）
TARGET_DB = {"Q": -22.0, "N": -22.0, "F": -30.0}


def apply_condition(x: np.ndarray, sr: int, condition: str, f_variant: str, rng: np.random.Generator,
                    babble_pool: list[np.ndarray] | None = None) -> np.ndarray:
    """按录音条件处理整段对话：Q 安静房间、N 嘈杂教室、F 口袋（f_variant="pocket"）或 2—3 米外（"far"）。"""
    n = len(x)
    if condition == "Q":
        y = reverb(x, sr, 0.25, 0.12, rng)[:n] + pink_noise(n, rng) * db(-62)
    elif condition == "N":
        if not babble_pool:
            raise ValueError("嘈杂教室（N）条件需要别的剧本的台词来做嘈杂声（babble_pool）")
        voice = reverb(x, sr, 0.35, 0.2, rng)[:n]
        snr = rng.uniform(8.0, 12.0)
        noise = babble(n, sr, babble_pool, rng) * speech_rms(voice, sr) * db(-snr)
        y = voice + noise + pink_noise(n, rng) * db(-50)
        for _ in range(int(n / sr / 15)):  # 每 15 秒左右一次桌椅碰撞
            at = int(rng.uniform(0, n / sr - 1) * sr)
            event = make_event("thump", sr, rng) * 0.5
            y[at: at + len(event)] += event[: n - at]
    elif condition == "F" and f_variant == "pocket":
        y = bandpass(x, sr, None, 2800) * db(-10) + pink_noise(n, rng) * db(-55)
        for _ in range(int(n / sr / 12)):  # 衣服摩擦声
            length = int(rng.uniform(0.2, 0.8) * sr)
            at = int(rng.uniform(0, max(0.0, n / sr - 1)) * sr)
            rustle = bandpass(rng.standard_normal(length).astype(np.float32), sr, 1000, 4000) * db(-32)
            rustle *= np.hanning(length).astype(np.float32)
            y[at: at + length] += rustle[: n - at]
    elif condition == "F":
        y = reverb(x, sr, 0.7, 0.65, rng)[:n] * db(-12) + pink_noise(n, rng) * db(-52)
    else:
        raise ValueError(f"不认识的录音条件：{condition}（只有 Q、N、F）")
    y = y * db(TARGET_DB[condition]) / (speech_rms(y, sr) + 1e-12)
    peak = np.max(np.abs(y))
    if peak > 0.89:  # 不让削波：最大不超过 -1 dBFS
        y *= 0.89 / peak
    return y.astype(np.float32)


def render_recording(script: dict, condition: str, voices: dict, disclaimer_voice: dict, tts: Callable,
                     seed: int, babble_pool: list[np.ndarray] | None = None,
                     cache: dict | None = None) -> tuple[np.ndarray, dict]:
    """合成一段完整的录音：开头一句清楚的声明 + 按条件处理的对话。

    返回 (采样（SR 采样率）, 标准答案 {"turns": [(开始, 结束, 说话人)], "clips": [(开始, 结束, 类别)],
    "reference": 参考文本, "timeline": 每句的时间线, "duration": 秒})。
    每个条件用不同的随机种子，停顿和重叠的时间不一样，就像同一个剧本录了三遍。
    """
    rng = np.random.default_rng(seed)
    cache = {} if cache is None else cache
    group = int(script["id"].split("-")[0][1:])
    script_no = int(script["id"].split("-")[1][1:])
    f_variant = "pocket" if (group + script_no) % 2 == 1 else "far"

    dialog, timeline = mix_dialog(script, voices, tts, rng, cache)
    dialog = apply_condition(dialog, SR, condition, f_variant, rng, babble_pool)

    notice = synthesize(tts, AI_DISCLAIMER, disclaimer_voice["sid"], disclaimer_voice.get("speed", 1.0), cache)
    notice = notice * db(-20) / (speech_rms(notice, SR) + 1e-12)
    lead, gap = 0.3, 1.0
    offset = lead + len(notice) / SR + gap
    audio = np.concatenate([np.zeros(int(lead * SR), np.float32), notice.astype(np.float32),
                            np.zeros(int(gap * SR), np.float32), dialog])
    peak = np.max(np.abs(audio))
    if peak > 0.89:
        audio *= 0.89 / peak

    shifted = [dict(item, start=round(item["start"] + offset, 3), end=round(item["end"] + offset, 3))
               for item in timeline]
    turns = [(lead, round(lead + len(notice) / SR, 3), AI_DISCLAIMER_SPEAKER)]
    turns += [(item["start"], item["end"], item["speaker"]) for item in shifted]
    truth = {
        "turns": turns,
        "clips": clips_from_lines(shifted),
        "reference": script_reference_text(script["id"], ai_disclaimer=True),
        "timeline": shifted,
        "duration": round(len(audio) / SR, 2),
        "f_variant": f_variant if condition == "F" else "",
    }
    return audio.astype(np.float32), truth


# ---------------- 声音分配 ----------------

FEMALE_SIDS = range(3, 58)   # Kokoro v1.1：3—57 号是中文女声
MALE_SIDS = range(58, 103)   # 58—102 号是中文男声

VOICE_PLAN = Path(__file__).resolve().parents[1] / "data" / "tts_voice_plan.csv"
VOICE_PLAN_COLUMNS = ["script_id", "role", "sid", "gender", "speed", "note"]
DISCLAIMER_ROW = "*"  # 声音分配表里"声明"那一行的 script_id

# 剧本对角色声音的要求：从剧本的说明（focus）和台词里的称呼（"大姐""大哥""阿姨""老公""闺女"……）看出来的。
# 没写的角色按 _default_gender 交替男女。改了这里要重新运行 tools/synthesize_recordings.py --make-voice-plan。
VOICE_RULES = {
    "G1-S2": {"游客乙": {"gender": "女", "note": "游客甲的母亲（被称作阿姨）"}},
    "G1-S3": {"游客乙": {"gender": "女", "note": "被称作大姐"}},
    "G2-S1": {"游客甲": {"gender": "男", "note": "被称作大哥"}, "游客乙": {"gender": "女", "note": "被称作美女"},
              "游客丙": {"gender": "女", "note": "被称作大姐"}},
    "G2-S2": {"游客甲": {"gender": "男", "note": "两位大哥之一"}, "游客乙": {"gender": "男", "note": "两位大哥之一"}},
    "G2-S3": {"游客甲": {"gender": "女", "note": "母亲（被称作阿姨）"}, "游客乙": {"gender": "女", "note": "女儿"}},
    "G3-S1": {"游客乙": {"similar_to": "游客甲", "note": "剧本要求和游客甲声线相近"}},
    "G3-S2": {"游客乙": {"similar_to": "游客甲", "note": "剧本要求和游客甲声线相近"}},
    "G3-S3": {"游客甲": {"gender": "女", "note": "姐妹"},
              "游客乙": {"similar_to": "游客甲", "note": "姐妹，剧本要求和游客甲腔调相近"}},
    "G4-S2": {"游客丙": {"gender": "女", "note": "被称作大姐"}},
    "G4-S3": {"游客甲": {"gender": "女", "note": "被称作大姐"}, "游客乙": {"gender": "男", "note": "被称作大哥"}},
    "G5-S1": {"游客甲": {"gender": "男", "note": "被称作大哥"}, "游客乙": {"gender": "女", "note": "被称作大姐"}},
    "G5-S3": {"游客乙": {"gender": "女", "note": "称游客丙老公"}, "游客丙": {"gender": "男", "note": "游客乙的丈夫"}},
    "G6-S1": {"游客甲": {"gender": "女", "note": "被称作大姐"}, "游客乙": {"gender": "男", "note": "被称作大哥"}},
    "G7-S2": {"游客甲": {"gender": "女", "note": "游客丙的女儿"},
              "游客丙": {"gender": "男", "speed": 0.9, "note": "七十二岁的父亲，说话慢一点"}},
    "G7-S3": {"游客乙": {"gender": "女", "high": True, "speed": 1.1, "note": "孩子：音调最高的声音，稍快"}},
}


def pick_disclaimer_voice(profiles: dict) -> int:
    """开头声明用的声音：和别的声音平均最不像的女声（说话人分离时不容易和角色混在一起）。"""
    sids = [s for s in profiles if s in FEMALE_SIDS]
    embs = np.array([profiles[s]["emb"] for s in profiles])

    def mean_similarity(sid):
        return float(np.mean(embs @ np.array(profiles[sid]["emb"])))

    return min(sids, key=mean_similarity)


def write_voice_plan(path, plan: list[dict], disclaimer_sid: int) -> None:
    """写声音分配表（UTF-8 带 BOM，Excel 直接打开不乱码）：第一行是开头声明的声音，后面每个剧本的每个角色一行。"""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VOICE_PLAN_COLUMNS)
        writer.writeheader()
        writer.writerow({"script_id": DISCLAIMER_ROW, "role": AI_DISCLAIMER_SPEAKER, "sid": disclaimer_sid,
                         "gender": "女", "speed": 1.0, "note": "每段录音开头的 AI 合成声明"})
        for item in plan:
            writer.writerow({key: item[key] for key in VOICE_PLAN_COLUMNS})


def voice_plan_rows(path=None) -> list[dict]:
    """声音分配表的每一行（字典，值都是文字）。"""
    from pipeline.textio import read_csv_dicts

    return read_csv_dicts(path or VOICE_PLAN, hint="请用 Excel 另存为\"CSV UTF-8\"格式。")[1]


def load_voice_plan(path=None) -> tuple[dict, dict]:
    """读声音分配表（默认 data/tts_voice_plan.csv），返回 ({剧本编号: {角色: {"sid", "speed"}}}, 声明的 {"sid", "speed"})。

    有写错的行（声音编号不是整数、语速不是数字）或缺少"声明"那一行时抛 ValueError（说明第几行）。
    """
    voices: dict[str, dict] = {}
    disclaimer = None
    for number, row in enumerate(voice_plan_rows(path), start=2):  # 第 1 行是表头
        try:
            voice = {"sid": int(row["sid"]), "speed": float(row.get("speed") or 1.0)}
        except (TypeError, ValueError):
            raise ValueError(f"声音分配表第 {number} 行写错了：sid 要是整数，speed 要是数字（{row}）") from None
        if row["script_id"] == DISCLAIMER_ROW:
            disclaimer = voice
        else:
            voices.setdefault(row["script_id"], {})[row["role"]] = voice
    if disclaimer is None:
        raise ValueError(f"声音分配表里没有\"{DISCLAIMER_ROW},{AI_DISCLAIMER_SPEAKER}\"那一行（开头声明用哪个声音）")
    return voices, disclaimer


def make_voice_plan(scripts: list[dict], profiles: dict, rules: dict | None = None) -> list[dict]:
    """给每个剧本的每个角色挑一个合成声音，返回 [{script_id, role, sid, gender, speed, note}, ...]。

    profiles：{声音编号: {"f0": 音高, "emb": 声纹向量（CAM++，已归一化）}}。
    挑法：同一个剧本里的声音尽量不像（声纹余弦距离大），男女交替；整个数据集里尽量多用不同的声音。
    rules（可选）：{剧本编号: {角色: {"gender": "女"/"男", "similar_to": 另一个角色, "speed": 1.1, "high": True, "note": "…"}}}，
    用来写"声线相近""母女""夫妻""孩子"这类剧本要求（similar_to 的角色要排在后面，没写性别时跟它一样）。
    """
    rules = rules or {}
    usage: dict[int, int] = {}
    plan = []
    for index, script in enumerate(scripts):
        sid_rules = rules.get(script["id"], {})
        chosen: dict[str, int] = {}
        genders: dict[str, str] = {}
        # 剧本里男女各几个：先算上规则里定好性别的角色，没定的角色补少的那一边
        counts = {"女": 0, "男": 0}
        for role in script["roles"]:
            if sid_rules.get(role, {}).get("gender"):
                counts[sid_rules[role]["gender"]] += 1
        for k, role in enumerate(script["roles"]):
            rule = sid_rules.get(role, {})
            gender = rule.get("gender") or genders.get(rule.get("similar_to"))
            if not gender:
                gender = _default_gender(role, k, index, counts)
            if not rule.get("gender"):
                counts[gender] += 1
            pool = [s for s in (FEMALE_SIDS if gender == "女" else MALE_SIDS) if s in profiles and s not in chosen.values()]
            if rule.get("similar_to") in chosen:
                ref = np.array(profiles[chosen[rule["similar_to"]]]["emb"])
                sid = max(pool, key=lambda s: float(np.dot(ref, profiles[s]["emb"])))
            elif rule.get("high"):
                sid = max(pool, key=lambda s: profiles[s]["f0"])
            else:
                def score(s):
                    sims = [float(np.dot(profiles[s]["emb"], profiles[c]["emb"])) for c in chosen.values()]
                    return (max(sims) if sims else 0.0) + 0.05 * usage.get(s, 0)
                sid = min(pool, key=score)
            chosen[role], genders[role] = sid, gender
            usage[sid] = usage.get(sid, 0) + 1
            plan.append({"script_id": script["id"], "role": role, "sid": sid, "gender": gender,
                         "speed": rule.get("speed", 1.0), "note": rule.get("note", "")})
    return plan


def _default_gender(role: str, k: int, script_index: int, counts: dict) -> str:
    """没有特别要求时的性别：司机是男声；导游按剧本单双交替；其余角色补这个剧本里少的那一边，一样多时交替。"""
    if role == "司机":
        return "男"
    if role == "导游":
        return "女" if script_index % 2 == 0 else "男"
    if counts["女"] != counts["男"]:
        return "女" if counts["女"] < counts["男"] else "男"
    return "男" if (k + script_index) % 2 == 0 else "女"
