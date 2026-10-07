"""音频工具：找 ffmpeg、查看音频信息、转成统一格式、读写 WAV、算文件指纹。

整个项目只处理一种音频格式：16000 Hz（SR）、单声道、16 位的 WAV。
手机录音、视频等各种格式，先用 ffmpeg 转成这种格式，后面的步骤才能读。

基线做法：
    - find_ffmpeg：依次找环境变量 FFMPEG_BINARY → 系统 PATH → imageio-ffmpeg 自带的程序。
    - probe：运行 "ffmpeg -i 文件"，从它打印的信息里读出时长、采样率、声道数。
    - convert_to_wav：ffmpeg -y -i 原文件 -vn -ac 1 -ar 16000 -sample_fmt s16 输出.wav
      （-vn 不要画面，-ac 1 单声道，-ar 16000 采样率，-sample_fmt s16 16 位）。
    - read_wav / write_wav：用 soundfile 读写；读出来是 float32 数组，数值在 -1 到 1 之间。
    - sha256_file：文件指纹（SHA-256），用来证明处理的是哪一份文件、文件没被改过。
    - has_non_ascii：路径里有没有中文等非英文字符（Windows 上中文路径容易出问题）。
可改进方向：
    这是全班共用的工具，学生不改。
测评指标：
    不涉及识别效果；由 tests/test_step1_ingest.py 检查转换结果格式、指纹是否正确。

注意：soundfile、imageio-ffmpeg 在函数里导入，import pipeline.audio 时只需要 numpy。
所有路径用 pathlib 处理，调用 ffmpeg 不经过命令行外壳（shell），Windows 和 Linux 都能用。
"""
import hashlib
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 统一采样率：16000 Hz（识别、端点检测、说话人分离的模型都要求这个采样率）
SR = 16000


class FfmpegNotFound(RuntimeError):
    """找不到 ffmpeg 程序。"""


def find_ffmpeg() -> str:
    """返回 ffmpeg 程序的完整路径；都找不到时抛 FfmpegNotFound（带中文提示）。

    查找顺序：
      1. 环境变量 FFMPEG_BINARY（老师可以用它指定某个 ffmpeg.exe）
      2. 系统 PATH 里的 ffmpeg
      3. pip 包 imageio-ffmpeg 自带的 ffmpeg（便携包用的就是它）
    """
    env = os.environ.get("FFMPEG_BINARY", "").strip()
    if env:
        if Path(env).is_file():
            return env
        # 设了但文件不存在：提醒一下，接着往下找（机房电脑上可能有别的软件留下的旧设置）
        logger.warning("环境变量 FFMPEG_BINARY 指向的文件不存在：%s，改为在 PATH 和 imageio-ffmpeg 里查找。", env)

    exe = shutil.which("ffmpeg")
    if exe:
        return exe

    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return exe
    except (ImportError, RuntimeError):
        pass

    raise FfmpegNotFound(
        "找不到 ffmpeg（转换音频格式要用它）。程序依次查找了：环境变量 FFMPEG_BINARY、系统 PATH、"
        "imageio-ffmpeg 自带的程序。解决办法（任选一个）：运行 pip install imageio-ffmpeg；"
        "或安装 ffmpeg 并把它所在的文件夹加入 PATH；或把环境变量 FFMPEG_BINARY 设为 ffmpeg.exe 的完整路径。"
    )


def _run_ffmpeg(args: list[str]) -> tuple[int, str]:
    """运行 ffmpeg，返回 (退出码, ffmpeg 打印的信息)。

    ffmpeg 的信息按 UTF-8 解码（不用系统默认编码，中文 Windows 上也不会因为文件名里的中文出错）。
    """
    proc = subprocess.run(
        [find_ffmpeg(), "-hide_banner", "-nostdin", *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
    )
    return proc.returncode, proc.stderr.decode("utf-8", errors="replace")


def _last_lines(text: str, n: int = 20) -> str:
    """取一段文字的最后 n 行（报错时附上 ffmpeg 的输出，方便查原因）。"""
    return "\n".join(text.strip().splitlines()[-n:])


# 声道布局名 → 声道数（ffmpeg 打印 "mono"、"stereo"、"5.1" 或 "2 channels" 等）
_LAYOUT_CHANNELS = {"mono": 1, "stereo": 2, "2.1": 3, "3.0": 3, "quad": 4, "4.0": 4,
                    "5.0": 5, "5.1": 6, "6.1": 7, "7.1": 8}


def probe(path) -> dict:
    """读出音频（或视频里的音轨）的基本信息：{"duration": 秒, "sample_rate": Hz, "channels": 声道数}。

    做法：运行 "ffmpeg -i 文件"（不指定输出，ffmpeg 只打印信息），从打印的文字里找：
        Duration: 00:00:03.03, ...
        Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, ...
    读不出时长时 duration 为 0.0；读不出声道数时 channels 为 0。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到文件：{path}")
    _, err = _run_ffmpeg(["-i", str(path)])  # 没有指定输出，ffmpeg 会以出错退出，这是正常的

    audio_line = re.search(r"Stream #\d+:\d+.*?: Audio: (.*)", err)
    if audio_line is None:
        raise RuntimeError(f"ffmpeg 在这个文件里没有找到声音（音轨），文件可能已损坏或不是音视频文件：{path}\n"
                           + _last_lines(err))
    rate = re.search(r"(\d+) Hz, ([^,]+)", audio_line.group(1))
    sample_rate = int(rate.group(1)) if rate else 0
    channels = 0
    if rate:
        layout = rate.group(2).strip()
        many = re.match(r"(\d+) channels", layout)
        if many:
            channels = int(many.group(1))
        else:
            channels = _LAYOUT_CHANNELS.get(layout.split("(")[0], 0)  # "5.1(side)" 按 "5.1" 算

    duration = 0.0
    found = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", err)
    if found:
        hours, minutes, seconds = found.groups()
        duration = round(int(hours) * 3600 + int(minutes) * 60 + float(seconds), 2)

    return {"duration": duration, "sample_rate": sample_rate, "channels": channels}


def convert_to_wav(src, dst) -> None:
    """用 ffmpeg 把任意音频或视频转成 16000 Hz、单声道、16 位的 WAV（视频只取声音）。

    只转格式，不降噪、不调音量。转换失败抛 RuntimeError，信息里带 ffmpeg 输出的最后 20 行。
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    code, err = _run_ffmpeg(["-loglevel", "error", "-y", "-i", str(src),
                             "-vn", "-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", str(dst)])
    if code != 0 or not dst.is_file():
        raise RuntimeError(f"ffmpeg 转换失败：{src}\n" + _last_lines(err))


def read_wav(path) -> np.ndarray:
    """读 16000 Hz 单声道 WAV，返回 float32 数组（数值在 -1 到 1 之间）。

    采样率不是 16000 或不是单声道时抛 ValueError：请先用 convert_to_wav 转换。
    """
    import soundfile as sf

    samples, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SR:
        raise ValueError(f"录音采样率是 {sr} Hz，需要 {SR} Hz，请先用 convert_to_wav 转换：{path}")
    if samples.ndim != 1:
        raise ValueError(f"录音有 {samples.shape[1]} 个声道，需要单声道，请先用 convert_to_wav 转换：{path}")
    return samples


def write_wav(path, samples, sr: int = SR) -> None:
    """把 float32 数组写成 16 位 WAV。超出 -1 到 1 的值先截断，避免写入时正负颠倒产生爆音。"""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    sf.write(str(path), samples, sr, subtype="PCM_16")


def sha256_file(path) -> str:
    """计算文件的 SHA-256 指纹（64 位十六进制字符）。按 1 MB 一块读，大文件也不占太多内存。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def has_non_ascii(path) -> bool:
    """路径里有没有中文等非英文字符。例如 "C:/Users/张三/x" 返回 True，"D:/asr/x" 返回 False。"""
    return not str(path).isascii()
