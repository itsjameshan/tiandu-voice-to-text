"""步骤 1：上传与格式统一（教师模板，学生不改）。

做三件事：
  1. 把原始文件原样复制一份到 工作文件夹/original/，设为只读，并算出 SHA-256 指纹
     （证明处理的是哪一份文件、文件没被改过）；
  2. 用 ffmpeg 转成 16000 Hz、单声道、16 位的 WAV（工作文件夹/audio_16k.wav），后面的步骤都读这个文件；
  3. 质检：只报告、不修改。这一步不降噪、不调音量，否则分不清第 1、2 组的改进是谁带来的。

基线做法：
    质检五项（阈值在 config.yaml 的 qc 下面，可以用试录数据调整）：
      - 时长：实际时长与剧本预计时长相差超过 duration_tolerance（默认 40%）；
      - 音量太小：整段录音的音量（RMS，换算成 dBFS）低于 min_rms_dbfs（默认 -40 dBFS）；
      - 削波：声音太大被"削平"的采样点比例超过 clip_ratio（默认 0.1%）；
      - 大段静音：每 0.1 秒算一次音量，低于 silence_dbfs（默认 -50 dBFS）算静音，
        最长一段连续静音超过 long_silence_seconds（默认 10 秒）；
      - 原始采样率低于 min_sample_rate（默认 16000 Hz）。
    dBFS：以数字音频能表示的最大值为 0 dB，越小越轻；这里用 20 × log10(RMS) 计算。
可改进方向：
    本步骤是教师模板，不开放替换。可以讨论的是质检阈值：第 1 组用全班试录数据校准音量和静音阈值。
测评指标：
    转换后为 16000 Hz、单声道、PCM_16；原始文件 SHA-256 不变；质检能报出大段静音、采样率低等问题
    （见 tests/test_step1_ingest.py）。
"""
import os
import shutil
import stat
from pathlib import Path

import numpy as np

from pipeline.audio import SR, convert_to_wav, probe, read_wav, sha256_file

# 支持的文件格式（音频 + 视频；视频只取声音）
SUPPORTED_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".mp4", ".mov", ".mkv"}

# 转换后的 WAV 文件名（固定用英文名，避免原文件名里的中文在 Windows 上出问题）
WAV_NAME = "audio_16k.wav"

# 采样点的绝对值达到这个数就算"削波"（16 位录音的最大值约为 0.99997）
CLIP_LEVEL = 0.999

# 算静音时每一小段的长度（秒）
FRAME_SECONDS = 0.1

# 文件的"可写"权限位（用户、同组、其他人）
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def _dbfs(rms: float) -> float:
    """把音量（RMS，0 到 1）换算成 dBFS。完全无声时返回一个很小的数（-200），不报错。"""
    return 20 * np.log10(max(rms, 1e-10))


def _fmt_seconds(seconds: float) -> str:
    """把秒数写成好读的样子：不到 1 分钟写 "12.3 秒"，否则写 "2 分 05 秒"。"""
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    total = int(round(seconds))
    return f"{total // 60} 分 {total % 60:02d} 秒"


def _longest_run(flags: np.ndarray) -> tuple[int, int]:
    """找最长的一段连续 True，返回 (长度, 开始位置)。没有 True 时返回 (0, 0)。"""
    best_len, best_start = 0, 0
    run_len, run_start = 0, 0
    for i, flag in enumerate(flags):
        if flag:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len > best_len:
                best_len, best_start = run_len, run_start
        else:
            run_len = 0
    return best_len, best_start


def quality_check(samples, original_sr: int, cfg: dict, expected_seconds: float | None = None) -> list[str]:
    """录音质检：返回中文问题列表，空列表表示合格。只报告，不修改录音。

    samples：已转成 16000 Hz 单声道的采样（read_wav 读出来的数组）。
    original_sr：原始文件的采样率（probe 读出来的）。
    cfg：完整配置（load_config() 的结果），阈值在 cfg["qc"] 里。
    expected_seconds：剧本预计时长（秒）；不知道时不检查时长。
    """
    qc = cfg["qc"]
    samples = np.asarray(samples, dtype=np.float32)
    problems = []

    duration = len(samples) / SR
    if len(samples) == 0:
        problems.append("录音时长为 0 秒，文件可能是空的或已损坏，请重新导出或重录")

    # 1. 时长与剧本预计时长相差太多
    if expected_seconds and expected_seconds > 0:
        diff = abs(duration - expected_seconds) / expected_seconds
        if diff > qc["duration_tolerance"]:
            problems.append(
                f"时长与剧本预计时长相差较大：实际 {_fmt_seconds(duration)}，预计 {_fmt_seconds(expected_seconds)}"
                f"（相差 {diff:.0%}），请确认没有漏录、多录或录错剧本"
            )

    if len(samples) > 0:
        # 2. 整体音量太小
        rms_db = _dbfs(float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))))
        if rms_db < qc["min_rms_dbfs"]:
            level = "几乎没有声音" if rms_db < -100 else f"整体音量 {rms_db:.1f} dBFS"
            problems.append(
                f"音量太小：{level}，低于 {qc['min_rms_dbfs']:g} dBFS，请让手机离说话人近一些后重录"
            )

        # 3. 削波：声音太大被削平的采样点太多
        ratio = float(np.mean(np.abs(samples) >= CLIP_LEVEL))
        if ratio > qc["clip_ratio"]:
            problems.append(
                f"削波（声音太大被削平）：{ratio:.2%} 的采样点达到最大值，超过 {qc['clip_ratio']:.2%}，"
                "请让手机离说话人远一些或说话轻一些后重录"
            )

        # 4. 大段静音：每 0.1 秒算一次音量，找最长的一段连续静音
        frame = int(SR * FRAME_SECONDS)
        n_frames = len(samples) // frame
        if n_frames > 0:
            frames = samples[: n_frames * frame].astype(np.float64).reshape(n_frames, frame)
            frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
            silent = 20 * np.log10(np.maximum(frame_rms, 1e-10)) < qc["silence_dbfs"]
            run_len, run_start = _longest_run(silent)
            longest = run_len * FRAME_SECONDS
            if longest > qc["long_silence_seconds"]:
                problems.append(
                    f"有一大段静音：从第 {_fmt_seconds(run_start * FRAME_SECONDS)}开始，"
                    f"连续 {_fmt_seconds(longest)}没有声音（超过 {qc['long_silence_seconds']:g} 秒），"
                    "请确认录音中间没有中断"
                )

    # 5. 原始采样率太低（读不出采样率时 probe 给出 0）
    if original_sr <= 0:
        problems.append("读不出原始采样率，文件可能不完整或格式特殊，请用手机自带的录音机重新导出或重录")
    elif original_sr < qc["min_sample_rate"]:
        problems.append(
            f"原始采样率只有 {original_sr} Hz，低于 {qc['min_sample_rate']} Hz，声音细节不够，"
            "请用手机自带的录音机、选高音质重录"
        )

    return problems


def _make_writable(path: Path) -> None:
    """给文件加上写权限（Windows 上就是去掉"只读"属性），之后才能删除或覆盖。"""
    os.chmod(path, os.stat(path).st_mode | stat.S_IWUSR)


def _set_read_only(path: Path) -> None:
    """去掉文件的写权限（Windows 上就是勾选"只读"属性）。"""
    os.chmod(path, os.stat(path).st_mode & ~_WRITE_BITS)


def ingest(src_path, work_dir, cfg: dict, expected_seconds: float | None = None) -> dict:
    """上传与格式统一：保存只读原件、算指纹、转成 16k 单声道 WAV、质检。

    返回字典：
      file                  原始文件名
      original              工作文件夹/original/ 里的只读原件路径
      wav                   转换后的 WAV 路径
      sha256                原始文件的 SHA-256 指纹
      duration              转换后的时长（秒，保留 2 位小数）
      original_sample_rate  原始文件的采样率（Hz）
      qc                    质检问题列表（空列表表示合格）
    """
    src = Path(src_path)
    ext = src.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(
            f"不支持这种文件格式：{src.name}。支持的格式：{'、'.join(sorted(SUPPORTED_EXTS))}。"
            "微信语音（.amr/.silk）请先导出为 m4a 或 mp3。"
        )
    if not src.is_file():
        raise FileNotFoundError(f"找不到文件：{src}")

    work_dir = Path(work_dir)
    original_dir = work_dir / "original"
    original_dir.mkdir(parents=True, exist_ok=True)

    # 1. 原样复制一份原始文件并设为只读
    original = original_dir / src.name
    if not (original.exists() and os.path.samefile(src, original)):
        if original.exists():
            # 同一个工作文件夹里再处理一次：旧副本是只读的，先改回可写再删掉
            _make_writable(original)
            original.unlink()
        shutil.copy2(src, original)
    _set_read_only(original)
    sha256 = sha256_file(original)

    # 2. 读原始采样率，转成 16000 Hz 单声道 16 位 WAV
    info = probe(original)
    wav = work_dir / WAV_NAME
    convert_to_wav(original, wav)
    samples = read_wav(wav)

    # 3. 质检（只报告，不修改）
    problems = quality_check(samples, info["sample_rate"], cfg, expected_seconds)

    return {
        "file": src.name,
        "original": str(original),
        "wav": str(wav),
        "sha256": sha256,
        "duration": round(len(samples) / SR, 2),
        "original_sample_rate": info["sample_rate"],
        "qc": problems,
    }
