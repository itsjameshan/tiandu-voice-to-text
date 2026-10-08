"""音频工具（pipeline/audio.py）和步骤 1 上传与格式统一（pipeline/step1_ingest.py）的测试。

测试音频全部用 ffmpeg 现场生成（正弦波、静音），不用语音合成（红线第 2 条）。
"""
import hashlib
import os
import shutil
import stat

import numpy as np
import pytest
import soundfile as sf

from pipeline import audio
from pipeline.audio import (
    SR,
    FfmpegNotFound,
    convert_to_wav,
    find_ffmpeg,
    has_non_ascii,
    probe,
    read_wav,
    sha256_file,
    write_wav,
)
from pipeline.config import load_config
from pipeline.step1_ingest import SUPPORTED_EXTS, ingest, quality_check

WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def _sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _is_read_only(path) -> bool:
    return not (os.stat(path).st_mode & WRITE_BITS)


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def work_dir(tmp_path):
    """处理结果放这里。测试结束后把只读文件改回可写：Windows 上只读文件删不掉，pytest 就清理不了临时文件夹。"""
    d = tmp_path / "work"
    yield d
    if d.exists():
        for p in d.rglob("*"):
            if p.is_file():
                os.chmod(p, stat.S_IREAD | stat.S_IWRITE)


# ---------- 计划里列出的测试 ----------


@pytest.mark.parametrize("ext", ["mp3", "m4a", "mp4", "wav"])
def test_convert_formats(make_audio, work_dir, cfg, ext):
    src = make_audio("tone", ext, seconds=3, sr=44100, channels=2)
    sha_before = _sha256(src)

    result = ingest(src, work_dir, cfg)

    info = sf.info(result["wav"])
    assert info.samplerate == 16000
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    # 原始文件没被改动，指纹正确；保存的副本与原始文件一模一样且是只读
    assert _sha256(src) == sha_before
    assert result["sha256"] == sha_before
    assert _sha256(result["original"]) == sha_before
    assert _is_read_only(result["original"])
    assert os.path.dirname(result["original"]) == str(work_dir / "original")
    # 其他返回值
    assert result["file"] == src.name
    assert result["original_sample_rate"] == 44100
    assert abs(result["duration"] - 3.0) < 0.2
    assert isinstance(result["qc"], list)


def test_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "data.bin"
    data = os.urandom(3 * 1024 * 1024 + 123)  # 超过 1 MB，检验分块读取
    p.write_bytes(data)
    assert sha256_file(p) == hashlib.sha256(data).hexdigest()


def test_qc_reports_long_silence(make_audio, work_dir, cfg):
    src = make_audio("tone_gap_tone", "wav")
    result = ingest(src, work_dir, cfg)
    assert any("静音" in problem for problem in result["qc"]), result["qc"]


def test_qc_low_sample_rate(make_audio, work_dir, cfg):
    src = make_audio("tone", "wav", seconds=3, sr=8000, channels=1)
    result = ingest(src, work_dir, cfg)
    assert result["original_sample_rate"] == 8000
    assert any("采样率" in problem for problem in result["qc"]), result["qc"]


def test_qc_ok_for_normal_tone(make_audio, work_dir, cfg):
    src = make_audio("tone", "wav", seconds=5)
    result = ingest(src, work_dir, cfg, expected_seconds=5)
    assert result["qc"] == []


def test_unsupported_ext(tmp_path, work_dir, cfg):
    src = tmp_path / "voice.amr"
    src.write_bytes(b"#!AMR\n")
    with pytest.raises(ValueError, match="微信"):
        ingest(src, work_dir, cfg)


def test_has_non_ascii():
    assert has_non_ascii("D:/asr/x") is False
    assert has_non_ascii("C:/Users/张三/x") is True


def test_find_ffmpeg_env_override(monkeypatch):
    real = shutil.which("ffmpeg")
    if real is None:
        import imageio_ffmpeg

        real = imageio_ffmpeg.get_ffmpeg_exe()
    monkeypatch.setenv("FFMPEG_BINARY", real)
    assert find_ffmpeg() == real


# ---------- 补充的测试 ----------


def test_supported_exts():
    assert SUPPORTED_EXTS == {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".mp4", ".mov", ".mkv"}
    assert SR == 16000


def test_uppercase_ext_accepted(make_audio, work_dir, cfg):
    """手机导出的文件常是大写扩展名（如 .WAV），也要能处理。"""
    src = make_audio("tone", "wav", seconds=2, name="UPPER.WAV")
    result = ingest(src, work_dir, cfg)
    assert sf.info(result["wav"]).samplerate == 16000


def test_find_ffmpeg_not_found(monkeypatch):
    import imageio_ffmpeg

    def no_exe():
        raise RuntimeError("no ffmpeg")

    monkeypatch.delenv("FFMPEG_BINARY", raising=False)
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", no_exe)
    with pytest.raises(FfmpegNotFound, match="FFMPEG_BINARY"):
        find_ffmpeg()


def test_find_ffmpeg_bad_env_falls_back(monkeypatch, tmp_path):
    """FFMPEG_BINARY 指向不存在的文件时，接着找 PATH 和 imageio-ffmpeg，不直接失败。"""
    bad = str(tmp_path / "no_such_ffmpeg.exe")
    monkeypatch.setenv("FFMPEG_BINARY", bad)
    exe = find_ffmpeg()
    assert exe != bad
    assert os.path.isfile(exe)


def test_probe_video_and_mono(make_audio):
    info = probe(make_audio("tone", "mp4", seconds=3, sr=44100, channels=2))
    assert info["sample_rate"] == 44100
    assert info["channels"] == 2
    assert abs(info["duration"] - 3.0) < 0.2

    info = probe(make_audio("tone", "wav", seconds=2, sr=8000, channels=1))
    assert info == {"duration": pytest.approx(2.0, abs=0.05), "sample_rate": 8000, "channels": 1}


def test_probe_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe(tmp_path / "missing.wav")


def test_convert_to_wav_error(tmp_path):
    bad = tmp_path / "not_audio.mp3"
    bad.write_text("这不是音频文件", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ffmpeg"):
        convert_to_wav(bad, tmp_path / "out.wav")


def test_convert_to_wav_creates_folder(make_audio, tmp_path):
    dst = tmp_path / "sub" / "dir" / "out.wav"
    convert_to_wav(make_audio("tone", "m4a", seconds=1), dst)
    info = sf.info(str(dst))
    assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")


def test_read_write_wav_roundtrip(tmp_path):
    t = np.arange(SR) / SR
    samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "x.wav"
    write_wav(path, samples)
    assert sf.info(str(path)).subtype == "PCM_16"
    back = read_wav(path)
    assert back.dtype == np.float32
    assert back.ndim == 1
    assert np.allclose(back, samples, atol=1e-3)


def test_write_wav_clips_out_of_range(tmp_path):
    path = tmp_path / "loud.wav"
    write_wav(path, np.array([2.0, -2.0, 0.0], dtype=np.float32))
    back = read_wav(path)
    assert back[0] > 0.99 and back[1] < -0.99  # 超出范围的值被截到 ±1，不会正负颠倒


def test_read_wav_rejects_other_rate_and_stereo(tmp_path):
    p1 = tmp_path / "44k.wav"
    sf.write(str(p1), np.zeros(4410, dtype=np.float32), 44100, subtype="PCM_16")
    with pytest.raises(ValueError, match="16000"):
        read_wav(p1)
    p2 = tmp_path / "stereo.wav"
    sf.write(str(p2), np.zeros((1600, 2), dtype=np.float32), SR, subtype="PCM_16")
    with pytest.raises(ValueError, match="单声道"):
        read_wav(p2)


def _tone(seconds, amplitude=0.1):
    t = np.arange(int(SR * seconds)) / SR
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_qc_too_quiet(cfg):
    problems = quality_check(_tone(3, amplitude=0.001), 44100, cfg)
    assert len(problems) == 1 and "音量" in problems[0], problems


def test_qc_clipping(cfg):
    problems = quality_check(np.clip(_tone(3, amplitude=3.0), -1.0, 1.0), 44100, cfg)
    assert len(problems) == 1 and "削波" in problems[0], problems


def test_qc_duration_mismatch(cfg):
    assert quality_check(_tone(5), 44100, cfg, expected_seconds=10) != []
    assert "时长" in quality_check(_tone(5), 44100, cfg, expected_seconds=10)[0]
    # 偏差在允许范围（40%）以内不报告
    assert quality_check(_tone(5), 44100, cfg, expected_seconds=6) == []


def test_qc_silence_and_empty_do_not_crash(cfg):
    problems = quality_check(np.zeros(SR * 12, dtype=np.float32), 16000, cfg)
    assert any("音量" in p for p in problems)
    assert any("静音" in p for p in problems)
    assert quality_check(np.zeros(0, dtype=np.float32), 16000, cfg) != []


def test_ingest_twice_same_work_dir(make_audio, work_dir, cfg):
    """同一个工作文件夹再处理一次：已有的只读副本要能被替换，不报错。"""
    src = make_audio("tone", "wav", seconds=2)
    first = ingest(src, work_dir, cfg)
    second = ingest(src, work_dir, cfg)
    assert first["sha256"] == second["sha256"]
    assert _is_read_only(second["original"])


def test_ingest_missing_file(tmp_path, work_dir, cfg):
    with pytest.raises(FileNotFoundError):
        ingest(tmp_path / "missing.m4a", work_dir, cfg)


def test_remove_tree_deletes_read_only_files(tmp_path):
    """原始文件被设为只读后，remove_tree 也能把整个文件夹删掉（Windows 上 shutil.rmtree 会失败）。"""
    folder = tmp_path / "work"
    (folder / "original").mkdir(parents=True)
    f = folder / "original" / "a.wav"
    f.write_bytes(b"x")
    os.chmod(f, stat.S_IREAD)
    audio.remove_tree(folder)
    assert not folder.exists()
    audio.remove_tree(folder)  # 不存在时什么也不做


def test_qc_unknown_sample_rate():
    """读不出原始采样率（0）时提示"读不出"，而不是说"只有 0 Hz"。"""
    from pipeline.config import load_config

    samples = (0.1 * np.sin(np.linspace(0, 2000, SR * 3))).astype("float32")
    problems = quality_check(samples, 0, load_config())
    assert any("读不出" in p for p in problems)
    assert not any("0 Hz" in p for p in problems)
