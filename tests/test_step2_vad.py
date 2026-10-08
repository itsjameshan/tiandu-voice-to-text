"""步骤 2 降噪、远距离增强、端点检测（pipeline/step2_vad.py）的测试。

需要 Silero VAD 模型的测试带 requires_models 标记，没下载模型时自动跳过。
测试音频只用模型自带的四人测试音频和程序生成的静音、正弦波（不用语音合成）。
"""
import subprocess
import sys

import numpy as np
import pytest
from conftest import FOUR_SPEAKERS_WAV, ROOT, requires_models

from pipeline import step2_vad
from pipeline.audio import SR, read_wav
from pipeline.config import load_config
from pipeline.methods import available, get_method
from pipeline.step2_vad import detect_speech


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def four_speakers():
    return read_wav(FOUR_SPEAKERS_WAV)


def _tone(seconds=2.0, freq=440.0):
    t = np.arange(int(seconds * SR)) / SR
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------- 降噪、远距离增强 ----------

def test_baseline_denoise_identity(cfg):
    samples = _tone()
    out = get_method("denoise", "baseline")(samples, SR, cfg)
    assert np.array_equal(out, samples)


def test_enhance_identity(cfg):
    samples = _tone()
    out = get_method("enhance", "baseline")(samples, SR, cfg)
    assert np.array_equal(out, samples)


def test_methods_registered():
    assert available("denoise")[:1] == ["baseline"]
    assert "noisereduce" in available("denoise")
    assert "baseline" in available("enhance")
    assert "baseline" in available("vad")


def test_noisereduce_float32_same_length(cfg):
    rng = np.random.default_rng(0)
    samples = (_tone(3.0) + 0.05 * rng.standard_normal(3 * SR)).astype(np.float32)
    out = get_method("denoise", "noisereduce")(samples, SR, cfg)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == samples.shape


# ---------- 端点检测 ----------

@requires_models
def test_vad_silence_returns_empty(cfg):
    vad = get_method("vad", "baseline")
    assert vad(np.zeros(3 * SR, dtype=np.float32), SR, cfg) == []


@requires_models
def test_vad_on_four_speakers(cfg, four_speakers):
    spans = get_method("vad", "baseline")(four_speakers, SR, cfg)
    assert len(spans) >= 5
    previous_end = 0.0
    for start, end in spans:
        assert end > start
        assert 0 <= start and end <= 57
        assert start >= previous_end  # 按时间递增，互不重叠
        assert round(start, 2) == start and round(end, 2) == end  # 保留 2 位小数
        previous_end = end


@requires_models
def test_vad_repeatable(cfg, four_speakers):
    # 检测器有内部状态：每次调用都要用新的检测器，第二次结果不能接着第一次的时间往后算
    vad = get_method("vad", "baseline")
    assert vad(four_speakers, SR, cfg) == vad(four_speakers, SR, cfg)


@requires_models
def test_vad_uses_config_params(cfg, four_speakers):
    vad = get_method("vad", "baseline")
    normal = vad(four_speakers, SR, cfg)
    # 停顿要超过 2 秒才切开：几句话会连成一段，段数变少
    merged = vad(four_speakers, SR, load_config(overrides={"vad": {"min_silence_duration": 2.0}}))
    assert 0 < len(merged) < len(normal)
    assert max(end - start for start, end in merged) > max(end - start for start, end in normal)


def test_vad_config_cached():
    model = str(ROOT / "models" / "silero_vad.onnx")
    first = step2_vad._vad_config(model, 0.5, 0.3, 0.25, 15.0)
    assert step2_vad._vad_config(model, 0.5, 0.3, 0.25, 15.0) is first
    other = step2_vad._vad_config(model, 0.6, 0.3, 0.25, 15.0)
    assert other is not first
    assert other.silero_vad.threshold == pytest.approx(0.6)
    assert first.silero_vad.min_silence_duration == pytest.approx(0.3)
    assert first.sample_rate == SR


def test_vad_missing_model(tmp_path):
    cfg = load_config(overrides={"paths": {"models": str(tmp_path)}})
    with pytest.raises(FileNotFoundError, match="download_models"):
        get_method("vad", "baseline")(_tone(), SR, cfg)


def test_vad_rejects_wrong_sample_rate(cfg):
    with pytest.raises(ValueError, match="16000"):
        get_method("vad", "baseline")(_tone(), 8000, cfg)


# ---------- detect_speech：依次 降噪 → 增强 → 端点检测 ----------

@requires_models
def test_detect_speech_uses_methods(cfg, four_speakers):
    def mute(samples, sr, cfg):
        return samples * 0

    processed, segments = detect_speech(four_speakers, SR, cfg, methods={"denoise": mute})
    assert segments == []
    assert not processed.any()


@requires_models
def test_detect_speech_returns_segments(cfg, four_speakers):
    processed, segments = detect_speech(four_speakers, SR, cfg)
    assert processed.dtype == np.float32
    assert np.array_equal(processed, four_speakers)  # 基线降噪、增强都原样返回
    assert len(segments) >= 5
    for seg in segments:
        assert seg["end"] > seg["start"]
        assert seg["speaker"] == "未知"
        assert seg["text"] == "" and seg["label"] == ""
        assert seg["review"] == "未复核"


def test_detect_speech_rejects_length_change(cfg):
    def cut(samples, sr, cfg):
        return samples[:100]

    with pytest.raises(ValueError, match="长度"):
        detect_speech(_tone(), SR, cfg, methods={"enhance": cut})


def test_light_import():
    code = (
        "import sys, pipeline.step2_vad; "
        "heavy = [m for m in ('sherpa_onnx', 'noisereduce', 'gradio', 'tensorflow') if m in sys.modules]; "
        "assert not heavy, heavy"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
