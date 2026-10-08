"""步骤 3 语音识别（pipeline/step3_asr.py）的测试。

需要模型的测试用 SenseVoice 自带的公开测试音频 test_wavs/zh.wav（不用语音合成，红线第 2 条），
没有下载模型时自动跳过。分批、进度条这些逻辑用一个假的识别器测，不需要模型。
"""
import re
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest
from conftest import ROOT, requires_models

from pipeline import methods, step3_asr
from pipeline.audio import SR, read_wav
from pipeline.config import load_config
from pipeline.models import model_path
from pipeline.schema import new_segment


# ---------- 假的识别器：只记下收到了什么，不做真正的识别 ----------

class FakeStream:
    def __init__(self, use_itn):
        self.use_itn = use_itn
        self.sample_rate = None
        self.n_samples = None
        self.result = None

    def accept_waveform(self, sample_rate, samples):
        self.sample_rate = sample_rate
        self.n_samples = len(samples)


class FakeResult:
    def __init__(self, text):
        self.text = text


class FakeRecognizer:
    def __init__(self, use_itn):
        self.use_itn = use_itn
        self.batches = []  # 每次 decode_streams 收到几个 stream

    def create_stream(self):
        return FakeStream(self.use_itn)

    def decode_streams(self, streams):
        self.batches.append(len(streams))
        for stream in streams:
            word = "显示" if self.use_itn else "测评"
            stream.result = FakeResult(f" {word}{stream.n_samples} ")


@pytest.fixture
def fake_recognizers(monkeypatch):
    """把 get_recognizer 换成返回假识别器（显示模式、测评模式各一个）。"""
    recognizers = {True: FakeRecognizer(True), False: FakeRecognizer(False)}
    calls = []

    def fake_get_recognizer(cfg, use_itn):
        calls.append(use_itn)
        return recognizers[use_itn]

    monkeypatch.setattr(step3_asr, "get_recognizer", fake_get_recognizer)
    return recognizers, calls


def _cfg(batch_size=20) -> dict:
    return load_config(overrides={"asr": {"batch_size": batch_size}})


# ---------- 计划里列出的测试 ----------

@requires_models
def test_recognize_zh_modes():
    cfg = load_config()
    samples = read_wav(model_path(cfg, "sense_voice_test_zh"))
    # 不依赖步骤 2：直接用一个覆盖整段录音的段落
    segments = [new_segment(0, len(samples) / SR)]
    result = step3_asr.recognize(samples, SR, segments, cfg, mode="both")
    assert len(result) == 1
    # 显示模式（use_itn=True）：阿拉伯数字，如"早上9点至下午5点"
    assert "9" in result[0]["text"]
    # 测评模式（use_itn=False）：汉字读法，不含阿拉伯数字
    assert "九" in result[0]["text_raw"]
    assert not re.search(r"\d", result[0]["text_raw"])


def test_recognize_empty():
    fake = mock.Mock()
    with mock.patch.object(step3_asr, "get_recognizer", fake):
        assert step3_asr.recognize(np.zeros(SR, dtype=np.float32), SR, [], _cfg(), mode="both") == []
    fake.assert_not_called()


# ---------- 其他测试 ----------

def test_recognize_rejects_unknown_mode():
    with pytest.raises(ValueError, match="display"):
        step3_asr.recognize(np.zeros(SR, dtype=np.float32), SR, [new_segment(0, 1)], _cfg(), mode="fast")


def test_recognize_batches_and_progress(fake_recognizers):
    recognizers, calls = fake_recognizers
    samples = np.zeros(10 * SR, dtype=np.float32)
    segments = [new_segment(i, i + 0.5) for i in range(5)]
    before = [dict(seg) for seg in segments]
    progress_calls = []

    result = step3_asr.recognize(samples, SR, segments, _cfg(batch_size=2), mode="both",
                                 progress=lambda fraction, desc: progress_calls.append((fraction, desc)))

    # 5 段、每批 2 段 → 每种模式 3 批（2、2、1）
    assert recognizers[True].batches == [2, 2, 1]
    assert recognizers[False].batches == [2, 2, 1]
    assert sorted(calls) == [False, True]
    # 显示模式写 text，测评模式写 text_raw；前后空格去掉；每段送进去的是 0.5 秒（8000 个采样）
    assert [seg["text"] for seg in result] == ["显示8000"] * 5
    assert [seg["text_raw"] for seg in result] == ["测评8000"] * 5
    # 其他字段保留
    assert [(seg["start"], seg["end"], seg["review"]) for seg in result] == [(i, i + 0.5, "未复核") for i in range(5)]
    # 每批更新一次进度：两种模式共 6 批，进度从小到大，最后是 1
    fractions = [fraction for fraction, _ in progress_calls]
    assert len(progress_calls) == 6
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(1.0)
    assert all(isinstance(desc, str) and desc for _, desc in progress_calls)
    # 不改调用方传进来的段落
    assert segments == before


def test_recognize_display_only(fake_recognizers):
    recognizers, calls = fake_recognizers
    segments = [new_segment(0, 1), new_segment(1, 1.5)]
    result = step3_asr.recognize(np.zeros(2 * SR, dtype=np.float32), SR, segments, _cfg(), mode="display")
    assert calls == [True]
    assert [seg["text"] for seg in result] == ["显示16000", "显示8000"]
    assert all("text_raw" not in seg for seg in result)


def test_recognize_eval_only(fake_recognizers):
    recognizers, calls = fake_recognizers
    segments = [new_segment(0, 1, text="原来的文字")]
    result = step3_asr.recognize(np.zeros(2 * SR, dtype=np.float32), SR, segments, _cfg(), mode="eval")
    assert calls == [False]
    assert result[0]["text_raw"] == "测评16000"
    assert result[0]["text"] == "原来的文字"  # 测评模式不动 text


def test_recognize_clips_segment_to_audio(fake_recognizers):
    # 段落结束时间超过录音长度时，只取到录音结尾，不报错
    segments = [new_segment(1.5, 3.0)]
    result = step3_asr.recognize(np.zeros(2 * SR, dtype=np.float32), SR, segments, _cfg())
    assert result[0]["text"] == "显示8000"


def test_get_recognizer_missing_model(tmp_path):
    cfg = load_config(overrides={"paths": {"models": str(tmp_path)}})
    with pytest.raises(FileNotFoundError, match="download_models"):
        step3_asr.get_recognizer(cfg, True)


@requires_models
def test_get_recognizer_cached():
    cfg = load_config()
    display = step3_asr.get_recognizer(cfg, True)
    assert step3_asr.get_recognizer(cfg, True) is display
    evaluation = step3_asr.get_recognizer(cfg, False)
    assert evaluation is not display
    assert step3_asr.get_recognizer(cfg, False) is evaluation


def test_importing_step3_registers_hotword_baseline():
    # methods.load_all() 只导入 pipeline.step3_asr（不直接导入 pipeline.hotwords），
    # 所以导入 step3_asr 时热词纠错的基线做法也要登记进来。在新的子进程里检查，不受别的测试影响。
    code = (
        "import pipeline.step3_asr; from pipeline import methods; "
        "assert methods.available('hotword') == ['baseline'], methods.available('hotword')"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
    from pipeline.hotwords import hotword_baseline

    assert methods.get_method("hotword", "baseline") is hotword_baseline
