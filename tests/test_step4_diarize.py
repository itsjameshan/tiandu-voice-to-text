"""步骤 4 说话人分离（pipeline/step4_diarize.py）的测试。

需要模型的测试带 requires_models 标记，没下载模型时自动跳过。
测试音频只用模型自带的四人测试音频和程序生成的正弦波（不用语音合成）。
段落用 pipeline.schema.new_segment 直接造，不依赖步骤 2、3。
"""
import subprocess
import sys

import numpy as np
import pytest
from conftest import FOUR_SPEAKERS_WAV, ROOT, requires_models

from pipeline import step4_diarize
from pipeline.audio import SR, read_wav
from pipeline.config import load_config
from pipeline.methods import available, get_method
from pipeline.schema import new_segment
from pipeline.step4_diarize import (
    SPEAKER_ROLES,
    apply_speaker_map,
    attach_speakers,
    diarize_turns,
    speaker_durations,
)


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def four_speakers():
    return read_wav(FOUR_SPEAKERS_WAV)


@pytest.fixture(scope="module")
def four_turns(four_speakers):
    # 分离一次要几秒钟，同一个文件里的几个测试共用这一次的结果
    cfg = load_config(overrides={"diarize": {"num_speakers": 4}})
    return diarize_turns(four_speakers, SR, cfg)


def _tone(seconds=2.0, freq=440.0):
    t = np.arange(int(seconds * SR)) / SR
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _no_model(*args, **kwargs):
    raise AssertionError("这里不应该加载说话人分离模型")


# ---------- 按重叠时长配说话人 ----------

def test_attach_by_overlap():
    turns = [(0, 5, 0), (5, 10, 1)]
    segments = [new_segment(1, 4), new_segment(4.5, 9), new_segment(11, 12)]
    out = attach_speakers(segments, turns)
    assert [seg["speaker"] for seg in out] == ["说话人1", "说话人2", "未知"]
    assert [seg["speaker_id"] for seg in out] == [1, 2, None]


def test_attach_keeps_other_fields_and_input():
    segments = [new_segment(1, 4, text="你好", label="")]
    out = attach_speakers(segments, [(0, 5, 2)])
    assert out[0]["speaker"] == "说话人3" and out[0]["speaker_id"] == 3
    assert out[0]["text"] == "你好" and out[0]["start"] == 1.0 and out[0]["end"] == 4.0
    # 传进来的段落不被改动
    assert segments[0]["speaker"] == "未知" and "speaker_id" not in segments[0]


def test_attach_touching_turn_is_not_overlap():
    # 只是首尾相接（重叠 0 秒）不算重叠
    out = attach_speakers([new_segment(5, 6)], [(0, 5, 0)])
    assert out[0]["speaker"] == "未知" and out[0]["speaker_id"] is None


def test_attach_empty():
    assert attach_speakers([], [(0, 5, 0)]) == []
    out = attach_speakers([new_segment(0, 1)], [])
    assert out[0]["speaker"] == "未知" and out[0]["speaker_id"] is None


# ---------- 说话人映射、说话时长 ----------

def test_apply_speaker_map():
    segments = attach_speakers([new_segment(1, 4), new_segment(6, 8)], [(0, 5, 0), (5, 10, 1)])
    out = apply_speaker_map(segments, {"说话人1": "导游"})
    assert out[0]["speaker"] == "导游"
    assert out[0]["speaker_id"] == 1
    assert out[1]["speaker"] == "说话人2" and out[1]["speaker_id"] == 2  # 映射里没写的不变
    assert segments[0]["speaker"] == "说话人1"  # 传进来的段落不被改动


def test_apply_speaker_map_again_uses_speaker_id():
    # 已经映射过一次，改了主意再应用一次：按编号仍然能对上
    segments = attach_speakers([new_segment(1, 4), new_segment(6, 8)], [(0, 5, 0), (5, 10, 1)])
    first = apply_speaker_map(segments, {"说话人1": "导游", "说话人2": "游客"})
    second = apply_speaker_map(first, {"说话人1": "游客", "说话人2": "导游"})
    assert [seg["speaker"] for seg in second] == ["游客", "导游"]
    assert [seg["speaker_id"] for seg in second] == [1, 2]


def test_apply_speaker_map_blank_and_unknown():
    segments = [new_segment(0, 1, speaker="说话人1", speaker_id=1), new_segment(1, 2, speaker_id=None)]
    out = apply_speaker_map(segments, {"说话人1": "", "说话人2": "导游"})
    assert out[0]["speaker"] == "说话人1"  # 没选角色（空字符串）就不改
    assert out[1]["speaker"] == "未知" and out[1]["speaker_id"] is None


def test_speaker_durations():
    segments = [
        new_segment(0, 2, speaker="说话人1"),
        new_segment(2, 5, speaker="说话人2"),
        new_segment(5, 6.5, speaker="说话人1"),
        new_segment(7, 7.25, speaker="未知"),
    ]
    durations = speaker_durations(segments)
    assert durations == {"说话人1": 3.5, "说话人2": 3.0, "未知": 0.25}
    assert list(durations) == ["说话人1", "说话人2", "未知"]  # 说话时间长的排前面
    assert speaker_durations([]) == {}


def test_speaker_roles():
    assert SPEAKER_ROLES == ["导游", "游客", "店员", "司机", "经理", "未知"]


# ---------- 说话人分离 ----------

def test_short_audio_single_turn(cfg, monkeypatch):
    # 短于 2 秒：不加载模型，整段算一个人
    monkeypatch.setattr(step4_diarize, "get_diarizer", _no_model)
    assert diarize_turns(_tone(1.5), SR, cfg) == [(0.0, 1.5, 0)]
    assert diarize_turns(np.zeros(0, dtype=np.float32), SR, cfg) == [(0.0, 0.0, 0)]


def test_diarize_rejects_wrong_sample_rate(cfg):
    with pytest.raises(ValueError, match="16000"):
        diarize_turns(_tone(3.0), 8000, cfg)


def test_diarize_missing_model(tmp_path):
    cfg = load_config(overrides={"paths": {"models": str(tmp_path)}})
    with pytest.raises(FileNotFoundError, match="download_models"):
        diarize_turns(_tone(3.0), SR, cfg)


def test_baseline_registered():
    assert available("diarize")[:1] == ["baseline"]
    assert get_method("diarize", "baseline") is step4_diarize.diarize_baseline


def test_baseline_empty_segments_skips_model(cfg, monkeypatch):
    # 没有人声段落（例如全静音录音）：直接返回空列表，不加载模型
    monkeypatch.setattr(step4_diarize, "get_diarizer", _no_model)
    assert get_method("diarize", "baseline")(_tone(5.0), SR, [], cfg) == []


@requires_models
def test_four_speakers(four_turns):
    assert len({speaker for _, _, speaker in four_turns}) == 4


@requires_models
def test_four_speakers_turns_well_formed(four_turns):
    assert set(speaker for _, _, speaker in four_turns) == {0, 1, 2, 3}  # 编号从 0 开始
    previous_start = 0.0
    for start, end, speaker in four_turns:
        assert 0 <= start < end <= 57
        assert start >= previous_start  # 按开始时间排好
        assert round(start, 2) == start and round(end, 2) == end  # 保留 2 位小数
        previous_start = start


@requires_models
def test_diarizer_cached(cfg):
    first = step4_diarize.get_diarizer(cfg, 4, 0.5)
    assert step4_diarize.get_diarizer(cfg, 4, 0.5) is first
    assert step4_diarize.get_diarizer(cfg, 3, 0.5) is not first
    assert step4_diarize.get_diarizer(cfg, 4, 0.6) is not first


@requires_models
def test_baseline_on_four_speakers(four_speakers, four_turns):
    cfg = load_config(overrides={"diarize": {"num_speakers": 4}})
    # 每 5 秒造一个段落（不依赖步骤 2）；基线 = 先分离、再按重叠时长配说话人
    segments = [new_segment(t, t + 5) for t in range(0, 55, 5)]
    out = get_method("diarize", "baseline")(four_speakers, SR, segments, cfg)
    assert out == attach_speakers(segments, four_turns)
    assert {seg["speaker_id"] for seg in out} <= {1, 2, 3, 4}
    assert len({seg["speaker_id"] for seg in out}) >= 2
    assert all(seg["speaker"] == f"说话人{seg['speaker_id']}" for seg in out)


def test_light_import():
    code = (
        "import sys, pipeline.step4_diarize; "
        "heavy = [m for m in ('sherpa_onnx', 'gradio', 'tensorflow') if m in sys.modules]; "
        "assert not heavy, heavy"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
