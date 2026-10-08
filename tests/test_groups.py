"""各组文件（pipeline/groups/）的测试：都登记好了，每个做法返回的格式都对（"接口测试"）。

这里只检查每个做法返回的东西格式对不对（类型、长度、时间范围、类别名），不检查和基线一样不一样：
各组改进以后结果和基线不同是正常的，这些测试照样通过。
如果提交后这里有一项变红，说明本组函数返回的格式错了（流程后面的步骤会出错），要先改好再提交，不要改这个测试。
每个槽位登记的所有做法（baseline、gN、noisereduce、tf_model……）都按同样的要求检查。
"""
import warnings

import numpy as np
import pytest
from conftest import FOUR_SPEAKERS_WAV, requires_models

from pipeline import methods
from pipeline.config import load_config
from pipeline.data import LABEL_NAMES
from pipeline.schema import new_segment

SR = 16000

methods.load_all()


def _names(slot: str) -> list[str]:
    """这个槽位登记的全部做法名，如 ["baseline", "g1", "noisereduce"]。"""
    return methods.available(slot)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _tone(seconds=3.0):
    t = np.arange(int(SR * seconds)) / SR
    return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _segments():
    return [
        new_segment(0.2, 1.4, text="我们报的是雾影行走旅行社，团费两千八百块", text_raw="我们报的是雾影行走旅行社团费两千八百块",
                    label="疑似·费用"),
        new_segment(1.6, 2.8, text="大家注意安全，跟紧队伍", text_raw="大家注意安全跟紧队伍"),
    ]


def test_groups_registered():
    expected = {
        "denoise": ["g1"], "vad": ["g1", "g2"], "enhance": ["g2"], "diarize": ["g3"],
        "normalize": ["g4"], "hotword": ["g5"], "classify": ["g6", "g7", "tf_model"], "clips": ["g8"],
    }
    for slot, names in expected.items():
        available = methods.available(slot)
        assert available[0] == "baseline"
        for name in names:
            assert name in available, (slot, name)


def test_load_all_is_strict(monkeypatch):
    """所有步骤和各组文件都已写好：缺模块时不再悄悄跳过。"""
    monkeypatch.setattr(methods, "_METHOD_MODULES", methods._METHOD_MODULES + ("pipeline.no_such_module",))
    with pytest.raises(ModuleNotFoundError):
        methods.load_all()


@pytest.mark.parametrize("slot,name", [(slot, name) for slot in ("denoise", "enhance") for name in _names(slot)])
def test_audio_method_returns_samples(cfg, slot, name):
    """降噪、远距离增强：返回一维的小数数组，长度和输入一样（后面按时间切段要用），没有 NaN、无穷大。"""
    samples = _tone()
    result = methods.get_method(slot, name)(samples, SR, cfg)
    assert isinstance(result, np.ndarray) and result.ndim == 1, f"{slot}={name} 要返回一维的 numpy 数组"
    assert len(result) == len(samples), f"{slot}={name} 返回的长度要和输入一样（{len(samples)} 个采样）"
    assert np.issubdtype(result.dtype, np.floating) and np.isfinite(result).all(), f"{slot}={name} 返回了 NaN 或无穷大"


@requires_models
@pytest.mark.parametrize("name", _names("vad"))
def test_vad_method_returns_spans(cfg, name):
    """端点检测：返回 [(开始秒, 结束秒), ...]，按时间先后排列，每段开始 < 结束，都在录音时长之内。"""
    from pipeline.audio import read_wav

    samples = read_wav(FOUR_SPEAKERS_WAV)
    duration = len(samples) / SR
    spans = methods.get_method("vad", name)(samples, SR, cfg)
    assert isinstance(spans, list) and spans, f"vad={name} 在测试音频上一段人声也没检测到"
    starts = [float(s) for s, _ in spans]
    assert starts == sorted(starts), f"vad={name} 返回的段要按时间先后排列"
    for start, end in spans:
        assert 0 <= float(start) < float(end) <= duration + 0.01, f"vad={name} 返回了不合理的段 ({start}, {end})"


@pytest.mark.parametrize("name", _names("hotword"))
def test_hotword_method_keeps_segments(name):
    """热词纠错：返回同样多的段落，时间不变，text、text_raw 都还是文字；纠错记录是 [{"from", "to", ...}]。"""
    on = load_config(overrides={"hotword": {"enabled": True}})
    segments = _segments()
    result = methods.get_method("hotword", name)([dict(seg) for seg in segments], None, on)
    assert isinstance(result, list) and len(result) == len(segments), f"hotword={name} 要返回同样多的段落"
    for before, after in zip(segments, result):
        assert (after["start"], after["end"]) == (before["start"], before["end"])
        assert isinstance(after["text"], str) and isinstance(after["text_raw"], str)
        for item in after.get("corrections", []):
            assert {"from", "to"} <= set(item), f"hotword={name} 的纠错记录要有 from 和 to"


@pytest.mark.parametrize("name", _names("normalize"))
@pytest.mark.parametrize("mode", ["spoken", "display"])
def test_normalize_method_returns_segment(cfg, name, mode):
    """数字规范化：返回一个段落（时间不变），text 是文字，numbers、entities 是文字列表。"""
    seg = _segments()[0]
    result = methods.get_method("normalize", name)(dict(seg), mode, cfg)
    assert isinstance(result, dict) and (result["start"], result["end"]) == (seg["start"], seg["end"])
    assert isinstance(result["text"], str)
    for key in ("numbers", "entities"):
        value = result.get(key, [])
        assert isinstance(value, list) and all(isinstance(x, str) for x in value), f"normalize={name} 的 {key} 要是文字列表"


@pytest.mark.parametrize("name", _names("classify"))
def test_classify_method_returns_category_names(cfg, name):
    """话术分类：每句话返回一个类别名（data/labels.json 里的 7 个之一，如"费用"，不是"疑似·费用"），句数一样。"""
    texts = ["这个手镯今天优惠价两千八百块", "大家注意安全，跟紧队伍", "不买的话下午的行程就不好安排了"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 没有训练好的模型时会提示改用规则
        labels = methods.get_method("classify", name)(texts, cfg)
    assert isinstance(labels, list) and len(labels) == len(texts), f"classify={name} 每句话要有一个结果"
    assert all(label in LABEL_NAMES for label in labels), f"classify={name} 返回了不认识的类别：{labels}"


@pytest.mark.parametrize("name", _names("clips"))
def test_clips_method_returns_clips(cfg, name):
    """疑似片段：返回 [(开始秒, 结束秒, 段落序号), ...]，开始 < 结束，在录音时长之内，序号指向一个段落。"""
    segments = _segments()
    samples = _tone()
    clips = methods.get_method("clips", name)(segments, samples, SR, cfg)
    assert isinstance(clips, list)
    for start, end, index in clips:
        assert 0 <= start < end <= len(samples) / SR + 0.01, f"clips={name} 返回了不合理的片段 ({start}, {end})"
        assert isinstance(index, int) and 0 <= index < len(segments), f"clips={name} 的段落序号 {index} 不对"


@requires_models
@pytest.mark.parametrize("name", _names("diarize"))
def test_diarize_method_labels_speakers(cfg, name):
    """说话人分离：每段都有 speaker_id（从 1 开始的整数，或 None）和 speaker（"说话人N"或"未知"），时间合理。

    段数可以变（例如把中间换了人的一段切成两段），但每段都要有说话人。
    """
    from pipeline.audio import read_wav
    from pipeline.step2_vad import detect_speech

    samples = read_wav(FOUR_SPEAKERS_WAV)
    _, segs = detect_speech(samples, SR, cfg)
    four = load_config(overrides={"diarize": {"num_speakers": 4}})
    result = methods.get_method("diarize", name)(samples, SR, [dict(seg) for seg in segs], four)
    assert isinstance(result, list) and result, f"diarize={name} 一段也没返回"
    for seg in result:
        assert 0 <= seg["start"] < seg["end"] <= len(samples) / SR + 0.01
        assert seg["speaker_id"] is None or (isinstance(seg["speaker_id"], int) and seg["speaker_id"] >= 1)
        assert isinstance(seg["speaker"], str) and seg["speaker"]


def test_classifier_groups_have_build_model():
    from pipeline.groups import g6_classifier_a, g7_classifier_b

    assert callable(g6_classifier_a.build_model)
    assert callable(g7_classifier_b.build_model)
