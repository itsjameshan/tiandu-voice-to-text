"""各组文件（pipeline/groups/）的测试：都登记好了，初始做法和基线结果一样。"""
import warnings

import numpy as np
import pytest
from conftest import FOUR_SPEAKERS_WAV, requires_models

from pipeline import methods
from pipeline.config import load_config
from pipeline.schema import new_segment

SR = 16000


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _tone(seconds=3.0):
    t = np.arange(int(SR * seconds)) / SR
    return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_groups_registered():
    methods.load_all()
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


def test_audio_stubs_equal_baseline(cfg):
    samples = _tone()
    get = methods.get_method
    assert np.array_equal(get("denoise", "g1")(samples, SR, cfg), get("denoise", "baseline")(samples, SR, cfg))
    assert np.array_equal(get("enhance", "g2")(samples, SR, cfg), get("enhance", "baseline")(samples, SR, cfg))
    for name in ["g1", "g2"]:
        assert get("vad", name)(samples, SR, cfg) == get("vad", "baseline")(samples, SR, cfg)


def test_text_stubs_equal_baseline(cfg):
    get = methods.get_method
    seg = new_segment(0, 3, text="我们报的是雾影行走旅行社，团费两千八百块", text_raw="我们报的是雾影行走旅行社团费两千八百块")
    on = load_config(overrides={"hotword": {"enabled": True}})
    assert get("hotword", "g5")([seg], None, on) == get("hotword", "baseline")([seg], None, on)
    for mode in ["spoken", "display"]:
        assert get("normalize", "g4")(seg, mode, cfg) == get("normalize", "baseline")(seg, mode, cfg)
    texts = ["这个手镯今天优惠价两千八百块", "大家注意安全，跟紧队伍", "不买的话下午的行程就不好安排了"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 没有训练好的模型时会提示改用规则
        for name in ["g6", "g7"]:
            assert get("classify", name)(texts, cfg) == get("classify", "baseline")(texts, cfg)
    segs = [new_segment(0.5, 2.0, label="疑似·费用"), new_segment(2.5, 2.9)]
    samples = _tone()
    assert get("clips", "g8")(segs, samples, SR, cfg) == get("clips", "baseline")(segs, samples, SR, cfg)


@requires_models
def test_diarize_stub_equal_baseline(cfg):
    from pipeline.audio import read_wav
    from pipeline.step2_vad import detect_speech

    samples = read_wav(FOUR_SPEAKERS_WAV)
    _, segs = detect_speech(samples, SR, cfg)
    four = load_config(overrides={"diarize": {"num_speakers": 4}})
    get = methods.get_method
    assert get("diarize", "g3")(samples, SR, segs, four) == get("diarize", "baseline")(samples, SR, segs, four)


def test_classifier_groups_have_build_model():
    from pipeline.groups import g6_classifier_a, g7_classifier_b

    assert callable(g6_classifier_a.build_model)
    assert callable(g7_classifier_b.build_model)
