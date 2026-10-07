"""做法登记（pipeline/methods.py）和配置读取（pipeline/config.py）的测试。

测试里临时登记的做法，测试结束后都会删掉，不影响别的测试。
"""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from conftest import ROOT

from pipeline import methods
from pipeline.config import load_config


@pytest.fixture
def clean_registry():
    """先把能找到的做法都登记好，记下当时的登记表；测试结束后恢复原样。"""
    methods.load_all()
    before = {slot: dict(methods._REGISTRY[slot]) for slot in methods.SLOTS}
    yield
    for slot in methods.SLOTS:
        methods._REGISTRY[slot].clear()
        methods._REGISTRY[slot].update(before[slot])


def _ensure_baseline(slot):
    """步骤模块还没写好时，先手动登记一个 baseline 占位（已经有了就不动）。"""
    if "baseline" not in methods.available(slot):
        methods.register(slot, "baseline")(lambda *args, **kwargs: None)


# ---------- 做法登记 ----------

def test_register_and_get(clean_registry):
    with pytest.raises(ValueError):
        methods.register("unknown", "x")
    _ensure_baseline("denoise")

    @methods.register("denoise", "_test")
    def my_denoise(samples, sr, cfg):
        return samples

    assert methods.get_method("denoise", "_test") is my_denoise
    # "_test" 按字母排在 "baseline" 前面，但 baseline 必须排第一
    assert methods.available("denoise")[0] == "baseline"
    assert "_test" in methods.available("denoise")


def test_duplicate_name_rejected(clean_registry):
    methods.register("denoise", "_dup")(lambda samples, sr, cfg: samples)
    with pytest.raises(ValueError, match="_dup"):
        methods.register("denoise", "_dup")(lambda samples, sr, cfg: samples * 0)


def test_get_unknown_lists_available(clean_registry):
    _ensure_baseline("denoise")
    with pytest.raises(KeyError, match="baseline"):
        methods.get_method("denoise", "no_such_method")


def test_slots_and_titles():
    assert methods.SLOTS == ("denoise", "enhance", "vad", "hotword", "diarize", "normalize", "classify", "clips")
    assert methods.SLOT_TITLES == {
        "denoise": "降噪", "enhance": "远距离增强", "vad": "端点检测", "hotword": "热词纠错",
        "diarize": "说话人分离", "normalize": "数字规范化", "classify": "话术分类", "clips": "疑似片段",
    }
    # config.yaml 的 methods 正好列出全部槽位
    assert set(load_config()["methods"]) == set(methods.SLOTS)


def test_available_order(clean_registry):
    _ensure_baseline("clips")
    for name in ["_test_b", "_test_a"]:
        methods.register("clips", name)(lambda segments, samples, sr, cfg: [])
    names = methods.available("clips")
    assert names[0] == "baseline"
    assert names[1:] == sorted(names[1:])
    with pytest.raises(ValueError):
        methods.available("unknown")


def test_load_all_repeatable(clean_registry):
    methods.load_all()
    methods.load_all()
    for slot in methods.SLOTS:
        assert isinstance(methods.available(slot), list)


def test_resolve_uses_config_methods(clean_registry):
    funcs = {}
    for slot in methods.SLOTS:
        def fn(*args, **kwargs):
            return None
        funcs[slot] = methods.register(slot, "_test")(fn)
    cfg = load_config(overrides={"methods": {slot: "_test" for slot in methods.SLOTS}})
    chosen = methods.resolve(cfg)
    assert set(chosen) == set(methods.SLOTS)
    for slot in methods.SLOTS:
        assert chosen[slot] is funcs[slot]


def test_resolve_defaults_to_baseline(clean_registry):
    for slot in methods.SLOTS:
        _ensure_baseline(slot)
    chosen = methods.resolve({"methods": {"denoise": "baseline"}})
    for slot in methods.SLOTS:
        assert chosen[slot] is methods.get_method(slot, "baseline")


def test_resolve_rejects_unknown(clean_registry):
    for slot in methods.SLOTS:
        _ensure_baseline(slot)
    with pytest.raises(KeyError, match="baseline"):
        methods.resolve({"methods": {"denoise": "g99"}})
    # config.yaml 里把槽位名写错了，要明确报错
    with pytest.raises(ValueError, match="denoize"):
        methods.resolve({"methods": {"denoize": "baseline"}})


# ---------- 配置读取 ----------

def test_load_config_overrides():
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg = load_config(overrides={"vad": {"threshold": 0.3}})
    assert cfg["vad"]["threshold"] == 0.3
    for key, value in raw["vad"].items():
        if key != "threshold":
            assert cfg["vad"][key] == value
    assert cfg["asr"] == raw["asr"]
    models_dir = cfg["paths"]["models"]
    assert isinstance(models_dir, str)
    assert Path(models_dir).is_absolute()
    assert Path(models_dir).resolve() == (ROOT / "models").resolve()


def test_load_config_does_not_change_inputs():
    overrides = {"vad": {"threshold": 0.3}, "paths": {"outputs": "my_outputs"}}
    load_config(overrides=overrides)
    assert overrides == {"vad": {"threshold": 0.3}, "paths": {"outputs": "my_outputs"}}
    first = load_config()
    first["vad"]["threshold"] = 0.9
    assert load_config()["vad"]["threshold"] != 0.9


def test_load_config_custom_file(tmp_path):
    pool = tmp_path / "pool"
    text = (
        "# 用记事本另存的配置文件，开头可能带 BOM\n"
        "paths:\n"
        "  models: my_models\n"
        f"  data_pool: {pool.as_posix()}\n"
        "vad:\n"
        "  threshold: 0.4\n"
    )
    path = tmp_path / "custom.yaml"
    path.write_text(text, encoding="utf-8-sig")
    cfg = load_config(path)
    assert cfg["vad"]["threshold"] == 0.4
    assert Path(cfg["paths"]["models"]).resolve() == (ROOT / "my_models").resolve()
    assert Path(cfg["paths"]["data_pool"]) == pool
    assert load_config(str(path)) == cfg


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="配置文件"):
        load_config(tmp_path / "no_such_config.yaml")


# ---------- 轻量导入 ----------

def test_light_imports():
    code = (
        "import pipeline, pipeline.methods, pipeline.config; import sys; "
        "assert 'sherpa_onnx' not in sys.modules and 'gradio' not in sys.modules "
        "and 'tensorflow' not in sys.modules, sorted(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
