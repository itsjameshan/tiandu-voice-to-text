"""仓库结构检查：交接包已移到 handoff/，数据和配置齐全，大文件不会进 git。"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_handoff_moved():
    assert (ROOT / "handoff" / "HANDOFF.md").is_file()
    assert (ROOT / "handoff" / "04_项目介绍视频").is_dir()
    assert not (ROOT / "HANDOFF.md").exists()


def test_data_present():
    for i in range(1, 9):
        assert (ROOT / "data" / "scripts" / f"group_{i}.json").is_file()
    for name in ["lines.csv", "labels.json", "hotwords.txt", "recording_plan.csv",
                 "fictional_names.csv", "hotword_variants.csv", "scripts_overview.csv"]:
        assert (ROOT / "data" / name).is_file(), name


def test_gitignore_blocks_media():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ["models/*", "!models/download_models.py", "data_pool/", "*.wav", ".venv/"]:
        assert pattern in text.splitlines(), pattern


def test_config_keys():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    for key in ["paths", "methods", "vad", "asr", "hotword", "diarize", "qc", "clips", "script_demo", "app"]:
        assert key in cfg, key
    assert set(cfg["methods"]) == {"denoise", "enhance", "vad", "hotword", "diarize", "normalize", "classify", "clips"}
    assert set(cfg["methods"].values()) == {"baseline"}


def test_requirements_need_gradio_6():
    """app.py 用了 Gradio 6 才有的写法（gr.Audio 的 buttons、launch 的 theme/css）：依赖要写 gradio>=6，
    不然已经装了 Gradio 5 的电脑 pip install 不会升级，网页工具启动就出错。"""
    line = next(l for l in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines() if l.startswith("gradio"))
    assert line.replace(" ", "").startswith("gradio>=6"), line
