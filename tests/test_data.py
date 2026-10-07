"""剧本与表格读取（pipeline/data.py）的测试。"""
import re
import subprocess
import sys

import pytest
from conftest import ROOT

# 直接运行 .venv/bin/pytest 时仓库根目录不一定在 sys.path 里，先加进去才能 import pipeline
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import data
from pipeline.data import (
    DATA_DIR,
    FLAG_LABELS,
    LABEL_NAMES,
    get_script,
    label_output,
    load_fictional_names,
    load_hotword_variants,
    load_hotwords,
    load_lines,
    load_recording_plan,
    load_scripts,
    script_reference_text,
)


# ---------- 计划里列出的测试 ----------

def test_load_scripts():
    scripts = load_scripts()
    assert len(scripts) == 24
    ids = [s["id"] for s in scripts]
    assert ids == [f"G{g}-S{s}" for g in range(1, 9) for s in range(1, 4)]
    assert ids[0] == "G1-S1" and ids[-1] == "G8-S3"
    for s in scripts:
        assert isinstance(s["group"], int)
        assert s["id"].startswith(f"G{s['group']}-")
    assert len(load_lines()) == 1055


def test_label_output():
    assert label_output("费用") == "疑似·费用"
    assert label_output("正常讲解") == ""
    with pytest.raises(ValueError):
        label_output("违规")


def test_reference_text_excludes_direction():
    text = script_reference_text("G1-S1")
    assert "（用车载扩音器）" not in text
    assert len(text.split("\n")) == 53


def test_recording_plan():
    plan = load_recording_plan()
    assert len(plan) == 72
    for row in plan:
        assert re.match(r"^G[1-8]-S[1-3]-[QNF]\.wav$", row["file_name"]), row["file_name"]
        assert isinstance(row["expected_minutes"], float)


# ---------- 补充测试 ----------

def test_data_dir():
    assert DATA_DIR == ROOT / "data"


def test_get_script():
    script = get_script("G8-S3")
    assert script["id"] == "G8-S3"
    assert script["group"] == 8
    assert script["lines"]
    with pytest.raises(KeyError):
        get_script("G9-S1")


def test_load_lines_types_and_lists():
    lines = load_lines()
    first = lines[0]
    assert first["script_id"] == "G1-S1"
    assert first["group"] == 1 and first["line_no"] == 1
    assert isinstance(first["effective_chars"], int)
    assert first["numbers"] == [] and first["hotwords"] == []
    third = lines[2]
    assert third["numbers"] == ["20多年"]
    assert third["hotwords"] == ["云栖野渡旅行社", "周导"]
    for line in lines:
        assert isinstance(line["numbers"], list) and isinstance(line["hotwords"], list)
        assert len(line["numbers"]) == len(set(line["numbers"]))
        assert len(line["hotwords"]) == len(set(line["hotwords"]))
        assert "" not in line["numbers"] and "" not in line["hotwords"]


def test_load_lines_dedupe_keeps_order():
    # lines.csv 里有一行 numbers 是 "190元；190元；1090元"，去重后保持原来的先后顺序
    lines = load_lines()
    assert any(line["numbers"] == ["190元", "1090元"] for line in lines)


def test_reference_text_matches_lines_csv():
    for script_id in ["G1-S1", "G5-S2", "G8-S3"]:
        texts = [line["text"] for line in load_lines() if line["script_id"] == script_id]
        assert script_reference_text(script_id) == "\n".join(texts)


def test_labels():
    assert LABEL_NAMES == ["购物安排", "费用", "行程变更", "服务态度", "威胁消费", "正常讲解", "其他"]
    assert FLAG_LABELS == ["购物安排", "费用", "行程变更", "服务态度", "威胁消费"]
    for name in FLAG_LABELS:
        assert label_output(name) == "疑似·" + name
    assert label_output("其他") == ""
    with pytest.raises(ValueError):
        label_output("")


def test_every_line_label_is_known():
    for line in load_lines():
        assert line["label"] in LABEL_NAMES


def test_load_hotwords():
    words = load_hotwords()
    assert len(words) == 121
    assert words[0] == "云栖野渡旅行社"
    assert "雾隐行舟旅行社" in words
    assert all(w and w == w.strip() for w in words)


def test_load_hotword_variants():
    rows = load_hotword_variants()
    assert len(rows) == 8
    assert rows[0]["spoken_variant"] == "雾隐晚渡"
    assert rows[0]["correct_name"] == "雾隐行舟旅行社"


def test_load_fictional_names():
    rows = load_fictional_names()
    assert len(rows) == 39
    assert set(rows[0]) == {"type", "name", "groups", "note"}
    assert {"type": "旅行社", "name": "雾隐行舟旅行社"}.items() <= next(
        r for r in rows if r["name"] == "雾隐行舟旅行社").items()


def test_cached_results_are_copies():
    # 结果有缓存，但调用方改了返回的列表，下次读到的仍是原样
    lines = load_lines()
    lines[0]["text"] = "被改掉了"
    lines.clear()
    assert load_lines()[0]["text"] != "被改掉了"
    script = get_script("G1-S1")
    script["lines"].clear()
    assert len(get_script("G1-S1")["lines"]) == 53
    scripts = load_scripts()
    scripts[0]["id"] = "XX"
    assert load_scripts()[0]["id"] == "G1-S1"
    words = load_hotwords()
    words.clear()
    assert len(load_hotwords()) == 121


def test_module_docstring_has_three_parts():
    for part in ["基线做法", "可改进方向", "测评指标"]:
        assert part in data.__doc__


def test_data_light_import():
    code = ("import sys; import pipeline.data; "
            "assert not {'sherpa_onnx', 'gradio', 'tensorflow', 'yaml'} & set(sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
