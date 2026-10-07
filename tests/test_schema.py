"""统一中间格式（pipeline/schema.py）的测试：字段、表头、JSON/CSV 往返。"""
import json
import subprocess
import sys

import pytest
from conftest import ROOT

from pipeline import schema
from pipeline.schema import (
    FIELDS,
    HEADERS,
    NOTICE,
    OPTIONAL_FIELDS,
    REVIEW_CHOICES,
    new_segment,
    read_csv,
    read_json,
    write_csv,
    write_json,
)


def _two_segments():
    """两个示例段落：带数字、热词纠错记录和复核结论。"""
    first = new_segment(12.4, 18.9, speaker="导游", text="这个手镯今天优惠价2800元。", label="疑似·费用",
                        numbers=["2800元"], corrections=[{"from": "雾影行走旅行社", "to": "雾隐行舟旅行社"}],
                        review="确认", review_note="已听原声")
    second = new_segment(20.0, 25.55, speaker="游客", text="电话是0871-0000-6688。",
                         numbers=["0871-0000-6688", "15:40"], review="未复核")
    return [first, second]


# ---------- 计划里列出的测试 ----------

def test_json_roundtrip(tmp_path):
    segments = _two_segments()
    path = tmp_path / "segments.json"
    write_json(path, segments, {"file": "G1-S1-Q.wav", "duration": 268.4})
    got_segments, got_meta = read_json(path)
    assert got_segments == segments
    assert got_meta["notice"] == NOTICE
    assert got_meta["file"] == "G1-S1-Q.wav"


def test_csv_headers_and_order(tmp_path):
    segments = _two_segments()
    path = tmp_path / "segments.csv"
    write_csv(path, segments)
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "CSV 要以 UTF-8 BOM 开头，Excel 才不乱码"
    first_line = raw.decode("utf-8-sig").splitlines()[0]
    assert first_line.split(",")[:5] == ["开始时间（秒）", "结束时间（秒）", "说话人", "文字内容", "标签"]
    assert read_csv(path) == segments


def test_new_segment_defaults():
    seg = new_segment(1.234, 2.0)
    assert seg["start"] == 1.23
    assert seg["end"] == 2.0
    assert seg["speaker"] == "未知"
    assert seg["review"] == "未复核"
    assert seg["label"] == ""
    assert seg["text"] == ""


# ---------- 补充测试 ----------

def test_constants():
    assert FIELDS == ["start", "end", "speaker", "text", "label"]
    assert list(HEADERS)[:5] == FIELDS
    assert HEADERS["start"] == "开始时间（秒）"
    assert HEADERS["review"] == "复核结论"
    assert HEADERS["text_raw"] == "识别原文（汉字读法）"
    assert OPTIONAL_FIELDS == [
        "text_raw", "speaker_id", "category", "score", "numbers", "entities",
        "corrections", "review", "review_note", "clip", "source",
    ]
    assert list(HEADERS) == FIELDS + OPTIONAL_FIELDS
    assert NOTICE == "识别可能有误；所有标注均为疑似、待核查，必须人工复核"
    assert REVIEW_CHOICES == ["未复核", "确认", "修改", "驳回"]


def test_new_segment_extra_fields_and_types():
    seg = new_segment(3, 4.567, speaker="导游", source="asr")
    assert list(seg)[:5] == FIELDS
    assert seg["speaker"] == "导游"
    assert seg["source"] == "asr"
    assert isinstance(seg["start"], float) and seg["start"] == 3.0
    assert seg["end"] == 4.57


def test_write_json_adds_notice_and_ai_mark(tmp_path):
    meta = {"file": "a.wav"}
    path = tmp_path / "sub" / "out.json"
    write_json(path, [], meta)
    assert "notice" not in meta, "write_json 不应改动调用方传进来的 meta"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["meta"]["notice"] == NOTICE
    assert "人工智能技术自动生成" in data["meta"]["generated_by"]
    assert data["segments"] == []
    # 中文直接写出，不转成 \uXXXX
    assert "识别可能有误" in path.read_text(encoding="utf-8")


def test_write_json_accepts_numpy_numbers(tmp_path):
    import numpy as np

    seg = new_segment(0, 1, score=np.float32(0.5), speaker_id=np.int64(2))
    path = tmp_path / "np.json"
    write_json(path, [seg], {"duration": np.float64(3.25), "wav": tmp_path / "a.wav"})
    segments, meta = read_json(path)
    assert segments[0]["score"] == 0.5 and segments[0]["speaker_id"] == 2
    assert meta["duration"] == 3.25
    assert meta["wav"] == str(tmp_path / "a.wav")


def test_read_json_fills_missing_notice(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"meta": {"file": "x.wav"}, "segments": []}, ensure_ascii=False), encoding="utf-8")
    segments, meta = read_json(path)
    assert segments == []
    assert meta["notice"] == NOTICE


def test_csv_only_writes_optional_columns_that_appear(tmp_path):
    segments = [new_segment(0, 1, text="你好")]
    path = tmp_path / "s.csv"
    write_csv(path, segments)
    header = path.read_text(encoding="utf-8-sig").splitlines()[0].split(",")
    # new_segment 默认带 review，所以有“复核结论”一列；没出现过的可选字段不写
    assert header == ["开始时间（秒）", "结束时间（秒）", "说话人", "文字内容", "标签", "复核结论"]


def test_csv_list_and_correction_format(tmp_path):
    seg = new_segment(0, 1, numbers=["2800元", "15:40"], entities=["雾隐行舟旅行社"],
                      corrections=[{"from": "雾影行走", "to": "雾隐行舟"}, {"from": "晓月阁", "to": "晓月银坊"}])
    path = tmp_path / "s.csv"
    write_csv(path, [seg])
    text = path.read_text(encoding="utf-8-sig")
    assert "2800元；15:40" in text
    assert "雾影行走→雾隐行舟；晓月阁→晓月银坊" in text
    got = read_csv(path)[0]
    assert got["numbers"] == ["2800元", "15:40"]
    assert got["entities"] == ["雾隐行舟旅行社"]
    assert got["corrections"] == [{"from": "雾影行走", "to": "雾隐行舟"}, {"from": "晓月阁", "to": "晓月银坊"}]


def test_csv_number_columns_types(tmp_path):
    seg = new_segment(1.5, 2.25, speaker="说话人1", speaker_id=1, score=0.875, category="费用", label="疑似·费用")
    path = tmp_path / "s.csv"
    write_csv(path, [seg])
    got = read_csv(path)[0]
    assert got == seg
    assert isinstance(got["start"], float) and isinstance(got["end"], float)
    assert isinstance(got["score"], float)
    assert got["speaker_id"] == 1 and isinstance(got["speaker_id"], int)


def test_csv_empty_optional_cell_is_left_out(tmp_path):
    # 一段有 clip、另一段没有：没有的那段读回来也没有 clip 键
    a = new_segment(0, 1, label="疑似·费用", clip="clip_001.wav", speaker_id=None)
    b = new_segment(1, 2)
    path = tmp_path / "s.csv"
    write_csv(path, [a, b])
    got_a, got_b = read_csv(path)
    assert got_a["clip"] == "clip_001.wav"
    assert "clip" not in got_b
    assert "speaker_id" not in got_a


def test_csv_ignores_unknown_keys(tmp_path):
    seg = new_segment(0, 1, gold_label="费用")
    path = tmp_path / "s.csv"
    write_csv(path, [seg])
    assert "gold_label" not in path.read_text(encoding="utf-8-sig")


def test_read_csv_missing_column_is_clear(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("开始时间（秒）,说话人\n1,导游\n", encoding="utf-8-sig")
    with pytest.raises(ValueError, match="结束时间"):
        read_csv(path)


def test_read_csv_bad_number_is_clear(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("开始时间（秒）,结束时间（秒）,说话人,文字内容,标签\n一点五,2,导游,好,\n", encoding="utf-8-sig")
    with pytest.raises(ValueError, match="第 2 行"):
        read_csv(path)


def test_empty_segment_list_csv(tmp_path):
    path = tmp_path / "empty.csv"
    write_csv(path, [])
    assert path.read_text(encoding="utf-8-sig").splitlines() == ["开始时间（秒）,结束时间（秒）,说话人,文字内容,标签"]
    assert read_csv(path) == []


def test_module_docstring_has_three_parts():
    for part in ["基线做法", "可改进方向", "测评指标"]:
        assert part in schema.__doc__


def test_schema_light_import():
    code = ("import sys; import pipeline.schema; "
            "assert not {'sherpa_onnx', 'gradio', 'tensorflow'} & set(sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
