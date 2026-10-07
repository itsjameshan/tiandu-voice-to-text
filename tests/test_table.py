"""界面表格与段落互转（pipeline/table.py）的测试。

用户在界面表格里改了说话人、文字、标签、复核结论、复核意见后再导出，导出必须用改过的内容
（计划 Review Focus 第 4 条）。标签只能是 5 个"疑似·…"或空；复核结论只能是 REVIEW_CHOICES 之一。
"""
import pytest

from pipeline.data import FLAG_OUTPUTS
from pipeline.schema import REVIEW_CHOICES, new_segment
from pipeline.step8_report import FORBIDDEN_WORDS
from pipeline.table import TABLE_HEADERS, rows_to_segments, segments_to_rows

LEGAL_LABELS = {""} | set(FLAG_OUTPUTS)


def _segments() -> list[dict]:
    return [
        new_segment(1.0, 3.5, speaker="说话人1", speaker_id=1, text="这个手镯今天2800元。", label="疑似·费用",
                    category="费用", numbers=["2800元"], entities=["雾隐行舟旅行社"]),
        new_segment(4.0, 6.0, speaker="说话人2", speaker_id=2, text="大家注意安全。", category="正常讲解"),
        new_segment(7.0, 9.25, speaker="说话人1", speaker_id=1, text="下午3点集合。", category="其他",
                    numbers=["15:00", "3点"]),
    ]


def test_table_headers():
    assert TABLE_HEADERS == ["序号", "开始", "结束", "说话人", "文字", "标签", "数字", "复核结论", "复核意见"]


def test_segments_to_rows():
    rows = segments_to_rows(_segments())
    assert len(rows) == 3
    assert all(len(row) == len(TABLE_HEADERS) for row in rows)
    assert rows[0] == [1, 1.0, 3.5, "说话人1", "这个手镯今天2800元。", "疑似·费用", "2800元", "未复核", ""]
    assert rows[2][6] == "15:00；3点"
    assert rows[1][5] == ""


def test_rows_roundtrip_unchanged():
    segments = _segments()
    assert rows_to_segments(segments_to_rows(segments), segments) == segments


def test_rows_roundtrip_with_edits():
    segments = _segments()
    rows = segments_to_rows(segments)
    rows[1][3] = "导游"
    rows[1][7] = "确认"
    rows[1][8] = "已听原声"
    rows[2][4] = "下午3点在停车场集合。"
    result = rows_to_segments(rows, segments)

    assert result[1]["speaker"] == "导游"
    assert result[1]["speaker_id"] == 2  # 编号不变
    assert result[1]["review"] == "确认"
    assert result[1]["review_note"] == "已听原声"
    assert result[2]["text"] == "下午3点在停车场集合。"
    assert result[0] == segments[0]  # 没改的行不变
    # 传进来的段落没有被改动
    assert segments[1]["speaker"] == "说话人2" and "review_note" not in segments[1]

    # 把标签改成不合法的词：标签被清空，复核意见里记一条提示
    rows[0][5] = "违规"
    result = rows_to_segments(rows, segments)
    assert result[0]["label"] == ""
    assert "标签" in result[0]["review_note"]
    for word in FORBIDDEN_WORDS:  # 提示里不能把不合法的词抄进导出文件
        assert word not in result[0]["review_note"]


def test_invalid_label_keeps_user_note():
    segments = _segments()
    rows = segments_to_rows(segments)
    rows[0][5] = "投诉"
    rows[0][8] = "导游原话如此"
    result = rows_to_segments(rows, segments)
    assert result[0]["label"] == ""
    assert result[0]["review_note"].startswith("导游原话如此")
    assert len(result[0]["review_note"]) > len("导游原话如此")


@pytest.mark.parametrize("typed, label, category", [
    ("疑似·消费施压", "疑似·消费施压", "威胁消费"),
    ("消费施压", "疑似·消费施压", "威胁消费"),
    ("威胁消费", "疑似·消费施压", "威胁消费"),
    ("疑似·威胁消费", "疑似·消费施压", "威胁消费"),
    ("费用", "疑似·费用", "费用"),
    ("疑似费用", "疑似·费用", "费用"),
    (" 疑似.行程变更 ", "疑似·行程变更", "行程变更"),
    ("疑似：服务态度", "疑似·服务态度", "服务态度"),
    ("疑似·购物安排", "疑似·购物安排", "购物安排"),
    ("正常讲解", "", "正常讲解"),
    ("其他", "", "其他"),
])
def test_label_accepts_common_spellings(typed, label, category):
    """"·"在键盘上不好打，常见写法都认；类别跟着标签改（消费施压 → 威胁消费）。"""
    segments = _segments()
    rows = segments_to_rows(segments)
    rows[1][5] = typed
    result = rows_to_segments(rows, segments)
    assert result[1]["label"] == label
    assert result[1]["category"] == category
    assert "review_note" not in result[1]


def test_clear_label_drops_flag_category():
    """人工把"疑似·费用"清空后，类别不能还写着"费用"（前后矛盾）。"""
    segments = _segments()
    rows = segments_to_rows(segments)
    rows[0][5] = ""
    result = rows_to_segments(rows, segments)
    assert result[0]["label"] == ""
    assert result[0].get("category") != "费用"


def test_labels_always_legal():
    segments = _segments()
    rows = segments_to_rows(segments)
    for typed in ["疑似·违约", "疑似·", "abc", "疑似·费用费用", "疑似·正常讲解"]:
        rows[0][5] = typed
        assert rows_to_segments(rows, segments)[0]["label"] in LEGAL_LABELS


def test_review_choice_validated():
    segments = _segments()
    rows = segments_to_rows(segments)
    rows[0][7] = "通过"
    rows[1][7] = " 驳回 "
    rows[2][7] = ""
    result = rows_to_segments(rows, segments)
    assert result[0]["review"] == "未复核"
    assert result[1]["review"] == "驳回"
    assert result[2]["review"] == "未复核"
    assert {seg["review"] for seg in result} <= set(REVIEW_CHOICES)


def test_empty_speaker_becomes_unknown():
    segments = _segments()
    rows = segments_to_rows(segments)
    rows[0][3] = "  "
    assert rows_to_segments(rows, segments)[0]["speaker"] == "未知"


def test_rows_matched_by_number():
    """表格行的顺序被打乱（例如按说话人排序）时，按"序号"列对回原来的段落。"""
    segments = _segments()
    rows = segments_to_rows(segments)
    rows[2][3] = "导游"
    rows.reverse()
    result = rows_to_segments(rows, segments)
    assert [seg["start"] for seg in result] == [1.0, 4.0, 7.0]
    assert result[2]["speaker"] == "导游"


def test_rows_from_dataframe():
    """界面的 gr.Dataframe 可能交来 pandas 表格，空单元格是 NaN 或 None。"""
    pd = pytest.importorskip("pandas")
    segments = _segments()
    rows = segments_to_rows(segments)
    rows[0][8] = None
    rows[1][3] = "导游"
    frame = pd.DataFrame(rows, columns=TABLE_HEADERS)
    frame.loc[2, "复核意见"] = float("nan")
    result = rows_to_segments(frame, segments)
    assert result[1]["speaker"] == "导游"
    assert "review_note" not in result[0]
    assert "review_note" not in result[2]
    assert result[0]["label"] == "疑似·费用"
