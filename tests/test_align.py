"""测试 pipeline/align.py：字错率拆分（错字、漏字、多字）和逐段对照。"""
import pytest

from pipeline.align import align_segments, cer_details
from pipeline.text_norm import normalize_for_cer


def test_cer_details():
    # 视频里的例子：错 2 个字，参考文本 7 个字
    d = cer_details("雾隐行舟旅行社", "雾影行走旅行社")
    assert d["sub"] == 2
    assert d["dele"] == 0
    assert d["ins"] == 0
    assert d["n_ref"] == 7
    assert d["cer"] == pytest.approx(2 / 7)

    d = cer_details("你好", "你好啊")
    assert d["ins"] == 1
    assert d["sub"] == 0 and d["dele"] == 0
    assert d["cer"] == pytest.approx(1 / 2)


def test_cer_details_counts_deletions():
    d = cer_details("你好啊", "你好")
    assert (d["sub"], d["dele"], d["ins"], d["n_ref"]) == (0, 1, 0, 3)
    assert d["cer"] == pytest.approx(1 / 3)


def test_cer_details_ignores_punctuation_width_and_case():
    # 标点、空格、全角半角、大小写的差别都不算错（两边先 normalize_for_cer）
    d = cer_details("你好，ＡＢＣ！\n今天 12 点。", "你好abc今天12点")
    assert d["cer"] == 0.0
    assert d["n_ref"] == len("你好ABC今天12点")


def test_cer_details_empty_reference():
    assert cer_details("", "")["cer"] == 0.0
    assert cer_details("。！", " ")["cer"] == 0.0  # 归一后两边都是空的
    d = cer_details("", "啊")
    assert d["cer"] == 1.0
    assert d["n_ref"] == 0
    assert d["ins"] == 1


def test_align_segments():
    rows = align_segments(["今天去石林", "下午回昆名"], "今天去石林。下午回昆明。")
    assert len(rows) == 2
    assert rows[0]["hyp"] == "今天去石林"
    assert rows[0]["ref"] == "今天去石林。"
    assert rows[0]["diff"] is False
    assert rows[0]["cer"] == 0.0
    assert rows[1]["hyp"] == "下午回昆名"
    assert rows[1]["diff"] is True
    assert rows[1]["ref"] == "下午回昆明。"
    assert rows[1]["cer"] == pytest.approx(1 / 5)


def test_align_segments_missing_char_is_shown():
    # 识别漏了"林"：参考文本里的"林"仍要出现在某一段里，这一段标为不同
    rows = align_segments(["今天去石", "下午回昆明"], "今天去石林。下午回昆明。")
    assert rows[0]["ref"] == "今天去石林。"
    assert rows[0]["diff"] is True
    assert rows[1]["ref"] == "下午回昆明。"
    assert rows[1]["diff"] is False


def test_align_segments_missing_first_char_goes_to_next_segment():
    # 识别漏了后一句开头的"下"：在句号处分开，"下"归第 2 段，第 1 段仍然一致
    rows = align_segments(["今天去石林", "午回昆明"], "今天去石林。下午回昆明。")
    assert rows[0]["ref"] == "今天去石林。"
    assert rows[0]["diff"] is False
    assert rows[1]["ref"] == "下午回昆明。"
    assert rows[1]["diff"] is True
    assert rows[1]["cer"] == pytest.approx(1 / 5)


def test_align_segments_missed_sentence_at_edges():
    # 第 1 段什么也没识别出来：第 1 句仍然分给第 1 段，不会挤到第 2 段里
    rows = align_segments(["", "下午回昆明"], "今天去石林。下午回昆明。")
    assert [r["ref"] for r in rows] == ["今天去石林。", "下午回昆明。"]
    assert [r["diff"] for r in rows] == [True, False]
    # 最后一段什么也没识别出来：最后一句分给最后一段
    rows = align_segments(["今天去石林", ""], "今天去石林。下午回昆明。")
    assert [r["ref"] for r in rows] == ["今天去石林。", "下午回昆明。"]
    assert [r["diff"] for r in rows] == [False, True]


def test_align_segments_multiline_reference():
    # 参考文本按台词分行；对照里每段的参考文字两头不带换行
    reference = "喂，听得到吗？\n听得到！\n好，那我先简单介绍一下。"
    rows = align_segments(["喂听得到吗", "听得到", "好那我先简单介绍一下"], reference)
    assert [r["ref"] for r in rows] == ["喂，听得到吗？", "听得到！", "好，那我先简单介绍一下。"]
    assert [r["diff"] for r in rows] == [False, False, False]


def test_align_segments_covers_whole_reference():
    # 不管识别错成什么样，参考文本的每个字都恰好分到一段里，不丢不重
    reference = "我们现在已经出了昆明，往石林方向开。\n走高速大概还要一个多小时。"
    segments = ["嗯我们现在出了昆名往", "", "石林方向开开走高速大概", "还要一个多小时啊"]
    rows = align_segments(segments, reference)
    assert len(rows) == len(segments)
    assert "".join(normalize_for_cer(r["ref"]) for r in rows) == normalize_for_cer(reference)
    assert [r["hyp"] for r in rows] == segments
    for r in rows:
        assert r["diff"] == (normalize_for_cer(r["ref"]) != normalize_for_cer(r["hyp"]))
        assert r["cer"] >= 0.0


def test_align_segments_empty_inputs():
    assert align_segments([], "今天去石林。") == []
    rows = align_segments(["你好"], "")
    assert rows == [{"hyp": "你好", "ref": "", "diff": True, "cer": 1.0}]
