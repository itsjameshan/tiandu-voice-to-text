"""pipeline/metrics.py 的测试：说话人标错比例、混淆矩阵、各类准确率召回率、误报率、片段起止误差。

例子都很小，答案可以手算对照（每个测试里写了怎么算）。
"""
import subprocess
import sys

import pytest

from conftest import ROOT
from pipeline.metrics import (
    clip_boundary_error,
    confusion_matrix,
    false_positive_rate,
    per_class_pr,
    speaker_error_rate,
)

# ---------------- 说话人标错的时长比例 ----------------


def test_speaker_error_perfect_and_swapped():
    ref = [(0.0, 5.0, "导游"), (5.0, 10.0, "游客")]
    # 完全相同 → 0
    assert speaker_error_rate(ref, ref) == pytest.approx(0.0)
    # 标签互换（1↔2）：最优对应把"说话人2"对到"导游"，仍然全对 → 0
    swapped = [(0.0, 5.0, "说话人2"), (5.0, 10.0, "说话人1")]
    assert speaker_error_rate(ref, swapped) == pytest.approx(0.0)
    # 一半时间标错：结果把 0~10 秒全标成同一个人，只能对上其中 5 秒 → 0.5
    half = [(0.0, 10.0, "说话人1")]
    assert speaker_error_rate(ref, half) == pytest.approx(0.5, abs=0.01)


def test_speaker_error_half_with_two_hyp_speakers():
    # 标准答案：导游 0~10，游客 10~20；结果每 5 秒换一次人。
    # 重叠表：导游-说话人1 5 秒、导游-说话人2 5 秒、游客-说话人1 5 秒、游客-说话人2 5 秒，
    # 怎么对应都只能对上 10 秒（共 20 秒）→ 0.5
    ref = [(0.0, 10.0, "导游"), (10.0, 20.0, "游客")]
    hyp = [(0.0, 5.0, "说话人1"), (5.0, 10.0, "说话人2"), (10.0, 15.0, "说话人1"), (15.0, 20.0, "说话人2")]
    assert speaker_error_rate(ref, hyp) == pytest.approx(0.5, abs=0.01)


def test_speaker_error_only_counts_time_both_sides_speak():
    # 标准答案 0~10 秒是导游；结果只在 0~4 秒有人说话（4~10 秒漏了）。
    # 只看两边都有人说话的 0~4 秒，这 4 秒对上了 → 0（漏掉的时间不算进这个指标）
    ref = [(0.0, 10.0, "导游")]
    hyp = [(0.0, 4.0, "说话人1")]
    assert speaker_error_rate(ref, hyp) == pytest.approx(0.0)
    # 结果多出一个人：2~4 秒标成说话人2，最优对应是 导游↔说话人1，所以 2 秒标错（共 4 秒）→ 0.5
    hyp2 = [(0.0, 2.0, "说话人1"), (2.0, 4.0, "说话人2")]
    assert speaker_error_rate(ref, hyp2) == pytest.approx(0.5, abs=0.01)


def test_speaker_error_accepts_segment_dicts_and_handles_empty():
    ref = [(0.0, 3.0, "导游"), (3.0, 6.0, "游客")]
    hyp = [
        {"start": 0.0, "end": 3.0, "speaker": "说话人1", "text": "大家好", "label": ""},
        {"start": 3.0, "end": 6.0, "speaker": "说话人2", "text": "你好", "label": ""},
    ]
    assert speaker_error_rate(ref, hyp) == pytest.approx(0.0)
    # 两边没有同时说话的时间：没有可比的，返回 0.0
    assert speaker_error_rate(ref, []) == 0.0
    assert speaker_error_rate([], []) == 0.0


def test_speaker_error_step_changes_grid_not_answer():
    ref = [(0.0, 1.0, "导游"), (1.0, 2.0, "游客")]
    hyp = [(0.0, 1.5, "说话人1"), (1.5, 2.0, "说话人2")]
    # 1.0~1.5 秒标错（共 2 秒）→ 0.25，换成 0.05 秒一格也一样
    assert speaker_error_rate(ref, hyp) == pytest.approx(0.25, abs=0.01)
    assert speaker_error_rate(ref, hyp, step=0.05) == pytest.approx(0.25, abs=0.01)


# ---------------- 混淆矩阵、各类准确率和召回率 ----------------


def test_pr_and_confusion():
    labels = ["费用", "购物安排", "正常讲解"]
    gold = ["费用", "费用", "费用", "购物安排", "正常讲解"]
    pred = ["费用", "费用", "购物安排", "购物安排", "费用"]
    # 行是标准答案，列是预测：
    #   费用 → 费用 2 句、购物安排 1 句
    #   购物安排 → 购物安排 1 句
    #   正常讲解 → 费用 1 句
    assert confusion_matrix(gold, pred, labels) == [
        [2, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
    ]
    pr = per_class_pr(gold, pred, labels)
    assert list(pr) == labels
    # 费用：预测成费用的 3 句里对 2 句 → 准确率 2/3；标准答案 3 句费用找回 2 句 → 召回率 2/3
    assert pr["费用"]["precision"] == pytest.approx(2 / 3)
    assert pr["费用"]["recall"] == pytest.approx(2 / 3)
    assert (pr["费用"]["tp"], pr["费用"]["n_pred"], pr["费用"]["n_gold"]) == (2, 3, 3)
    # 购物安排：预测 2 句对 1 句 → 0.5；标准答案 1 句找回 1 句 → 1.0
    assert pr["购物安排"]["precision"] == pytest.approx(0.5)
    assert pr["购物安排"]["recall"] == pytest.approx(1.0)
    # 正常讲解：一句也没预测成它 → 准确率按 0 算；标准答案 1 句没找回 → 0
    assert pr["正常讲解"]["precision"] == 0.0
    assert pr["正常讲解"]["recall"] == 0.0
    assert pr["正常讲解"]["n_pred"] == 0


def test_confusion_rejects_bad_input():
    labels = ["费用", "其他"]
    with pytest.raises(ValueError):
        confusion_matrix(["费用"], ["费用", "其他"], labels)  # 长度不一样
    with pytest.raises(ValueError):
        confusion_matrix(["费用"], ["投诉"], labels)  # 不在类别表里
    with pytest.raises(ValueError):
        per_class_pr(["导游"], ["费用"], labels)


# ---------------- 误报率 ----------------


def test_false_positive_rate():
    gold = ["正常讲解", "正常讲解", "费用"]
    pred = ["费用", "正常讲解", "费用"]
    # 标准答案是正常讲解的 2 句里，1 句被标成疑似类别 → 0.5
    assert false_positive_rate(gold, pred) == pytest.approx(0.5)


def test_false_positive_rate_details():
    # 正常讲解被分成"其他"不算误报（其他不会被标出来）
    assert false_positive_rate(["正常讲解", "正常讲解"], ["其他", "正常讲解"]) == 0.0
    # 5 个疑似类别都算误报（类别名"威胁消费"不变，不是显示名"消费施压"）
    gold = ["正常讲解"] * 5
    pred = ["购物安排", "费用", "行程变更", "服务态度", "威胁消费"]
    assert false_positive_rate(gold, pred) == pytest.approx(1.0)
    # 标准答案是费用、却被分成购物安排：分错了，但不算误报
    assert false_positive_rate(["费用", "正常讲解"], ["购物安排", "正常讲解"]) == 0.0
    # 没有正常讲解的句子：没有可比的，返回 0.0
    assert false_positive_rate(["费用"], ["费用"]) == 0.0
    with pytest.raises(ValueError):
        false_positive_rate(["正常讲解"], [])


# ---------------- 片段起止误差 ----------------


def test_clip_boundary_error():
    ref = [(10.0, 20.0, "费用"), (30.0, 40.0, "威胁消费"), (50.0, 55.0, "服务态度")]
    hyp = [(11.0, 19.0), (28.0, 41.0), (70.0, 80.0)]
    result = clip_boundary_error(ref, hyp)
    # 配上 2 对：(10,20)↔(11,19)，(30,40)↔(28,41)
    # 起点误差 |11-10|=1、|28-30|=2 → 平均 1.5；终点误差 |19-20|=1、|41-40|=1 → 平均 1.0
    assert result["start_mae"] == pytest.approx(1.5)
    assert result["end_mae"] == pytest.approx(1.0)
    assert result["n_matched"] == 2
    assert result["n_ref"] == 3 and result["n_hyp"] == 3
    # 标准答案里 (50,55) 没配上（漏标）；结果里 (70,80) 没配上（多标）
    assert result["unmatched_ref"] == 1
    assert result["unmatched_hyp"] == 1
    assert result["unmatched"] == 2


def test_clip_boundary_pairs_by_largest_overlap():
    # 结果 (1,11) 和 (0,10) 重叠 9 秒、和 (10,12) 重叠 1 秒 → 配给 (0,10)
    ref = [(0.0, 10.0, "费用"), (10.0, 12.0, "费用")]
    hyp = [{"start": 1.0, "end": 11.0, "label": "疑似·费用"}]
    result = clip_boundary_error(ref, hyp)
    assert result["n_matched"] == 1
    assert result["start_mae"] == pytest.approx(1.0)
    assert result["end_mae"] == pytest.approx(1.0)
    assert (result["unmatched_ref"], result["unmatched_hyp"]) == (1, 0)


def test_clip_boundary_no_overlap_means_unmatched():
    # 只是挨着、没有重叠，不算配上；没有配上的片段时平均误差为 None
    result = clip_boundary_error([(0.0, 5.0, "费用")], [(5.0, 8.0, 0)])
    assert result["n_matched"] == 0
    assert result["start_mae"] is None and result["end_mae"] is None
    assert (result["unmatched_ref"], result["unmatched_hyp"]) == (1, 1)
    empty = clip_boundary_error([], [])
    assert empty["n_matched"] == 0 and empty["unmatched"] == 0


# ---------------- 轻量导入 ----------------


def test_metrics_import_is_light():
    # 训练脚本会在只装了 TensorFlow、numpy 的机房电脑上用到这个模块：顶层只能导入标准库和 numpy
    code = (
        "import sys, pipeline.metrics\n"
        "heavy = {'scipy', 'sherpa_onnx', 'gradio', 'tensorflow', 'matplotlib'}\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in heavy))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", check=True
    )
    assert proc.stdout.strip() == "[]"
