"""tools/train_classifier.py（第 6、7 组的 TensorFlow 话术分类训练脚本）的测试。

本环境（.venv）没有装 TensorFlow：需要 TensorFlow 的测试用 pytest.importorskip 自动跳过；
不需要 TensorFlow 的部分（导入是否轻量、按组留一的划分、分层随机划分、混淆矩阵、误报率、报告）照常测。
"""
import csv
import importlib.util
import json
import subprocess
import sys

import pytest
from conftest import ROOT

from pipeline.data import FLAG_LABELS, LABEL_NAMES, load_lines
from pipeline.tf_classifier import is_tf_available, load_classifier


def _load_tool():
    spec = importlib.util.spec_from_file_location("train_classifier", ROOT / "tools" / "train_classifier.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------- 计划里列出的测试 ----------


def test_train_light_imports():
    """导入训练脚本和它用到的模块（含第 6、7 组的 build_model）后，不能把 sherpa_onnx、gradio 带进来。"""
    code = (
        "import sys; import tools.train_classifier; "
        "import pipeline.text_norm, pipeline.data, pipeline.tf_classifier; "
        "import pipeline.groups.g6_classifier_a, pipeline.groups.g7_classifier_b; "
        "bad = [m for m in ('sherpa_onnx', 'gradio') if m in sys.modules]; "
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("model_name", ["baseline", "g6", "g7"])
def test_train_smoke(tmp_path, model_name):
    pytest.importorskip("tensorflow")
    train = _load_tool()
    out = tmp_path / "model"
    rc = train.main(["--model", model_name, "--epochs", "1", "--eval", "none",
                     "--out", str(out), "--report-dir", str(tmp_path / "reports")])
    assert rc == 0
    for name in ("model.h5", "vocab.json", "meta.json"):
        assert (out / name).is_file(), name
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["labels"] == LABEL_NAMES
    assert meta["max_len"] == train.MAX_LEN
    assert meta["model"] == model_name
    assert meta["used_extra"] is True
    assert meta["trained_at"]
    assert meta["evaluation"]["mode"] == "none"
    predict = load_classifier(out)
    assert predict is not None
    result = predict(["这个手镯两千八百块，买不买？", "大家跟紧我，前面台阶有点滑。"])
    assert len(result) == 2
    assert all(label in LABEL_NAMES for label in result)


# ---------- 补充的测试：需要 TensorFlow ----------


def test_train_logo_report(tmp_path):
    """按组留一（1 轮训练，只看流程）：报告、混淆矩阵、预测表都写出来，测试集只有 1055 句剧本台词。"""
    pytest.importorskip("tensorflow")
    train = _load_tool()
    out = tmp_path / "model"
    report_dir = tmp_path / "reports"
    rc = train.main(["--model", "baseline", "--epochs", "1", "--eval", "logo", "--no-extra",
                     "--out", str(out), "--report-dir", str(report_dir)])
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["used_extra"] is False
    evaluation = meta["evaluation"]
    assert evaluation["mode"] == "logo"
    matrix = evaluation["confusion"]
    assert len(matrix) == 7 and all(len(row) == 7 for row in matrix)
    assert sum(map(sum, matrix)) == len(load_lines())  # 只测人写的剧本台词
    assert 0.0 <= evaluation["fp_rate"] <= 1.0
    assert 0.0 <= evaluation["fp_rate_g8"] <= 1.0
    assert len(evaluation["by_group"]) == 8

    report = report_dir / "train_baseline_logo_no_extra.md"
    text = report.read_text(encoding="utf-8")
    assert train.LIMITS_NOTE in text
    assert "按组留一" in text and "混淆矩阵" in text and "误报率" in text and "第 8 组" in text
    with open(report_dir / "train_baseline_logo_no_extra_predictions.csv", encoding="utf-8-sig", newline="") as f:
        assert len(list(csv.DictReader(f))) == len(load_lines())


# ---------- 补充的测试：不需要 TensorFlow ----------


def test_script_import_is_lab_safe():
    """机房的 Python 只装了 TensorFlow 和 numpy：只导入训练脚本时，不能带进 TensorFlow、PyYAML、cn2an、pypinyin 等。"""
    code = (
        "import sys; import tools.train_classifier; "
        "bad = [m for m in ('sherpa_onnx', 'gradio', 'tensorflow', 'yaml', 'cn2an', 'pypinyin', 'scipy') "
        "if m in sys.modules]; assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_main_without_tensorflow(tmp_path, capsys):
    if is_tf_available():
        pytest.skip("这台电脑装了 TensorFlow")
    train = _load_tool()
    out = tmp_path / "model"
    rc = train.main(["--model", "g6", "--epochs", "1", "--eval", "none", "--out", str(out),
                     "--report-dir", str(tmp_path / "reports")])
    assert rc == 1
    assert "TensorFlow" in capsys.readouterr().out
    assert not out.exists()


def test_get_build_model():
    from pipeline.groups import g6_classifier_a, g7_classifier_b
    from pipeline.tf_classifier import build_baseline_model

    train = _load_tool()
    assert train.get_build_model("baseline") is build_baseline_model
    assert train.get_build_model("g6") is g6_classifier_a.build_model
    assert train.get_build_model("g7") is g7_classifier_b.build_model
    with pytest.raises(ValueError):
        train.get_build_model("g9")


def test_default_dirs():
    train = _load_tool()
    assert train.default_out_dir("g6") == ROOT / "models" / "classifier_g6"
    assert train.default_out_dir("g7") == ROOT / "models" / "classifier_g7"
    # 基线模型存到步骤 6 的 tf_model 做法读取的文件夹 models/classifier/
    assert train.default_out_dir("baseline") == ROOT / "models" / "classifier"
    assert train.default_report_dir("g6") == ROOT / "reports" / "g6"
    assert train.default_report_dir("g7") == ROOT / "reports" / "g7"
    assert train.default_report_dir("baseline") == ROOT / "reports" / "baseline"


def test_load_extra():
    train = _load_tool()
    extra = train.load_extra()
    assert len(extra) >= 900
    assert all(row["text"] and row["label"] in LABEL_NAMES for row in extra)


def test_logo_folds():
    train = _load_tool()
    lines = load_lines()
    folds = train.logo_folds(lines)
    assert [group for group, _, _ in folds] == list(range(1, 9))
    seen = []
    for group, train_idx, test_idx in folds:
        assert test_idx and all(lines[i]["group"] == group for i in test_idx)
        assert all(lines[i]["group"] != group for i in train_idx)
        assert len(train_idx) + len(test_idx) == len(lines)
        seen.extend(test_idx)
    assert sorted(seen) == list(range(len(lines)))  # 每句台词正好被测一次


def test_random_split_stratified():
    train = _load_tool()
    labels = [line["label"] for line in load_lines()]
    train_idx, test_idx = train.random_split(labels, test_ratio=0.2, seed=0)
    assert not set(train_idx) & set(test_idx)
    assert sorted(train_idx + test_idx) == list(range(len(labels)))
    for label in LABEL_NAMES:
        total = labels.count(label)
        in_test = sum(1 for i in test_idx if labels[i] == label)
        assert abs(in_test - round(total * 0.2)) <= 1, label
    assert train.random_split(labels, test_ratio=0.2, seed=0) == (train_idx, test_idx)  # 同一个种子结果一样


def test_metric_helpers():
    train = _load_tool()
    labels = ["费用", "正常讲解", "其他"]
    gold = ["费用", "费用", "正常讲解", "正常讲解", "其他"]
    pred = ["费用", "其他", "费用", "正常讲解", "其他"]
    assert train._confusion_matrix(gold, pred, labels) == [[1, 0, 1], [1, 1, 0], [0, 0, 1]]
    pr = train._per_class_pr(gold, pred, labels)
    assert pr["费用"]["precision"] == pytest.approx(0.5)
    assert pr["费用"]["recall"] == pytest.approx(0.5)
    assert pr["其他"]["precision"] == pytest.approx(0.5)
    assert pr["其他"]["recall"] == pytest.approx(1.0)
    assert pr["正常讲解"]["support"] == 2
    # 计划 Task 19 的例子：正常讲解 2 句，1 句被标成疑似 → 0.5
    assert train._false_positive_rate(["正常讲解", "正常讲解", "费用"], ["费用", "正常讲解", "费用"]) == 0.5
    # 没有正常讲解的句子时为 0；"其他"不算疑似
    assert train._false_positive_rate(["费用"], ["费用"]) == 0.0
    assert train._false_positive_rate(["正常讲解"], ["其他"]) == 0.0


def test_summarize_and_write_report(tmp_path):
    train = _load_tool()
    lines = load_lines()
    gold = [line["label"] for line in lines]
    pred = ["正常讲解" if label == "正常讲解" else label for label in gold]
    # 让第 8 组的前两句正常讲解被标成疑似
    g8_normal = [i for i, line in enumerate(lines) if line["group"] == 8 and line["label"] == "正常讲解"]
    for i in g8_normal[:2]:
        pred[i] = FLAG_LABELS[0]
    summary = train.summarize(lines, pred, mode="logo")
    assert summary["mode"] == "logo"
    assert summary["n_test"] == len(lines)
    assert summary["fp_count_g8"] == 2
    assert summary["fp_rate_g8"] == pytest.approx(2 / len(g8_normal))
    assert summary["fp_rate"] == pytest.approx(2 / gold.count("正常讲解"))
    assert len(summary["by_group"]) == 8

    info = {"model": "g6", "eval": "logo", "used_extra": True, "n_extra": 964, "epochs": 15,
            "max_len": train.MAX_LEN, "trained_at": "2026-10-07T10:00:00", "tensorflow": "2.x"}
    md_path, csv_path = train.write_report(tmp_path / "reports", info, summary, lines, pred)
    assert md_path.name == "train_g6_logo_extra.md"
    text = md_path.read_text(encoding="utf-8")
    assert train.LIMITS_NOTE in text
    assert "混淆矩阵" in text and "误报率" in text and "第 8 组" in text and "补充句子" in text
    for label in LABEL_NAMES:
        assert label in text
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(lines)
    assert {"script_id", "group", "text", "label", "predicted"} <= set(rows[0])
