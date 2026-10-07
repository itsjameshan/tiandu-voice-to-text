"""步骤 6 话术分类（pipeline/step6_classify.py、pipeline/tf_classifier.py）的测试。

- 关键词规则基线：输出只能是 7 个类别之一，标签只能是 5 个"疑似·…"或空字符串；
- 关键词表：短、可解释，不能把台词原句抄成关键词；
- 在 1055 句剧本台词上打印各类召回率和"正常讲解"的误报率（只打印，不设下限）；
- 选了 TensorFlow 模型做法但用不了（没装 TensorFlow、没训练模型）时，退回关键词规则并给出中文警告。

本环境没有安装 TensorFlow，需要 TensorFlow 的测试用 pytest.importorskip 自动跳过。
"""
import json
import re
import subprocess
import sys
import warnings

import pytest
from conftest import ROOT

from pipeline import methods, step6_classify, tf_classifier
from pipeline.data import FLAG_LABELS, FLAG_OUTPUTS, LABEL_NAMES, label_output, load_lines
from pipeline.schema import new_segment
from pipeline.step6_classify import (
    RULES_PATH,
    classifier_dir,
    classify_segments,
    classify_with_model,
    count_hits,
    load_keywords,
    load_priority,
    pick_category,
)
from pipeline.text_norm import normalize_for_classification
from pipeline.tf_classifier import build_vocab, encode, is_tf_available, load_classifier, save_classifier

LEGAL_LABELS = {""} | set(FLAG_OUTPUTS)

# 台词去标点：只留汉字、字母、数字
_PUNCT = re.compile(r"[^0-9A-Za-z〇㐀-䶿一-鿿]")


def _baseline():
    return methods.get_method("classify", "baseline")


def _segments(texts):
    return [new_segment(i, i + 1, text=t) for i, t in enumerate(texts)]


# ---------- 计划里列出的测试 ----------


@pytest.mark.parametrize("index", range(200))
def test_labels_only_legal(index):
    line = load_lines()[index]
    result = classify_segments(_segments([line["text"]]), {})
    assert len(result) == 1
    assert result[0]["category"] in LABEL_NAMES
    assert result[0]["label"] in LEGAL_LABELS
    assert result[0]["label"] == label_output(result[0]["category"])


def test_keywords_not_copied_from_lines():
    lines = load_lines()
    raw_lines = {_PUNCT.sub("", line["text"]) for line in lines}
    norm_lines = {normalize_for_classification(line["text"]) for line in lines}
    for category, words in load_keywords().items():
        for word in words:
            assert len(word) <= 6, f"{category} 的关键词“{word}”太长（最多 6 个字）"
            assert _PUNCT.sub("", word) not in raw_lines, f"{category} 的关键词“{word}”和一句台词全文相同"
            assert normalize_for_classification(word) not in norm_lines, f"{category} 的关键词“{word}”和一句台词全文相同"


def test_rules_baseline_report():
    lines = load_lines()
    gold = [line["label"] for line in lines]
    pred = _baseline()([line["text"] for line in lines], {})
    assert len(pred) == len(gold) == 1055
    assert set(pred) <= set(LABEL_NAMES)

    print("\n关键词规则基线（24 个剧本 1055 句台词；剧本数据上的结果不代表真实录音上的效果）")
    print(f"{'类别':<6}{'句数':>6}{'命中':>6}{'召回率':>8}{'预测数':>8}{'准确率':>8}")
    for name in LABEL_NAMES:
        total = sum(1 for g in gold if g == name)
        hit = sum(1 for g, p in zip(gold, pred) if g == name and p == name)
        predicted = sum(1 for p in pred if p == name)
        recall = hit / total if total else 0.0
        precision = hit / predicted if predicted else 0.0
        assert 0.0 <= recall <= 1.0 and 0.0 <= precision <= 1.0
        print(f"{name:<6}{total:>6}{hit:>6}{recall:>8.2f}{predicted:>8}{precision:>8.2f}")
    accuracy = sum(1 for g, p in zip(gold, pred) if g == p) / len(gold)
    print(f"7 类总体正确率：{accuracy:.2f}")

    # 误报率 = 标准答案为"正常讲解"的句子中，被标成任一疑似类别的比例
    normal = [(g, p) for g, p in zip(gold, pred) if g == "正常讲解"]
    fp = sum(1 for _, p in normal if p in FLAG_LABELS)
    fp_rate = fp / len(normal)
    print(f"正常讲解误报率（全部 24 个剧本）：{fp}/{len(normal)} = {fp_rate:.2f}")
    normal_g8 = [(g, p) for line, g, p in zip(lines, gold, pred) if line["group"] == 8 and g == "正常讲解"]
    fp_g8 = sum(1 for _, p in normal_g8 if p in FLAG_LABELS)
    print(f"正常讲解误报率（第 8 组剧本）：{fp_g8}/{len(normal_g8)} = {fp_g8 / len(normal_g8):.2f}")
    assert 0.0 <= fp_rate <= 1.0


def test_tf_model_fallback(tmp_path):
    cfg = {"paths": {"models": str(tmp_path)}}  # 空目录：里面没有 classifier/ 模型
    texts = ["这个手镯两千八百块", "大家系好安全带", "大家多少支持一下，后面的安排我也好协调", "嗯"]
    with pytest.warns(UserWarning, match="关键词规则"):
        result = methods.get_method("classify", "tf_model")(texts, cfg)
    assert result == _baseline()(texts, cfg)


# ---------- 关键词表 ----------


def test_keywords_file_structure():
    data = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    assert data["_说明"]
    assert set(data["keywords"]) == set(LABEL_NAMES)  # 7 个类别都有键，包括"威胁消费"
    assert data["keywords"]["其他"] == []
    for name in LABEL_NAMES:
        if name != "其他":
            assert data["keywords"][name], f"{name} 没有关键词"
    assert sorted(data["priority"]) == sorted(LABEL_NAMES)
    assert load_keywords() == data["keywords"]
    assert load_priority() == data["priority"]


def test_keywords_unique_and_meaningful_after_norm():
    seen = {}
    for category, words in load_keywords().items():
        for word in words:
            norm = normalize_for_classification(word)
            # 归一后不能变成空的或只剩 #，否则什么句子都会命中
            assert norm.replace("#", ""), f"{category} 的关键词“{word}”归一后没有剩下文字"
            assert norm not in seen, f"关键词“{word}”同时出现在 {seen.get(norm)} 和 {category}"
            seen[norm] = category


def test_bad_keywords_file_gives_chinese_error(tmp_path, monkeypatch):
    bad = {"_说明": "测试", "priority": LABEL_NAMES, "keywords": {"费用": ["多少钱"]}}  # 少了 6 个类别
    path = tmp_path / "rules_keywords.json"
    path.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(step6_classify, "RULES_PATH", path)
    with pytest.raises(ValueError, match="类别"):
        load_keywords()


# ---------- 规则基线的小零件 ----------


def test_count_hits_and_pick_category():
    keywords = {"费用": ["#元", "多少钱"], "购物安排": ["手镯"], "其他": []}
    counts = count_hits(normalize_for_classification("这个手镯多少钱？2800元"), keywords)
    assert counts == {"费用": 2, "购物安排": 1, "其他": 0}
    priority = ["购物安排", "费用", "其他"]
    assert pick_category(counts, priority) == "费用"  # 次数多的胜
    assert pick_category({"费用": 1, "购物安排": 1, "其他": 0}, priority) == "购物安排"  # 平局按 priority
    assert pick_category({"费用": 0, "购物安排": 0, "其他": 0}, priority) == "其他"  # 都没命中


def test_baseline_examples():
    texts = ["这个手镯两千八百块", "大家多少支持一下，后面的安排我也好协调", "", "嗯"]
    result = _baseline()(texts, {})
    assert result[0] == "费用"
    assert result[1] == "威胁消费"
    assert result[2] == "其他" and result[3] == "其他"
    assert _baseline()([], {}) == []


# ---------- classify_segments ----------


def test_classify_segments_writes_category_and_label():
    segments = _segments(["甲", "乙", "丙", "丁"])
    fixed = ["威胁消费", "正常讲解", "费用", "其他"]
    result = classify_segments(segments, {}, method=lambda texts, cfg: list(fixed))
    assert [s["category"] for s in result] == fixed
    assert [s["label"] for s in result] == ["疑似·消费施压", "", "疑似·费用", ""]
    assert [s["label"] for s in result] == [label_output(c) for c in fixed]
    # 其他字段原样保留，传进来的段落不被修改
    assert [s["text"] for s in result] == ["甲", "乙", "丙", "丁"]
    assert all("category" not in s for s in segments)
    assert all(s["label"] == "" for s in segments)


def test_classify_segments_default_is_baseline():
    texts = ["这个手镯两千八百块", "大家系好安全带"]
    result = classify_segments(_segments(texts), {})
    assert [s["category"] for s in result] == _baseline()(texts, {})


def test_classify_segments_rejects_bad_method_output():
    segments = _segments(["甲", "乙"])
    with pytest.raises(ValueError, match="2"):
        classify_segments(segments, {}, method=lambda texts, cfg: ["费用"])  # 少了一个
    with pytest.raises(ValueError):
        classify_segments(segments, {}, method=lambda texts, cfg: ["费用", "违规"])  # 不认识的类别


def test_methods_registered():
    names = methods.available("classify")
    assert names[0] == "baseline"
    assert "tf_model" in names


def test_classifier_dir(tmp_path):
    assert classifier_dir({"paths": {"models": str(tmp_path)}}) == tmp_path / "classifier"
    assert classifier_dir({"paths": {"models": str(tmp_path)}}, "classifier_g6") == tmp_path / "classifier_g6"
    assert classifier_dir({}) == ROOT / "models" / "classifier"


# ---------- 模型做法：用得上时用模型，用不上时退回规则 ----------


def test_classify_with_model_uses_model_when_available(tmp_path, monkeypatch):
    monkeypatch.setattr(tf_classifier, "is_tf_available", lambda: True)
    monkeypatch.setattr(tf_classifier, "load_classifier", lambda model_dir: (lambda texts: ["行程变更"] * len(texts)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 能用模型时不应该有警告
        assert classify_with_model(["甲", "乙"], {}, tmp_path) == ["行程变更", "行程变更"]


def test_classify_with_model_no_tensorflow(tmp_path, monkeypatch):
    monkeypatch.setattr(tf_classifier, "is_tf_available", lambda: False)
    with pytest.warns(UserWarning, match="TensorFlow"):
        result = classify_with_model(["这个手镯两千八百块"], {}, tmp_path)
    assert result == _baseline()(["这个手镯两千八百块"], {})


def test_classify_with_model_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(tf_classifier, "is_tf_available", lambda: True)
    monkeypatch.setattr(tf_classifier, "load_classifier", lambda model_dir: None)
    with pytest.warns(UserWarning, match="model.h5"):
        result = classify_with_model(["大家系好安全带"], {}, tmp_path)
    assert result == _baseline()(["大家系好安全带"], {})


def test_classify_with_model_load_error(tmp_path, monkeypatch):
    def broken(model_dir):
        raise OSError("文件损坏")

    monkeypatch.setattr(tf_classifier, "is_tf_available", lambda: True)
    monkeypatch.setattr(tf_classifier, "load_classifier", broken)
    with pytest.warns(UserWarning, match="文件损坏"):
        result = classify_with_model(["嗯"], {}, tmp_path)
    assert result == ["其他"]


# ---------- tf_classifier（不需要 TensorFlow 的部分） ----------


def test_is_tf_available_returns_bool():
    assert isinstance(is_tf_available(), bool)


def test_load_classifier_missing_files_returns_none(tmp_path):
    assert load_classifier(tmp_path) is None
    assert load_classifier(tmp_path / "no_such_dir") is None


def test_encode():
    vocab = {"#": 2, "块": 3, "好": 4}
    # 先归一（两千八百 → #），按字查表，不认识的字是 1，不够长补 0，太长截断
    assert encode(["两千八百块", "好", "好呀好好好"], vocab, 4) == [[2, 3, 0, 0], [4, 0, 0, 0], [4, 1, 4, 4]]
    assert encode([], vocab, 4) == []


def test_build_vocab():
    vocab = build_vocab(["好好", "两千八百块，好"])
    # 0 留给补位，1 留给不认识的字；出现多的字编号小
    assert vocab["好"] == 2
    assert set(vocab) == {"好", "#", "块"}
    assert sorted(vocab.values()) == [2, 3, 4]


def test_tf_modules_light_import():
    code = (
        "import sys; import pipeline.step6_classify, pipeline.tf_classifier, pipeline.text_norm; "
        "bad = [m for m in ('sherpa_onnx', 'gradio', 'tensorflow') if m in sys.modules]; "
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ---------- 需要 TensorFlow 的测试（没装时自动跳过） ----------


def test_tf_save_and_load_roundtrip(tmp_path):
    tf = pytest.importorskip("tensorflow")
    texts = [line["text"] for line in load_lines()[:50]]
    vocab = build_vocab(texts)
    max_len = 20
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(max_len,)),
        tf.keras.layers.Embedding(len(vocab) + 2, 8),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(len(LABEL_NAMES), activation="softmax"),
    ])
    save_classifier(model, vocab, {"labels": LABEL_NAMES, "max_len": max_len}, tmp_path / "classifier")
    for name in ["model.h5", "vocab.json", "meta.json"]:
        assert (tmp_path / "classifier" / name).is_file()
    predict = load_classifier(tmp_path / "classifier")
    assert predict is not None
    result = predict(texts[:5])
    assert len(result) == 5 and set(result) <= set(LABEL_NAMES)
    # 通过做法登记使用：有模型就不再退回规则
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        labels = methods.get_method("classify", "tf_model")(texts[:5], {"paths": {"models": str(tmp_path)}})
    assert labels == result
    assert not any("关键词规则" in str(w.message) for w in caught)
