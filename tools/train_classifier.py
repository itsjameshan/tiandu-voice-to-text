"""第 6、7 组：训练 TensorFlow 话术分类模型，并用"按组留一"测评。

用法（在项目文件夹里运行；需要 TensorFlow 2.10 及以上和 numpy，用机房自带的 Python 即可）：
    python tools/train_classifier.py --model g6 --eval logo              按组留一（主结果），用补充句子
    python tools/train_classifier.py --model g6 --eval logo --no-extra   同上，但不用补充句子（两种结果都要报告）
    python tools/train_classifier.py --model g7 --eval random            分层随机划分 8:2（对照，结果会虚高）
    python tools/train_classifier.py --model g6 --eval none              不测评，只用全部数据训练并保存模型

参数：
    --model baseline|g6|g7   用哪个模型结构：baseline 是模板的字级卷积网络（pipeline/tf_classifier.py 的
                             build_baseline_model），g6、g7 是 pipeline/groups/g6_classifier_a.py、
                             g7_classifier_b.py 里的 build_model（默认 baseline）
    --extra / --no-extra     训练时用不用 AI 生成的补充句子（data/classification_extra.csv），默认用。
                             补充句子只进训练集，测试集永远只有人写的剧本台词
    --eval logo|random|none  测评方式（默认 logo）
    --epochs 15              训练轮数
    --out DIR                模型存到哪里。默认 models/classifier_g6、models/classifier_g7；
                             baseline 存到 models/classifier（步骤 6 的 tf_model 做法读这个文件夹）
    --report-dir DIR         报告存到哪里，默认 reports/g6、reports/g7、reports/baseline

做了什么：
    1. 读数据：剧本台词 1055 句（data/lines.csv，人写的，每句有组号和类别）；补充句子（AI 生成）。
    2. 测评（--eval）：
       logo   按组留一：用 7 个组的台词训练、测剩下 1 个组，轮 8 次，每句台词正好被测一次，
              汇总成一个混淆矩阵。相当于"用别的同学写的句子来测"，是主结果；
       random 分层随机划分：每个类别随机取 20% 的台词做测试，其余训练。同一个剧本的句子会同时
              出现在训练和测试里，结果会虚高，只作对照；
       none   不测评。
    3. 用全部台词（加上补充句子）训练最终模型，存进 --out：model.h5、vocab.json、meta.json
       （网页工具和 tools/evaluate.py 用的就是这个模型）。
    4. 报告写进 --report-dir：
       train_<模型>_<测评方式>_<extra 或 no_extra>.md      正确率、误报率、各类准确率和召回率、混淆矩阵
       train_<模型>_<测评方式>_<extra 或 no_extra>_predictions.csv   每句台词的标准答案和预测，方便找分错的句子

基线做法：
    字级模型：文字先归一（数字串换成 #、去标点），每个字换成编号（字表只用训练句子建，测试句里
    没见过的字算"不认识的字"），每句取前 MAX_LEN 个字；模型结构由 build_model 决定；
    不加类别权重，训练 15 轮，每批 32 句；固定随机种子，同一台电脑上重复运行结果基本一样。
可改进方向：
    - 模型结构在各组自己的 build_model 里改（卷积核、LSTM、Dropout……），本脚本一般不用改；
    - 类别权重（威胁消费的台词很少）、训练轮数、MAX_LEN；
    - 审核、增补补充句子，比较"用 / 不用补充句子"的差别。
测评指标：
    7 类正确率；各类准确率（精确率）和召回率；混淆矩阵（行：标准答案，列：预测）；
    误报率 = 标准答案为"正常讲解"的句子中被预测成 5 个疑似类别之一的比例
    （全部剧本一个，只看第 8 组剧本一个）。剧本数据上的测评结果不代表真实场景的效果。

注意：本脚本只导入 pipeline 里的轻量模块（pipeline.data、pipeline.text_norm、pipeline.tf_classifier），
TensorFlow 和 numpy 只在函数里面导入，不导入 sherpa_onnx、gradio，所以机房自带的 Python 就能运行。
选 g6、g7 时只导入那一组的文件（pipeline/groups/g6_classifier_a.py 或 g7_classifier_b.py），只要 TensorFlow 和 numpy。
混淆矩阵用 pipeline/metrics.py（只依赖 numpy）；各类准确率召回率、误报率的计算写在本文件里，和 tools/evaluate.py 的口径相同。
"""
import argparse
import csv
import importlib
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 先把项目文件夹加进 sys.path 才能导入 pipeline；这几个模块都很轻，不会带进 TensorFlow、sherpa_onnx、gradio
from pipeline.data import FLAG_LABELS, LABEL_NAMES, load_lines  # noqa: E402
from pipeline.metrics import confusion_matrix  # noqa: E402  只依赖 numpy，机房自带的 Python 也能用
from pipeline.tf_classifier import (  # noqa: E402
    build_baseline_model,
    build_vocab,
    encode,
    is_tf_available,
    save_classifier,
)

# 可选的模型：baseline 是模板的基线模型，g6、g7 是两个组文件里的 build_model
MODELS = ("baseline", "g6", "g7")
GROUP_MODULES = {"g6": "pipeline.groups.g6_classifier_a", "g7": "pipeline.groups.g7_classifier_b"}

# 测评方式
EVAL_MODES = ("logo", "random", "none")
EVAL_TITLES = {
    "logo": "按组留一（8 折）",
    "random": "分层随机划分（训练 8 : 测试 2）",
    "none": "不测评",
}

DEFAULT_EPOCHS = 15  # 训练轮数
MAX_LEN = 64  # 每句最多取几个字（剧本台词 1055 句里只有 14 句超过 64 个字）
BATCH_SIZE = 32  # 每批训练几句
SEED = 0  # 随机种子：固定下来，重复运行结果基本一样
TEST_RATIO = 0.2  # 随机划分时测试集的比例

NORMAL_LABEL = "正常讲解"  # 算误报率用：这一类被标成疑似就是误报
FP_GROUP = 8  # 第 8 组的剧本是"正常讲解"对照组，单独算一个误报率

# AI 生成的补充句子（只进训练集）
EXTRA_PATH = ROOT / "data" / "classification_extra.csv"

# 固定的局限说明，写进每份报告
LIMITS_NOTE = "剧本数据上的测评结果不代表真实场景的效果"


# ---------------- 文件夹、模型结构、数据 ----------------

def default_out_dir(model_name: str) -> Path:
    """模型默认存放的文件夹：g6 → models/classifier_g6；baseline → models/classifier（tf_model 做法读这里）。"""
    name = "classifier" if model_name == "baseline" else f"classifier_{model_name}"
    return ROOT / "models" / name


def default_report_dir(model_name: str) -> Path:
    """报告默认存放的文件夹：reports/g6、reports/g7、reports/baseline。"""
    return ROOT / "reports" / model_name


def get_build_model(model_name: str):
    """取出建模型的函数 build_model(vocab_size, num_classes, max_len)，返回编译好的 Keras 模型。"""
    if model_name == "baseline":
        return build_baseline_model
    if model_name not in GROUP_MODULES:
        raise ValueError(f"不认识的模型“{model_name}”，只能是：{'、'.join(MODELS)}")
    module = importlib.import_module(GROUP_MODULES[model_name])
    return module.build_model


def load_extra(path=EXTRA_PATH) -> list[dict]:
    """读补充句子，返回 [{"id", "text", "label"}, ...]。类别不合法或句子为空时报错，提示先运行检查脚本。"""
    from pipeline.textio import read_csv_dicts  # Excel 另存时可能是 GBK 或带 BOM 的 UTF-8，都能读

    path = Path(path)
    rows = read_csv_dicts(path, hint="请用 Excel 另存为\"CSV UTF-8（逗号分隔）\"后再试。")[1]
    result = []
    for line_no, row in enumerate(rows, start=2):  # 表头是第 1 行
        text = (row.get("text") or "").strip()
        label = (row.get("label") or "").strip()
        if not text or label not in LABEL_NAMES:
            raise ValueError(
                f"{path.name} 第 {line_no} 行的句子为空或类别“{label}”不合法。"
                f"请先运行 python tools/check_classification_data.py 检查"
            )
        result.append({"id": row.get("id", ""), "text": text, "label": label})
    return result


# ---------------- 怎么分训练集和测试集 ----------------

def logo_folds(lines: list[dict]) -> list[tuple[int, list[int], list[int]]]:
    """按组留一：每个组轮流当测试集。返回 [(测试组号, 训练句下标, 测试句下标), ...]，按组号排。"""
    folds = []
    for group in sorted({line["group"] for line in lines}):
        train_idx = [i for i, line in enumerate(lines) if line["group"] != group]
        test_idx = [i for i, line in enumerate(lines) if line["group"] == group]
        folds.append((group, train_idx, test_idx))
    return folds


def random_split(labels: list[str], test_ratio: float = TEST_RATIO, seed: int = SEED) -> tuple[list[int], list[int]]:
    """分层随机划分：每个类别各自打乱，取 test_ratio 的句子做测试。返回 (训练句下标, 测试句下标)，都从小到大排。"""
    rng = random.Random(seed)
    test_idx = []
    for label in sorted(set(labels)):  # 按固定顺序处理各类，同一个种子每次结果一样
        indices = [i for i, x in enumerate(labels) if x == label]
        rng.shuffle(indices)
        test_idx.extend(indices[:round(len(indices) * test_ratio)])
    test_set = set(test_idx)
    train_idx = [i for i in range(len(labels)) if i not in test_set]
    return train_idx, sorted(test_idx)


# ---------------- 训练和预测（TensorFlow 只在这里面导入） ----------------

def train_model(build_model, texts: list[str], labels: list[str], epochs: int, verbose: int = 0):
    """用 texts、labels 训练一个新模型，返回 (模型, 字表)。字表只用这些训练句子建。"""
    import numpy as np
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)  # 固定随机种子（初始权重、打乱顺序）
    vocab = build_vocab(texts)
    x = np.array(encode(texts, vocab, MAX_LEN), dtype="int32")
    y = np.array([LABEL_NAMES.index(label) for label in labels], dtype="int32")
    # 字表大小要加 2：编号 0 留给补位，1 留给不认识的字
    model = build_model(len(vocab) + 2, len(LABEL_NAMES), MAX_LEN)
    model.fit(x, y, epochs=epochs, batch_size=BATCH_SIZE, shuffle=True, verbose=verbose)
    return model, vocab


def predict_labels(model, vocab: dict[str, int], texts: list[str]) -> list[str]:
    """用训练好的模型给每句话预测一个类别（取概率最大的那一类）。"""
    import numpy as np

    if not texts:
        return []
    x = np.array(encode(texts, vocab, MAX_LEN), dtype="int32")
    probs = model.predict(x, batch_size=256, verbose=0)
    return [LABEL_NAMES[int(i)] for i in probs.argmax(axis=1)]


def run_eval(mode: str, build_model, lines: list[dict], extra: list[dict], epochs: int) -> tuple[list[int], list[str]]:
    """按 mode（logo 或 random）测评，返回 (测试句在 lines 里的下标, 对应的预测类别)。

    补充句子 extra 每一折都全部加进训练集，测试集只有剧本台词。
    """
    texts = [line["text"] for line in lines]
    labels = [line["label"] for line in lines]
    extra_texts = [row["text"] for row in extra]
    extra_labels = [row["label"] for row in extra]

    if mode == "logo":
        splits = [(f"第 {group} 折（测第 {group} 组）", train_idx, test_idx)
                  for group, train_idx, test_idx in logo_folds(lines)]
    elif mode == "random":
        train_idx, test_idx = random_split(labels)
        splits = [("随机划分", train_idx, test_idx)]
    else:
        raise ValueError(f"不认识的测评方式“{mode}”，只能是 logo 或 random")

    test_indices, predictions = [], []
    for title, train_idx, test_idx in splits:
        start = time.time()
        model, vocab = train_model(build_model,
                                   [texts[i] for i in train_idx] + extra_texts,
                                   [labels[i] for i in train_idx] + extra_labels,
                                   epochs)
        pred = predict_labels(model, vocab, [texts[i] for i in test_idx])
        right = sum(1 for i, p in zip(test_idx, pred) if labels[i] == p)
        print(f"  {title}：训练 {len(train_idx) + len(extra)} 句，测试 {len(test_idx)} 句，"
              f"正确率 {right / len(test_idx):.2f}，用时 {time.time() - start:.1f} 秒")
        test_indices.extend(test_idx)
        predictions.extend(pred)
    return test_indices, predictions


# ---------------- 测评指标（和 tools/evaluate.py classify 的口径相同） ----------------

def _per_class_pr(gold: list[str], pred: list[str], labels: list[str]) -> dict[str, dict]:
    """各类的准确率（精确率）和召回率。

    准确率 = 预测成这一类的句子里，真的是这一类的比例；召回率 = 这一类的句子里，被找出来的比例。
    分母为 0 时记为 0.0。support 是标准答案里这一类的句数，predicted 是预测成这一类的句数。
    """
    result = {}
    for label in labels:
        correct = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        support = sum(1 for g in gold if g == label)
        predicted = sum(1 for p in pred if p == label)
        result[label] = {
            "precision": correct / predicted if predicted else 0.0,
            "recall": correct / support if support else 0.0,
            "support": support,
            "predicted": predicted,
            "correct": correct,
        }
    return result


def _fp_counts(gold: list[str], pred: list[str]) -> tuple[int, int]:
    """(被预测成疑似类别的"正常讲解"句数, "正常讲解"总句数)。"""
    normal = [p for g, p in zip(gold, pred) if g == NORMAL_LABEL]
    return sum(1 for p in normal if p in FLAG_LABELS), len(normal)


def _false_positive_rate(gold: list[str], pred: list[str]) -> float:
    """误报率：标准答案为"正常讲解"的句子中被预测成 5 个疑似类别之一的比例；没有正常讲解时为 0.0。"""
    flagged, total = _fp_counts(gold, pred)
    return flagged / total if total else 0.0


def summarize(test_lines: list[dict], pred: list[str], mode: str) -> dict:
    """汇总测评结果（写进 meta.json 和报告）。test_lines 是测试的台词（有 label、group），pred 是预测类别。"""
    gold = [line["label"] for line in test_lines]
    groups = [line["group"] for line in test_lines]
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    fp_count, normal_count = _fp_counts(gold, pred)
    in_fp_group = [i for i, group in enumerate(groups) if group == FP_GROUP]
    fp_count_g8, normal_count_g8 = _fp_counts([gold[i] for i in in_fp_group], [pred[i] for i in in_fp_group])

    by_group = []
    for group in sorted(set(groups)):
        idx = [i for i, x in enumerate(groups) if x == group]
        right = sum(1 for i in idx if gold[i] == pred[i])
        by_group.append({"group": group, "n": len(idx), "correct": right, "accuracy": right / len(idx)})

    return {
        "mode": mode,
        "n_test": len(gold),
        "correct": correct,
        "accuracy": correct / len(gold) if gold else 0.0,
        "fp_rate": fp_count / normal_count if normal_count else 0.0,
        "fp_count": fp_count,
        "normal_count": normal_count,
        "fp_rate_g8": fp_count_g8 / normal_count_g8 if normal_count_g8 else 0.0,
        "fp_count_g8": fp_count_g8,
        "normal_count_g8": normal_count_g8,
        "labels": list(LABEL_NAMES),
        "per_class": _per_class_pr(gold, pred, LABEL_NAMES),
        "confusion": confusion_matrix(gold, pred, LABEL_NAMES),
        "by_group": by_group,
    }


# ---------------- 报告 ----------------

def report_stem(model_name: str, mode: str, used_extra: bool) -> str:
    """报告文件名（不含扩展名），如 train_g6_logo_extra、train_g6_logo_no_extra。"""
    return f"train_{model_name}_{mode}_{'extra' if used_extra else 'no_extra'}"


def _ratio(count: int, total: int) -> str:
    """写成 0.62（650/1055）这样；分母为 0 时写 —（0 句）。"""
    return f"{count / total:.2f}（{count}/{total}）" if total else "—（0 句）"


def write_report(report_dir, info: dict, summary: dict, test_lines: list[dict], pred: list[str]) -> tuple[Path, Path]:
    """写 Markdown 报告和每句预测的 CSV，返回 (md 路径, csv 路径)。

    info：model、eval、used_extra、n_extra、epochs、max_len、trained_at、tensorflow。
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = report_stem(info["model"], info["eval"], info["used_extra"])
    md_path = report_dir / f"{stem}.md"
    csv_path = report_dir / f"{stem}_predictions.csv"

    if info["model"] == "baseline":
        model_text = "baseline（模板的字级卷积网络，pipeline/tf_classifier.py 的 build_baseline_model）"
    else:
        model_text = f"{info['model']}（{GROUP_MODULES[info['model']].replace('.', '/')}.py 的 build_model）"
    if info["used_extra"]:
        extra_text = f"用了：{info.get('n_extra', 0)} 句 AI 生成的补充句子，只进训练集"
    else:
        extra_text = "没用"
    if info["eval"] == "logo":
        eval_text = "按组留一（8 折）：每次用 7 个组的剧本台词训练、测剩下 1 个组，轮 8 次，汇总成一个混淆矩阵"
    else:
        eval_text = "分层随机划分：每个类别随机取 20% 的剧本台词做测试，其余训练（对照，会虚高）"

    out = [
        f"# 话术分类模型训练报告（{info['model']}，{EVAL_TITLES[info['eval']]}）",
        "",
        f"> {LIMITS_NOTE}。工具只提示、不判定，所有标注均为疑似、待核查，必须人工复核。",
        "",
        "## 基本信息",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| 模型 | {model_text} |",
        f"| 测评方式 | {eval_text} |",
        f"| 补充句子 | {extra_text} |",
        f"| 测试集 | 只有人写的剧本台词，共 {summary['n_test']} 句 |",
        f"| 训练轮数 | {info['epochs']} |",
        f"| 每句最多取几个字 | {info['max_len']} |",
        f"| TensorFlow 版本 | {info.get('tensorflow', '')} |",
        f"| 生成时间 | {info['trained_at']} |",
        "",
        "## 总体结果",
        "",
        "| 指标 | 结果 |",
        "|---|---|",
        f"| 7 类正确率 | {_ratio(summary['correct'], summary['n_test'])} |",
        f"| 误报率（全部剧本） | {_ratio(summary['fp_count'], summary['normal_count'])} |",
        f"| 误报率（只看第 {FP_GROUP} 组剧本） | {_ratio(summary['fp_count_g8'], summary['normal_count_g8'])} |",
        "",
        "误报率 = 标准答案为“正常讲解”的句子中，被模型预测成 5 个疑似类别之一的比例。误报会冤枉人，一定要报告。",
        "",
        "## 各类准确率与召回率",
        "",
        "准确率（精确率）= 预测成这一类的句子里真的是这一类的比例；召回率 = 这一类的句子里被找出来的比例。",
        "",
        "| 类别 | 标准答案句数 | 预测成该类的句数 | 预测对的句数 | 准确率 | 召回率 |",
        "|---|---|---|---|---|---|",
    ]
    for label in LABEL_NAMES:
        item = summary["per_class"][label]
        out.append(f"| {label} | {item['support']} | {item['predicted']} | {item['correct']} | "
                   f"{item['precision']:.2f} | {item['recall']:.2f} |")

    out += [
        "",
        "## 混淆矩阵",
        "",
        "行是标准答案，列是模型预测，对角线上是分对的句数。类别名用数据里的写法（“威胁消费”在界面上显示为“消费施压”）。",
        "",
        "| 标准答案 \\ 预测 | " + " | ".join(LABEL_NAMES) + " | 合计 |",
        "|---" * (len(LABEL_NAMES) + 2) + "|",
    ]
    for label, row in zip(LABEL_NAMES, summary["confusion"]):
        out.append(f"| {label} | " + " | ".join(str(n) for n in row) + f" | {sum(row)} |")

    group_title = "各组结果（按组留一时每一折测一个组）" if info["eval"] == "logo" else "测试句按组统计"
    out += ["", f"## {group_title}", "", "| 测试组 | 句数 | 分对 | 正确率 |", "|---|---|---|---|"]
    for item in summary["by_group"]:
        out.append(f"| 第 {item['group']} 组 | {item['n']} | {item['correct']} | {item['accuracy']:.2f} |")

    out += [
        "",
        "## 局限",
        "",
        f"- {LIMITS_NOTE}：剧本台词是照着类别写的，说法比真实录音规整；模型也没有见过识别出错的文字。",
        f"- 测试集里有些类别的句子很少（例如“威胁消费”只有 {summary['per_class']['威胁消费']['support']} 句），"
        "这些类别的准确率、召回率波动很大。",
    ]
    if info["used_extra"]:
        out.append("- 补充句子由 AI 参照全部剧本台词写成，可能带有测试句的说法，结果可能偏乐观；"
                   "补充句子在第 6、7 组审核完之前可能有标错的类别或固定的说法。")
    if info["eval"] == "random":
        out.append("- 随机划分时同一个剧本的句子同时出现在训练和测试里，结果会虚高，只作对照，主结果看按组留一。")
    out.append("- 随机种子是固定的，但换一台电脑或换 TensorFlow 版本，结果会有小幅差别。")
    out.append("")
    md_path.write_text("\n".join(out), encoding="utf-8")

    # 每句台词的预测结果（用 Excel 打开可以筛选"错"的行，看看哪些句子分错了）
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["script_id", "group", "line_no", "text", "label", "predicted", "correct"])
        for line, p in zip(test_lines, pred):
            writer.writerow([line.get("script_id", ""), line["group"], line.get("line_no", ""), line["text"],
                             line["label"], p, "对" if p == line["label"] else "错"])
    return md_path, csv_path


# ---------------- 命令行 ----------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="训练 TensorFlow 话术分类模型（第 6、7 组），用按组留一测评")
    parser.add_argument("--model", choices=MODELS, default="baseline",
                        help="模型结构：baseline（模板）、g6、g7（各组文件里的 build_model），默认 baseline")
    parser.add_argument("--extra", action=argparse.BooleanOptionalAction, default=True,
                        help="训练时用不用 AI 生成的补充句子（默认用；--no-extra 不用）。补充句子只进训练集")
    parser.add_argument("--eval", dest="eval_mode", choices=EVAL_MODES, default="logo",
                        help="测评方式：logo 按组留一（主结果，默认）、random 分层随机 8:2（对照）、none 不测评")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"训练轮数（默认 {DEFAULT_EPOCHS}）")
    parser.add_argument("--out", help="模型存放的文件夹（默认 models/classifier_<模型>，baseline 为 models/classifier）")
    parser.add_argument("--report-dir", help="报告存放的文件夹（默认 reports/<模型>）")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs 至少是 1")

    # 命令行窗口显示不了的字符用 ? 代替，不让程序因此出错
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    print("== 话术分类模型训练 ==")
    if not is_tf_available():
        print("这台电脑的 Python 没有安装 TensorFlow，不能训练。请用机房自带的、装了 TensorFlow 的 Python 运行，"
              "或者先安装：python -m pip install -r requirements-tf.txt")
        return 1
    try:
        build_model = get_build_model(args.model)
    except ImportError as err:
        print(f"导入 {args.model} 的模型文件失败：{err}。请检查本组文件 pipeline/groups/ 里是不是写错了，"
              f"或者新加的 import 用了这台电脑没有的库")
        return 1

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 少打印一些 TensorFlow 的内部日志
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")  # 不打印"函数重新追踪"之类的提示，只留错误
    lines = load_lines()
    extra = load_extra() if args.extra else []
    out_dir = Path(args.out) if args.out else default_out_dir(args.model)
    report_dir = Path(args.report_dir) if args.report_dir else default_report_dir(args.model)
    print(f"模型：{args.model}；测评方式：{EVAL_TITLES[args.eval_mode]}；训练轮数：{args.epochs}；TensorFlow {tf.__version__}")
    print(f"数据：剧本台词 {len(lines)} 句；补充句子 {'用了 ' + str(len(extra)) + ' 句（只进训练集）' if args.extra else '没用'}")

    info = {
        "model": args.model,
        "eval": args.eval_mode,
        "used_extra": bool(args.extra),
        "n_extra": len(extra),
        "epochs": args.epochs,
        "max_len": MAX_LEN,
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tensorflow": tf.__version__,
    }

    # 1. 测评
    summary = {"mode": "none"}
    test_lines, pred = [], []
    if args.eval_mode != "none":
        print(f"\n测评（{EVAL_TITLES[args.eval_mode]}）：")
        start = time.time()
        test_indices, pred = run_eval(args.eval_mode, build_model, lines, extra, args.epochs)
        test_lines = [lines[i] for i in test_indices]
        summary = summarize(test_lines, pred, args.eval_mode)
        summary["seconds"] = round(time.time() - start, 1)
        print(f"  7 类正确率 {_ratio(summary['correct'], summary['n_test'])}；"
              f"误报率（全部剧本）{_ratio(summary['fp_count'], summary['normal_count'])}；"
              f"误报率（第 {FP_GROUP} 组）{_ratio(summary['fp_count_g8'], summary['normal_count_g8'])}")
        print("  各类召回率：" + "，".join(f"{label} {summary['per_class'][label]['recall']:.2f}"
                                       for label in LABEL_NAMES))

    # 2. 用全部数据训练最终模型并保存
    print("\n用全部数据训练最终模型：")
    start = time.time()
    texts = [line["text"] for line in lines] + [row["text"] for row in extra]
    labels = [line["label"] for line in lines] + [row["label"] for row in extra]
    model, vocab = train_model(build_model, texts, labels, args.epochs, verbose=2)
    meta = {
        "labels": list(LABEL_NAMES),
        "max_len": MAX_LEN,
        "model": args.model,
        "trained_at": info["trained_at"],
        "used_extra": info["used_extra"],
        "n_lines": len(lines),
        "n_extra": len(extra),
        "epochs": args.epochs,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "tensorflow": tf.__version__,
        "train_seconds": round(time.time() - start, 1),
        "evaluation": summary,
        "notice": LIMITS_NOTE,
    }
    save_classifier(model, vocab, meta, out_dir)
    print(f"模型已保存到：{out_dir}（model.h5、vocab.json、meta.json）")

    # 3. 报告
    if args.eval_mode == "none":
        print("没有测评（--eval none），不写报告。")
    else:
        md_path, csv_path = write_report(report_dir, info, summary, test_lines, pred)
        print(f"报告：{md_path}")
        print(f"每句的预测：{csv_path}")
    print(f"注意：{LIMITS_NOTE}。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
