"""TensorFlow 话术分类模型的保存、加载和文字编码（给步骤 6 和第 6、7 组的训练脚本用）。

TensorFlow 是可选依赖（requirements-tf.txt），便携包里不带。所以：
    - 本文件导入时不导入 TensorFlow，只在 load_classifier 函数里面导入；
    - 没装 TensorFlow 或模型还没训练时，load_classifier 返回 None，步骤 6 就退回关键词规则。

一个训练好的模型是一个文件夹（如 models/classifier/），里面三个文件：
    model.h5     Keras 模型（输入：字编号序列；输出：7 个类别各自的概率）
    vocab.json   字表：{"字": 编号}。编号 0 留给补位，1 留给字表里没有的字，真正的字从 2 开始
    meta.json    其他信息，至少有 {"labels": [7 个类别名，顺序同模型输出], "max_len": 每句最多取几个字}
                 训练脚本还可以写进训练日期、用了哪些数据、测评结果等

基线做法：
    字级模型：先用 text_norm.normalize_for_classification 归一（数字串换成 #、去标点），
    每个字查字表换成编号（encode），不够 max_len 补 0，超过就截断；
    模型给出 7 个概率，取最大的那个类别。
可改进方向：
    第 6、7 组在 pipeline/groups/ 里改模型结构（卷积、LSTM 等）、句子长度、字表大小，
    或者用补充句子扩充训练数据；保存、加载的格式不要改，这样界面和测评工具不用跟着改。
测评指标：
    各类准确率、召回率、混淆矩阵、"正常讲解"误报率；主结果用"按组留一"（见 tools/train_classifier.py）。
"""
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Callable

from pipeline.data import LABEL_NAMES
from pipeline.text_norm import normalize_for_classification

# 模型文件夹里必须有的三个文件
MODEL_FILES = ("model.h5", "vocab.json", "meta.json")

# 字编号：0 = 补位（句子不够长时补在后面），1 = 字表里没有的字
PAD_ID = 0
UNK_ID = 1

# 已经加载过的模型：(文件夹, model.h5 的修改时间) → 预测函数。重新训练后修改时间变了，会重新加载
_LOADED: dict[tuple[str, float], Callable[[list[str]], list[str]]] = {}


def is_tf_available() -> bool:
    """这台电脑装没装 TensorFlow。只查找、不导入（导入 TensorFlow 要好几秒）。"""
    return importlib.util.find_spec("tensorflow") is not None


def build_vocab(texts: list[str]) -> dict[str, int]:
    """用训练句子建字表：先归一，再统计每个字出现的次数，出现多的字编号小，从 2 开始编号。"""
    counts = Counter()
    for text in texts:
        counts.update(normalize_for_classification(text))
    # 按出现次数从多到少排；次数一样的按字本身排序，保证每次建出来的字表都一样
    chars = sorted(counts, key=lambda ch: (-counts[ch], ch))
    return {ch: index + 2 for index, ch in enumerate(chars)}


def encode(texts: list[str], vocab: dict[str, int], max_len: int) -> list[list[int]]:
    """把句子变成等长的字编号列表：先归一，再按字查表（不认识的字是 1），不够 max_len 补 0，太长截断。

    例：encode(["两千八百块"], {"#": 2, "块": 3}, 4) → [[2, 3, 0, 0]]
    """
    result = []
    for text in texts:
        ids = [vocab.get(ch, UNK_ID) for ch in normalize_for_classification(text)][:max_len]
        ids += [PAD_ID] * (max_len - len(ids))
        result.append(ids)
    return result


def save_classifier(model, vocab: dict[str, int], meta: dict, model_dir) -> None:
    """把训练好的模型存进 model_dir：model.h5、vocab.json、meta.json。

    meta 至少要有 labels（7 个类别名，顺序同模型输出）和 max_len。
    """
    for key in ("labels", "max_len"):
        if key not in meta:
            raise ValueError(f"meta 里缺少“{key}”：至少要有 labels（类别名列表）和 max_len（每句最多几个字）")
    _check_labels(meta["labels"])
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir / "model.h5"))
    (model_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")
    (model_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_classifier(model_dir) -> Callable[[list[str]], list[str]] | None:
    """加载 model_dir 里训练好的模型，返回一个函数：输入句子列表，输出类别名列表。

    没装 TensorFlow，或者文件夹里缺 model.h5、vocab.json、meta.json 任何一个，返回 None。
    文件都在、但模型坏了或 meta.json 写错了，会抛出异常（由调用方决定怎么处理）。
    """
    model_dir = Path(model_dir)
    if not is_tf_available():
        return None
    if not all((model_dir / name).is_file() for name in MODEL_FILES):
        return None

    key = (str(model_dir.resolve()), (model_dir / "model.h5").stat().st_mtime)
    if key in _LOADED:
        return _LOADED[key]

    try:
        import numpy as np
        import tensorflow as tf  # 可选依赖，只在这里导入
    except ImportError:
        return None

    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    labels = list(meta["labels"])
    max_len = int(meta["max_len"])
    _check_labels(labels)
    model = tf.keras.models.load_model(str(model_dir / "model.h5"), compile=False)

    def predict(texts: list[str]) -> list[str]:
        texts = list(texts)
        if not texts:
            return []
        x = np.array(encode(texts, vocab, max_len), dtype="int32")
        probs = model.predict(x, verbose=0)
        return [labels[int(i)] for i in probs.argmax(axis=1)]

    _LOADED[key] = predict
    return predict


def _check_labels(labels: list[str]) -> None:
    """meta 里的类别名必须是 data/labels.json 里的 7 个类别之一，不然标签会出错。"""
    unknown = [name for name in labels if name not in LABEL_NAMES]
    if unknown or not labels:
        raise ValueError(f"meta.json 的 labels 里有不认识的类别：{unknown}，只能是：{'、'.join(LABEL_NAMES)}")
