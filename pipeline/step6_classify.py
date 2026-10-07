"""步骤 6：话术分类（第 6、7 组）。

给每个段落的文字定一个类别（7 选 1，见 data/labels.json）：
    购物安排、费用、行程变更、服务态度、威胁消费、正常讲解、其他。
前 5 类会被标出来，标签写成"疑似·显示名"（如"疑似·费用"；"威胁消费"显示为"疑似·消费施压"），
正常讲解和其他的标签留空。标签一律用 pipeline.data.label_output(类别) 得到，不要自己拼字符串。
工具只提示、不判定：所有标签都要人工复核。

槽位 classify 的做法：函数 (文字列表, 配置) → 类别列表（与文字一一对应，每个都是 7 个类别名之一）。
    baseline   关键词规则（本文件）
    tf_model   用 models/classifier/ 里训练好的 TensorFlow 模型；没装 TensorFlow 或没有模型时，
               退回关键词规则，并发出中文警告说明原因
    g6、g7     第 6、7 组的模型（pipeline/groups/），用 classify_with_model 加载自己的模型文件夹

基线做法（关键词规则）：
    1. 关键词表在 pipeline/rules_keywords.json：每个类别一组短词，例如"费用"有"多少钱""团费""#元"；
    2. 句子和关键词都先用 text_norm.normalize_for_classification 归一（数字串换成 #、去掉标点）；
    3. 数一数每个类别的关键词在句子里一共出现了几次（count_hits），次数最多的类别胜出；
    4. 次数一样多时，按 rules_keywords.json 里 priority 的先后顺序选（pick_category）；
    5. 一个关键词都没命中，就是"其他"。
    每个结果都能说清楚"因为句子里有哪个词"，方便人工复核时判断。
可改进方向：
    - 第 6、7 组：训练 TensorFlow 模型（tools/train_classifier.py），和关键词规则、和对方的模型比较；
    - 改关键词表：加词、删掉容易误报的词、调整 priority。注意不要把剧本原句抄成关键词，
      那样在剧本上测出来的结果会虚高，换个说法就失效；
    - "正常讲解"也有关键词，它命中多时句子就不会被标成疑似，这是压低误报的简单办法。
测评指标：
    各类准确率、召回率、混淆矩阵；误报率 = 标准答案为"正常讲解"的句子中被标成任一疑似类别的比例
    （见 tests/test_step6_classify.py 和 tools/evaluate.py classify）。剧本上的结果不代表真实录音上的效果。
"""
import copy
import json
import warnings
from pathlib import Path
from typing import Callable

from pipeline import tf_classifier
from pipeline.config import ROOT
from pipeline.data import LABEL_NAMES, label_output
from pipeline.methods import register
from pipeline.text_norm import normalize_for_classification

# 关键词表文件
RULES_PATH = Path(__file__).resolve().parent / "rules_keywords.json"

# 一个关键词都没命中时用的类别
DEFAULT_CATEGORY = "其他"


# ---------------- 关键词表 ----------------

def _read_rules() -> dict:
    """读关键词表并检查格式；写错时给出中文提示。每次都重新读，改了文件不用重启。"""
    data = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    keywords = data.get("keywords")
    if not isinstance(keywords, dict) or set(keywords) != set(LABEL_NAMES):
        found = "、".join(keywords) if isinstance(keywords, dict) else "（没有 keywords）"
        raise ValueError(
            f"{RULES_PATH.name} 的 keywords 必须正好有 7 个类别：{'、'.join(LABEL_NAMES)}；现在是：{found}"
        )
    for category, words in keywords.items():
        if not isinstance(words, list) or not all(isinstance(w, str) and w for w in words):
            raise ValueError(f"{RULES_PATH.name} 里类别“{category}”的关键词必须是非空字符串的列表")
    priority = data.get("priority")
    if not isinstance(priority, list) or sorted(priority) != sorted(LABEL_NAMES):
        raise ValueError(f"{RULES_PATH.name} 的 priority 必须把 7 个类别各写一次：{'、'.join(LABEL_NAMES)}")
    return data


def load_keywords() -> dict[str, list[str]]:
    """关键词表：类别 → 关键词列表（照文件里写的样子，没有归一）。7 个类别都有，"其他"是空表。"""
    return copy.deepcopy(_read_rules()["keywords"])


def load_priority() -> list[str]:
    """平局时的先后顺序：排在前面的类别优先。"""
    return list(_read_rules()["priority"])


def _normalized_keywords() -> dict[str, list[str]]:
    """把关键词也用同样的方法归一（如"２８００元"→"#元"），这样才能和归一后的句子比较。"""
    result = {}
    for category, words in load_keywords().items():
        result[category] = []
        for word in words:
            norm = normalize_for_classification(word)
            if not norm.replace("#", ""):
                # 归一后只剩 # 或什么都不剩，会命中所有带数字的句子，不能用
                raise ValueError(f"{RULES_PATH.name} 里类别“{category}”的关键词“{word}”归一后没有剩下文字，请换一个词")
            result[category].append(norm)
    return result


# ---------------- 关键词规则的两个小零件 ----------------

def count_hits(text: str, keywords: dict[str, list[str]]) -> dict[str, int]:
    """数一数每个类别的关键词在 text 里一共出现了几次。text 和 keywords 都应该已经归一过。

    例：count_hits("这个手镯多少钱#元", {"费用": ["#元", "多少钱"], "购物安排": ["手镯"]})
        → {"费用": 2, "购物安排": 1}
    """
    return {category: sum(text.count(word) for word in words) for category, words in keywords.items()}


def pick_category(counts: dict[str, int], priority: list[str]) -> str:
    """次数最多的类别胜出；一样多时按 priority 的先后；全是 0 时返回"其他"。"""
    best = max(counts.values(), default=0)
    if best == 0:
        return DEFAULT_CATEGORY
    for category in priority:
        if counts.get(category, 0) == best:
            return category
    return DEFAULT_CATEGORY


@register("classify", "baseline")
def classify_baseline(texts: list[str], cfg: dict) -> list[str]:
    """关键词规则基线：每句话归一后数各类关键词命中次数，最多者胜，平局按 priority，全不命中为"其他"。"""
    keywords = _normalized_keywords()
    priority = load_priority()
    return [pick_category(count_hits(normalize_for_classification(text), keywords), priority) for text in texts]


# ---------------- TensorFlow 模型做法 ----------------

def classifier_dir(cfg: dict, name: str = "classifier") -> Path:
    """模型文件夹：config.yaml 里 paths.models 下面的 name 子文件夹（默认 models/classifier/）。"""
    models = (cfg.get("paths") or {}).get("models")
    return Path(models) / name if models else ROOT / "models" / name


def classify_with_model(texts: list[str], cfg: dict, model_dir) -> list[str]:
    """用 model_dir 里训练好的 TensorFlow 模型分类；用不了时退回关键词规则，并发出中文警告。

    第 6、7 组的做法也调用这个函数，只是换成自己的模型文件夹（如 models/classifier_g6/）。
    """
    texts = list(texts)
    if not tf_classifier.is_tf_available():
        warnings.warn("这台电脑没有安装 TensorFlow，话术分类改用关键词规则（基线做法）。", UserWarning, stacklevel=2)
        return classify_baseline(texts, cfg)
    try:
        predict = tf_classifier.load_classifier(model_dir)
    except Exception as err:  # 模型文件损坏、TensorFlow 版本不兼容等：不让整条流程停下
        warnings.warn(f"分类模型加载失败（{model_dir}：{err}），话术分类改用关键词规则（基线做法）。", UserWarning, stacklevel=2)
        return classify_baseline(texts, cfg)
    if predict is None:
        warnings.warn(
            f"在 {model_dir} 里没有找到训练好的分类模型（需要 model.h5、vocab.json、meta.json 三个文件），"
            f"话术分类改用关键词规则（基线做法）。训练方法见 tools/train_classifier.py。",
            UserWarning,
            stacklevel=2,
        )
        return classify_baseline(texts, cfg)
    return predict(texts)


@register("classify", "tf_model")
def classify_tf_model(texts: list[str], cfg: dict) -> list[str]:
    """用 models/classifier/ 里的模型分类；没装 TensorFlow 或没有模型时退回关键词规则。"""
    return classify_with_model(texts, cfg, classifier_dir(cfg))


# ---------------- 给段落写上类别和标签 ----------------

def classify_segments(segments: list[dict], cfg: dict, method: Callable | None = None) -> list[dict]:
    """给每个段落写上 category（7 类之一）和 label（label_output(category)），返回新的段落列表。

    method：分类做法 (文字列表, 配置) → 类别列表；不填就用关键词规则基线。
            整条流程里由 pipeline.methods.resolve(cfg)["classify"] 按配置选好再传进来。
    传进来的段落不会被修改。
    """
    method = method or classify_baseline
    texts = [segment.get("text") or "" for segment in segments]
    categories = list(method(texts, cfg))
    if len(categories) != len(texts):
        raise ValueError(f"分类做法返回了 {len(categories)} 个类别，但输入有 {len(texts)} 段文字，两者必须一样多")
    result = []
    for segment, category in zip(segments, categories):
        segment = dict(segment)
        segment["category"] = category
        segment["label"] = label_output(category)  # 不认识的类别会抛 ValueError
        result.append(segment)
    return result
