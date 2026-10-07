"""第 7 组：TensorFlow 话术分类模型 B（槽位 classify）

你们组负责什么
    流水线第 6 步：给每段文字分 7 类之一，前 5 类在初稿上标成"疑似·……"（威胁消费显示为"消费施压"）。
    你们做"模型 B"：把模板的字级卷积网络改成 LSTM（循环神经网络，按顺序一个字一个字地读），
    和第 6 组的模型 A（卷积）在同一份数据上对比。
    基线做法是关键词规则（pipeline/step6_classify.py + rules_keywords.json）。

在哪里改
    1. 下面的 build_model：现在和模板的基线模型一样（卷积），请改成 LSTM，例如
       Embedding → LSTM(64)（或 Bidirectional(LSTM(64))）→ Dense(7, softmax)。
    2. 训练：python tools/train_classifier.py --model g7 --eval logo
       （需要 TensorFlow，用机房自带的 Python 运行即可；训练好的模型存在 models/classifier_g7/）
    3. classify_g7 用训练好的模型分类；模型不存在或没装 TensorFlow 时自动退回关键词规则，并给出提示。
    函数名、参数、返回值的格式不要改。

怎么测
    python tools/train_classifier.py --model g7 --eval logo
    python tools/evaluate.py classify --method classify=g7 --out reports/g7
    和第 6 组用同样的数据、同样的测评方式比较（按组留一的混淆矩阵、各类召回率、误报率）。

注意
    - LSTM 训练比卷积慢，先用少的轮数试；Embedding 记得设 mask_zero=True，让补位的 0 不参与计算。
    - 误报率一定要报告；AI 补充句子只进训练集。
    - 不联网，不用公共大模型分类。

可以试的方向（由易到难）
    1. 单向 LSTM → 双向 LSTM；调隐藏单元数、句子长度。
    2. 类别权重或补充句子，改善样本少的类别。
    3. LSTM 后面接注意力或池化，和第 6 组的卷积对比"哪种更适合短句分类"。
"""
from pipeline.methods import register
from pipeline.step6_classify import classifier_dir, classify_with_model
from pipeline.tf_classifier import build_baseline_model

MODEL_DIR_NAME = "classifier_g7"


def build_model(vocab_size: int, num_classes: int, max_len: int):
    """第 7 组的模型 B。TensorFlow 只能在函数里面导入（网页工具在没有 TensorFlow 的电脑上也要能启动）。"""
    # TODO 第 7 组：把这里改成 LSTM。现在先用模板的基线模型（字级卷积网络）。
    return build_baseline_model(vocab_size, num_classes, max_len)


@register("classify", "g7")
def classify_g7(texts: list[str], cfg: dict) -> list[str]:
    """用 models/classifier_g7/ 里训练好的模型分类；用不了时退回关键词规则。"""
    return classify_with_model(texts, cfg, classifier_dir(cfg, MODEL_DIR_NAME))
