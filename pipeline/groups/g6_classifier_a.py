"""第 6 组：TensorFlow 话术分类模型 A（槽位 classify）

你们组负责什么
    流水线第 6 步：给每段文字分 7 类之一（购物安排、费用、行程变更、服务态度、威胁消费、正常讲解、其他），
    前 5 类在初稿上标成"疑似·……"（威胁消费显示为"消费施压"）。
    你们做"模型 A"：在模板的字级卷积网络基础上改进；第 7 组做"模型 B"（LSTM），两组最后对比。
    基线做法是关键词规则（pipeline/step6_classify.py + rules_keywords.json）。

在哪里改
    1. 下面的 build_model：模型结构。现在和模板的基线模型一样（字级卷积网络）。
    2. 训练：python tools/train_classifier.py --model g6 --eval logo
       （需要 TensorFlow，用机房自带的 Python 运行即可；训练好的模型存在 models/classifier_g6/）
    3. classify_g6 用训练好的模型分类；模型不存在或没装 TensorFlow 时自动退回关键词规则，并给出提示。
    函数名、参数、返回值的格式不要改。

怎么测
    python tools/train_classifier.py --model g6 --eval logo     （按组留一，主结果）
    python tools/train_classifier.py --model g6 --eval random   （随机划分，对照，会虚高）
    python tools/evaluate.py classify --method classify=g6 --out reports/g6/check
        （只用来确认模型能加载、能跑：模型最后是用全部台词训练的，在同样的台词上测会虚高，成绩看上面按组留一的报告）

注意
    - "按组留一"：用 7 个组的台词训练、测剩下 1 个组，轮 8 次，汇总成一个混淆矩阵——相当于"用别的同学写的句子测"。
    - 误报率（正常讲解被标成疑似）一定要报告：误报会冤枉人。
    - AI 生成的补充句子（data/classification_extra.csv）只用来训练，报告里写清楚用没用。
    - 不联网，不用公共大模型分类。

可以试的方向（由易到难）
    1. 调 Embedding 维度、卷积核个数和宽度、句子长度 max_len、训练轮数。
    2. 类别很不均匀（威胁消费只有 41 句）：试试类别权重，或者加入补充句子。
    3. 多个宽度的卷积核（2、3、4）拼起来；加 Dropout 防止过拟合。
"""
from pipeline.methods import register
from pipeline.step6_classify import classifier_dir, classify_with_model
from pipeline.tf_classifier import build_baseline_model

MODEL_DIR_NAME = "classifier_g6"


def build_model(vocab_size: int, num_classes: int, max_len: int):
    """第 6 组的模型 A。TensorFlow 只能在函数里面导入（网页工具在没有 TensorFlow 的电脑上也要能启动）。"""
    # TODO 第 6 组：在这里改模型结构。现在先用模板的基线模型（字级卷积网络）。
    return build_baseline_model(vocab_size, num_classes, max_len)


@register("classify", "g6")
def classify_g6(texts: list[str], cfg: dict) -> list[str]:
    """用 models/classifier_g6/ 里训练好的模型分类；用不了时退回关键词规则。"""
    return classify_with_model(texts, cfg, classifier_dir(cfg, MODEL_DIR_NAME))
