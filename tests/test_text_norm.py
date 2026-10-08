"""文字归一（pipeline/text_norm.py）的测试。

两个函数：
- normalize_for_cer：算字错率前用，只留汉字、字母、数字；
- normalize_for_classification：话术分类前用（规则和 TensorFlow 模型都用它），数字串统一换成 #。
"""
import subprocess
import sys

import pytest
from conftest import ROOT

from pipeline.text_norm import normalize_for_cer, normalize_for_classification

# ---------- 计划里列出的测试 ----------


def test_cer_norm():
    assert normalize_for_cer("ＡＢ，你好 12！") == "AB你好12"


def test_classification_norm_consistent():
    # 汉字读法和阿拉伯数字写法归一后要一样，否则模型训练时见的和使用时见的不是一回事
    assert normalize_for_classification("两千八百块") == normalize_for_classification("2800块") == "#块"
    # 只有一个汉字数字的不是数字串
    assert "#" not in normalize_for_classification("等一下")
    assert normalize_for_classification("等一下") == "等一下"


# ---------- 补充测试 ----------


def test_cer_norm_letters_and_full_width():
    assert normalize_for_cer("abc１２３") == "ABC123"
    assert normalize_for_cer("  ") == ""
    assert normalize_for_cer("") == ""
    # 标点、空白、换行都去掉；汉字数字原样保留（测评模式本来就是汉字读法）
    assert normalize_for_cer("早上九点，\n至下午五点。") == "早上九点至下午五点"
    assert normalize_for_cer("二〇二六年") == "二〇二六年"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("0871-0000-6688", "#"),  # 电话：数字中间的 - 算在数字串里
        ("零八七一 零零零零 六六八八", "#"),  # 逐位读的电话（中间有空格）也是一个数字串
        ("YN-0000-3721", "YN#"),
        ("yn 零零零零 三七二一", "YN#"),  # 字母转大写
        ("15:40集合", "#集合"),
        # 已知不足（留给改进）："三点四十" 里 "三" 是单个数字字，不替换，和 "15:40" 归一后不一样
        ("下午三点四十集合", "下午三点#集合"),
        ("5.5元", "#元"),
        ("20元/克", "#元克"),  # 标点去掉
        ("ＡＢ，你好 12！", "AB你好#"),
        ("第三天", "第三天"),  # 只有一个汉字数字
        ("二十多年", "#多年"),
        ("一百五十的那是平时喝的", "#的那是平时喝的"),
        ("", ""),
    ],
)
def test_classification_norm_examples(text, expected):
    assert normalize_for_classification(text) == expected


def test_classification_norm_is_idempotent():
    # 归一两次和归一一次结果相同（# 不会被再改掉）
    for text in ["两千八百块", "0871-0000-6688", "大家多少支持一下，后面的安排我也好协调。"]:
        once = normalize_for_classification(text)
        assert normalize_for_classification(once) == once


def test_text_norm_light_import():
    # text_norm 只用标准库：只装了 TensorFlow 的机房 Python 也能导入
    code = (
        "import sys; import pipeline.text_norm; "
        "bad = [m for m in ('sherpa_onnx', 'gradio', 'tensorflow') if m in sys.modules]; "
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
