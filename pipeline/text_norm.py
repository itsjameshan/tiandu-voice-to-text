"""文字归一：比较或分类之前，先把文字整理成统一的样子。

为什么要归一？
    同一句话可能有好几种写法：全角"ＡＢ"和半角"AB"，"两千八百块"和"2800块"，带不带标点……
    算字错率、做话术分类之前不先统一，就会把"写法不同"误当成"内容不同"。

本文件只用 Python 标准库（unicodedata、re），不导入 sherpa-onnx、numpy、TensorFlow，
也不调用步骤 5 的数字转换。这样：
    - 第 6、7 组可以用机房自带的、只装了 TensorFlow 的 Python 运行训练脚本；
    - 第 4 组改了步骤 5，也不会悄悄改变分类模型看到的文字。

两个函数：
    normalize_for_cer(text)              算字错率前用
    normalize_for_classification(text)   话术分类前用（关键词规则和 TensorFlow 模型都用它，训练和使用口径一致）

基线做法：
    normalize_for_cer：
        1. NFKC 规范化：全角字母、数字、标点变成半角（"ＡＢ１２" → "AB12"）；
        2. 只保留汉字、ASCII 字母和数字，其余（标点、空格、换行）全部去掉；
        3. 字母转成大写。
    normalize_for_classification：
        1. NFKC 规范化，字母转大写；
        2. 去掉空白（这样"零八七一 零零零零 六六八八"这种分组念的号码会连成一串）；
        3. 连续的阿拉伯数字（包括数字中间的 . : / -，如 0871-0000-6688、15:40、5.5）换成一个 #；
        4. 两个及以上连续的汉字数字（零〇一二两三四五六七八九十百千万亿）换成一个 #。
           只有一个汉字数字的不换，所以"等一下""第三天"保持原样；
        5. 去掉标点，只留汉字、字母和 #。
        例："两千八百块" 和 "2800块" 都变成 "#块"。
可改进方向：
    - "三点四十"里"三"只有一个字，不会被替换，归一后是"三点#"，而"15:40"是"#"，两者不一致；
    - "万一""千万别"也会被当成数字串换成 #；
    - 可以把"块""块钱"统一成"元"，把繁体转简体等。
    改这里会同时改变关键词规则和模型的输入，改完要重新训练模型、重新测评。
测评指标：
    不直接测；由 tests/test_text_norm.py 检查例子，分类和字错率的测评结果间接反映归一是否合理。
"""
import re
import unicodedata

# 汉字数字（用于识别"两千八百""零八七一"这样的数字串）
CHINESE_DIGITS = "零〇一二两三四五六七八九十百千万亿"

# 汉字的范围：常用汉字（4E00–9FFF）、扩展 A 区（3400–4DBF），再加上"〇"
_HAN = "〇㐀-䶿一-鿿"

# 算字错率时要去掉的字：不是 ASCII 数字、字母、汉字的都去掉
_NOT_CER_CHAR = re.compile(f"[^0-9A-Za-z{_HAN}]")

# 分类时要去掉的字：不是字母、汉字、# 的都去掉（这时数字已经都换成 # 了）
_NOT_CLS_CHAR = re.compile(f"[^0-9A-Za-z#{_HAN}]")

# 空白：空格、制表符、换行等
_SPACE = re.compile(r"\s+")

# 阿拉伯数字串：一段数字，后面可以跟着若干个"一个 .:/- 符号 + 一段数字"
_ARABIC_NUMBER = re.compile(r"[0-9]+(?:[.:/\-][0-9]+)*")

# 汉字数字串：两个及以上连续的汉字数字
_CHINESE_NUMBER = re.compile(f"[{CHINESE_DIGITS}]{{2,}}")

# 数字串统一换成这个占位符
NUMBER_PLACEHOLDER = "#"


def normalize_for_cer(text: str) -> str:
    """算字错率前的归一：全角转半角，只留汉字、ASCII 字母、数字，字母大写。

    例：normalize_for_cer("ＡＢ，你好 12！") → "AB你好12"
    """
    text = unicodedata.normalize("NFKC", text)
    text = _NOT_CER_CHAR.sub("", text)
    return text.upper()


def normalize_for_classification(text: str) -> str:
    """话术分类前的归一：数字串换成 #，去掉标点和空白，字母大写。

    例：normalize_for_classification("这个手镯两千八百块！") → "这个手镯#块"
        normalize_for_classification("这个手镯2800块")     → "这个手镯#块"
    """
    text = unicodedata.normalize("NFKC", text).upper()
    text = _SPACE.sub("", text)
    text = _ARABIC_NUMBER.sub(NUMBER_PLACEHOLDER, text)
    text = _CHINESE_NUMBER.sub(NUMBER_PLACEHOLDER, text)
    return _NOT_CLS_CHAR.sub("", text)
