"""步骤 5：数字、证号、金额规范化与提取（第 4 组的工位，槽位 normalize）。

这一步给每个段落补两个字段：
    numbers   文字里提到的数字的规范写法，如 ["2800元", "15:40", "0871-0000-6688", "YN-0000-3721"]
    entities  文字里提到的旅行社、店铺、酒店、品牌、演出与设施的名称，如 ["雾隐行舟旅行社"]
有两种模式：
    display  显示模式：识别时已经做了逆文本规范化（SenseVoice use_itn=True），文字里已经是阿拉伯数字。
             只做提取，不改 text。
    spoken   汉字读法：剧本文本演示、测评模式的识别结果，数字是汉字读法（"两千八百块"）。
             先把原文存进 text_raw（已经有就不动），再用 spoken_to_digits 把 text 转成阿拉伯数字，然后提取。

基线做法：
    1. spoken_to_digits（汉字读法 → 阿拉伯数字），按顺序做五件事：
       ① 逐位读的号码：前面带两个大写字母的证号、合同号、订单号（"YN 零零零零 三七二一"→"YN-0000-3721"，
          "HT 二零二六 四个零 幺幺八"→"HT-20260000-118"），以及用空格隔开的号码段
          （"零八七一 零零零零 六六八八"→"0871-0000-6688"）。这一步必须最先做，
          因为 cn2an 会把"零八七一"当成一个数，变成"871"，开头的 0 就丢了。
       ② 时刻：前面紧挨着"上午、下午、早上、晚上、中午"的"X点""X点Y""X点Y分""X点半""X点整"，
          以及"X点钟"，转成"15:40"这样的写法（下午、晚上加 12；中午一点到五点也加 12）。
          cn2an 认不出"三点四十"，所以放在 cn2an 前面做。没有这些时间词的"九点五十"不转（怕把"一点"之类误转）。
       ③ 保护词：热词表里的名称（"拾光滇程旅行社"）和 PROTECTED_WORDS 里的词（"一下""一般"……）
          先换成占位符，cn2an 转完再换回来，否则会变成"10光滇程旅行社""1下""1般"。
          前面紧挨着数字的不算保护词（"十一点半"里的"一点"、"一万一千"里的"万一"）。
          另外，量词前面的"两"先改成"2"（cn2an 会漏掉"两盒""两家"）。
       ④ cn2an.transform(text, "cn2an")：把其余的汉字数字转成阿拉伯数字（"两千八百"→"2800"）。
       ⑤ 紧跟在数字后面的"块""块钱"改成"元"（"2800块"→"2800元"）。
    2. extract_numbers：一组正则表达式按优先级依次找（合同号 → 证号/订单号 → 电话 → 时刻 → 日期 → 金额 →
       "第几" → 时长 → 数量 → 公共热线 → 不带单位的数），认出来的部分先涂掉，免得被后面的规则重复认；
       最后按出现顺序去重。最后一条规则：剩下的、不带任何单位的两位以上的数一律当成金额
       （口语里说价钱常常不说"块"，如"标价三万九千八"；后面跟着"岁""周年"等别的单位的不算，见 OTHER_UNITS）。
    3. extract_entities：在文字里找 data/fictional_names.csv 中类型为旅行社、店铺、酒店、品牌、演出与设施的名称，
       长名优先（先找"听松阁玉器行"，再找"听松阁"这种短的）。
    4. number_type（给一个规范写法判断类型，测评时按类型分开统计）。按下面的顺序，先符合哪条就是哪类：
       ① 含"元"                                   → 金额（如 2800元、2800多元、20元/克、30-40元/盒）
       ② 形如 0871-0000-6688（3～4 位-4 位-4 位）   → 电话
       ③ 以"HT-"开头                              → 合同号
       ④ 以"DD-"开头                              → 订单号
       ⑤ 两个大写字母-4 位-4 位（如 YN-0000-3721）   → 证号
       ⑥ 形如 15:40（1～2 位:2 位）                → 时刻
       ⑦ 2016年、10月5日、15日、2026年10月18日 这样的写法（年份必须是 4 位数字，
          所以"20年""7个工作日""5日游"都不算日期）  → 日期
       ⑧ 以数字开头，并且以 分钟、小时、天、晚、年、个月、个多月、个工作日 结尾，或者含"小时""分钟"
                                                   → 时长（如 40分钟、1个多小时、20多年、6天5晚）
       ⑨ 以量词结尾（人、家、盒、克、张、饼、个、泡、样、笔、排、页、条、号门、房间、折、%、毫米、公里），
          或者以"第"开头                           → 数量（如 3盒、10%、第2天）
       ⑩ 其余                                     → 其他（如 12301、999）
可改进方向（第 4 组）：
    - 没有时间词的时刻（"九点五十""十一点二十"，剧本里大多数时刻都是这样说的），要判断上午还是下午；
      还有"差十分八点"、改口时省掉的"晚上"（"晚上八点二十，不对，八点五十"）；
    - "两个半小时"被 cn2an 转成"2个0.5小时"，"半个小时"变成"0.5个小时"；
      "三四十块""八九百"这种约数 cn2an 转不了；
    - 不带单位的数一律当成钱太粗：足银"九九九"、房号"三零六"、电话尾号"六六八八"也被当成了钱
      （提示：逐位读的数多半是编号）；"一克三十八"其实是单价"38元/克"；"一块银板"的"块"是量词；
    - "一百五十一块二""四毛钱"这种带角、毛的金额；
    - "四个零""两个六两个八"这种口语读号码的说法（证号、合同号里的"四个零"基线已经认识）；
    - 补充 PROTECTED_WORDS（cn2an 还会把"一口价""一段""对一对"转成"1口价""1段""对1对"）；
      反过来，保护了"一点"以后，小数"一点五"也不转了。
测评指标：
    numbers_accuracy：拿剧本每行的 numbers 字段当金标准，与提取结果按行比较（同一行里去重后当集合比），
    按类型分开算正确率（precision = 命中数 / 提取数）和召回率（recall = 命中数 / 金标准数）。
    主指标是金额、电话、证号、合同号、订单号、时刻、日期（与纠纷核查直接相关）；次要指标是数量、时长、其他。
    剧本上的结果不代表真实录音的效果。
"""
import re
import warnings
from typing import Callable

import cn2an  # 纯 Python 小库，导入很快，不是重依赖

from pipeline.data import load_fictional_names, load_hotwords
from pipeline.methods import register
from pipeline.schema import LIST_SEP

# 数字的类型；前 7 个是主指标（与纠纷核查直接相关），后 3 个是次要指标
NUMBER_TYPES = ["金额", "电话", "证号", "合同号", "订单号", "时刻", "日期", "数量", "时长", "其他"]
MAIN_TYPES = NUMBER_TYPES[:7]

# 两种模式：spoken（输入是汉字读法）、display（输入已经是阿拉伯数字）
MODES = ("spoken", "display")

# cn2an 之前要保护起来的词（不然"一下"会变成"1下"、"收拾"会变成"收10"）。第 4 组可以往里加。
# 另外，data/hotwords.txt 里的名称也整个保护起来（"拾光滇程旅行社"不能变成"10光滇程旅行社"）
PROTECTED_WORDS = [
    "一下", "一般", "一点", "一个是", "一起", "一直", "一样", "统一", "万一",
    "一定", "一些", "一会", "一共", "一块儿", "一路", "一切", "一边", "一遍", "一声", "一律", "一眼",
    "收拾", "零头",
]

# 计数用的量词（number_type 用）。长的写在前面
QUANTIFIERS = ["号门", "房间", "毫米", "公里", "人", "家", "盒", "克", "张", "饼", "个", "泡", "样", "笔", "排",
               "页", "条", "折", "%"]

# 不带"块""元"的数后面如果跟着这些字，就不当成金额（extract_numbers 最后一条规则用）：
# 时间、年龄、次数等单位，以及"来""余""几"（"二十来分钟"）
OTHER_UNITS = "点分秒岁周回次对遍趟层楼号届度斤件套只片句来余几月米公站路级"

# 时长的结尾（number_type 用）
DURATION_ENDINGS = ["分钟", "小时", "天", "晚", "年", "个月", "个多月", "个工作日"]

# extract_entities 要找的名称类型（人物称呼、号码不算）
ENTITY_TYPES = ["旅行社", "店铺", "酒店", "品牌", "演出与设施"]

# 逐位读号码时，每个汉字对应的数字（"幺"是电话里"一"的读法）
DIGIT_OF = {"零": "0", "〇": "0", "一": "1", "幺": "1", "二": "2", "两": "2", "三": "3", "四": "4",
            "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}

# 汉字数字用到的所有字（判断"前面是不是紧挨着数字"时用）
NUMERAL_CHARS = "零〇一二两三四五六七八九十百千万亿幺"


# ---------------- 类型判断 ----------------

_PHONE_FORMAT = re.compile(r"\d{3,4}-\d{4}-\d{4}")
_CERT_FORMAT = re.compile(r"[A-Z]{2}-\d{4}-\d{4}")
_TIME_FORMAT = re.compile(r"\d{1,2}:\d{2}")
_DATE_FORMAT = re.compile(r"(\d{4}年)?(\d{1,2}月)?(\d{1,2}[日号])?")


def number_type(value: str) -> str:
    """判断一个规范写法属于哪一类（10 类之一）。规则和顺序见本文件开头的说明。"""
    value = value.strip()
    if "元" in value:
        return "金额"
    if _PHONE_FORMAT.fullmatch(value):
        return "电话"
    if value.startswith("HT-"):
        return "合同号"
    if value.startswith("DD-"):
        return "订单号"
    if _CERT_FORMAT.fullmatch(value):
        return "证号"
    if _TIME_FORMAT.fullmatch(value):
        return "时刻"
    if value and _DATE_FORMAT.fullmatch(value):
        return "日期"
    starts_with_digit = value[:1].isdigit()
    if starts_with_digit and (value.endswith(tuple(DURATION_ENDINGS)) or "小时" in value or "分钟" in value):
        return "时长"
    if value.endswith(tuple(QUANTIFIERS)) or value.startswith("第"):
        return "数量"
    return "其他"


# ---------------- 汉字读法 → 阿拉伯数字 ----------------

_D = "[零〇一二三四五六七八九幺]"  # 逐位读时的一个数字
_SEP = r"[\s，,、杠\-]*"  # 号码段之间的分隔：空格、逗号、顿号、"杠"
_G4 = rf"(?:{_D}{{4}}|四个零)"  # 四位号码段（"四个零"就是 0000）

# 合同号：HT + 8 位 + 3 位，如"HT 二零二六 零零零零 幺幺八"
_SPOKEN_CONTRACT = re.compile(rf"(?<![A-Za-z])HT{_SEP}({_G4}{_SEP}{_G4}){_SEP}({_D}{{3}})(?!{_D})")
# 证号、订单号等：两个大写字母 + 4 位 + 4 位，如"YN 零零零零 三七二一"
_SPOKEN_CODE = re.compile(rf"(?<![A-Za-z])(?!HT)([A-Z]{{2}}){_SEP}({_G4}){_SEP}({_G4})(?!{_D})")
# 用空格（半角或全角 \u3000）隔开的号码段（每段至少 3 位），如"零八七一 零零零零 六六八八"
_SPOKEN_GROUPS = re.compile(rf"(?<!{_D}){_D}{{3,}}(?:[ \u3000]+{_D}{{3,}})+(?!{_D})")


def _read_digits(text: str) -> str:
    """把逐位读的号码变成数字串："二零二六 四个零" → "20260000"（分隔符丢掉）。"""
    text = text.replace("四个零", "零零零零")
    return "".join(DIGIT_OF[ch] for ch in text if ch in DIGIT_OF)


def _convert_codes(text: str) -> str:
    """第 ① 步：逐位读的合同号、证号、订单号、电话 → 带连字符的写法。"""
    text = _SPOKEN_CONTRACT.sub(lambda m: f"HT-{_read_digits(m.group(1))}-{_read_digits(m.group(2))}", text)
    text = _SPOKEN_CODE.sub(
        lambda m: f"{m.group(1)}-{_read_digits(m.group(2))}-{_read_digits(m.group(3))}", text)
    text = _SPOKEN_GROUPS.sub(lambda m: "-".join(_read_digits(part) for part in m.group(0).split()), text)
    return text


# 时刻前面的时间词。基线只在这些词后面认时刻；"下午""晚上"要把小时加 12
TIME_WORDS = ["上午", "下午", "早上", "晚上", "中午"]

_HOUR = r"[零一二两三四五六七八九十\d]{1,3}"
# 点后面的部分：半、整、钟、"Y分"、"Y十"之类的分钟数（"三点四十"）、"零五"、两位阿拉伯数字。
# 注意没有单独的"五"：小数"一点五"不是时刻
_MINUTE = (r"半|整|钟|[零一二三四五六七八九十\d]{1,3}分|[一二三四五]?十[一二三四五六七八九]?"
           r"|零[一二三四五六七八九]|\d{2}")
# 时间词 + 小时 + 点 + （分钟），如"下午三点四十"
_TIME_WITH_WORD = re.compile(rf"({'|'.join(TIME_WORDS)})({_HOUR})点({_MINUTE})?")
# 没有时间词的"X点钟"，如"八点钟"（不知道上午还是下午，小时不加 12）
_TIME_OCLOCK = re.compile(rf"(?<![第{NUMERAL_CHARS}\d])({_HOUR})点钟")


def _small_number(text: str) -> int | None:
    """把 0～99 的数字（汉字或阿拉伯数字）变成整数："四十"→40、"十五"→15、"零五"→5、"12"→12。认不出返回 None。"""
    if text.isdigit():
        return int(text)
    try:
        if "十" in text:
            tens, _, ones = text.partition("十")
            tens_value = int(DIGIT_OF[tens]) if tens else 1  # "十五"前面没有字，就是一十五
            ones_value = int(DIGIT_OF[ones]) if ones else 0
            return tens_value * 10 + ones_value
        value = 0
        for ch in text:
            value = value * 10 + int(DIGIT_OF[ch])
        return value
    except KeyError:
        return None


def _time_text(word: str, hour_text: str, minute_text: str) -> str | None:
    """拼出"15:40"这样的时刻（小时前面不补 0）。数字不合理返回 None。

    时间词是"下午""晚上"、小时小于 12 时加 12；"中午"后面的一点到五点也加 12（"中午一点"是 13:00）。
    """
    hour = _small_number(hour_text)
    if minute_text in ("", "整", "钟"):
        minute = 0
    elif minute_text == "半":
        minute = 30
    else:
        minute = _small_number(minute_text.rstrip("分"))
    if hour is None or minute is None or hour > 24 or minute > 59:
        return None
    if word in ("下午", "晚上") and hour < 12:
        hour += 12
    elif word == "中午" and 1 <= hour <= 5:
        hour += 12
    return f"{hour}:{minute:02d}"


def _convert_times(text: str) -> str:
    """第 ② 步：时间词后面的"X点""X点Y""X点Y分""X点半""X点整"，以及"X点钟" → "15:40"。

    小时、分钟写成汉字或阿拉伯数字都可以（显示模式里的"下午3点40"也能转）。
    后面紧跟着数字或"多"的不转（"两点五"是小数，"八点多"说不准几分）。
    没有时间词的"九点五十"不转：不知道是上午还是下午，也怕把"便宜一点"的"一点"当成时刻。
    """

    def followed_by_number(m) -> bool:
        next_char = m.string[m.end():m.end() + 1]
        return bool(next_char) and (next_char in NUMERAL_CHARS or next_char.isdigit() or next_char == "多")

    def with_word(m):
        result = _time_text(m.group(1), m.group(2), m.group(3) or "")
        if result is None or followed_by_number(m):
            return m.group(0)
        return m.group(1) + result  # 时间词保留，如"下午15:40"

    def oclock(m):
        result = _time_text("", m.group(1), "")
        return m.group(0) if result is None or followed_by_number(m) else result

    text = _TIME_WITH_WORD.sub(with_word, text)
    return _TIME_OCLOCK.sub(oclock, text)


# 保护词：前面不能紧挨着数字（"十一点半"里的"一点"、"一万一千"里的"万一"不算）
# "一点"后面紧跟数字时是小数（"一点五元"），不保护
_PROTECTED = re.compile(
    rf"(?<![{NUMERAL_CHARS}\d])("
    + "|".join(re.escape(w) + (rf"(?![{NUMERAL_CHARS}\d])" if w == "一点" else "")
               for w in sorted(PROTECTED_WORDS, key=len, reverse=True))
    + ")")
_PLACEHOLDER_START = 0xE000  # Unicode 私用区的字符，cn2an 不认识，不会动它
# 紧跟在量词前面、前面又不是数字的"两"
_LIANG = re.compile(rf"(?<![{NUMERAL_CHARS}\d])两(?=[人家盒克张饼个泡样笔排页条折晚天年位次回对岁])")


def _protected_names() -> re.Pattern | None:
    """热词表里的名称（长名优先）拼成一个正则，用来整个保护起来。热词表为空时返回 None。"""
    names = sorted((n for n in load_hotwords() if n), key=len, reverse=True)
    if not names:
        return None
    return re.compile("|".join(re.escape(name) for name in names))


def spoken_to_digits(text: str) -> str:
    """把汉字读法的数字转成阿拉伯数字（步骤见本文件开头 ①～⑤）。

    例："这个手镯两千八百块" → "这个手镯2800元"；"下午三点四十集合" → "下午15:40集合"。
    """
    text = _convert_codes(text)  # ① 逐位读的号码
    text = _convert_times(text)  # ② 时刻

    # ③ 名称和保护词换成占位符
    saved = []

    def hide(m):
        saved.append(m.group(0))
        return chr(_PLACEHOLDER_START + len(saved) - 1)

    names = _protected_names()
    if names is not None:
        text = names.sub(hide, text)
    text = _PROTECTED.sub(hide, text)

    # cn2an 有时不转"两"（"两盒""两家"原样保留），量词前面的"两"先改成"2"
    text = _LIANG.sub("2", text)

    # ④ cn2an 转换。转不了的（如"三四十"）它会发警告并原样保留，这里不让警告刷屏
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text = cn2an.transform(text, "cn2an")

    # 占位符换回原来的词
    for i, word in enumerate(saved):
        text = text.replace(chr(_PLACEHOLDER_START + i), word)

    # ⑤ 数字后面的"块""块钱"改成"元"（"多块"也算："100多块"→"100多元"）
    return re.sub(r"(\d)(多)?(?:块钱|块)", r"\1\2元", text)


# ---------------- 从文字里提取规范写法 ----------------

_NUM = r"(?<![\d.])(\d+(?:\.\d+)?)"  # 一个数（整数或小数），前面不能紧挨着别的数字


def _money(m):
    """金额：2800元、2800多元、5.5元、2-3万元、20元/克（"块""块钱"也算元）。"""
    low, high, wan, more, per = m.groups()
    value = low + (f"-{high}" if high else "") + (wan or "") + (more or "") + "元"
    return value + (f"/{per}" if per else "")


def _day(m):
    """单独的"15日""15号"：只认 1～31。"""
    day = int(m.group(1))
    return f"{day}日" if 1 <= day <= 31 else None


def _time(m):
    """时刻：08:30 → 8:30（小时前面不补 0，与剧本写法一致）。"""
    hour, minute = int(m.group(1)), int(m.group(2))
    return f"{hour}:{minute:02d}" if hour <= 24 and minute <= 59 else None


def _duration(m):
    """时长：40分钟、1个多小时、2小时（"2个小时"去掉"个"）、20多年、1个多月、7个工作日。"""
    number, middle, unit = m.group(1), m.group(2) or "", m.group(3)
    if unit in ("月", "工作日"):
        if "个" not in middle:
            return None  # "10月"是日期，不是时长
        return f"{number}{middle}{unit}"
    if unit == "钟头":
        unit = "小时"
    if middle == "个":
        middle = ""
    return f"{number}{middle}{unit}"


def _quantity(m):
    """数量：3盒、357克、10%、80多公里；"3个人""3位"都写成"3人"。"""
    number, more, ge, unit = m.group(1), m.group(2) or "", m.group(3) or "", m.group(4)
    if unit in ("人", "位"):
        return f"{number}{more}人"
    return f"{number}{more}{ge}{unit}"


# 提取规则：(正则, 把匹配结果变成规范写法的函数)。按顺序找，前面的优先
_EXTRACT_RULES: list[tuple[re.Pattern, Callable]] = [
    # 合同号 HT-20260000-118（显示模式里也可能是"HT20260000118"）
    (re.compile(r"(?<![A-Za-z])HT[- ]?(\d{8})[- ]?(\d{3})(?!\d)"), lambda m: f"HT-{m.group(1)}-{m.group(2)}"),
    # 证号、订单号 YN-0000-3721、DD-0000-5566
    (re.compile(r"(?<![A-Za-z])([A-Z]{2})[- ]?(\d{4})[- ]?(\d{4})(?!\d)"),
     lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
    # 固定电话 0871-0000-6688、手机 138-0000-0000
    (re.compile(r"(?<![\d-])(0\d{2,3}|1[3-9]\d)[- ]?(\d{4})[- ]?(\d{4})(?![\d-])"),
     lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
    # 时刻 15:40
    (re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)"), _time),
    # 日期 2026年10月18日、10月18日（"号"写成"日"）
    (re.compile(r"(?<!\d)(?:(\d{4})年)?(\d{1,2})月(\d{1,2})[日号]"),
     lambda m: (f"{m.group(1)}年" if m.group(1) else "") + f"{int(m.group(2))}月{int(m.group(3))}日"),
    # 年份 2016年、2026年10月
    (re.compile(r"(?<!\d)(\d{4})年(?:(\d{1,2})月)?"),
     lambda m: f"{m.group(1)}年" + (f"{int(m.group(2))}月" if m.group(2) else "")),
    # 单独的日 15日、15号（"5日游""2号门""3号楼"不算）
    (re.compile(r"(?<![\d.])(\d{1,2})(?:日(?!游)|号(?![门楼线车房座桌]))"), _day),
    # 金额
    (re.compile(_NUM + r"(?:-(\d+(?:\.\d+)?))?(万)?(多)?(?:元|块钱|块)(?:/(克|盒|个|张|斤|位|人|件|套|饼|只))?"),
     _money),
    # 第几：第2天、第1排、第3家、第2
    (re.compile(r"第(\d+)(天|家|排|页|条|个|次|站|名)?"), lambda m: f"第{m.group(1)}{m.group(2) or ''}"),
    # 几天几晚 6天5晚
    (re.compile(r"(?<![\d.])(\d+)天(\d+)晚(?!上)"), lambda m: f"{m.group(1)}天{m.group(2)}晚"),
    # 时长
    (re.compile(_NUM + r"(个多|个|多)?(分钟|小时|钟头|工作日|月|天|晚(?!上)|年)"), _duration),
    # 数量（带量词）
    (re.compile(_NUM + r"(多)?(个)?(号门|房间|毫米|公里|人|位|家|盒|克|张|饼|个|泡|样|笔|排|页|条|折|%)"),
     _quantity),
    # 公共服务热线 12301、12345 等（放在金额后面，免得把"12345元"拆开）
    (re.compile(r"(?<![\d.\-])(123\d{2})(?![\d.\-])"), lambda m: m.group(1)),
    # 剩下的、不带任何单位的数（两位以上、不以 0 开头）：口语里说价钱常常不说"块"（"标价三万九千八"），
    # 基线一律当成金额。后面跟着 OTHER_UNITS 里的字（"72岁""12周年"）的不算。
    # 会认错的：足银999、房号1218、电话尾号6688（留给第 4 组结合上下文改进）
    (re.compile(rf"(?<![A-Za-z\d.\-:/])([1-9]\d+(?:\.\d+)?)(多)?(?![\d.:\-/A-Za-z{OTHER_UNITS}])"),
     lambda m: f"{m.group(1)}{m.group(2) or ''}元"),
]


def _apply_rule(pattern: re.Pattern, make: Callable, work: str, found: list) -> str:
    """用一条提取规则在 work 里找：认出来的 (位置, 规范写法) 放进 found，并把那一段涂掉。返回涂过的文字。"""

    def take(m):
        value = make(m)
        if not value:
            return m.group(0)  # 这条规则不认，原样留给后面的规则
        found.append((m.start(), value))
        return "\x00" * len(m.group(0))  # 涂掉（长度不变，后面算的位置不会乱）

    return pattern.sub(take, work)


def extract_numbers(text: str) -> list[str]:
    """从（阿拉伯数字的）文字里提取数字的规范写法，去重，按在文字里出现的先后排列。

    例："这个手镯2800元，下午3点40集合" → ["2800元", "15:40"]
    """
    work = _convert_times(text)  # 显示模式里的"下午3点40"也转成"下午15:40"
    found = []  # (位置, 规范写法)
    for pattern, make in _EXTRACT_RULES:
        work = _apply_rule(pattern, make, work, found)

    found.sort(key=lambda item: item[0])
    result = []
    for _, value in found:
        if value not in result:
            result.append(value)
    return result


def extract_entities(text: str) -> list[str]:
    """找出文字里提到的旅行社、店铺、酒店、品牌、演出与设施的名称（长名优先），按出现顺序去重。"""
    names = [row["name"] for row in load_fictional_names() if row["type"] in ENTITY_TYPES]
    names.sort(key=len, reverse=True)  # 长名优先
    work = text
    found = []  # (位置, 名称)
    for name in names:
        start = work.find(name)
        while start != -1:
            found.append((start, name))
            work = work[:start] + "\x00" * len(name) + work[start + len(name):]  # 涂掉，短名不会再在里面找到
            start = work.find(name, start + len(name))
    found.sort(key=lambda item: item[0])
    result = []
    for _, name in found:
        if name not in result:
            result.append(name)
    return result


# ---------------- 登记为基线做法 ----------------

@register("normalize", "baseline")
def normalize_baseline(segment: dict, mode: str, cfg: dict) -> dict:
    """数字规范化的基线做法。返回一个新的段落（不改传进来的段落）。

    mode="spoken"：没有 text_raw 时先把原文存进 text_raw，再把 text 转成阿拉伯数字；
    mode="display"：不改 text。
    两种模式都写 numbers 和 entities。
    """
    if mode not in MODES:
        raise ValueError(f"不认识的模式“{mode}”，只能是 spoken（汉字读法）或 display（显示模式）")
    segment = dict(segment)
    text = segment.get("text", "")
    if mode == "spoken":
        if not segment.get("text_raw"):
            segment["text_raw"] = text
        text = spoken_to_digits(text)
        segment["text"] = text
    segment["numbers"] = extract_numbers(text)
    segment["entities"] = extract_entities(text)
    return segment


# ---------------- 测评 ----------------

def _rates(counts: dict) -> dict:
    """在 gold、hit、pred 三个计数后面补上正确率和召回率（分母为 0 时记 0.0，请同时看计数）。"""
    gold, hit, pred = counts["gold"], counts["hit"], counts["pred"]
    return {
        "gold": gold,
        "hit": hit,
        "pred": pred,
        "precision": round(hit / pred, 4) if pred else 0.0,
        "recall": round(hit / gold, 4) if gold else 0.0,
    }


def numbers_accuracy(lines: list[dict], method: Callable | None = None, cfg: dict | None = None) -> dict:
    """在剧本台词上测数字提取：每行的 numbers 是金标准，按行比较（同一行里去重后当集合比）。

    lines：load_lines() 的结果（每行要有 text 和 numbers）。
    method：要测的 normalize 做法（如第 4 组的 g4），不填就用 spoken_to_digits + extract_numbers；
            填了就用 method(段落, "spoken", cfg) 返回的 numbers。
    返回：{"by_type": {类型: {"gold", "hit", "pred", "precision", "recall"}}, "main": {...}, "secondary": {...}}
        gold = 金标准个数，pred = 提取出的个数，hit = 两边都有的个数；类型用 number_type 判断。
    """
    counts = {name: {"gold": 0, "hit": 0, "pred": 0} for name in NUMBER_TYPES}
    for line in lines:
        gold = line.get("numbers") or []
        if isinstance(gold, str):  # 也接受没拆开的 "2800元；15:40"
            gold = gold.split(LIST_SEP)
        gold = {value.strip() for value in gold if value.strip()}
        if method is None:
            pred = set(extract_numbers(spoken_to_digits(line["text"])))
        else:
            segment = {"start": 0.0, "end": 0.0, "speaker": line.get("speaker", "未知"),
                       "text": line["text"], "label": ""}
            pred = set(method(segment, "spoken", cfg or {}).get("numbers") or [])
        for value in gold:
            counts[number_type(value)]["gold"] += 1
        for value in pred:
            counts[number_type(value)]["pred"] += 1
        for value in gold & pred:
            counts[number_type(value)]["hit"] += 1

    def total(type_names):
        return {key: sum(counts[name][key] for name in type_names) for key in ("gold", "hit", "pred")}

    return {
        "by_type": {name: _rates(counts[name]) for name in NUMBER_TYPES},
        "main": _rates(total(MAIN_TYPES)),
        "secondary": _rates(total(NUMBER_TYPES[7:])),
    }
