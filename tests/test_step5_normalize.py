"""步骤 5（数字、证号、金额规范化与提取）的测试。

剧本上的正确率只打印、不要求 100%；已知做不到的例子写成 xfail，并写明原因，留给第 4 组改进。
"""
import pytest

from pipeline.data import load_lines
from pipeline.methods import get_method
from pipeline.step5_normalize import (
    MAIN_TYPES,
    NUMBER_TYPES,
    extract_entities,
    extract_numbers,
    normalize_baseline,
    number_type,
    numbers_accuracy,
    spoken_to_digits,
)


# ---------------- 计划里列出的测试 ----------------

@pytest.mark.parametrize(
    "value, expected",
    [
        ("2800元", "金额"),
        ("0871-0000-6688", "电话"),
        ("YN-0000-3721", "证号"),
        ("HT-20260000-118", "合同号"),
        ("DD-0000-5566", "订单号"),
        ("15:40", "时刻"),
        ("10月5日", "日期"),
        ("40分钟", "时长"),
        ("3盒", "数量"),
        ("第2天", "数量"),
        ("99999", "其他"),
    ],
)
def test_number_type(value, expected):
    assert number_type(value) == expected


def test_spoken_phone():
    assert "0871-0000-6688" in spoken_to_digits("零八七一 零零零零 六六八八")


def test_spoken_cert():
    assert "YN-0000-3721" in spoken_to_digits("YN 零零零零 三七二一")


def test_spoken_money():
    text = spoken_to_digits("这个手镯两千八百块")
    assert "2800元" in text
    assert extract_numbers(text) == ["2800元"]


def test_protected_words():
    assert "1下" not in spoken_to_digits("大家等一下")
    assert "1般" not in spoken_to_digits("一般来说")


def test_time():
    assert "15:40" in spoken_to_digits("下午三点四十集合")


def _fmt(value) -> str:
    return f"{value:.2f}"


def test_script_accuracy_report():
    result = numbers_accuracy(load_lines())
    print("\n剧本台词上的数字提取（spoken_to_digits + extract_numbers，按行集合比较）")
    print(f"{'类型':<6}{'金标准':>6}{'命中':>6}{'提取':>6}{'正确率':>8}{'召回率':>8}")
    for name in NUMBER_TYPES:
        row = result["by_type"][name]
        print(f"{name:<6}{row['gold']:>6}{row['hit']:>6}{row['pred']:>6}"
              f"{_fmt(row['precision']):>8}{_fmt(row['recall']):>8}")
    for key, title in (("main", "主指标"), ("secondary", "次要指标")):
        row = result[key]
        print(f"{title:<5}{row['gold']:>6}{row['hit']:>6}{row['pred']:>6}"
              f"{_fmt(row['precision']):>8}{_fmt(row['recall']):>8}")
    assert set(result["by_type"]) == set(NUMBER_TYPES)
    assert result["main"]["recall"] >= 0.6


@pytest.mark.xfail(reason="已知失败：cn2an 把“两个半小时”转成“2个0.5小时”，基线没有处理“X个半”的规则（留给第 4 组）")
def test_known_failure_two_and_half_hours():
    assert "2.5小时" in extract_numbers(spoken_to_digits("石林给大家留足两个半小时"))


@pytest.mark.xfail(reason="已知失败：cn2an 认不出“三四十”这种约数，原样保留汉字，基线提取不到“30-40元”（留给第 4 组）")
def test_known_failure_rough_range():
    assert "30-40元" in "".join(extract_numbers(spoken_to_digits("鲜花饼外面也就三四十块一盒吧")))


@pytest.mark.xfail(reason="已知失败：“九点五十”前面没有上午、下午等时间词，基线不敢当成时刻；cn2an 也转不了（留给第 4 组）")
def test_known_failure_time_without_context():
    assert "9:50" in spoken_to_digits("现在是九点五十")


def test_extract_entities():
    assert extract_entities("我们是雾隐行舟旅行社的") == ["雾隐行舟旅行社"]


# ---------------- 补充的测试 ----------------

def test_type_constants():
    assert NUMBER_TYPES == ["金额", "电话", "证号", "合同号", "订单号", "时刻", "日期", "数量", "时长", "其他"]
    assert MAIN_TYPES == ["金额", "电话", "证号", "合同号", "订单号", "时刻", "日期"]


@pytest.mark.parametrize(
    "value, expected",
    [
        ("30-40元/盒", "金额"),
        ("38元/克", "金额"),
        ("2800多元", "金额"),
        ("2016年", "日期"),
        ("2026年10月18日", "日期"),
        ("15日", "日期"),
        ("20多年", "时长"),
        ("1个多小时", "时长"),
        ("7个工作日", "时长"),
        ("6天5晚", "时长"),
        ("第1排", "数量"),
        ("2号门", "数量"),
        ("806房间", "数量"),
        ("10%", "数量"),
        ("5日游", "其他"),
        ("12301", "其他"),
    ],
)
def test_number_type_more(value, expected):
    assert number_type(value) == expected


@pytest.mark.parametrize(
    "spoken",
    [
        "合同编号 HT 二零二六 零零零零 幺幺八，第二页",
        "HT 杠 二零二六 四个零 杠 一一八。",
        "合同编号是 HT 杠 二零二六零零零零 杠 一一八。",
        "HT，二零二六，四个零，幺幺八。",
    ],
)
def test_spoken_contract(spoken):
    assert "HT-20260000-118" in spoken_to_digits(spoken)


@pytest.mark.parametrize(
    "spoken, expected",
    [
        ("我念啊：DD 零零零零 五五六六。", "DD-0000-5566"),
        ("对，DD 杠 四个零 杠 五五六六，就是这个。", "DD-0000-5566"),
        ("YN，四个零，三七二幺，是这个吧？", "YN-0000-3721"),
    ],
)
def test_spoken_codes_other_forms(spoken, expected):
    assert expected in extract_numbers(spoken_to_digits(spoken))


def test_protected_words_inside_numbers_still_convert():
    # “十一点半”里有“一点”、“一万一千”里有“万一”，它们是数字的一部分，不能当成保护词
    assert "11" in spoken_to_digits("十一点半在那边吃午饭")
    assert "11000" in spoken_to_digits("一共一万一千块")


def test_time_context_words():
    assert "13:00" in spoken_to_digits("中午一点在景区外面吃团餐")
    assert "8:30" in spoken_to_digits("早上八点半出发")
    assert "20:20" in spoken_to_digits("我是晚上八点二十的")
    assert "15:10" in spoken_to_digits("下午三点十分准时上车")
    assert "14:00" in spoken_to_digits("下午两点那班还有位置")


def test_extract_from_display_text():
    # 显示模式：SenseVoice 已经把数字转成阿拉伯数字，只做格式统一和提取
    text = "导游证号YN00003721，电话0871 0000 6688，下午3点40集合，一共2800块。"
    assert extract_numbers(text) == ["YN-0000-3721", "0871-0000-6688", "15:40", "2800元"]
    # SenseVoice 自带 zh.wav 在显示模式下的识别结果（见 docs/progress.md）
    assert extract_numbers("开放时间早上9点至下午5点。") == ["9:00", "17:00"]
    assert extract_numbers("合同HT20260000118，订单DD 0000 5566") == ["HT-20260000-118", "DD-0000-5566"]


def test_extract_dates():
    assert extract_numbers(spoken_to_digits("二零二六年十月十八日")) == ["2026年10月18日"]
    assert extract_numbers(spoken_to_digits("十月十四号骑马")) == ["10月14日"]
    assert extract_numbers(spoken_to_digits("今天十五号，第三天")) == ["15日", "第3天"]
    assert extract_numbers(spoken_to_digits("二零一六年压的古树茶")) == ["2016年"]


def test_extract_money_forms():
    assert extract_numbers(spoken_to_digits("家里电水壶才一百多块")) == ["100多元"]
    assert extract_numbers("票价每人280元，补5.5元") == ["280元", "5.5元"]
    assert extract_numbers("一克20元/克") == ["20元/克"]
    assert extract_numbers("六十八块钱喝了三泡茶") == []  # 没转数字前什么也提取不到
    assert extract_numbers(spoken_to_digits("六十八块钱喝了三泡茶")) == ["68元", "3泡"]


def test_extract_durations_and_quantities():
    assert extract_numbers(spoken_to_digits("走高速大概还要一个多小时")) == ["1个多小时"]
    assert extract_numbers(spoken_to_digits("我们会在暮山特产超市停四十分钟")) == ["40分钟"]
    assert extract_numbers(spoken_to_digits("约两个小时")) == ["2小时"]
    assert extract_numbers(spoken_to_digits("我们一共二十二个人")) == ["22人"]
    assert extract_numbers(spoken_to_digits("这六天五晚的自费账单")) == ["6天5晚"]
    assert extract_numbers(spoken_to_digits("一般七个工作日到账")) == ["7个工作日"]
    assert extract_numbers(spoken_to_digits("A区是第一排到第十排")) == ["第1排", "第10排"]
    assert extract_numbers(spoken_to_digits("景区要扣百分之十的手续费")) == ["10%"]


def test_bare_numbers_are_money():
    # 口语里说价钱常常不说"块"，基线把不带单位的两位以上的数当成金额
    assert extract_numbers(spoken_to_digits("这只标价十八万八千八，标签在这儿")) == ["188800元"]
    assert extract_numbers(spoken_to_digits("熟普外面卖六百八，今天收四百八")) == ["680元", "480元"]
    assert extract_numbers(spoken_to_digits("差不多的熟普才两百多。")) == ["200多元"]
    # 带了别的单位的数不算钱
    assert extract_numbers(spoken_to_digits("我爸今年七十二岁了")) == []
    assert extract_numbers(spoken_to_digits("开店十二周年")) == []
    # 一位数不算（"一、二、三"是在点人数）
    assert extract_numbers(spoken_to_digits("好，一、二、三，三个人")) == ["3人"]


def test_day_before_you_chuan():
    assert extract_numbers(spoken_to_digits("十五号游船，每人一百二")) == ["15日", "120元"]


def test_liang_before_measure_word():
    # cn2an 有时不转"两"（"两盒""两家"原样保留），基线补一条规则
    assert extract_numbers(spoken_to_digits("买两盒送一个手提袋")) == ["2盒", "1个"]
    assert extract_numbers(spoken_to_digits("你把两家搞混了")) == ["2家"]
    assert "2000" in spoken_to_digits("两千")


def test_names_not_converted():
    # 名称里的"拾""七"不能转成数字
    assert spoken_to_digits("拾光滇程旅行社出") == "拾光滇程旅行社出"
    assert extract_entities(spoken_to_digits("鹿鸣七彩旅行社的马导")) == ["鹿鸣七彩旅行社"]
    assert "收拾" in spoken_to_digits("大家先回房间收拾一下")


def test_extract_numbers_dedup_keeps_order():
    assert extract_numbers("12:30集合，不是，12:30，还有58元和58元") == ["12:30", "58元"]


def test_extract_entities_longest_first_and_order():
    text = "先去晓月银坊，再回澄溪假日酒店，晓月银坊明天还去"
    assert extract_entities(text) == ["晓月银坊", "澄溪假日酒店"]
    # 人物称呼（如“周导”）和号码不算名称
    assert extract_entities("我是周导，电话0871-0000-6688") == []


def test_normalize_spoken_mode():
    seg = {"start": 0.0, "end": 2.0, "speaker": "导游", "text": "这个手镯两千八百块", "label": ""}
    out = normalize_baseline(seg, "spoken", {})
    assert out["text_raw"] == "这个手镯两千八百块"
    assert out["text"] == "这个手镯2800元"
    assert out["numbers"] == ["2800元"]
    assert out["entities"] == []
    assert seg["text"] == "这个手镯两千八百块"  # 不改调用方传进来的段落


def test_normalize_spoken_keeps_existing_text_raw():
    seg = {"start": 0.0, "end": 2.0, "speaker": "导游", "text": "停四十分钟", "label": "",
           "text_raw": "停四十分钟呀"}
    out = normalize_baseline(seg, "spoken", {})
    assert out["text_raw"] == "停四十分钟呀"
    assert out["numbers"] == ["40分钟"]


def test_normalize_display_mode():
    seg = {"start": 0.0, "end": 2.0, "speaker": "导游", "text": "雾隐行舟旅行社，团费3800元。", "label": ""}
    out = normalize_baseline(seg, "display", {})
    assert out["text"] == "雾隐行舟旅行社，团费3800元。"
    assert "text_raw" not in out
    assert out["numbers"] == ["3800元"]
    assert out["entities"] == ["雾隐行舟旅行社"]


def test_normalize_bad_mode():
    with pytest.raises(ValueError):
        normalize_baseline({"text": "一百块"}, "eval", {})


def test_normalize_registered_as_baseline():
    assert get_method("normalize", "baseline") is normalize_baseline


def test_numbers_accuracy_counts():
    lines = [
        {"text": "这个手镯两千八百块", "numbers": ["2800元"]},
        {"text": "下午三点四十集合，停四十分钟", "numbers": ["15:40", "40分钟", "1盒"]},
        {"text": "现在是九点五十", "numbers": ["9:50"]},
    ]
    result = numbers_accuracy(lines)
    assert result["by_type"]["金额"] == {"gold": 1, "hit": 1, "pred": 1, "precision": 1.0, "recall": 1.0}
    assert result["by_type"]["时刻"]["gold"] == 2
    assert result["by_type"]["时刻"]["hit"] == 1
    assert result["by_type"]["数量"]["recall"] == 0.0
    assert result["by_type"]["电话"]["precision"] == 0.0  # 没有提取到也没有金标准时记 0.0
    assert result["main"]["gold"] == 3
    assert result["main"]["hit"] == 2
    assert result["secondary"]["gold"] == 2
    assert result["secondary"]["hit"] == 1


def test_numbers_accuracy_with_method():
    # 可以换成别的做法（例如第 4 组的 g4）来算正确率；这里用一个什么都提取不到的做法
    def empty_method(segment, mode, cfg):
        return dict(segment, numbers=[])

    result = numbers_accuracy([{"text": "两千八百块", "numbers": ["2800元"]}], method=empty_method)
    assert result["main"]["hit"] == 0
    assert result["main"]["recall"] == 0.0


# ---- 集成时补的回归测试（审查意见）----

def test_decimal_after_yidian_not_protected():
    """"一点五元"是小数，不能因为保护"一点"而变成"一点5元"（再被提取成 5元）。"""
    assert extract_numbers(spoken_to_digits("一共一点五元")) == ["1.5元"]
    assert "1" not in spoken_to_digits("再便宜一点吧")


def test_bare_numbers_with_other_units_not_money():
    """带"月""米"等单位的数不能被当成金额。"""
    assert extract_numbers("我们10月份报的团，海拔3000米") == []


def test_empty_hotword_list_does_not_break(monkeypatch):
    """热词表为空时，保护名称的正则不能匹配到每个位置。"""
    import pipeline.step5_normalize as s5

    monkeypatch.setattr(s5, "load_hotwords", lambda: [])
    assert s5.spoken_to_digits("这个手镯两千八百块") == "这个手镯2800元"
