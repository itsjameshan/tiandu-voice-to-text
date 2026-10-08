"""热词纠错（pipeline/hotwords.py）的测试。

热词纠错的基线做法：识别以后，在文字里找"拼音相近、字不同"的片段，换成热词表里的正确名称。
这里的测试只用文字，不需要模型。
"""
import subprocess
import sys

from conftest import ROOT

from pipeline import methods
from pipeline.config import load_config
from pipeline.data import load_hotword_variants, load_hotwords
from pipeline.hotwords import correct_text, fuzzy_pinyin, hotword_baseline
from pipeline.schema import new_segment


def _cfg(enabled: bool, **hotword) -> dict:
    """读 config.yaml，再把热词纠错打开或关闭（其余参数用配置文件里的默认值）。"""
    return load_config(overrides={"hotword": {"enabled": enabled, **hotword}})


# ---------- 模糊拼音 ----------

def test_fuzzy_pinyin_rules():
    # 声母 zh→z、ch→c、sh→s，开头 l→n；韵母 ing→in、eng→en、ang→an；不带声调
    assert fuzzy_pinyin("张") == ["zan"]
    assert fuzzy_pinyin("吃") == ["ci"]
    assert fuzzy_pinyin("上") == ["san"]
    assert fuzzy_pinyin("李") == ["ni"]
    assert fuzzy_pinyin("星") == ["xin"]
    assert fuzzy_pinyin("风") == ["fen"]
    assert fuzzy_pinyin("旅") == ["nv"]
    assert fuzzy_pinyin("雾影行走") == fuzzy_pinyin("雾隐行舟")
    assert fuzzy_pinyin("晚渡") != fuzzy_pinyin("行舟")


def test_fuzzy_pinyin_one_item_per_char():
    # 每个字一个元素；不是汉字的字符原样保留，不做模糊归并
    assert fuzzy_pinyin("A区l") == ["A", "qu", "l"]
    text = "我们报的是雾影行走旅行社，好吗？"
    assert len(fuzzy_pinyin(text)) == len(text)


# ---------- correct_text ----------

def test_correct_text_fixes_homophone():
    text, records = correct_text("我们报的是雾影行走旅行社", ["雾隐行舟旅行社"])
    assert "雾隐行舟旅行社" in text
    assert text == "我们报的是雾隐行舟旅行社"
    assert len(records) == 1
    assert records[0]["from"] == "雾影行走旅行社"
    assert records[0]["to"] == "雾隐行舟旅行社"


def test_correct_text_keeps_variant():
    # 长度不同：不会把"松风行舟"改成"松风晚渡旅行社"
    text, records = correct_text("我们报的是松风行舟", ["松风晚渡旅行社"])
    assert text == "我们报的是松风行舟"
    assert records == []
    # 拼音不同（wan du ≠ xing zhou）：不会把"雾隐晚渡"改成"雾隐行舟"
    text, records = correct_text("雾隐晚渡", ["雾隐行舟"])
    assert text == "雾隐晚渡"
    assert records == []


def test_short_hotwords_ignored():
    # "周导"只有 2 个字，默认 min_len=3，不参与纠错（"州导"和"周导"拼音一样也不改）
    text, records = correct_text("这是州导说的", ["周导"])
    assert text == "这是州导说的"
    assert records == []
    # 把 min_len 调成 2 就会参与
    text, records = correct_text("这是州导说的", ["周导"], min_len=2)
    assert text == "这是周导说的"
    assert records == [{"from": "州导", "to": "周导"}]


def test_long_hotword_first():
    # "雾隐行舟"和"雾隐行舟旅行社"都能对上时，用长的那个
    text, records = correct_text("雾影行走旅行社", ["雾隐行舟", "雾隐行舟旅行社"])
    assert text == "雾隐行舟旅行社"
    assert records == [{"from": "雾影行走旅行社", "to": "雾隐行舟旅行社"}]


def test_correct_name_untouched():
    text, records = correct_text("雾隐行舟旅行社的周导", ["雾隐行舟旅行社", "雾隐行舟"])
    assert text == "雾隐行舟旅行社的周导"
    assert records == []


def test_only_inside_han_runs():
    # 标点、空格、数字把汉字隔开时，不跨过它们去配热词
    for text in ["雾影，行走旅行社", "雾影 行走旅行社", "雾影1行走旅行社"]:
        fixed, records = correct_text(text, ["雾隐行舟旅行社"])
        assert fixed == text
        assert records == []
    # 热词前后有标点不影响
    fixed, records = correct_text("是雾影行走旅行社，对吧？", ["雾隐行舟旅行社"])
    assert fixed == "是雾隐行舟旅行社，对吧？"
    assert len(records) == 1


def test_repeated_occurrences_recorded_each_time():
    fixed, records = correct_text("雾影行走旅行社和雾影行走旅行社", ["雾隐行舟旅行社"])
    assert fixed == "雾隐行舟旅行社和雾隐行舟旅行社"
    assert len(records) == 2


def test_max_mismatch_allows_more():
    # 允许 2 个音节对不上时，"雾隐晚渡"就会被改成"雾隐行舟"（这就是"过度纠正"的风险）
    text, records = correct_text("雾隐晚渡", ["雾隐行舟"], max_mismatch=2)
    assert text == "雾隐行舟"
    assert records == [{"from": "雾隐晚渡", "to": "雾隐行舟"}]


def test_empty_inputs():
    assert correct_text("", ["雾隐行舟旅行社"]) == ("", [])
    assert correct_text("雾影行走旅行社", []) == ("雾影行走旅行社", [])


def test_default_settings_keep_all_spoken_variants():
    # 第 5 组剧本里故意说错、简称的名称，用默认参数和完整热词表都不会被改（不过度纠正）
    hotwords = load_hotwords()
    for row in load_hotword_variants():
        variant = row["spoken_variant"]
        text, records = correct_text(variant, hotwords)
        assert text == variant, f"{variant} 被改成了 {text}"
        assert records == []


# ---------- 热词纠错的基线做法（槽位 hotword） ----------

def test_hotword_disabled_noop():
    segments = [new_segment(0, 2, text="我们报的是雾影行走旅行社", text_raw="我们报的是雾影行走旅行社")]
    result = hotword_baseline(segments, ["雾隐行舟旅行社"], _cfg(enabled=False))
    assert result == segments
    assert all("corrections" not in seg for seg in result)


def test_hotword_baseline_fixes_text_and_text_raw():
    segments = [
        new_segment(1.5, 4.0, text="我们报的是雾影行走旅行社。", text_raw="我们报的是雾影行走旅行社"),
        new_segment(4.0, 6.0, text="今天天气不错。", text_raw="今天天气不错"),
    ]
    before = [dict(seg) for seg in segments]
    result = hotword_baseline(segments, ["雾隐行舟旅行社"], _cfg(enabled=True))
    assert result[0]["text"] == "我们报的是雾隐行舟旅行社。"
    assert result[0]["text_raw"] == "我们报的是雾隐行舟旅行社"
    # text 和 text_raw 里是同一处纠错，只记一次
    assert result[0]["corrections"] == [{"from": "雾影行走旅行社", "to": "雾隐行舟旅行社", "start": 1.5}]
    # 没有纠错的段落不加 corrections
    assert result[1] == segments[1]
    assert "corrections" not in result[1]
    # 不改调用方传进来的段落
    assert segments == before


def test_hotword_baseline_text_raw_only():
    # 测评模式只有 text_raw 时也要纠错（第 5 组用 text_raw 算专名正确率）
    segments = [new_segment(0, 2, text_raw="我们报的是雾影行走旅行社")]
    result = hotword_baseline(segments, ["雾隐行舟旅行社"], _cfg(enabled=True))
    assert result[0]["text_raw"] == "我们报的是雾隐行舟旅行社"
    assert result[0]["text"] == ""
    assert result[0]["corrections"] == [{"from": "雾影行走旅行社", "to": "雾隐行舟旅行社", "start": 0.0}]


def test_hotword_baseline_uses_cfg_params():
    segments = [new_segment(0, 2, text="这是州导说的")]
    assert hotword_baseline(segments, ["周导"], _cfg(enabled=True))[0]["text"] == "这是州导说的"
    result = hotword_baseline(segments, ["周导"], _cfg(enabled=True, min_len=2))
    assert result[0]["text"] == "这是周导说的"
    segments = [new_segment(0, 2, text="雾隐晚渡")]
    result = hotword_baseline(segments, ["雾隐行舟"], _cfg(enabled=True, max_syllable_mismatch=2))
    assert result[0]["text"] == "雾隐行舟"


def test_hotword_baseline_default_hotword_list():
    # 热词表传 None 时用 data/hotwords.txt
    segments = [new_segment(0, 2, text="我们报的是雾影行走旅行社")]
    result = hotword_baseline(segments, None, _cfg(enabled=True))
    assert result[0]["text"] == "我们报的是雾隐行舟旅行社"


def test_hotword_baseline_registered():
    assert methods.get_method("hotword", "baseline") is hotword_baseline


def test_light_import():
    # 导入热词模块和识别模块都不加载 sherpa-onnx、gradio、tensorflow（重依赖在函数里才导入）
    code = (
        "import sys; import pipeline.hotwords, pipeline.step3_asr; "
        "heavy = [m for m in ('sherpa_onnx', 'gradio', 'tensorflow') if m in sys.modules]; "
        "assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
