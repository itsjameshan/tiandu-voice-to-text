"""热词纠错：识别以后，把"拼音相近、字不同"的名称改成热词表里的正确名称（步骤 3 的槽位 hotword）。

为什么要这样做：
    剧本里有很多虚构的旅行社、商店、景点名称（如"雾隐行舟旅行社"），识别模型没见过，
    常常写成同音或近音的别字（如"雾影行走旅行社"）。sherpa-onnx 里的 SenseVoice 不支持
    "解码时热词"，所以基线在识别之后按拼音找、按拼音换。

基线做法：
    1. 只用热词表（data/hotwords.txt）里 3 个字以上、全是汉字的名称（min_len）；
       两个字的名称（如"周导"）太容易误改，先不用。
    2. 把文字和热词都转成"模糊拼音"（fuzzy_pinyin）：不带声调，再把容易混的音归成一个：
       声母 zh→z、ch→c、sh→s，开头的 l→n；韵母 ing→in、eng→en、ang→an。
    3. 只在连续的汉字里找（标点、空格、数字、字母会把汉字隔开），从左往右逐字看：
       在每个位置先试长的热词、再试短的；取和热词一样长的一段字，字面不同、
       模糊拼音逐个音节比较，对不上的音节数 ≤ max_mismatch（默认 0）就换成热词。
    4. 每次替换记一条 {"from": 原来的字, "to": 热词}。
    5. 槽位 hotword 的基线做法 hotword_baseline：config.yaml 里 hotword.enabled 为 false（默认）时
       原样返回；打开时对每个段落的 text（显示模式）和 text_raw（测评模式）都纠错，
       记录写进段落的 corrections（多记一个 start，方便对照录音）。
可改进方向（第 5 组）：
    两个字的名称怎么纠又不误改；按字形、按常见错法加规则；只在上下文像名称的地方纠错；
    换成真正的"解码时热词"（transducer 模型 + hotwords_file，见 docs/build_spec.md 5.3）。
    注意"过度纠正"：剧中人物故意说错或简称的名称（data/hotword_variants.csv）不能被改成正确名称。
测评指标：
    专名正确率（参考文本里出现的热词，在识别结果里也出现的比例）；
    过度纠正次数（故意说错的名称被改成正确名称的次数）。用 tools/evaluate.py hotwords 计算。
"""
import re
from functools import lru_cache

from pypinyin import lazy_pinyin

from pipeline.data import load_hotwords
from pipeline.methods import register

# 连续的汉字（常用汉字的编码范围）
HAN_RUN = re.compile(r"[\u4e00-\u9fff]+")

# 模糊归并规则：声母（开头）和韵母（结尾）各自按顺序检查，只改第一个对上的
_INITIAL_RULES = [("zh", "z"), ("ch", "c"), ("sh", "s"), ("l", "n")]
_FINAL_RULES = [("ing", "in"), ("eng", "en"), ("ang", "an")]


def _fuzzy_syllable(syllable: str) -> str:
    """把一个拼音音节做模糊归并，例如 zhang → zang → zan，ling → ning → nin。"""
    for old, new in _INITIAL_RULES:
        if syllable.startswith(old):
            syllable = new + syllable[len(old):]
            break
    for old, new in _FINAL_RULES:
        if syllable.endswith(old):
            syllable = syllable[: -len(old)] + new
            break
    return syllable


def fuzzy_pinyin(text: str) -> list[str]:
    """把文字转成模糊拼音，每个字一个元素，例如 "雾影行走" → ["wu", "yin", "xin", "zou"]。

    汉字用 pypinyin.lazy_pinyin 转成不带声调的拼音（一整串一起转，多音字能按词读对），再做模糊归并；
    不是汉字的字符（标点、数字、字母）原样保留，不做模糊归并。
    """
    result = []
    # 把文字切成"一串汉字"和"单个非汉字字符"两种小块
    for part in re.findall(r"[\u4e00-\u9fff]+|.", text, flags=re.S):
        if HAN_RUN.fullmatch(part):
            result.extend(_fuzzy_syllable(syllable) for syllable in lazy_pinyin(part))
        else:
            result.append(part)
    return result


@lru_cache(maxsize=None)
def _word_pinyin(word: str) -> tuple[str, ...]:
    """热词的模糊拼音。热词表每段都要用一遍，算过一次就记住，不重复算。"""
    return tuple(fuzzy_pinyin(word))


def _count_mismatch(pinyin_a, pinyin_b) -> int:
    """两串一样长的拼音，逐个音节比较，有几个对不上。"""
    return sum(1 for a, b in zip(pinyin_a, pinyin_b) if a != b)


def _match_at(run: str, run_pinyin: list[str], i: int, words: list[str],
              word_pinyin: dict[str, tuple[str, ...]], max_mismatch: int) -> str | None:
    """从 run 的第 i 个字开始，找能对上的热词（words 已经按长到短排好）；找不到返回 None。

    "对上"有两种：字面完全一样（本来就是对的，不用改），或者模糊拼音对不上的音节数 ≤ max_mismatch。
    """
    for word in words:
        n = len(word)
        window = run[i:i + n]
        if len(window) < n:
            continue  # 剩下的字不够这个热词长
        if window == word:
            return word
        if _count_mismatch(run_pinyin[i:i + n], word_pinyin[word]) <= max_mismatch:
            return word
    return None


def _correct_run(run: str, words: list[str], word_pinyin: dict[str, tuple[str, ...]],
                 max_mismatch: int, records: list[dict]) -> str:
    """纠正一串连续的汉字，替换记录追加到 records 里，返回纠正后的汉字串。"""
    run_pinyin = fuzzy_pinyin(run)
    pieces = []
    i = 0
    while i < len(run):
        word = _match_at(run, run_pinyin, i, words, word_pinyin, max_mismatch)
        if word is None:
            pieces.append(run[i])  # 这个位置没有热词，保留这个字，往后挪一个字
            i += 1
            continue
        window = run[i:i + len(word)]
        if window != word:
            records.append({"from": window, "to": word})
        pieces.append(word)  # 换成热词（或本来就是热词），跳过这几个字
        i += len(word)
    return "".join(pieces)


def correct_text(text: str, hotwords: list[str], min_len: int = 3,
                 max_mismatch: int = 0) -> tuple[str, list[dict]]:
    """按拼音把 text 里的近音别字换成热词，返回 (纠正后的文字, 替换记录列表)。

    - 只用 min_len 个字以上、全是汉字的热词；长的热词优先；
    - 只在连续汉字里找，窗口长度 = 热词长度；字面不同、模糊拼音对不上的音节数 ≤ max_mismatch 时替换；
    - 替换记录形如 {"from": "雾影行走旅行社", "to": "雾隐行舟旅行社"}，出现几次记几条。

    例：correct_text("我们报的是雾影行走旅行社", ["雾隐行舟旅行社"])
        → ("我们报的是雾隐行舟旅行社", [{"from": "雾影行走旅行社", "to": "雾隐行舟旅行社"}])
    """
    # 去掉重复的热词，只留够长、全是汉字的，再按长到短排好（长热词优先）
    words = [w for w in dict.fromkeys(hotwords) if len(w) >= min_len and HAN_RUN.fullmatch(w)]
    words.sort(key=len, reverse=True)
    if not text or not words:
        return text, []

    word_pinyin = {word: _word_pinyin(word) for word in words}
    records: list[dict] = []
    # 对每一串连续汉字分别纠正，汉字以外的字符原样保留
    fixed = HAN_RUN.sub(lambda m: _correct_run(m.group(), words, word_pinyin, max_mismatch, records), text)
    return fixed, records


@register("hotword", "baseline")
def hotword_baseline(segments: list[dict], hotwords: list[str] | None, cfg: dict) -> list[dict]:
    """热词纠错的基线做法：(段落列表, 热词表, 配置) → 段落列表。

    - cfg["hotword"]["enabled"] 为假（默认）时原样返回，不加 corrections；
    - hotwords 为 None 时用 data/hotwords.txt；
    - 打开时对每段的 text 和 text_raw 都纠错（参数 min_len、max_syllable_mismatch 取自 cfg["hotword"]），
      替换记录写进 corrections，每条形如 {"from", "to", "start"}；
      text 和 text_raw 里的同一处替换只记一次；没有替换的段落不加 corrections。
    返回新的段落列表，不改调用方传进来的段落。
    """
    settings = cfg.get("hotword") or {}
    if not settings.get("enabled"):
        return segments
    if hotwords is None:
        hotwords = load_hotwords()
    min_len = int(settings.get("min_len", 3))
    max_mismatch = int(settings.get("max_syllable_mismatch", 0))

    result = []
    for seg in segments:
        seg = dict(seg)  # 复制一份再改
        records = []
        pairs_in_text = set()  # text 里已经记过的 (原来的字, 热词)
        for field in ("text", "text_raw"):
            if not seg.get(field):
                continue
            seg[field], found = correct_text(seg[field], hotwords, min_len, max_mismatch)
            for item in found:
                pair = (item["from"], item["to"])
                if field == "text":
                    pairs_in_text.add(pair)
                elif pair in pairs_in_text:
                    continue  # text_raw 里的同一处替换，text 里已经记过了
                records.append({"from": item["from"], "to": item["to"], "start": seg.get("start")})
        if records:
            seg["corrections"] = list(seg.get("corrections") or []) + records
        result.append(seg)
    return result
