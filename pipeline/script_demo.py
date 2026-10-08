"""剧本文本演示：没有录音时，直接拿剧本台词当输入，跑步骤 5（数字规范化）和步骤 6（话术分类），
再和剧本里的标注对照，看哪些句子标对了、哪些是误报。可以接着用步骤 8 导出核查初稿。

注意：这里没有经过语音识别，时间是按语速估算的（config.yaml 的 script_demo.chars_per_minute，默认每分钟 220 字），
所以只能用来演示和测"分类、规范化"本身，不能代表录音上的效果。
"""
import re
from datetime import datetime

from pipeline import methods as method_registry
from pipeline.config import load_config
from pipeline.data import FLAG_LABELS, get_script, load_lines
from pipeline.schema import NOTICE, new_segment
from pipeline.step6_classify import classify_segments

DEMO_NOTICE = "演示模式：直接使用剧本文字，没有经过语音识别；时间为估算"

_EFFECTIVE = re.compile(r"[一-鿿A-Za-z0-9]")


def script_segments(script_id: str, chars_per_minute: int = 220) -> list[dict]:
    """把一个剧本的台词变成段落：说话人用剧本角色，时间按有效字数和语速累计估算。

    每段带 source="script_demo"，以及剧本标注 gold_label（类别）和 gold_numbers（数字标准写法）。
    """
    script = get_script(script_id)  # 剧本不存在时抛 KeyError
    lines = [line for line in load_lines() if line["script_id"] == script_id]
    segments, t = [], 0.0
    for line in lines:
        chars = line.get("effective_chars") or len(_EFFECTIVE.findall(line["text"]))
        seconds = max(chars, 1) / chars_per_minute * 60
        segments.append(new_segment(
            t, t + seconds, speaker=line["speaker"], text=line["text"], source="script_demo",
            gold_label=line["label"], gold_numbers=list(line.get("numbers") or [])))
        t += seconds
    assert len(segments) == len(script["lines"])
    return segments


def compare_with_gold(segments: list[dict]) -> dict:
    """和剧本标注对照，返回统计：
    total 句数；flagged 被标疑似的句数；correct_flags 标对的（类别和剧本一致）；
    false_positives 误报（剧本标"正常讲解"却被标疑似）；missed 漏标（剧本是 5 类之一却没标）；
    fp_rate 误报率 = 误报 ÷ 剧本里正常讲解的句数。
    """
    normal = [s for s in segments if s.get("gold_label") == "正常讲解"]
    flagged = [s for s in segments if s.get("label")]
    false_positives = [s for s in normal if s.get("label")]
    return {
        "total": len(segments),
        "flagged": len(flagged),
        "correct_flags": sum(1 for s in flagged if s.get("category") == s.get("gold_label")),
        "false_positives": len(false_positives),
        "missed": sum(1 for s in segments if s.get("gold_label") in FLAG_LABELS and not s.get("label")),
        "normal_total": len(normal),
        "fp_rate": round(len(false_positives) / len(normal), 3) if normal else 0.0,
    }


def run_script_demo(script_id: str, cfg: dict | None = None, methods: dict | None = None):
    """剧本文本演示：返回 (段落列表, meta, 对照统计)。methods 可以指定 normalize、classify 用哪种做法。"""
    cfg = dict(cfg or load_config())
    if methods:
        cfg["methods"] = {**cfg.get("methods", {}), **methods}
    funcs = method_registry.resolve(cfg)
    cpm = int((cfg.get("script_demo") or {}).get("chars_per_minute", 220))

    segments = script_segments(script_id, cpm)
    segments = [funcs["normalize"](seg, "spoken", cfg) for seg in segments]
    segments = classify_segments(segments, cfg, method=funcs["classify"])
    stats = compare_with_gold(segments)

    script = get_script(script_id)
    meta = {
        "file": script_id,
        "title": script.get("title", ""),
        "duration": round(segments[-1]["end"], 2) if segments else 0.0,
        "sha256": "（剧本文本演示，没有录音文件）",
        "processed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "models": {"asr": "未使用（剧本文本演示）"},
        "options": {"chars_per_minute": cpm},
        "methods": {slot: (cfg.get("methods") or {}).get(slot, "baseline") for slot in ("normalize", "classify")},
        "notice": NOTICE,
        "demo_notice": DEMO_NOTICE,
        "source": "script_demo",
    }
    return segments, meta, stats
