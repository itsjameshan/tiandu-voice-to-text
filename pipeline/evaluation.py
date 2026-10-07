"""测评：在数据池的录音（或剧本台词文字）上算指标，写成 CSV 表格和 Markdown 报告。

tools/evaluate.py（测一种做法）和 tools/compare.py（基线和改进各测一遍、出对比表）都调用这里的函数，
命令行工具本身只负责读参数、打印进度。指标怎么算、怎么看，见 docs/guides/evaluation.md。

现在有三个指标（说话人、话术分类、片段起止三个指标以后加在本文件后面）：
    cer       字错率：录音 → 降噪 → 增强 → 端点检测 → 识别（测评模式，汉字读法）→ 热词纠错，
              把各段 text_raw 连起来，和参考文本 references/<文件编号>.txt 比（pipeline.align.cer_details）。
    hotwords  专名正确率、过度纠正次数：识别流程同上，数法见 hotword_stats 的说明。
    numbers   数字提取的正确率、召回率：不用录音，在剧本 1055 句台词上测（pipeline.step5_normalize.numbers_accuracy）。

基线做法：
    1. 要测哪些录音：只看数据池清单 manifest.csv（入池时写的），不去翻 normalized/ 文件夹，
       这样没入池、入池失败的文件不会混进来。可以用 files 只测其中几段。
    2. 识别结果缓存：72 段录音约 5.4 小时，在 4 核电脑上识别一遍要 40 分钟左右。识别结果（带 text_raw 的段落）
       存进 数据池/asr_cache/<文件编号>.eval-<钥匙>.json。"钥匙"是下面这些东西合起来算的一个短指纹：
         - 录音文件本身的 SHA-256（复录换了文件 → 指纹变了）；
         - 降噪、增强、端点检测三个槽位用的做法名字，以及这三个做法所在代码文件的指纹（改了代码就重新识别）；
         - config.yaml 的 vad、asr 两节参数，以及与这三个做法同名的一节参数（如各组自己加的 g1: 一节）；
         - 识别模型的文件夹名。
       只要这些都没变，就直接用缓存，不再识别；所以只换热词纠错（hotword）做法、开关热词纠错时，
       不用重新识别，几秒钟就出结果。入池工具在复录时会删掉 asr_cache/<文件编号>.* ，缓存也跟着失效。
       怀疑缓存有问题时，命令行加 --no-cache 强制重新识别。
    3. 汇总：全体、按录音条件（Q 安静 / N 嘈杂教室 / F 口袋或远距离）、按组。
       汇总行的字错率 =（各段错字 + 漏字 + 多字之和）÷（各段参考文本字数之和），不是各段字错率的平均。
    4. 报告：write_report 写 <名字>.csv（UTF-8 带 BOM，Excel 能直接打开）和 <名字>.md，
       Markdown 开头写数据池版本、做法、关键参数、生成时间和固定的局限说明 FIXED_LIMITATION。
可改进方向：
    这是教师模板，各组不改本文件（改自己的 pipeline/groups/gN_*.py，再用这里的工具测）。可以讨论的有：
    过度纠正现在按次数估算，可以改成逐字对齐后逐处判断；字错率还可以按说话人、按段落长短分开看。
测评指标：
    本文件就是测评工具；tests/test_evaluation.py 用模型自带的测试音频改名建一个临时数据池，
    检查流程能跑通、缓存有效、报告格式正确。在云端开发环境里不报告任何字错率数字。

注意：本文件导入时只用标准库；识别模型、ffmpeg 等在函数里用到时才加载。
"""
import copy
import csv
import hashlib
import inspect
import json
import logging
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# 每份 Markdown 报告开头都写这句话（红线第 7 条：如实报告）
FIXED_LIMITATION = "剧本数据上的测评结果不代表真实场景的效果"

# 数据池还没冻结过任何版本时，报告里"数据池版本"写这个
NOT_FROZEN = "未冻结（请先运行 tools/freeze_pool.py）"

# 识别（步骤 2）用到的槽位：这几个槽位换了做法，识别结果就会变，要重新识别
ASR_SLOTS = ("denoise", "enhance", "vad")

# 字错率、专名正确率涉及的槽位（报告里列出这几个槽位用的做法）
AUDIO_SLOTS = ("denoise", "enhance", "vad", "hotword")

# 报告的中文标题
REPORT_TITLES = {
    "cer": "字错率（cer）",
    "hotwords": "专名正确率与过度纠正（hotwords）",
    "numbers": "数字提取正确率（numbers）",
}

# CSV 和 Markdown 表格里，英文字段名对应的中文列名（本来就是中文的列名原样写出）
COLUMN_TITLES = {
    "stem": "文件编号",
    "group": "组",
    "condition": "录音条件",
    "files": "录音数",
    "seconds": "录音时长（秒）",
    "n_ref": "参考文本字数",
    "sub": "错字",
    "dele": "漏字",
    "ins": "多字",
    "cer": "字错率",
    "names_ref": "参考文本中的专名次数",
    "names_hit": "识别结果中命中的次数",
    "name_accuracy": "专名正确率",
    "variants_ref": "说错或简称的名称次数",
    "overcorrected": "过度纠正次数",
    "corrections": "热词纠错次数",
    "gold": "金标准个数",
    "pred": "提取个数",
    "hit": "命中个数",
    "precision": "正确率",
    "recall": "召回率",
}

# 对比表（compare.py）里每个指标比较哪几个数：[(字段名, 中文名)]。
# 只有一个数时分组项只写"全体""Q（安静）"……；有几个数时写成"全体·专名正确率"。
COMPARE_VALUES = {
    "cer": [("cer", "字错率")],
    "hotwords": [("name_accuracy", "专名正确率"), ("overcorrected", "过度纠正次数")],
    "numbers": [("precision", "正确率"), ("recall", "召回率")],
}

# 每个指标的报告里写的说明（数法、口径）
CER_NOTES = [
    "字错率 =（错字 + 漏字 + 多字）÷ 参考文本字数。两边先去掉标点和空格、全角转半角、字母转大写，再逐字对齐。"
    "越低越好；多字很多时可能超过 1。",
    "识别用测评模式（不把数字转成阿拉伯数字，输出汉字读法），和参考文本的口径一致；"
    "一段录音的各段识别文字连起来，和整篇参考文本比。",
    "汇总行的字错率是把各段录音的错字、漏字、多字和字数分别加起来再除，不是各段字错率的平均。",
    "热词纠错（hotword 槽位）会改识别文字，所以它的做法和开关（hotword.enabled）也会影响字错率。",
    "比例都写成小数，如 0.1234 即 12.34%。",
]

HOTWORD_NOTES = [
    "专名：data/hotwords.txt 里的名称（虚构旅行社、店名、地名、行话）。识别结果用热词纠错之后的测评模式文字（text_raw）。",
    "数之前，参考文本和识别结果都先去掉标点和空格、全角转半角、字母转大写；"
    "再把 data/hotword_variants.csv 里人物故意说错或简称的名称（spoken_variant）挖掉，这些位置不算专名。",
    "数专名：从左往右找，同一个位置有长短几个名称都对得上时只算最长的那个"
    "（\"雾隐行舟旅行社\"不再另算\"雾隐行舟\"和\"雾隐\"），找到后跳过这几个字接着找。",
    "专名正确率 = 命中次数 ÷ 参考文本里的专名次数；每个名称的命中次数 = min(它在参考文本里的次数, 它在识别结果里的次数)。"
    "越高越好。参考文本里一个专名都没有时记 0，请同时看次数。",
    "过度纠正：按正确名称（correct_name）分别数。参考文本里它的各种说错、简称的说法（spoken_variant）一共出现 n 次。"
    "用下面两种办法数，取大的那个，最多记 n 次。越少越好。",
    "办法 A（纠错改掉的）：每种说法，纠错前的识别文字里有 b 次、纠错后剩 a 次、参考文本里有 r 次，"
    "记 min(r, b) − a 次（小于 0 记 0），各种说法加起来。也就是：识别对了的说错名称被热词纠错改掉了，不管改成了什么。",
    "办法 B（多出来的正确名称）：数正确名称\"一家\"在纠错后的识别文字里比参考文本里多出几次（小于 0 记 0）。"
    "\"一家\"是正确名称本身，加上热词表里是它的一部分、又不在它的说错说法里的简称："
    "听松阁玉器行 一家还有 听松阁，雾隐行舟旅行社 一家还有 雾隐行舟，松风晚渡旅行社 一家还有 松风晚渡，百年茶语 一家还有 茶语。"
    "（热词纠错只能换成一样长的热词，说错的\"听松坊\"只会被改成\"听松阁\"，所以简称也要算。）",
    "过度纠正是按次数估算的，没有逐字对齐：办法 B 在同一段录音里会互相抵消（一处正确名称被识别错、"
    "另一处说错的名称又被改成正确名称），这种情况靠办法 A 数出来；纠错前就识别错了的说错名称被改成正确名称时，靠办法 B 数。"
    "识别时就把说错的名称写成了正确名称（没有经过纠错），办法 B 也会算进去，所以热词纠错关着时过度纠正也可能不是 0。",
    "热词纠错开关（hotword.enabled）关闭时，基线做法不做任何纠错。专名正确率和过度纠正要一起看，只看一个会误导。",
]

NUMBERS_NOTES = [
    "在剧本台词文字上测（data/lines.csv，不用录音）：每句台词（汉字读法）用 normalize 做法（mode=\"spoken\"）"
    "转换并提取数字，和这一句的 numbers（标准写法）比较，同一句里去重后按集合比。",
    "正确率 = 命中个数 ÷ 提取个数；召回率 = 命中个数 ÷ 金标准个数；分母为 0 时记 0，请同时看个数。",
    "主指标：金额、电话、证号、合同号、订单号、时刻、日期；次要指标：数量、时长、其他。"
    "类型判断规则见 pipeline/step5_normalize.py 的 number_type。",
]

NOTES = {"cer": CER_NOTES, "hotwords": HOTWORD_NOTES, "numbers": NUMBERS_NOTES}


# ======================== 做法 ========================


def check_methods(methods: dict | None) -> None:
    """检查 {槽位: 做法名} 里的槽位和做法都存在；不存在时抛 ValueError（中文信息里列出可用的做法）。"""
    from pipeline.methods import SLOT_TITLES, SLOTS, available, load_all

    load_all()  # 让各组文件里的做法都登记进来
    for slot, name in (methods or {}).items():
        if slot not in SLOTS:
            raise ValueError(f"不认识的槽位“{slot}”。可用的槽位：{'、'.join(SLOTS)}")
        names = available(slot)
        if name not in names:
            raise ValueError(f"槽位“{slot}”（{SLOT_TITLES[slot]}）没有叫“{name}”的做法。可用的做法：{'、'.join(names)}")


def with_methods(cfg: dict, methods: dict | None) -> dict:
    """返回一份新的配置：在 cfg["methods"] 的基础上，换上 methods 里指定的做法（{槽位: 做法名}）。

    methods 里的做法和 config.yaml 里写的做法都检查一遍，写错了（如把 g1 写成 g11）就抛 ValueError，
    中文信息里列出可用的做法；不然要等到识别时才报一大段英文错。
    """
    from pipeline.methods import SLOTS

    check_methods(methods)
    cfg = copy.deepcopy(cfg)
    cfg["methods"] = dict(cfg.get("methods") or {})
    cfg["methods"].update(methods or {})
    from_config = {slot: _method_name(cfg, slot) for slot in SLOTS if slot not in (methods or {})}
    try:
        check_methods(from_config)
    except ValueError as e:
        raise ValueError(f"config.yaml 的 methods 一节写错了：{e}") from None
    return cfg


def _method_name(cfg: dict, slot: str) -> str:
    """配置里这个槽位用的做法名字，没写就是 baseline。"""
    return (cfg.get("methods") or {}).get(slot) or "baseline"


def _get_method(cfg: dict, slot: str) -> Callable:
    from pipeline.methods import get_method, load_all

    load_all()
    return get_method(slot, _method_name(cfg, slot))


# ======================== 数据池 ========================


def pool_version(root) -> str:
    """数据池最新冻结的版本名（如 v1，见 pipeline.pool.current_version）；还没冻结过时返回 NOT_FROZEN。"""
    from pipeline.pool import current_version

    return current_version(root) or NOT_FROZEN


def pool_items(root, files: list[str] | None = None) -> list[dict]:
    """数据池清单（manifest.csv）里的录音，按文件编号排好。

    每项：stem（文件编号，如 G1-S1-Q）、wav（转换后的录音路径）、reference（参考文本路径）、
    group（组号，整数）、condition（录音条件代码 Q / N / F）。
    files：只要其中几段，写文件编号（如 ["G1-S1-Q"]，带不带 .wav 都行）；有不在清单里的就抛 ValueError。
    """
    from pipeline.pool import parse_recording_name, pool_paths, read_csv_rows

    root = Path(root)
    paths = pool_paths(root)
    items = {}
    for row in read_csv_rows(paths["manifest"]):
        stem = (row.get("文件编号") or "").strip()
        if not stem:
            continue
        info = parse_recording_name(f"{stem}.wav")
        wav = (row.get("转换后文件") or "").strip() or f"normalized/{stem}.wav"
        items[stem] = {
            "stem": stem,
            "wav": root / wav,
            "reference": paths["references"] / f"{stem}.txt",
            "group": info["group"],
            "condition": info["condition"],
        }

    if files:
        wanted = [Path(name.strip()).stem if "." in name else name.strip() for name in files]  # 去掉扩展名
        unknown = [stem for stem in wanted if stem not in items]
        if unknown:
            raise ValueError(f"这些录音不在数据池清单（manifest.csv）里：{'、'.join(unknown)}。"
                             f"文件编号形如 G1-S1-Q；请先入池（tools/ingest_pool.py），或检查编号有没有写错")
        return [items[stem] for stem in sorted(set(wanted))]
    return [items[stem] for stem in sorted(items)]


def _checked_items(root, files) -> list[dict]:
    """取要测的录音，并在开始识别之前检查录音和参考文本都在（不要等了几十分钟才发现少文件）。"""
    from pipeline.pool import pool_paths

    items = pool_items(root, files)
    if not items:
        raise ValueError(f"数据池里还没有入池的录音（{pool_paths(root)['manifest']} 不存在或是空的）。"
                         f"请先把录音放进 raw 文件夹，运行 tools/ingest_pool.py")
    no_wav = [item["stem"] for item in items if not item["wav"].is_file()]
    if no_wav:
        raise FileNotFoundError(f"清单里有、但找不到转换后的录音：{'、'.join(no_wav)}（应该在 normalized 文件夹里）。"
                                f"请检查数据池是否拷全了")
    no_ref = [item["stem"] for item in items if not item["reference"].is_file()]
    if no_ref:
        raise FileNotFoundError(f"找不到参考文本：{'、'.join(no_ref)}（应该在 references 文件夹里）。"
                                f"请先运行 python tools/export_references.py 生成参考文本，再校对")
    return items


def read_reference(path) -> str:
    """读参考文本。记事本存的 UTF-8（带不带 BOM 都行）或 GBK 都能读。"""
    data = Path(path).read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("gb18030")


# ======================== 识别（带缓存） ========================


def _source_hash(func: Callable) -> str:
    """做法函数所在代码文件的指纹：同学改了自己文件里的代码，指纹就变了，缓存跟着失效。"""
    try:
        source = Path(inspect.getsourcefile(func)).read_bytes()
    except (TypeError, OSError):  # 找不到源文件（很少见）：只用函数名
        return f"{func.__module__}.{func.__qualname__}"
    return hashlib.sha256(source).hexdigest()[:12]


def cache_key(wav, cfg: dict) -> str:
    """识别结果缓存的"钥匙"：12 位的短指纹。会影响识别结果的东西变了，钥匙就变（见本文件开头的说明）。

    热词纠错、说话人分离、话术分类等识别之后才做的步骤不算在里面，换它们的做法时可以接着用缓存。
    """
    from pipeline.audio import sha256_file
    from pipeline.models import model_path

    model = model_path(cfg, "sense_voice_model")
    parts = {
        "wav": sha256_file(wav),
        "vad": cfg.get("vad"),
        "asr": cfg.get("asr"),
        "model": f"{model.parent.name}/{model.name}",
        "methods": {},
        "code": {},
        "group_params": {},
    }
    for slot in ASR_SLOTS:
        name = _method_name(cfg, slot)
        parts["methods"][slot] = name
        parts["code"][slot] = _source_hash(_get_method(cfg, slot))
        if name in cfg:  # 各组在 config.yaml 里加的同名一节参数（如 g1:）
            parts["group_params"][name] = cfg[name]
    text = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _save_cache(path: Path, segments: list[dict], meta: dict) -> None:
    """写缓存：先写临时文件再改名，写到一半断电或几台电脑同时写也不会留下半个文件。写不进去就算了（只是慢一点）。"""
    from pipeline.schema import write_json

    tmp = path.with_name(path.name + ".tmp")
    try:
        write_json(tmp, segments, meta)
        os.replace(tmp, path)
    except OSError as e:
        logger.warning("识别结果缓存写不进去（不影响这次测评，只是下次要重新识别）：%s", e)
        if tmp.exists():
            tmp.unlink()


def recognize_item(root, item: dict, cfg: dict, use_cache: bool = True) -> tuple[list[dict], dict]:
    """识别数据池里的一段录音（测评模式），返回 (段落列表, 情况)。

    段落带 text_raw（汉字读法、没有标点），还没做热词纠错。
    情况：{"seconds": 录音时长（秒）, "cached": 是否用了缓存, "elapsed": 这次识别用了几秒}。
    cfg["methods"] 决定降噪、增强、端点检测用哪个做法（见 with_methods）。
    """
    from pipeline import step2_vad, step3_asr
    from pipeline.audio import SR, read_wav
    from pipeline.pool import pool_paths
    from pipeline.schema import read_json

    key = cache_key(item["wav"], cfg)
    folder = pool_paths(root)["asr_cache"]
    cache = folder / f"{item['stem']}.eval-{key}.json"  # 以文件编号开头：复录时入池工具会删掉它
    if use_cache and cache.is_file():
        try:
            segments, meta = read_json(cache)
            if meta.get("key") == key:
                return segments, {"seconds": float(meta["seconds"]), "cached": True, "elapsed": 0.0}
        except (OSError, ValueError, KeyError, TypeError):
            pass  # 缓存文件坏了：重新识别

    started = time.perf_counter()
    samples = read_wav(item["wav"])
    seconds = round(len(samples) / SR, 2)
    processed, segments = step2_vad.detect_speech(samples, SR, cfg)
    segments = step3_asr.recognize(processed, SR, segments, cfg, mode="eval")
    elapsed = time.perf_counter() - started

    folder.mkdir(parents=True, exist_ok=True)
    meta = {
        "stem": item["stem"],
        "key": key,
        "mode": "eval",
        "methods": {slot: _method_name(cfg, slot) for slot in ASR_SLOTS},
        "seconds": seconds,
        "elapsed": round(elapsed, 1),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_cache(cache, segments, meta)
    return segments, {"seconds": seconds, "cached": False, "elapsed": elapsed}


def _run_audio_eval(root, cfg: dict, files, progress, use_cache: bool,
                    score: Callable[[str, str, list[dict], str], tuple[dict, str]]) -> list[dict]:
    """cer 和 hotwords 共用的流程：逐段录音 识别（或用缓存）→ 热词纠错 → score 算这一段的指标。

    score(参考文本, 识别文字, 纠错后的段落, 纠错前的识别文字) 返回 (指标字典, 进度里显示的一句话)。
    识别文字 = 各段 text_raw 连起来。返回每段录音一行：stem、group、condition、指标……、seconds。
    """
    def say(message: str) -> None:
        if progress is not None:
            progress(message)
        else:
            logger.info(message)

    items = _checked_items(root, files)
    hotword = _get_method(cfg, "hotword")
    rows = []
    cached = 0
    started = time.perf_counter()
    for n, item in enumerate(items, start=1):
        head = f"[{n}/{len(items)}] {item['stem']}"
        say(f"{head}：处理中……")
        segments, info = recognize_item(root, item, cfg, use_cache)
        # 和 run_pipeline 的测评模式一样：纠错之前先把 text 设成 text_raw（有的做法可能只看 text）
        segments = [dict(seg, text=seg.get("text_raw") or "") for seg in segments]
        before = "".join(seg["text"] for seg in segments)  # 纠错之前的识别文字（数过度纠正要用）
        segments = hotword(segments, None, cfg)  # 热词纠错（hotwords=None 表示用 data/hotwords.txt）
        hypothesis = "".join(seg.get("text_raw") or "" for seg in segments)
        values, text = score(read_reference(item["reference"]), hypothesis, segments, before)
        rows.append({"stem": item["stem"], "group": item["group"], "condition": item["condition"],
                     **values, "seconds": info["seconds"]})
        if info["cached"]:
            cached += 1
            how = "用了缓存的识别结果"
        else:
            how = f"识别用时 {info['elapsed']:.0f} 秒"
        say(f"{head}：完成（{how}），{text}")
    minutes = (time.perf_counter() - started) / 60
    say(f"全部完成：{len(items)} 段录音，用时 {minutes:.1f} 分钟（其中 {cached} 段用了缓存的识别结果）")
    return rows


# ======================== 汇总 ========================


def _summarize(rows: list[dict], sum_keys: list[str], add_rates: Callable[[dict], dict]) -> dict:
    """把每段录音的行加起来：全体、按录音条件、按组。add_rates 在加好的计数后面补上比例。

    返回 {"all": {...}, "by_condition": {"Q": {...}}, "by_group": {1: {...}}, "overview": [带"分组项"的行]}。
    """
    from pipeline.pool import CONDITIONS

    def total(part: list[dict]) -> dict:
        stats = {"files": len(part)}
        for key in sum_keys:
            stats[key] = sum(row[key] for row in part)
        stats["seconds"] = round(sum(row["seconds"] for row in part), 1)
        return add_rates(stats)

    by_condition = {code: total([r for r in rows if r["condition"] == code])
                    for code in CONDITIONS if any(r["condition"] == code for r in rows)}
    by_group = {group: total([r for r in rows if r["group"] == group])
                for group in sorted({r["group"] for r in rows})}
    summary = {"all": total(rows), "by_condition": by_condition, "by_group": by_group}

    overview = [{"分组项": "全体", **summary["all"]}]
    overview += [{"分组项": f"{code}（{CONDITIONS[code]}）", **stats} for code, stats in by_condition.items()]
    overview += [{"分组项": f"第 {group} 组", **stats} for group, stats in by_group.items()]
    summary["overview"] = overview
    return summary


def _key_params(cfg: dict) -> dict:
    """报告里写的关键参数：端点检测、识别、热词纠错的参数和识别模型。"""
    from pipeline.models import model_path

    params = {}
    for section in ("vad", "asr", "hotword"):
        for key, value in (cfg.get(section) or {}).items():
            params[f"{section}.{key}"] = value
    params["识别模型"] = model_path(cfg, "sense_voice_model").parent.name
    return params


def _audio_info(root, cfg: dict) -> dict:
    return {"pool": str(root), "pool_version": pool_version(root),
            "methods": {slot: _method_name(cfg, slot) for slot in AUDIO_SLOTS},
            "params": _key_params(cfg)}


# ======================== 字错率 ========================


def _cer_rates(stats: dict) -> dict:
    errors = stats["sub"] + stats["dele"] + stats["ins"]
    stats["cer"] = errors / stats["n_ref"] if stats["n_ref"] else 0.0
    return stats


def eval_cer(root, cfg: dict, methods: dict | None = None, files: list[str] | None = None,
             progress: Callable[[str], None] | None = None, use_cache: bool = True) -> tuple[list[dict], dict]:
    """字错率测评。返回 (每段录音一行, 汇总)。

    methods：{槽位: 做法名}，如 {"denoise": "g1"}；没写的槽位按 cfg["methods"]，那里也没写就用 baseline。
    files：只测这几段录音（文件编号）；不填就测清单里的全部录音。
    progress：每段录音开始、结束时各调用一次 progress(中文说明)，命令行传 print 就能看到进度。
    use_cache：False 时不用缓存，全部重新识别。
    每行：stem、group、condition、cer、sub（错字）、dele（漏字）、ins（多字）、n_ref（参考文本字数）、seconds（录音时长）。
    汇总：all（全体）、by_condition（按条件）、by_group（按组）、overview（这三样排成表）、tables、info（报告开头用）。
    """
    from pipeline.align import cer_details

    cfg = with_methods(cfg, methods)

    def score(reference: str, hypothesis: str, segments: list[dict], before: str):
        details = cer_details(reference, hypothesis)
        return details, f"字错率 {details['cer']:.4f}"

    rows = _run_audio_eval(root, cfg, files, progress, use_cache, score)
    summary = _summarize(rows, ["n_ref", "sub", "dele", "ins"], _cer_rates)
    summary["info"] = _audio_info(root, cfg)
    summary["tables"] = [{"title": "汇总（全体、按录音条件、按组）", "rows": summary["overview"]}]
    return rows, summary


# ======================== 专名正确率、过度纠正 ========================


def count_names(text: str, names: list[str]) -> Counter:
    """数 text 里每个名称出现几次：从左往右找，同一个位置长短几个名称都对得上时只算最长的，找到后跳过这几个字。

    例：count_names("雾隐行舟旅行社和雾隐", ["雾隐", "雾隐行舟旅行社"]) → {"雾隐行舟旅行社": 1, "雾隐": 1}
    """
    words = sorted({name for name in names if name}, key=len, reverse=True)  # 长的排前面
    counts = Counter()
    i = 0
    while i < len(text):
        for word in words:
            if text.startswith(word, i):
                counts[word] += 1
                i += len(word)
                break
        else:  # 这个位置没有名称，往后挪一个字
            i += 1
    return counts


def name_families(variants: list[dict], hotwords: list[str]) -> dict[str, list[str]]:
    """每个正确名称的"一家"：正确名称本身，加上热词表里是它的一部分、又不在它任何一种说错/简称说法里的名称。

    热词纠错只能把一段字换成一样长的热词，所以说错的"听松坊"（3 个字）只会被改成"听松阁"，
    不会被改成"听松阁玉器行"；数过度纠正时这些简称也要算。用 data/hotword_variants.csv 和 data/hotwords.txt：
        听松阁玉器行   → [听松阁玉器行, 听松阁]
        雾隐行舟旅行社 → [雾隐行舟旅行社, 雾隐行舟]（"雾隐"也在说错的"雾隐晚渡"里，不算）
        松风晚渡旅行社 → [松风晚渡旅行社, 松风晚渡]（"松风"也在说错的"松风行舟"里，不算）
        晓月银坊       → [晓月银坊]（"晓月"也在说错的"晓月阁"里，不算）
    variants 的名称和 hotwords 要用同样的办法归一过（hotword_stats 里都先 normalize_for_cer）。
    """
    spoken_of: dict[str, list[str]] = {}
    for row in variants:
        spoken_of.setdefault(row["correct_name"], []).append(row["spoken_variant"])
    families = {}
    for correct, spoken in spoken_of.items():
        family = [correct] + [name for name in hotwords
                              if name and name in correct and not any(name in variant for variant in spoken)]
        families[correct] = list(dict.fromkeys(family))  # 去掉重复，保持顺序
    return families


def hotword_stats(reference: str, hypothesis: str, hotwords: list[str], variants: list[dict],
                  before: str | None = None) -> dict:
    """一段录音的专名正确率和过度纠正次数（数法也写在报告的说明 HOTWORD_NOTES 里）。

    reference：参考文本；hypothesis：热词纠错之后的识别文字（测评模式 text_raw 连起来）；
    hotwords：热词表；variants：data/hotword_variants.csv 的行（spoken_variant 说错或简称的说法，correct_name 正确名称）；
    before：热词纠错之前的识别文字（测评时一定会给；不给就只用下面第 3 步的办法 B）。

    1. 所有文字都先用 normalize_for_cer 归一（去标点和空格、全角转半角、字母大写），名称也一样归一。
    2. 专名：先把两边文字里的说错/简称的说法挖掉（这些位置不算专名），再用 count_names 数每个热词出现几次
       （长名称优先）。每个名称的命中次数 = min(参考文本里的次数, 识别结果里的次数)。
       专名正确率 = 命中次数之和 ÷ 参考文本里的次数之和（参考文本里没有专名时记 0.0）。
    3. 过度纠正：按正确名称分别数。参考文本里它的各种说错/简称说法共出现 n 次。用两种办法数，取大的那个，最多记 n 次：
       A. 纠错改掉的：每种说法，纠错前的识别文字里有 b 次、纠错后剩 a 次，参考文本里有 r 次，
          记 min(r, b) − a 次（小于 0 记 0）；各种说法加起来。意思是：识别对了的说错名称被纠错改掉了（不管改成什么）。
       B. 多出来的正确名称：用 count_names 数正确名称"一家"（见 name_families，如 听松阁玉器行 和 听松阁）
          一共出现几次，纠错后的识别文字比参考文本多出几次就记几次（小于 0 记 0）。
          意思是：说错的名称最后变成了正确名称（被纠错改的，或者识别时就写成了正确名称）。
       两种办法都没有逐字对齐，是按次数估算：B 在同一段录音里会互相抵消（一处正确名称识别错了、
       另一处说错的名称又被改成正确名称），这种情况靠 A 数出来；纠错前就识别错的说法被改成正确名称时，靠 B 数。
    返回 names_ref、names_hit、name_accuracy、variants_ref、overcorrected，
    以及 by_name（{名称: {"ref", "hit"}}）、by_correct（{正确名称: {"variants", "over"}}）给报告的明细表用。
    """
    from pipeline.text_norm import normalize_for_cer

    ref = normalize_for_cer(reference)
    hyp = normalize_for_cer(hypothesis)
    names = [normalize_for_cer(word) for word in hotwords]
    pairs = [(normalize_for_cer(row["spoken_variant"]), normalize_for_cer(row["correct_name"])) for row in variants]
    spoken = [variant for variant, _ in pairs]

    # 专名：先挖掉说错/简称的说法（换成一个"|"，名称不会跨过它），长的说法先挖
    ref_masked, hyp_masked = ref, hyp
    for variant in sorted(set(spoken), key=len, reverse=True):
        if variant:
            ref_masked = ref_masked.replace(variant, "|")
            hyp_masked = hyp_masked.replace(variant, "|")
    ref_counts = count_names(ref_masked, names)
    hyp_counts = count_names(hyp_masked, names)
    by_name = {name: {"ref": n, "hit": min(n, hyp_counts[name])} for name, n in ref_counts.items()}
    names_ref = sum(item["ref"] for item in by_name.values())
    names_hit = sum(item["hit"] for item in by_name.values())

    # 过度纠正：按正确名称分别数
    families = name_families([{"spoken_variant": v, "correct_name": c} for v, c in pairs], names)
    spoken_ref = count_names(ref, spoken)
    spoken_before = count_names(normalize_for_cer(before), spoken) if before is not None else None
    spoken_after = count_names(hyp, spoken)
    by_correct = {}
    removed = Counter()  # 办法 A：纠错改掉的说错名称，按正确名称加起来
    for variant, correct in pairs:
        if not spoken_ref[variant]:
            continue
        entry = by_correct.setdefault(correct, {"variants": 0, "over": 0})
        entry["variants"] += spoken_ref[variant]
        if spoken_before is not None:
            removed[correct] += max(0, min(spoken_ref[variant], spoken_before[variant]) - spoken_after[variant])
    for correct, entry in by_correct.items():
        family = families[correct]
        extra = sum(count_names(hyp, family).values()) - sum(count_names(ref, family).values())  # 办法 B
        entry["over"] = min(entry["variants"], max(removed[correct], extra, 0))

    return {
        "names_ref": names_ref,
        "names_hit": names_hit,
        "name_accuracy": names_hit / names_ref if names_ref else 0.0,
        "variants_ref": sum(entry["variants"] for entry in by_correct.values()),
        "overcorrected": sum(entry["over"] for entry in by_correct.values()),
        "by_name": by_name,
        "by_correct": by_correct,
    }


def _hotword_rates(stats: dict) -> dict:
    stats["name_accuracy"] = stats["names_hit"] / stats["names_ref"] if stats["names_ref"] else 0.0
    return stats


def eval_hotwords(root, cfg: dict, methods: dict | None = None, files: list[str] | None = None,
                  progress: Callable[[str], None] | None = None, use_cache: bool = True) -> tuple[list[dict], dict]:
    """专名正确率、过度纠正测评（第 5 组主指标）。参数同 eval_cer；数法见 hotword_stats。

    每行：stem、group、condition、names_ref、names_hit、name_accuracy、variants_ref、overcorrected、
    corrections（热词纠错一共改了几处）、seconds。
    汇总同 eval_cer，另外 tables 里多两张明细表：各专名的命中情况、过度纠正明细。
    """
    from pipeline.data import load_hotword_variants, load_hotwords

    cfg = with_methods(cfg, methods)
    hotwords = load_hotwords()
    variants = load_hotword_variants()
    by_name: dict[str, Counter] = {}
    by_correct: dict[str, Counter] = {}

    def score(reference: str, hypothesis: str, segments: list[dict], before: str):
        stats = hotword_stats(reference, hypothesis, hotwords, variants, before=before)
        for name, item in stats.pop("by_name").items():
            by_name.setdefault(name, Counter()).update(item)
        for name, item in stats.pop("by_correct").items():
            by_correct.setdefault(name, Counter()).update(item)
        stats["corrections"] = sum(len(seg.get("corrections") or []) for seg in segments)
        text = f"专名正确率 {stats['name_accuracy']:.4f}，过度纠正 {stats['overcorrected']} 次"
        return stats, text

    rows = _run_audio_eval(root, cfg, files, progress, use_cache, score)
    summary = _summarize(rows, ["names_ref", "names_hit", "variants_ref", "overcorrected", "corrections"],
                         _hotword_rates)
    summary["info"] = _audio_info(root, cfg)

    # 明细：命中最少（错得最多）的名称排前面，方便看哪些名称最需要纠错
    name_rows = [{"名称": name, "参考文本中出现": c["ref"], "识别结果中命中": c["hit"],
                  "专名正确率": c["hit"] / c["ref"] if c["ref"] else 0.0}
                 for name, c in by_name.items()]
    name_rows.sort(key=lambda r: (r["专名正确率"], -r["参考文本中出现"], r["名称"]))
    over_rows = [{"正确名称": name, "说错或简称的出现次数": c["variants"], "过度纠正次数": c["over"]}
                 for name, c in sorted(by_correct.items())]
    summary["tables"] = [
        {"title": "汇总（全体、按录音条件、按组）", "rows": summary["overview"]},
        {"title": "各专名的命中情况（错得多的排前面）", "rows": name_rows},
        {"title": "过度纠正明细（按正确名称）", "rows": over_rows},
    ]
    return rows, summary


# ======================== 数字提取（剧本文字） ========================


def eval_numbers_lines(cfg: dict, methods: dict | None = None) -> dict:
    """在剧本 1055 句台词上测数字提取（第 4 组主指标），不用录音。

    methods：如 {"normalize": "g4"}；不填就用 cfg["methods"]["normalize"]（默认 baseline）。
    返回 numbers_accuracy 的结果：by_type（每种类型的 gold、hit、pred、precision、recall）、main（主指标）、
    secondary（次要指标）；另加 overview（主指标、次要指标、各类型排成的表）和 info（报告开头用）。
    """
    from pipeline.data import load_lines
    from pipeline.step5_normalize import NUMBER_TYPES, numbers_accuracy

    cfg = with_methods(cfg, methods)
    name = _method_name(cfg, "normalize")
    lines = load_lines()
    result = numbers_accuracy(lines, method=_get_method(cfg, "normalize"), cfg=cfg)
    result["overview"] = ([{"分组项": "主指标", **result["main"]}, {"分组项": "次要指标", **result["secondary"]}]
                          + [{"分组项": kind, **result["by_type"][kind]} for kind in NUMBER_TYPES])
    result["tables"] = []
    result["info"] = {"pool": None, "pool_version": "不用数据池（在剧本台词文字上测）",
                      "methods": {"normalize": name}, "params": {"台词句数": len(lines)}}
    return result


# ======================== 对比 ========================


def compare_rows(metric: str, base_summary: dict, new_summary: dict) -> list[dict]:
    """把基线和改进两次测评的汇总排成对比表：每行 分组项、基线、改进、差值（= 改进 − 基线）。

    分组项的顺序和汇总一样：字错率等是 全体 → Q/N/F → 各组；数字提取是 主指标 → 次要指标 → 各类型。
    """
    values = COMPARE_VALUES[metric]
    new_rows = {row["分组项"]: row for row in new_summary["overview"]}
    rows = []
    for base_row in base_summary["overview"]:
        label = base_row["分组项"]
        new_row = new_rows.get(label)
        if new_row is None:
            continue
        for key, title in values:
            base, new = base_row[key], new_row[key]
            rows.append({"分组项": f"{label}·{title}" if len(values) > 1 else label,
                         "基线": round(base, 4), "改进": round(new, 4), "差值": round(new - base, 4)})
    return rows


# ======================== 写报告 ========================


def _columns(rows: list[dict]) -> list[str]:
    """表格的列：按各行字段第一次出现的顺序。"""
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def _cell(key: str, value) -> str:
    """表格里一格的写法：小数保留 4 位（录音时长保留 1 位），没有值写空。"""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.1f}" if key == "seconds" else f"{value:.4f}"
    return str(value)


def _markdown_table(rows: list[dict]) -> list[str]:
    if not rows:
        return ["（没有数据）"]
    columns = _columns(rows)
    lines = ["| " + " | ".join(COLUMN_TITLES.get(c, c) for c in columns) + " |",
             "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(_cell(c, row.get(c)) for c in columns) + " |")
    return lines


def _header_lines(info: dict) -> list[str]:
    """报告开头：数据池版本、数据池、做法、关键参数、生成时间、局限说明。"""
    from pipeline.methods import SLOT_TITLES

    methods = "；".join(f"{SLOT_TITLES.get(slot, slot)} {slot}={name}"
                        for slot, name in (info.get("methods") or {}).items())
    params = "；".join(f"{key}={value}" for key, value in (info.get("params") or {}).items())
    lines = [f"- 数据池版本：{info.get('pool_version') or NOT_FROZEN}"]
    if info.get("pool"):
        lines.append(f"- 数据池：{info['pool']}")
    lines += [
        f"- 做法：{methods or '（无）'}",
        f"- 关键参数：{params or '（无）'}",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 局限：{FIXED_LIMITATION}",
    ]
    return lines


def write_report(out_dir, name: str, rows: list[dict], summary: dict, notes: list[str]) -> tuple[Path, Path]:
    """写测评报告：<out_dir>/<name>.csv（rows，UTF-8 带 BOM、中文表头）和 <out_dir>/<name>.md。

    Markdown 的内容依次是：标题；数据池版本、做法、关键参数、生成时间、局限说明（取自 summary["info"]）；
    说明（notes，一条一行）；summary["tables"] 里的各张表（每张 {"title", "rows"}）；
    最后是 rows 这张表（标题用 summary["rows_title"]，没有就写"明细"）。
    返回 (CSV 路径, Markdown 路径)。文件正用 Excel 打开着时，Windows 会报 PermissionError。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    md_path = out_dir / f"{name}.md"

    columns = _columns(rows)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([COLUMN_TITLES.get(c, c) for c in columns])
        for row in rows:
            writer.writerow([_cell(c, row.get(c)) for c in columns])

    title = REPORT_TITLES.get(name) or REPORT_TITLES.get(name.removeprefix("compare_")) or name
    if name.startswith("compare_"):
        title = f"对比 · {title}"
    lines = [f"# 测评报告：{title}", ""]
    lines += _header_lines(summary.get("info") or {})
    if notes:
        lines += ["", "## 说明", ""] + [f"- {note}" for note in notes]
    for table in summary.get("tables") or []:
        lines += ["", f"## {table['title']}", ""] + _markdown_table(table["rows"])
    rows_title = summary.get("rows_title") or "明细"
    lines += ["", f"## {rows_title}（和 {csv_path.name} 的内容相同）", ""] + _markdown_table(rows)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path
