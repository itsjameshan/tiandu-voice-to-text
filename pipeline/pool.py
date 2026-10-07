"""数据池：全班录音的命名检查、入池、参考文本、校对记录（教师模板，学生不改）。

数据池是全班录音、参考文本、标注和清单的统一存放处，放在老师电脑或 U 盘上，
不进代码仓库、不上传网络（老师的操作步骤见 docs/teacher/data_pool.md）。文件夹结构：

    raw/                   学生交来的原始录音（入池后设为只读）
    normalized/            统一格式后的 16000 Hz 单声道 16 位 WAV，如 G1-S1-Q.wav
    references/            每段录音一份参考文本，如 G1-S1-Q.txt
    annotations/speakers/  说话人时间标注
    annotations/clips/     疑似片段起止标注
    asr_cache/             "数据校对"页的识别结果缓存
    digits/                选做：中文数字录音
    versions/              冻结的数据池版本（如 versions/v1/）
    manifest.csv           清单：每段入池的录音一行
    qc_report.csv          质检报告：每段入池的录音一行
    proofread_log.csv      校对记录：每保存一次参考文本记一行
    acceptance.csv         验收表：每段计划录音一行
    acceptance_summary.csv 验收表的各组汇总

基线做法：
    - 命名检查：文件名必须是"组号-剧本号-条件代码.扩展名"，如 G1-S1-Q.m4a，用正则表达式 FILENAME_RE 检查。
      不合格就报错并说明原因（小写、空格、"(1)"、组号超出 1–8……），不自动猜、不自动改名。
    - 入池：用 SHA-256 指纹判断 raw/ 里哪些文件还没入池；新文件用 ffmpeg 转成 normalized/<编号>.wav，
      质检（只报告、不修改；阈值在 config.yaml 的 qc 下面，预计时长来自 data/recording_plan.csv），
      结果写进 qc_report.csv，把原始文件设为只读，最后写进 manifest.csv。
      清单里有这一行才算入池，所以中途出错（如表格正用 Excel 打开着）时，再运行一次就能补完。
    - 复录：同一个编号换了新文件（指纹不同）→ 替换清单和质检报告里的旧行，删掉旧的识别结果缓存。
    - 参考文本：为录音计划里的 72 段录音各写一份初稿（内容照抄剧本台词），已存在的不覆盖（校对过的不会丢）。
    - 校对记录：每次保存参考文本，在 proofread_log.csv 末尾追加一行（录音、校对人、时间、第几遍）。
可改进方向：
    本模块是教师模板，学生不改。可以讨论的是质检阈值：第 1 组用全班试录数据校准音量和静音阈值。
测评指标：
    不涉及识别效果；由 tests/test_pool.py 检查命名检查、入池、跳过已入池文件、复录替换、参考文本、校对记录。

表格（CSV）一律用 UTF-8 带 BOM（utf-8-sig）读写，老师用 Excel 直接打开不会乱码。
表格里的路径都相对于数据池文件夹（如 raw/G1-S1-Q.m4a），整个数据池拷到 U 盘或共享文件夹后照样能用。
"""
import csv
import os
import re
from datetime import datetime
from pathlib import Path

from pipeline.audio import SR, convert_to_wav, find_ffmpeg, probe, read_wav, sha256_file
from pipeline.data import load_recording_plan, script_reference_text

# 支持的格式、质检、"设为只读"都和步骤 1（上传与格式统一）用同一份代码，两边的规则不会不一样
from pipeline.step1_ingest import SUPPORTED_EXTS, _set_read_only, quality_check

# ======================== 文件命名 ========================

# 合格的文件名：G组号(1–8)-S剧本号(1–3)-条件代码(Q/N/F).扩展名，如 G1-S1-Q.m4a
FILENAME_RE = re.compile(r"^G([1-8])-S([1-3])-([QNF])\.([A-Za-z0-9]+)$")

# 录音条件代码 → 中文名称
CONDITIONS = {"Q": "安静", "N": "嘈杂教室", "F": "口袋或远距离"}

# 报错时给出的正确示例
NAME_EXAMPLE = "G1-S1-Q.m4a"


def _name_problems(name: str) -> list[str]:
    """文件名不合格时，找出常见的错误原因（只用来提示，不会据此自动改名）。"""
    problems = []
    # base 是第一个点前面的部分（应该是 G1-S1-Q 这样），rest 是后面的扩展名
    base, dot, rest = name.partition(".")
    if not dot:
        problems.append("没有扩展名（如 .m4a、.wav）")
    elif not rest:
        problems.append("最后多了一个点，点后面没有扩展名（如 .m4a、.wav），可能改名时把扩展名删掉了")
    if "." in rest:
        problems.append("有两个点，扩展名可能重复了（Windows 默认隐藏扩展名，改名时容易写成 G1-S1-Q.m4a.m4a，"
                        "请在资源管理器的\"查看\"里勾选\"文件扩展名\"后再改）")
    if "(" in base or "（" in base:
        problems.append("多了\"(1)\"这样的编号（同一个文件复制两次时电脑会自动加上），请删掉")
    if any(ch.isspace() for ch in name):
        problems.append("有空格，请删掉")
    if any(ch.islower() for ch in base):
        problems.append("用了小写字母，G、S 和条件代码都要大写")
    if "_" in base:
        problems.append("用了下划线\"_\"，要用英文连字符\"-\"")
    if not name.isascii():
        problems.append("有中文或全角符号，条件要写代码 Q、N、F，连字符要用英文的\"-\"")

    # 再按"宽松"的写法拆开看（先把小写问题放一边，连字符可以缺、条件代码后面可以多出东西），
    # 这样能说清楚具体哪一部分不对：G(组号) 连字符 S(剧本号) 连字符 (条件代码)(多出来的部分)
    loose = re.fullmatch(r"G(\d+)([-_]?)S(\d+)([-_]?)([A-Z]*)(.*)", base.strip().upper())
    if loose:
        group, dash1, script, dash2, code, extra = loose.groups()
        if not (dash1 and dash2):
            problems.append("少了连字符\"-\"，组号、剧本号、条件代码之间都要用\"-\"隔开")
        if (len(group) > 1 and group.startswith("0")) or (len(script) > 1 and script.startswith("0")):
            problems.append("组号、剧本号前面不要加 0：写 G1、S1，不写 G01、S01")
        if not 1 <= int(group) <= 8:
            problems.append(f"组号 G{int(group)} 不存在，只有第 1–8 组（G1 到 G8）")
        if not 1 <= int(script) <= 3:
            problems.append(f"剧本号 S{int(script)} 不存在，每组只有 S1、S2、S3 三个剧本")
        extra = extra.strip()
        if not code:
            problems.append("缺少条件代码，最后一部分只能是 Q（安静）、N（嘈杂教室）、F（口袋或远距离）")
        elif code not in CONDITIONS:
            problems.append(f"条件代码 {code} 不对，只能是 Q（安静）、N（嘈杂教室）、F（口袋或远距离）")
        elif extra and not extra.startswith(("(", "（")):  # "(1)" 上面已经说过了
            problems.append(f"条件代码后面多了\"{extra}\"，请删掉（复录的录音也用原来的名字）")

    if not problems:
        problems.append("格式不对")
    return problems


def parse_recording_name(name: str) -> dict:
    """检查录音文件名，合格时返回它代表哪段录音；不合格抛 ValueError（中文说明原因和正确写法）。

    例如 parse_recording_name("G3-S2-F.mp3") 返回
        {"stem": "G3-S2-F", "script_id": "G3-S2", "group": 3, "condition": "F",
         "condition_name": "口袋或远距离", "ext": ".mp3"}
    stem 是"文件编号"（去掉扩展名的文件名），数据池里这段录音的所有文件都用它命名。
    扩展名大小写都可以（手机导出的文件常是 .WAV、.M4A）；扩展名是不是支持的格式由入池时检查。
    """
    name = str(name)
    found = FILENAME_RE.fullmatch(name)  # fullmatch：末尾多一个换行符也不算合格
    if found is None:
        reasons = "；".join(_name_problems(name))
        raise ValueError(
            f"文件名不合格：{name}。原因：{reasons}。正确写法是\"组号-剧本号-条件代码.扩展名\"，"
            f"全部大写、用英文连字符，例如 {NAME_EXAMPLE}；条件代码 Q 安静、N 嘈杂教室、F 口袋或远距离。"
            "请改名后重新放进 raw 文件夹（工具不会自动猜）。"
        )
    group, script, code, ext = found.groups()
    return {
        "stem": f"G{group}-S{script}-{code}",
        "script_id": f"G{group}-S{script}",
        "group": int(group),
        "condition": code,
        "condition_name": CONDITIONS[code],
        "ext": "." + ext.lower(),
    }


def _plan_by_stem() -> dict[str, dict]:
    """录音计划（data/recording_plan.csv，72 段）：文件编号（如 G1-S1-Q）→ 计划里的那一行。"""
    return {Path(item["file_name"]).stem: item for item in load_recording_plan()}


# ======================== 数据池文件夹 ========================

# 数据池里的文件夹（init_pool 会建好；表格在用到时才写）
_POOL_DIRS = ["raw", "normalized", "references", "annotations_speakers", "annotations_clips",
              "asr_cache", "digits", "versions"]


def pool_paths(root) -> dict[str, Path]:
    """数据池里各个文件夹和表格的路径（见本文件开头的文件夹结构）。只算路径，不建文件夹。"""
    root = Path(root)
    return {
        "raw": root / "raw",
        "normalized": root / "normalized",
        "references": root / "references",
        "annotations_speakers": root / "annotations" / "speakers",
        "annotations_clips": root / "annotations" / "clips",
        "asr_cache": root / "asr_cache",
        "digits": root / "digits",
        "versions": root / "versions",
        "manifest": root / "manifest.csv",
        "qc_report": root / "qc_report.csv",
        "proofread_log": root / "proofread_log.csv",
        "acceptance": root / "acceptance.csv",
        "acceptance_summary": root / "acceptance_summary.csv",
    }


def init_pool(root) -> None:
    """建好数据池的全部文件夹（已经存在的不动）。"""
    paths = pool_paths(root)
    for key in _POOL_DIRS:
        paths[key].mkdir(parents=True, exist_ok=True)


# ======================== 表格读写 ========================

# 清单 manifest.csv 的列
MANIFEST_COLUMNS = ["文件编号", "剧本编号", "录音条件", "原始文件", "转换后文件",
                    "时长（秒）", "原始采样率", "质检结果", "上传时间", "SHA-256"]

# 质检报告 qc_report.csv 的列（"质检结果"是"合格"或用"；"隔开的问题）
QC_COLUMNS = ["文件编号", "质检时间", "时长（秒）", "预计时长（秒）", "原始采样率", "原始声道数",
              "问题数", "质检结果"]

# 校对记录 proofread_log.csv 的列
PROOFREAD_COLUMNS = ["文件编号", "校对人", "校对时间", "第几遍", "备注"]

# 质检没有发现问题时写的文字
QC_OK = "合格"

# 时间的写法，如 2026-10-07 09:30:00（这样写的时间可以直接按文字比较先后）
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _now() -> str:
    return datetime.now().strftime(TIME_FORMAT)


def read_csv_rows(path) -> list[dict]:
    """读数据池里的一个表格，每行一个字典（键是表头）。文件不存在时返回空列表。"""
    path = Path(path)
    if not path.is_file():
        return []
    # 老师用 Excel 打开后按"CSV（逗号分隔）"另存，中文 Windows 会存成 GBK 编码，所以读不了时再按 GBK 读一次
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with open(path, encoding=encoding, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"表格的文字编码认不出来：{path}。请用 Excel 另存为\"CSV UTF-8（逗号分隔）\"后再试。")


def write_csv_rows(path, columns: list[str], rows: list[dict]) -> None:
    """把整张表格写回文件（UTF-8 带 BOM）。表格正用 Excel 打开着时，Windows 会报 PermissionError。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _replace_row(path, columns: list[str], row: dict) -> None:
    """表格里每段录音只留一行：先去掉同一个文件编号的旧行，再把新行加在最后。"""
    rows = [old for old in read_csv_rows(path) if old.get("文件编号") != row["文件编号"]]
    rows.append(row)
    write_csv_rows(path, columns, rows)


def _can_append(path: Path, columns: list[str]) -> bool:
    """这个表格能不能直接在末尾追加一行：要是 UTF-8 编码、表头一样、最后一行后面有换行。

    老师用 Excel 另存过（可能存成了 GBK 编码）或手工改过的表格，直接追加会乱码或接错行，就不能追加。
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return False
    header = next(csv.reader(text.splitlines()), [])
    return header == columns and text.endswith("\n")


def append_csv_row(path, columns: list[str], row: dict) -> None:
    """在表格末尾追加一行（UTF-8 带 BOM），不重写整张表。

    为什么要"追加"：几台电脑可能同时往共享文件夹里的同一张表写（如几个同学同时在"数据校对"页保存）。
    "整张读出来、加一行、再整张写回去"时，别人在这中间写进去的行会被冲掉；只在末尾追加就不会。
    文件不存在或是空的：先写表头。不能直接追加时（见 _can_append）：只好整张重写，统一存回 UTF-8 带 BOM。
    """
    path = Path(path)
    is_empty = not path.is_file() or path.stat().st_size == 0
    if not is_empty and not _can_append(path, columns):
        write_csv_rows(path, columns, read_csv_rows(path) + [row])
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # 用 "a"（追加）模式打开：文件不是空的时，Python 不会在中间再写一个 BOM
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="", extrasaction="ignore")
        if is_empty:
            writer.writeheader()
        writer.writerow(row)


# ======================== 入池 ========================

# raw/ 里这些系统自动生成的文件不是录音，直接忽略（不报错）
_SYSTEM_FILES = {"thumbs.db", "desktop.ini", ".ds_store"}


def _is_system_file(path: Path) -> bool:
    """Windows、苹果电脑、Office 自动生成的隐藏文件（Thumbs.db、.DS_Store、~$ 开头的临时文件）。"""
    name = path.name.lower()
    return name in _SYSTEM_FILES or name.startswith(".") or name.startswith("~$")


def _check_raw_files(raw: Path, errors: list[dict]) -> dict[str, list[Path]]:
    """检查 raw/ 里每个文件的名字和格式，合格的按文件编号归类；不合格的记进 errors。"""
    by_stem: dict[str, list[Path]] = {}
    for path in sorted(raw.iterdir()):
        if _is_system_file(path):
            continue
        if path.is_dir():
            errors.append({"file": path.name, "reason": "这是一个文件夹。入池工具只处理直接放在 raw 文件夹里的录音，"
                                                        "请把里面的录音拿出来放到 raw 文件夹下"})
            continue
        try:
            info = parse_recording_name(path.name)
        except ValueError as e:
            errors.append({"file": path.name, "reason": str(e)})
            continue
        if info["ext"] not in SUPPORTED_EXTS:
            errors.append({"file": path.name, "reason": (
                f"不支持这种文件格式（{info['ext']}）。支持的格式：{'、'.join(sorted(SUPPORTED_EXTS))}。"
                "微信语音请先导出为 m4a 或 mp3；最好直接用手机自带的录音机录")})
            continue
        by_stem.setdefault(info["stem"], []).append(path)
    return by_stem


def _convert_to_pool_wav(src: Path, wav: Path):
    """把原始录音转成 normalized/<文件编号>.wav，返回读出来的采样（float32 数组）。

    先转到临时文件 <文件编号>.tmp.wav，转换和读取都成功了，才把它改名成正式的文件名。
    这样复录时如果新文件转到一半失败，旧的 normalized/<文件编号>.wav 还完好（清单里记的也还是它）。
    """
    tmp = wav.with_name(f"{wav.stem}.tmp.wav")
    try:
        convert_to_wav(src, tmp)
        samples = read_wav(tmp)
        os.replace(tmp, wav)  # 改名；正式文件已经存在时直接替换（Windows 上也可以）
    finally:
        if tmp.exists():  # 出错时删掉转了一半的临时文件
            tmp.unlink()
    return samples


def _ingest_one(path: Path, stem: str, sha: str, paths: dict, cfg: dict) -> dict:
    """处理一个新文件：转格式、质检、写质检报告、原始文件设只读、删旧缓存，最后写清单。返回入池结果。

    为什么清单放在最后写：manifest.csv 里有这一行，就表示"这个文件已经入池"，下次运行会跳过它。
    如果中途出错（比如 qc_report.csv 正用 Excel 打开着，写不进去），清单里还没有这一行，
    关掉 Excel 再运行一次，这个文件会从头重新处理，质检报告和只读设置都不会漏掉。
    """
    plan = _plan_by_stem().get(stem)
    expected = round(plan["expected_minutes"] * 60, 1) if plan else None
    name = parse_recording_name(path.name)

    # 1. 读原始采样率，转成 16000 Hz 单声道 16 位 WAV，质检（只报告、不修改）
    info = probe(path)
    samples = _convert_to_pool_wav(path, paths["normalized"] / f"{stem}.wav")
    duration = round(len(samples) / SR, 2)
    problems = quality_check(samples, info["sample_rate"], cfg, expected)
    qc_text = "；".join(problems) if problems else QC_OK
    now = _now()

    # 2. 写质检报告（复录或上次中途出错时，替换这段录音的旧行）
    _replace_row(paths["qc_report"], QC_COLUMNS, {
        "文件编号": stem,
        "质检时间": now,
        "时长（秒）": duration,
        "预计时长（秒）": expected if expected is not None else "",
        "原始采样率": info["sample_rate"],
        "原始声道数": info["channels"],
        "问题数": len(problems),
        "质检结果": qc_text,
    })

    # 3. 原始文件设为只读
    _set_read_only(path)

    # 4. 删掉这段录音以前的识别结果缓存（复录时旧缓存已经不对了），下次"数据校对"时重新识别
    for cache in paths["asr_cache"].glob(f"{stem}.*"):
        cache.unlink()

    # 5. 最后写清单：写进去了才算入池
    _replace_row(paths["manifest"], MANIFEST_COLUMNS, {
        "文件编号": stem,
        "剧本编号": name["script_id"],
        "录音条件": name["condition_name"],
        "原始文件": f"raw/{path.name}",
        "转换后文件": f"normalized/{stem}.wav",
        "时长（秒）": duration,
        "原始采样率": info["sample_rate"],
        "质检结果": qc_text,
        "上传时间": now,
        "SHA-256": sha,
    })
    return {"file": path.name, "stem": stem, "duration": duration, "qc": problems}


def ingest_pool(root, cfg: dict) -> dict:
    """入池：处理 raw/ 里还没入池的录音。可以放心重复运行，已入池的文件会跳过。

    root：数据池文件夹；cfg：完整配置（load_config() 的结果），质检阈值在 cfg["qc"] 里。
    返回 {"added": [...], "skipped": [...], "errors": [...]}：
      added    新入池的文件，每个是 {"file", "stem", "duration", "qc"（问题列表，空表示合格）,
               "replaced"（True 表示复录：这个编号以前入过池，这次换了新文件）}
      skipped  已经入池的文件名（按 SHA-256 指纹判断，内容没变）
      errors   没处理的文件，每个是 {"file": 文件名, "reason": 中文原因}
               （文件名不合格、格式不支持、同一段录音交了两个文件、和别的录音内容完全一样、文件损坏）
    """
    init_pool(root)
    paths = pool_paths(root)
    result = {"added": [], "skipped": [], "errors": []}

    by_stem = _check_raw_files(paths["raw"], result["errors"])
    if not by_stem:
        return result
    find_ffmpeg()  # 先确认找得到 ffmpeg；找不到时直接报错（带中文提示），不必每个文件都报一遍

    manifest = read_csv_rows(paths["manifest"])
    known = {row.get("SHA-256"): row.get("文件编号") for row in manifest}  # 已入池的指纹 → 文件编号
    old_stems = {row.get("文件编号") for row in manifest}

    for stem in sorted(by_stem):
        files = by_stem[stem]
        if len(files) > 1:
            names = "、".join(p.name for p in files)
            for path in files:
                result["errors"].append({"file": path.name, "reason": (
                    f"raw 文件夹里有 {len(files)} 个文件都是录音 {stem}（{names}），一段录音只能交一个文件。"
                    "如果是复录，请把旧文件移出 raw 文件夹后再运行")})
            continue

        path = files[0]
        sha = sha256_file(path)
        if sha in known:
            if known[sha] == stem:
                if os.access(path, os.W_OK):  # 有人去掉了"只读"（或旧版本工具没来得及设）：补设上
                    _set_read_only(path)
                result["skipped"].append(path.name)
            else:
                result["errors"].append({"file": path.name, "reason": (
                    f"内容和已入池的录音 {known[sha]} 完全一样（SHA-256 指纹相同），可能是同一个录音交了两次"
                    "或改错了名字，请核对后重新命名或移出 raw 文件夹")})
            continue

        try:
            item = _ingest_one(path, stem, sha, paths, cfg)
        except (RuntimeError, ValueError) as e:  # ffmpeg 读不了、转换失败等：记下来，接着处理下一个文件
            last_line = str(e).strip().splitlines()[-1] if str(e).strip() else ""
            result["errors"].append({"file": path.name, "reason": (
                f"转换失败，文件可能已损坏或不是录音文件，请重新从手机拷贝后再放进 raw 文件夹（{last_line}）")})
            continue

        item["replaced"] = stem in old_stems
        known[sha] = stem
        result["added"].append(item)
    return result


# ======================== 参考文本 ========================


def export_references(root, overwrite: bool = False) -> list[Path]:
    """为录音计划里的 72 段录音各写一份参考文本初稿 references/<文件编号>.txt（内容照抄剧本台词）。

    已经存在的文件默认不覆盖（可能已经校对过）；overwrite=True 时全部按剧本重新写（校对结果会丢失）。
    返回这次写了的文件路径。
    """
    init_pool(root)
    folder = pool_paths(root)["references"]
    written = []
    for stem, item in _plan_by_stem().items():
        path = folder / f"{stem}.txt"
        if path.exists() and not overwrite:
            continue
        path.write_text(script_reference_text(item["script_id"]), encoding="utf-8")
        written.append(path)
    return written


# ======================== 校对记录 ========================


def log_proofread(root, stem, proofreader, note="") -> None:
    """记一笔校对：proofread_log.csv 追加一行（文件编号、校对人、校对时间、第几遍、备注）。

    "第几遍"按不同的校对人数：第一个校对这段录音的人是第 1 遍，第二个不同的人是第 2 遍；
    同一个人改了再保存，仍记作他的那一遍。验收时要有 2 个不同的人校对过才算通过。
    """
    if stem not in _plan_by_stem():
        raise ValueError(f"没有这个录音编号：{stem}。编号形如 G1-S1-Q（组号-剧本号-条件代码）。")
    proofreader = str(proofreader).strip()
    if not proofreader:
        raise ValueError("请填写校对人（学号后四位）：验收时要数有几个不同的人校对过。")

    path = pool_paths(root)["proofread_log"]
    people = []  # 按先后顺序，校对过这段录音的人（读已有的记录只是为了算"第几遍"）
    for row in read_csv_rows(path):
        if row.get("文件编号") == stem and row.get("校对人") not in people:
            people.append(row.get("校对人"))
    if proofreader not in people:
        people.append(proofreader)
    # 追加一行，不重写整张表：几个同学同时在共享文件夹里保存，也不会冲掉别人的记录
    append_csv_row(path, PROOFREAD_COLUMNS, {
        "文件编号": stem,
        "校对人": proofreader,
        "校对时间": _now(),
        "第几遍": people.index(proofreader) + 1,
        "备注": str(note).strip(),
    })
