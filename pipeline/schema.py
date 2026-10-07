"""统一中间格式：段落的字段、中文表头，以及 JSON / CSV 的读写。

八个步骤之间传递的都是"段落列表"：每个段落是一个字典，一段话一条。
五个必有字段（含义和顺序都不能改）：
    start   开始时间（秒），保留 2 位小数
    end     结束时间（秒）
    speaker 说话人（"导游""游客""未知"，映射之前是"说话人1""说话人2"）
    text    文字内容
    label   标签（"疑似·费用"等 5 类之一；正常讲解和其他留空字符串）
可选字段见 OPTIONAL_FIELDS，可以有也可以没有。

基线做法：
    用标准库 json 和 csv 读写。JSON 里放 {"meta": …, "segments": […]}，meta 一定带通知语 NOTICE；
    CSV 用 UTF-8 带 BOM（Excel 双击打开不乱码），表头用中文，前 5 列顺序固定，可选列只写出现过的。
可改进方向：
    这是全班共用的格式，学生不改。要加新信息就加新的可选字段，不能改 5 个必有字段的含义。
测评指标：
    往返一致：写出再读回，内容与写入时完全相同（见 tests/test_schema.py）。
"""
import csv
import json
from pathlib import Path, PurePath

# 五个必有字段，顺序固定
FIELDS = ["start", "end", "speaker", "text", "label"]

# 代码里的键 → 导出文件里的中文表头（前 5 个是必有字段，后面是可选字段）
HEADERS = {
    "start": "开始时间（秒）",
    "end": "结束时间（秒）",
    "speaker": "说话人",
    "text": "文字内容",
    "label": "标签",
    "text_raw": "识别原文（汉字读法）",
    "speaker_id": "说话人编号",
    "category": "类别",
    "score": "置信度",
    "numbers": "数字",
    "entities": "名称",
    "corrections": "热词纠错",
    "review": "复核结论",
    "review_note": "复核意见",
    "clip": "片段文件",
    "source": "来源",
}

# 可选字段：HEADERS 里除前 5 个以外的键，顺序同上
OPTIONAL_FIELDS = [key for key in HEADERS if key not in FIELDS]

# 界面、初稿、导出文件里都要出现的通知语
NOTICE = "识别可能有误；所有标注均为疑似、待核查，必须人工复核"

# 写进 JSON 的 meta，标明这是机器生成的（红线第 6 条）
GENERATED_BY = "由人工智能技术自动生成（旅游纠纷录音材料整理工具·教学原型）"

# 人工复核结论只有这四种
REVIEW_CHOICES = ["未复核", "确认", "修改", "驳回"]

# CSV 里的分隔写法
LIST_SEP = "；"  # 列表（数字、名称、热词纠错）里多个值之间用全角分号
ARROW = "→"  # 热词纠错写成"错→对"
LIST_FIELDS = ["numbers", "entities"]  # 这些字段是字符串列表
FLOAT_FIELDS = ["start", "end", "score"]  # 这些列读回来是小数
INT_FIELDS = ["speaker_id"]  # 这些列读回来是整数


def new_segment(start: float, end: float, **extra) -> dict:
    """新建一个段落。时间保留 2 位小数；没给的字段用默认值。

    例：new_segment(1.234, 2.0, speaker="导游")
        → {"start": 1.23, "end": 2.0, "speaker": "导游", "text": "", "label": "", "review": "未复核"}
    """
    segment = {
        "start": round(float(start), 2),
        "end": round(float(end), 2),
        "speaker": "未知",
        "text": "",
        "label": "",
        "review": "未复核",
    }
    segment.update(extra)  # 额外传进来的字段（可以覆盖上面的默认值）
    return segment


# ---------------- JSON ----------------

def _json_default(value):
    """json 不认识的类型怎么写：numpy 的数字和数组转成普通数字和列表，路径转成文字，其余报错。"""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, PurePath):
        return str(value)
    raise TypeError(f"不能写进 JSON 的类型：{type(value).__name__}")


def write_json(path, segments: list[dict], meta: dict) -> None:
    """把段落和元信息写成 JSON 文件（UTF-8，中文直接写出）。meta 里没有通知语时自动补上。"""
    meta = dict(meta or {})  # 复制一份，不改调用方传进来的字典
    meta.setdefault("notice", NOTICE)
    meta.setdefault("generated_by", GENERATED_BY)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"meta": meta, "segments": segments}
    text = json.dumps(data, ensure_ascii=False, indent=2, default=_json_default)
    path.write_text(text, encoding="utf-8")


def read_json(path) -> tuple[list[dict], dict]:
    """读回 write_json 写的文件，返回 (段落列表, 元信息)。元信息缺通知语时补上。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    meta = data.get("meta", {})
    meta.setdefault("notice", NOTICE)
    return segments, meta


# ---------------- CSV ----------------

def _cell_text(key: str, value) -> str:
    """把一个字段的值变成 CSV 单元格里的文字。"""
    if value is None:
        return ""
    if isinstance(value, str):
        # 已经是连接好的文字（例如界面表格里编辑过的"2800元；15:40"），原样写出
        return value
    if key in LIST_FIELDS:
        return LIST_SEP.join(str(v) for v in value)
    if key == "corrections":
        # [{"from": "雾影行走", "to": "雾隐行舟"}] → "雾影行走→雾隐行舟"
        return LIST_SEP.join(f"{c['from']}{ARROW}{c['to']}" for c in value)
    return str(value)


def _cell_value(key: str, text: str, row_no: int):
    """把 CSV 单元格里的文字变回字段的值（_cell_text 的反过程）。"""
    if key in LIST_FIELDS:
        return [v.strip() for v in text.split(LIST_SEP) if v.strip()]
    if key == "corrections":
        result = []
        for item in text.split(LIST_SEP):
            if ARROW in item:
                wrong, right = item.split(ARROW, 1)
                result.append({"from": wrong.strip(), "to": right.strip()})
        return result
    if key in FLOAT_FIELDS or key in INT_FIELDS:
        try:
            number = float(text)
        except ValueError:
            raise ValueError(f"CSV 第 {row_no} 行的“{HEADERS[key]}”不是数字：{text!r}") from None
        if key in INT_FIELDS:
            return int(number)
        return number
    return text


def write_csv(path, segments: list[dict]) -> None:
    """把段落写成 CSV（UTF-8 带 BOM、中文表头）。

    - 前 5 列固定是 开始时间、结束时间、说话人、文字内容、标签；
    - 可选列只写至少一个段落里出现过的，按 HEADERS 的顺序排在后面；
    - 不在 HEADERS 里的键（例如剧本演示用的 gold_label）不写出；
    - 数字、名称用"；"连接；热词纠错写成"错→对"再用"；"连接
      （纠错记录里的其他信息，如 start，不写进 CSV；要完整保存请用 JSON）。
    """
    keys = list(FIELDS)
    for key in OPTIONAL_FIELDS:
        if any(key in seg for seg in segments):
            keys.append(key)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([HEADERS[key] for key in keys])
        for seg in segments:
            writer.writerow([_cell_text(key, seg.get(key)) for key in keys])


def read_csv(path) -> list[dict]:
    """读回 write_csv 写的文件（也接受用 Excel 改过、另存为 CSV UTF-8 的文件）。

    - 表头可以是中文表头，也可以是英文键名；不认识的列忽略；
    - 必须有前 5 列，缺了会报错并说明缺哪一列；
    - 开始时间、结束时间、置信度读回小数，说话人编号读回整数；
    - 可选列里空着的单元格，读回时这个段落就没有这个字段。
    """
    header_to_key = {header: key for key, header in HEADERS.items()}
    segments = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header_row = next(reader, [])
        # 每一列对应哪个键（不认识的列为 None）
        column_keys = []
        for header in header_row:
            header = header.strip()
            if header in header_to_key:
                column_keys.append(header_to_key[header])
            elif header in HEADERS:
                column_keys.append(header)
            else:
                column_keys.append(None)
        missing = [HEADERS[key] for key in FIELDS if key not in column_keys]
        if missing:
            raise ValueError(f"CSV 缺少必有的列：{'、'.join(missing)}")

        for row_no, row in enumerate(reader, start=2):  # 第 1 行是表头
            if not any(cell.strip() for cell in row):
                continue  # 跳过空行
            row = row + [""] * (len(column_keys) - len(row))  # 行比表头短时补上空单元格
            seg = {}
            for key, cell in zip(column_keys, row):
                if key is None:
                    continue
                if key not in FIELDS and cell == "":
                    continue  # 可选字段空着就不放进段落
                seg[key] = _cell_value(key, cell, row_no)
            # 按 HEADERS 的顺序排好键，看起来和写入时一样
            segments.append({key: seg[key] for key in HEADERS if key in seg})
    return segments
