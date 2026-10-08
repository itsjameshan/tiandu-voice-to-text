"""界面表格 ↔ 段落：把段落列表变成界面上可编辑的表格，再把人工改过的表格变回段落。

界面"整理录音"标签页里的表格有 9 列（TABLE_HEADERS）：
    序号、开始、结束、说话人、文字、标签、数字、复核结论、复核意见
工作人员可以在表格里改说话人、文字、标签、复核结论、复核意见；点"导出"时，
rows_to_segments 用表格里的内容覆盖对应的段落，导出的 Word、CSV、JSON 用的就是改过的内容。

基线做法：
    - segments_to_rows：每个段落一行；数字列表用"；"连起来显示；没有复核结论的显示"未复核"。
    - rows_to_segments：按"序号"列找到对应的段落（表格被排序过也不会对错），覆盖 5 个可改的列：
        说话人：空着就写"未知"；
        文字：原样用；
        标签：只能是 5 个"疑似·…"之一或空。"·"在键盘上不好打，所以"费用""疑似费用""疑似.费用"
              "威胁消费""消费施压"这些常见写法都认，统一改成标准写法（如"疑似·消费施压"）；
              写成"正常讲解""其他"就是不标（标签为空）。认不出来的写法：标签清空，
              并在复核意见后面加一句提示，请人重新选；
              标签改了，类别（category）跟着改（"疑似·消费施压"对应类别"威胁消费"）；
        复核结论：只能是"未复核、确认、修改、驳回"之一，其他写法一律改成"未复核"；
        复核意见：原样用，空着就去掉这一项。
      开始、结束、数字三列只是给人看的，改了也不会写回段落。
可改进方向：
    这是界面用的工具，学生一般不改。要在表格里多显示一列，就同时改 TABLE_HEADERS、
    segments_to_rows 和 rows_to_segments，并补上测试。
测评指标：
    不涉及识别效果；由 tests/test_table.py 检查"改了再导出，导出的是改过的内容"、标签和复核结论都合法。
"""
import re

from pipeline.data import DISPLAY_NAMES, FLAG_LABELS, FLAG_OUTPUTS, LABEL_NAMES, label_output
from pipeline.schema import LIST_SEP, REVIEW_CHOICES

TABLE_HEADERS = ["序号", "开始", "结束", "说话人", "文字", "标签", "数字", "复核结论", "复核意见"]

# 每一列在一行里的位置
COL_NO, COL_START, COL_END, COL_SPEAKER, COL_TEXT, COL_LABEL, COL_NUMBERS, COL_REVIEW, COL_NOTE = range(9)

# 标签写得不对时，加在复核意见后面的提示（不把写错的词抄进来，免得导出文件里出现不该有的说法）
BAD_LABEL_HINT = f"系统提示：表格里填写的标签不是可用的标签，已清空，请重新选择（{'、'.join(FLAG_OUTPUTS)}，或留空）"

# 比较标签写法时忽略的字符：空白和各种点、冒号、横线
_IGNORED = re.compile(r"[\s·・.．。:：、\-－—_]")


def _squash(text: str) -> str:
    """去掉空白和分隔符号，例如 " 疑似.费用 " → "疑似费用"。"""
    return _IGNORED.sub("", text)


def _build_label_lookup() -> dict[str, tuple[str, str]]:
    """标签的各种写法 → (标准标签, 类别)。

    每个类别认 4 种写法：类别名、显示名、"疑似"+类别名、"疑似"+显示名（去掉分隔符号后比较）。
    例："消费施压""威胁消费""疑似·消费施压""疑似威胁消费" → ("疑似·消费施压", "威胁消费")；
        "正常讲解" → ("", "正常讲解")。
    """
    lookup = {}
    for category in LABEL_NAMES:
        label = label_output(category)
        display = DISPLAY_NAMES[category]
        for name in (category, display, "疑似" + category, "疑似" + display):
            lookup[_squash(name)] = (label, category)
    return lookup


_LABEL_LOOKUP = _build_label_lookup()


def _cell_text(value) -> str:
    """表格格子里的值 → 去掉两头空白的文字。空格子可能是 None 或 NaN（pandas 表格），都当成空。"""
    if value is None or (isinstance(value, float) and value != value):  # NaN 不等于它自己
        return ""
    return str(value).strip()


def _as_rows(rows) -> list[list]:
    """界面交来的表格可能是 pandas 表格、numpy 数组或列表的列表，统一成列表的列表。"""
    if hasattr(rows, "columns") and hasattr(rows, "values"):  # pandas 表格
        return rows.values.tolist()
    if hasattr(rows, "tolist"):  # numpy 数组
        return rows.tolist()
    return [list(row) for row in rows]


def segments_to_rows(segments: list[dict]) -> list[list]:
    """段落列表 → 表格行（每行 9 列，顺序同 TABLE_HEADERS）。序号从 1 开始。"""
    rows = []
    for n, segment in enumerate(segments, start=1):
        numbers = segment.get("numbers") or []
        rows.append([
            n,
            segment["start"],
            segment["end"],
            segment.get("speaker", "未知"),
            segment.get("text", ""),
            segment.get("label", ""),
            numbers if isinstance(numbers, str) else LIST_SEP.join(str(v) for v in numbers),
            segment.get("review") or REVIEW_CHOICES[0],
            segment.get("review_note", ""),
        ])
    return rows


def _row_index(row: list, count: int) -> int | None:
    """这一行对应第几个段落（从 0 开始）；序号空着或超出范围（例如界面里新加的空行）返回 None。"""
    try:
        index = int(float(_cell_text(row[COL_NO]))) - 1
    except (ValueError, OverflowError, IndexError):
        return None
    return index if 0 <= index < count else None


def _apply_label(segment: dict, typed: str) -> bool:
    """把表格里填的标签写进段落，类别跟着改。认不出来时清空标签并返回 False。"""
    found = _LABEL_LOOKUP.get(_squash(typed)) if typed else None
    if found is None:
        segment["label"] = ""
        if segment.get("category") in FLAG_LABELS:
            segment.pop("category")  # 标签清空了，类别不能还写着某个疑似类（前后矛盾）
        return not typed  # 空着是合法的（不标），写了认不出的词才不合法
    label, category = found
    if "category" in segment or label != segment.get("label", ""):
        segment["category"] = category
    segment["label"] = label
    return True


def rows_to_segments(rows, segments: list[dict]) -> list[dict]:
    """用表格里改过的说话人、文字、标签、复核结论、复核意见覆盖对应的段落，返回新的段落列表。

    rows：表格行（列表的列表，或 pandas 表格），列的顺序同 TABLE_HEADERS；按"序号"列对应段落。
    segments：生成这个表格时用的段落列表（不会被修改）。表格里没有的段落原样保留。
    """
    result = [dict(segment) for segment in segments]
    for row in _as_rows(rows):
        row = list(row) + [""] * (len(TABLE_HEADERS) - len(row))  # 行太短时补空格子
        index = _row_index(row, len(result))
        if index is None:
            continue
        segment = result[index]

        segment["speaker"] = _cell_text(row[COL_SPEAKER]) or "未知"
        segment["text"] = _cell_text(row[COL_TEXT])

        review = _cell_text(row[COL_REVIEW])
        segment["review"] = review if review in REVIEW_CHOICES else REVIEW_CHOICES[0]

        note = _cell_text(row[COL_NOTE])
        if not _apply_label(segment, _cell_text(row[COL_LABEL])):
            note = f"{note}；{BAD_LABEL_HINT}" if note else BAD_LABEL_HINT
        if note:
            segment["review_note"] = note
        else:
            segment.pop("review_note", None)
    return result
