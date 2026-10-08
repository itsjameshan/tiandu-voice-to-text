"""标注表格：把 Audacity 导出的标签转成表格，再读回来给验收和测评用（教师模板，学生不改）。

有两种人工标注（怎么用 Audacity 标，见 docs/guides/annotation.md）：
    说话人时间  数据池/annotations/speakers/G3-S1-Q.csv，三列 start,end,speaker（开始秒、结束秒、角色名）
    疑似片段    数据池/annotations/clips/G8-S1-Q.csv，  三列 start,end,label  （开始秒、结束秒、类别名）

Audacity"导出标签"得到一个文本文件，每个标签一行，三项之间用制表符（Tab）隔开，例如：
    1.500000	4.250000	导游
在语谱图上框选过的标签，下面还会多一行以反斜杠"\\"开头的频率范围，例如"\\	200.000000	3000.000000"，
这种行不是标签，读的时候跳过。

基线做法：
    - read_audacity_labels：一行一行读，按制表符拆成 开始、结束、标签；跳过空行和以"\\"开头的频率行；
      去掉标签两边的空格。文字编码先按 UTF-8（带不带 BOM 都行）读，读不了再按 GBK 读
      （中文 Windows 上的 Audacity 可能按 ANSI，也就是 GBK 写文件）；记事本另存的"Unicode"（UTF-16）也认。
    - write_turns_csv / write_clips_csv：先逐个检查（要拖选一段、起止不能相同、不能是负数、要写标签；
      片段的标签只能是 5 个疑似类别），有错就一条也不写、把所有错误一起报出来；没错就按开始时间排好，
      时间保留 3 位小数（毫秒），用 UTF-8 带 BOM 写（老师用 Excel 直接打开不乱码）。
    - read_turns_csv / read_clips_csv：读回来，做同样的检查，返回 [(开始, 结束, 标签), ...]。
    - annotated_seconds：标注覆盖了多少秒。把各段时间合并起来算：两人同时说话（标签重叠）的部分只算一次，
      没标的停顿不算。验收表里的"说话人标注（秒）"就是它。
可改进方向：
    本模块是教师模板，学生不改。第 3 组、第 8 组定标注规范时可以讨论：多短的停顿算同一段话、
    片段从哪一句开始算，这些写进标注指南，而不是改代码。
测评指标：
    不涉及识别效果；由 tests/test_annotations.py 检查读 Audacity 标签（含频率行、各种编码）、
    表格往返、错误提示和覆盖时长的算法。

类别名：数据里一律写 data/labels.json 里的类别名。"威胁消费"在界面上显示为"消费施压"，
所以片段标注里写"消费施压"也认，保存时改成"威胁消费"。
"""
import csv
import io
import math
from pathlib import Path

from pipeline.data import DISPLAY_NAMES, FLAG_LABELS

# 说话人时间表格的列：开始秒、结束秒、角色名
TURN_COLUMNS = ["start", "end", "speaker"]

# 疑似片段表格的列：开始秒、结束秒、类别名
CLIP_COLUMNS = ["start", "end", "label"]

# 片段标注能用的类别：会被标成"疑似·…"的 5 个类别（购物安排、费用、行程变更、服务态度、威胁消费）
CLIP_LABELS = list(FLAG_LABELS)

# 显示名 → 类别名（只有"消费施压"→"威胁消费"这一个不一样）
_DISPLAY_TO_NAME = {display: name for name, display in DISPLAY_NAMES.items() if name in CLIP_LABELS}

# 片段标注的标签写错时，告诉同学能写什么
CLIP_LABEL_RULE = (f"片段标注的标签只能写这 5 个类别名之一：{'、'.join(CLIP_LABELS)}"
                   "（\"威胁消费\"也可以写成\"消费施压\"）；正常讲解和其他不用标")


# ======================== 读文本 ========================


def _read_text(path) -> str:
    """读一个文本文件。依次试 UTF-16（记事本另存为"Unicode"时）、UTF-8（带不带 BOM 都行）、GBK（见 pipeline.textio）。"""
    from pipeline.textio import read_text

    return read_text(path, hint="请在 Audacity 里重新导出标签，或用记事本另存为 UTF-8 后再试。")


def _to_seconds(text: str, where: str) -> float:
    """把"1.500000"这样的文字转成秒数（小数）。转不了时抛 ValueError，说明是哪里的哪个值。"""
    value = text.strip()
    try:
        # 有的电脑把小数点写成逗号（如 1,5），也认
        seconds = float(value.replace(",", ".")) if value else None
    except ValueError:
        seconds = None
    if seconds is None or not math.isfinite(seconds):
        raise ValueError(f"{where}：时间\"{value}\"不是数字（应该是秒数，如 1.5）")
    return seconds


# ======================== Audacity 标签 ========================


def read_audacity_labels(path) -> list[tuple[float, float, str]]:
    """读 Audacity"导出标签"得到的文本文件，返回 [(开始秒, 结束秒, 标签), ...]（按文件里的先后顺序）。

    跳过空行和以"\\"开头的频率范围行；标签两边的空格去掉；没写标签时标签是空字符串。
    某一行读不懂时抛 ValueError，说明是第几行。文件不存在时抛 FileNotFoundError。
    """
    path = Path(path)
    labels = []
    for number, line in enumerate(_read_text(path).splitlines(), start=1):
        if not line.strip() or line.startswith("\\"):
            continue
        # 正常是用制表符隔开的；手工改过、用空格隔开的也认（最多拆成 3 部分，标签里可以有空格）
        parts = line.split("\t", 2) if "\t" in line else line.split(None, 2)
        where = f"{path.name} 第 {number} 行"
        if len(parts) < 2:
            raise ValueError(f"{where}：应该是\"开始时间<Tab>结束时间<Tab>标签\"，实际是\"{line.strip()}\"。"
                             "请用 Audacity 的\"文件 → 导出 → 导出标签\"重新导出")
        start = _to_seconds(parts[0], where)
        end = _to_seconds(parts[1], where)
        label = parts[2].strip() if len(parts) > 2 else ""
        labels.append((start, end, label))
    return labels


# ======================== 检查 ========================


def clip_label(name: str) -> str:
    """检查片段标注的标签，返回数据里用的类别名。

    只能是 5 个疑似类别之一：购物安排、费用、行程变更、服务态度、威胁消费；
    写显示名"消费施压"也认，返回"威胁消费"。其他（正常讲解、其他、角色名、"疑似·费用"……）抛 ValueError。
    """
    name = str(name).strip()
    name = _DISPLAY_TO_NAME.get(name, name)
    if name not in CLIP_LABELS:
        raise ValueError(f"标签\"{name}\"不对。{CLIP_LABEL_RULE}")
    return name


def _check_rows(rows, what: str, clips: bool = False, title: str = "标注有问题，没有保存") -> list:
    """逐个检查标注，全部合格时返回按开始时间排好的 [(开始, 结束, 标签), ...]；有错时把所有错误一起报出来。

    what：标签写的是什么（"角色名"或"类别名"），用在错误提示里。
    clips：True 表示疑似片段标注，标签要是 5 个类别名之一（"消费施压"改成"威胁消费"）；
           False 表示说话人标注，标签不是空的就行。
    title：报错时第一行写什么。
    时间先保留 3 位小数（毫秒）再检查，这样写进表格再读回来，结果不会变。
    """
    checked, problems = [], []
    wrong_label = False
    for number, row in enumerate(rows, start=1):
        start, end, label = round(float(row[0]), 3), round(float(row[1]), 3), str(row[2]).strip()
        where = f"第 {number} 个标签（{start:.3f}—{end:.3f} 秒，\"{label}\"）"
        if not (math.isfinite(start) and math.isfinite(end)):
            problems.append(f"{where}：时间不是数字")
        elif start < 0 or end < 0:
            problems.append(f"{where}：时间是负数")
        elif end == start:
            problems.append(f"{where}：起止时间相同，是一个\"点\"标签。请在 Audacity 里先拖选一段，再按 Ctrl+B 加标签")
        elif end < start:
            problems.append(f"{where}：结束时间早于开始时间")
        if not label:
            problems.append(f"{where}：没有写{what}")
        elif clips:
            label = _DISPLAY_TO_NAME.get(label, label)
            if label not in CLIP_LABELS:
                problems.append(f"{where}：{what}不对")
                wrong_label = True
        checked.append((start, end, label))
    if wrong_label:  # 能写哪几个类别名，最后说一次就够了
        problems.append(CLIP_LABEL_RULE)
    if problems:
        raise ValueError(f"{title}：\n" + "\n".join(problems))
    return sorted(checked)


# ======================== 表格读写 ========================


def _write_rows(path, columns: list[str], rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for start, end, label in rows:
            writer.writerow([f"{start:.3f}", f"{end:.3f}", label])


def _read_rows(path, columns: list[str]) -> list[tuple]:
    """读一个三列的标注表格，返回 [(开始秒, 结束秒, 标签), ...]（还没逐个检查）。

    Excel 另存过的表格（GBK 编码、空行）也能读。缺列、时间不是数字时抛 ValueError，说明是哪个文件的第几行。
    """
    path = Path(path)
    reader = csv.DictReader(io.StringIO(_read_text(path), newline=""))
    missing = [col for col in columns if col not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"标注表格 {path} 缺少列：{'、'.join(missing)}。第一行应该是 {','.join(columns)}")
    rows = []
    for row in reader:
        if not any(value.strip() for value in row.values() if isinstance(value, str)):
            continue  # 空行（Excel 另存时可能留下",,"这样的行）
        where = f"标注表格 {path} 第 {reader.line_num} 行"  # line_num：读到了文件的第几行
        rows.append((_to_seconds(row[columns[0]] or "", where), _to_seconds(row[columns[1]] or "", where),
                     row[columns[2]] or ""))
    return rows


def _checked_from_file(path, columns, what, clips=False):
    """读一个标注表格并逐个检查（和写的时候检查的一样）。"""
    return _check_rows(_read_rows(path, columns), what, clips, title=f"标注表格 {path} 有问题")


def write_turns_csv(path, rows) -> None:
    """写说话人时间表格（列 start,end,speaker）。rows 是 [(开始秒, 结束秒, 角色名), ...]。

    有不合格的标注时抛 ValueError（说明第几个、什么问题），一条也不写。
    """
    _write_rows(path, TURN_COLUMNS, _check_rows(rows, "角色名"))


def read_turns_csv(path) -> list[tuple[float, float, str]]:
    """读说话人时间表格，返回按开始时间排好的 [(开始秒, 结束秒, 角色名), ...]。"""
    return _checked_from_file(path, TURN_COLUMNS, "角色名")


def write_clips_csv(path, rows) -> None:
    """写疑似片段表格（列 start,end,label）。rows 是 [(开始秒, 结束秒, 类别名), ...]。

    类别名只能是 5 个疑似类别（"消费施压"会存成"威胁消费"）；有不合格的标注时抛 ValueError，一条也不写。
    """
    _write_rows(path, CLIP_COLUMNS, _check_rows(rows, "类别名", clips=True))


def read_clips_csv(path) -> list[tuple[float, float, str]]:
    """读疑似片段表格，返回按开始时间排好的 [(开始秒, 结束秒, 类别名), ...]。"""
    return _checked_from_file(path, CLIP_COLUMNS, "类别名", clips=True)


# ======================== 统计 ========================


def annotated_seconds(rows) -> float:
    """标注覆盖了多少秒：把各段合并起来算，重叠的部分只算一次，没标的空白不算。

    例如 [(0, 10, 导游), (5, 15, 游客甲), (20, 25, 导游)] → 0—15 秒和 20—25 秒，共 20 秒。
    """
    total = 0.0
    current_start = current_end = None
    for start, end in sorted((float(row[0]), float(row[1])) for row in rows):
        if current_end is None or start > current_end:  # 和前面的不相连：前面那一块算完，开始新的一块
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:  # 和前面的重叠或相连：并进去
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total
