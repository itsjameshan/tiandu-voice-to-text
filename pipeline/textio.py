"""读文本文件和 CSV 表格时自动认文字编码（只用 Python 自带的库，机房自带的 Python 也能用）。

同学们常用记事本、Excel 改文件，存出来的编码不一样：
    记事本：UTF-8（带不带 BOM）、ANSI（中文 Windows 上就是 GBK）、"Unicode"（UTF-16，文件开头有 BOM）；
    Excel：另存为"CSV UTF-8（逗号分隔）"是带 BOM 的 UTF-8，另存为"CSV（逗号分隔）"是 GBK。
这里依次试：开头是 UTF-16 的 BOM 就按 UTF-16 读，否则先试 UTF-8，再试 GB18030（包含 GBK）。

基线做法 / 可改进方向 / 测评指标：这是读文件的小工具，没有可改进的做法，也不参与测评。
"""
import codecs
import csv
import io
from pathlib import Path

DEFAULT_HINT = "请用记事本或 Excel 另存为 UTF-8 后再试"


def read_text(path, hint: str = DEFAULT_HINT) -> str:
    """读一个文本文件，返回文字。编码认不出来时抛 ValueError（中文说明，hint 是告诉同学怎么办的那句话）。"""
    path = Path(path)
    data = path.read_bytes()
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16")
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"文件的文字编码认不出来：{path}。{hint}")


def read_csv_dicts(path, hint: str = DEFAULT_HINT) -> tuple[list[str], list[dict]]:
    """读一个 CSV 表格，返回 (表头, 每行一个字典)。编码同 read_text；格子里有换行也能读。"""
    reader = csv.DictReader(io.StringIO(read_text(path, hint), newline=""))
    rows = [dict(row) for row in reader]
    return list(reader.fieldnames or []), rows
