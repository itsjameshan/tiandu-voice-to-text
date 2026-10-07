"""读取 data/ 目录里的虚构剧本和表格。

data/ 里有什么（详见 data/README.md）：
    scripts/group_1.json … group_8.json  24 个剧本（8 组 × 3 个），每个剧本有 id、title、roles、lines 等
    lines.csv              同样 1055 行台词的扁平表（多了有效字数、备注）
    labels.json            7 个话术类别；flag=true 的 5 类输出为"疑似·类别"
    hotwords.txt           热词表（只含正确名称，一行一个）
    hotword_variants.csv   第 5 组剧本里故意说错或简称的名称
    fictional_names.csv    全部虚构名称与号码
    recording_plan.csv     录音计划（72 个录音文件）

基线做法：
    只用标准库（json、csv）读取；CSV 一律用 utf-8-sig 编码读（文件带 BOM）。
    读过一次的文件用 functools.lru_cache 记住，不重复读盘；
    每次返回的都是新的列表和字典，调用方随便改也不会影响下一次读到的内容。
可改进方向：
    数据由老师维护，学生一般不改这个文件；要加新的数据表，照着下面的写法加一个 load_xxx() 函数。
测评指标：
    数量对得上：24 个剧本、1055 行台词、7 个类别、121 个热词、72 个计划录音（见 tests/test_data.py）。
"""
import copy
import csv
import json
from functools import lru_cache
from pathlib import Path

from pipeline.schema import LIST_SEP  # 与统一中间格式共用同一个分隔符"；"

# 仓库根目录下的 data/
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# 列表型字段（如 numbers、hotwords）里多个值之间用全角分号分隔


# ---------------- 读文件的小工具（带缓存） ----------------

@lru_cache(maxsize=None)
def _read_json(name: str):
    """读 data/ 下的一个 JSON 文件（读一次后缓存）。"""
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _read_csv(name: str) -> tuple[dict, ...]:
    """读 data/ 下的一个 CSV 文件，每行一个字典（读一次后缓存）。"""
    with open(DATA_DIR / name, encoding="utf-8-sig", newline="") as f:
        return tuple(csv.DictReader(f))


def _split_unique(text: str) -> list[str]:
    """把 "2800元；15:40；2800元" 拆成 ["2800元", "15:40"]：去掉空项、去重、保持原来的先后顺序。"""
    result = []
    for item in text.split(LIST_SEP):
        item = item.strip()
        if item and item not in result:
            result.append(item)
    return result


# ---------------- 剧本与台词 ----------------

@lru_cache(maxsize=None)
def _all_scripts() -> tuple[dict, ...]:
    """按组号顺序读出 24 个剧本，每个剧本加上 group（组号，整数）。"""
    scripts = []
    for group in range(1, 9):
        data = _read_json(f"scripts/group_{group}.json")
        for script in data["scripts"]:
            script = dict(script)
            script["group"] = int(data["group"])
            scripts.append(script)
    return tuple(scripts)


def load_scripts() -> list[dict]:
    """返回全部 24 个剧本（G1-S1 … G8-S3），每个都带 group（整数）。"""
    return copy.deepcopy(list(_all_scripts()))


def get_script(script_id: str) -> dict:
    """按编号取一个剧本，如 get_script("G1-S1")。编号不存在时抛 KeyError。"""
    for script in _all_scripts():
        if script["id"] == script_id:
            return copy.deepcopy(script)
    raise KeyError(f"没有这个剧本编号：{script_id}（编号形如 G1-S1，从 G1-S1 到 G8-S3）")


def load_lines() -> list[dict]:
    """返回 lines.csv 的 1055 行台词。

    numbers、hotwords 拆成列表（去重、保持顺序）；group、line_no、effective_chars 转成整数。
    """
    lines = []
    for row in _read_csv("lines.csv"):
        line = dict(row)
        line["group"] = int(line["group"])
        line["line_no"] = int(line["line_no"])
        line["effective_chars"] = int(line["effective_chars"])
        line["numbers"] = _split_unique(line["numbers"])
        line["hotwords"] = _split_unique(line["hotwords"])
        lines.append(line)
    return lines


def script_reference_text(script_id: str) -> str:
    """一个剧本的参考文本：全部台词的 text 按顺序用换行连接。

    不含说话人，也不含 direction（动作、环境提示，录音时不念）。
    """
    script = get_script(script_id)
    return "\n".join(line["text"] for line in script["lines"])


# ---------------- 话术类别 ----------------

def _load_labels() -> list[dict]:
    return _read_json("labels.json")["labels"]


# 7 个类别名称，顺序同 labels.json
LABEL_NAMES: list[str] = [item["name"] for item in _load_labels()]

# 会被标成"疑似·…"的 5 个类别（flag=true）
FLAG_LABELS: list[str] = [item["name"] for item in _load_labels() if item["flag"]]

# 5 个合法的显示标签，如"疑似·费用"（取自 labels.json 的 output）
FLAG_OUTPUTS: list[str] = [item["output"] for item in _load_labels() if item["flag"]]

# 类别名 → 界面和初稿上显示的名称。老师决定"威胁消费"显示为"消费施压"，其余不变
DISPLAY_NAMES: dict[str, str] = {item["name"]: item["display"] for item in _load_labels()}


def label_output(category: str) -> str:
    """类别 → 段落里的标签：前 5 类返回 labels.json 里的 output（如"疑似·费用""疑似·消费施压"），
    正常讲解和其他返回 ""。

    不在 7 个类别里的名称（例如"投诉"）一律抛 ValueError，保证标签只会是合法的几种。
    """
    if category not in LABEL_NAMES:
        raise ValueError(f"未知的类别：{category!r}，只能是：{'、'.join(LABEL_NAMES)}")
    for item in _load_labels():
        if item["name"] == category and item["flag"]:
            return item["output"]
    return ""


# ---------------- 热词、虚构名称、录音计划 ----------------

@lru_cache(maxsize=None)
def _hotwords() -> tuple[str, ...]:
    text = (DATA_DIR / "hotwords.txt").read_text(encoding="utf-8-sig")
    words = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):  # 跳过空行和 # 开头的说明行
            words.append(line)
    return tuple(words)


def load_hotwords() -> list[str]:
    """热词表（121 个正确名称），一行一个。"""
    return list(_hotwords())


def load_hotword_variants() -> list[dict]:
    """故意说错或简称的名称：spoken_variant（实际说的）、correct_name（正确名称）、kind、scripts。"""
    return [dict(row) for row in _read_csv("hotword_variants.csv")]


def load_fictional_names() -> list[dict]:
    """全部虚构名称与号码：type（类型）、name（名称）、groups（出现的组）、note（说明）。"""
    return [dict(row) for row in _read_csv("fictional_names.csv")]


def load_recording_plan() -> list[dict]:
    """录音计划，72 行（24 个剧本 × 3 种录音条件）。

    列：file_name、script_id、group、condition、condition_code、how_to_record、expected_minutes。
    expected_minutes（预计分钟数）转成小数，group 转成整数。
    """
    plan = []
    for row in _read_csv("recording_plan.csv"):
        item = dict(row)
        item["group"] = int(item["group"])
        item["expected_minutes"] = float(item["expected_minutes"])
        plan.append(item)
    return plan
