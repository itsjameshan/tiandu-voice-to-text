"""检查话术分类补充句子（data/classification_extra.csv）：格式、类别、长度、重复、红线词。

用法：
    python tools/check_classification_data.py                 检查 data/classification_extra.csv
    python tools/check_classification_data.py 我审过的.csv     检查别的文件（列要和补充句子表一样）

第 6、7 组审核补充句子（改 label、改 text、把 reviewed 改成"已审核"）之后，用它查一遍。
有问题时逐条列出"第几行（编号）：什么问题"，退出码为 1；全部合格时退出码为 0。
最后打印每个类别有多少句，和计划的目标数量对照。

检查哪些（行号按用 Excel 打开时的行号算，表头是第 1 行）：
    1. 列：必须有 id、text、label、source、reviewed、reviewer、note 这 7 列；
    2. 编号 id：形如 X0001，不能重复；
    3. 类别 label：只能是 data/labels.json 里的 7 个类别之一；
    4. 长度：有效字数（只数汉字、字母、数字，不数标点和空格）在 6—60 之间；
    5. 重复：去掉标点后，不能和文件里另一句相同，也不能和剧本台词（data/lines.csv）任何一句相同
       （测评时测试集是剧本台词，补充句子和台词一样就等于"把考题放进了练习册"）；
    6. 红线词：不能出现 RED_LINE_WORDS 里的词（来自 docs/script_spec.md 的"内容红线"）；
    7. source 必须是"AI生成"；reviewed 只能是"未审核"或"已审核"，"已审核"的要填 reviewer。

基线做法：
    逐行检查上面 7 项，只报告、不修改文件。查重复时用 text_norm.normalize_for_cer
    （全角变半角、只留汉字、字母、数字）把标点和空格去掉再比较。
可改进方向：
    - 近似重复：两句只差一两个字也算重复（例如用编辑距离）；
    - 名称检查：句子里的旅行社、店名、人名是否都在 data/fictional_names.csv 里（现在要靠人工审核）；
    - 数字写法：统计汉字读法和阿拉伯数字的比例是否接近"八成、两成"。
测评指标：
    不涉及模型效果；由 tests/test_classification_data.py 检查：仓库里的补充句子全部合格，
    故意写错的句子都能被查出来。
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.data import LABEL_NAMES, load_lines  # noqa: E402  先把项目文件夹加进 sys.path 才能导入
from pipeline.text_norm import normalize_for_cer  # noqa: E402

# 默认检查的文件
EXTRA_PATH = ROOT / "data" / "classification_extra.csv"

# 必须有的列（顺序同补充句子表）
COLUMNS = ["id", "text", "label", "source", "reviewed", "reviewer", "note"]

# 来源只能是这个（补充句子都是 AI 生成的，要如实标明）
SOURCE = "AI生成"

# 审核状态只能是这两种
REVIEW_STATES = ("未审核", "已审核")

# 每句的有效字数范围（只数汉字、字母、数字）
MIN_CHARS = 6
MAX_CHARS = 60

# 编号的格式：X 加 4 位数字，如 X0001
ID_PATTERN = re.compile(r"X\d{4}")

# 红线词：来自 docs/script_spec.md 的"内容红线"（不写违法违规、报警警察、辱骂暴力，
# 不涉及政治、民族、宗教、地域，不出现"开光"等宗教说法和带歧视的说法，不写治病疗效）。
# 发现新的不合适说法，就加在这里。
RED_LINE_WORDS = (
    "违法", "违规", "警察", "报警", "开光", "民族", "宗教", "政府", "政治", "国籍",
    "外地人", "乡下人", "骂", "脏话", "打人", "滚", "恐吓", "治病", "疗效",
)

# 计划里每类的目标句数（只用来打印对照，不作为检查条件）
TARGETS = {"威胁消费": 200, "服务态度": 150, "行程变更": 150, "正常讲解": 200, "购物安排": 110, "费用": 110, "其他": 80}


def read_rows(path) -> tuple[list[str], list[dict]]:
    """读 CSV，返回 (表头, 每行一个字典)。Excel 另存的 GBK、带 BOM 的 UTF-8 都能读（见 pipeline.textio）。"""
    from pipeline.textio import read_csv_dicts

    return read_csv_dicts(path, hint="请用 Excel 另存为\"CSV UTF-8（逗号分隔）\"后再试。")


def script_texts() -> dict[str, str]:
    """剧本台词去标点后的文字 → 出处（如"G1-S1 第 3 句"），用来查补充句子是否和台词重复。"""
    result = {}
    for line in load_lines():
        result.setdefault(normalize_for_cer(line["text"]), f"{line['script_id']} 第 {line['line_no']} 句")
    return result


def check_rows(rows: list[dict], scripts: dict[str, str] | None = None) -> list[str]:
    """逐行检查，返回问题列表（每条形如"第 3 行（X0002）：类别……"）；没有问题时返回空列表。

    rows：read_rows 读出的行；scripts：script_texts() 的结果，不填就现读剧本台词。
    """
    if scripts is None:
        scripts = script_texts()
    problems = []
    seen_ids: dict[str, int] = {}  # 编号 → 第几行
    seen_texts: dict[str, tuple[int, str]] = {}  # 去标点后的文字 → (第几行, 编号)

    for index, row in enumerate(rows):
        line_no = index + 2  # 表头是第 1 行
        row_id = (row.get("id") or "").strip()
        text = row.get("text") or ""
        label = (row.get("label") or "").strip()
        where = f"第 {line_no} 行（{row_id or '没有编号'}）"

        # 编号
        if not ID_PATTERN.fullmatch(row_id):
            problems.append(f"{where}：编号“{row_id}”格式不对，应该是 X 加 4 位数字，如 X0001")
        elif row_id in seen_ids:
            problems.append(f"{where}：编号 {row_id} 和第 {seen_ids[row_id]} 行重复")
        else:
            seen_ids[row_id] = line_no

        # 类别
        if label not in LABEL_NAMES:
            problems.append(f"{where}：类别“{label}”不合法，只能是：{'、'.join(LABEL_NAMES)}")

        # 长度（有效字数）
        plain = normalize_for_cer(text)
        if len(plain) < MIN_CHARS:
            problems.append(f"{where}：句子太短（有效字数 {len(plain)}，至少 {MIN_CHARS}）")
        elif len(plain) > MAX_CHARS:
            problems.append(f"{where}：句子太长（有效字数 {len(plain)}，最多 {MAX_CHARS}）")

        # 重复：文件里的另一句、剧本台词
        if plain:
            if plain in seen_texts:
                other_line, other_id = seen_texts[plain]
                problems.append(f"{where}：去掉标点后和第 {other_line} 行（{other_id}）重复")
            else:
                seen_texts[plain] = (line_no, row_id)
            if plain in scripts:
                problems.append(f"{where}：去掉标点后和剧本台词（{scripts[plain]}）相同，测试集是剧本台词，不能重复")

        # 红线词
        found = [word for word in RED_LINE_WORDS if word in text]
        if found:
            problems.append(f"{where}：含有红线词“{'、'.join(found)}”（见 docs/script_spec.md 的内容红线），请改写或删掉")

        # 来源、审核状态
        if (row.get("source") or "").strip() != SOURCE:
            problems.append(f"{where}：source 必须是“{SOURCE}”，现在是“{row.get('source') or ''}”")
        reviewed = (row.get("reviewed") or "").strip()
        if reviewed not in REVIEW_STATES:
            problems.append(f"{where}：reviewed 只能是“未审核”或“已审核”，现在是“{reviewed}”")
        elif reviewed == "已审核" and not (row.get("reviewer") or "").strip():
            problems.append(f"{where}：已审核的句子要在 reviewer 里填审核人（学号后四位）")

    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="检查话术分类补充句子：列、类别、长度、重复、红线词")
    parser.add_argument("file", nargs="?", default=str(EXTRA_PATH),
                        help="要检查的 CSV 文件，不填就检查 data/classification_extra.csv")
    args = parser.parse_args(argv)

    # 命令行窗口显示不了的字符用 ? 代替，不让程序因此出错
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    path = Path(args.file)
    print("== 话术分类补充句子检查 ==")
    print(f"文件：{path}")
    if not path.is_file():
        print(f"找不到文件：{path}")
        return 1

    columns, rows = read_rows(path)
    missing = [name for name in COLUMNS if name not in columns]
    if missing:
        print(f"缺少列：{'、'.join(missing)}。必须有这 {len(COLUMNS)} 列：{','.join(COLUMNS)}")
        return 1

    problems = check_rows(rows)

    # 每类句数（和目标对照）
    counts = Counter((row.get("label") or "").strip() for row in rows)
    print(f"共 {len(rows)} 句。各类句数：")
    for label in LABEL_NAMES:
        print(f"  {label}：{counts.get(label, 0)} 句（计划目标约 {TARGETS.get(label, 0)} 句）")
    reviewed = sum(1 for row in rows if (row.get("reviewed") or "").strip() == "已审核")
    with_digits = sum(1 for row in rows if re.search(r"[0-9]", row.get("text") or ""))
    print(f"已审核 {reviewed} 句，未审核 {len(rows) - reviewed} 句；含阿拉伯数字的句子 {with_digits} 句。")

    if problems:
        print(f"\n发现 {len(problems)} 个问题：")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("\n全部合格。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
