"""数据池验收：生成验收表 acceptance.csv 和各组汇总 acceptance_summary.csv，并打印各组情况（第 6 周末老师运行）。

用法：
    python tools/check_pool.py                       数据池用 config.yaml 里的 paths.data_pool
    python tools/check_pool.py --pool D:\\data_pool   指定数据池文件夹

验收表每段计划录音一行：文件、组、条件、已交、命名、质检、校对遍数、说话人标注（秒）、片段标注、通过。
通过 = 已交 且 质检合格 且 至少 2 个不同的人校对过（入池之后）。
各组汇总：已交/应交（每组 9 段）、通过数、说话人标注总分钟、是否达到 10 分钟。
只读数据池、写这两张表格，不改录音和参考文本，可以随时重复运行。详见 docs/teacher/data_pool.md。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _not_passed_reasons(row: dict) -> list[str]:
    """已交但没通过的录音，说清楚卡在哪里、怎么办。"""
    from pipeline.pool import MIN_PROOFREADERS, NAME_OK, NOT_INGESTED, QC_OK

    reasons = []
    if row["命名"] != NAME_OK:
        reasons.append(f"命名{row['命名']}，请改名后重新放进 raw 文件夹再入池")
    elif row["质检"] == NOT_INGESTED:
        reasons.append("还没入池，请运行 tools/ingest_pool.py（看它的提示）")
    elif row["质检"] != QC_OK:
        reasons.append(f"质检：{row['质检']}")
    if row["校对遍数"] < MIN_PROOFREADERS:
        reasons.append(f"校对遍数 {row['校对遍数']}（要 {MIN_PROOFREADERS} 个不同的人校对）")
    return reasons


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="数据池验收：生成验收表和各组汇总")
    parser.add_argument("--pool", help="数据池文件夹，如 D:\\data_pool（不填就用 config.yaml 里的 paths.data_pool）")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # 命令行窗口显示不了的字用 ? 代替，不让程序因此出错

    from pipeline.config import load_config
    from pipeline.pool import MIN_PROOFREADERS, MIN_SPEAKER_MINUTES, acceptance, acceptance_summary, write_acceptance

    root = Path(args.pool) if args.pool else Path(load_config()["paths"]["data_pool"])
    print("== 数据池验收 ==")
    print(f"数据池：{root}")
    if not root.is_dir():
        print(f"数据池文件夹不存在：{root}。请检查路径是否写对、U 盘或共享文件夹是否连上。")
        return 1

    try:
        rows = acceptance(root)
        acceptance_path, summary_path = write_acceptance(root)
    except PermissionError as e:
        print(f"没有权限写入：{e.filename or root}。如果这个表格正用 Excel 打开着，请先关闭再重新运行。")
        return 1
    except ValueError as e:  # 标注表格有错，或表格的文字编码认不出来
        print(e)
        print("请改好后重新运行。")
        return 1
    except OSError as e:
        print(f"无法访问数据池文件夹：{e}。请检查路径是否写对、U 盘或共享文件夹是否连上。")
        return 1

    print(f"\n各组汇总（每组应交 9 段；通过 = 已交、质检合格、至少 {MIN_PROOFREADERS} 个不同的人校对过；"
          f"说话人标注每组至少 {MIN_SPEAKER_MINUTES} 分钟）：")
    for item in acceptance_summary(rows):
        minutes = item["说话人标注总分钟"]
        if item["是否达到 10 分钟"]:
            reached = "已达到"
        else:
            reached = f"还差 {MIN_SPEAKER_MINUTES - minutes:.1f} 分钟"
        print(f"  第 {item['组']} 组：已交 {item['已交']}/{item['应交']}，通过 {item['通过数']}，"
              f"说话人标注 {minutes:.1f} 分钟（{reached}）")
        missing = [row["文件"] for row in rows if row["组"] == item["组"] and not row["已交"]]
        if 0 < len(missing) < item["应交"]:
            print(f"      未交：{'、'.join(missing)}")

    submitted = sum(1 for row in rows if row["已交"])
    passed = sum(1 for row in rows if row["通过"])
    print(f"全班：已交 {submitted}/{len(rows)}，通过 {passed}/{len(rows)}。")

    waiting = [row for row in rows if row["已交"] and not row["通过"]]
    if waiting:
        print(f"\n已交但还没通过的录音 {len(waiting)} 段（请按原因处理）：")
        for row in waiting:
            print(f"  {row['文件']}：{'；'.join(_not_passed_reasons(row))}")

    print(f"\n验收表：{acceptance_path}")
    print(f"各组汇总：{summary_path}")
    print("老师确认后，用 tools/freeze_pool.py --version v1 冻结数据池版本。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
