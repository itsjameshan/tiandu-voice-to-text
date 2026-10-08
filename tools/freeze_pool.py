"""冻结数据池版本（验收后老师运行）：之后所有测评报告都写明用的是哪个版本，各组的结果才能比较。

用法：
    python tools/freeze_pool.py --version v1                       数据池用 config.yaml 里的 paths.data_pool
    python tools/freeze_pool.py --pool D:\\data_pool --version v1   指定数据池文件夹

把清单、质检报告、校对记录、验收表、参考文本、标注复制到 数据池/versions/v1/，
并在 checksums.json 里记下每个文件（包括 normalized/ 下的录音）的 SHA-256 指纹；录音太大，只记指纹、不复制。
同一个版本名不能覆盖：参考文本或标注改过后，请冻结一个新版本（如 v2），基线和改进要用同一个版本重新测。
详见 docs/teacher/data_pool.md。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="冻结数据池版本（复制清单、参考文本、标注等，记下每个文件的指纹）")
    parser.add_argument("--version", required=True, help="版本名，如 v1（只用英文字母、数字）")
    parser.add_argument("--pool", help="数据池文件夹，如 D:\\data_pool（不填就用 config.yaml 里的 paths.data_pool）")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # 命令行窗口显示不了的字用 ? 代替，不让程序因此出错

    from pipeline.config import load_config
    from pipeline.pool import current_version, freeze, read_csv_rows

    root = Path(args.pool) if args.pool else Path(load_config()["paths"]["data_pool"])
    print("== 冻结数据池版本 ==")
    print(f"数据池：{root}")
    if not root.is_dir():
        print(f"数据池文件夹不存在：{root}。请检查路径是否写对、U 盘或共享文件夹是否连上。")
        return 1

    try:
        target = freeze(root, args.version)
    except FileExistsError as e:
        print(e)
        print(f"现在最新的版本是 {current_version(root) or '（没有）'}。")
        return 1
    except FileNotFoundError as e:  # 清单里的录音不见了
        print(e)
        return 1
    except PermissionError as e:
        print(f"没有权限写入：{e.filename or root}。如果这个文件正用 Excel 或别的软件打开着，请先关闭再重新运行。")
        return 1
    except ValueError as e:  # 版本名不合格、标注表格有错
        print(e)
        return 1
    except OSError as e:
        print(f"无法访问数据池文件夹：{e}。请检查路径是否写对、U 盘或共享文件夹是否连上。")
        return 1

    files = json.loads((target / "checksums.json").read_text(encoding="utf-8"))["files"]
    recordings = sum(1 for name in files if name.startswith("normalized/"))
    rows = read_csv_rows(target / "acceptance.csv")
    passed = sum(1 for row in rows if row.get("通过") == "是")

    print(f"已冻结数据池版本 {args.version.strip()}：{target}")
    print(f"复制了 {len(files) - recordings} 个文件（清单、质检报告、校对记录、验收表、参考文本、标注），"
          f"记下了 {len(files)} 个文件的 SHA-256 指纹（其中录音 {recordings} 段，录音只记指纹、不复制）。")
    print(f"验收：{len(rows)} 段计划录音中通过 {passed} 段（详见 {target / 'acceptance.csv'}）。")
    if recordings == 0:
        print("注意：清单里还没有入池的录音，这个版本里没有录音。")
    print(f"之后所有测评报告都写明\"数据池版本 {args.version.strip()}\"。参考文本或标注再改动时，"
          "请冻结一个新版本（如 v2），并通知各组用新版本重新测基线和改进（基线和改进必须用同一个版本）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
