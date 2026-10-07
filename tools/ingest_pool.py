"""数据池入池：检查文件名、统一格式、算指纹、质检（老师或数据负责人在老师电脑上运行）。

用法：
    python tools/ingest_pool.py                       数据池用 config.yaml 里的 paths.data_pool
    python tools/ingest_pool.py --pool D:\\data_pool   指定数据池文件夹

先把录音（文件名如 G1-S1-Q.m4a）拷进 数据池/raw/，再运行。工具会：
    - 检查文件名，不合格的列出原因、不处理（不会自动猜、不会自动改名）；
    - 把合格的转成 normalized/G1-S1-Q.wav（16000 Hz 单声道），写进 manifest.csv，原始文件设为只读；
    - 质检（时长、音量、削波、大段无声、原始采样率），只报告、不修改录音，结果写进 qc_report.csv。
已经入池的文件（按 SHA-256 指纹判断）会跳过，可以放心重复运行。
有没处理的文件时退出码为 1，全部正常时为 0。详见 docs/teacher/data_pool.md。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _print_added(item: dict, normalized: Path) -> None:
    """打印一个新入池文件的结果。"""
    from pipeline.step1_ingest import _fmt_seconds  # 时长的写法和质检提示里的一样（同一个函数）

    wav = normalized / f"{item['stem']}.wav"
    head = f"  {item['file']}：转成 {wav}，时长 {_fmt_seconds(item['duration'])}"
    if item["replaced"]:
        head += "（复录：已替换旧版本，参考文本要按新录音重新校对）"
    if not item["qc"]:
        print(head + "，质检合格")
        return
    print(head + f"，质检发现 {len(item['qc'])} 个问题：")
    for problem in item["qc"]:
        print(f"      - {problem}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="数据池入池：检查文件名、统一格式、算指纹、质检")
    parser.add_argument("--pool", help="数据池文件夹，如 D:\\data_pool（不填就用 config.yaml 里的 paths.data_pool）")
    args = parser.parse_args(argv)

    # 文件名里可能有命令行窗口显示不了的字符（如表情符号），显示不了的用 ? 代替，不让程序因此出错
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from pipeline.audio import FfmpegNotFound
    from pipeline.config import load_config
    from pipeline.pool import ingest_pool, pool_paths

    cfg = load_config()
    root = Path(args.pool) if args.pool else Path(cfg["paths"]["data_pool"])
    paths = pool_paths(root)
    print("== 数据池入池 ==")
    print(f"数据池：{root}")

    try:
        result = ingest_pool(root, cfg)
    except FfmpegNotFound as e:
        print(e)
        return 1
    except PermissionError as e:
        target = e.filename2 or e.filename or root  # 改名替换文件时，filename2 才是被占用的那个文件
        print(f"没有权限写入：{target}。如果这个文件正用 Excel、Audacity 或别的软件打开着，请先关闭再重新运行"
              "（已经入池的录音会自动跳过，这次没处理完的会从头重新处理）。")
        return 1
    except OSError as e:
        print(f"无法访问数据池文件夹：{e}。请检查路径是否写对、U 盘或共享文件夹是否连上。")
        return 1
    except ValueError as e:  # 清单或质检报告读不出来（例如被 Excel 另存成了别的编码）
        print(e)
        return 1

    added, skipped, errors = result["added"], result["skipped"], result["errors"]
    if not (added or skipped or errors):
        print(f"raw 文件夹里还没有录音。请把录音（文件名如 G1-S1-Q.m4a）拷进 {paths['raw']} 后再运行。")
        return 0

    if added:
        print(f"\n新入池 {len(added)} 个：")
        for item in added:
            _print_added(item, paths["normalized"])
    if skipped:
        print(f"\n已经入池、内容没变，跳过 {len(skipped)} 个。")
    if errors:
        print(f"\n未处理 {len(errors)} 个（请按提示处理后重新运行）：")
        for error in errors:
            print(f"  {error['file']}：{error['reason']}")

    with_problems = sum(1 for item in added if item["qc"])
    print(f"\n汇总：新入池 {len(added)} 个（其中质检有问题 {with_problems} 个），跳过 {len(skipped)} 个，"
          f"未处理 {len(errors)} 个。")
    print(f"清单：{paths['manifest']}")
    print(f"质检报告：{paths['qc_report']}")
    if with_problems:
        print("质检只报告、不修改录音；是否重录由老师和数据负责人判断（阈值在 config.yaml 的 qc 一节）。")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
