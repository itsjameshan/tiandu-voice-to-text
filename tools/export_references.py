"""为 72 段计划录音生成待校对的参考文本（第 0 周老师运行一次）。

用法：
    python tools/export_references.py                       数据池用 config.yaml 里的 paths.data_pool
    python tools/export_references.py --pool D:\\data_pool   指定数据池文件夹
    python tools/export_references.py --overwrite            已有的也按剧本重新写（会丢掉校对结果，慎用）

每段录音一份 数据池/references/G1-S1-Q.txt，内容先照抄剧本台词（不含说话人和动作提示），
再由同学对照录音改成"实际说了什么"（见 docs/guides/proofreading.md）。
已经存在的文件默认不覆盖，校对过的不会丢，可以放心重复运行。详见 docs/teacher/data_pool.md。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="为 72 段计划录音生成待校对的参考文本（内容照抄剧本台词）")
    parser.add_argument("--pool", help="数据池文件夹，如 D:\\data_pool（不填就用 config.yaml 里的 paths.data_pool）")
    parser.add_argument("--overwrite", action="store_true",
                        help="已有的参考文本也按剧本重新写（会丢掉校对结果，慎用）")
    args = parser.parse_args(argv)

    from pipeline.config import load_config
    from pipeline.data import load_recording_plan
    from pipeline.pool import export_references, pool_paths

    root = Path(args.pool) if args.pool else Path(load_config()["paths"]["data_pool"])
    folder = pool_paths(root)["references"]
    print("== 生成待校对的参考文本 ==")
    print(f"数据池：{root}")

    try:
        written = export_references(root, overwrite=args.overwrite)
    except PermissionError as e:
        print(f"没有权限写入：{e.filename or root}。如果这个文件正用记事本或别的软件打开着，请先关闭再运行。")
        return 1
    except OSError as e:
        print(f"无法访问数据池文件夹：{e}。请检查路径是否写对、U 盘或共享文件夹是否连上。")
        return 1

    total = len(load_recording_plan())
    kept = total - len(written)
    if args.overwrite:
        print(f"按剧本重新写了 {len(written)} 份参考文本（原有内容已覆盖）：{folder}")
    else:
        print(f"新写了 {len(written)} 份参考文本：{folder}")
        if kept:
            print(f"已存在 {kept} 份，没有覆盖（校对过的不会丢）。"
                  "确实要全部按剧本重新生成，加 --overwrite（会丢掉校对结果）。")
    print(f"共 {total} 段计划录音。下一步：同学们在网页\"数据校对\"页对照录音校对，见 docs/guides/proofreading.md。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
