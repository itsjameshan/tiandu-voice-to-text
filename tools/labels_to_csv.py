"""把 Audacity 导出的标签文件转成标注表格（说话人时间或疑似片段起止）。

用法：
    python tools/labels_to_csv.py 标签.txt 输出.csv                    说话人标注（默认）
    python tools/labels_to_csv.py 标签.txt 输出.csv --kind clips        疑似片段标注
    python tools/labels_to_csv.py 标签.txt --pool D:\\data_pool --file G3-S1-Q --kind speakers
        直接写到 数据池/annotations/speakers/G3-S1-Q.csv（--kind clips 写到 annotations/clips/G3-S1-Q.csv）
        不写 --pool 时用 config.yaml 里的 paths.data_pool。

转换后的表格三列，第一行是列名：
    说话人标注  start,end,speaker  （开始秒、结束秒、角色名，如"导游"）
    片段标注    start,end,label    （开始秒、结束秒、类别名：购物安排、费用、行程变更、服务态度、威胁消费；
                                   写"消费施压"也可以，会存成"威胁消费"；写别的会报错）
标签文件是 UTF-8 还是 GBK 编码都能读；以"\\"开头的频率信息行自动跳过。
有错时一条也不写，说明第几个标签有什么问题，退出码为 1。详见 docs/guides/annotation.md。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KIND_NAMES = {"speakers": "说话人标注", "clips": "疑似片段标注"}


def _pool_warnings(labels, kind: str, root: Path, plan_item: dict) -> list[str]:
    """转到数据池时额外核对：角色名在不在剧本的角色表里、标签有没有超出录音时长。只提醒，不拦着。"""
    from pipeline.data import get_script
    from pipeline.pool import pool_paths, read_csv_rows

    stem = Path(plan_item["file_name"]).stem
    warnings = []
    if kind == "speakers":
        roles = get_script(plan_item["script_id"])["roles"]
        unknown = sorted({label for _, _, label in labels if label not in roles})
        if unknown:
            warnings.append(f"这些角色名不在剧本 {plan_item['script_id']} 的角色表（{'、'.join(roles)}）里："
                            f"{'、'.join(unknown)}。如果是写错了（如\"导游\"写成\"导\"），请在 Audacity 里改好后"
                            "重新导出、再转换；客串的同学也按剧本角色名写")

    # 录音已经入池的话，清单里有它的时长：标签超出录音时长，多半是标错了录音
    try:
        manifest = read_csv_rows(pool_paths(root)["manifest"])
        durations = [float(row["时长（秒）"]) for row in manifest if row.get("文件编号") == stem]
    except (ValueError, KeyError):  # 清单读不了：这只是额外的提醒，跳过
        durations = []
    last_end = max(end for _, end, _ in labels)
    if durations and last_end > durations[0] + 1:
        warnings.append(f"最后一个标签结束在 {last_end:.1f} 秒，比录音 {stem}（{durations[0]:.1f} 秒）还长。"
                        f"是不是标错了录音？要标数据池里的 normalized/{stem}.wav")
    return warnings


def _role_like(label: str) -> bool:
    """片段标注里这个标签不是类别名（可能是角色名，选错了 --kind）。"""
    from pipeline.annotations import clip_label

    try:
        clip_label(label)
    except ValueError:
        return bool(label)
    return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="把 Audacity 导出的标签文件转成标注表格（说话人时间或疑似片段起止）")
    parser.add_argument("labels", help="Audacity 导出的标签文件，如 G3-S1-Q_speakers.txt")
    parser.add_argument("output", nargs="?", help="输出的表格，如 G3-S1-Q.csv（用 --pool --file 时不写）")
    parser.add_argument("--pool", help="数据池文件夹，如 D:\\data_pool（和 --file 一起用；不填就用 config.yaml 里的 "
                                       "paths.data_pool）")
    parser.add_argument("--file", help="录音的文件编号，如 G3-S1-Q：表格直接写到数据池的 annotations 文件夹下")
    parser.add_argument("--kind", choices=["speakers", "clips"], default="speakers",
                        help="speakers 说话人标注（默认），clips 疑似片段标注")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # 命令行窗口显示不了的字用 ? 代替，不让程序因此出错

    from pipeline.annotations import (
        CLIP_LABELS,
        annotated_seconds,
        clip_label,
        read_audacity_labels,
        write_clips_csv,
        write_turns_csv,
    )
    from pipeline.config import load_config
    from pipeline.data import load_recording_plan
    from pipeline.pool import pool_paths

    print(f"== Audacity 标签 → {KIND_NAMES[args.kind]}表格 ==")

    # 1. 两种写法只能选一种：给输出文件名，或者用 --pool --file 直接写进数据池
    if args.output and (args.file or args.pool):
        print("两种写法只能选一种：要么写输出表格的文件名，要么用 --pool 和 --file 直接写进数据池。")
        return 1
    if not args.output and not args.file:
        print("请写输出表格的文件名（如 G3-S1-Q.csv），或者用 --file G3-S1-Q 直接写进数据池。")
        return 1

    # 2. 算出要写到哪里
    root = plan_item = None
    if args.output:
        target = Path(args.output)
    else:
        plan = {Path(item["file_name"]).stem: item for item in load_recording_plan()}
        stem = args.file.strip()
        if stem not in plan:
            print(f"没有这段录音：{stem}。--file 只写文件编号（组号-剧本号-条件代码），如 G3-S1-Q，"
                  "不带扩展名，全部大写。")
            return 1
        root = Path(args.pool) if args.pool else Path(load_config()["paths"]["data_pool"])
        if not root.is_dir():
            print(f"数据池文件夹不存在：{root}。请检查路径是否写对、U 盘或共享文件夹是否连上。")
            return 1
        plan_item = plan[stem]
        target = pool_paths(root)[f"annotations_{args.kind}"] / f"{stem}.csv"

    # 3. 读标签文件
    source = Path(args.labels)
    if not source.is_file():
        print(f"找不到标签文件：{source}。请检查文件名和路径（在 Audacity 里用\"文件 → 导出 → 导出标签\"保存）。")
        return 1
    if target.resolve() == source.resolve():
        print("输出表格和标签文件是同一个文件，会把标签文件覆盖掉。请换一个输出文件名（如 G3-S1-Q.csv）。")
        return 1
    try:
        labels = read_audacity_labels(source)
    except ValueError as e:
        print(e)
        return 1
    if not labels:
        print(f"标签文件里没有标签：{source}。请确认在 Audacity 里加了标签（Ctrl+B）后再导出。")
        return 1

    # 4. 检查并写表格（有错时一条也不写）
    existed = target.exists()
    try:
        if args.kind == "speakers":
            write_turns_csv(target, labels)
        else:
            write_clips_csv(target, labels)
    except ValueError as e:
        print(e)
        if args.kind == "clips" and any(_role_like(label) for _, _, label in labels):
            print("如果这其实是说话人标注（标签写的是角色名），请把 --kind clips 改成 --kind speakers。")
        return 1
    except PermissionError:
        print(f"没有权限写入：{target}。如果这个表格正用 Excel 打开着，请先关闭再运行。")
        return 1
    except OSError as e:
        print(f"写不进去：{target}（{e}）。请检查路径是否写对、U 盘或共享文件夹是否连上。")
        return 1

    # 5. 打印汇总和提醒
    print(f"读到 {len(labels)} 个标签，已写入：{target}" + ("（原来的表格已覆盖）" if existed else ""))
    if args.kind == "speakers":
        names = sorted({label for _, _, label in labels})
        seconds = annotated_seconds(labels)
        print(f"说话人 {len(names)} 个：{'、'.join(names)}；标注覆盖 {seconds:.1f} 秒（{seconds / 60:.1f} 分钟，"
              "重叠的部分只算一次）。")
        like_clips = [name for name in names if name in CLIP_LABELS or name == "消费施压"]
        if like_clips:
            print(f"提醒：标签里有类别名（{'、'.join(like_clips)}）。如果这是疑似片段标注，请加 --kind clips 重新转换。")
    else:
        counts = {}
        for _, _, label in labels:
            name = clip_label(label)
            counts[name] = counts.get(name, 0) + 1
        print("片段：" + "，".join(f"{name} {n} 个" for name, n in counts.items()) + "。")

    if plan_item is not None:
        for warning in _pool_warnings(labels, args.kind, root, plan_item):
            print("提醒：" + warning + "。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
