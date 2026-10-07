"""对比：同一份数据、同样的参数，先用基线做法、再用某组的改进做法各测一遍，写出对比表。

用法（Windows 便携包里把 python 换成 python\\python.exe）：
    python tools/compare.py --slot denoise --method g1 --metric cer --pool D:\\data_pool --out reports/g1
    python tools/compare.py --slot hotword --method g5 --metric hotwords --pool D:\\data_pool --hotword on --out reports/g5
    python tools/compare.py --slot normalize --method g4 --metric numbers --out reports/g4
    python tools/compare.py --slot diarize --method g3 --metric speakers --pool D:\\data_pool --speakers ref --out reports/g3
    python tools/compare.py --slot clips --method g8 --metric clips --pool D:\\data_pool --out reports/g8

生成 <out>/compare_<指标>.md 和 compare_<指标>.csv，表格列：分组项、基线、改进、差值（= 改进 − 基线）。
分组项：字错率、专名正确率、说话人标错比例、片段起止误差是 全体 → Q（安静）/N（嘈杂教室）/F（口袋或远距离）→ 各组；
数字提取是 主指标 → 次要指标 → 各类型；话术分类是 7 类正确率 → 误报率（全部剧本、第 8 组剧本）→ 各类准确率、召回率 → 各组正确率。
这张表直接放进本组报告（reports/gN/README.md）的"对比表"一节。

各组用哪个槽位、哪个指标：
    第 1 组 denoise（或 vad）→ cer；第 2 组 enhance（或 vad）→ cer；第 3 组 diarize → speakers；
    第 4 组 normalize → numbers；第 5 组 hotword → hotwords；第 6、7 组 classify → classify；第 8 组 clips → clips。
其余参数（--pool、--files、--hotword、--no-cache）和 tools/evaluate.py 一样；--speakers（说话人数）只在 speakers 时有用：
auto 自动判断、ref 每段录音按标注里的人数（设对人数）、或一个正整数。
比较热词纠错（--slot hotword）时记得加 --hotword on：热词纠错默认是关闭的，关着时基线做法不做任何纠错。
话术分类（classify）在剧本台词上比：第 6、7 组训练好的模型是用全部剧本台词训练的，在同样的台词上比会虚高，
模型和关键词规则的正式对比用 tools/train_classifier.py 的按组留一报告。
识别结果有缓存：基线测过一次以后，再比较别的做法时基线部分几秒钟就出结果。
中途按 Ctrl+C 停下也没关系，再运行一次会接着往下测。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "reports" / "eval"
STOPPED_MESSAGE = "\n已停止（按了 Ctrl+C）。已经识别完的录音都有缓存，再运行一次会接着往下测。"

# 每个指标能比较哪些槽位（别的槽位不影响这个指标，比了也白比）
METRIC_SLOTS = {
    "cer": ("denoise", "enhance", "vad", "hotword"),
    "hotwords": ("denoise", "enhance", "vad", "hotword"),
    "numbers": ("normalize",),
    "speakers": ("denoise", "enhance", "vad", "diarize"),
    "classify": ("classify",),
    "clips": ("denoise", "enhance", "vad", "hotword", "normalize", "classify", "clips"),
}

# 差值怎么看：这些指标越小越好（差值为负表示变好），其余越大越好
LOWER_IS_BETTER = "字错率、过度纠正次数、说话人标错比例、误报率、起止误差、没配上的片段数、和标注都不重叠的工具片段"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对比：基线做法和改进做法在同一份数据上各测一遍，写出对比表")
    parser.add_argument("--slot", required=True, help="比较哪个槽位，如 denoise")
    parser.add_argument("--method", required=True, help="改进做法的名字，如 g1")
    parser.add_argument("--metric", required=True, choices=list(METRIC_SLOTS), help="用哪个指标比较")
    parser.add_argument("--pool", help="数据池文件夹（不填就用 config.yaml 里的 paths.data_pool；numbers、classify 用不到）")
    parser.add_argument("--files", nargs="+", metavar="文件编号", help="只用这几段录音，如 G1-S1-Q G1-S1-N")
    parser.add_argument("--out", help="报告写到哪个文件夹（默认 reports/eval）")
    parser.add_argument("--hotword", choices=["on", "off"], help="临时打开（on）或关闭（off）热词纠错")
    parser.add_argument("--no-cache", action="store_true", help="不用缓存的识别结果，全部重新识别")
    parser.add_argument("--speakers", metavar="人数",
                        help="说话人数（只在 --metric speakers 时有用）：auto 自动判断、ref 每段录音按标注里的人数（设对人数）、"
                             "或一个正整数（如 4）；不填就按 config.yaml 的 diarize.num_speakers")
    return parser


def _evaluate(metric: str, root: Path, cfg: dict, methods: dict, args, num_speakers=None) -> dict:
    """用指定的做法测一遍，返回汇总。"""
    from pipeline.evaluation import (
        eval_cer,
        eval_classify_rules,
        eval_clips,
        eval_hotwords,
        eval_numbers_lines,
        eval_speakers,
    )

    def say(message: str) -> None:
        print(message, flush=True)

    use_cache = not args.no_cache
    if metric == "cer":
        return eval_cer(root, cfg, methods, args.files, progress=say, use_cache=use_cache)[1]
    if metric == "hotwords":
        return eval_hotwords(root, cfg, methods, args.files, progress=say, use_cache=use_cache)[1]
    if metric == "speakers":
        return eval_speakers(root, cfg, methods, args.files, progress=say, use_cache=use_cache,
                             num_speakers=num_speakers)[1]
    if metric == "clips":
        return eval_clips(root, cfg, methods, args.files, progress=say, use_cache=use_cache)[1]
    if metric == "classify":
        return eval_classify_rules(cfg, method=methods["classify"])
    return eval_numbers_lines(cfg, methods)


def _show(value) -> str:
    """对比表里一个数的显示：没有值（如一对片段也没配上）写 —。"""
    return "—" if value is None else str(value)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    slots = METRIC_SLOTS[args.metric]
    if args.slot not in slots:
        print(f"槽位“{args.slot}”不影响指标“{args.metric}”，比较不出差别。这个指标可以比较的槽位：{'、'.join(slots)}")
        return 2

    from pipeline.config import load_config
    from pipeline.evaluation import NOTES, check_methods, compare_rows, describe_num_speakers, parse_num_speakers, write_report
    from pipeline.methods import SLOT_TITLES

    try:
        check_methods({args.slot: args.method})
        num_speakers = parse_num_speakers(args.speakers) if args.speakers else None
    except ValueError as e:
        print(e)
        return 2
    if args.speakers and args.metric != "speakers":
        print("提示：--speakers 只在 --metric speakers 时有用，这次用不到。")

    overrides = {}
    if args.hotword:
        overrides["hotword"] = {"enabled": args.hotword == "on"}
    cfg = load_config(overrides=overrides)
    root = Path(args.pool) if args.pool else Path(cfg["paths"]["data_pool"])
    out = Path(args.out) if args.out else DEFAULT_OUT

    print(f"== 对比：{SLOT_TITLES[args.slot]}（{args.slot}）baseline 与 {args.method}，指标 {args.metric} ==")
    if args.metric not in ("numbers", "classify"):
        print(f"数据池：{root}")
    if args.metric == "speakers":
        setting = num_speakers if num_speakers is not None else (cfg.get("diarize") or {}).get("num_speakers")
        print(f"说话人数：{describe_num_speakers(setting)}")
    hotword_off = args.slot == "hotword" and not cfg["hotword"].get("enabled")
    if hotword_off:
        print("提示：热词纠错现在是关闭的（config.yaml 的 hotword.enabled），基线做法不会做任何纠错；"
              "要和\"基线纠错\"比，加 --hotword on。")

    summaries = {}
    try:
        for label, method in (("基线", "baseline"), ("改进", args.method)):
            print(f"\n--- {label}：{args.slot}={method} ---", flush=True)
            summaries[label] = _evaluate(args.metric, root, cfg, {args.slot: method}, args, num_speakers)
    except (ValueError, FileNotFoundError) as e:
        print(e)
        return 1
    except OSError as e:
        print(f"无法访问数据池文件夹：{e}。请检查路径是否写对、U 盘或共享文件夹是否连上。")
        return 1
    except KeyboardInterrupt:
        print(STOPPED_MESSAGE)
        return 1

    rows = compare_rows(args.metric, summaries["基线"], summaries["改进"])
    info = dict(summaries["改进"]["info"])
    info["methods"] = dict(info["methods"])
    info["methods"][args.slot] = f"baseline（基线）／{args.method}（改进）"
    # 两次测评里做法发出的提示（如第 6、7 组的模型用不了、退回关键词规则）都写进报告开头
    warnings = list(dict.fromkeys((summaries["基线"].get("warnings") or []) + (summaries["改进"].get("warnings") or [])))
    notes = [
        f"两次测评用同一份数据、同样的参数，只换了{SLOT_TITLES[args.slot]}（{args.slot}）的做法："
        f"基线 baseline，改进 {args.method}。",
        f"差值 = 改进 − 基线。{LOWER_IS_BETTER}：差值为负表示变好；专名正确率、正确率、准确率、召回率：差值为正表示变好。"
        "没有值（如一对片段也没配上）的格子空着。",
        *NOTES[args.metric],
    ]
    if hotword_off:
        notes.append("这次对比时热词纠错是关闭的（hotword.enabled: false），基线做法没有做任何纠错。")

    for message in warnings:
        print(f"注意：{message}")
    print("\n对比（差值 = 改进 − 基线）：")
    for row in rows:
        print(f"  {row['分组项']}：基线 {_show(row['基线'])}，改进 {_show(row['改进'])}，差值 {_show(row['差值'])}")
    try:
        csv_path, md_path = write_report(out, f"compare_{args.metric}", rows,
                                         {"info": info, "tables": [], "rows_title": "对比表", "warnings": warnings},
                                         notes)
    except PermissionError as e:
        print(f"报告写不进去：{e.filename or out}。如果这个文件正用 Excel 或别的软件打开着，请先关闭再重新运行。")
        return 1
    print(f"\n对比报告：{md_path}")
    print(f"对比表格：{csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
