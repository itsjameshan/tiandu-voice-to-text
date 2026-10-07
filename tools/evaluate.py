"""测评：在数据池的录音（或剧本台词文字）上算指标，结果写成 CSV 表格和 Markdown 报告。

用法（Windows 便携包里把 python 换成 python\\python.exe，路径里的 / 也可以写成 \\）：
    python tools/evaluate.py cer --pool D:\\data_pool --out reports/g1
    python tools/evaluate.py cer --pool D:\\data_pool --method denoise=g1 --method vad=g1 --out reports/g1
    python tools/evaluate.py cer --pool D:\\data_pool --files G1-S1-Q G1-S1-N        只测这几段录音
    python tools/evaluate.py hotwords --pool D:\\data_pool --method hotword=g5 --hotword on --out reports/g5
    python tools/evaluate.py numbers --method normalize=g4 --out reports/g4         不用录音，不用 --pool

子命令（指标）：
    cer       字错率，按录音、录音条件（Q/N/F）、组汇总                  —— 全员；第 1、2 组主指标
    hotwords  专名正确率、过度纠正次数                                  —— 第 5 组
    numbers   数字提取正确率、召回率，按类型分开（在剧本台词文字上测）    —— 第 4 组
    speakers、classify、clips 将在后续任务中加入。

常用参数：
    --pool DIR          数据池文件夹（不填就用 config.yaml 里的 paths.data_pool）
    --method 槽位=做法   换某个槽位的做法，可以写好几个（如 --method denoise=g1 --method vad=g1）；
                        没写的槽位按 config.yaml 的 methods
    --files 编号 ...     只测这几段录音（文件编号，如 G1-S1-Q）
    --out DIR           报告写到哪个文件夹（默认 reports/eval），生成 <指标>.csv 和 <指标>.md
    --hotword on|off    临时打开或关闭热词纠错（不填就按 config.yaml 的 hotword.enabled，默认关闭）；
                        测热词纠错的做法（如 --method hotword=g5）时要加 --hotword on，不然基线不做任何纠错
    --no-cache          不用缓存的识别结果，全部重新识别

识别结果会缓存在 数据池/asr_cache/ 里：第一次测 72 段录音要几十分钟（每段录音都会显示进度），
以后只换热词纠错的做法、或者再测一遍时，几秒钟就出结果（缓存的规则见 pipeline/evaluation.py 开头）。
中途按 Ctrl+C 停下也没关系：已经识别完的录音都有缓存，再运行一次会接着往下测。
指标怎么算、怎么看，见 docs/guides/evaluation.md。报告里写明数据池版本；剧本数据上的测评结果不代表真实场景的效果。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 不写 --out 时，报告放在这里
DEFAULT_OUT = ROOT / "reports" / "eval"

# 还没做的子命令打印这句话
LATER_MESSAGE = "这个测评将在后续任务中加入"

# 中途按 Ctrl+C 停下时打印这句话
STOPPED_MESSAGE = "\n已停止（按了 Ctrl+C）。已经识别完的录音都有缓存，再运行一次会接着往下测。"

# 热词纠错开关关着、却要测热词纠错的做法时，打印这句提示
HOTWORD_OFF_HINT = ("提示：热词纠错现在是关闭的（config.yaml 的 hotword.enabled），基线做法不会做任何纠错"
                    "（各组的做法如果直接调用基线，也一样）；要测纠错的效果，加 --hotword on。")

# 每个子命令的说明（--help 里显示）
METRIC_HELP = {
    "cer": "字错率（错字、漏字、多字），按录音、录音条件、组汇总",
    "hotwords": "专名正确率、过度纠正次数（第 5 组）",
    "numbers": "数字提取正确率、召回率，按类型分开（在剧本台词文字上测，第 4 组）",
    "speakers": "说话人标错的时长比例（第 3 组）——将在后续任务中加入",
    "classify": "话术分类的准确率、召回率、误报率（第 6、7、8 组）——将在后续任务中加入",
    "clips": "疑似片段起止误差（第 8 组）——将在后续任务中加入",
}


def build_parser() -> argparse.ArgumentParser:
    """命令行参数：一个子命令对应一个指标，每个子命令都认下面这些参数。"""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pool", help="数据池文件夹，如 D:\\data_pool（不填就用 config.yaml 里的 paths.data_pool）")
    common.add_argument("--method", action="append", default=[], metavar="槽位=做法",
                        help="换某个槽位的做法，如 denoise=g1；可以写好几次")
    common.add_argument("--files", nargs="+", metavar="文件编号", help="只测这几段录音，如 G1-S1-Q G1-S1-N")
    common.add_argument("--out", help="报告写到哪个文件夹（默认 reports/eval）")
    common.add_argument("--hotword", choices=["on", "off"], help="临时打开（on）或关闭（off）热词纠错")
    common.add_argument("--no-cache", action="store_true", help="不用缓存的识别结果，全部重新识别")

    parser = argparse.ArgumentParser(description="测评：算指标，写 CSV 和 Markdown 报告（详见 docs/guides/evaluation.md）")
    sub = parser.add_subparsers(dest="metric", required=True, metavar="指标")
    for name, text in METRIC_HELP.items():
        sub.add_parser(name, parents=[common], help=text, description=text)
    return parser


def parse_methods(values: list[str]) -> dict[str, str]:
    """把 ["denoise=g1", "vad=g1"] 变成 {"denoise": "g1", "vad": "g1"}，并检查槽位和做法都存在。

    写法不对、槽位或做法不存在时抛 ValueError（中文信息里列出可用的槽位或做法）。
    """
    from pipeline.evaluation import check_methods

    methods = {}
    for value in values or []:
        slot, sep, name = value.partition("=")
        slot, name = slot.strip(), name.strip()
        if not sep or not slot or not name:
            raise ValueError(f"--method 的写法是“槽位=做法”，如 --method denoise=g1，现在写的是“{value}”")
        methods[slot] = name
    check_methods(methods)
    return methods


def _print_methods(methods: dict) -> None:
    if methods:
        print("换了做法：" + "，".join(f"{slot}={name}" for slot, name in methods.items())
              + "（其余槽位按 config.yaml 的 methods）")
    else:
        print("做法：全部按 config.yaml 的 methods")


def _hint_hotword_off(cfg: dict, methods: dict, always: bool = False) -> None:
    """热词纠错开关关着时提醒一句：always 为真（hotwords 指标）、换了热词纠错的做法、
    或者 config.yaml 里热词纠错用的不是基线时才提醒。"""
    if (cfg.get("hotword") or {}).get("enabled"):
        return
    name = methods.get("hotword") or (cfg.get("methods") or {}).get("hotword") or "baseline"
    if always or "hotword" in methods or name != "baseline":
        print(HOTWORD_OFF_HINT)


def _pool_root(args, cfg) -> Path:
    return Path(args.pool) if args.pool else Path(cfg["paths"]["data_pool"])


def _print_overview(summary: dict, keys: list[tuple[str, str]]) -> None:
    """在屏幕上打印汇总：每个分组项一行。keys 是 [(字段, 中文名)]。"""
    for row in summary["overview"]:
        values = "，".join(f"{title} {row[key]:.4f}" if isinstance(row[key], float) else f"{title} {row[key]}"
                          for key, title in keys)
        print(f"  {row['分组项']}：{values}")


def _write(out: Path, name: str, rows, summary, notes) -> int:
    """写报告；写不进去时给中文提示。成功返回 0，失败返回 1。"""
    from pipeline.evaluation import write_report

    try:
        csv_path, md_path = write_report(out, name, rows, summary, notes)
    except PermissionError as e:
        print(f"报告写不进去：{e.filename or out}。如果这个文件正用 Excel 或别的软件打开着，请先关闭再重新运行。")
        return 1
    print(f"\n报告：{md_path}")
    print(f"表格：{csv_path}")
    return 0


# ======================== 每个指标一个函数 ========================


def run_cer(args, cfg: dict, methods: dict, out: Path) -> int:
    """字错率：每段录音识别（测评模式）→ 热词纠错 → 和参考文本比。"""
    from pipeline.evaluation import CER_NOTES, eval_cer

    root = _pool_root(args, cfg)
    print("== 测评：字错率（cer） ==")
    print(f"数据池：{root}")
    _print_methods(methods)
    _hint_hotword_off(cfg, methods)
    try:
        rows, summary = eval_cer(root, cfg, methods, args.files, progress=_say, use_cache=not args.no_cache)
    except (ValueError, FileNotFoundError) as e:
        print(e)
        return 1
    except OSError as e:
        print(f"无法访问数据池文件夹：{e}。请检查路径是否写对、U 盘或共享文件夹是否连上。")
        return 1
    print("\n汇总（字错率越低越好）：")
    _print_overview(summary, [("cer", "字错率"), ("files", "录音数")])
    return _write(out, "cer", rows, summary, CER_NOTES)


def run_hotwords(args, cfg: dict, methods: dict, out: Path) -> int:
    """专名正确率、过度纠正：识别流程同字错率，再按热词表和说错名称表数次数。"""
    from pipeline.evaluation import HOTWORD_NOTES, eval_hotwords

    root = _pool_root(args, cfg)
    print("== 测评：专名正确率与过度纠正（hotwords） ==")
    print(f"数据池：{root}")
    _print_methods(methods)
    _hint_hotword_off(cfg, methods, always=True)
    try:
        rows, summary = eval_hotwords(root, cfg, methods, args.files, progress=_say, use_cache=not args.no_cache)
    except (ValueError, FileNotFoundError) as e:
        print(e)
        return 1
    except OSError as e:
        print(f"无法访问数据池文件夹：{e}。请检查路径是否写对、U 盘或共享文件夹是否连上。")
        return 1
    print("\n汇总（专名正确率越高越好，过度纠正越少越好）：")
    _print_overview(summary, [("name_accuracy", "专名正确率"), ("overcorrected", "过度纠正次数")])
    return _write(out, "hotwords", rows, summary, HOTWORD_NOTES)


def run_numbers(args, cfg: dict, methods: dict, out: Path) -> int:
    """数字提取：在剧本台词文字上测，不用录音。"""
    from pipeline.evaluation import NUMBERS_NOTES, eval_numbers_lines

    print("== 测评：数字提取正确率（numbers），在剧本台词文字上测 ==")
    _print_methods(methods)
    if args.pool or args.files:
        print("提示：这个指标在剧本台词文字上测，用不到 --pool 和 --files。")
    try:
        result = eval_numbers_lines(cfg, methods)
    except ValueError as e:  # 如 config.yaml 里的做法名写错了
        print(e)
        return 1
    print("\n汇总（正确率 = 命中 ÷ 提取，召回率 = 命中 ÷ 金标准）：")
    _print_overview(result, [("precision", "正确率"), ("recall", "召回率"), ("gold", "金标准个数")])
    return _write(out, "numbers", result["overview"], result, NUMBERS_NOTES)


def run_speakers(args, cfg: dict, methods: dict, out: Path) -> int:
    """说话人标错的时长比例（第 3 组）。TODO：后续任务加入。"""
    print(LATER_MESSAGE)
    return 2


def run_classify(args, cfg: dict, methods: dict, out: Path) -> int:
    """话术分类的准确率、召回率、误报率（第 6、7、8 组）。TODO：后续任务加入。"""
    print(LATER_MESSAGE)
    return 2


def run_clips(args, cfg: dict, methods: dict, out: Path) -> int:
    """疑似片段起止误差（第 8 组）。TODO：后续任务加入。"""
    print(LATER_MESSAGE)
    return 2


# 子命令 → 处理函数
RUNNERS = {
    "cer": run_cer,
    "hotwords": run_hotwords,
    "numbers": run_numbers,
    "speakers": run_speakers,
    "classify": run_classify,
    "clips": run_clips,
}


def _say(message: str) -> None:
    """打印进度，马上显示出来（不等缓冲区满）。"""
    print(message, flush=True)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # 命令行窗口显示不了的字用 ? 代替，不让程序出错

    try:
        methods = parse_methods(args.method)
    except ValueError as e:
        print(e)
        return 2

    from pipeline.config import load_config

    overrides = {}
    if args.hotword:
        overrides["hotword"] = {"enabled": args.hotword == "on"}
    cfg = load_config(overrides=overrides)
    out = Path(args.out) if args.out else DEFAULT_OUT
    try:
        return RUNNERS[args.metric](args, cfg, methods, out)
    except KeyboardInterrupt:
        print(STOPPED_MESSAGE)
        return 1


if __name__ == "__main__":
    sys.exit(main())
