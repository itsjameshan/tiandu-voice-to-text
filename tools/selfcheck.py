"""便携包自检：检查这台电脑能不能正常运行本工具。

用法（Windows 便携包里双击 selfcheck.bat，或在命令行运行）：
    python tools/selfcheck.py           完整自检（约 1 分钟，会用模型自带的测试音频跑一遍完整流程）
    python tools/selfcheck.py --quick   只检查环境，不跑识别

检查项目：Python 版本、安装路径是否含中文、ffmpeg、必需模型、完整流程、导出核查初稿。
测试音频是模型自带的公开测试音频，只用于自检，不会出现在网页界面里。
全部通过时退出码为 0。
"""
import argparse
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _print(ok: bool, name: str, detail: str = "") -> bool:
    mark = "通过  " if ok else "不通过"
    print(f"[{mark}] {name}" + (f"：{detail}" if detail else ""), flush=True)
    return ok


def check_python() -> bool:
    v = sys.version_info
    return _print(v >= (3, 10), "Python 版本", f"{v.major}.{v.minor}.{v.micro}（需要 3.10 及以上）")


def check_path() -> bool:
    from pipeline.audio import has_non_ascii

    if has_non_ascii(ROOT):
        # 只警告，不算失败
        print(f"[警告  ] 安装路径含中文或特殊字符：{ROOT}\n"
              f"         模型可能加载失败。请把整个文件夹移到纯英文路径，例如 D:\\asr\\tiandu", flush=True)
    else:
        _print(True, "安装路径", str(ROOT))
    return True


def check_ffmpeg() -> bool:
    from pipeline.audio import find_ffmpeg

    try:
        return _print(True, "ffmpeg", find_ffmpeg())
    except Exception as e:  # noqa: BLE001  自检要把所有错误都显示出来
        return _print(False, "ffmpeg", str(e))


def check_models(cfg) -> bool:
    from pipeline.models import missing_models

    missing = missing_models(cfg)
    if missing:
        return _print(False, "模型文件", "缺少 " + "、".join(missing) + "；请运行 python models/download_models.py")
    return _print(True, "模型文件", "必需模型都在")


def check_pipeline(cfg) -> bool:
    from pipeline import run_pipeline
    from pipeline.models import model_path
    from pipeline.step8_report import export_bundle

    wav = model_path(cfg, "test_wav")
    try:
        # 原始文件会被设为只读，所以清理时忽略删除失败（Windows 上删除只读文件会报错）
        with tempfile.TemporaryDirectory(dir=cfg["paths"]["tmp"], ignore_cleanup_errors=True) as tmp:
            t0 = time.time()
            segments, meta = run_pipeline(str(wav), {"num_speakers": 4, "out_dir": str(Path(tmp) / "run")}, cfg=cfg)
            used = time.time() - t0
            ok = len(segments) >= 5 and all(s.get("text") for s in segments)
            _print(ok, "完整流程（测试音频）",
                   f"{len(segments)} 段，音频 {meta['duration']:.1f} 秒，用时 {used:.1f} 秒，实时率 {meta['rtf']:.2f}")
            if not ok:
                return False
            zip_path = export_bundle(segments, meta, str(Path(tmp) / "export"))
            return _print(Path(zip_path).is_file(), "导出核查初稿", Path(zip_path).name)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return _print(False, "完整流程（测试音频）", str(e))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="便携包自检")
    parser.add_argument("--quick", action="store_true", help="只检查环境，不跑识别")
    args = parser.parse_args(argv)

    print("== 旅游纠纷录音材料整理 · 自检 ==", flush=True)
    from pipeline.config import load_config

    cfg = load_config()
    Path(cfg["paths"]["tmp"]).mkdir(parents=True, exist_ok=True)
    results = [check_python(), check_path(), check_ffmpeg(), check_models(cfg)]
    if not args.quick and all(results):
        results.append(check_pipeline(cfg))
    if all(results):
        print("== 全部通过。双击 start.bat（或运行 python app.py）启动网页工具。 ==", flush=True)
        return 0
    print("== 有项目没有通过，请按上面的提示处理；解决不了请截图发给老师。 ==", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
