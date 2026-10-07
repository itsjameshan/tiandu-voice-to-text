"""长录音测试：把模型自带的四人测试音频重复拼接成长录音，跑一遍完整流程，记录每一步的用时、实时率和最大内存。

用法：
    python tools/long_audio_test.py --minutes 60            拼成约 60 分钟再测（较慢，单独运行，不放进默认的 pytest）
    python tools/long_audio_test.py --minutes 10 --log      测完把结果追加到 docs/progress.md

说明：
    - 测试音频是模型自带的公开测试音频，只用于自动测试，不会出现在网页界面里；不用语音合成（红线第 2 条）。
    - 拼接出来的长录音是同一段话的重复，说话人分离会把重复的同一个人归到一起，所以只用来测速度和内存，不测准确率。
    - 处理速度和电脑有关；这里测出来的数字只代表跑测试的这台电脑，机房和办公电脑要另测，实测之前不要对外说处理时长。
"""
import argparse
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def peak_memory_mb() -> float | None:
    """本进程到目前为止用过的最大内存（MB）。Linux/Mac 用 resource；Windows 上有 psutil 才能测，没有就返回 None。"""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux 的单位是 KB，Mac 是字节
        return peak / 1024 / 1024 if sys.platform == "darwin" else peak / 1024
    except ImportError:
        try:
            import psutil

            return psutil.Process().memory_info().peak_wset / 1024 / 1024
        except Exception:  # noqa: BLE001  测不了内存不影响测速度
            return None


def make_long_audio(src: Path, minutes: float, out: Path) -> Path:
    """用 ffmpeg 把 src 重复拼接到约 minutes 分钟，写成 16kHz 单声道 WAV。"""
    import soundfile as sf

    from pipeline.audio import find_ffmpeg

    seconds = sf.info(str(src)).duration
    repeat = max(1, round(minutes * 60 / seconds))
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [find_ffmpeg(), "-y", "-loglevel", "error", "-stream_loop", str(repeat - 1), "-i", str(src),
           "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(out)]
    subprocess.run(cmd, check=True)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="长录音测试：测每一步的用时、实时率和最大内存")
    parser.add_argument("--minutes", type=float, default=60, help="拼成多少分钟（默认 60）")
    parser.add_argument("--speakers", type=int, default=4, help="说话人数（测试音频是 4 人）")
    parser.add_argument("--log", action="store_true", help="把结果追加到 docs/progress.md")
    args = parser.parse_args(argv)

    from pipeline import run_pipeline
    from pipeline.audio import remove_tree
    from pipeline.config import load_config
    from pipeline.models import model_path

    cfg = load_config()
    tmp = Path(cfg["paths"]["tmp"]) / "long_audio_test"
    remove_tree(tmp)
    src = model_path(cfg, "test_wav")
    print(f"== 长录音测试：拼成约 {args.minutes:g} 分钟 ==", flush=True)
    wav = make_long_audio(Path(src), args.minutes, tmp / f"long_{args.minutes:g}min.wav")

    t0 = time.time()
    segments, meta = run_pipeline(str(wav), {"num_speakers": args.speakers, "out_dir": str(tmp / "run")}, cfg=cfg,
                                  progress=lambda f, d: print(f"  {f:5.0%} {d}", flush=True))
    total = time.time() - t0
    mem = peak_memory_mb()
    mem_text = f"最大内存 {mem:.0f} MB" if mem else "最大内存未测（Windows 上需要 psutil）"

    lines = [
        f"- 长录音测试（{datetime.now():%Y-%m-%d %H:%M}，{platform.system()}，{platform.processor() or platform.machine()}，"
        f"Python {platform.python_version()}，识别线程数 {cfg['asr']['num_threads']}）：",
        f"  录音 {meta['duration'] / 60:.1f} 分钟（四人测试音频重复拼接），{len(segments)} 段；"
        f"总用时 {total:.1f} 秒（{total / 60:.1f} 分钟），实时率 {total / meta['duration']:.3f}；{mem_text}",
        "  各步用时（秒）：" + "，".join(f"{name} {sec}" for name, sec in meta["timings"].items()),
    ]
    print("\n".join(lines), flush=True)
    if args.log:
        with open(ROOT / "docs" / "progress.md", "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(lines) + "\n")
        print("已追加到 docs/progress.md")
    remove_tree(tmp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
