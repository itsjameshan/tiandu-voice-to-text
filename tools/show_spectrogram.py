"""画一段录音的波形、语谱图和 MFCC，存成 PNG 图片（原理课、语谱图质检作业用）。

用法：
    python tools/show_spectrogram.py 录音.m4a                  图片存到 outputs/spectrograms/录音_spectrogram.png
    python tools/show_spectrogram.py 录音.wav --out 图.png     指定图片保存位置

任何格式（wav、mp3、m4a、视频……）都可以：先用 ffmpeg 转成 16000 Hz 单声道 WAV
（临时文件放在项目的 tmp/ 文件夹里，用完自动删除），再画图。
同时打印录音时长、原始采样率和质检结果（"质检：合格"，或发现的问题，如"音量太小：整体音量 …… dBFS"），对照图看；
这里不和剧本预计时长比较（文件名不一定是录音编号）。
全部成功时退出码为 0，出错时为 1。
"""
import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="画录音的波形、语谱图和 MFCC，存成 PNG 图片。")
    parser.add_argument("audio", help="录音或视频文件（任何格式）")
    parser.add_argument("--out", help="图片保存路径（默认：outputs/spectrograms/<文件名>_spectrogram.png）")
    args = parser.parse_args(argv)

    from pipeline.audio import SR, convert_to_wav, probe, read_wav
    from pipeline.config import load_config
    from pipeline.features import plot_recording
    from pipeline.step1_ingest import quality_check

    src = Path(args.audio)
    if not src.is_file():
        print(f"找不到录音文件：{src}")
        return 1
    cfg = load_config()
    # 默认存到 outputs/ 下，不往录音旁边写文件（录音可能在只读的数据池 raw 文件夹里）
    out = Path(args.out) if args.out else Path(cfg["paths"]["outputs"]) / "spectrograms" / f"{src.stem}_spectrogram.png"
    if not out.suffix:
        out = out.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(cfg["paths"]["tmp"])  # 项目内的 tmp/：避免临时文件落到含中文的系统临时文件夹
    tmp_root.mkdir(parents=True, exist_ok=True)
    try:
        original_sr = probe(src)["sample_rate"]
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_dir:
            wav = Path(tmp_dir) / "audio_16k.wav"
            convert_to_wav(src, wav)
            samples = read_wav(wav)
        plot_recording(samples, SR, path=out, title=src.name)
        problems = quality_check(samples, original_sr, cfg)
    except Exception as e:  # noqa: BLE001  命令行工具：把错误原因用中文打印出来，不打印一大串报错
        print(f"处理失败：{e}")
        return 1

    print(f"录音：{src}（{len(samples) / SR:.1f} 秒，原始采样率 {original_sr} Hz）")
    print(f"图片已保存：{out}")
    if problems:
        print("质检发现的问题：")
        for p in problems:
            print(f"  - {p}")
    else:
        print("质检：合格")
    return 0


if __name__ == "__main__":
    sys.exit(main())
