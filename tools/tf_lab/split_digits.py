"""选做实验 · 第 1 步：把"零到九念 5 遍"的录音自动切成 50 段，按数字分好文件夹。

用法（在项目文件夹里运行，用便携包的 Python 或装好 requirements.txt 的 Python）：
    python tools/tf_lab/split_digits.py digits-0123.m4a --speaker 0123
    python tools/tf_lab/split_digits.py digits-0123.m4a --speaker 0123 --out D:/digits    指定存到哪里

参数：
    录音文件     任何格式（m4a、mp3、wav、视频……），先用 ffmpeg 转成 16000 Hz 单声道 WAV
                （临时文件放在项目的 tmp/ 文件夹里，用完自动删除）
    --speaker   学号后四位（只能是数字或英文字母，不要写姓名），用作文件名
    --out       存到哪个文件夹，默认是数据池里的 digits 文件夹（config.yaml 的 paths.data_pool 下面）

怎么录（详见 docs/guides/recording.md 第六节）：
    安静环境，手机离嘴约 30 厘米，按顺序念"零、一、二、三、四、五、六、七、八、九"，
    每个字之间停顿约 1 秒，连念 5 遍，中间不要说别的话。

做了什么：
    1. 转格式：任何格式 → 16000 Hz、单声道、16 位 WAV；
    2. 端点检测：用 Silero VAD（和步骤 2 同一个模型）找出"哪几段有人在说话"。
       这里把两个参数改得更适合念单个字：
         min_speech_duration = 0.1   一个字很短（约 0.3 秒），有 0.1 秒的人声就算一段；
         min_silence_duration = 0.4  停顿超过 0.4 秒才算一个字念完（所以每个字之间要停顿约 1 秒）；
    3. 数段数：必须正好 50 段（10 个数字 × 5 遍）。
       不是 50 段时：打印每一段的时长，提示"念慢一点、每个字之间停顿约 1 秒，重录"，退出码 1，不写任何文件；
       是 50 段时：第 1 段是"零"、第 2 段是"一"……第 11 段又是"零"（第 2 遍），
       每段前后各多留 0.1 秒（免得把字头、字尾切掉），存成：
           <out>/<数字>/<学号后四位>_<第几遍>.wav      例如 digits/3/0123_2.wav 是第 2 遍念的"三"
       同一个人重新切一次，会覆盖原来的 50 个文件。

基线做法：
    Silero VAD 切段 + 按顺序编号。只要念的时候停顿够长、中间没有杂音，就能一次切对。
可改进方向：
    - 段数不对时自动调 min_silence_duration 再试几次；
    - 用每段的音量、时长找出"可能切错"的段（比如一个字特别长，可能是两个字连在一起了）；
    - 切好后用 tools/show_spectrogram.py 看几段的语谱图，比较不同数字的样子有什么不同。
测评指标：
    一次切对的比例（段数正好 50 的录音占多少）；随机听几段，看数字和文件夹是否对得上。

隐私：录音和切好的文件只放在本机（机房电脑或老师电脑），不上传网络、不进代码仓库；
文件名只用学号后四位，不用姓名。
"""
import argparse
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 念 5 遍"零到九"，一共 50 个字
DIGITS = 10
REPEATS = 5
EXPECTED_SEGMENTS = DIGITS * REPEATS

# 端点检测参数：只改这两个，其余（threshold 等）沿用 config.yaml 的 vad
VAD_OVERRIDES = {"vad": {"min_speech_duration": 0.1, "min_silence_duration": 0.4}}

# 每段前后各多留多少秒（端点检测可能把轻的字头、字尾切掉，如"四"的 s、"七"的 q）
PAD_SECONDS = 0.1

# 段数不对时的固定提示
ADVICE = "念慢一点、每个字之间停顿约 1 秒，重录"

# --speaker 只能是 1 到 10 个数字或英文字母（学号后四位），不能是姓名
SPEAKER_PATTERN = re.compile(r"[0-9A-Za-z]{1,10}")

# 数字的汉字，打印时用
DIGIT_NAMES = "零一二三四五六七八九"


def find_segments(samples, cfg) -> list[tuple[float, float]]:
    """用 Silero VAD 找出每一段人声，返回 [(开始秒, 结束秒)]，按时间先后排列。"""
    from pipeline.audio import SR
    from pipeline.step2_vad import vad_baseline  # 里面才导入 sherpa_onnx

    return vad_baseline(samples, SR, cfg)


def print_durations(spans) -> None:
    """打印每一段的时长，每行 10 段（正好对应念一遍"零到九"）。"""
    for first in range(0, len(spans), DIGITS):
        group = spans[first:first + DIGITS]
        durations = " ".join(f"{end - start:.2f}" for start, end in group)
        print(f"  第 {first + 1}–{first + len(group)} 段（秒）：{durations}")


def save_segments(samples, spans, speaker: str, out_dir: Path) -> list[Path]:
    """把 50 段按顺序存成 <out_dir>/<数字>/<speaker>_<第几遍>.wav，返回存好的文件列表。

    第 i 段（从 0 数起）是数字 i % 10，是第 i // 10 + 1 遍。
    """
    from pipeline.audio import SR, write_wav

    saved = []
    for i, (start, end) in enumerate(spans):
        digit = i % DIGITS
        rep = i // DIGITS + 1
        a = max(0, int((start - PAD_SECONDS) * SR))
        b = min(len(samples), int((end + PAD_SECONDS) * SR))
        path = out_dir / str(digit) / f"{speaker}_{rep}.wav"
        write_wav(path, samples[a:b], SR)
        saved.append(path)
    return saved


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # 命令行窗口显示不了的字用 ? 代替，不让程序因此出错

    parser = argparse.ArgumentParser(description="把'零到九念 5 遍'的录音切成 50 段，按数字分文件夹存好。")
    parser.add_argument("audio", help="录音文件（任何格式）")
    parser.add_argument("--speaker", required=True, help="学号后四位（只能是数字或英文字母，不要写姓名）")
    parser.add_argument("--out", help="存到哪个文件夹（默认：数据池里的 digits 文件夹）")
    args = parser.parse_args(argv)

    speaker = args.speaker.strip()
    if not SPEAKER_PATTERN.fullmatch(speaker):
        print(f"--speaker 写的是“{args.speaker}”：只能用学号后四位（数字或英文字母，例如 0123），不要写姓名。")
        return 1
    src = Path(args.audio)
    if not src.is_file():
        print(f"找不到录音文件：{src}")
        return 1

    from pipeline.audio import SR, convert_to_wav, read_wav
    from pipeline.config import load_config

    cfg = load_config(overrides=VAD_OVERRIDES)
    out_dir = Path(args.out) if args.out else Path(cfg["paths"]["data_pool"]) / "digits"
    tmp_root = Path(cfg["paths"]["tmp"])  # 项目内的 tmp/：避免临时文件落到含中文的系统临时文件夹
    tmp_root.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_dir:
            wav = Path(tmp_dir) / "digits_16k.wav"
            convert_to_wav(src, wav)
            samples = read_wav(wav)
        spans = find_segments(samples, cfg)
    except Exception as e:  # noqa: BLE001  命令行工具：把错误原因用中文打印出来，不打印一大串报错
        print(f"处理失败：{e}")
        return 1

    print(f"录音：{src}（{len(samples) / SR:.1f} 秒）")
    print(f"端点检测找到 {len(spans)} 段，需要正好 {EXPECTED_SEGMENTS} 段（"
          f"“{DIGIT_NAMES}”念 {REPEATS} 遍）。")
    print_durations(spans)

    if len(spans) != EXPECTED_SEGMENTS:
        if len(spans) == 0:
            print("没有找到人声：录音可能是空的、音量太小，或者选错了文件。")
        elif len(spans) < EXPECTED_SEGMENTS:
            print("段数少了：可能有两个字连在一起（停顿太短），或者有的字声音太小没检测到。")
        else:
            print("段数多了：可能有杂音、咳嗽、说了别的话，或者一个字被切成了两段。")
        print(f"提示：{ADVICE}。")
        print("没有保存任何文件。")
        return 1

    try:
        saved = save_segments(samples, spans, speaker, out_dir)
    except Exception as e:  # noqa: BLE001
        print(f"保存失败：{e}")
        return 1
    long_ones = [i + 1 for i, (start, end) in enumerate(spans) if end - start > 1.0]
    if long_ones:
        print(f"注意：第 {'、'.join(map(str, long_ones))} 段超过 1 秒，训练时只取前 1 秒；可以听一听是不是两个字连在一起了。")
    print(f"已保存 {len(saved)} 段到 {out_dir}（每个数字一个文件夹，文件名 {speaker}_第几遍.wav；同名文件已覆盖）。")
    print("建议随便点开几段听一听，确认数字和文件夹对得上。录音只放在本机，不要上传到网上。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
