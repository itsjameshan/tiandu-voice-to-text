"""选做实验 · MFCC 对照：同一段录音，用 numpy 和 TensorFlow 各算一遍 MFCC，看结果是不是一样。

用法（在项目文件夹里运行）：
    python tools/tf_lab/mfcc_compare.py 录音.wav                  图片存到 outputs/tf_lab/录音_mfcc_compare.png
    python tools/tf_lab/mfcc_compare.py 录音.m4a --out 图.png      指定图片保存位置

没装 TensorFlow 的 Python（例如便携包的 Python）也能运行，只算、只画 numpy 版；
用机房自带的 Python 运行（装了 TensorFlow），就能看到两种算法的对照。
本脚本还要用 matplotlib（画图）、soundfile（读录音）、PyYAML（读 config.yaml）；
机房的 Python 缺哪个，脚本会提示，按提示 pip install 即可（例如 pip install matplotlib soundfile PyYAML）。

MFCC（梅尔频率倒谱系数）是什么：把一小段声音（25 毫秒的一"帧"）压缩成 13 个数，
很多识别模型（包括本实验的小网络）都用它做输入。两种算法的步骤一一对应（对应教材项目 2）：

    步骤                         numpy 版（pipeline/features.py 的 mfcc）   TensorFlow 版（本文件的 tf_mfcc）
    1. 预加重 y[n]=x[n]-0.97x[n-1]  numpy 数组相减                            TensorFlow 张量相减
    2. 分帧、加汉明窗（400 点、移 160 点） sliding_window_view × np.hamming       tf.signal.stft 里一起做
    3. 傅里叶变换、算能量         np.fft.rfft                                 tf.signal.stft
    4. 40 个梅尔三角滤波器         自己写的 _mel_filterbank                     tf.signal.linear_to_mel_weight_matrix
    5. 取对数                     np.log                                      tf.math.log
    6. DCT-II 取前 13 个           自己写的 _dct_matrix                         tf.signal.mfccs_from_log_mel_spectrograms

两种算法的小差别（所以结果很像，但不是一模一样）：
    - TensorFlow 的梅尔滤波器在"梅尔刻度"上画三角形，numpy 版在 Hz 上画，形状略有不同；
    - TensorFlow 的第 0 维比 numpy 版大 √2 倍（两边 DCT 的"归一化"写法不同）。
所以对每一维分别算相关系数（只看起伏像不像，不看大小），再取 13 维的平均。越接近 1 越像。

基线做法：
    上面的 6 步；图上每一维先减平均值、除以标准差再画（和 tools/show_spectrogram.py 的 MFCC 图一样），
    红色表示比平时大、蓝色表示比平时小。
可改进方向：
    - 把 13 维改成 20 维、40 个滤波器改成 80 个，看图有什么变化；
    - 不做预加重，或者把汉明窗换成汉宁窗（tf.signal.hann_window），相关系数会变多少；
    - 画出每一维各自的相关系数，找出差别最大的是哪几维、为什么。
测评指标：
    两种算法的平均相关系数（同一段录音，一般在 0.99 以上）。

注意：TensorFlow、matplotlib 都只在函数里导入；图只存成 PNG，不弹窗口。
"""
import argparse
import functools
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 和 pipeline/features.py 的 mfcc 用同样的参数
N_MFCC = 13
N_MELS = 40
N_FFT = 400  # 每帧 400 个采样点（25 毫秒）
HOP = 160  # 每隔 160 个采样点（10 毫秒）取一帧
EPS = 1e-10  # 取对数前加的一个很小的数，避免 log(0)

# 缺少某个库时提示怎么装：导入时的名字 → pip install 时的名字（只列两者不一样的）
PIP_NAMES = {"yaml": "PyYAML"}


def tf_mfcc(samples, sr, n_mfcc=N_MFCC, n_mels=N_MELS, n_fft=N_FFT, hop=HOP) -> np.ndarray:
    """用 TensorFlow 的 tf.signal 算 MFCC，返回 [帧数, n_mfcc] 的 numpy 数组。没装 TensorFlow 时报 ImportError。"""
    import tensorflow as tf

    x = tf.constant(np.asarray(samples, dtype=np.float32))
    x = tf.concat([x[:1], x[1:] - 0.97 * x[:-1]], axis=0)  # 1. 预加重
    if x.shape[0] < n_fft:  # 录音比一帧还短：后面补 0 凑够一帧（和 numpy 版一样）
        x = tf.pad(x, [[0, n_fft - x.shape[0]]])
    # 2、3. 分帧、加汉明窗、傅里叶变换。periodic=False 是"对称"的汉明窗，和 np.hamming 一样
    window = functools.partial(tf.signal.hamming_window, periodic=False)
    stft = tf.signal.stft(x, frame_length=n_fft, frame_step=hop, fft_length=n_fft, window_fn=window)
    power = tf.abs(stft) ** 2 / n_fft  # 能量谱 [帧数, 201]
    # 4. 梅尔滤波：[201 个频率格, 40 个滤波器] 的矩阵，频率范围 0 到 sr/2（和 numpy 版一样）
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels, num_spectrogram_bins=n_fft // 2 + 1, sample_rate=sr,
        lower_edge_hertz=0.0, upper_edge_hertz=sr / 2)
    mel_energy = tf.matmul(power, mel_matrix)  # [帧数, 40]
    log_mel = tf.math.log(mel_energy + EPS)  # 5. 取对数
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[:, :n_mfcc]  # 6. DCT-II 取前 13 个
    return mfccs.numpy()


def mean_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """两组 MFCC（[帧数, 维数]）每一维分别算相关系数，再取平均。

    相关系数在 -1 到 1 之间：1 表示两条曲线起伏完全一致（不管整体放大多少倍），0 表示没关系。
    某一维在整段录音里几乎不变（算不出相关系数）时跳过这一维。
    """
    frames = min(len(a), len(b))
    values = []
    for j in range(a.shape[1]):
        x, y = a[:frames, j], b[:frames, j]
        if x.std() < EPS or y.std() < EPS:
            continue
        values.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def _standardize(feats: np.ndarray) -> np.ndarray:
    """每一维减去平均值、除以标准差（画图用：红色偏大、蓝色偏小）。返回 [维数, 帧数]，方便画成"横轴是时间"。"""
    feats = feats.T
    return (feats - feats.mean(axis=1, keepdims=True)) / (feats.std(axis=1, keepdims=True) + EPS)


def plot_compare(results: list[tuple[str, np.ndarray]], sr: int, path: Path, title: str) -> None:
    """把一组或两组 MFCC 上下排画在一张图里，存成 PNG。results 是 [(小标题, [帧数, 维数] 的 MFCC)]。"""
    import matplotlib

    matplotlib.use("Agg")  # 只画图片、不弹窗口
    from matplotlib.figure import Figure

    from pipeline.features import _use_chinese_font  # 找一个中文字体，图上的中文才不会变成方框

    _use_chinese_font()
    fig = Figure(figsize=(10, 3.2 * len(results) + 0.6), layout="constrained")
    axes = fig.subplots(len(results), 1, sharex=True, squeeze=False)[:, 0]
    fig.suptitle(title)
    for ax, (name, feats) in zip(axes, results):
        duration = (len(feats) * HOP + N_FFT) / sr
        ax.imshow(_standardize(feats), origin="lower", aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
                  extent=[0, duration, -0.5, feats.shape[1] - 0.5], interpolation="nearest")
        ax.set_title(name, fontsize=10, loc="left")
        ax.set_ylabel("第几维")
    axes[-1].set_xlabel("时间（秒）")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # 命令行窗口显示不了的字用 ? 代替，不让程序因此出错

    parser = argparse.ArgumentParser(description="同一段录音用 numpy 和 TensorFlow 各算一遍 MFCC，画图对照。")
    parser.add_argument("audio", help="录音文件（任何格式）")
    parser.add_argument("--out", help="图片保存路径（默认：outputs/tf_lab/<文件名>_mfcc_compare.png）")
    args = parser.parse_args(argv)

    from pipeline.audio import SR, convert_to_wav, read_wav
    from pipeline.config import load_config
    from pipeline.features import mfcc

    src = Path(args.audio)
    if not src.is_file():
        print(f"找不到录音文件：{src}")
        return 1

    try:
        cfg = load_config()
        out = Path(args.out) if args.out else Path(cfg["paths"]["outputs"]) / "tf_lab" / f"{src.stem}_mfcc_compare.png"
        if not out.suffix:
            out = out.with_suffix(".png")
        tmp_root = Path(cfg["paths"]["tmp"])  # 项目内的 tmp/：避免临时文件落到含中文的系统临时文件夹
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_dir:
            wav = Path(tmp_dir) / "audio_16k.wav"
            convert_to_wav(src, wav)
            samples = read_wav(wav)
        numpy_feats = mfcc(samples, SR, n_mfcc=N_MFCC, n_mels=N_MELS, n_fft=N_FFT, hop=HOP)
    except ImportError as e:  # 机房的 Python 可能没装 PyYAML（读 config.yaml 用）或 soundfile（读 WAV 用）
        print(f"缺少 {e.name}：请先运行 pip install {PIP_NAMES.get(e.name, e.name)} 再试。")
        return 1
    except Exception as e:  # noqa: BLE001  命令行工具：把错误原因用中文打印出来，不打印一大串报错
        print(f"处理失败：{e}")
        return 1
    print(f"录音：{src}（{len(samples) / SR:.1f} 秒）")
    print(f"numpy 版 MFCC：{numpy_feats.shape[0]} 帧 × {numpy_feats.shape[1]} 维")
    results = [("numpy 版（pipeline/features.py 的 mfcc）", numpy_feats)]

    try:
        tf_feats = tf_mfcc(samples, SR)
    except ImportError:
        tf_feats = None
        print("这个 Python 没装 TensorFlow，只算、只画 numpy 版。想看对照：用机房自带的 Python 运行本脚本。")
    if tf_feats is not None:
        corr = mean_correlation(numpy_feats, tf_feats)
        print(f"TensorFlow 版 MFCC：{tf_feats.shape[0]} 帧 × {tf_feats.shape[1]} 维")
        print(f"两种算法的相关系数（13 维的平均）：{corr:.4f}（越接近 1 越像）")
        results.append(("TensorFlow 版（tf.signal）", tf_feats))

    try:
        plot_compare(results, SR, out, title=f"{src.name} 的 MFCC")
    except ImportError as e:
        print(f"画图失败：缺少 {e.name}。请先运行 pip install matplotlib 再试。")
        return 1
    print(f"图片已保存：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
