"""语谱图与 MFCC：把声音"画出来"（原理课、"录音质检"标签页、tools/show_spectrogram.py 用）。

一段录音就是一长串数字（每秒 16000 个采样点）。直接看这串数字看不出什么，
所以把它切成很多很短的"帧"，每帧算一下里面有哪些频率的声音、各有多强：
    帧：每帧 400 个采样点（25 毫秒），每隔 160 个采样点（10 毫秒）取一帧，相邻帧有重叠；
    语谱图：横轴是时间，纵轴是频率，颜色深浅表示这个时刻、这个频率的声音有多强（单位 dB）；
    MFCC（梅尔频率倒谱系数）：把每帧的频谱压缩成 13 个数，是很多识别模型的输入特征。

从语谱图上能看出录音质量问题：
    - 噪声：没人说话的地方也有一片灰（嘈杂教室的录音尤其明显）；
    - 削波：声音太大，波形顶到上下两条虚线，被"削平"；
    - 口袋录音、远距离录音：高频（图的上半部分）明显变淡，声音发闷。

基线做法（只用 numpy 和 matplotlib）：
    spectrogram_db：分帧 → 乘汉明窗 → 傅里叶变换（rfft）→ 能量取 10×log10 换成 dB。
    mfcc：
        1. 预加重：y[n] = x[n] - 0.97 × x[n-1]，把高频提上来一些；
        2. 分帧，每帧乘汉明窗（让帧两头平滑地变小，减少频谱"泄漏"）；
        3. 傅里叶变换，算每个频率的能量；
        4. 梅尔滤波：用 40 个三角形滤波器把能量按"梅尔刻度"分组相加——
           人耳对低频的变化更敏感，所以低频的滤波器窄、高频的宽；
        5. 取对数（人耳对音量的感觉也接近对数）；
        6. DCT-II（离散余弦变换），只留前 13 个数。
    plot_recording：三张图上下排：波形、语谱图、MFCC。
可改进方向：
    - 加上一阶、二阶差分（delta），表示特征随时间的变化；
    - 画出每帧的音量曲线，标出质检报出的长静音、削波位置；
    - 第 1、2 组可以把降噪前后、近处和口袋录音的语谱图并排对比。
测评指标：
    不直接测识别效果；tests/test_features.py 检查：1 秒录音约 98 帧、MFCC 13 维、
    440 Hz 正弦波在语谱图上 440 Hz 附近最亮、能画出 PNG 图片。

注意：matplotlib 在 plot_recording 里才导入，并且只用不弹窗口的 Agg 方式画图
（Windows 便携包里的 Python 没有 tkinter，弹不出窗口）。
"""
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 中文字体候选，按顺序找第一个装了的：
# Windows 自带微软雅黑、黑体；Linux 常见 Noto Sans CJK、文泉驿正黑；苹果电脑是苹方
CHINESE_FONTS = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC"]

# 画图时最多画多少帧：屏幕也就这么宽，再多也看不出来，还占内存、画得慢
MAX_PLOT_FRAMES = 4000

# 算对数前加的一个很小的数，避免 log(0) 得到负无穷（全静音的帧）
EPS = 1e-10


def _to_float_mono(samples) -> np.ndarray:
    """把录音整理成一维的 float64 数组：整数采样（如 16 位的 -32768~32767）换算到 -1~1，多声道取平均。"""
    x = np.asarray(samples)
    if np.issubdtype(x.dtype, np.integer):
        x = x / float(np.iinfo(x.dtype).max)
    x = x.astype(np.float64)
    if x.ndim == 2:  # soundfile 读多声道是 [采样点, 声道]
        x = x.mean(axis=1)
    return x


def _frames(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """分帧：第 i 帧是 x[i×hop : i×hop+n_fft]，返回 [帧数, n_fft]。

    录音比一帧还短时，后面补 0 凑够一帧。
    """
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    # sliding_window_view 列出所有"从第 0、1、2……个点开始的 n_fft 个点"，不复制数据；
    # 再每隔 hop 个取一个，就是一帧一帧了
    return np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]


def spectrogram_db(samples, sr, n_fft=400, hop=160) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """算语谱图，返回 (times, freqs, dB)。

    times：每帧中心的时间（秒），长度 = 帧数；
    freqs：每个频率格的频率（Hz），从 0 到 sr/2，长度 = n_fft/2 + 1（默认 201 个，相邻相差 40 Hz）；
    dB：形状 [频率格数, 帧数]，每个数是这一帧、这个频率的能量（10×log10，单位 dB）。
    """
    x = _to_float_mono(samples)
    frames = _frames(x, n_fft, hop) * np.hamming(n_fft)
    power = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2  # [帧数, 频率格数]
    db = 10.0 * np.log10(power + EPS)

    times = (np.arange(len(frames)) * hop + n_fft / 2) / sr
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    return times, freqs, db.T  # 转置成 [频率, 时间]，画图时频率在纵轴


def _hz_to_mel(hz):
    """频率（Hz）换成梅尔刻度：低频变化大、高频变化小，接近人耳的感觉。"""
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel):
    """梅尔刻度换回频率（Hz）。"""
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_filterbank(sr, n_fft, n_mels) -> np.ndarray:
    """梅尔滤波器组：n_mels 个三角形滤波器，返回 [n_mels, 频率格数]。

    在梅尔刻度上把 0 到 sr/2 等分，第 m 个三角形从第 m 个点升到第 m+1 个点、再降到第 m+2 个点。
    换回 Hz 后，低频的三角形窄、高频的宽。
    """
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    hz_points = _mel_to_hz(np.linspace(0.0, _hz_to_mel(sr / 2), n_mels + 2))
    bank = np.zeros((n_mels, len(fft_freqs)))
    for m in range(n_mels):
        left, center, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        rising = (fft_freqs - left) / (center - left)     # 三角形左边：从 0 升到 1
        falling = (right - fft_freqs) / (right - center)  # 三角形右边：从 1 降到 0
        bank[m] = np.maximum(0.0, np.minimum(rising, falling))
    return bank


def _dct_matrix(n_in, n_out) -> np.ndarray:
    """DCT-II 变换矩阵（正交归一化，与 scipy.fft.dct(type=2, norm="ortho") 相同），形状 [n_out, n_in]。

    第 k 行是一条余弦曲线 cos(π × k × (2n+1) / (2 × n_in))：k 越大，曲线起伏越快。
    """
    n = np.arange(n_in)
    k = np.arange(n_out)[:, None]
    matrix = np.cos(np.pi * k * (2 * n + 1) / (2 * n_in)) * np.sqrt(2.0 / n_in)
    matrix[0] /= np.sqrt(2.0)
    return matrix


def mfcc(samples, sr, n_mfcc=13, n_mels=40, n_fft=400, hop=160) -> np.ndarray:
    """算 MFCC 特征，返回形状 [帧数, n_mfcc] 的数组（1 秒 16 kHz 录音约 98 帧）。

    步骤：预加重 0.97 → 分帧、汉明窗 → 能量谱 → 梅尔滤波 → 取对数 → DCT-II 取前 n_mfcc 个。
    """
    x = _to_float_mono(samples)
    x = np.append(x[:1], x[1:] - 0.97 * x[:-1])                     # 1. 预加重
    frames = _frames(x, n_fft, hop) * np.hamming(n_fft)              # 2. 分帧、加窗
    power = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2 / n_fft        # 3. 能量谱 [帧数, 频率格数]
    mel_energy = power @ _mel_filterbank(sr, n_fft, n_mels).T        # 4. 梅尔滤波 [帧数, n_mels]
    log_mel = np.log(mel_energy + EPS)                               # 5. 取对数
    return log_mel @ _dct_matrix(n_mels, n_mfcc).T                   # 6. DCT-II [帧数, n_mfcc]


def use_chinese_font():
    """在电脑上找一个中文字体给 matplotlib 用，返回字体名；一个都没有时返回 None（中文会显示成方框）。"""
    from matplotlib import font_manager, rcParams

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in CHINESE_FONTS:
        if name in installed:
            others = [f for f in rcParams["font.sans-serif"] if f != name]
            rcParams["font.sans-serif"] = [name, *others]
            rcParams["axes.unicode_minus"] = False  # 中文字体常缺"−"号，负号改用普通减号
            return name
    logger.warning("没有找到中文字体（%s），图上的中文可能显示成方框。", "、".join(CHINESE_FONTS))
    return None


def plot_recording(samples, sr, path=None, title=""):
    """画一段录音的三张图（上下排）：波形、语谱图、MFCC。返回 matplotlib.figure.Figure。

    path：给了就把图存成 PNG 图片（文件夹不存在会自动建）；
    title：图最上方的标题，例如录音文件名。
    长录音会自动加大帧移，最多画 MAX_PLOT_FRAMES 帧（每隔一段取一帧），所以几十分钟的录音也能很快画完。
    """
    import matplotlib

    matplotlib.use("Agg")  # 只画图片、不弹窗口（Windows 便携包的 Python 没有 tkinter）
    # 不用 pyplot，直接建 Figure：pyplot 会把画过的图都记在内存里，网页工具一直开着会越积越多
    from matplotlib.figure import Figure

    use_chinese_font()
    x = _to_float_mono(samples)
    n_fft = 400
    duration = max(len(x), n_fft) / sr  # 比一帧还短（甚至是空的）录音按一帧的长度画，不报错
    hop = max(160, len(x) // MAX_PLOT_FRAMES + 1)  # 帧数 ≈ 采样点数 ÷ 帧移，不超过 MAX_PLOT_FRAMES

    fig = Figure(figsize=(10, 7.5), layout="constrained")
    ax_wave, ax_spec, ax_mfcc = fig.subplots(3, 1, sharex=True)
    if title:
        fig.suptitle(title)

    # 1. 波形：纵轴固定在 -1 到 1，音量小的录音看起来就"矮"；碰到虚线就是削波
    if len(x) <= 2 * MAX_PLOT_FRAMES:
        ax_wave.plot(np.arange(len(x)) / sr, x, color="#2a6fb0", linewidth=0.6)
    else:
        # 点太多时，每一小块只画最高点和最低点之间的竖条（和 Audacity 的画法一样，削波不会被漏掉）
        block = len(x) // MAX_PLOT_FRAMES
        pieces = x[: block * MAX_PLOT_FRAMES].reshape(MAX_PLOT_FRAMES, block)
        t = np.arange(MAX_PLOT_FRAMES) * block / sr
        ax_wave.fill_between(t, pieces.min(axis=1), pieces.max(axis=1), color="#2a6fb0", linewidth=0.6)
    for level in (1.0, -1.0):
        ax_wave.axhline(level, color="#888888", linestyle="--", linewidth=0.8)
    ax_wave.set_ylim(-1.1, 1.1)
    ax_wave.set_ylabel("幅度")
    ax_wave.set_title("波形（碰到虚线说明声音太大、被削波）", fontsize=10, loc="left")

    # 2. 语谱图：颜色越深，这个时刻、这个频率的声音越强；只显示最强处往下 80 dB 的范围
    _, _, db = spectrogram_db(x, sr, n_fft=n_fft, hop=hop)
    top = float(db.max())
    ax_spec.imshow(db, origin="lower", aspect="auto", cmap="Greys", vmin=top - 80, vmax=top,
                   extent=[0, duration, 0, sr / 2], interpolation="nearest")
    ax_spec.set_ylabel("频率（Hz）")
    ax_spec.set_title("语谱图（颜色越深，声音越强；口袋录音上半部分会变淡）", fontsize=10, loc="left")

    # 3. MFCC：每一维先减去平均值、除以标准差，红色表示比平时大、蓝色表示比平时小
    feats = mfcc(x, sr, n_fft=n_fft, hop=hop).T  # [13, 帧数]
    feats = (feats - feats.mean(axis=1, keepdims=True)) / (feats.std(axis=1, keepdims=True) + EPS)
    ax_mfcc.imshow(feats, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
                   extent=[0, duration, -0.5, feats.shape[0] - 0.5], interpolation="nearest")
    ax_mfcc.set_yticks(range(0, feats.shape[0], 2))
    ax_mfcc.set_ylabel("第几维")
    ax_mfcc.set_xlabel("时间（秒）")
    ax_mfcc.set_title(f"MFCC（{feats.shape[0]} 维，红色偏大、蓝色偏小）", fontsize=10, loc="left")

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=100)
    return fig
