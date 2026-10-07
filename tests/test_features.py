"""测试 pipeline/features.py（语谱图、MFCC、画图）和 tools/show_spectrogram.py。

测试音频都是用 numpy 或 ffmpeg 生成的正弦波、噪声，不用语音合成。
"""
import importlib.util

import numpy as np
import pytest
from conftest import ROOT

from pipeline.features import mfcc, plot_recording, spectrogram_db

SR = 16000


def _tone(seconds=1.0, freq=440.0, sr=SR, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _load_tool():
    spec = importlib.util.spec_from_file_location("show_spectrogram", ROOT / "tools" / "show_spectrogram.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mfcc_shape():
    samples = _tone(1.0)
    feats = mfcc(samples, SR)
    assert feats.ndim == 2
    assert abs(feats.shape[0] - 98) <= 2
    assert feats.shape[1] == 13
    assert np.all(np.isfinite(feats))

    times, freqs, db = spectrogram_db(samples, SR)
    peak = freqs[np.argmax(db.mean(axis=1))]
    assert abs(peak - 440) <= 40  # n_fft=400 时相邻频率格相差 40 Hz


def test_spectrogram_shapes():
    samples = _tone(1.0)
    times, freqs, db = spectrogram_db(samples, SR, n_fft=400, hop=160)
    assert len(freqs) == 201
    assert freqs[0] == 0 and freqs[-1] == pytest.approx(8000)
    assert db.shape == (len(freqs), len(times))
    assert np.all(np.diff(times) > 0)
    assert times[0] == pytest.approx(200 / SR)  # 第一帧的中心在 200 个采样点处
    assert np.all(np.isfinite(db))


def test_mfcc_parameters_and_short_input():
    samples = _tone(1.0)
    assert mfcc(samples, SR, n_mfcc=20, n_mels=26).shape[1] == 20
    # 比一帧还短的录音：补零成一帧，不报错
    short = mfcc(samples[:100], SR)
    assert short.shape == (1, 13)
    assert np.all(np.isfinite(short))
    # 全静音也不能出现 -inf
    assert np.all(np.isfinite(mfcc(np.zeros(SR, dtype=np.float32), SR)))


def test_mfcc_distinguishes_tones():
    # 不同音高的声音，MFCC 应该不一样；同一个音高，音量变了 MFCC 第 1 维以外变化很小
    low = mfcc(_tone(1.0, 300), SR).mean(axis=0)
    high = mfcc(_tone(1.0, 3000), SR).mean(axis=0)
    quiet = mfcc(_tone(1.0, 300, amp=0.03), SR).mean(axis=0)
    assert np.linalg.norm(low[1:] - high[1:]) > 5 * np.linalg.norm(low[1:] - quiet[1:])


def test_dct_matches_scipy():
    scipy_fft = pytest.importorskip("scipy.fft")
    from pipeline.features import _dct_matrix

    x = np.random.default_rng(0).normal(size=(5, 40))
    ours = x @ _dct_matrix(40, 13).T
    theirs = scipy_fft.dct(x, type=2, norm="ortho", axis=1)[:, :13]
    assert np.allclose(ours, theirs)


def test_plot_recording_writes_png(tmp_path):
    from matplotlib.figure import Figure

    out = tmp_path / "sub" / "tone.png"
    fig = plot_recording(_tone(2.0), SR, path=out, title="测试：440Hz 正弦波")
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 3
    assert out.is_file()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_recording_without_path_returns_figure():
    fig = plot_recording(_tone(0.5), SR)
    assert len(fig.axes) >= 3


def test_plot_recording_empty_audio_no_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # 空录音也不能出"坐标范围相同"之类的警告
        fig = plot_recording(np.zeros(0, dtype=np.float32), SR)
    assert len(fig.axes) >= 3


def test_int16_and_stereo_input():
    # Gradio 的 gr.Audio(type="numpy") 给的是 16 位整数；soundfile 读立体声是 [采样点, 2]
    tone = _tone(1.0)
    as_int16 = np.round(tone * 32767).astype(np.int16)
    stereo = np.stack([tone, tone], axis=1)
    _, _, db_float = spectrogram_db(tone, SR)
    _, _, db_int = spectrogram_db(as_int16, SR)
    _, _, db_stereo = spectrogram_db(stereo, SR)
    peak = np.argmax(db_float.mean(axis=1))
    assert db_int[peak].mean() == pytest.approx(db_float[peak].mean(), abs=0.1)
    assert np.allclose(db_stereo, db_float)


def test_plot_recording_long_audio(tmp_path):
    # 10 分钟的录音也能画（画图时帧数有上限，不会占用大量内存）
    rng = np.random.default_rng(1)
    samples = (0.05 * rng.standard_normal(SR * 600)).astype(np.float32)
    out = tmp_path / "long.png"
    plot_recording(samples, SR, path=out)
    assert out.is_file()


def test_show_spectrogram_cli(make_audio, tmp_path):
    tool = _load_tool()
    src = make_audio("tone", "mp3", seconds=2.0)
    out = tmp_path / "out.png"
    assert tool.main([str(src), "--out", str(out)]) == 0
    assert out.is_file()


def test_show_spectrogram_default_out(make_audio):
    tool = _load_tool()
    src = make_audio("tone", "m4a", seconds=1.0, name="G1-S1-Q.m4a")
    assert tool.main([str(src)]) == 0
    assert (src.parent / "G1-S1-Q_spectrogram.png").is_file()


def test_show_spectrogram_missing_file(tmp_path, capsys):
    tool = _load_tool()
    assert tool.main([str(tmp_path / "nothing.wav")]) == 1
    assert "找不到" in capsys.readouterr().out
