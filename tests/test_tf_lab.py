"""tools/tf_lab/（选做的中文数字 TensorFlow 小实验）的测试。

不用语音合成（红线第 2 条）：测试音频只用 ffmpeg 或 numpy 生成的正弦波和静音。
- split_digits：段数不是 50 时退出码 1、不写任何文件，并提示怎么重录；
  段数正好 50 时（把端点检测换成假的）按"数字/学号后四位_第几遍.wav"的顺序存好；
- mfcc_compare：没装 TensorFlow 时只画 numpy 版，退出码 0，写出 PNG；装了时两种算法结果高度相关；
  模拟机房自带的 Python（没有 soundfile、PyYAML、ffmpeg）时，切好的 16 kHz WAV 照样能对照、画图；
  其他格式转不了时提示用便携包的 Python；没有 matplotlib 时照样打印相关系数；
- train_digits：说话人不够 3 个、没装 TensorFlow 时给出中文提示并退出码 1；
  装了 TensorFlow 时用很小的合成数据集（每个"数字"是一个不同频率的正弦波）跑通训练。

本环境（.venv）没有装 TensorFlow：需要 TensorFlow 的测试用 pytest.importorskip 自动跳过。
"""
import importlib.util
import re
import subprocess
import sys
import wave

import numpy as np
import pytest
from conftest import ROOT, ZH_WAV, ffmpeg_exe, requires_models

TF_LAB = ROOT / "tools" / "tf_lab"
SR = 16000
ADVICE = "念慢一点、每个字之间停顿约 1 秒，重录"

# 走真的端点检测既要模型文件，也要 sherpa_onnx（只装了 TensorFlow 的环境里没有它，这时跳过）
requires_sherpa = pytest.mark.skipif(importlib.util.find_spec("sherpa_onnx") is None,
                                     reason="没装 sherpa_onnx（端点检测要用）")


def _load(name: str):
    """按文件路径导入 tools/tf_lab/ 里的脚本（tools 不是包）。"""
    spec = importlib.util.spec_from_file_location(f"tf_lab_{name}", TF_LAB / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_wav(path, samples, sr=SR):
    """用 Python 自带的 wave 模块写 16 位单声道 WAV（装 TensorFlow 的虚拟环境里可能没有 soundfile）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(data.tobytes())


def _three_bursts_m4a(tmp_path):
    """用 ffmpeg 生成 3 段"1 秒正弦 + 1 秒静音"的 m4a 文件（44.1 kHz 立体声，检验先转格式）。"""
    out = tmp_path / "three_bursts.m4a"
    graph = "sine=frequency=440:sample_rate=44100:duration=1,apad=pad_dur=1,aloop=loop=2:size=88200,volume=0.2"
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", graph, "-ac", "2", str(out)],
                   check=True)
    return out


def _files_under(folder):
    return sorted(p for p in folder.rglob("*") if p.is_file()) if folder.exists() else []


def _synthetic_digits(root, speakers, reps=2, seconds=0.5):
    """做一个很小的"数字"数据集：数字 d 用 (300 + 150×d) Hz 的正弦波代替，每人的频率略有不同、加一点噪声。

    存成 root/<数字>/<说话人>_<遍>.wav，和 split_digits.py 的输出一样。
    """
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * SR)) / SR
    for s_index, speaker in enumerate(speakers):
        for digit in range(10):
            for rep in range(1, reps + 1):
                freq = (300 + 150 * digit) * (1 + 0.01 * s_index)
                tone = 0.3 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(len(t))
                _write_wav(root / str(digit) / f"{speaker}_{rep}.wav", tone)


# ---------- 计划里列出的测试 ----------


@requires_models
@requires_sherpa
@pytest.mark.parametrize("kind", ["three_bursts", "tone_gap_tone", "tone"])
def test_split_digits_wrong_count(make_audio, tmp_path, capsys, kind):
    """正弦波录音切出来的段数不是 50（Silero 对正弦不一定检出）：退出码 1、一个文件也不写、提示重录。"""
    if kind == "three_bursts":
        audio = _three_bursts_m4a(tmp_path)
    else:
        audio = make_audio(kind, ext="m4a" if kind == "tone_gap_tone" else "wav", seconds=3.0)
    out = tmp_path / "digits"
    split = _load("split_digits")
    rc = split.main([str(audio), "--speaker", "0123", "--out", str(out)])
    assert rc == 1
    assert _files_under(out) == []
    printed = capsys.readouterr().out
    assert ADVICE in printed
    assert "50" in printed


def test_mfcc_compare_without_tf(make_audio, tmp_path, monkeypatch, capsys):
    """没装 TensorFlow 时只画 numpy 版，退出码 0，PNG 图片写出来。"""
    monkeypatch.setitem(sys.modules, "tensorflow", None)  # 让 import tensorflow 失败，模拟没装
    audio = make_audio("tone", ext="m4a", seconds=1.5)
    png = tmp_path / "out" / "compare.png"
    compare = _load("mfcc_compare")
    rc = compare.main([str(audio), "--out", str(png)])
    assert rc == 0
    assert png.is_file()
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    printed = capsys.readouterr().out
    assert "TensorFlow" in printed
    assert "13" in printed  # 打印了 MFCC 的形状（帧数 × 13 维）


# ---------- 补充的测试：split_digits ----------


def _fake_spans(count, start=0.5, step=1.0, length=0.4):
    return [(round(start + i * step, 2), round(start + i * step + length, 2)) for i in range(count)]


def test_split_digits_saves_50_in_order(tmp_path, monkeypatch, capsys):
    """端点检测换成假的、正好返回 50 段：按顺序存成 <数字>/<speaker>_<遍>.wav，第 i 段是数字 i%10、第 i//10+1 遍。"""
    spans = _fake_spans(50)
    # 每段放一个音量不同的正弦波（第 i 段音量 0.01×(i+1)），存好后按音量认出是第几段
    samples = np.zeros(int(52 * SR))
    for i, (start, end) in enumerate(spans):
        a, b = int(start * SR), int(end * SR)
        samples[a:b] = 0.01 * (i + 1) * np.sin(2 * np.pi * 440 * np.arange(b - a) / SR)
    audio = tmp_path / "digits-0123.wav"
    _write_wav(audio, samples)

    split = _load("split_digits")
    monkeypatch.setattr(split, "find_segments", lambda samples, cfg: spans)
    out = tmp_path / "digits"
    rc = split.main([str(audio), "--speaker", "0123", "--out", str(out)])
    assert rc == 0
    files = _files_under(out)
    assert len(files) == 50
    for digit in range(10):
        names = sorted(p.name for p in (out / str(digit)).iterdir())
        assert names == [f"0123_{rep}.wav" for rep in range(1, 6)]
    for i in (0, 13, 49):  # 第 1 段 → 0/0123_1；第 14 段 → 3/0123_2；第 50 段 → 9/0123_5
        path = out / str(i % 10) / f"0123_{i // 10 + 1}.wav"
        with wave.open(str(path), "rb") as f:
            assert f.getframerate() == SR and f.getnchannels() == 1 and f.getsampwidth() == 2
            data = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2") / 32768
        assert abs(np.abs(data).max() - 0.01 * (i + 1)) < 0.003
    assert "50" in capsys.readouterr().out


@requires_models
@requires_sherpa
def test_split_digits_real_vad_50_segments(tmp_path, capsys):
    """走真的端点检测：模型自带的测试音频 zh.wav（一句话，约 4.4 秒人声）重复 50 遍、中间隔 1 秒静音，
    应该正好切出 50 段并存好（这里每段都比 1 秒长，会提示"超过 1 秒"）。只用于自动测试，不是数字录音。"""
    with wave.open(str(ZH_WAV), "rb") as f:
        assert f.getframerate() == SR and f.getnchannels() == 1
        speech = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2") / 32768
    gap = np.zeros(SR)
    audio = tmp_path / "fifty.wav"
    _write_wav(audio, np.concatenate([gap] + [np.concatenate([speech, gap]) for _ in range(50)]))
    out = tmp_path / "digits"
    split = _load("split_digits")
    rc = split.main([str(audio), "--speaker", "0123", "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert len(_files_under(out)) == 50
    assert (out / "9" / "0123_5.wav").is_file()
    assert "超过 1 秒" in printed


def test_split_digits_prints_every_duration(tmp_path, monkeypatch, capsys):
    """段数不对时把每一段的时长都打印出来，并提示怎么重录；不写任何文件。"""
    audio = tmp_path / "digits-0123.wav"
    _write_wav(audio, np.zeros(5 * SR))
    split = _load("split_digits")
    monkeypatch.setattr(split, "find_segments", lambda samples, cfg: [(0.5, 0.92), (1.5, 1.83), (2.5, 3.77)])
    out = tmp_path / "digits"
    rc = split.main([str(audio), "--speaker", "0123", "--out", str(out)])
    assert rc == 1
    assert _files_under(out) == []
    printed = capsys.readouterr().out
    for text in ("0.42", "0.33", "1.27", ADVICE):
        assert text in printed


@pytest.mark.parametrize("speaker", ["张三", "zhang san", "0123/45", ""])
def test_split_digits_rejects_bad_speaker(tmp_path, capsys, speaker):
    """--speaker 只能用学号后四位（数字或英文字母），不能写姓名：直接退出码 1，不处理录音。"""
    audio = tmp_path / "digits.wav"
    _write_wav(audio, np.zeros(SR))
    split = _load("split_digits")
    rc = split.main([str(audio), "--speaker", speaker, "--out", str(tmp_path / "digits")])
    assert rc == 1
    assert "学号后四位" in capsys.readouterr().out
    assert _files_under(tmp_path / "digits") == []


def test_split_digits_missing_file(tmp_path, capsys):
    split = _load("split_digits")
    rc = split.main([str(tmp_path / "nope.m4a"), "--speaker", "0123", "--out", str(tmp_path / "digits")])
    assert rc == 1
    assert "找不到" in capsys.readouterr().out


# ---------- 补充的测试：mfcc_compare ----------


def test_mfcc_compare_with_tf(make_audio, tmp_path, capsys):
    """装了 TensorFlow 时：tf.signal 算的 MFCC 和 numpy 版形状一样、高度相关，图里画两张。"""
    pytest.importorskip("tensorflow")
    pytest.importorskip("matplotlib")
    audio = make_audio("tone_gap_tone", ext="wav", sr=16000, channels=1)
    png = tmp_path / "compare.png"
    compare = _load("mfcc_compare")
    rc = compare.main([str(audio), "--out", str(png)])
    assert rc == 0
    assert png.is_file()
    printed = capsys.readouterr().out
    found = re.search(r"相关系数.*?：(-?\d+\.\d+)", printed)  # "两种算法的相关系数（13 维的平均）：0.9999"
    assert found, printed
    assert float(found.group(1)) > 0.99


def test_tf_mfcc_matches_numpy():
    """直接比较两个函数：形状一样，每一维的相关系数都接近 1。"""
    pytest.importorskip("tensorflow")
    from pipeline.features import mfcc

    compare = _load("mfcc_compare")
    rng = np.random.default_rng(1)
    t = np.arange(SR) / SR
    x = (0.3 * np.sin(2 * np.pi * 500 * t) + 0.01 * rng.standard_normal(SR)).astype(np.float32)
    a = mfcc(x, SR)
    b = compare.tf_mfcc(x, SR)
    assert a.shape == b.shape == (98, 13)
    assert compare.mean_correlation(a, b) > 0.99


def test_mean_correlation_ignores_scale():
    """相关系数只看"起伏像不像"，和整体放大多少倍无关（TF 的第 0 维比 numpy 版大 √2 倍也不影响）。"""
    compare = _load("mfcc_compare")
    rng = np.random.default_rng(2)
    a = rng.standard_normal((50, 13))
    b = a * np.r_[np.sqrt(2.0), np.ones(12)]
    assert compare.mean_correlation(a, b) == pytest.approx(1.0)
    assert compare.mean_correlation(a, -a) == pytest.approx(-1.0)


def _hide_modules(monkeypatch, *names):
    """让 import 这些库失败，模拟"这个 Python 没装"。"""
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)


def _hide_ffmpeg(monkeypatch, tmp_path):
    """让 pipeline.audio.find_ffmpeg 找不到 ffmpeg：不设 FFMPEG_BINARY、PATH 里没有、也没装 imageio-ffmpeg。"""
    empty = tmp_path / "empty_path"
    empty.mkdir(exist_ok=True)
    monkeypatch.delenv("FFMPEG_BINARY", raising=False)
    monkeypatch.setenv("PATH", str(empty))
    _hide_modules(monkeypatch, "imageio_ffmpeg")


def _split_digit_wav(tmp_path):
    """和 split_digits.py 切出来的文件一样：digits/3/0123_1.wav，16000 Hz、单声道、16 位，约 0.6 秒正弦 + 静音。"""
    t = np.arange(SR) / SR
    path = tmp_path / "digits" / "3" / "0123_1.wav"
    _write_wav(path, 0.3 * np.sin(2 * np.pi * 440 * t) * (t < 0.6))
    return path


def test_mfcc_compare_machine_room_python(tmp_path, monkeypatch, capsys):
    """模拟机房自带的 Python（没有 soundfile、PyYAML、ffmpeg）：split_digits.py 切好的 WAV
    用 Python 自带的 wave 模块直接读，不转格式；读不了 config.yaml 时图片存到项目的 outputs/tf_lab/。退出码 0。"""
    _hide_modules(monkeypatch, "soundfile", "yaml")
    _hide_ffmpeg(monkeypatch, tmp_path)
    audio = _split_digit_wav(tmp_path)
    compare = _load("mfcc_compare")
    monkeypatch.setattr(compare, "ROOT", tmp_path)  # 项目文件夹换成临时文件夹，不往仓库的 outputs/ 里写
    rc = compare.main([str(audio)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    png = tmp_path / "outputs" / "tf_lab" / "3_0123_1_mfcc_compare.png"
    assert png.is_file(), printed
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("missing", ["ffmpeg", "soundfile", "unnamed"])
def test_mfcc_compare_other_formats_need_portable_python(make_audio, tmp_path, monkeypatch, capsys, missing):
    """不是 16000 Hz 单声道 16 位的 WAV（这里是 44.1 kHz 立体声）要先用 ffmpeg 转、用 soundfile 读。
    缺了时说清楚缺什么、建议对照切好的 WAV 或用便携包的 Python；退出码 1。
    不提示 pip install（机房不联网、重启会还原）；提示里也不能出现 "None"（有的 ImportError 不带模块名）。"""
    import pipeline.audio

    audio = make_audio("tone", ext="wav", seconds=1.0)  # 44.1 kHz 立体声（先生成，再"拿走" ffmpeg）
    if missing == "ffmpeg":
        _hide_ffmpeg(monkeypatch, tmp_path)
    elif missing == "soundfile":
        _hide_modules(monkeypatch, "soundfile")
    else:
        def broken_read_wav(path):
            raise ImportError("cannot import name 'x' from 'y'")  # 这种 ImportError 的 name 是 None

        monkeypatch.setattr(pipeline.audio, "read_wav", broken_read_wav)
    png = tmp_path / "compare.png"
    compare = _load("mfcc_compare")
    rc = compare.main([str(audio), "--out", str(png)])
    printed = capsys.readouterr().out
    assert rc == 1, printed
    assert not png.exists()
    for words in ("便携包", "split_digits.py", {"ffmpeg": "ffmpeg", "soundfile": "soundfile",
                                                 "unnamed": "cannot import name"}[missing]):
        assert words in printed, words
    assert "None" not in printed
    assert "pip install" not in printed


@pytest.mark.parametrize("relative, name", [
    ("digits/3/0123_1.wav", "3_0123_1_mfcc_compare.png"),  # 切好的文件：前面加上数字，不同数字的图不会互相覆盖
    ("digits/8/0123_1.wav", "8_0123_1_mfcc_compare.png"),
    ("digits-0123.m4a", "digits-0123_mfcc_compare.png"),
    ("录音.m4a", "audio_mfcc_compare.png"),  # 中文文件名换成英文（Windows 上路径最好只有英文）
])
def test_mfcc_compare_default_png_name(tmp_path, relative, name):
    compare = _load("mfcc_compare")
    out = compare.default_out_path(tmp_path / relative, tmp_path / "outputs")
    assert out == tmp_path / "outputs" / "tf_lab" / name


def test_mfcc_compare_without_matplotlib_and_tf(tmp_path, monkeypatch, capsys):
    """既没有 TensorFlow 也没有 matplotlib：对照不了、也画不了图，说明原因，退出码 1。"""
    _hide_modules(monkeypatch, "tensorflow", "matplotlib")
    audio = _split_digit_wav(tmp_path)
    png = tmp_path / "compare.png"
    compare = _load("mfcc_compare")
    rc = compare.main([str(audio), "--out", str(png)])
    printed = capsys.readouterr().out
    assert rc == 1, printed
    assert not png.exists()
    assert "没有画图" in printed
    assert "pip install" not in printed


def test_mfcc_compare_with_tf_without_matplotlib(tmp_path, monkeypatch, capsys):
    """装了 TensorFlow、没装 matplotlib（机房的 Python 可能这样）：照样打印相关系数，只是不画图，退出码 0。"""
    pytest.importorskip("tensorflow")
    _hide_modules(monkeypatch, "matplotlib", "soundfile", "yaml")
    audio = _split_digit_wav(tmp_path)
    png = tmp_path / "compare.png"
    compare = _load("mfcc_compare")
    rc = compare.main([str(audio), "--out", str(png)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert not png.exists()
    assert "相关系数" in printed
    assert "没有画图" in printed


# ---------- 补充的测试：train_digits ----------


def test_pick_test_speakers_holds_out_people_not_clips():
    train = _load("train_digits")
    speakers = [f"{n:04d}" for n in range(10)]
    test = train.pick_test_speakers(speakers)
    assert len(test) == 2  # 10 人的 20%
    assert set(test) <= set(speakers)
    assert train.pick_test_speakers(speakers) == test  # 固定随机种子，每次一样
    assert len(train.pick_test_speakers(speakers[:3])) == 1  # 人少时至少留 1 人测试


def test_load_dataset_pads_to_one_second(tmp_path):
    """每段录音补零或截断到 1 秒，MFCC 都是 98 帧 × 13 维；说话人从文件名里取。"""
    train = _load("train_digits")
    _write_wav(tmp_path / "3" / "0123_1.wav", 0.1 * np.ones(int(0.4 * SR)))
    _write_wav(tmp_path / "7" / "4567_2.wav", 0.1 * np.ones(int(1.6 * SR)))
    features, labels, speakers = train.load_dataset(tmp_path)
    assert features.shape == (2, 98, 13)
    assert sorted(zip(labels.tolist(), speakers)) == [(3, "0123"), (7, "4567")]


def test_train_digits_too_few_speakers(tmp_path, capsys):
    """不到 3 个人的录音：说明原因，退出码 1（不需要 TensorFlow 就能检查出来）。"""
    data = tmp_path / "digits"
    _synthetic_digits(data, ["0001", "0002"], reps=1, seconds=0.3)
    train = _load("train_digits")
    rc = train.main(["--data", str(data)])
    assert rc == 1
    assert "至少需要 3" in capsys.readouterr().out


def test_train_digits_empty_folder(tmp_path, capsys):
    train = _load("train_digits")
    rc = train.main(["--data", str(tmp_path / "no_such_folder")])
    assert rc == 1
    assert "split_digits.py" in capsys.readouterr().out


def test_train_digits_without_tf(tmp_path, monkeypatch, capsys):
    """没装 TensorFlow：提示用机房自带的 Python 运行，退出码 1，不报一大串错。"""
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    data = tmp_path / "digits"
    _synthetic_digits(data, ["0001", "0002", "0003"], reps=1, seconds=0.3)
    train = _load("train_digits")
    rc = train.main(["--data", str(data)])
    assert rc == 1
    assert "用机房自带的 Python 运行" in capsys.readouterr().out


def test_train_digits_smoke(tmp_path, capsys):
    """装了 TensorFlow 时：小数据集跑通训练，打印测试用的说话人、准确率和混淆矩阵。"""
    pytest.importorskip("tensorflow")
    data = tmp_path / "digits"
    _synthetic_digits(data, ["0001", "0002", "0003", "0004", "0005"], reps=2)
    train = _load("train_digits")
    rc = train.main(["--data", str(data), "--epochs", "40"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "混淆矩阵" in printed
    found = re.search(r"准确率[^0-9]*(\d+\.\d+)%", printed)
    assert found, printed
    assert 0.0 <= float(found.group(1)) <= 100.0


# ---------- 补充的测试：导入轻量、讲义 ----------


def test_tf_lab_scripts_import_light():
    """三个脚本导入时不能把 tensorflow、sherpa_onnx、gradio、matplotlib 带进来（重依赖在函数里导入）。"""
    code = (
        "import importlib.util, sys\n"
        "for name in ('mfcc_compare', 'split_digits', 'train_digits'):\n"
        f"    spec = importlib.util.spec_from_file_location(name, r'{TF_LAB}' + '/' + name + '.py')\n"
        "    spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
        "bad = [m for m in ('tensorflow', 'sherpa_onnx', 'gradio', 'matplotlib') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_readme_handout():
    text = (TF_LAB / "README.md").read_text(encoding="utf-8")
    for words in ("选做", "学号后四位", "MFCC", "项目 2", "项目 4", "split_digits.py", "train_digits.py",
                  "mfcc_compare.py", "用机房自带的 Python"):
        assert words in text, words
