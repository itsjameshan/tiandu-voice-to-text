"""步骤 7 疑似片段导出（pipeline/step7_clips.py）的测试。

- 基线做法：只给有标签的段出片段，前后各留 padding 秒，不超出录音的开头和结尾；
- 片段文件名只用 ASCII，格式固定；
- 导出的 WAV 按采样点切，长度精确；另附 clips_index.csv（UTF-8 带 BOM、中文表头）；
- 做法登记里有 clips/baseline。

测试音频用 numpy 生成的正弦波，不用语音合成（红线第 2 条）。
"""
import csv

import numpy as np
import pytest
import soundfile as sf

from pipeline import methods
from pipeline.audio import SR, read_wav
from pipeline.schema import new_segment
from pipeline.step7_clips import (
    CLIPS_INDEX_HEADERS,
    CLIPS_INDEX_NAME,
    clip_filename,
    clips_baseline,
    export_clips,
)

CFG = {"clips": {"padding": 1.0}}


def _tone(seconds: float, sr: int = SR) -> np.ndarray:
    """440Hz 正弦波，音量约 -20 dBFS。"""
    t = np.arange(int(round(seconds * sr))) / sr
    return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_clip_padding_and_bounds():
    samples = _tone(3.0)
    segments = [
        new_segment(0.5, 2.0, text="这个要另外交钱", label="疑似·费用"),
        new_segment(2.2, 2.8, text="大家注意安全"),  # 没有标签，不出片段
    ]
    assert clips_baseline(segments, samples, SR, CFG) == [(0.0, 3.0, 0)]


def test_clip_padding_from_config():
    samples = _tone(10.0)
    segments = [new_segment(4.0, 5.0, label="疑似·行程变更")]
    assert clips_baseline(segments, samples, SR, {"clips": {"padding": 0.5}}) == [(3.5, 5.5, 0)]
    # 配置里没写 clips 时按默认 1 秒
    assert clips_baseline(segments, samples, SR, {}) == [(3.0, 6.0, 0)]


def test_clip_index_is_segment_position():
    samples = _tone(20.0)
    segments = [
        new_segment(1.0, 2.0),
        new_segment(5.0, 6.0, label="疑似·购物安排"),
        new_segment(8.0, 9.0),
        new_segment(12.0, 13.0, label="疑似·消费施压"),
    ]
    assert clips_baseline(segments, samples, SR, CFG) == [(4.0, 7.0, 1), (11.0, 14.0, 3)]


def test_clip_filename():
    assert clip_filename(3, 12.4, 18.9) == "clip_003_000012.4-000018.9.wav"
    assert clip_filename(12, 0.0, 3605.5) == "clip_012_000000.0-003605.5.wav"
    assert clip_filename(1, 1.0, 2.0).isascii()


def test_baseline_registered():
    methods.load_all()
    assert "baseline" in methods.available("clips")
    assert methods.get_method("clips", "baseline") is clips_baseline


def test_export_clips_exact_samples(tmp_path):
    samples = _tone(10.0)
    segments = [
        new_segment(3.0, 5.5, speaker="导游", text="这个手镯两千八百块", label="疑似·费用"),
        new_segment(6.0, 7.0, speaker="游客", text="好的"),
        new_segment(8.25, 9.1, speaker="导游", text="不买的话晚上就不安排了", label="疑似·消费施压"),
    ]
    out_dir = tmp_path / "clips"
    result = export_clips(segments, samples, SR, out_dir, CFG)

    wavs = sorted(out_dir.glob("*.wav"))
    assert [p.name for p in wavs] == [clip_filename(1, 2.0, 6.5), clip_filename(2, 7.25, 10.0)]
    for path, (start, end) in zip(wavs, [(2.0, 6.5), (7.25, 10.0)]):
        info = sf.info(str(path))
        assert info.samplerate == SR and info.channels == 1 and info.subtype == "PCM_16"
        assert abs(info.frames - (end - start) * SR) <= 1
        # 内容就是原录音里对应的那一段（16 位量化误差以内）
        clip = read_wav(path)
        first = int(round(start * SR))
        assert np.allclose(clip, samples[first:first + len(clip)], atol=1e-3)

    # 对应段落写上了片段文件名，没有标签的段落没有
    assert result[0]["clip"] == wavs[0].name
    assert "clip" not in result[1]
    assert result[2]["clip"] == wavs[1].name
    # 传进来的段落没有被改动
    assert all("clip" not in seg for seg in segments)


def test_export_clips_index_csv(tmp_path):
    samples = _tone(10.0)
    segments = [new_segment(3.0, 5.5, speaker="导游", text="这个手镯两千八百块", label="疑似·费用")]
    out_dir = tmp_path / "clips"
    export_clips(segments, samples, SR, out_dir, CFG)

    index_path = out_dir / CLIPS_INDEX_NAME
    assert index_path.read_bytes().startswith(b"\xef\xbb\xbf")  # UTF-8 带 BOM，Excel 双击不乱码
    with open(index_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == CLIPS_INDEX_HEADERS
    assert CLIPS_INDEX_HEADERS == ["序号", "开始", "结束", "说话人", "文字", "标签", "文件"]
    assert rows[1] == ["1", "2.0", "6.5", "导游", "这个手镯两千八百块", "疑似·费用", clip_filename(1, 2.0, 6.5)]


def test_export_clips_removes_old_clips(tmp_path):
    """同一个文件夹再导出一次（例如人工改了标签），上一次的片段不能混进来。"""
    samples = _tone(10.0)
    out_dir = tmp_path / "clips"
    export_clips([new_segment(3.0, 5.5, label="疑似·费用")], samples, SR, out_dir, CFG)
    result = export_clips([new_segment(3.0, 5.5, label=""), new_segment(8.0, 8.5, label="疑似·费用")],
                          samples, SR, out_dir, CFG)
    assert [p.name for p in out_dir.glob("*.wav")] == [clip_filename(1, 7.0, 9.5)]
    assert "clip" not in result[0]


def test_export_clips_uses_given_method(tmp_path):
    samples = _tone(10.0)
    segments = [new_segment(3.0, 5.5, label="疑似·费用"), new_segment(6.0, 7.0)]

    def no_padding(segs, smp, sr, cfg):
        return [(seg["start"], seg["end"], i) for i, seg in enumerate(segs)]

    result = export_clips(segments, samples, SR, tmp_path, CFG, method=no_padding)
    assert [r["clip"] for r in result] == [clip_filename(1, 3.0, 5.5), clip_filename(2, 6.0, 7.0)]


def test_export_clips_bad_index(tmp_path):
    samples = _tone(5.0)

    def bad(segs, smp, sr, cfg):
        return [(0.0, 1.0, 5)]

    with pytest.raises(ValueError, match="段落序号"):
        export_clips([new_segment(0.0, 1.0, label="疑似·费用")], samples, SR, tmp_path, CFG, method=bad)


def test_export_clips_no_labels(tmp_path):
    samples = _tone(3.0)
    result = export_clips([new_segment(0.0, 1.0)], samples, SR, tmp_path / "clips", CFG)
    assert result == [new_segment(0.0, 1.0)]
    assert list((tmp_path / "clips").glob("*.wav")) == []
    assert (tmp_path / "clips" / CLIPS_INDEX_NAME).exists()  # 索引表只有表头
