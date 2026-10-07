"""整条流程 run_pipeline 的测试。"""
from pathlib import Path

from conftest import FOUR_SPEAKERS_WAV, requires_models

from pipeline import run_pipeline
from pipeline.data import FLAG_OUTPUTS

LEGAL_LABELS = {""} | set(FLAG_OUTPUTS)


def test_pipeline_silence(make_audio, tmp_path):
    """没有人声的录音：返回空列表和提示，不报错（Review Focus 第 2 条）。"""
    wav = make_audio("silence", "wav", seconds=2.0, sr=16000, channels=1)
    segments, meta = run_pipeline(str(wav), {"out_dir": str(tmp_path / "run")})
    assert segments == []
    assert meta["message"] == "没有检测到人声，请检查录音"
    assert meta["duration"] > 1.5
    assert meta["notice"]


@requires_models
def test_pipeline_four_speakers(tmp_path):
    segments, meta = run_pipeline(str(FOUR_SPEAKERS_WAV), {"num_speakers": 4, "out_dir": str(tmp_path / "run")})
    assert len(segments) >= 5
    for seg in segments:
        for key in ["start", "end", "speaker", "text", "label"]:
            assert key in seg
        assert seg["end"] > seg["start"]
        assert seg["label"] in LEGAL_LABELS
        assert seg["source"] == "asr"
    assert len({s["speaker_id"] for s in segments if s.get("speaker_id")}) == 4
    assert meta["rtf"] > 0
    assert len(meta["timings"]) >= 6
    assert meta["methods"]["denoise"] == "baseline"
    assert len(meta["sha256"]) == 64
    for seg in segments:
        if seg.get("clip"):
            assert (Path(meta["work_dir"]) / "clips" / seg["clip"]).is_file()


@requires_models
def test_pipeline_method_override_and_eval_mode(tmp_path):
    segments, meta = run_pipeline(
        str(FOUR_SPEAKERS_WAV),
        {"num_speakers": 4, "methods": {"denoise": "g1", "classify": "g6"}, "asr_mode": "both",
         "out_dir": str(tmp_path / "run")})
    assert meta["methods"]["denoise"] == "g1"
    assert meta["methods"]["classify"] == "g6"
    assert all(s.get("text_raw") for s in segments)
    # 没装 TensorFlow 或没训练模型时，g6 退回规则，并把原因写进 meta["warnings"]
    assert isinstance(meta["warnings"], list)
