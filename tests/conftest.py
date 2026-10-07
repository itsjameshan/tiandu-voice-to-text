"""测试共用的工具。

- make_audio：用 ffmpeg 生成正弦波、静音等测试音频（不用语音合成，红线第 2 条）。
- requires_models：需要已下载模型的测试用这个标记，没有模型时自动跳过。
"""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SENSE_VOICE = MODELS / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
FOUR_SPEAKERS_WAV = MODELS / "0-four-speakers-zh.wav"
ZH_WAV = SENSE_VOICE / "test_wavs" / "zh.wav"

_REQUIRED_MODEL_FILES = [
    SENSE_VOICE / "model.int8.onnx",
    SENSE_VOICE / "tokens.txt",
    MODELS / "silero_vad.onnx",
    MODELS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx",
    MODELS / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
    FOUR_SPEAKERS_WAV,
]


def models_available() -> bool:
    return all(p.exists() for p in _REQUIRED_MODEL_FILES)


requires_models = pytest.mark.skipif(not models_available(), reason="模型未下载（运行 python models/download_models.py）")


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _run_ffmpeg(args):
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", *args], check=True)


@pytest.fixture
def make_audio(tmp_path):
    """生成测试音频文件，返回路径。

    kind:
      "tone"           440Hz 正弦波（约 -20 dBFS）
      "silence"        全静音
      "tone_gap_tone"  1 秒正弦 + 12 秒静音 + 1 秒正弦（seconds 参数忽略）
    ext: 文件扩展名，如 "wav"、"mp3"、"m4a"、"mp4"
    """

    def _make(kind="tone", ext="wav", seconds=3.0, sr=44100, channels=2, name=None):
        out = tmp_path / (name or f"{kind}_{sr}_{channels}.{ext}")
        layout = "stereo" if channels == 2 else "mono"
        if kind == "tone":
            src = ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={sr}:duration={seconds}"]
            filt = ["-af", "volume=0.2", "-ac", str(channels)]
        elif kind == "silence":
            src = ["-f", "lavfi", "-i", f"anullsrc=r={sr}:cl={layout}", "-t", str(seconds)]
            filt = ["-ac", str(channels)]
        elif kind == "tone_gap_tone":
            src = ["-f", "lavfi", "-i",
                   f"sine=frequency=440:sample_rate={sr}:duration=1,apad=pad_dur=12[a];"
                   f"sine=frequency=440:sample_rate={sr}:duration=1[b];[a][b]concat=n=2:v=0:a=1"]
            filt = ["-af", "volume=0.2", "-ac", str(channels)]
        else:
            raise ValueError(kind)
        extra = []
        if ext == "mp4":
            # 视频文件：加一个黑色画面轨道，检验“只取音轨”
            extra = ["-f", "lavfi", "-i", "color=c=black:s=64x64:d=%s" % (14 if kind == "tone_gap_tone" else seconds),
                     "-shortest", "-c:v", "libx264" if _has_encoder("libx264") else "mpeg4"]
            _run_ffmpeg([*src, *extra[:4], *filt, *extra[4:], str(out)])
            return out
        _run_ffmpeg([*src, *filt, str(out)])
        return out

    return _make


def _has_encoder(name: str) -> bool:
    out = subprocess.run([ffmpeg_exe(), "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return f" {name} " in out
