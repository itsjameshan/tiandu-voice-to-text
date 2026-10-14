"""tools/synthesize_recordings.py 的测试：用假的合成函数跑一遍（不需要语音合成模型），检查输出的文件和标准答案。"""
import csv
import importlib.util
import subprocess
import sys

import pytest
from test_tts_dataset import fake_tts

from pipeline.annotations import read_clips_csv, read_turns_csv
from pipeline.audio import find_ffmpeg, probe
from pipeline.config import ROOT, load_config
from pipeline.data import AI_DISCLAIMER, AI_DISCLAIMER_SPEAKER, get_script
from pipeline.pool import parse_recording_name


def _load_tool():
    spec = importlib.util.spec_from_file_location("synthesize_recordings", ROOT / "tools" / "synthesize_recordings.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    """用假的合成函数合成 G6-S3（角色最少的剧本）的 Q 和 F 两段 m4a。"""
    tool = _load_tool()
    out = tmp_path_factory.mktemp("synth") / "ai_recordings"
    calls = []

    def counting_tts(text, sid, speed):
        calls.append(text)
        return fake_tts(text, sid, speed)

    cfg = load_config(overrides={"paths": {"tmp": str(out.parent / "tmp")}})
    assert tool.synthesize_all(cfg, counting_tts, ["G6-S3"], ["Q", "F"], out, "m4a", seed=1, overwrite=False) == 0
    return tool, out, cfg, calls


def test_outputs(made):
    _, out, _, _ = made
    names = sorted(p.name for p in (out / "recordings").iterdir())
    assert names == ["G6-S3-F-TTS.m4a", "G6-S3-Q-TTS.m4a"]
    for name in names:  # 文件名合格，入池后编号不带 -TTS
        assert parse_recording_name(name)["ai_synthesized"] is True
    readme = (out / "README.txt").read_text(encoding="utf-8")
    assert AI_DISCLAIMER in readme and "不是真实录音" in readme and "偏乐观" in readme
    assert not list(out.rglob("*.partial.*")) and not list(out.parent.joinpath("tmp").glob("*.synth.wav"))


def test_metadata_says_ai_generated(made):
    _, out, _, _ = made
    proc = subprocess.run([find_ffmpeg(), "-hide_banner", "-i", str(out / "recordings" / "G6-S3-Q-TTS.m4a")],
                          capture_output=True)
    info = proc.stderr.decode("utf-8", errors="replace")
    assert "AI合成·虚构教学录音 G6-S3-Q" in info and "人工智能技术" in info and "非真人" in info


def test_answer_key(made):
    _, out, _, _ = made
    key = out / "answer_key"
    script = get_script("G6-S3")
    reference = (key / "references" / "G6-S3-Q.txt").read_text(encoding="utf-8")
    assert reference.splitlines()[0] == AI_DISCLAIMER
    turns = read_turns_csv(key / "annotations" / "speakers" / "G6-S3-Q.csv")
    assert turns[0][2] == AI_DISCLAIMER_SPEAKER and len(turns) == len(script["lines"]) + 1
    duration = probe(out / "recordings" / "G6-S3-Q-TTS.m4a")["duration"]
    assert max(end for _, end, _ in turns) < duration
    read_clips_csv(key / "annotations" / "clips" / "G6-S3-F.csv")  # 格式合格（能读回来）
    with open(key / "timeline" / "G6-S3-Q.csv", encoding="utf-8-sig", newline="") as f:
        timeline = list(csv.DictReader(f))
    assert [row["text"] for row in timeline] == [line["text"] for line in script["lines"]]
    with open(key / "summary.csv", encoding="utf-8-sig", newline="") as f:
        summary = {row["file"]: row for row in csv.DictReader(f)}
    assert set(summary) == {"G6-S3-Q-TTS", "G6-S3-F-TTS"}
    assert summary["G6-S3-F-TTS"]["f_variant"].startswith("口袋")  # 6 + 3 是单数
    assert float(summary["G6-S3-Q-TTS"]["duration_s"]) == pytest.approx(duration, abs=0.1)
    assert (key / "voice_plan.csv").is_file() and "老师" in (key / "README.txt").read_text(encoding="utf-8")


def test_answer_key_audacity_labels(made):
    """标准答案另有 Audacity 标签文件，老师可以和同学的标签一起导入 Audacity 对照；读回来和表格一样。"""
    from pipeline.annotations import read_audacity_labels

    _, out, _, _ = made
    key = out / "answer_key"
    for kind, read in (("speakers", read_turns_csv), ("clips", read_clips_csv)):
        labels = read_audacity_labels(key / "audacity_labels" / f"G6-S3-F.{kind}.txt")
        assert labels == read(key / "annotations" / kind / "G6-S3-F.csv")


def test_each_line_synthesized_once_and_rerun_skips(made):
    tool, out, cfg, calls = made
    script = get_script("G6-S3")
    texts = {tool_text for tool_text in calls}
    assert len(calls) == len(texts)  # Q 和 F 共用同一份台词合成结果
    assert len(calls) <= len(script["lines"]) + 1
    before = (out / "recordings" / "G6-S3-Q-TTS.m4a").stat().st_mtime_ns

    def must_not_run(*args):
        raise AssertionError("已经合成好的录音不应该再合成")

    assert tool.synthesize_all(cfg, must_not_run, ["G6-S3"], ["Q", "F"], out, "m4a", seed=1, overwrite=False) == 0
    assert (out / "recordings" / "G6-S3-Q-TTS.m4a").stat().st_mtime_ns == before


def test_seed_for_differs_by_condition():
    tool = _load_tool()
    seeds = {tool.seed_for(f"G{g}-S{s}", c, 1) for g in range(1, 9) for s in range(1, 4) for c in "QNF"}
    assert len(seeds) == 72


def test_cli_rejects_unknown_script():
    import os

    env = dict(os.environ, PYTHONIOENCODING="utf-8")  # 中文 Windows 默认 GBK，这里统一按 UTF-8 读输出
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "synthesize_recordings.py"), "--scripts", "G9-S1"],
                          capture_output=True, text=True, encoding="utf-8", env=env)
    assert proc.returncode == 2 and "G9-S1" in proc.stdout


def test_digits(tmp_path):
    """选做的数字实验：每个声音把"零"到"九"各念 3 遍，存成 tf_lab 认得的 digits/<数字>/tts<声音编号>_<第几遍>.wav。"""
    import soundfile as sf

    tool = _load_tool()
    said = []

    def digit_tts(text, sid, speed):
        said.append(text)
        return fake_tts(text, sid, speed)

    out = tmp_path / "ai_recordings"
    assert tool.synthesize_digits(digit_tts, out, sids=[5, 60]) == 60
    assert sorted(set(said)) == sorted(tool.DIGITS)
    for digit in range(10):
        names = sorted(p.name for p in (out / "digits" / str(digit)).glob("*.wav"))
        assert names == [f"tts{sid}_{rep}.wav" for sid in (5, 60) for rep in (1, 2, 3)]
    info = sf.info(str(out / "digits" / "3" / "tts5_2.wav"))
    assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16" and info.duration < 1.0
    with sf.SoundFile(str(out / "digits" / "3" / "tts5_2.wav")) as f:
        assert "AI合成" in f.title
    assert "不是真实录音" in (out / "digits" / "README.txt").read_text(encoding="utf-8")
    # 训练脚本按 "_" 前面的部分分人：同一个声音的 3 遍算同一个人
    spec = importlib.util.spec_from_file_location("train_digits", ROOT / "tools" / "tf_lab" / "train_digits.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    _, labels, speakers = train.load_dataset(out / "digits")
    assert len(labels) == 60 and sorted(set(speakers)) == ["tts5", "tts60"]
