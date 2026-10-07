"""剧本文本演示（pipeline/script_demo.py）的测试：不用录音，直接拿剧本台词跑步骤 5—8。"""
import zipfile

import docx

from pipeline.data import FLAG_OUTPUTS, get_script
from pipeline.script_demo import DEMO_NOTICE, run_script_demo, script_segments
from pipeline.step8_report import DISCLAIMER, export_bundle


def test_script_segments_times_and_speakers():
    segs = script_segments("G1-S1")
    script = get_script("G1-S1")
    assert len(segs) == len(script["lines"]) == 53
    for a, b in zip(segs, segs[1:]):
        assert a["end"] <= b["start"] + 1e-6
    assert {s["speaker"] for s in segs} <= set(script["roles"])
    assert all(s["source"] == "script_demo" for s in segs)
    assert all("gold_label" in s for s in segs)


def test_run_script_demo_g8(tmp_path):
    segments, meta, stats = run_script_demo("G8-S1")
    assert len(segments) == len(get_script("G8-S1")["lines"])
    assert all(s["label"] in {""} | set(FLAG_OUTPUTS) for s in segments)
    for key in ["total", "flagged", "correct_flags", "false_positives", "missed", "fp_rate"]:
        assert key in stats
    assert meta["demo_notice"] == DEMO_NOTICE
    zip_path = export_bundle(segments, meta, tmp_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extract("review_draft.docx", tmp_path / "x")
    text = "\n".join(p.text for p in docx.Document(tmp_path / "x" / "review_draft.docx").paragraphs)
    assert DISCLAIMER in text


def test_print_g8_false_positive_rates():
    for sid in ["G8-S1", "G8-S2", "G8-S3"]:
        _, _, stats = run_script_demo(sid)
        print(f"{sid}：误报 {stats['false_positives']}，误报率 {stats['fp_rate']:.2f}")
        assert 0.0 <= stats["fp_rate"] <= 1.0
