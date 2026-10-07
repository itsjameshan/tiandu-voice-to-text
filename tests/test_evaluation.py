"""测评库（pipeline/evaluation.py）和两个命令行工具（tools/evaluate.py、tools/compare.py）的测试。

红线：云端开发环境里不报告任何字错率数字。这里只检查流程能跑通、字段齐全、缓存有效，
不打印、不记录字错率的具体数值。
测试音频：模型自带的 0-four-speakers-zh.wav 改名成 G1-S1-Q.wav 放进临时数据池（不用语音合成）。
数据池建在 pytest 的临时文件夹里，不碰仓库里的 data_pool/。
"""
import copy
import csv
import importlib.util
import os
import shutil
import stat

import numpy as np
import pytest
import soundfile as sf

from conftest import FOUR_SPEAKERS_WAV, ROOT, requires_models
from pipeline.config import load_config
from pipeline.evaluation import (
    FIXED_LIMITATION,
    NOT_FROZEN,
    cache_key,
    count_names,
    eval_cer,
    eval_hotwords,
    eval_numbers_lines,
    hotword_stats,
    pool_items,
    pool_version,
    write_report,
)
from pipeline.pool import MANIFEST_COLUMNS, export_references, ingest_pool, pool_paths, write_csv_rows

LIMITATION = "剧本数据上的测评结果不代表真实场景的效果"


@pytest.fixture
def cfg():
    return load_config()


def _make_writable(root):
    """入池会把原始录音设为只读；测试结束后改回可写，Windows 上 pytest 才能删掉临时文件夹。"""
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file():
                os.chmod(p, stat.S_IREAD | stat.S_IWRITE)


@pytest.fixture(scope="module")
def renamed_pool(tmp_path_factory):
    """把模型自带的四人测试音频复制成 raw/G1-S1-Q.wav，入池并导出参考文本。整个文件的测试共用这个数据池。"""
    root = tmp_path_factory.mktemp("eval") / "pool"
    raw = pool_paths(root)["raw"]
    raw.mkdir(parents=True)
    shutil.copy2(FOUR_SPEAKERS_WAV, raw / "G1-S1-Q.wav")
    result = ingest_pool(root, load_config())
    assert [item["stem"] for item in result["added"]] == ["G1-S1-Q"]
    export_references(root)
    yield root
    _make_writable(root)


def _load_tool(script):
    spec = importlib.util.spec_from_file_location(script.removesuffix(".py"), ROOT / "tools" / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ======================== 计划里列的 4 个测试 ========================


@requires_models
def test_eval_cer_runs_on_renamed_test_audio(renamed_pool, cfg):
    rows, summary = eval_cer(renamed_pool, cfg)
    assert len(rows) == 1
    row = rows[0]
    for key in ("stem", "group", "condition", "cer", "sub", "dele", "ins", "n_ref", "seconds"):
        assert key in row
    assert row["stem"] == "G1-S1-Q" and row["group"] == 1 and row["condition"] == "Q"
    assert 0 <= row["cer"]
    assert row["n_ref"] > 0
    assert 50 < row["seconds"] < 60  # 测试音频约 57 秒
    # 汇总：全体、按条件、按组
    assert summary["all"]["files"] == 1
    assert set(summary["by_condition"]) == {"Q"}
    assert set(summary["by_group"]) == {1}
    assert summary["all"]["cer"] == pytest.approx(row["cer"])


def test_eval_numbers_lines(cfg):
    result = eval_numbers_lines(cfg)
    assert "main" in result and "by_type" in result
    assert result["main"]["gold"] > 0
    assert "金额" in result["by_type"]
    # 指定做法：g4 初始直接调用基线，结果应该和基线一样
    g4 = eval_numbers_lines(cfg, methods={"normalize": "g4"})
    assert g4["main"] == result["main"]
    assert g4["info"]["methods"] == {"normalize": "g4"}


def test_compare_writes_report(tmp_path, capsys):
    compare = _load_tool("compare.py")
    code = compare.main(["--metric", "numbers", "--slot", "normalize", "--method", "g4", "--out", str(tmp_path)])
    assert code == 0
    md = tmp_path / "compare_numbers.md"
    csv_path = tmp_path / "compare_numbers.csv"
    assert md.is_file() and csv_path.is_file()
    rows = _read_csv(csv_path)
    assert list(rows[0]) == ["分组项", "基线", "改进", "差值"]
    assert rows[0]["分组项"].startswith("主指标")
    for row in rows:  # g4 初始等于基线
        assert float(row["基线"]) == float(row["改进"])
        assert float(row["差值"]) == 0
    text = md.read_text(encoding="utf-8")
    assert "| 分组项 | 基线 | 改进 | 差值 |" in text
    assert LIMITATION in text
    assert "g4" in text


def test_report_header(tmp_path):
    rows = [{"stem": "G1-S1-Q", "group": 1, "condition": "Q", "cer": 0.5, "sub": 1, "dele": 0, "ins": 0,
             "n_ref": 2, "seconds": 3.0}]
    summary = {"info": {"pool": str(tmp_path), "pool_version": "v1",
                        "methods": {"denoise": "baseline", "vad": "g1"},
                        "params": {"vad.threshold": 0.5}},
               "tables": [{"title": "汇总", "rows": [{"分组项": "全体", "cer": 0.5}]}]}
    csv_path, md_path = write_report(tmp_path / "out", "cer", rows, summary, ["这是一条说明"])
    assert csv_path == tmp_path / "out" / "cer.csv" and md_path == tmp_path / "out" / "cer.md"
    text = md_path.read_text(encoding="utf-8")
    assert LIMITATION in text and FIXED_LIMITATION == LIMITATION
    assert "数据池版本：v1" in text
    assert "vad=g1" in text
    assert "vad.threshold=0.5" in text
    assert "生成时间" in text
    assert "这是一条说明" in text
    # 头几行就写明版本和局限
    head = "\n".join(text.splitlines()[:12])
    assert "数据池版本" in head and LIMITATION in head
    # CSV 用中文表头、UTF-8 带 BOM
    assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")
    header = _read_csv(csv_path)[0]
    assert "文件编号" in header and "字错率" in header


# ======================== 补充测试 ========================


@requires_models
def test_eval_cer_reuses_cache(renamed_pool, cfg, monkeypatch):
    first, _ = eval_cer(renamed_pool, cfg)
    caches = list(pool_paths(renamed_pool)["asr_cache"].glob("G1-S1-Q.eval-*.json"))
    assert len(caches) == 1  # 缓存文件名以文件编号开头，复录时入池工具会删掉它

    def boom(*args, **kwargs):
        raise AssertionError("应该用缓存，不应该再识别")

    monkeypatch.setattr("pipeline.step3_asr.recognize", boom)
    monkeypatch.setattr("pipeline.step2_vad.detect_speech", boom)
    messages = []
    second, _ = eval_cer(renamed_pool, cfg, progress=messages.append)
    assert second[0]["cer"] == first[0]["cer"]
    assert any("缓存" in m for m in messages)
    # 只换下游的热词纠错做法、打开热词纠错：识别结果照样用缓存
    hot_cfg = load_config(overrides={"hotword": {"enabled": True}})
    eval_cer(renamed_pool, hot_cfg, methods={"hotword": "g5"})
    # 不用缓存时必须重新识别
    with pytest.raises(AssertionError, match="应该用缓存"):
        eval_cer(renamed_pool, cfg, use_cache=False)


@requires_models
def test_eval_hotwords_runs(renamed_pool, cfg):
    rows, summary = eval_hotwords(renamed_pool, cfg)
    assert len(rows) == 1
    for key in ("stem", "names_ref", "names_hit", "name_accuracy", "variants_ref", "overcorrected", "corrections"):
        assert key in rows[0]
    assert summary["all"]["files"] == 1
    assert 0 <= summary["all"]["name_accuracy"] <= 1


@requires_models
def test_evaluate_cli_cer_and_compare(renamed_pool, tmp_path, capsys):
    evaluate = _load_tool("evaluate.py")
    out = tmp_path / "g1"
    code = evaluate.main(["cer", "--pool", str(renamed_pool), "--files", "G1-S1-Q", "--out", str(out)])
    assert code == 0
    assert (out / "cer.csv").is_file()
    text = (out / "cer.md").read_text(encoding="utf-8")
    assert LIMITATION in text and "未冻结" in text
    assert "[1/1] G1-S1-Q" in capsys.readouterr().out  # 每个文件都显示进度

    compare = _load_tool("compare.py")
    code = compare.main(["--slot", "hotword", "--method", "g5", "--metric", "cer",
                         "--pool", str(renamed_pool), "--out", str(out)])
    assert code == 0
    rows = _read_csv(out / "compare_cer.csv")
    assert [row["分组项"] for row in rows] == ["全体", "Q（安静）", "第 1 组"]
    assert all(float(row["差值"]) == 0 for row in rows)  # g5 初始等于基线


def test_cache_key(tmp_path, cfg):
    wav = tmp_path / "a.wav"
    sf.write(wav, np.zeros(1600, dtype=np.float32), 16000, subtype="PCM_16")
    key = cache_key(wav, cfg)
    assert key == cache_key(wav, load_config())  # 同样的录音、同样的设置 → 同一个缓存
    assert len(key) <= 16 and key.isalnum()
    # 影响识别结果的设置变了 → 换一个缓存
    assert cache_key(wav, load_config(overrides={"vad": {"threshold": 0.3}})) != key
    assert cache_key(wav, load_config(overrides={"asr": {"language": "auto"}})) != key
    assert cache_key(wav, load_config(overrides={"methods": {"denoise": "noisereduce"}})) != key
    # 只影响识别之后的设置（热词纠错、说话人分离、分类）不换缓存
    assert cache_key(wav, load_config(overrides={"hotword": {"enabled": True}})) == key
    assert cache_key(wav, load_config(overrides={"methods": {"hotword": "g5", "classify": "g6"}})) == key
    # 录音内容变了 → 换一个缓存
    sf.write(wav, np.ones(1600, dtype=np.float32) * 0.1, 16000, subtype="PCM_16")
    assert cache_key(wav, cfg) != key


def test_count_names_longest_first():
    names = ["雾隐", "雾隐行舟", "雾隐行舟旅行社"]
    counts = count_names("雾隐行舟旅行社和雾隐，还有雾隐行舟旅行社", names)
    assert counts == {"雾隐行舟旅行社": 2, "雾隐": 1}


def test_hotword_stats_counting_rule():
    hotwords = ["雾隐行舟旅行社", "雾隐行舟", "雾隐", "松风晚渡旅行社", "松风", "百年茶语"]
    variants = [{"spoken_variant": "松风行舟", "correct_name": "松风晚渡旅行社"},
                {"spoken_variant": "百年茶叶", "correct_name": "百年茶语"}]
    reference = "我们报的是雾隐行舟旅行社，不是松风行舟的团。买了百年茶叶的茶。"
    # 识别结果（纠错后）：雾隐行舟旅行社对了；说错的"松风行舟"被改成了正确名称（过度纠正）；
    # "百年茶叶"照实记录（没有过度纠正）
    hypothesis = "我们报的是雾隐行舟旅行社不是松风晚渡旅行社的团买了百年茶叶的茶"
    stats = hotword_stats(reference, hypothesis, hotwords, variants)
    # 说错的名称所在位置不算专名（"松风行舟"里的"松风"不算），参考文本里的专名只有 1 个
    assert stats["names_ref"] == 1 and stats["names_hit"] == 1
    assert stats["variants_ref"] == 2
    assert stats["overcorrected"] == 1
    # 识别结果把专名听错了：专名没命中
    wrong = hotword_stats(reference, "我们报的是雾影行走旅行社不是松风行舟的团买了百年茶叶的茶", hotwords, variants)
    assert wrong["names_hit"] == 0 and wrong["overcorrected"] == 0


def _hotword_data():
    from pipeline.data import load_hotword_variants, load_hotwords

    return load_hotwords(), load_hotword_variants()


@pytest.mark.parametrize("reference, hypothesis", [
    ("去那个听松坊", "去那个听松阁"),
    ("老公，要不回头去那个听松坊，看看糯种的？", "老公要不回头去那个听松阁看看糯种的"),
    ("定金已经交给了雾隐晚渡", "定金已经交给了雾隐行舟"),
    ("我们报的是松风那家的团", "我们报的是松风晚渡的团"),
])
def test_overcorrection_to_short_name(reference, hypothesis):
    """纠错只能换成一样长的热词：说错的名称被改成正确名称的简称（听松阁、雾隐行舟、松风晚渡）也算过度纠正。"""
    hotwords, variants = _hotword_data()
    stats = hotword_stats(reference, hypothesis, hotwords, variants)
    assert stats["variants_ref"] == 1
    assert stats["overcorrected"] == 1
    # 识别结果照实记录了说错的名称：没有过度纠正
    assert hotword_stats(reference, reference, hotwords, variants)["overcorrected"] == 0


def test_overcorrection_uses_text_before_correction():
    """纠错之前识别对了的说错名称，被纠错改掉了：用纠错前的文字（before）能数出来，不会被别处抵消。"""
    hotwords, variants = _hotword_data()
    reference = "雾隐行舟旅行社的导游说是雾隐晚渡"
    before = "雾影行走旅行社的导游说是雾隐晚渡"  # 正确名称识别错了，说错的名称识别对了
    after = "雾影行走旅行社的导游说是雾隐行舟"  # 纠错只改了说错的名称
    # 只看纠错后的文字：识别结果里多出的"雾隐行舟"和识别错的正确名称互相抵消，数不出来
    assert hotword_stats(reference, after, hotwords, variants)["overcorrected"] == 0
    assert hotword_stats(reference, after, hotwords, variants, before=before)["overcorrected"] == 1
    # 改成什么都算：纠错前识别对了的"晓月阁"被改掉了
    stats = hotword_stats("去晓月阁看看", "去小乐歌看看", hotwords, variants, before="去晓月阁看看")
    assert stats["overcorrected"] == 1
    # 纠错前就识别错了、纠错也没动：不算过度纠正
    stats = hotword_stats("去晓月阁看看", "去小月哥看看", hotwords, variants, before="去小月哥看看")
    assert stats["overcorrected"] == 0


def test_overcorrection_on_script_text():
    """第 5 组要试的第一个实验：max_syllable_mismatch=1 会把剧本 G5-S3 里两处"听松坊"改成"听松阁"。"""
    from pipeline.data import script_reference_text
    from pipeline.hotwords import correct_text

    hotwords, variants = _hotword_data()
    reference = script_reference_text("G5-S3")
    loose, _ = correct_text(reference, hotwords, min_len=3, max_mismatch=1)
    stats = hotword_stats(reference, loose, hotwords, variants, before=reference)
    assert stats["by_correct"]["听松阁玉器行"]["over"] == 2
    assert stats["overcorrected"] >= 2
    strict, _ = correct_text(reference, hotwords, min_len=3, max_mismatch=0)
    assert hotword_stats(reference, strict, hotwords, variants, before=reference)["overcorrected"] == 0


@requires_models
def test_eval_hotword_step_sees_text_like_run_pipeline(renamed_pool, cfg, monkeypatch):
    """和 run_pipeline 的测评模式一样：热词纠错拿到的段落里 text 等于 text_raw。"""
    from pipeline.methods import _REGISTRY

    seen = []

    def spy(segments, hotwords, cfg):
        seen.extend(segments)
        return segments

    monkeypatch.setitem(_REGISTRY["hotword"], "spy", spy)
    rows, _ = eval_hotwords(renamed_pool, cfg, methods={"hotword": "spy"})
    assert seen and all(seg["text"] == seg["text_raw"] for seg in seen)
    assert len(rows) == 1


def test_config_method_typo_gives_chinese_error(monkeypatch, capsys):
    """config.yaml 里写错了做法名（如 g11）：给出中文提示，不是一大段英文报错。"""
    bad = load_config(overrides={"methods": {"denoise": "g11"}})
    with pytest.raises(ValueError, match="g11"):
        eval_numbers_lines(bad)
    monkeypatch.setattr("pipeline.config.load_config", lambda *args, **kwargs: copy.deepcopy(bad))
    evaluate = _load_tool("evaluate.py")
    assert evaluate.main(["numbers"]) != 0
    assert "g11" in capsys.readouterr().out
    compare = _load_tool("compare.py")
    assert compare.main(["--metric", "numbers", "--slot", "normalize", "--method", "g4"]) != 0
    assert "g11" in capsys.readouterr().out


def test_cli_ctrl_c_stops_quietly(monkeypatch, capsys):
    """中途按 Ctrl+C：打印一句中文说明，不打印 Python 报错。"""
    def stop(*args, **kwargs):
        raise KeyboardInterrupt

    evaluate = _load_tool("evaluate.py")
    monkeypatch.setitem(evaluate.RUNNERS, "numbers", stop)
    assert evaluate.main(["numbers"]) == 1
    assert "已停止" in capsys.readouterr().out
    compare = _load_tool("compare.py")
    monkeypatch.setattr(compare, "_evaluate", stop)
    assert compare.main(["--metric", "numbers", "--slot", "normalize", "--method", "g4"]) == 1
    assert "已停止" in capsys.readouterr().out


def test_cer_hints_when_hotword_switch_off(tmp_path, capsys):
    """cer 换了热词纠错做法、但热词纠错开关是关的：提示加 --hotword on。"""
    evaluate = _load_tool("evaluate.py")
    empty = tmp_path / "empty_pool"
    assert evaluate.main(["cer", "--pool", str(empty), "--method", "hotword=g5"]) == 1  # 空数据池
    assert "--hotword on" in capsys.readouterr().out
    assert evaluate.main(["cer", "--pool", str(empty), "--method", "hotword=g5", "--hotword", "on"]) == 1
    assert "--hotword on" not in capsys.readouterr().out
    assert evaluate.main(["cer", "--pool", str(empty)]) == 1
    assert "--hotword on" not in capsys.readouterr().out


def test_pool_items_from_manifest(tmp_path):
    root = tmp_path / "pool"
    rows = []
    for stem in ["G2-S1-N", "G1-S1-Q"]:
        rows.append({"文件编号": stem, "剧本编号": stem[:5], "录音条件": "", "原始文件": f"raw/{stem}.m4a",
                     "转换后文件": f"normalized/{stem}.wav", "时长（秒）": "1.0", "原始采样率": "44100",
                     "质检结果": "合格", "上传时间": "", "SHA-256": stem})
    write_csv_rows(pool_paths(root)["manifest"], MANIFEST_COLUMNS, rows)
    items = pool_items(root)
    assert [item["stem"] for item in items] == ["G1-S1-Q", "G2-S1-N"]  # 按文件编号排好
    first = items[0]
    assert first["wav"] == root / "normalized" / "G1-S1-Q.wav"
    assert first["reference"] == root / "references" / "G1-S1-Q.txt"
    assert first["group"] == 1 and first["condition"] == "Q"
    assert [item["stem"] for item in pool_items(root, files=["G2-S1-N"])] == ["G2-S1-N"]
    assert [item["stem"] for item in pool_items(root, files=["G2-S1-N.wav"])] == ["G2-S1-N"]
    with pytest.raises(ValueError, match="G3-S1-Q"):
        pool_items(root, files=["G3-S1-Q"])


def test_eval_cer_empty_pool_message(tmp_path, cfg):
    with pytest.raises(ValueError, match="还没有入池"):
        eval_cer(tmp_path / "empty_pool", cfg)


def test_eval_cer_missing_reference(tmp_path, cfg):
    root = tmp_path / "pool"
    write_csv_rows(pool_paths(root)["manifest"], MANIFEST_COLUMNS,
                   [{"文件编号": "G1-S1-Q", "转换后文件": "normalized/G1-S1-Q.wav"}])
    (root / "normalized").mkdir()
    sf.write(root / "normalized" / "G1-S1-Q.wav", np.zeros(1600, dtype=np.float32), 16000, subtype="PCM_16")
    # 没有参考文本：在开始识别之前就报错（不白等几十分钟）
    with pytest.raises(FileNotFoundError, match="export_references"):
        eval_cer(root, cfg)


def test_pool_version(tmp_path):
    root = tmp_path / "pool"
    root.mkdir()
    assert pool_version(root) == NOT_FROZEN
    assert "freeze_pool" in NOT_FROZEN
    version = pool_paths(root)["versions"] / "v1"
    version.mkdir(parents=True)
    (version / "checksums.json").write_text("{}", encoding="utf-8")
    assert pool_version(root) == "v1"
    # 按版本名的自然顺序取最新的（v10 在 v9 后面），和 pipeline.pool.current_version 一致
    for name in ("v9", "v10"):
        folder = pool_paths(root)["versions"] / name
        folder.mkdir()
        (folder / "checksums.json").write_text("{}", encoding="utf-8")
    assert pool_version(root) == "v10"


def test_evaluate_cli_numbers(tmp_path):
    evaluate = _load_tool("evaluate.py")
    code = evaluate.main(["numbers", "--method", "normalize=g4", "--out", str(tmp_path)])
    assert code == 0
    text = (tmp_path / "numbers.md").read_text(encoding="utf-8")
    assert LIMITATION in text and "normalize=g4" in text
    rows = _read_csv(tmp_path / "numbers.csv")
    assert {"金额", "电话", "主指标"} <= {row["分组项"] for row in rows}


@pytest.mark.parametrize("metric", ["speakers", "classify", "clips"])
def test_evaluate_cli_later_metrics(metric, capsys):
    evaluate = _load_tool("evaluate.py")
    assert evaluate.main([metric]) == 2
    assert "这个测评将在后续任务中加入" in capsys.readouterr().out


def test_evaluate_cli_method_repeatable_and_checked(capsys):
    evaluate = _load_tool("evaluate.py")
    parser = evaluate.build_parser()
    args = parser.parse_args(["cer", "--method", "denoise=g1", "--method", "hotword=g5"])
    assert args.method == ["denoise=g1", "hotword=g5"]
    assert evaluate.parse_methods(args.method) == {"denoise": "g1", "hotword": "g5"}
    for bad in (["normalize"], ["nope=g1"], ["normalize=nope"]):
        with pytest.raises(ValueError):
            evaluate.parse_methods(bad)
    assert evaluate.main(["numbers", "--method", "normalize=nope"]) == 2
    assert "baseline" in capsys.readouterr().out  # 报错时列出可用的做法


def test_compare_rejects_unrelated_slot(capsys):
    compare = _load_tool("compare.py")
    assert compare.main(["--slot", "denoise", "--method", "g1", "--metric", "numbers"]) == 2
    assert "不影响" in capsys.readouterr().out
