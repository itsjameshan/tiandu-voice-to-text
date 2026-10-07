"""测评库（pipeline/evaluation.py）和两个命令行工具（tools/evaluate.py、tools/compare.py）的测试。

红线：云端开发环境里不报告任何字错率数字。这里只检查流程能跑通、字段齐全、缓存有效，
不打印、不记录字错率的具体数值。
测试音频：模型自带的 0-four-speakers-zh.wav 改名成 G1-S1-Q.wav 放进临时数据池（不用语音合成）。
说话人标注、片段标注是测试里手写的（不是这段音频真正的说话时间），只用来检查流程和计算接得对不对。
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
from pipeline.annotations import write_clips_csv, write_turns_csv
from pipeline.config import load_config
from pipeline.evaluation import (
    FIXED_LIMITATION,
    NOT_FROZEN,
    annotated_items,
    cache_key,
    clip_details,
    count_names,
    eval_cer,
    eval_classify_rules,
    eval_clips,
    eval_hotwords,
    eval_numbers_lines,
    eval_speakers,
    hotword_stats,
    parse_num_speakers,
    pool_items,
    pool_version,
    write_report,
)
from pipeline.metrics import false_positive_rate, speaker_error_details
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


@pytest.mark.parametrize("metric", ["speakers", "clips"])
def test_evaluate_cli_annotation_metrics_on_empty_pool(metric, tmp_path, capsys):
    """speakers、clips 已经做好（不再提示"后续任务"）；数据池是空的时给中文提示，退出码 1。"""
    evaluate = _load_tool("evaluate.py")
    assert evaluate.main([metric, "--pool", str(tmp_path / "empty_pool")]) == 1
    out = capsys.readouterr().out
    assert "还没有入池" in out
    assert "后续任务" not in out


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


# ======================== Task 19：speakers / classify / clips ========================


def _fake_pool(root, stems):
    """只有清单和转换后录音（0.1 秒静音）的小数据池，不用模型。"""
    rows = [{"文件编号": stem, "转换后文件": f"normalized/{stem}.wav"} for stem in stems]
    write_csv_rows(pool_paths(root)["manifest"], MANIFEST_COLUMNS, rows)
    (root / "normalized").mkdir(exist_ok=True)
    for stem in stems:
        sf.write(root / "normalized" / f"{stem}.wav", np.zeros(1600, dtype=np.float32), 16000, subtype="PCM_16")


GROUP_CLASSIFIERS = {"g6": "pipeline.groups.g6_classifier_a", "g7": "pipeline.groups.g7_classifier_b"}


def _no_g6_model(monkeypatch, tmp_path, method="g6"):
    """让第 6 组（或第 7 组）的做法找不到训练好的模型（不管这台电脑有没有 TensorFlow、有没有训练过），一定退回关键词规则。"""
    import importlib

    importlib.import_module(GROUP_CLASSIFIERS[method])
    monkeypatch.setattr(f"{GROUP_CLASSIFIERS[method]}.classifier_dir",
                        lambda cfg, name="classifier": tmp_path / "no_models" / name)


def test_eval_classify_rules(cfg):
    from pipeline.data import LABEL_NAMES, load_lines

    result = eval_classify_rules(cfg)
    matrix = result["confusion"]
    assert len(matrix) == 7 and all(len(row) == 7 for row in matrix)
    assert sum(sum(row) for row in matrix) == len(load_lines()) == 1055
    assert "fp_rate_g8" in result
    assert 0 <= result["fp_rate"] <= 1 and 0 <= result["fp_rate_g8"] <= 1
    assert result["normal_count_g8"] > 0  # 第 8 组（正常讲解对照组）剧本里有正常讲解
    assert result["normal_count"] >= result["normal_count_g8"]
    assert result["labels"] == LABEL_NAMES and list(result["per_class"]) == LABEL_NAMES
    assert result["warnings"] == []
    assert result["info"]["pool_version"]  # 每份报告都写明数据来源
    assert result["info"]["methods"] == {"classify": "baseline"}
    # 每句台词一行；误报率和 pipeline.metrics 的定义一致（用类别名，不是"疑似·…"标签）
    predictions = result["predictions"]
    assert len(predictions) == 1055
    gold = [row["label"] for row in predictions]
    pred = [row["predicted"] for row in predictions]
    assert result["fp_rate"] == false_positive_rate(gold, pred)
    g8 = [i for i, row in enumerate(predictions) if row["group"] == 8]
    assert result["fp_rate_g8"] == false_positive_rate([gold[i] for i in g8], [pred[i] for i in g8])
    # 对比表用的汇总：比例 = 分子 ÷ 分母
    labels = [row["分组项"] for row in result["overview"]]
    assert labels[:3] == ["7 类正确率", "误报率（全部剧本）", "误报率（第 8 组剧本）"]
    assert "威胁消费·召回率" in labels and "费用·准确率" in labels and "第 8 组·7 类正确率" in labels
    for row in result["overview"]:
        assert row["rate"] == (row["count"] / row["n"] if row["n"] else 0.0)


@pytest.mark.parametrize("method", ["g6", "g7"])
def test_eval_classify_group_model_falls_back_to_rules_with_warning(cfg, monkeypatch, tmp_path, method):
    _no_g6_model(monkeypatch, tmp_path, method)
    base = eval_classify_rules(cfg)
    result = eval_classify_rules(cfg, method=method)
    assert result["warnings"] and all("关键词规则" in w for w in result["warnings"])
    assert result["confusion"] == base["confusion"]
    assert result["info"]["methods"] == {"classify": method}
    with pytest.raises(ValueError, match="g99"):
        eval_classify_rules(cfg, method="g99")


def test_eval_classify_rejects_display_labels(cfg, monkeypatch):
    """做法返回了"疑似·费用"这样的显示标签（应该返回类别名"费用"）：明确报错。"""
    from pipeline.methods import _REGISTRY

    monkeypatch.setitem(_REGISTRY["classify"], "shown", lambda texts, cfg: ["疑似·费用"] * len(texts))
    with pytest.raises(ValueError, match="疑似·费用"):
        eval_classify_rules(cfg, method="shown")
    monkeypatch.setitem(_REGISTRY["classify"], "short", lambda texts, cfg: ["费用"])
    with pytest.raises(ValueError, match="1055"):
        eval_classify_rules(cfg, method="short")


def test_evaluate_cli_classify(tmp_path, capsys):
    evaluate = _load_tool("evaluate.py")
    assert evaluate.main(["classify", "--out", str(tmp_path)]) == 0
    text = (tmp_path / "classify.md").read_text(encoding="utf-8")
    assert LIMITATION in text and "数据池版本" in text
    assert "参照剧本台词写的" in text  # 关键词规则是参照剧本写的，数字偏乐观
    assert "误报率（第 8 组剧本）" in text and "混淆矩阵" in text
    assert "classify=baseline" in text
    rows = _read_csv(tmp_path / "classify.csv")
    assert len(rows) == 1055
    assert {"剧本编号", "台词", "标准答案", "预测", "对错"} <= set(rows[0])
    assert "误报率" in capsys.readouterr().out


def test_compare_classify_g6_without_model(tmp_path, monkeypatch):
    _no_g6_model(monkeypatch, tmp_path)
    compare = _load_tool("compare.py")
    out = tmp_path / "g6"
    assert compare.main(["--metric", "classify", "--slot", "classify", "--method", "g6", "--out", str(out)]) == 0
    rows = _read_csv(out / "compare_classify.csv")
    labels = [row["分组项"] for row in rows]
    assert labels[:3] == ["7 类正确率", "误报率（全部剧本）", "误报率（第 8 组剧本）"]
    assert "第 8 组·7 类正确率" in labels
    assert all(float(row["差值"]) == 0 for row in rows)  # 没有模型：g6 退回关键词规则，和基线一样
    text = (out / "compare_classify.md").read_text(encoding="utf-8")
    assert LIMITATION in text and "关键词规则" in text  # 退回规则的提示写进了报告
    assert compare.main(["--metric", "classify", "--slot", "denoise", "--method", "g1"]) == 2


def test_speaker_error_details_counts_seconds():
    ref = [(0, 10, "导游"), (10, 20, "游客甲")]
    hyp = [(0, 10, "说话人1"), (10, 15, "说话人2"), (15, 20, "说话人1")]
    details = speaker_error_details(ref, hyp)
    # 最优对应：导游↔说话人1（重叠 10 秒）、游客甲↔说话人2（5 秒）；15—20 秒的"说话人1"标错
    assert details["speaker_error_rate"] == pytest.approx(0.25)
    assert details["scored_seconds"] == pytest.approx(20.0)
    assert details["error_seconds"] == pytest.approx(5.0)
    # 只有一边有人说话的时间不比对
    one_side = speaker_error_details([(0, 10, "导游")], [(5, 20, "说话人1")])
    assert one_side["scored_seconds"] == pytest.approx(5.0) and one_side["error_seconds"] == 0
    assert speaker_error_details(ref, [])["scored_seconds"] == 0
    # 两人同时说话：按人数算
    both = speaker_error_details([(0, 10, "导游"), (0, 10, "游客甲")], [(0, 10, "说话人1"), (0, 10, "说话人2")])
    assert both["scored_seconds"] == pytest.approx(20.0) and both["speaker_error_rate"] == 0


def test_clip_details_counts_category_and_outside():
    ref = [(10.0, 20.0, "费用"), (40.0, 50.0, "购物安排")]
    tool = [(9.0, 21.0, "费用"), (41.0, 45.0, "行程变更"), (45.0, 50.0, "购物安排"), (70.0, 75.0, "费用")]
    details = clip_details(ref, tool)
    assert details["ref_clips"] == 2 and details["tool_clips"] == 4 and details["matched"] == 2
    # 配对：(10,20)↔(9,21) 起点差 1、终点差 1；(40,50)↔(45,50)（重叠 5 秒，比 (41,45) 的 4 秒多）起点差 5、终点差 0
    assert details["start_mae"] == pytest.approx(3.0) and details["end_mae"] == pytest.approx(0.5)
    assert details["unmatched_ref"] == 0 and details["unmatched_hyp"] == 2 and details["unmatched"] == 2
    assert details["wrong_category"] == 1  # (41,45) 的"行程变更"落在"购物安排"的标注里
    assert details["tool_outside"] == 1  # (70,75) 和哪个标注片段都不重叠
    empty = clip_details(ref, [])
    assert empty["start_mae"] is None and empty["end_mae"] is None and empty["unmatched"] == 2


def test_parse_num_speakers():
    assert parse_num_speakers("auto") == -1 and parse_num_speakers("自动") == -1 and parse_num_speakers("-1") == -1
    assert parse_num_speakers("4") == 4 and parse_num_speakers(" 3 ") == 3
    assert parse_num_speakers("ref") == "ref" and parse_num_speakers("标注") == "ref"
    for bad in ("四", "x", "2.5", ""):
        with pytest.raises(ValueError, match="--speakers"):
            parse_num_speakers(bad)


def test_annotated_items_only_files_with_annotations(tmp_path):
    root = tmp_path / "pool"
    _fake_pool(root, ["G1-S1-Q", "G3-S1-Q"])
    with pytest.raises(ValueError, match="说话人标注"):
        annotated_items(root, "speakers")
    write_turns_csv(pool_paths(root)["annotations_speakers"] / "G3-S1-Q.csv", [(0.0, 0.1, "导游")])
    items = annotated_items(root, "speakers")
    assert [item["stem"] for item in items] == ["G3-S1-Q"]  # 没有标注的 G1-S1-Q 不评
    assert items[0]["annotation"] == [(0.0, 0.1, "导游")]
    assert items[0]["group"] == 3 and items[0]["condition"] == "Q"
    with pytest.raises(ValueError, match="G1-S1-Q"):  # 指定了没有标注的录音：明确报错
        annotated_items(root, "speakers", files=["G1-S1-Q"])
    with pytest.raises(ValueError, match="疑似片段标注"):
        annotated_items(root, "clips")
    write_clips_csv(pool_paths(root)["annotations_clips"] / "G1-S1-Q.csv", [(0.0, 0.1, "消费施压")])
    assert annotated_items(root, "clips")[0]["annotation"] == [(0.0, 0.1, "威胁消费")]
    with pytest.raises(ValueError, match="还没有入池"):
        annotated_items(tmp_path / "empty_pool", "speakers")


def test_evaluate_cli_rejects_bad_speakers_value(capsys):
    evaluate = _load_tool("evaluate.py")
    assert evaluate.main(["speakers", "--speakers", "四"]) == 2
    assert "--speakers" in capsys.readouterr().out
    compare = _load_tool("compare.py")
    assert compare.main(["--slot", "diarize", "--method", "g3", "--metric", "speakers", "--speakers", "四"]) == 2
    assert "--speakers" in capsys.readouterr().out


# ---------- 需要模型：四人测试音频 + 手写的标注 ----------

# 手写的说话人标注和片段标注（不是这段音频真正的说话时间，只检查流程）
HAND_TURNS = [(0.0, 14.0, "导游"), (14.0, 28.0, "游客甲"), (28.0, 42.0, "游客乙"), (42.0, 56.0, "店员")]
HAND_CLIPS = [(5.0, 15.0, "费用"), (30.0, 40.0, "购物安排")]


@pytest.fixture(scope="module")
def annotated_pool(renamed_pool):
    paths = pool_paths(renamed_pool)
    write_turns_csv(paths["annotations_speakers"] / "G1-S1-Q.csv", HAND_TURNS)
    write_clips_csv(paths["annotations_clips"] / "G1-S1-Q.csv", HAND_CLIPS)
    return renamed_pool


@requires_models
def test_eval_speakers_runs_on_renamed_test_audio(annotated_pool, cfg):
    rows, summary = eval_speakers(annotated_pool, cfg)
    assert len(rows) == 1
    row = rows[0]
    for key in ("stem", "group", "condition", "speaker_error_rate", "annotated_seconds", "scored_seconds",
                "error_seconds", "ref_speakers", "hyp_speakers", "num_speakers", "seconds"):
        assert key in row
    assert row["stem"] == "G1-S1-Q" and row["group"] == 1 and row["condition"] == "Q"
    assert 0 <= row["speaker_error_rate"] <= 1
    assert row["annotated_seconds"] == pytest.approx(56.0)
    assert 0 < row["scored_seconds"] <= row["annotated_seconds"]
    assert row["ref_speakers"] == 4 and row["hyp_speakers"] >= 1
    assert row["num_speakers"] == "自动"  # config.yaml 默认 -1
    assert summary["all"]["files"] == 1
    assert set(summary["by_condition"]) == {"Q"} and set(summary["by_group"]) == {1}
    assert summary["all"]["speaker_error_rate"] == pytest.approx(row["speaker_error_rate"])
    assert summary["info"]["pool_version"] == NOT_FROZEN
    assert summary["info"]["methods"]["diarize"] == "baseline"


@requires_models
def test_eval_speakers_zero_when_annotation_copies_tool_output(annotated_pool, cfg, tmp_path):
    """标注照抄工具的结果（只把"说话人N"换成角色名）：标错比例是 0，说明时间、对应关系都接对了。"""
    from pipeline.audio import read_wav
    from pipeline.evaluation import recognize_item
    from pipeline.step4_diarize import diarize_baseline

    recognize_item(annotated_pool, pool_items(annotated_pool, ["G1-S1-Q"])[0], cfg)  # 确保有识别结果缓存
    copy_root = tmp_path / "pool_copy"
    shutil.copytree(annotated_pool, copy_root)  # 连识别结果缓存一起复制，不用重新识别
    try:
        item = pool_items(copy_root, ["G1-S1-Q"])[0]
        segments, info = recognize_item(copy_root, item, cfg)
        assert info["cached"]
        four = load_config(overrides={"diarize": {"num_speakers": 4}})
        result = diarize_baseline(read_wav(item["wav"]), 16000, segments, four)
        roles = {}
        turns = [(seg["start"], seg["end"], roles.setdefault(seg["speaker"], f"角色{len(roles) + 1}"))
                 for seg in result if seg["speaker"] != "未知"]
        write_turns_csv(pool_paths(copy_root)["annotations_speakers"] / "G1-S1-Q.csv", turns)

        rows, summary = eval_speakers(copy_root, cfg, num_speakers=4)
        assert rows[0]["speaker_error_rate"] == 0
        assert rows[0]["hyp_speakers"] == rows[0]["ref_speakers"] == len(roles)
        assert rows[0]["num_speakers"] == "4"
        assert summary["info"]["params"]["diarize.num_speakers"] == 4
        # 按标注的人数（设对人数）
        rows, summary = eval_speakers(copy_root, cfg, num_speakers="ref")
        assert rows[0]["num_speakers"] == str(len(roles))
        assert "标注" in str(summary["info"]["params"]["diarize.num_speakers"])
    finally:
        _make_writable(copy_root)


@requires_models
def test_eval_clips_with_fixed_methods(annotated_pool, cfg, monkeypatch):
    """用固定的分类和片段做法，检查片段测评的接线：片段做法拿到原始录音、类别取自段落。"""
    from pipeline.methods import _REGISTRY

    seen = []

    def fixed_clips(segments, samples, sr, cfg):
        seen.append((segments, len(samples), sr))
        return [(5.5, 14.0, 0)]

    monkeypatch.setitem(_REGISTRY["classify"], "all_fee", lambda texts, cfg: ["费用"] * len(texts))
    monkeypatch.setitem(_REGISTRY["classify"], "all_shop", lambda texts, cfg: ["购物安排"] * len(texts))
    monkeypatch.setitem(_REGISTRY["clips"], "fixed", fixed_clips)

    rows, summary = eval_clips(annotated_pool, cfg, methods={"classify": "all_fee", "clips": "fixed"})
    assert len(rows) == 1
    row = rows[0]
    assert row["ref_clips"] == 2 and row["tool_clips"] == 1 and row["matched"] == 1
    assert row["start_mae"] == pytest.approx(0.5) and row["end_mae"] == pytest.approx(1.0)
    assert row["unmatched"] == 1 and row["wrong_category"] == 0 and row["tool_outside"] == 0
    segments, n_samples, sr = seen[0]
    assert sr == 16000 and n_samples > 50 * 16000  # 整段原始录音
    assert segments and all(seg["label"] == "疑似·费用" for seg in segments)
    assert all(seg["text_raw"] for seg in segments)  # 测评模式：保留汉字读法
    assert summary["all"]["start_mae"] == pytest.approx(0.5)
    assert summary["info"]["methods"]["clips"] == "fixed" and summary["info"]["pool_version"] == NOT_FROZEN

    rows, _ = eval_clips(annotated_pool, cfg, methods={"classify": "all_shop", "clips": "fixed"})
    assert rows[0]["wrong_category"] == 1  # 工具片段是"购物安排"，标注是"费用"


@requires_models
def test_eval_clips_baseline_and_g6_warning(annotated_pool, cfg, monkeypatch, tmp_path):
    rows, summary = eval_clips(annotated_pool, cfg)
    row = rows[0]
    for key in ("ref_clips", "tool_clips", "matched", "unmatched_ref", "unmatched_hyp", "unmatched",
                "start_mae", "end_mae", "wrong_category", "tool_outside", "seconds"):
        assert key in row
    assert row["ref_clips"] == 2
    assert row["unmatched"] == row["ref_clips"] + row["tool_clips"] - 2 * row["matched"]
    assert summary["warnings"] == []
    _no_g6_model(monkeypatch, tmp_path)
    _, summary = eval_clips(annotated_pool, cfg, methods={"classify": "g6"})
    assert summary["warnings"] and "关键词规则" in summary["warnings"][0]


@requires_models
def test_evaluate_cli_speakers_clips_and_compare(annotated_pool, tmp_path, capsys):
    evaluate = _load_tool("evaluate.py")
    compare = _load_tool("compare.py")

    out = tmp_path / "g3"
    assert evaluate.main(["speakers", "--pool", str(annotated_pool), "--speakers", "4", "--out", str(out)]) == 0
    text = (out / "speakers.md").read_text(encoding="utf-8")
    assert LIMITATION in text and "未冻结" in text and "diarize.num_speakers=4" in text
    rows = _read_csv(out / "speakers.csv")
    assert rows[0]["文件编号"] == "G1-S1-Q" and rows[0]["人数设置"] == "4"
    assert "[1/1] G1-S1-Q" in capsys.readouterr().out

    assert compare.main(["--slot", "diarize", "--method", "g3", "--metric", "speakers",
                         "--pool", str(annotated_pool), "--speakers", "ref", "--out", str(out)]) == 0
    rows = _read_csv(out / "compare_speakers.csv")
    assert [row["分组项"] for row in rows] == ["全体", "Q（安静）", "第 1 组"]
    assert all(float(row["差值"]) == 0 for row in rows)  # g3 初始等于基线

    out = tmp_path / "g8"
    assert evaluate.main(["clips", "--pool", str(annotated_pool), "--out", str(out)]) == 0
    text = (out / "clips.md").read_text(encoding="utf-8")
    assert LIMITATION in text and "起点平均误差" in text
    assert compare.main(["--slot", "clips", "--method", "g8", "--metric", "clips",
                         "--pool", str(annotated_pool), "--out", str(out)]) == 0
    rows = _read_csv(out / "compare_clips.csv")
    assert rows[0]["分组项"].startswith("全体·")
    assert all(row["差值"] == "" or float(row["差值"]) == 0 for row in rows)  # g8 初始等于基线


def test_recognize_item_without_writable_cache(tmp_path, monkeypatch, make_audio, cfg):
    """数据池不能写入（例如只读的共享文件夹，建不了 asr_cache）：照样识别、照样测，只是不留缓存。"""
    import shutil

    from pipeline import step2_vad, step3_asr
    from pipeline.evaluation import recognize_item
    from pipeline.schema import new_segment

    root = tmp_path / "pool"
    root.mkdir()
    (root / "asr_cache").write_text("这里是一个文件，所以建不了 asr_cache 文件夹", encoding="utf-8")
    wav = tmp_path / "G1-S1-Q.wav"
    shutil.copyfile(make_audio("tone", "wav", seconds=1.0, sr=16000, channels=1), wav)
    monkeypatch.setattr(step2_vad, "detect_speech",
                        lambda samples, sr, cfg, methods=None: (samples, [new_segment(0.0, 1.0)]))
    monkeypatch.setattr(step3_asr, "recognize",
                        lambda samples, sr, segs, cfg, mode="display", progress=None:
                        [dict(seg, text_raw="今天去石林") for seg in segs])
    segments, info = recognize_item(root, {"stem": "G1-S1-Q", "wav": wav}, cfg)
    assert [seg["text_raw"] for seg in segments] == ["今天去石林"]
    assert info["cached"] is False
