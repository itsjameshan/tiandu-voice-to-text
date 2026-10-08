"""数据池（pipeline/pool.py）和两个命令行工具（tools/ingest_pool.py、tools/export_references.py）的测试。

测试音频全部用 ffmpeg 现场生成（正弦波），不用语音合成（红线第 2 条）。
数据池建在 pytest 的临时文件夹里，不碰仓库里的 data_pool/。
"""
import csv
import importlib.util
import os
import shutil
import stat
import subprocess
import sys

import pytest
import soundfile as sf

from conftest import ROOT
from pipeline.audio import sha256_file
from pipeline.config import load_config
from pipeline.data import script_reference_text
from pipeline.pool import (
    CONDITIONS,
    FILENAME_RE,
    MANIFEST_COLUMNS,
    PROOFREAD_COLUMNS,
    export_references,
    ingest_pool,
    init_pool,
    log_proofread,
    parse_recording_name,
    pool_paths,
)

WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def pool(tmp_path):
    """临时数据池。入池会把原始录音设为只读；测试结束后改回可写，Windows 上 pytest 才能删掉临时文件夹。"""
    root = tmp_path / "pool"
    yield root
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file():
                os.chmod(p, stat.S_IREAD | stat.S_IWRITE)


def _put_raw(pool, src, name):
    """把一个文件以指定名字放进数据池的 raw/。"""
    raw = pool_paths(pool)["raw"]
    raw.mkdir(parents=True, exist_ok=True)
    dst = raw / name
    shutil.copy2(src, dst)
    return dst


def _read_csv(path):
    """读 CSV，返回 [表头, 第 1 行, ...]（每行是列表）。"""
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


def _read_dicts(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _is_read_only(path) -> bool:
    return not (os.stat(path).st_mode & WRITE_BITS)


# ---------- 计划里列出的测试 ----------


@pytest.mark.parametrize("name, ok", [
    ("G1-S1-Q.wav", True),
    ("g1-s1-q.wav", False),
    ("G1-S1-Q (1).m4a", False),
    ("G9-S1-Q.wav", False),
    ("G1-S4-Q.wav", False),
    ("G1-S1-X.wav", False),
    ("G1_S1_Q.wav", False),
])
def test_parse_names(name, ok):
    if ok:
        assert parse_recording_name(name)["stem"] == "G1-S1-Q"
    else:
        with pytest.raises(ValueError) as err:
            parse_recording_name(name)
        assert "G1-S1-Q" in str(err.value)


def test_ingest_pool(make_audio, pool, cfg):
    _put_raw(pool, make_audio("tone", "m4a", seconds=3, name="tone.m4a"), "G1-S1-Q.m4a")
    _put_raw(pool, make_audio("tone", "wav", seconds=1, name="other.wav"), "bad name.wav")

    result = ingest_pool(pool, cfg)
    assert len(result["added"]) == 1
    assert len(result["errors"]) == 1
    assert result["errors"][0]["file"] == "bad name.wav"
    assert "G1-S1-Q" in result["errors"][0]["reason"]

    paths = pool_paths(pool)
    assert (paths["normalized"] / "G1-S1-Q.wav").is_file()
    rows = _read_csv(paths["manifest"])
    assert rows[0] == ["文件编号", "剧本编号", "录音条件", "原始文件", "转换后文件",
                       "时长（秒）", "原始采样率", "质检结果", "上传时间", "SHA-256"]
    assert len(rows) == 2  # 表头 + 一行

    again = ingest_pool(pool, cfg)
    assert len(again["skipped"]) == 1
    assert len(again["added"]) == 0
    assert len(_read_csv(paths["manifest"])) == 2


def test_export_references(pool):
    written = export_references(pool)
    folder = pool_paths(pool)["references"]
    assert len(written) == 72
    assert len(list(folder.glob("*.txt"))) == 72

    first = folder / "G1-S1-Q.txt"
    assert first.read_text(encoding="utf-8") == script_reference_text("G1-S1")

    # 改写（模拟校对过）后再运行：不覆盖
    first.write_text("校对过的参考文本", encoding="utf-8")
    assert export_references(pool) == []
    assert first.read_text(encoding="utf-8") == "校对过的参考文本"


def test_log_proofread(pool):
    log_proofread(pool, "G1-S1-Q", "1234")
    log_proofread(pool, "G1-S1-Q", "5678", note="第 3 句听不清")
    rows = _read_dicts(pool_paths(pool)["proofread_log"])
    assert len(rows) == 2
    assert [r["校对人"] for r in rows] == ["1234", "5678"]


# ---------- 补充的测试：文件名 ----------


def test_constants():
    assert FILENAME_RE.pattern == r"^G([1-8])-S([1-3])-([QNF])\.([A-Za-z0-9]+)$"
    assert CONDITIONS == {"Q": "安静", "N": "嘈杂教室", "F": "口袋或远距离"}
    assert MANIFEST_COLUMNS == ["文件编号", "剧本编号", "录音条件", "原始文件", "转换后文件",
                                "时长（秒）", "原始采样率", "质检结果", "上传时间", "SHA-256"]


def test_parse_fields():
    info = parse_recording_name("G3-S2-F.mp3")
    assert info["stem"] == "G3-S2-F"
    assert info["script_id"] == "G3-S2"
    assert info["group"] == 3
    assert info["condition"] == "F"
    # 手机导出的文件常是大写扩展名，也算合格
    assert parse_recording_name("G8-S3-N.WAV")["stem"] == "G8-S3-N"


@pytest.mark.parametrize("name, hint", [
    ("g1-s1-q.wav", "小写"),
    ("G1-S1-Q (1).m4a", "(1)"),
    ("bad name.wav", "空格"),
    ("G9-S1-Q.wav", "组号"),
    ("G1-S4-Q.wav", "剧本号"),
    ("G1-S1-X.wav", "条件代码"),
    ("G1_S1_Q.wav", "下划线"),
    ("G1-S1-Q.m4a.m4a", "两个点"),
    ("G1-S1-安静.m4a", "中文"),
    ("G1-S1-Q", "没有扩展名"),
    ("G1-S1-Q.", "点后面没有扩展名"),
    ("G01-S1-Q.wav", "不要加 0"),
    ("G1-S01-Q.wav", "不要加 0"),
    ("G1S1Q.wav", "连字符"),
    ("G1-S1Q.wav", "连字符"),
    ("G1-S1-Q-2.m4a", "\"-2\""),
    ("G1-S1-Q2.m4a", "\"2\""),
])
def test_parse_names_explain_reason(name, hint):
    """报错时说清楚哪里不对（不只是笼统的"格式不对"），并给出正确示例。"""
    with pytest.raises(ValueError) as err:
        parse_recording_name(name)
    message = str(err.value)
    assert hint in message, message
    assert "格式不对" not in message, message
    assert "G1-S1-Q.m4a" in message


def test_parse_double_extension_not_called_lowercase():
    """G1-S1-Q.m4a.m4a 的问题是扩展名重复，不能误报"用了小写字母"。"""
    with pytest.raises(ValueError) as err:
        parse_recording_name("G1-S1-Q.m4a.m4a")
    assert "小写" not in str(err.value)


def test_parse_rejects_trailing_newline():
    with pytest.raises(ValueError):
        parse_recording_name("G1-S1-Q.wav\n")


# ---------- 补充的测试：文件夹 ----------


def test_pool_paths_and_init(tmp_path):
    root = tmp_path / "pool"
    paths = pool_paths(root)
    for key in ["raw", "normalized", "references", "annotations_speakers", "annotations_clips", "digits",
                "versions", "asr_cache", "manifest", "qc_report", "proofread_log", "acceptance",
                "acceptance_summary"]:
        assert key in paths, key
    assert paths["manifest"] == root / "manifest.csv"
    assert paths["annotations_speakers"] == root / "annotations" / "speakers"
    assert paths["asr_cache"] == root / "asr_cache"

    init_pool(root)
    for key in ["raw", "normalized", "references", "annotations_speakers", "annotations_clips", "digits",
                "versions", "asr_cache"]:
        assert paths[key].is_dir(), key
    assert not paths["manifest"].exists()  # 表格在用到时才写
    init_pool(root)  # 再运行一次也不报错


# ---------- 补充的测试：入池 ----------


def test_ingest_details(make_audio, pool, cfg):
    raw_file = _put_raw(pool, make_audio("tone", "m4a", seconds=3, name="tone.m4a"), "G1-S1-Q.m4a")
    sha = sha256_file(raw_file)

    result = ingest_pool(pool, cfg)
    added = result["added"][0]
    assert added["file"] == "G1-S1-Q.m4a"
    assert added["stem"] == "G1-S1-Q"
    assert abs(added["duration"] - 3.0) < 0.2
    assert any("时长" in p for p in added["qc"])  # 3 秒，剧本预计约 4.5 分钟
    assert added["replaced"] is False

    paths = pool_paths(pool)
    info = sf.info(str(paths["normalized"] / "G1-S1-Q.wav"))
    assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")
    assert _is_read_only(raw_file)
    assert sha256_file(raw_file) == sha  # 原始文件没被改

    row = _read_dicts(paths["manifest"])[0]
    assert row["文件编号"] == "G1-S1-Q"
    assert row["剧本编号"] == "G1-S1"
    assert row["录音条件"] == "安静"
    assert row["原始文件"] == "raw/G1-S1-Q.m4a"
    assert row["转换后文件"] == "normalized/G1-S1-Q.wav"
    assert row["原始采样率"] == "44100"
    assert row["SHA-256"] == sha
    assert "时长" in row["质检结果"]
    assert abs(float(row["时长（秒）"]) - 3.0) < 0.2
    assert len(row["上传时间"]) == len("2026-10-07 09:30:00")

    qc_rows = _read_dicts(paths["qc_report"])
    assert len(qc_rows) == 1
    assert qc_rows[0]["文件编号"] == "G1-S1-Q"
    assert qc_rows[0]["预计时长（秒）"] == "267.6"  # 4.46 分钟
    assert qc_rows[0]["问题数"] == "1"


def test_ingest_good_recording_is_ok(make_audio, pool, cfg):
    """时长与剧本预计时长接近、音量正常的录音：质检合格。"""
    plan_seconds = 4.46 * 60  # G1-S1 的预计时长
    src = make_audio("tone", "wav", seconds=plan_seconds, sr=16000, channels=1, name="long.wav")
    _put_raw(pool, src, "G1-S1-Q.wav")
    result = ingest_pool(pool, cfg)
    assert result["added"][0]["qc"] == []
    assert _read_dicts(pool_paths(pool)["manifest"])[0]["质检结果"] == "合格"


def test_ingest_rerecord_replaces_old_version(make_audio, pool, cfg):
    """复录：同一个编号换了新文件（指纹不同）→ 替换旧版本，清单里仍只有一行，旧的识别缓存删掉。"""
    first = _put_raw(pool, make_audio("tone", "wav", seconds=2, name="a.wav"), "G2-S1-N.wav")
    ingest_pool(pool, cfg)
    cache = pool_paths(pool)["asr_cache"] / "G2-S1-N.json"
    cache.write_text("{}", encoding="utf-8")

    os.chmod(first, stat.S_IREAD | stat.S_IWRITE)  # 老师删掉旧文件（Windows 上会提示"只读"）
    first.unlink()
    new = _put_raw(pool, make_audio("tone", "wav", seconds=4, name="b.wav"), "G2-S1-N.wav")

    result = ingest_pool(pool, cfg)
    assert len(result["added"]) == 1
    assert result["added"][0]["replaced"] is True
    rows = _read_dicts(pool_paths(pool)["manifest"])
    assert len(rows) == 1
    assert rows[0]["SHA-256"] == sha256_file(new)
    assert len(_read_dicts(pool_paths(pool)["qc_report"])) == 1
    assert not cache.exists()
    assert sf.info(str(pool_paths(pool)["normalized"] / "G2-S1-N.wav")).duration == pytest.approx(4, abs=0.1)


@pytest.mark.parametrize("locked", ["qc_report.csv", "manifest.csv"])
def test_ingest_interrupted_then_rerun(make_audio, pool, cfg, monkeypatch, locked):
    """入池中途出错（如 qc_report.csv 正用 Excel 打开着，写不进去）：关掉 Excel 再运行一次，
    这个文件要重新完整入池（质检报告有一行、原始文件只读），不能被当成"已经入池"跳过。"""
    import pipeline.pool

    raw_file = _put_raw(pool, make_audio("tone", "m4a", seconds=2, name="a.m4a"), "G1-S1-Q.m4a")
    real_write = pipeline.pool.write_csv_rows

    def write_but_locked(path, columns, rows):
        if os.path.basename(path) == locked:
            raise PermissionError(13, "Permission denied", str(path))
        real_write(path, columns, rows)

    monkeypatch.setattr(pipeline.pool, "write_csv_rows", write_but_locked)
    with pytest.raises(PermissionError):
        ingest_pool(pool, cfg)
    paths = pool_paths(pool)
    assert not paths["manifest"].exists()  # 清单最后写：还没写进清单，就还不算入池

    monkeypatch.setattr(pipeline.pool, "write_csv_rows", real_write)  # 关掉 Excel 后再运行
    result = ingest_pool(pool, cfg)
    assert [a["stem"] for a in result["added"]] == ["G1-S1-Q"]
    assert result["skipped"] == []
    assert [r["文件编号"] for r in _read_dicts(paths["qc_report"])] == ["G1-S1-Q"]
    assert [r["文件编号"] for r in _read_dicts(paths["manifest"])] == ["G1-S1-Q"]
    assert _is_read_only(raw_file)
    assert not list(paths["normalized"].glob("*.tmp.wav"))


def test_ingest_skipped_file_set_read_only_again(make_audio, pool, cfg):
    """已入池的原始文件被人去掉了"只读"：再运行时跳过它，但把只读补设上。"""
    raw_file = _put_raw(pool, make_audio("tone", "wav", seconds=1, name="a.wav"), "G1-S1-Q.wav")
    ingest_pool(pool, cfg)
    os.chmod(raw_file, stat.S_IREAD | stat.S_IWRITE)
    assert not _is_read_only(raw_file)

    result = ingest_pool(pool, cfg)
    assert result["skipped"] == ["G1-S1-Q.wav"]
    assert _is_read_only(raw_file)


def test_ingest_rerecord_failed_conversion_keeps_old_wav(make_audio, pool, cfg, monkeypatch):
    """复录的新文件转到一半失败：旧的 normalized/<编号>.wav 不能被写坏，清单也不变。"""
    import pipeline.pool

    first = _put_raw(pool, make_audio("tone", "wav", seconds=2, name="a.wav"), "G2-S1-N.wav")
    ingest_pool(pool, cfg)
    paths = pool_paths(pool)
    wav = paths["normalized"] / "G2-S1-N.wav"
    old_bytes = wav.read_bytes()
    old_manifest = _read_dicts(paths["manifest"])

    os.chmod(first, stat.S_IREAD | stat.S_IWRITE)
    first.unlink()
    _put_raw(pool, make_audio("tone", "wav", seconds=4, name="b.wav"), "G2-S1-N.wav")

    def broken_convert(src, dst):
        with open(dst, "wb") as f:  # ffmpeg 写了一半就出错
            f.write(b"RIFF only half written")
        raise RuntimeError("ffmpeg 转换失败：G2-S1-N.wav\nError while decoding stream")

    monkeypatch.setattr(pipeline.pool, "convert_to_wav", broken_convert)
    result = ingest_pool(pool, cfg)
    assert result["added"] == []
    assert [e["file"] for e in result["errors"]] == ["G2-S1-N.wav"]
    assert wav.read_bytes() == old_bytes
    assert not list(paths["normalized"].glob("*.tmp.wav"))
    assert _read_dicts(paths["manifest"]) == old_manifest


def test_ingest_two_files_same_recording(make_audio, pool, cfg):
    """同一个录音交了两个文件（扩展名不同）：两个都报错，不猜用哪个。"""
    _put_raw(pool, make_audio("tone", "wav", seconds=1, name="a.wav"), "G1-S1-Q.wav")
    _put_raw(pool, make_audio("tone", "m4a", seconds=1, name="b.m4a"), "G1-S1-Q.m4a")
    result = ingest_pool(pool, cfg)
    assert result["added"] == []
    assert sorted(e["file"] for e in result["errors"]) == ["G1-S1-Q.m4a", "G1-S1-Q.wav"]
    assert all("只能" in e["reason"] for e in result["errors"])
    assert not pool_paths(pool)["manifest"].exists()


def test_ingest_same_content_under_two_names(make_audio, pool, cfg):
    """同一个录音改成两个名字交了两次（指纹相同）：第二个报错，提醒可能交重了。"""
    src = make_audio("tone", "wav", seconds=1, name="a.wav")
    _put_raw(pool, src, "G1-S1-Q.wav")
    _put_raw(pool, src, "G2-S1-Q.wav")
    result = ingest_pool(pool, cfg)
    assert [a["stem"] for a in result["added"]] == ["G1-S1-Q"]
    assert [e["file"] for e in result["errors"]] == ["G2-S1-Q.wav"]
    assert "G1-S1-Q" in result["errors"][0]["reason"]


def test_ingest_bad_files_do_not_crash(make_audio, pool, cfg):
    paths = pool_paths(pool)
    raw = paths["raw"]
    raw.mkdir(parents=True)
    (raw / "G1-S1-Q.amr").write_bytes(b"#!AMR\n")  # 不支持的格式
    (raw / "G1-S1-N.m4a").write_text("这不是录音", encoding="utf-8")  # 损坏的文件
    (raw / "Thumbs.db").write_bytes(b"x")  # Windows 自动生成的文件：忽略
    (raw / "desktop.ini").write_text("[.ShellClassInfo]", encoding="utf-8")
    (raw / "G1").mkdir()  # 文件夹
    _put_raw(pool, make_audio("tone", "wav", seconds=1, name="ok.wav"), "G1-S1-F.wav")

    result = ingest_pool(pool, cfg)
    assert [a["stem"] for a in result["added"]] == ["G1-S1-F"]
    reasons = {e["file"]: e["reason"] for e in result["errors"]}
    assert set(reasons) == {"G1-S1-Q.amr", "G1-S1-N.m4a", "G1"}
    assert "格式" in reasons["G1-S1-Q.amr"]
    assert "文件夹" in reasons["G1"]
    assert [r["文件编号"] for r in _read_dicts(paths["manifest"])] == ["G1-S1-F"]


def test_ingest_empty_pool(tmp_path, cfg):
    root = tmp_path / "new_pool"
    assert ingest_pool(root, cfg) == {"added": [], "skipped": [], "errors": []}
    assert pool_paths(root)["raw"].is_dir()


# ---------- 补充的测试：参考文本、校对记录 ----------


def test_export_references_overwrite(pool):
    export_references(pool)
    first = pool_paths(pool)["references"] / "G8-S3-F.txt"
    first.write_text("改过", encoding="utf-8")
    written = export_references(pool, overwrite=True)
    assert len(written) == 72
    assert first.read_text(encoding="utf-8") == script_reference_text("G8-S3")


def test_log_proofread_round_numbers(pool):
    """第几遍：第一个校对人是第 1 遍，第二个不同的人是第 2 遍；同一个人再保存一次仍是他的那一遍。"""
    log_proofread(pool, "G1-S1-Q", "1234")
    log_proofread(pool, "G1-S1-Q", "5678", note="第 3 句听不清")
    log_proofread(pool, "G1-S1-Q", "1234")
    log_proofread(pool, "G2-S1-Q", "5678")
    path = pool_paths(pool)["proofread_log"]
    assert _read_csv(path)[0] == PROOFREAD_COLUMNS
    rows = _read_dicts(path)
    assert [r["第几遍"] for r in rows] == ["1", "2", "1", "1"]
    assert rows[1]["备注"] == "第 3 句听不清"
    assert rows[0]["校对时间"]


def test_log_proofread_appends(pool, monkeypatch):
    """校对记录是在文件末尾追加一行，不重写整张表：几台电脑同时往共享文件夹里的同一张表写时，
    不会把别人刚写进去的行冲掉。"""
    import pipeline.pool

    path = pool_paths(pool)["proofread_log"]
    log_proofread(pool, "G1-S1-Q", "1234")
    before = path.read_bytes()

    real_read = pipeline.pool.read_csv_rows

    def read_then_other_computer_saves(p):
        rows = real_read(p)
        # 刚读完，另一台电脑上的同学也保存了一次（在文件末尾追加了一行）
        with open(p, "a", encoding="utf-8", newline="") as f:
            f.write("G1-S1-Q,9999,2026-10-07 10:00:00,2,\r\n")
        monkeypatch.setattr(pipeline.pool, "read_csv_rows", real_read)
        return rows

    monkeypatch.setattr(pipeline.pool, "read_csv_rows", read_then_other_computer_saves)
    log_proofread(pool, "G1-S1-Q", "5678")
    after = path.read_bytes()
    assert after.startswith(before)
    assert after.count(b"\xef\xbb\xbf") == 1  # 只在文件开头有一个 BOM
    assert [r["校对人"] for r in _read_dicts(path)] == ["1234", "9999", "5678"]


@pytest.mark.parametrize("encoding, ending", [
    ("gb18030", "\r\n"),  # 老师用 Excel 另存为普通 CSV：中文 Windows 存成 GBK 编码
    ("utf-8-sig", ""),  # 最后一行后面没有换行
])
def test_log_proofread_after_excel_resave(pool, encoding, ending):
    """表格被 Excel 另存过、不能直接追加时：整张表读出来再统一存回 UTF-8 带 BOM，旧行不丢。"""
    path = pool_paths(pool)["proofread_log"]
    path.parent.mkdir(parents=True, exist_ok=True)
    old = "\r\n".join([",".join(PROOFREAD_COLUMNS), "G1-S1-Q,1234,2026-10-07 09:30:00,1,第 3 句听不清"]) + ending
    path.write_bytes(old.encode(encoding))

    log_proofread(pool, "G1-S1-Q", "5678")
    rows = _read_dicts(path)  # 用 UTF-8 能读
    assert [(r["校对人"], r["第几遍"]) for r in rows] == [("1234", "1"), ("5678", "2")]
    assert rows[0]["备注"] == "第 3 句听不清"


def test_log_proofread_rejects_bad_input(pool):
    with pytest.raises(ValueError, match="G1-S1-Q"):
        log_proofread(pool, "G9-S1-Q", "1234")
    with pytest.raises(ValueError, match="校对人"):
        log_proofread(pool, "G1-S1-Q", "   ")


# ---------- 补充的测试：命令行工具 ----------


def _run_tool(script, *args):
    """像老师在命令行里那样运行 python tools/xxx.py（在别的文件夹里运行，检验工具自己能找到 pipeline）。"""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, str(ROOT / "tools" / script), *args], capture_output=True,
                          text=True, encoding="utf-8", env=env, cwd=str(ROOT.parent))


def _load_tool(script):
    spec = importlib.util.spec_from_file_location(script.removesuffix(".py"), ROOT / "tools" / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_ingest_pool(make_audio, pool):
    _put_raw(pool, make_audio("tone", "m4a", seconds=2, name="a.m4a"), "G1-S1-Q.m4a")
    bad = _put_raw(pool, make_audio("tone", "wav", seconds=1, name="b.wav"), "G1-S1-Q (1).wav")

    proc = _run_tool("ingest_pool.py", "--pool", str(pool))
    assert proc.returncode == 1, proc.stdout + proc.stderr  # 有不合格的文件
    assert "G1-S1-Q (1).wav" in proc.stdout
    assert "新入池 1 个" in proc.stdout
    assert "未处理 1 个" in proc.stdout

    bad.unlink()
    proc = _run_tool("ingest_pool.py", "--pool", str(pool))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "跳过 1 个" in proc.stdout


def test_cli_ingest_table_locked_then_rerun(make_audio, pool, monkeypatch, capsys):
    """质检报告正用 Excel 打开着：提示关掉再运行；关掉后再运行一次，这个文件正常入池。"""
    import pipeline.pool

    _put_raw(pool, make_audio("tone", "m4a", seconds=2, name="a.m4a"), "G1-S1-Q.m4a")
    real_write = pipeline.pool.write_csv_rows

    def write_but_locked(path, columns, rows):
        if os.path.basename(path) == "qc_report.csv":
            raise PermissionError(13, "Permission denied", str(path))
        real_write(path, columns, rows)

    tool = _load_tool("ingest_pool.py")
    monkeypatch.setattr(pipeline.pool, "write_csv_rows", write_but_locked)
    assert tool.main(["--pool", str(pool)]) == 1
    out = capsys.readouterr().out
    assert "qc_report.csv" in out
    assert "请先关闭" in out

    monkeypatch.setattr(pipeline.pool, "write_csv_rows", real_write)
    assert tool.main(["--pool", str(pool)]) == 0
    out = capsys.readouterr().out
    assert "新入池 1 个" in out
    assert "时长 2.0 秒" in out


def test_cli_ingest_empty_raw(pool):
    proc = _run_tool("ingest_pool.py", "--pool", str(pool))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "还没有录音" in proc.stdout


def test_cli_export_references(pool):
    proc = _run_tool("export_references.py", "--pool", str(pool))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "72" in proc.stdout
    assert len(list(pool_paths(pool)["references"].glob("*.txt"))) == 72

    proc = _run_tool("export_references.py", "--pool", str(pool))
    assert proc.returncode == 0
    assert "没有覆盖" in proc.stdout

    proc = _run_tool("export_references.py", "--pool", str(pool), "--overwrite")
    assert proc.returncode == 0
    assert "覆盖" in proc.stdout


def test_cli_uses_config_pool_by_default(tmp_path, monkeypatch, capsys):
    """不写 --pool 时用 config.yaml 里的 paths.data_pool。"""
    import pipeline.config

    cfg = load_config()
    cfg["paths"]["data_pool"] = str(tmp_path / "cfg_pool")
    monkeypatch.setattr(pipeline.config, "load_config", lambda *a, **k: cfg)

    assert _load_tool("export_references.py").main([]) == 0
    assert len(list((tmp_path / "cfg_pool" / "references").glob("*.txt"))) == 72
    assert _load_tool("ingest_pool.py").main([]) == 0
    assert str(tmp_path / "cfg_pool") in capsys.readouterr().out


@pytest.mark.parametrize("locked", ["qc_report.csv", "manifest.csv"])
def test_rerecord_interrupted_keeps_old_wav(make_audio, pool, cfg, monkeypatch, locked):
    """复录时写表格出错（如 manifest.csv 正用 Excel 打开着）：normalized 里要换回旧录音，
    和清单里记的指纹、时长一致；关掉 Excel 再运行一次，新录音才正式换上。"""
    import pipeline.pool
    from pipeline.audio import sha256_file

    _put_raw(pool, make_audio("tone", "wav", seconds=2, name="a.wav"), "G1-S1-Q.wav")
    ingest_pool(pool, cfg)
    paths = pool_paths(pool)
    wav = paths["normalized"] / "G1-S1-Q.wav"
    old_sha = sha256_file(wav)
    old_manifest = _read_dicts(paths["manifest"])

    old_raw = paths["raw"] / "G1-S1-Q.wav"  # 复录：先把旧文件移走，新录音用同一个名字放进去
    os.chmod(old_raw, stat.S_IREAD | stat.S_IWRITE)
    old_raw.unlink()
    _put_raw(pool, make_audio("tone", "wav", seconds=4, name="b.wav"), "G1-S1-Q.wav")
    real_write = pipeline.pool.write_csv_rows

    def write_but_locked(path, columns, rows):
        if os.path.basename(path) == locked:
            raise PermissionError(13, "Permission denied", str(path))
        real_write(path, columns, rows)

    monkeypatch.setattr(pipeline.pool, "write_csv_rows", write_but_locked)
    with pytest.raises(PermissionError):
        ingest_pool(pool, cfg)
    assert sha256_file(wav) == old_sha, "写表格失败时要换回旧录音，和清单一致"
    assert _read_dicts(paths["manifest"]) == old_manifest
    assert not list(paths["normalized"].glob("*.tmp.wav")), "临时文件要删掉"

    monkeypatch.setattr(pipeline.pool, "write_csv_rows", real_write)  # 关掉 Excel 后再运行
    result = ingest_pool(pool, cfg)
    assert [a["stem"] for a in result["added"]] == ["G1-S1-Q"]
    assert sha256_file(wav) != old_sha
    assert float(_read_dicts(paths["manifest"])[0]["时长（秒）"]) == pytest.approx(4, abs=0.1)
