"""数据池验收表、冻结版本（pipeline/pool.py 的 Task 16 部分）和 tools/check_pool.py、tools/freeze_pool.py 的测试。

测试音频全部用 ffmpeg 现场生成（正弦波），不用语音合成（红线第 2 条）。
数据池建在 pytest 的临时文件夹里，不碰仓库里的 data_pool/。
"""
import csv
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys

import pytest

from conftest import ROOT
from pipeline.annotations import write_clips_csv, write_turns_csv
from pipeline.audio import sha256_file
from pipeline.config import load_config
from pipeline.pool import (
    ACCEPTANCE_COLUMNS,
    PROOFREAD_COLUMNS,
    SUMMARY_COLUMNS,
    acceptance,
    acceptance_summary,
    current_version,
    export_references,
    freeze,
    ingest_pool,
    init_pool,
    log_proofread,
    pool_paths,
    read_csv_rows,
    version_changes,
    write_acceptance,
    write_csv_rows,
)

# G1-S1 剧本的预计时长是 4.46 分钟：生成这么长的正弦波，质检才"合格"
G1_S1_SECONDS = 4.46 * 60


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


@pytest.fixture
def good_recording(make_audio):
    """一段质检能合格的录音（时长与 G1-S1 剧本的预计时长一致、16k 单声道正弦波）。"""
    return make_audio("tone", "wav", seconds=G1_S1_SECONDS, sr=16000, channels=1, name="good.wav")


def _put_raw(pool, src, name):
    raw = pool_paths(pool)["raw"]
    raw.mkdir(parents=True, exist_ok=True)
    dst = raw / name
    shutil.copy2(src, dst)
    return dst


def _row(rows, stem):
    return next(row for row in rows if row["文件"] == stem)


def _read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


# ---------- 计划里列出的测试 ----------


def test_acceptance_flow(pool, cfg, good_recording):
    rows = acceptance(pool)
    assert len(rows) == 72
    assert not any(row["已交"] for row in rows)
    assert not any(row["通过"] for row in rows)

    _put_raw(pool, good_recording, "G1-S1-Q.wav")
    assert len(ingest_pool(pool, cfg)["added"]) == 1
    log_proofread(pool, "G1-S1-Q", "1234")
    log_proofread(pool, "G1-S1-Q", "5678")
    write_turns_csv(pool_paths(pool)["annotations_speakers"] / "G1-S1-Q.csv",
                    [(0.0, 300.0, "导游"), (300.0, 600.0, "游客甲")])

    rows = acceptance(pool)
    row = _row(rows, "G1-S1-Q")
    assert row["已交"] is True
    assert row["命名"] == "合格"
    assert row["质检"] == "合格"
    assert row["校对遍数"] == 2
    assert row["说话人标注（秒）"] == 600.0
    assert row["通过"] is True

    summary = acceptance_summary(rows)
    group1 = next(item for item in summary if item["组"] == 1)
    assert group1["说话人标注总分钟"] == 10.0
    assert group1["是否达到 10 分钟"] is True


def test_freeze(pool, cfg, make_audio):
    _put_raw(pool, make_audio("tone", "wav", seconds=2, name="a.wav"), "G1-S1-Q.wav")
    ingest_pool(pool, cfg)

    target = freeze(pool, "v1")
    assert target == pool_paths(pool)["versions"] / "v1"
    assert target.is_dir()
    assert (target / "checksums.json").is_file()
    with pytest.raises(FileExistsError):
        freeze(pool, "v1")
    assert current_version(pool) == "v1"


# ---------- 补充的测试：验收表 ----------


def test_acceptance_columns_and_plan_order(pool):
    rows = acceptance(pool)
    assert list(rows[0]) == ACCEPTANCE_COLUMNS == [
        "文件", "组", "条件", "已交", "命名", "质检", "校对遍数", "说话人标注（秒）", "片段标注", "通过"]
    assert [row["文件"] for row in rows[:3]] == ["G1-S1-Q", "G1-S1-N", "G1-S1-F"]
    first = rows[0]
    assert (first["组"], first["条件"], first["已交"], first["命名"], first["质检"]) == (1, "安静", False, "", "未入池")
    assert (first["校对遍数"], first["说话人标注（秒）"], first["片段标注"]) == (0, 0.0, False)
    assert not pool.exists()  # 只是看一看，不建文件夹


def test_acceptance_bad_name_counts_as_submitted_but_not_ok(pool, cfg, make_audio):
    """文件名不合格的录音：验收表里显示"已交"和不合格的原因，但不会入池、不会通过（入池时仍然报错、不猜）。"""
    _put_raw(pool, make_audio("tone", "wav", seconds=1, name="a.wav"), "g1-s1-q.wav")
    _put_raw(pool, make_audio("tone", "m4a", seconds=1, name="b.m4a"), "G2-S1-N (1).m4a")
    _put_raw(pool, make_audio("tone", "wav", seconds=1, name="c.wav"), "录音1.wav")  # 对不上任何录音
    ingest_pool(pool, cfg)

    rows = acceptance(pool)
    row = _row(rows, "G1-S1-Q")
    assert row["已交"] is True
    assert "小写" in row["命名"] and "g1-s1-q.wav" in row["命名"]
    assert row["质检"] == "未入池"
    assert row["通过"] is False
    assert "(1)" in _row(rows, "G2-S1-N")["命名"]
    assert sum(row["已交"] for row in rows) == 2


def test_acceptance_qc_problem_and_unsupported_format(pool, cfg, make_audio):
    _put_raw(pool, make_audio("tone", "wav", seconds=2, name="a.wav"), "G1-S1-Q.wav")  # 太短
    raw = pool_paths(pool)["raw"]
    (raw / "G1-S1-N.amr").write_bytes(b"#!AMR\n")  # 名字合格、格式不支持
    ingest_pool(pool, cfg)
    log_proofread(pool, "G1-S1-Q", "1234")
    log_proofread(pool, "G1-S1-Q", "5678")

    rows = acceptance(pool)
    short = _row(rows, "G1-S1-Q")
    assert "时长" in short["质检"]
    assert short["校对遍数"] == 2
    assert short["通过"] is False
    amr = _row(rows, "G1-S1-N")
    assert (amr["已交"], amr["命名"], amr["质检"], amr["通过"]) == (True, "合格", "未入池", False)


def test_acceptance_interrupted_ingest_is_not_ingested(pool):
    """qc_report 里有、manifest 里没有：入池中途出错了，还不算入池。"""
    paths = pool_paths(pool)
    init_pool(pool)
    from pipeline.pool import QC_COLUMNS

    write_csv_rows(paths["qc_report"], QC_COLUMNS, [{"文件编号": "G1-S1-Q", "质检结果": "合格"}])
    row = _row(acceptance(pool), "G1-S1-Q")
    assert row["质检"] == "未入池"
    assert row["通过"] is False


def test_proofread_count_distinct_and_after_upload(pool, cfg, good_recording):
    """校对遍数 = 入池（上传时间）之后校对过的不同的人数：同一个人两次只算一次；复录前的校对不算。"""
    _put_raw(pool, good_recording, "G1-S1-Q.wav")
    ingest_pool(pool, cfg)
    paths = pool_paths(pool)
    manifest = read_csv_rows(paths["manifest"])
    manifest[0]["上传时间"] = "2026-10-20 10:00:00"
    from pipeline.pool import MANIFEST_COLUMNS

    write_csv_rows(paths["manifest"], MANIFEST_COLUMNS, manifest)
    write_csv_rows(paths["proofread_log"], PROOFREAD_COLUMNS, [
        {"文件编号": "G1-S1-Q", "校对人": "1111", "校对时间": "2026-10-19 09:00:00", "第几遍": 1},  # 复录前
        {"文件编号": "G1-S1-Q", "校对人": "2222", "校对时间": "2026-10-20 10:00:00", "第几遍": 1},
        {"文件编号": "G1-S1-Q", "校对人": "2222", "校对时间": "2026-10-21 08:00:00", "第几遍": 1},
        {"文件编号": "G1-S1-N", "校对人": "3333", "校对时间": "2026-10-21 08:00:00", "第几遍": 1},  # 别的录音
    ])
    row = _row(acceptance(pool), "G1-S1-Q")
    assert row["校对遍数"] == 1
    assert row["通过"] is False

    # 时间认不出来的记录不算（没法确认是在入池之后校对的）
    rows = read_csv_rows(paths["proofread_log"])
    rows.append({"文件编号": "G1-S1-Q", "校对人": "5555", "校对时间": "昨天", "第几遍": 2})
    write_csv_rows(paths["proofread_log"], PROOFREAD_COLUMNS, rows)
    assert _row(acceptance(pool), "G1-S1-Q")["校对遍数"] == 1

    # 老师用 Excel 合并过校对记录：时间被改成了 Excel 的写法，也要认得
    rows = read_csv_rows(paths["proofread_log"])
    rows.append({"文件编号": "G1-S1-Q", "校对人": "4444", "校对时间": "2026/10/21 9:05", "第几遍": 2})
    write_csv_rows(paths["proofread_log"], PROOFREAD_COLUMNS, rows)
    row = _row(acceptance(pool), "G1-S1-Q")
    assert row["校对遍数"] == 2
    assert row["通过"] is True


def test_acceptance_clips_and_union_seconds(pool):
    paths = pool_paths(pool)
    write_turns_csv(paths["annotations_speakers"] / "G3-S1-Q.csv",
                    [(0.0, 10.0, "导游"), (5.0, 15.0, "游客甲")])  # 重叠的 5 秒只算一次
    write_clips_csv(paths["annotations_clips"] / "G3-S1-Q.csv", [(1.0, 9.0, "费用")])
    row = _row(acceptance(pool), "G3-S1-Q")
    assert row["说话人标注（秒）"] == 15.0
    assert row["片段标注"] is True
    assert _row(acceptance(pool), "G3-S1-N")["片段标注"] is False


def test_acceptance_broken_annotation_names_file(pool):
    folder = pool_paths(pool)["annotations_speakers"]
    folder.mkdir(parents=True)
    (folder / "G3-S1-Q.csv").write_text("start,end,speaker\n0,abc,导游\n", encoding="utf-8-sig")
    with pytest.raises(ValueError) as err:
        acceptance(pool)
    assert "G3-S1-Q.csv" in str(err.value)


def test_acceptance_missing_normalized_wav(pool, cfg, good_recording):
    """清单里有、但转换后的录音被删了：不能算通过，质检一栏说明原因。"""
    _put_raw(pool, good_recording, "G1-S1-Q.wav")
    ingest_pool(pool, cfg)
    (pool_paths(pool)["normalized"] / "G1-S1-Q.wav").unlink()
    row = _row(acceptance(pool), "G1-S1-Q")
    assert "normalized/G1-S1-Q.wav" in row["质检"]
    assert row["通过"] is False


def test_acceptance_summary_per_group():
    rows = []
    for group in (1, 2):
        for i in range(9):
            rows.append({"文件": f"G{group}-x{i}", "组": group, "已交": i < 3, "通过": i < 1,
                         "说话人标注（秒）": 100.0 if i < 3 else 0.0})
    summary = acceptance_summary(rows)
    assert [list(item) for item in summary] == [SUMMARY_COLUMNS] * 2
    assert SUMMARY_COLUMNS == ["组", "已交", "应交", "通过数", "说话人标注总分钟", "是否达到 10 分钟"]
    assert summary[0] == {"组": 1, "已交": 3, "应交": 9, "通过数": 1, "说话人标注总分钟": 5.0,
                          "是否达到 10 分钟": False}


def test_write_acceptance(pool, cfg, good_recording):
    _put_raw(pool, good_recording, "G1-S1-Q.wav")
    ingest_pool(pool, cfg)
    log_proofread(pool, "G1-S1-Q", "1234")
    log_proofread(pool, "G1-S1-Q", "5678")

    acc_path, sum_path = write_acceptance(pool)
    paths = pool_paths(pool)
    assert (acc_path, sum_path) == (paths["acceptance"], paths["acceptance_summary"])
    rows = _read_csv(acc_path)
    assert rows[0] == ACCEPTANCE_COLUMNS
    assert len(rows) == 73
    assert rows[1][:4] == ["G1-S1-Q", "1", "安静", "是"]  # 是/否：老师用 Excel 看得懂
    assert rows[1][-1] == "是"
    assert rows[2][3] == "否"
    summary = _read_csv(sum_path)
    assert summary[0] == SUMMARY_COLUMNS
    assert len(summary) == 9
    assert summary[1] == ["1", "1", "9", "1", "0.0", "否"]


# ---------- 补充的测试：冻结版本 ----------


def test_freeze_contents_and_checksums(pool, cfg, make_audio):
    _put_raw(pool, make_audio("tone", "wav", seconds=2, name="a.wav"), "G1-S1-Q.wav")
    ingest_pool(pool, cfg)
    export_references(pool)
    log_proofread(pool, "G1-S1-Q", "1234")
    paths = pool_paths(pool)
    write_turns_csv(paths["annotations_speakers"] / "G1-S1-Q.csv", [(0.0, 1.0, "导游")])
    write_clips_csv(paths["annotations_clips"] / "G1-S1-Q.csv", [(0.0, 1.0, "费用")])
    (paths["references"] / "Thumbs.db").write_bytes(b"x")  # 系统自动生成的文件不冻结
    (paths["normalized"] / "G1-S1-N.tmp.wav").write_bytes(b"half")  # 转了一半的临时文件不算

    target = freeze(pool, "v1")
    for name in ["manifest.csv", "qc_report.csv", "proofread_log.csv", "acceptance.csv", "acceptance_summary.csv",
                 "references/G1-S1-Q.txt", "annotations/speakers/G1-S1-Q.csv", "annotations/clips/G1-S1-Q.csv"]:
        assert (target / name).is_file(), name
    assert len(list((target / "references").glob("*.txt"))) == 72
    assert not (target / "references" / "Thumbs.db").exists()
    assert not (target / "normalized").exists()  # 录音不复制，只记指纹
    assert paths["acceptance"].is_file()  # 冻结前先按现在的数据池重新生成了验收表

    data = json.loads((target / "checksums.json").read_text(encoding="utf-8"))
    assert data["version"] == "v1"
    files = data["files"]
    wav = paths["normalized"] / "G1-S1-Q.wav"
    assert files["normalized/G1-S1-Q.wav"] == sha256_file(wav)
    assert "normalized/G1-S1-N.tmp.wav" not in files
    assert files["references/G1-S1-Q.txt"] == sha256_file(target / "references" / "G1-S1-Q.txt")
    assert files["manifest.csv"] == sha256_file(target / "manifest.csv")
    # 除了 checksums.json 自己，版本文件夹里的每个文件都有指纹
    copied = {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()} - {"checksums.json"}
    assert copied <= set(files)
    assert [p.name for p in paths["versions"].iterdir()] == ["v1"]  # 没有留下临时文件夹


@pytest.mark.parametrize("bad", ["", "../v1", "v1/x", "版本1", ".v1", "v 1"])
def test_freeze_rejects_bad_version_name(pool, bad):
    with pytest.raises(ValueError, match="v1"):
        freeze(pool, bad)


def test_freeze_missing_wav_leaves_nothing(pool, cfg, make_audio):
    """清单里的录音不见了：拒绝冻结，不留下半成品；补好后可以用同一个版本名再冻结。"""
    _put_raw(pool, make_audio("tone", "wav", seconds=2, name="a.wav"), "G1-S1-Q.wav")
    ingest_pool(pool, cfg)
    wav = pool_paths(pool)["normalized"] / "G1-S1-Q.wav"
    saved = wav.read_bytes()
    wav.unlink()
    with pytest.raises(FileNotFoundError) as err:
        freeze(pool, "v1")
    assert "normalized/G1-S1-Q.wav" in str(err.value)
    assert list(pool_paths(pool)["versions"].iterdir()) == []
    assert current_version(pool) is None

    wav.write_bytes(saved)
    assert freeze(pool, "v1").is_dir()


def test_current_version_natural_order(pool):
    assert current_version(pool) is None
    versions = pool_paths(pool)["versions"]
    for name in ["v1", "v2", "v9", "v10"]:
        (versions / name).mkdir(parents=True)
        (versions / name / "checksums.json").write_text("{}", encoding="utf-8")
    (versions / "v11").mkdir()  # 没有 checksums.json：不是冻结好的版本
    (versions / ".v12.tmp").mkdir()
    (versions / "notes.txt").write_text("x", encoding="utf-8")
    assert current_version(pool) == "v10"


# ---------- 补充的测试：命令行工具 ----------


def _run_tool(script, *args):
    """像老师在命令行里那样运行 python tools/xxx.py（在别的文件夹里运行，检验工具自己能找到 pipeline）。"""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, str(ROOT / "tools" / script), *map(str, args)], capture_output=True,
                          text=True, encoding="utf-8", env=env, cwd=str(ROOT.parent))


def _load_tool(script):
    spec = importlib.util.spec_from_file_location(script.removesuffix(".py"), ROOT / "tools" / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_check_pool(pool, cfg, good_recording):
    _put_raw(pool, good_recording, "G1-S1-Q.wav")
    ingest_pool(pool, cfg)
    log_proofread(pool, "G1-S1-Q", "1234")

    proc = _run_tool("check_pool.py", "--pool", pool)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "第 1 组" in out and "第 8 组" in out
    assert "已交 1/9" in out
    assert "G1-S1-Q" in out and "校对" in out  # 已交但没通过的录音，列出原因
    assert pool_paths(pool)["acceptance"].is_file()
    assert pool_paths(pool)["acceptance_summary"].is_file()


def test_cli_check_pool_missing_folder(tmp_path):
    proc = _run_tool("check_pool.py", "--pool", tmp_path / "nothing")
    assert proc.returncode == 1
    assert "不存在" in proc.stdout
    assert not (tmp_path / "nothing").exists()


def test_cli_check_pool_table_locked(pool, monkeypatch, capsys):
    import pipeline.pool

    init_pool(pool)
    real_write = pipeline.pool.write_csv_rows

    def write_but_locked(path, columns, rows):
        if os.path.basename(path) == "acceptance.csv":
            raise PermissionError(13, "Permission denied", str(path))
        real_write(path, columns, rows)

    monkeypatch.setattr(pipeline.pool, "write_csv_rows", write_but_locked)
    assert _load_tool("check_pool.py").main(["--pool", str(pool)]) == 1
    out = capsys.readouterr().out
    assert "acceptance.csv" in out and "关闭" in out


def test_cli_freeze_pool(pool, cfg, make_audio):
    _put_raw(pool, make_audio("tone", "wav", seconds=2, name="a.wav"), "G1-S1-Q.wav")
    ingest_pool(pool, cfg)

    proc = _run_tool("freeze_pool.py", "--pool", pool, "--version", "v1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "v1" in proc.stdout
    assert (pool_paths(pool)["versions"] / "v1" / "checksums.json").is_file()

    proc = _run_tool("freeze_pool.py", "--pool", pool, "--version", "v1")
    assert proc.returncode == 1
    assert "已经存在" in proc.stdout

    proc = _run_tool("freeze_pool.py", "--pool", pool)  # 必须写版本名
    assert proc.returncode != 0


def test_cli_freeze_uses_config_pool_by_default(tmp_path, monkeypatch, capsys):
    import pipeline.config

    cfg = load_config()
    cfg["paths"]["data_pool"] = str(tmp_path / "cfg_pool")
    init_pool(tmp_path / "cfg_pool")
    monkeypatch.setattr(pipeline.config, "load_config", lambda *a, **k: cfg)
    assert _load_tool("freeze_pool.py").main(["--version", "v1"]) == 0
    assert (tmp_path / "cfg_pool" / "versions" / "v1").is_dir()
    assert _load_tool("check_pool.py").main([]) == 0
    assert str(tmp_path / "cfg_pool") in capsys.readouterr().out


def test_ten_minutes_not_reached_by_rounding():
    """598 秒（9.97 分钟）显示成 10.0 分钟，但不能算达到 10 分钟。"""
    from pipeline.pool import acceptance_summary

    rows = [{"组": 1, "已交": True, "通过": False, "说话人标注（秒）": 598.0}]
    summary = [r for r in acceptance_summary(rows) if r["组"] == 1][0]
    assert summary["是否达到 10 分钟"] is False


def test_pool_version_flags_changes_after_freeze(pool, cfg, make_audio):
    """冻结以后，测评要读的文件（清单、参考文本、标注、录音）又改了：报告里的版本名要注明"和 v1 不一致"。

    校对记录、验收表这些记录表本来就会变，不算改动。
    """
    from pipeline.evaluation import pool_version

    _put_raw(pool, make_audio("tone", "wav", seconds=2, name="a.wav"), "G1-S1-Q.wav")
    ingest_pool(pool, cfg)
    export_references(pool)
    freeze(pool, "v1")
    assert version_changes(pool, "v1") == []
    assert pool_version(pool) == "v1"

    log_proofread(pool, "G1-S1-Q", "1234")  # 校对记录变了：不算
    write_acceptance(pool)
    assert pool_version(pool) == "v1"

    paths = pool_paths(pool)
    (paths["references"] / "G1-S1-Q.txt").write_text("冻结以后又改过的参考文本", encoding="utf-8")
    write_turns_csv(paths["annotations_speakers"] / "G1-S1-Q.csv", [(0.0, 1.0, "导游")])  # 冻结以后新加的标注
    assert version_changes(pool, "v1") == ["annotations/speakers/G1-S1-Q.csv", "references/G1-S1-Q.txt"]
    label = pool_version(pool)
    assert label.startswith("v1（") and "2 个文件" in label and "不一致" in label
