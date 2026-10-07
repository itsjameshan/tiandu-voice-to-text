"""标注表格（pipeline/annotations.py）和命令行工具 tools/labels_to_csv.py 的测试。

标签文件都是在测试里现场写的文本（模仿 Audacity"导出标签"的格式），不涉及任何录音。
"""
import csv
import importlib.util
import os
import subprocess
import sys

import pytest

from conftest import ROOT
from pipeline.annotations import (
    CLIP_COLUMNS,
    TURN_COLUMNS,
    annotated_seconds,
    clip_label,
    read_audacity_labels,
    read_clips_csv,
    read_turns_csv,
    write_clips_csv,
    write_turns_csv,
)
from pipeline.pool import init_pool, pool_paths

# Audacity 导出的标签：每行"开始<Tab>结束<Tab>标签"；在语谱图上框选过的标签，下面多一行以 \ 开头的频率范围
AUDACITY_TEXT = (
    "0.500000\t3.250000\t导游\n"
    "\\\t200.000000\t3000.000000\n"
    "3.400000\t6.000000\t  游客甲 \n"
)


def _write(path, text, encoding="utf-8"):
    path.write_bytes(text.encode(encoding))
    return path


def _read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


# ---------- 计划里列出的测试 ----------


def test_audacity_parse(tmp_path):
    labels = read_audacity_labels(_write(tmp_path / "labels.txt", AUDACITY_TEXT))
    assert len(labels) == 2
    assert labels == [(0.5, 3.25, "导游"), (3.4, 6.0, "游客甲")]
    for start, end, _ in labels:
        assert isinstance(start, float) and isinstance(end, float)


# ---------- 补充的测试：读 Audacity 标签 ----------


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "gbk", "utf-16"])
def test_audacity_parse_encodings(tmp_path, encoding):
    """UTF-8（带不带 BOM 都行）、中文 Windows 上的 GBK（ANSI）、记事本另存的 Unicode（UTF-16）都能读。"""
    text = AUDACITY_TEXT.replace("\n", "\r\n")  # Windows 换行
    labels = read_audacity_labels(_write(tmp_path / "labels.txt", text, encoding))
    assert labels == [(0.5, 3.25, "导游"), (3.4, 6.0, "游客甲")]


def test_audacity_parse_skips_blank_lines_and_keeps_empty_label(tmp_path):
    text = "\n1.000000\t2.000000\t\n\n2.5\t3\t费用\n"
    assert read_audacity_labels(_write(tmp_path / "a.txt", text)) == [(1.0, 2.0, ""), (2.5, 3.0, "费用")]


def test_audacity_parse_bad_line_says_where(tmp_path):
    path = _write(tmp_path / "a.txt", "1.0\t2.0\t导游\n两秒\t3.0\t游客甲\n")
    with pytest.raises(ValueError) as err:
        read_audacity_labels(path)
    message = str(err.value)
    assert "第 2 行" in message
    assert "两秒" in message


def test_audacity_parse_rejects_nan(tmp_path):
    with pytest.raises(ValueError, match="第 1 行"):
        read_audacity_labels(_write(tmp_path / "a.txt", "nan\t2.0\t导游\n"))


def test_audacity_parse_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_audacity_labels(tmp_path / "nothing.txt")


# ---------- 补充的测试：说话人表格 ----------


def test_turns_csv_roundtrip(tmp_path):
    path = tmp_path / "speakers" / "G3-S1-Q.csv"
    write_turns_csv(path, [(3.4, 6.0, "游客甲"), (0.5, 3.25, "导游")])
    rows = _read_csv(path)
    assert rows[0] == ["start", "end", "speaker"] == TURN_COLUMNS
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")  # UTF-8 带 BOM，Excel 打开不乱码
    assert read_turns_csv(path) == [(0.5, 3.25, "导游"), (3.4, 6.0, "游客甲")]  # 按开始时间排好


@pytest.mark.parametrize("rows, hint", [
    ([(2.0, 2.0, "导游")], "起止时间相同"),
    ([(3.0, 2.0, "导游")], "结束时间早于开始时间"),
    ([(-1.0, 2.0, "导游")], "负数"),
    ([(1.0, 2.0, "  ")], "没有写"),
])
def test_write_turns_rejects_bad_rows(tmp_path, rows, hint):
    path = tmp_path / "t.csv"
    with pytest.raises(ValueError) as err:
        write_turns_csv(path, [(0.0, 1.0, "导游"), *rows])
    assert hint in str(err.value)
    assert "第 2 个标签" in str(err.value)
    assert not path.exists()  # 有错就不写


def test_read_turns_csv_after_excel_resave(tmp_path):
    """老师用 Excel 打开后另存为普通 CSV（中文 Windows 存成 GBK）、时间被改成短写法：照样能读。"""
    path = tmp_path / "t.csv"
    _write(path, "start,end,speaker\r\n0.5,3.25,导游\r\n,,\r\n3.4,6,游客甲\r\n", "gbk")
    assert read_turns_csv(path) == [(0.5, 3.25, "导游"), (3.4, 6.0, "游客甲")]


def test_read_turns_csv_errors_name_file(tmp_path):
    missing_col = _write(tmp_path / "a.csv", "start,end,label\n0,1,费用\n", "utf-8-sig")
    with pytest.raises(ValueError, match="speaker"):
        read_turns_csv(missing_col)
    bad_number = _write(tmp_path / "b.csv", "start,end,speaker\n0,1,导游\nabc,2,导游\n", "utf-8-sig")
    with pytest.raises(ValueError) as err:
        read_turns_csv(bad_number)
    assert "b.csv" in str(err.value)
    assert "第 3 行" in str(err.value)


def test_annotated_seconds_counts_overlap_once():
    assert annotated_seconds([]) == 0.0
    assert annotated_seconds([(0.0, 600.0, "导游")]) == 600.0
    # 两人同时说话的 5 秒只算一次；没标的停顿不算
    rows = [(0.0, 10.0, "导游"), (5.0, 15.0, "游客甲"), (20.0, 25.0, "导游"), (21.0, 22.0, "游客乙")]
    assert annotated_seconds(rows) == pytest.approx(20.0)


# ---------- 补充的测试：疑似片段表格 ----------


def test_clip_label_names():
    for name in ["购物安排", "费用", "行程变更", "服务态度", "威胁消费"]:
        assert clip_label(name) == name
    assert clip_label(" 消费施压 ") == "威胁消费"  # 界面上的显示名也认，存成数据里的类别名


@pytest.mark.parametrize("name", ["正常讲解", "其他", "投诉", "疑似·费用", "导游", ""])
def test_clip_label_rejects_others(name):
    with pytest.raises(ValueError) as err:
        clip_label(name)
    message = str(err.value)
    assert "购物安排" in message and "威胁消费" in message  # 说清楚只能写哪几个


def test_clips_csv_roundtrip(tmp_path):
    path = tmp_path / "clips" / "G8-S1-Q.csv"
    write_clips_csv(path, [(30.0, 52.5, "消费施压"), (10.0, 20.0, "费用")])
    rows = _read_csv(path)
    assert rows[0] == ["start", "end", "label"] == CLIP_COLUMNS
    assert rows[1] == ["10.000", "20.000", "费用"]
    assert rows[2][2] == "威胁消费"
    assert read_clips_csv(path) == [(10.0, 20.0, "费用"), (30.0, 52.5, "威胁消费")]


def test_write_clips_rejects_bad_label(tmp_path):
    path = tmp_path / "c.csv"
    with pytest.raises(ValueError) as err:
        write_clips_csv(path, [(1.0, 2.0, "费用"), (3.0, 4.0, "导游")])
    assert "第 2 个标签" in str(err.value)
    assert "导游" in str(err.value)
    assert not path.exists()


# ---------- 补充的测试：命令行工具 labels_to_csv.py ----------


def _run_tool(*args):
    """像同学在命令行里那样运行（在别的文件夹里运行，检验工具自己能找到 pipeline）。"""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, str(ROOT / "tools" / "labels_to_csv.py"), *map(str, args)],
                          capture_output=True, text=True, encoding="utf-8", env=env, cwd=str(ROOT.parent))


def _load_tool():
    spec = importlib.util.spec_from_file_location("labels_to_csv", ROOT / "tools" / "labels_to_csv.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_two_file_form(tmp_path):
    labels = _write(tmp_path / "G3-S1-Q_speakers.txt", AUDACITY_TEXT, "gbk")
    out = tmp_path / "out" / "G3-S1-Q.csv"
    proc = _run_tool(labels, out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert read_turns_csv(out) == [(0.5, 3.25, "导游"), (3.4, 6.0, "游客甲")]
    assert "2 个" in proc.stdout


def test_cli_two_file_form_clips(tmp_path):
    labels = _write(tmp_path / "c.txt", "1\t5\t费用\n6\t9\t消费施压\n")
    out = tmp_path / "c.csv"
    proc = _run_tool(labels, out, "--kind", "clips")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert read_clips_csv(out) == [(1.0, 5.0, "费用"), (6.0, 9.0, "威胁消费")]


def test_cli_pool_form_speakers_and_clips(tmp_path):
    root = tmp_path / "pool"
    init_pool(root)
    labels = _write(tmp_path / "s.txt", AUDACITY_TEXT)
    proc = _run_tool(labels, "--pool", root, "--file", "G3-S1-Q", "--kind", "speakers")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    target = pool_paths(root)["annotations_speakers"] / "G3-S1-Q.csv"
    assert read_turns_csv(target) == [(0.5, 3.25, "导游"), (3.4, 6.0, "游客甲")]
    assert str(target) in proc.stdout

    clips = _write(tmp_path / "c.txt", "10\t20\t费用\n")
    proc = _run_tool(clips, "--pool", root, "--file", "G8-S1-Q", "--kind", "clips")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert read_clips_csv(pool_paths(root)["annotations_clips"] / "G8-S1-Q.csv") == [(10.0, 20.0, "费用")]


def test_cli_warns_unknown_role(tmp_path):
    """角色名不在剧本角色表里（如"导游"写成"导"）：提醒核对，但仍然转换。"""
    root = tmp_path / "pool"
    init_pool(root)
    labels = _write(tmp_path / "s.txt", "0\t2\t导\n2\t4\t导游\n")
    proc = _run_tool(labels, "--pool", root, "--file", "G3-S1-Q")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "导" in proc.stdout and "角色" in proc.stdout
    assert (pool_paths(root)["annotations_speakers"] / "G3-S1-Q.csv").is_file()


def test_cli_bad_clip_label_fails_in_chinese(tmp_path):
    labels = _write(tmp_path / "c.txt", "1\t5\t投诉\n")
    out = tmp_path / "c.csv"
    proc = _run_tool(labels, out, "--kind", "clips")
    assert proc.returncode == 1
    assert "投诉" in proc.stdout and "威胁消费" in proc.stdout
    assert not out.exists()


@pytest.mark.parametrize("args, hint", [
    (["--pool", "POOL", "--file", "G9-S1-Q"], "G9-S1-Q"),  # 没有这段录音
    (["--pool", "POOL"], "--file"),  # 没写输出文件，也没写 --file
    (["OUT.csv", "--file", "G3-S1-Q"], "选一种"),  # 两种写法混在一起
])
def test_cli_bad_arguments(tmp_path, capsys, args, hint):
    root = tmp_path / "pool"
    init_pool(root)
    labels = _write(tmp_path / "s.txt", AUDACITY_TEXT)
    args = [str(root) if a == "POOL" else str(tmp_path / a) if a == "OUT.csv" else a for a in args]
    assert _load_tool().main([str(labels), *args]) == 1
    assert hint in capsys.readouterr().out


def test_cli_missing_input(tmp_path, capsys):
    assert _load_tool().main([str(tmp_path / "nothing.txt"), str(tmp_path / "o.csv")]) == 1
    assert "找不到" in capsys.readouterr().out


def test_cli_empty_label_file(tmp_path, capsys):
    labels = _write(tmp_path / "s.txt", "\\\t100\t200\n")
    assert _load_tool().main([str(labels), str(tmp_path / "o.csv")]) == 1
    assert "没有标签" in capsys.readouterr().out
