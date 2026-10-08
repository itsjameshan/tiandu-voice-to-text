"""pipeline/textio.py：记事本、Excel 存出来的各种编码都能读。"""
import codecs

import pytest

from pipeline.textio import read_csv_dicts, read_text

TEXT = "今天去石林，下午回昆明。\n一共两千八百块。"


@pytest.mark.parametrize("encoding,bom", [("utf-8", b""), ("utf-8", codecs.BOM_UTF8), ("gbk", b""),
                                          ("utf-16-le", codecs.BOM_UTF16_LE), ("utf-16-be", codecs.BOM_UTF16_BE)])
def test_read_text_common_encodings(tmp_path, encoding, bom):
    """UTF-8（带不带 BOM）、ANSI（中文 Windows 上是 GBK）、记事本另存为"Unicode"（UTF-16）。"""
    path = tmp_path / "a.txt"
    path.write_bytes(bom + TEXT.encode(encoding))
    assert read_text(path) == TEXT


@pytest.mark.parametrize("encoding", ["utf-8-sig", "gbk", "utf-16"])
def test_read_csv_dicts(tmp_path, encoding):
    """Excel 另存为"CSV（逗号分隔）"是 GBK，"CSV UTF-8"带 BOM；格子里有换行也照样读。"""
    path = tmp_path / "a.csv"
    path.write_bytes('id,text,label\r\nX0001,"这个手镯\r\n今天优惠",购物安排\r\n'.encode(encoding))
    columns, rows = read_csv_dicts(path)
    assert columns == ["id", "text", "label"]
    assert rows == [{"id": "X0001", "text": "这个手镯\r\n今天优惠", "label": "购物安排"}]


def test_read_text_unknown_encoding(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\x81\x20\xff\xfe\x00")  # 哪种编码都对不上（开头也不是 UTF-16 的 BOM）
    with pytest.raises(ValueError, match="编码"):
        read_text(path, hint="请另存为 UTF-8")


def test_reference_saved_as_unicode_by_notepad(tmp_path):
    """记事本"另存为 → 编码选 Unicode"的参考文本（UTF-16），测评和"数据校对"页都能读。"""
    from pipeline.evaluation import read_reference

    path = tmp_path / "G1-S1-Q.txt"
    path.write_bytes(codecs.BOM_UTF16_LE + TEXT.encode("utf-16-le"))
    assert read_reference(path) == TEXT


def test_extra_sentences_saved_as_gbk_by_excel(tmp_path):
    """第 6、7 组用 Excel 审核补充句子后按"CSV（逗号分隔）"另存（GBK）：检查脚本和训练脚本都能读。"""
    import importlib.util

    from conftest import ROOT

    path = tmp_path / "classification_extra.csv"
    path.write_bytes("id,text,label,source,reviewed,reviewer,note\r\nX0001,这个手镯今天优惠，买不买随你,购物安排,AI生成,已审核,1234,\r\n"
                     .encode("gbk"))
    for name in ("train_classifier.py", "check_classification_data.py"):
        spec = importlib.util.spec_from_file_location(name[:-3], ROOT / "tools" / name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if name == "train_classifier.py":
            assert module.load_extra(path)[0]["label"] == "购物安排"
        else:
            columns, rows = module.read_rows(path)
            assert columns[0] == "id" and rows[0]["text"].startswith("这个手镯")
