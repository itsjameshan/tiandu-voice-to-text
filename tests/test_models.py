"""模型清单（pipeline/models.py）和下载脚本（models/download_models.py）的测试。

这些测试不联网：下载函数都被替换成假的，只检查“该不该下载、下载后怎么处理”。
只有 test_models_present 会去看仓库里真实的 models/ 文件夹（没下载模型时自动跳过）。
"""
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest
from conftest import ROOT, requires_models

from pipeline.config import load_config
from pipeline.models import MODEL_SPECS, missing_models, model_path

REQUIRED_NAMES = ["sense_voice", "silero_vad", "pyannote", "campplus", "test_wav"]
OPTIONAL_NAMES = ["eres2net", "itn_fst", "paraformer_small", "punct", "conformer_hotword"]


def load_download_script():
    """models/ 不是 Python 包，按文件路径导入下载脚本，拿到里面的 main、download_file。"""
    spec = importlib.util.spec_from_file_location("download_models", ROOT / "models" / "download_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def put_check_files(models_dir: Path, names) -> None:
    """按清单里的 check 在 models_dir 下放好空文件，假装这些模型已经下载好了。"""
    for spec in MODEL_SPECS:
        if spec["name"] in names:
            for rel in spec["check"]:
                p = models_dir / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()


def spec_of(name: str) -> dict:
    return next(s for s in MODEL_SPECS if s["name"] == name)


# ---------------- 计划里规定的 4 个测试 ----------------


def test_specs_complete():
    required = [s["name"] for s in MODEL_SPECS if s["required"]]
    assert len(required) == 5
    assert sorted(required) == sorted(REQUIRED_NAMES)
    for s in MODEL_SPECS:
        assert s["url"].startswith("https://github.com/k2-fsa/sherpa-onnx/releases/download/")
    sense_voice = spec_of("sense_voice")
    assert "2024-07-17" in sense_voice["url"]
    assert "2025-09-09" not in sense_voice["url"]


def test_missing_models_empty_dir(tmp_path):
    cfg = {"paths": {"models": str(tmp_path)}}
    missing = missing_models(cfg)
    assert len(missing) == 5
    assert sorted(missing) == sorted(REQUIRED_NAMES)


def test_download_skips_existing(tmp_path, monkeypatch):
    dm = load_download_script()
    put_check_files(tmp_path, REQUIRED_NAMES)
    calls = []
    monkeypatch.setattr(dm, "download_file", lambda url, dest: calls.append(url))
    assert dm.main(["--models-dir", str(tmp_path)]) == 0
    assert calls == []


@requires_models
def test_models_present():
    assert missing_models(load_config()) == []


# ---------------- 补充测试 ----------------


def test_spec_fields_and_names():
    """每项都有计划规定的 6 个键；名字不重复；URL 的文件名就是 archive；可选项是计划里那 5 个。"""
    names = [s["name"] for s in MODEL_SPECS]
    assert len(names) == len(set(names))
    assert sorted(n for n in names if n not in REQUIRED_NAMES) == sorted(OPTIONAL_NAMES)
    for s in MODEL_SPECS:
        assert {"name", "url", "archive", "check", "size_mb", "required"} <= set(s)
        assert s["url"].endswith("/" + s["archive"])
        assert isinstance(s["required"], bool)
        assert s["size_mb"] > 0
        assert s["check"] and all(isinstance(p, str) and not p.startswith("/") for p in s["check"])
        # 文件名只用 ASCII（Windows 解压时中文名可能乱码）
        assert s["archive"].isascii() and all(p.isascii() for p in s["check"])
    test_wav = spec_of("test_wav")
    assert test_wav["archive"] == "0-four-speakers-zh.wav"
    assert "/speaker-segmentation-models/" in test_wav["url"]


def test_model_path_keys(tmp_path):
    cfg = {"paths": {"models": str(tmp_path)}}
    sv = tmp_path / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
    expected = {
        "sense_voice_model": sv / "model.int8.onnx",
        "sense_voice_tokens": sv / "tokens.txt",
        "sense_voice_test_zh": sv / "test_wavs" / "zh.wav",
        "silero_vad": tmp_path / "silero_vad.onnx",
        "pyannote": tmp_path / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx",
        "campplus": tmp_path / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
        "test_wav": tmp_path / "0-four-speakers-zh.wav",
    }
    for key, path in expected.items():
        assert model_path(cfg, key) == path
        assert isinstance(model_path(cfg, key), Path)


def test_model_path_unknown_key_lists_keys(tmp_path):
    with pytest.raises(KeyError) as info:
        model_path({"paths": {"models": str(tmp_path)}}, "whisper")
    assert "sense_voice_model" in str(info.value)


def test_model_path_files_are_checked(tmp_path):
    """model_path 能给出的每个文件都在某个必需模型的 check 里，这样 missing_models 能发现它缺了。"""
    cfg = {"paths": {"models": str(tmp_path)}}
    checked = {tmp_path / rel for s in MODEL_SPECS if s["required"] for rel in s["check"]}
    for key in ["sense_voice_model", "sense_voice_tokens", "sense_voice_test_zh", "silero_vad",
                "pyannote", "campplus", "test_wav"]:
        assert model_path(cfg, key) in checked


def test_missing_models_partial(tmp_path):
    put_check_files(tmp_path, ["sense_voice", "silero_vad", "campplus", "test_wav"])
    assert missing_models({"paths": {"models": str(tmp_path)}}) == ["pyannote"]


def test_download_fetches_and_extracts(tmp_path, monkeypatch, capsys):
    """空文件夹：每个必需模型下载一次；.tar.bz2 解压后删掉压缩包；最后 missing_models 为空。"""
    dm = load_download_script()
    calls = []

    def fake_download(url, dest):
        calls.append(url)
        spec = next(s for s in MODEL_SPECS if s["url"] == url)
        dest = Path(dest)
        if spec["archive"].endswith(".tar.bz2"):
            # 做一个只含 check 文件的小压缩包，代替真的模型压缩包
            with tarfile.open(dest, "w:bz2") as tar:
                for rel in spec["check"]:
                    data = b"fake"
                    info = tarfile.TarInfo(rel)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
        else:
            dest.write_bytes(b"fake")

    monkeypatch.setattr(dm, "download_file", fake_download)
    assert dm.main(["--models-dir", str(tmp_path)]) == 0
    assert sorted(calls) == sorted(s["url"] for s in MODEL_SPECS if s["required"])
    assert missing_models({"paths": {"models": str(tmp_path)}}) == []
    assert list(tmp_path.glob("*.tar.bz2")) == []
    out = capsys.readouterr().out
    assert "OK" in out


def test_copied_archive_is_extracted(tmp_path, monkeypatch):
    """老师在别处下载好压缩包拷进 models/：不再下载，直接解压，解压后删除压缩包。"""
    dm = load_download_script()
    put_check_files(tmp_path, ["sense_voice", "silero_vad", "campplus", "test_wav"])
    spec = spec_of("pyannote")
    with tarfile.open(tmp_path / spec["archive"], "w:bz2") as tar:
        for rel in spec["check"]:
            info = tarfile.TarInfo(rel)
            info.size = 4
            tar.addfile(info, io.BytesIO(b"fake"))
    calls = []
    monkeypatch.setattr(dm, "download_file", lambda url, dest: calls.append(url))
    assert dm.main(["--models-dir", str(tmp_path)]) == 0
    assert calls == []
    assert (tmp_path / spec["check"][0]).read_bytes() == b"fake"
    assert not (tmp_path / spec["archive"]).exists()


def test_download_failure_exit_code(tmp_path, monkeypatch, capsys):
    """下载失败：退出码 1，并给出中文提示。"""
    dm = load_download_script()
    put_check_files(tmp_path, ["sense_voice", "silero_vad", "campplus", "test_wav"])

    def broken_download(url, dest):
        raise RuntimeError("网络断了")

    monkeypatch.setattr(dm, "download_file", broken_download)
    assert dm.main(["--models-dir", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "国内访问 GitHub 慢，可以先在别处下载好再拷进 models/" in out
    assert "缺失" in out and "pyannote" in out


def test_optional_download(tmp_path, monkeypatch):
    """--optional 只多下载点名的可选模型；可选模型下载失败不影响退出码。"""
    dm = load_download_script()
    put_check_files(tmp_path, REQUIRED_NAMES)
    calls = []

    def fake_download(url, dest):
        calls.append(url)
        Path(dest).write_bytes(b"fake")

    monkeypatch.setattr(dm, "download_file", fake_download)
    assert dm.main(["--models-dir", str(tmp_path), "--optional", "itn_fst"]) == 0
    assert calls == [spec_of("itn_fst")["url"]]
    assert (tmp_path / "itn_zh_number.fst").is_file()

    def broken_download(url, dest):
        raise RuntimeError("网络断了")

    monkeypatch.setattr(dm, "download_file", broken_download)
    assert dm.main(["--models-dir", str(tmp_path), "--optional", "eres2net"]) == 0


def test_list_does_not_download(tmp_path, monkeypatch, capsys):
    dm = load_download_script()
    calls = []
    monkeypatch.setattr(dm, "download_file", lambda url, dest: calls.append(url))
    assert dm.main(["--list", "--models-dir", str(tmp_path)]) == 0
    assert calls == []
    out = capsys.readouterr().out
    for s in MODEL_SPECS:
        assert s["name"] in out


def test_unknown_optional_name(tmp_path, monkeypatch, capsys):
    dm = load_download_script()
    calls = []
    monkeypatch.setattr(dm, "download_file", lambda url, dest: calls.append(url))
    assert dm.main(["--models-dir", str(tmp_path), "--optional", "whisper"]) == 2
    assert calls == []
    out = capsys.readouterr().out
    assert "whisper" in out and "punct" in out


class FakeResponse(io.BytesIO):
    """假装是 urllib 返回的响应：能 read()、有 headers、能用 with。"""

    def __init__(self, data: bytes):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}


def test_download_file_retries(tmp_path, monkeypatch):
    """前两次连接失败，第三次成功：文件内容正确，临时的 .part 文件不留下。"""
    dm = load_download_script()
    data = b"x" * 3000
    attempts = []

    def fake_urlopen(url, timeout=None):
        attempts.append(url)
        if len(attempts) < 3:
            raise OSError("连接被重置")
        return FakeResponse(data)

    monkeypatch.setattr(dm.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(dm.time, "sleep", lambda s: None)
    dest = tmp_path / "silero_vad.onnx"
    dm.download_file("https://example.invalid/silero_vad.onnx", dest)
    assert len(attempts) == 3
    assert dest.read_bytes() == data
    assert list(tmp_path.glob("*.part")) == []


def test_download_file_gives_up(tmp_path, monkeypatch):
    """一直失败：重试 3 次后报错，不留下半截文件（否则下次会被当成已下载）。"""
    dm = load_download_script()
    attempts = []

    def fake_urlopen(url, timeout=None):
        attempts.append(url)
        raise OSError("连接超时")

    monkeypatch.setattr(dm.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(dm.time, "sleep", lambda s: None)
    dest = tmp_path / "silero_vad.onnx"
    with pytest.raises(RuntimeError):
        dm.download_file("https://example.invalid/silero_vad.onnx", dest)
    assert len(attempts) == 3
    assert not dest.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_download_file_rejects_truncated(tmp_path, monkeypatch):
    """服务器说有 3000 字节、实际只给了 1000：算失败，不留下文件。"""
    dm = load_download_script()

    def fake_urlopen(url, timeout=None):
        resp = FakeResponse(b"x" * 1000)
        resp.headers = {"Content-Length": "3000"}
        return resp

    monkeypatch.setattr(dm.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(dm.time, "sleep", lambda s: None)
    dest = tmp_path / "silero_vad.onnx"
    with pytest.raises(RuntimeError):
        dm.download_file("https://example.invalid/silero_vad.onnx", dest)
    assert not dest.exists()
