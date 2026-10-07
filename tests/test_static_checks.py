"""静态检查：不运行代码，只读源文件，守住几条红线。

- 任何 .py 文件里都不能打开 Gradio 的公网分享（不开公网分享）；
- pipeline/ 里不能导入联网用的库（处理录音时数据不出本机）；
- 不能把密钥、令牌直接写在代码里；
- pipeline/ 和 app.py 里不能出现禁用的说法（只有 step8_report.py 定义 FORBIDDEN_WORDS 的那一行除外）；
- 界面文字不用"举报""曝光""证据"等字眼。
"""
import ast
import re
from pathlib import Path

from conftest import ROOT

from pipeline.step8_report import FORBIDDEN_WORDS

# 和 pyproject.toml 里 ruff 的 extend-exclude 一致：这些文件夹不是本项目写的代码
SKIP_DIRS = {".git", ".venv", "venv", "handoff", "data_pool", "outputs", "tmp", "dist", "__pycache__",
             "node_modules"}

# 公网分享的写法（这里故意拆开写，免得本文件自己被查出来）
PUBLIC_SHARE = re.compile(r"\bshare\s*=\s*" + "True")
SECRET = re.compile(r"""(api_key|secret|token)\s*=\s*["'][A-Za-z0-9]{16,}""", re.IGNORECASE)
NETWORK_MODULES = {"requests", "urllib.request", "http.client", "socket", "httpx", "aiohttp", "urllib3"}
UI_AVOID_WORDS = ["举报", "曝光", "证据"]


def _py_files(base: Path):
    for path in sorted(base.rglob("*.py")):
        parts = path.relative_to(ROOT).parts
        if any(part in SKIP_DIRS for part in parts):
            continue
        if parts[0] == "models" and len(parts) > 2:  # 下载来的模型文件夹（如 sherpa-onnx-*）
            continue
        yield path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_no_public_share():
    files = list(_py_files(ROOT))
    assert any(path.name == "app.py" for path in files)
    found = [f"{path.relative_to(ROOT)}:{n}" for path in files
             for n, line in enumerate(_read(path).splitlines(), start=1) if PUBLIC_SHARE.search(line)]
    assert not found, f"不能打开 Gradio 公网分享：{found}"


def _imported_modules(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)  # from urllib import request
    return names


def _is_network(module: str) -> bool:
    """例如 requests、requests.adapters、urllib.request、socket 都算联网用的库。"""
    return any(module == name or module.startswith(name + ".") for name in NETWORK_MODULES)


def test_pipeline_has_no_network_imports():
    found = []
    for path in _py_files(ROOT / "pipeline"):
        modules = _imported_modules(ast.parse(_read(path)))
        bad = {m for m in modules if _is_network(m)}
        if bad:
            found.append(f"{path.relative_to(ROOT)}: {sorted(bad)}")
    assert not found, f"pipeline/ 里不能联网：{found}"


def test_no_hardcoded_secrets():
    found = [f"{path.relative_to(ROOT)}:{n}" for path in _py_files(ROOT)
             for n, line in enumerate(_read(path).splitlines(), start=1) if SECRET.search(line)]
    assert not found, f"不要把密钥写在代码里（从环境变量读）：{found}"


def _pipeline_and_app() -> list[Path]:
    return [*_py_files(ROOT / "pipeline"), ROOT / "app.py"]


def test_no_forbidden_words_in_pipeline_and_app():
    found = []
    for path in _pipeline_and_app():
        for n, line in enumerate(_read(path).splitlines(), start=1):
            if path.name == "step8_report.py" and line.startswith("FORBIDDEN_WORDS ="):
                continue  # 定义禁用说法的那一行
            for word in FORBIDDEN_WORDS:
                if word in line:
                    found.append(f"{path.relative_to(ROOT)}:{n} {word}")
    assert not found, f"界面、初稿、导出文件里不能出现这些说法：{found}"


def test_ui_text_avoids_sensitive_words():
    found = [f"{path.name}: {word}" for path in (ROOT / "app.py", ROOT / "pipeline" / "ui_text.py")
             for word in UI_AVOID_WORDS if word in _read(path)]
    assert not found, f"界面不用这些字眼：{found}"
