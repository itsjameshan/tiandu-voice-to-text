"""读取参数配置 config.yaml。

基线做法：
    读项目文件夹里的 config.yaml，再用 overrides（例如界面“高级设置”里改的值）
    逐层覆盖；paths 下的相对路径一律换成以项目文件夹为起点的绝对路径，
    这样不管从哪个文件夹启动程序，找到的模型、数据池都是同一个位置。
可改进方向：
    检查参数取值是否合理（例如阈值必须在 0 到 1 之间），写错时给出中文提示。
测评指标：
    不涉及识别效果；由 tests/test_methods.py 检查覆盖是否正确、路径是否为绝对路径。

注意：本文件导入时不加载任何重依赖（PyYAML 也只在 load_config 里导入），
只装了 TensorFlow 的电脑也能 import pipeline.config 拿到 ROOT。
"""
import copy
import os
from pathlib import Path

# 项目文件夹（仓库根目录），即 pipeline 文件夹的上一级。
# 用 abspath 而不是 resolve：Windows 上 resolve 可能把映射的网络盘（如 Z:）改写成 \\服务器\共享 的形式。
ROOT = Path(os.path.abspath(__file__)).parents[1]

DEFAULT_CONFIG = ROOT / "config.yaml"


def _merge(base: dict, overrides: dict) -> dict:
    """把 overrides 逐层合并进 base 的副本：两边都是字典就往下一层合并，否则用 overrides 的值。"""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """读取配置，返回一个新的字典（调用方随便改，不会影响下一次读取）。

    path：配置文件路径，不填就用项目文件夹里的 config.yaml。
    overrides：要覆盖的参数，只写想改的部分，例如 {"vad": {"threshold": 0.3}}，其余参数保留。
    """
    import yaml  # PyYAML，在函数里导入，保持 import pipeline.config 轻量

    path = Path(path) if path is not None else DEFAULT_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"找不到配置文件：{path}。请确认项目文件夹里有 config.yaml。")
    # utf-8-sig：Windows 记事本保存的文件开头可能带 BOM，这样也能正确读取
    cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    if overrides:
        cfg = _merge(cfg, overrides)

    # paths 下的相对路径都以项目文件夹为起点，换成绝对路径字符串
    paths = cfg.get("paths") or {}
    for key, value in paths.items():
        if value is None:
            continue
        p = Path(str(value))
        if not p.is_absolute():
            p = ROOT / p
        paths[key] = os.path.abspath(p)
    if paths:
        cfg["paths"] = paths
    return cfg
