"""做法登记表：每个槽位有哪些做法、现在用哪一个。

基线做法：
    每个可替换的环节叫一个“槽位”（降噪、端点检测、热词纠错……），每个槽位登记若干“做法”。
    基线做法叫 baseline，写在 pipeline/stepN_*.py 里，永远保留；
    各组的改进做法叫 g1、g2……，写在 pipeline/groups/gN_*.py 里。
    写法：在函数上面加一行 @register("槽位", "做法名")，例如：

        @register("denoise", "g1")
        def denoise_g1(samples, sr, cfg):
            return samples

    用哪个做法由 config.yaml 的 methods 或界面“高级设置”决定，resolve(cfg) 负责按配置取出函数。
可改进方向：
    登记表本身不需要改；各组只改自己文件里的函数体。
测评指标：
    不涉及识别效果；由 tests/test_methods.py 检查登记、查找、排序和报错是否正确。

注意：本文件导入时不加载 sherpa-onnx、gradio、tensorflow 等重依赖。
"""
import importlib
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

# 全部槽位，顺序就是处理流程里的先后顺序
SLOTS = ("denoise", "enhance", "vad", "hotword", "diarize", "normalize", "classify", "clips")

# 槽位的中文名字（界面和报错信息里用）
SLOT_TITLES: dict[str, str] = {
    "denoise": "降噪",
    "enhance": "远距离增强",
    "vad": "端点检测",
    "hotword": "热词纠错",
    "diarize": "说话人分离",
    "normalize": "数字规范化",
    "classify": "话术分类",
    "clips": "疑似片段",
}

# 登记表：槽位 → {做法名: 函数}
_REGISTRY: dict[str, dict[str, Callable]] = {slot: {} for slot in SLOTS}

# load_all() 要导入的模块：导入时，模块里的 @register 就会把做法登记进来
_METHOD_MODULES = (
    "pipeline.step2_vad",
    "pipeline.step3_asr",
    "pipeline.step4_diarize",
    "pipeline.step5_normalize",
    "pipeline.step6_classify",
    "pipeline.step7_clips",
    "pipeline.groups",
)


def _check_slot(slot: str) -> None:
    """槽位名写错时给出中文提示。"""
    if slot not in _REGISTRY:
        raise ValueError(f"不认识的槽位“{slot}”。可用的槽位：{'、'.join(SLOTS)}")


def _check_new_name(slot: str, name: str) -> None:
    """做法名不能为空，也不能和这个槽位里已有的做法重名。"""
    if not isinstance(name, str) or not name:
        raise ValueError(f"槽位“{slot}”的做法名必须是非空的字符串，例如 g1")
    if name in _REGISTRY[slot]:
        raise ValueError(
            f"槽位“{slot}”（{SLOT_TITLES[slot]}）已经有叫“{name}”的做法了，"
            f"请换一个名字（各组的做法用自己的组号命名，如 g1）"
        )


def register(slot: str, name: str) -> Callable[[F], F]:
    """装饰器：把下面的函数登记为槽位 slot 的一种做法，名字叫 name。函数本身原样返回。

    槽位不存在或名字重复时抛 ValueError。
    """
    _check_slot(slot)
    _check_new_name(slot, name)

    def decorator(func: F) -> F:
        _check_new_name(slot, name)  # 同一个装饰器被用了两次时也要拦住
        _REGISTRY[slot][name] = func
        return func

    return decorator


def available(slot: str) -> list[str]:
    """槽位 slot 已登记的做法名：baseline 排第一，其余按名字排序。"""
    _check_slot(slot)
    names = sorted(_REGISTRY[slot])
    if "baseline" in names:
        names.remove("baseline")
        names.insert(0, "baseline")
    return names


def get_method(slot: str, name: str) -> Callable:
    """取出槽位 slot 里叫 name 的做法函数；找不到时抛 KeyError，信息里列出可用的做法。"""
    _check_slot(slot)
    if name not in _REGISTRY[slot]:
        names = "、".join(available(slot)) or "（还没有登记任何做法）"
        raise KeyError(
            f"槽位“{slot}”（{SLOT_TITLES[slot]}）没有叫“{name}”的做法。可用的做法：{names}。"
            f"请检查 config.yaml 的 methods 或界面“高级设置”里的选择"
        )
    return _REGISTRY[slot][name]


def load_all() -> None:
    """导入各步骤模块和各组文件，让里面的做法都登记进来。可以重复调用（模块只会导入一次）。"""
    for module_name in _METHOD_MODULES:
        importlib.import_module(module_name)


def resolve(cfg: dict) -> dict[str, Callable]:
    """按 cfg["methods"] 给每个槽位取出要用的做法函数；没写的槽位用 baseline。"""
    load_all()
    chosen = cfg.get("methods") or {}
    for slot in chosen:
        _check_slot(slot)  # config.yaml 里把槽位名写错了，要明确报错，不能悄悄用基线
    return {slot: get_method(slot, chosen.get(slot) or "baseline") for slot in SLOTS}
