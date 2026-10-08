"""测评：在数据池的录音（或剧本台词文字）上算指标，写成 CSV 表格和 Markdown 报告。

tools/evaluate.py（测一种做法）和 tools/compare.py（基线和改进各测一遍、出对比表）都调用这里的函数，
命令行工具本身只负责读参数、打印进度。指标怎么算、怎么看，见 docs/guides/evaluation.md。

六个指标：
    cer       字错率：录音 → 降噪 → 增强 → 端点检测 → 识别（测评模式，汉字读法）→ 热词纠错，
              把各段 text_raw 连起来，和参考文本 references/<文件编号>.txt 比（pipeline.align.cer_details）。
    hotwords  专名正确率、过度纠正次数：识别流程同上，数法见 hotword_stats 的说明。
    numbers   数字提取的正确率、召回率：不用录音，在剧本 1055 句台词上测（pipeline.step5_normalize.numbers_accuracy）。
    speakers  说话人标错的时长比例：只评有说话人标注 annotations/speakers/<文件编号>.csv 的录音。
              识别结果（缓存的段落）→ 说话人分离（用原始录音）→ 和标注比（pipeline.metrics.speaker_error_rate）。
    classify  话术分类的准确率、召回率、混淆矩阵、误报率：不用录音，在剧本 1055 句台词上测。
    clips     疑似片段起止误差：只评有片段标注 annotations/clips/<文件编号>.csv 的录音。
              识别结果 → 热词纠错 → 数字规范化 → 话术分类 → 片段做法定起止 → 和标注比（pipeline.metrics.clip_boundary_error）。

基线做法：
    1. 要测哪些录音：只看数据池清单 manifest.csv（入池时写的），不去翻 normalized/ 文件夹，
       这样没入池、入池失败的文件不会混进来。可以用 files 只测其中几段。
    2. 识别结果缓存：72 段录音约 5.4 小时，在 4 核电脑上识别一遍要 40 分钟左右。识别结果（带 text_raw 的段落）
       存进 数据池/asr_cache/<文件编号>.eval-<钥匙>.json。"钥匙"是下面这些东西合起来算的一个短指纹：
         - 录音文件本身的 SHA-256（复录换了文件 → 指纹变了）；
         - 降噪、增强、端点检测三个槽位用的做法名字，以及这三个做法所在代码文件的指纹（改了代码就重新识别）；
         - config.yaml 的 vad、asr 两节参数，以及与这三个做法同名的一节参数（如各组自己加的 g1: 一节）；
         - 识别模型的文件夹名。
       只要这些都没变，就直接用缓存，不再识别；所以只换热词纠错（hotword）做法、开关热词纠错时，
       不用重新识别，几秒钟就出结果。入池工具在复录时会删掉 asr_cache/<文件编号>.* ，缓存也跟着失效。
       怀疑缓存有问题时，命令行加 --no-cache 强制重新识别。
       说话人、片段两个指标也用同一份缓存：识别之后的步骤每次都重新做——分类、片段很快，说话人分离不缓存，是最慢的一步（比识别还慢）。
    3. 汇总：全体、按录音条件（Q 安静 / N 嘈杂教室 / F 口袋或远距离）、按组。
       汇总行的字错率 =（各段错字 + 漏字 + 多字之和）÷（各段参考文本字数之和），不是各段字错率的平均；
       说话人标错比例、片段起止误差的汇总也一样，先把分子、分母分别加起来再除。
    4. 报告：write_report 写 <名字>.csv（UTF-8 带 BOM，Excel 能直接打开）和 <名字>.md，
       Markdown 开头写数据池版本、做法、关键参数、生成时间和固定的局限说明 FIXED_LIMITATION。
       做法在运行时发出的提示（如第 6、7 组的模型用不了、退回关键词规则）也写在报告开头。
可改进方向：
    这是教师模板，各组不改本文件（改自己的 pipeline/groups/gN_*.py，再用这里的工具测）。可以讨论的有：
    过度纠正现在按次数估算，可以改成逐字对齐后逐处判断；字错率还可以按说话人、按段落长短分开看；
    说话人指标还可以另算漏掉、多出来的说话时间（合起来就是常见的 DER）。
测评指标：
    本文件就是测评工具；tests/test_evaluation.py 用模型自带的测试音频改名建一个临时数据池，
    检查流程能跑通、缓存有效、报告格式正确。在云端开发环境里不报告任何字错率数字。

注意：本文件导入时只用标准库；识别模型、ffmpeg 等在函数里用到时才加载。
"""
import copy
import csv
import hashlib
import inspect
import json
import logging
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# 每份 Markdown 报告开头都写这句话（红线第 7 条：如实报告）
FIXED_LIMITATION = "剧本数据上的测评结果不代表真实场景的效果"

# 数据池还没冻结过任何版本时，报告里"数据池版本"写这个
NOT_FROZEN = "未冻结（请先运行 tools/freeze_pool.py）"

# 在剧本台词文字上测的指标（numbers、classify），报告里"数据池版本"写这个
NO_POOL = "不用数据池（在剧本台词文字上测）"

# 识别（步骤 2）用到的槽位：这几个槽位换了做法，识别结果就会变，要重新识别
ASR_SLOTS = ("denoise", "enhance", "vad")

# 字错率、专名正确率涉及的槽位（报告里列出这几个槽位用的做法）
AUDIO_SLOTS = ("denoise", "enhance", "vad", "hotword")

# 说话人指标涉及的槽位（和整个流程一样，热词纠错之后再分说话人）；片段指标涉及的槽位（识别之后要经过热词纠错、数字规范化、话术分类才定片段）
SPEAKER_SLOTS = ("denoise", "enhance", "vad", "hotword", "diarize")
CLIP_SLOTS = ("denoise", "enhance", "vad", "hotword", "normalize", "classify", "clips")

# 说话人分离没配上任何人的段落写成这个（见 pipeline/step4_diarize.py），不算工具分出的说话人
UNKNOWN_SPEAKER = "未知"

# 命令行 --speakers ref：每段录音按它的标注里有几个人设说话人数（"设对人数"）
SPEAKERS_BY_ANNOTATION = "ref"

# 第 8 组的剧本是"正常讲解"对照组，单独算一个误报率
FP_GROUP = 8

# 两种人工标注：放在数据池的哪个文件夹（pipeline.pool.pool_paths 的键）、中文名
ANNOTATION_KINDS = {
    "speakers": ("annotations_speakers", "说话人标注"),
    "clips": ("annotations_clips", "疑似片段标注"),
}

# 报告的中文标题
REPORT_TITLES = {
    "cer": "字错率（cer）",
    "hotwords": "专名正确率与过度纠正（hotwords）",
    "numbers": "数字提取正确率（numbers）",
    "speakers": "说话人标错的时长比例（speakers）",
    "classify": "话术分类准确率、召回率与误报率（classify）",
    "clips": "疑似片段起止误差（clips）",
}

# CSV 和 Markdown 表格里，英文字段名对应的中文列名（本来就是中文的列名原样写出）
COLUMN_TITLES = {
    "stem": "文件编号",
    "group": "组",
    "condition": "录音条件",
    "files": "录音数",
    "seconds": "录音时长（秒）",
    "n_ref": "参考文本字数",
    "sub": "错字",
    "dele": "漏字",
    "ins": "多字",
    "cer": "字错率",
    "names_ref": "参考文本中的专名次数",
    "names_hit": "识别结果中命中的次数",
    "name_accuracy": "专名正确率",
    "variants_ref": "说错或简称的名称次数",
    "overcorrected": "过度纠正次数",
    "corrections": "热词纠错次数",
    "gold": "金标准个数",
    "pred": "提取个数",
    "hit": "命中个数",
    "precision": "正确率",
    "recall": "召回率",
    # 说话人（speakers）
    "speaker_error_rate": "说话人标错比例",
    "annotated_seconds": "标注的说话时长（秒）",
    "scored_seconds": "比对的时长（秒）",
    "error_seconds": "标错的时长（秒）",
    "ref_speakers": "标注的人数",
    "hyp_speakers": "分出的人数",
    "num_speakers": "人数设置",
    # 话术分类（classify）
    "rate": "比例",
    "count": "分子（句数）",
    "n": "分母（句数）",
    "script_id": "剧本编号",
    "line_no": "行号",
    "speaker": "角色",
    "text": "台词",
    "label": "标准答案",
    "predicted": "预测",
    "correct": "对错",
    "false_positive": "误报",
    # 片段（clips）
    "ref_clips": "标注片段数",
    "tool_clips": "工具片段数",
    "matched": "配上的对数",
    "unmatched_ref": "没配上的标注片段",
    "unmatched_hyp": "没配上的工具片段",
    "unmatched": "没配上的片段数",
    "tool_outside": "和标注都不重叠的工具片段",
    "wrong_category": "类别不同的工具片段",
    "start_mae": "起点平均误差（秒）",
    "end_mae": "终点平均误差（秒）",
}

# 对比表（compare.py）里每个指标比较哪几个数：[(字段名, 中文名)]。
# 只有一个数时分组项只写"全体""Q（安静）"……；有几个数时写成"全体·专名正确率"。
COMPARE_VALUES = {
    "cer": [("cer", "字错率")],
    "hotwords": [("name_accuracy", "专名正确率"), ("overcorrected", "过度纠正次数")],
    "numbers": [("precision", "正确率"), ("recall", "召回率")],
    "speakers": [("speaker_error_rate", "说话人标错比例")],
    "classify": [("rate", "比例")],
    "clips": [("start_mae", "起点平均误差（秒）"), ("end_mae", "终点平均误差（秒）"),
              ("unmatched", "没配上的片段数"), ("tool_outside", "和标注都不重叠的工具片段")],
}

# 每个指标的报告里写的说明（数法、口径）
CER_NOTES = [
    "字错率 =（错字 + 漏字 + 多字）÷ 参考文本字数。两边先去掉标点和空格、全角转半角、字母转大写，再逐字对齐。"
    "越低越好；多字很多时可能超过 1。",
    "识别用测评模式（不把数字转成阿拉伯数字，输出汉字读法），和参考文本的口径一致；"
    "一段录音的各段识别文字连起来，和整篇参考文本比。",
    "汇总行的字错率是把各段录音的错字、漏字、多字和字数分别加起来再除，不是各段字错率的平均。",
    "热词纠错（hotword 槽位）会改识别文字，所以它的做法和开关（hotword.enabled）也会影响字错率。",
    "比例都写成小数，如 0.1234 即 12.34%。",
]

HOTWORD_NOTES = [
    "专名：data/hotwords.txt 里的名称（虚构旅行社、店名、地名、行话）。识别结果用热词纠错之后的测评模式文字（text_raw）。",
    "数之前，参考文本和识别结果都先去掉标点和空格、全角转半角、字母转大写；"
    "再把 data/hotword_variants.csv 里人物故意说错或简称的名称（spoken_variant）挖掉，这些位置不算专名。",
    "数专名：从左往右找，同一个位置有长短几个名称都对得上时只算最长的那个"
    "（\"雾隐行舟旅行社\"不再另算\"雾隐行舟\"和\"雾隐\"），找到后跳过这几个字接着找。",
    "专名正确率 = 命中次数 ÷ 参考文本里的专名次数；每个名称的命中次数 = min(它在参考文本里的次数, 它在识别结果里的次数)。"
    "越高越好。参考文本里一个专名都没有时记 0，请同时看次数。",
    "过度纠正：按正确名称（correct_name）分别数。参考文本里它的各种说错、简称的说法（spoken_variant）一共出现 n 次。"
    "用下面两种办法数，取大的那个，最多记 n 次。越少越好。",
    "办法 A（纠错改掉的）：每种说法，纠错前的识别文字里有 b 次、纠错后剩 a 次、参考文本里有 r 次，"
    "记 min(r, b) − a 次（小于 0 记 0），各种说法加起来。也就是：识别对了的说错名称被热词纠错改掉了，不管改成了什么。",
    "办法 B（多出来的正确名称）：数正确名称\"一家\"在纠错后的识别文字里比参考文本里多出几次（小于 0 记 0）。"
    "\"一家\"是正确名称本身，加上热词表里是它的一部分、又不在它的说错说法里的简称："
    "听松阁玉器行 一家还有 听松阁，雾隐行舟旅行社 一家还有 雾隐行舟，松风晚渡旅行社 一家还有 松风晚渡，百年茶语 一家还有 茶语。"
    "（热词纠错只能换成一样长的热词，说错的\"听松坊\"只会被改成\"听松阁\"，所以简称也要算。）",
    "过度纠正是按次数估算的，没有逐字对齐：办法 B 在同一段录音里会互相抵消（一处正确名称被识别错、"
    "另一处说错的名称又被改成正确名称），这种情况靠办法 A 数出来；纠错前就识别错了的说错名称被改成正确名称时，靠办法 B 数。"
    "识别时就把说错的名称写成了正确名称（没有经过纠错），办法 B 也会算进去，所以热词纠错关着时过度纠正也可能不是 0。",
    "热词纠错开关（hotword.enabled）关闭时，基线做法不做任何纠错。专名正确率和过度纠正要一起看，只看一个会误导。",
]

NUMBERS_NOTES = [
    "在剧本台词文字上测（data/lines.csv，不用录音）：每句台词（汉字读法）用 normalize 做法（mode=\"spoken\"）"
    "转换并提取数字，和这一句的 numbers（标准写法）比较，同一句里去重后按集合比。",
    "正确率 = 命中个数 ÷ 提取个数；召回率 = 命中个数 ÷ 金标准个数；分母为 0 时记 0，请同时看个数。",
    "主指标：金额、电话、证号、合同号、订单号、时刻、日期；次要指标：数量、时长、其他。"
    "类型判断规则见 pipeline/step5_normalize.py 的 number_type。",
]

SPEAKER_NOTES = [
    "只评有人工说话人标注（数据池 annotations/speakers/<文件编号>.csv，列 start,end,speaker）的录音；"
    "怎么标见 docs/guides/annotation.md。",
    "流程：识别结果（测评模式的段落，和字错率共用缓存）→ diarize 做法用原始录音（没降噪的）给每段配说话人 → 和标注比。"
    "配成\"未知\"（和哪个说话人都不重叠）的段落不算工具分出的说话人。",
    "说话人标错比例（pipeline.metrics.speaker_error_rate）：时间切成 10 毫秒一格；工具的\"说话人1、说话人2\"和标注的角色名"
    "先找最佳对应（匈牙利算法，让对上的总时长最多，所以只是编号不同不算错）；只在两边都有人说话的时间里比，"
    "标错的时长 ÷ 比对的时长。越低越好。",
    "比对的时长：两边都有人说话的时间（两人同时说话时按人数算）。只有一边有人说话的时间（工具漏掉的、多出来的）不算在内，"
    "所以要同时看\"比对的时长\"和\"标注的说话时长\"：比对的时长很短时，标错比例再低也说明不了什么。",
    "工具的说话时间用的是端点检测切出来的段落（不是分离模型的原始结果），段落之间的停顿不算；"
    "一个段落中间换了人（没有停顿）时整段只算给一个人，这会算进标错的时长。",
    "人数设置：\"自动\"表示 diarize.num_speakers = -1（工具自己判断人数）；命令行 --speakers ref 表示每段录音按标注里的人数设（设对人数）。"
    "知道人数时一定要设人数，\"自动\"和\"设对人数\"两种都要报告。",
    "汇总行的标错比例 = 各段录音标错的时长之和 ÷ 比对的时长之和，不是各段比例的平均。",
]

CLASSIFY_NOTES = [
    "在剧本台词文字上测（data/lines.csv 的 1055 句，不用录音）：每句台词的文字直接交给 classify 做法，"
    "和这一句的 label（类别名）比较。剧本台词没有经过语音识别，真实录音的识别文字有错字，效果会更差。",
    "准确率（精确率）= 预测成这一类的句子里真是这一类的比例；召回率 = 这一类的句子里被找出来的比例；"
    "分母为 0 时记 0，请同时看句数。混淆矩阵：行是标准答案，列是预测，对角线上是分对的句数。",
    "误报率 = 标准答案为\"正常讲解\"的句子中，被预测成 5 个疑似类别（购物安排、费用、行程变更、服务态度、威胁消费）之一的比例。"
    "全部 24 个剧本算一次，第 8 组（正常讲解对照组）的剧本再单独算一次；正常讲解被分成\"其他\"不算误报，"
    "第 8 组里标为费用、行程变更、购物安排的句子分错了也不算误报。误报会冤枉人，越低越好。",
    "关键词规则（baseline）的关键词是参照剧本台词写的，在同一批剧本上测，数字偏乐观（虚高），换个说法就可能失效。",
    "训练好的模型（tf_model、g6、g7）最后是用全部剧本台词训练的，在同样的台词上测等于\"考原题\"，数字会明显虚高，"
    "不能当成模型的效果；模型的主结果看训练脚本的按组留一报告"
    "（如 reports/g6/train_g6_logo_extra.md、train_g6_logo_no_extra.md）。",
    "没装 TensorFlow 或没有训练好的模型时，tf_model、g6、g7 会退回关键词规则，报告开头的\"运行时的提示\"会写明；"
    "这时测出来的就是关键词规则的结果。",
    "类别名用数据里的写法：\"威胁消费\"在界面上显示为\"消费施压\"。比例都写成小数，如 0.1234 即 12.34%。",
]

CLIPS_NOTES = [
    "只评有人工片段标注（数据池 annotations/clips/<文件编号>.csv，列 start,end,label）的录音；怎么标见 docs/guides/annotation.md。",
    "流程和整理录音的测评模式一样：识别结果（和字错率共用缓存）→ 热词纠错 → 数字规范化（mode=\"spoken\"）→ 话术分类 → "
    "clips 做法定每个片段的起止（只算起止，不剪音频、不写文件）。为了省时间不做说话人分离，片段做法拿到的段落里没有说话人。",
    "配对（pipeline.metrics.clip_boundary_error）：两边的片段按重叠秒数从多到少一一配对，只是挨着（重叠 0 秒）不算配上。"
    "起点、终点平均误差 = 每对 |工具 − 标注| 的平均（秒），越小越好；一对也没配上时空着。",
    "没配上的片段数 = 没配上的标注片段（工具漏掉的）+ 没配上的工具片段（多出来的）。基线一个段落出一个片段，"
    "标注的一个片段往往包含好几句话，只有一个工具片段能和它配对，其余的都算\"没配上的工具片段\"——合并相邻的片段是第 8 组的改进方向。",
    "和标注都不重叠的工具片段：和任何标注片段都不重叠，即标注人认为没有纠纷的地方被剪成了疑似片段（片段层面的误报），越少越好。",
    "类别不同的工具片段：和它重叠最多的标注片段类别不一样（时间对上了，类别标错了）。这主要由话术分类（第 6、7 组）决定。",
    "汇总行的平均误差 = 各段录音误差之和 ÷ 配上的对数之和，不是各段平均误差的平均。",
    "句子层面的误报率（正常讲解被标成疑似的比例）用 tools/evaluate.py classify 在剧本台词上测。",
]

NOTES = {"cer": CER_NOTES, "hotwords": HOTWORD_NOTES, "numbers": NUMBERS_NOTES,
         "speakers": SPEAKER_NOTES, "classify": CLASSIFY_NOTES, "clips": CLIPS_NOTES}


# ======================== 做法 ========================


def check_methods(methods: dict | None) -> None:
    """检查 {槽位: 做法名} 里的槽位和做法都存在；不存在时抛 ValueError（中文信息里列出可用的做法）。"""
    from pipeline.methods import SLOT_TITLES, SLOTS, available, load_all

    load_all()  # 让各组文件里的做法都登记进来
    for slot, name in (methods or {}).items():
        if slot not in SLOTS:
            raise ValueError(f"不认识的槽位“{slot}”。可用的槽位：{'、'.join(SLOTS)}")
        names = available(slot)
        if name not in names:
            raise ValueError(f"槽位“{slot}”（{SLOT_TITLES[slot]}）没有叫“{name}”的做法。可用的做法：{'、'.join(names)}")


def with_methods(cfg: dict, methods: dict | None) -> dict:
    """返回一份新的配置：在 cfg["methods"] 的基础上，换上 methods 里指定的做法（{槽位: 做法名}）。

    methods 里的做法和 config.yaml 里写的做法都检查一遍，写错了（如把 g1 写成 g11）就抛 ValueError，
    中文信息里列出可用的做法；不然要等到识别时才报一大段英文错。
    """
    from pipeline.methods import SLOTS

    check_methods(methods)
    cfg = copy.deepcopy(cfg)
    cfg["methods"] = dict(cfg.get("methods") or {})
    cfg["methods"].update(methods or {})
    from_config = {slot: _method_name(cfg, slot) for slot in SLOTS if slot not in (methods or {})}
    try:
        check_methods(from_config)
    except ValueError as e:
        raise ValueError(f"config.yaml 的 methods 一节写错了：{e}") from None
    return cfg


def _method_name(cfg: dict, slot: str) -> str:
    """配置里这个槽位用的做法名字，没写就是 baseline。"""
    return (cfg.get("methods") or {}).get(slot) or "baseline"


def _get_method(cfg: dict, slot: str) -> Callable:
    from pipeline.methods import get_method, load_all

    load_all()
    return get_method(slot, _method_name(cfg, slot))


# ======================== 数据池 ========================


def pool_version(root) -> str:
    """数据池最新冻结的版本名（如 v1，见 pipeline.pool.current_version）；还没冻结过时返回 NOT_FROZEN。

    冻结之后，测评要读的文件（清单、参考文本、标注、录音）又改过的（例如有人在"数据校对"页又保存了参考文本），
    在版本名后面注明"和 v1 不一致"：这时测出来的结果和别的组用 v1 测的不能比，要冻结新版本、基线和改进都重测。
    """
    from pipeline.pool import current_version, version_changes

    version = current_version(root)
    if not version:
        return NOT_FROZEN
    changes = version_changes(root, version)
    if not changes:
        return version
    shown = "、".join(changes[:3]) + ("……" if len(changes) > 3 else "")
    logger.warning("数据池冻结为 %s 以后又改了 %d 个文件（%s），测评结果和 %s 不一致", version, len(changes), shown, version)
    return (f"{version}（冻结后有 {len(changes)} 个文件改动，和 {version} 不一致：{shown}；"
            f"请老师冻结新版本，基线和改进都用新版本重测）")


def pool_items(root, files: list[str] | None = None) -> list[dict]:
    """数据池清单（manifest.csv）里的录音，按文件编号排好。

    每项：stem（文件编号，如 G1-S1-Q）、wav（转换后的录音路径）、reference（参考文本路径）、
    group（组号，整数）、condition（录音条件代码 Q / N / F）。
    files：只要其中几段，写文件编号（如 ["G1-S1-Q"]，带不带 .wav 都行）；有不在清单里的就抛 ValueError。
    """
    from pipeline.pool import parse_recording_name, pool_paths, read_csv_rows

    root = Path(root)
    paths = pool_paths(root)
    items = {}
    for row in read_csv_rows(paths["manifest"]):
        stem = (row.get("文件编号") or "").strip()
        if not stem:
            continue
        info = parse_recording_name(f"{stem}.wav")
        wav = (row.get("转换后文件") or "").strip() or f"normalized/{stem}.wav"
        items[stem] = {
            "stem": stem,
            "wav": root / wav,
            "reference": paths["references"] / f"{stem}.txt",
            "group": info["group"],
            "condition": info["condition"],
        }

    if files:
        wanted = [Path(name.strip()).stem if "." in name else name.strip() for name in files]  # 去掉扩展名
        unknown = [stem for stem in wanted if stem not in items]
        if unknown:
            raise ValueError(f"这些录音不在数据池清单（manifest.csv）里：{'、'.join(unknown)}。"
                             f"文件编号形如 G1-S1-Q；请先入池（tools/ingest_pool.py），或检查编号有没有写错")
        return [items[stem] for stem in sorted(set(wanted))]
    return [items[stem] for stem in sorted(items)]


def _checked_items(root, files) -> list[dict]:
    """取要测的录音，并在开始识别之前检查录音和参考文本都在（不要等了几十分钟才发现少文件）。"""
    from pipeline.pool import pool_paths

    items = pool_items(root, files)
    if not items:
        raise ValueError(f"数据池里还没有入池的录音（{pool_paths(root)['manifest']} 不存在或是空的）。"
                         f"请先把录音放进 raw 文件夹，运行 tools/ingest_pool.py")
    no_wav = [item["stem"] for item in items if not item["wav"].is_file()]
    if no_wav:
        raise FileNotFoundError(f"清单里有、但找不到转换后的录音：{'、'.join(no_wav)}（应该在 normalized 文件夹里）。"
                                f"请检查数据池是否拷全了")
    no_ref = [item["stem"] for item in items if not item["reference"].is_file()]
    if no_ref:
        raise FileNotFoundError(f"找不到参考文本：{'、'.join(no_ref)}（应该在 references 文件夹里）。"
                                f"请先运行 python tools/export_references.py 生成参考文本，再校对")
    return items


def annotated_items(root, kind: str, files: list[str] | None = None) -> list[dict]:
    """有人工标注的录音（kind 为 "speakers" 说话人标注，或 "clips" 疑似片段标注），按文件编号排好。

    每项和 pool_items 一样，另加 annotation_path（标注表格的路径）和 annotation（读出来的标注）：
        speakers：[(开始秒, 结束秒, 角色名), ...]（pipeline.annotations.read_turns_csv）
        clips：   [(开始秒, 结束秒, 类别名), ...]（read_clips_csv；"消费施压"已换成"威胁消费"）
    没有标注的录音不评。files 里指定了没有标注的录音、一段有标注的录音也没有、找不到录音、
    标注表格有错时，在开始识别之前就报错（中文说明怎么办），不白等。
    """
    from pipeline.annotations import read_clips_csv, read_turns_csv
    from pipeline.pool import pool_paths

    key, title = ANNOTATION_KINDS[kind]
    folder = pool_paths(root)[key]
    candidates = pool_items(root, files)
    if not candidates:
        raise ValueError(f"数据池里还没有入池的录音（{pool_paths(root)['manifest']} 不存在或是空的）。"
                         f"请先把录音放进 raw 文件夹，运行 tools/ingest_pool.py")
    how = "标注方法见 docs/guides/annotation.md，用 tools/labels_to_csv.py 把 Audacity 导出的标签转成表格放进去"
    items = [dict(item, annotation_path=folder / f"{item['stem']}.csv") for item in candidates]
    missing = [item["stem"] for item in items if not item["annotation_path"].is_file()]
    if files and missing:
        raise ValueError(f"这些录音没有{title}：{'、'.join(missing)}（应该在 {folder} 里，文件名如 {missing[0]}.csv）。{how}")
    items = [item for item in items if item["annotation_path"].is_file()]
    if not items:
        raise ValueError(f"数据池里还没有{title}（{folder} 里没有和入池录音同名的表格）。{how}")
    no_wav = [item["stem"] for item in items if not item["wav"].is_file()]
    if no_wav:
        raise FileNotFoundError(f"清单里有、但找不到转换后的录音：{'、'.join(no_wav)}（应该在 normalized 文件夹里）。"
                                f"请检查数据池是否拷全了")
    read = read_turns_csv if kind == "speakers" else read_clips_csv
    for item in items:
        item["annotation"] = read(item["annotation_path"])  # 表格有错时抛 ValueError，说明是哪个文件的哪一行
    return items


def read_reference(path) -> str:
    """读参考文本。记事本存的 UTF-8（带不带 BOM 都行）、ANSI（GBK）、"Unicode"（UTF-16）都能读（见 pipeline.textio）。"""
    from pipeline.textio import read_text

    return read_text(path, hint="请用记事本打开这个参考文本，另存为时编码选 UTF-8。")


# ======================== 识别（带缓存） ========================


def _source_hash(func: Callable) -> str:
    """做法函数所在代码文件的指纹：同学改了自己文件里的代码，指纹就变了，缓存跟着失效。"""
    try:
        source = Path(inspect.getsourcefile(func)).read_bytes()
    except (TypeError, OSError):  # 找不到源文件（很少见）：只用函数名
        return f"{func.__module__}.{func.__qualname__}"
    return hashlib.sha256(source).hexdigest()[:12]


def cache_key(wav, cfg: dict) -> str:
    """识别结果缓存的"钥匙"：12 位的短指纹。会影响识别结果的东西变了，钥匙就变（见本文件开头的说明）。

    热词纠错、说话人分离、话术分类等识别之后才做的步骤不算在里面，换它们的做法时可以接着用缓存。
    """
    from pipeline.audio import sha256_file
    from pipeline.models import model_path

    model = model_path(cfg, "sense_voice_model")
    parts = {
        "wav": sha256_file(wav),
        "vad": cfg.get("vad"),
        "asr": cfg.get("asr"),
        "model": f"{model.parent.name}/{model.name}",
        "methods": {},
        "code": {},
        "group_params": {},
    }
    for slot in ASR_SLOTS:
        name = _method_name(cfg, slot)
        parts["methods"][slot] = name
        parts["code"][slot] = _source_hash(_get_method(cfg, slot))
        if name in cfg:  # 各组在 config.yaml 里加的同名一节参数（如 g1:）
            parts["group_params"][name] = cfg[name]
    text = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _save_cache(path: Path, segments: list[dict], meta: dict) -> None:
    """写缓存：先写临时文件再改名，写到一半断电或几台电脑同时写也不会留下半个文件。写不进去就算了（只是慢一点）。"""
    from pipeline.schema import write_json

    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)  # 只读的共享文件夹建不了：照样往下测，只是不留缓存
        write_json(tmp, segments, meta)
        os.replace(tmp, path)
    except OSError as e:
        logger.warning("识别结果缓存写不进去（不影响这次测评，只是下次要重新识别）：%s", e)
        if tmp.exists():
            tmp.unlink()


def recognize_item(root, item: dict, cfg: dict, use_cache: bool = True,
                   progress=None) -> tuple[list[dict], dict]:
    """识别数据池里的一段录音（测评模式），返回 (段落列表, 情况)。

    段落带 text_raw（汉字读法、没有标点），还没做热词纠错。
    情况：{"seconds": 录音时长（秒）, "cached": 是否用了缓存, "elapsed": 这次识别用了几秒}。
    cfg["methods"] 决定降噪、增强、端点检测用哪个做法（见 with_methods）。
    progress：可选，progress(完成比例 0~1, 中文说明)，识别时显示进度（网页的“数据校对”页用）。
    """
    from pipeline import step2_vad, step3_asr
    from pipeline.audio import SR, read_wav
    from pipeline.pool import pool_paths
    from pipeline.schema import read_json

    key = cache_key(item["wav"], cfg)
    folder = pool_paths(root)["asr_cache"]
    cache = folder / f"{item['stem']}.eval-{key}.json"  # 以文件编号开头：复录时入池工具会删掉它
    if use_cache and cache.is_file():
        try:
            segments, meta = read_json(cache)
            if meta.get("key") == key:
                return segments, {"seconds": float(meta["seconds"]), "cached": True, "elapsed": 0.0}
        except (OSError, ValueError, KeyError, TypeError):
            pass  # 缓存文件坏了：重新识别

    started = time.perf_counter()
    samples = read_wav(item["wav"])
    seconds = round(len(samples) / SR, 2)
    processed, segments = step2_vad.detect_speech(samples, SR, cfg)
    segments = step3_asr.recognize(processed, SR, segments, cfg, mode="eval", progress=progress)
    elapsed = time.perf_counter() - started

    meta = {
        "stem": item["stem"],
        "key": key,
        "mode": "eval",
        "methods": {slot: _method_name(cfg, slot) for slot in ASR_SLOTS},
        "seconds": seconds,
        "elapsed": round(elapsed, 1),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_cache(cache, segments, meta)
    return segments, {"seconds": seconds, "cached": False, "elapsed": elapsed}


def _run_items(root, cfg: dict, items: list[dict], progress, use_cache: bool,
               score: Callable[[dict, list[dict]], tuple[dict, str]]) -> list[dict]:
    """所有录音指标共用的流程：逐段录音 识别（或用缓存）→ score 算这一段的指标，并显示进度。

    score(录音, 识别出的段落) 返回 (指标字典, 进度里显示的一句话)。段落是测评模式的（有 text_raw），
    score 可以随意改它（每次都是从缓存新读出来的）。返回每段录音一行：stem、group、condition、指标……、seconds。
    """
    def say(message: str) -> None:
        if progress is not None:
            progress(message)
        else:
            logger.info(message)

    rows = []
    cached = 0
    started = time.perf_counter()
    for n, item in enumerate(items, start=1):
        head = f"[{n}/{len(items)}] {item['stem']}"
        say(f"{head}：处理中……")
        segments, info = recognize_item(root, item, cfg, use_cache)
        values, text = score(item, segments)
        rows.append({"stem": item["stem"], "group": item["group"], "condition": item["condition"],
                     **values, "seconds": info["seconds"]})
        if info["cached"]:
            cached += 1
            how = "用了缓存的识别结果"
        else:
            how = f"识别用时 {info['elapsed']:.0f} 秒"
        say(f"{head}：完成（{how}），{text}")
    minutes = (time.perf_counter() - started) / 60
    say(f"全部完成：{len(items)} 段录音，用时 {minutes:.1f} 分钟（其中 {cached} 段用了缓存的识别结果）")
    return rows


def _run_audio_eval(root, cfg: dict, files, progress, use_cache: bool,
                    score: Callable[[str, str, list[dict], str], tuple[dict, str]]) -> list[dict]:
    """cer 和 hotwords 共用的流程：逐段录音 识别（或用缓存）→ 热词纠错 → score 算这一段的指标。

    score(参考文本, 识别文字, 纠错后的段落, 纠错前的识别文字) 返回 (指标字典, 进度里显示的一句话)。
    识别文字 = 各段 text_raw 连起来。返回每段录音一行：stem、group、condition、指标……、seconds。
    """
    items = _checked_items(root, files)
    hotword = _get_method(cfg, "hotword")

    def score_item(item: dict, segments: list[dict]) -> tuple[dict, str]:
        # 和 run_pipeline 的测评模式一样：纠错之前先把 text 设成 text_raw（有的做法可能只看 text）
        segments = [dict(seg, text=seg.get("text_raw") or "") for seg in segments]
        before = "".join(seg["text"] for seg in segments)  # 纠错之前的识别文字（数过度纠正要用）
        segments = hotword(segments, None, cfg)  # 热词纠错（hotwords=None 表示用 data/hotwords.txt）
        hypothesis = "".join(seg.get("text_raw") or "" for seg in segments)
        return score(read_reference(item["reference"]), hypothesis, segments, before)

    return _run_items(root, cfg, items, progress, use_cache, score_item)


# ======================== 汇总 ========================


def _summarize(rows: list[dict], sum_keys: list[str], add_rates: Callable[[dict], dict]) -> dict:
    """把每段录音的行加起来：全体、按录音条件、按组。add_rates 在加好的计数后面补上比例。

    返回 {"all": {...}, "by_condition": {"Q": {...}}, "by_group": {1: {...}}, "overview": [带"分组项"的行]}。
    """
    from pipeline.pool import CONDITIONS

    def total(part: list[dict]) -> dict:
        stats = {"files": len(part)}
        for key in sum_keys:
            stats[key] = sum(row[key] for row in part)
        stats["seconds"] = round(sum(row["seconds"] for row in part), 1)
        return add_rates(stats)

    by_condition = {code: total([r for r in rows if r["condition"] == code])
                    for code in CONDITIONS if any(r["condition"] == code for r in rows)}
    by_group = {group: total([r for r in rows if r["group"] == group])
                for group in sorted({r["group"] for r in rows})}
    summary = {"all": total(rows), "by_condition": by_condition, "by_group": by_group}

    overview = [{"分组项": "全体", **summary["all"]}]
    overview += [{"分组项": f"{code}（{CONDITIONS[code]}）", **stats} for code, stats in by_condition.items()]
    overview += [{"分组项": f"第 {group} 组", **stats} for group, stats in by_group.items()]
    summary["overview"] = overview
    return summary


def _key_params(cfg: dict, sections=("vad", "asr", "hotword")) -> dict:
    """报告里写的关键参数：config.yaml 里 sections 这几节的参数（默认端点检测、识别、热词纠错）和识别模型。"""
    from pipeline.models import model_path

    params = {}
    for section in sections:
        for key, value in (cfg.get(section) or {}).items():
            params[f"{section}.{key}"] = value
    params["识别模型"] = model_path(cfg, "sense_voice_model").parent.name
    return params


def _audio_info(root, cfg: dict, slots=AUDIO_SLOTS, sections=("vad", "asr", "hotword")) -> dict:
    """录音指标报告开头用的信息：数据池、数据池版本、slots 这几个槽位用的做法、关键参数。"""
    return {"pool": str(root), "pool_version": pool_version(root),
            "methods": {slot: _method_name(cfg, slot) for slot in slots},
            "params": _key_params(cfg, sections)}


# ======================== 字错率 ========================


def _cer_rates(stats: dict) -> dict:
    errors = stats["sub"] + stats["dele"] + stats["ins"]
    stats["cer"] = errors / stats["n_ref"] if stats["n_ref"] else 0.0
    return stats


def eval_cer(root, cfg: dict, methods: dict | None = None, files: list[str] | None = None,
             progress: Callable[[str], None] | None = None, use_cache: bool = True) -> tuple[list[dict], dict]:
    """字错率测评。返回 (每段录音一行, 汇总)。

    methods：{槽位: 做法名}，如 {"denoise": "g1"}；没写的槽位按 cfg["methods"]，那里也没写就用 baseline。
    files：只测这几段录音（文件编号）；不填就测清单里的全部录音。
    progress：每段录音开始、结束时各调用一次 progress(中文说明)，命令行传 print 就能看到进度。
    use_cache：False 时不用缓存，全部重新识别。
    每行：stem、group、condition、cer、sub（错字）、dele（漏字）、ins（多字）、n_ref（参考文本字数）、seconds（录音时长）。
    汇总：all（全体）、by_condition（按条件）、by_group（按组）、overview（这三样排成表）、tables、info（报告开头用）。
    """
    from pipeline.align import cer_details

    cfg = with_methods(cfg, methods)

    def score(reference: str, hypothesis: str, segments: list[dict], before: str):
        details = cer_details(reference, hypothesis)
        return details, f"字错率 {details['cer']:.4f}"

    rows = _run_audio_eval(root, cfg, files, progress, use_cache, score)
    summary = _summarize(rows, ["n_ref", "sub", "dele", "ins"], _cer_rates)
    summary["info"] = _audio_info(root, cfg)
    summary["tables"] = [{"title": "汇总（全体、按录音条件、按组）", "rows": summary["overview"]}]
    return rows, summary


# ======================== 专名正确率、过度纠正 ========================


def count_names(text: str, names: list[str]) -> Counter:
    """数 text 里每个名称出现几次：从左往右找，同一个位置长短几个名称都对得上时只算最长的，找到后跳过这几个字。

    例：count_names("雾隐行舟旅行社和雾隐", ["雾隐", "雾隐行舟旅行社"]) → {"雾隐行舟旅行社": 1, "雾隐": 1}
    """
    words = sorted({name for name in names if name}, key=len, reverse=True)  # 长的排前面
    counts = Counter()
    i = 0
    while i < len(text):
        for word in words:
            if text.startswith(word, i):
                counts[word] += 1
                i += len(word)
                break
        else:  # 这个位置没有名称，往后挪一个字
            i += 1
    return counts


def name_families(variants: list[dict], hotwords: list[str]) -> dict[str, list[str]]:
    """每个正确名称的"一家"：正确名称本身，加上热词表里是它的一部分、又不在它任何一种说错/简称说法里的名称。

    热词纠错只能把一段字换成一样长的热词，所以说错的"听松坊"（3 个字）只会被改成"听松阁"，
    不会被改成"听松阁玉器行"；数过度纠正时这些简称也要算。用 data/hotword_variants.csv 和 data/hotwords.txt：
        听松阁玉器行   → [听松阁玉器行, 听松阁]
        雾隐行舟旅行社 → [雾隐行舟旅行社, 雾隐行舟]（"雾隐"也在说错的"雾隐晚渡"里，不算）
        松风晚渡旅行社 → [松风晚渡旅行社, 松风晚渡]（"松风"也在说错的"松风行舟"里，不算）
        晓月银坊       → [晓月银坊]（"晓月"也在说错的"晓月阁"里，不算）
    variants 的名称和 hotwords 要用同样的办法归一过（hotword_stats 里都先 normalize_for_cer）。
    """
    spoken_of: dict[str, list[str]] = {}
    for row in variants:
        spoken_of.setdefault(row["correct_name"], []).append(row["spoken_variant"])
    families = {}
    for correct, spoken in spoken_of.items():
        family = [correct] + [name for name in hotwords
                              if name and name in correct and not any(name in variant for variant in spoken)]
        families[correct] = list(dict.fromkeys(family))  # 去掉重复，保持顺序
    return families


def hotword_stats(reference: str, hypothesis: str, hotwords: list[str], variants: list[dict],
                  before: str | None = None) -> dict:
    """一段录音的专名正确率和过度纠正次数（数法也写在报告的说明 HOTWORD_NOTES 里）。

    reference：参考文本；hypothesis：热词纠错之后的识别文字（测评模式 text_raw 连起来）；
    hotwords：热词表；variants：data/hotword_variants.csv 的行（spoken_variant 说错或简称的说法，correct_name 正确名称）；
    before：热词纠错之前的识别文字（测评时一定会给；不给就只用下面第 3 步的办法 B）。

    1. 所有文字都先用 normalize_for_cer 归一（去标点和空格、全角转半角、字母大写），名称也一样归一。
    2. 专名：先把两边文字里的说错/简称的说法挖掉（这些位置不算专名），再用 count_names 数每个热词出现几次
       （长名称优先）。每个名称的命中次数 = min(参考文本里的次数, 识别结果里的次数)。
       专名正确率 = 命中次数之和 ÷ 参考文本里的次数之和（参考文本里没有专名时记 0.0）。
    3. 过度纠正：按正确名称分别数。参考文本里它的各种说错/简称说法共出现 n 次。用两种办法数，取大的那个，最多记 n 次：
       A. 纠错改掉的：每种说法，纠错前的识别文字里有 b 次、纠错后剩 a 次，参考文本里有 r 次，
          记 min(r, b) − a 次（小于 0 记 0）；各种说法加起来。意思是：识别对了的说错名称被纠错改掉了（不管改成什么）。
       B. 多出来的正确名称：用 count_names 数正确名称"一家"（见 name_families，如 听松阁玉器行 和 听松阁）
          一共出现几次，纠错后的识别文字比参考文本多出几次就记几次（小于 0 记 0）。
          意思是：说错的名称最后变成了正确名称（被纠错改的，或者识别时就写成了正确名称）。
       两种办法都没有逐字对齐，是按次数估算：B 在同一段录音里会互相抵消（一处正确名称识别错了、
       另一处说错的名称又被改成正确名称），这种情况靠 A 数出来；纠错前就识别错的说法被改成正确名称时，靠 B 数。
    返回 names_ref、names_hit、name_accuracy、variants_ref、overcorrected，
    以及 by_name（{名称: {"ref", "hit"}}）、by_correct（{正确名称: {"variants", "over"}}）给报告的明细表用。
    """
    from pipeline.text_norm import normalize_for_cer

    ref = normalize_for_cer(reference)
    hyp = normalize_for_cer(hypothesis)
    names = [normalize_for_cer(word) for word in hotwords]
    pairs = [(normalize_for_cer(row["spoken_variant"]), normalize_for_cer(row["correct_name"])) for row in variants]
    spoken = [variant for variant, _ in pairs]

    # 专名：先挖掉说错/简称的说法（换成一个"|"，名称不会跨过它），长的说法先挖
    ref_masked, hyp_masked = ref, hyp
    for variant in sorted(set(spoken), key=len, reverse=True):
        if variant:
            ref_masked = ref_masked.replace(variant, "|")
            hyp_masked = hyp_masked.replace(variant, "|")
    ref_counts = count_names(ref_masked, names)
    hyp_counts = count_names(hyp_masked, names)
    by_name = {name: {"ref": n, "hit": min(n, hyp_counts[name])} for name, n in ref_counts.items()}
    names_ref = sum(item["ref"] for item in by_name.values())
    names_hit = sum(item["hit"] for item in by_name.values())

    # 过度纠正：按正确名称分别数
    families = name_families([{"spoken_variant": v, "correct_name": c} for v, c in pairs], names)
    spoken_ref = count_names(ref, spoken)
    spoken_before = count_names(normalize_for_cer(before), spoken) if before is not None else None
    spoken_after = count_names(hyp, spoken)
    by_correct = {}
    removed = Counter()  # 办法 A：纠错改掉的说错名称，按正确名称加起来
    for variant, correct in pairs:
        if not spoken_ref[variant]:
            continue
        entry = by_correct.setdefault(correct, {"variants": 0, "over": 0})
        entry["variants"] += spoken_ref[variant]
        if spoken_before is not None:
            removed[correct] += max(0, min(spoken_ref[variant], spoken_before[variant]) - spoken_after[variant])
    for correct, entry in by_correct.items():
        family = families[correct]
        extra = sum(count_names(hyp, family).values()) - sum(count_names(ref, family).values())  # 办法 B
        entry["over"] = min(entry["variants"], max(removed[correct], extra, 0))

    return {
        "names_ref": names_ref,
        "names_hit": names_hit,
        "name_accuracy": names_hit / names_ref if names_ref else 0.0,
        "variants_ref": sum(entry["variants"] for entry in by_correct.values()),
        "overcorrected": sum(entry["over"] for entry in by_correct.values()),
        "by_name": by_name,
        "by_correct": by_correct,
    }


def _hotword_rates(stats: dict) -> dict:
    stats["name_accuracy"] = stats["names_hit"] / stats["names_ref"] if stats["names_ref"] else 0.0
    return stats


def eval_hotwords(root, cfg: dict, methods: dict | None = None, files: list[str] | None = None,
                  progress: Callable[[str], None] | None = None, use_cache: bool = True) -> tuple[list[dict], dict]:
    """专名正确率、过度纠正测评（第 5 组主指标）。参数同 eval_cer；数法见 hotword_stats。

    每行：stem、group、condition、names_ref、names_hit、name_accuracy、variants_ref、overcorrected、
    corrections（热词纠错一共改了几处）、seconds。
    汇总同 eval_cer，另外 tables 里多两张明细表：各专名的命中情况、过度纠正明细。
    """
    from pipeline.data import load_hotword_variants, load_hotwords

    cfg = with_methods(cfg, methods)
    hotwords = load_hotwords()
    variants = load_hotword_variants()
    by_name: dict[str, Counter] = {}
    by_correct: dict[str, Counter] = {}

    def score(reference: str, hypothesis: str, segments: list[dict], before: str):
        stats = hotword_stats(reference, hypothesis, hotwords, variants, before=before)
        for name, item in stats.pop("by_name").items():
            by_name.setdefault(name, Counter()).update(item)
        for name, item in stats.pop("by_correct").items():
            by_correct.setdefault(name, Counter()).update(item)
        stats["corrections"] = sum(len(seg.get("corrections") or []) for seg in segments)
        text = f"专名正确率 {stats['name_accuracy']:.4f}，过度纠正 {stats['overcorrected']} 次"
        return stats, text

    rows = _run_audio_eval(root, cfg, files, progress, use_cache, score)
    summary = _summarize(rows, ["names_ref", "names_hit", "variants_ref", "overcorrected", "corrections"],
                         _hotword_rates)
    summary["info"] = _audio_info(root, cfg)

    # 明细：命中最少（错得最多）的名称排前面，方便看哪些名称最需要纠错
    name_rows = [{"名称": name, "参考文本中出现": c["ref"], "识别结果中命中": c["hit"],
                  "专名正确率": c["hit"] / c["ref"] if c["ref"] else 0.0}
                 for name, c in by_name.items()]
    name_rows.sort(key=lambda r: (r["专名正确率"], -r["参考文本中出现"], r["名称"]))
    over_rows = [{"正确名称": name, "说错或简称的出现次数": c["variants"], "过度纠正次数": c["over"]}
                 for name, c in sorted(by_correct.items())]
    summary["tables"] = [
        {"title": "汇总（全体、按录音条件、按组）", "rows": summary["overview"]},
        {"title": "各专名的命中情况（错得多的排前面）", "rows": name_rows},
        {"title": "过度纠正明细（按正确名称）", "rows": over_rows},
    ]
    return rows, summary


# ======================== 数字提取（剧本文字） ========================


def eval_numbers_lines(cfg: dict, methods: dict | None = None) -> dict:
    """在剧本 1055 句台词上测数字提取（第 4 组主指标），不用录音。

    methods：如 {"normalize": "g4"}；不填就用 cfg["methods"]["normalize"]（默认 baseline）。
    返回 numbers_accuracy 的结果：by_type（每种类型的 gold、hit、pred、precision、recall）、main（主指标）、
    secondary（次要指标）；另加 overview（主指标、次要指标、各类型排成的表）和 info（报告开头用）。
    """
    from pipeline.data import load_lines
    from pipeline.step5_normalize import NUMBER_TYPES, numbers_accuracy

    cfg = with_methods(cfg, methods)
    name = _method_name(cfg, "normalize")
    lines = load_lines()
    result = numbers_accuracy(lines, method=_get_method(cfg, "normalize"), cfg=cfg)
    result["overview"] = ([{"分组项": "主指标", **result["main"]}, {"分组项": "次要指标", **result["secondary"]}]
                          + [{"分组项": kind, **result["by_type"][kind]} for kind in NUMBER_TYPES])
    result["tables"] = []
    result["info"] = {"pool": None, "pool_version": NO_POOL,
                      "methods": {"normalize": name}, "params": {"台词句数": len(lines)}}
    return result


# ======================== 做法发出的提示 ========================


def _capture_warnings(func: Callable, *args):
    """调用 func(*args)，把它发出的警告收集起来，返回 (结果, 警告文字列表（去掉重复）)。

    例如第 6、7 组的做法在没装 TensorFlow、没有训练好的模型时会退回关键词规则，并发出一句中文警告；
    测评报告开头要写明，不然会把关键词规则的结果当成模型的结果。
    """
    from pipeline import capture_warnings

    return capture_warnings(func, *args)


def _add_new(found: list[str], messages: list[str]) -> None:
    """把 messages 里还没有的提示加进 found（保持先后顺序）。"""
    for message in messages:
        if message not in found:
            found.append(message)


# ======================== 说话人标错的时长比例 ========================


def parse_num_speakers(value) -> int | str:
    """命令行 --speakers 的值 → eval_speakers 的 num_speakers。

    auto 或 自动（以及 0、负数）→ -1（工具自己判断人数）；ref 或 标注 → "ref"（每段录音按标注里的人数，即设对人数）；
    正整数（如 4）→ 这个人数。别的写法抛 ValueError（中文说明能写什么）。
    """
    text = str(value).strip()
    if text.lower() in ("auto", "自动"):
        return -1
    if text.lower() in (SPEAKERS_BY_ANNOTATION, "标注"):
        return SPEAKERS_BY_ANNOTATION
    try:
        number = int(text)
    except ValueError:
        raise ValueError(f"--speakers 的写法：auto（自动判断人数）、ref（每段录音按标注里的人数，即设对人数）"
                         f"或一个正整数（如 4）；现在写的是“{text}”") from None
    return number if number > 0 else -1


def describe_num_speakers(value) -> str:
    """说话人数设置的中文说明（命令行里打印）：-1 → 自动；"ref" → 按标注里的人数；4 → 4 人。"""
    if value == SPEAKERS_BY_ANNOTATION:
        return "每段录音按标注里的人数（设对人数）"
    value = int(value) if value is not None else -1
    return "自动（工具自己判断人数）" if value <= 0 else f"{value} 人"


def _speaker_rates(stats: dict) -> dict:
    """汇总：标错比例 = 标错的时长之和 ÷ 比对的时长之和。标错比例放在录音数后面（表格里排第一列数字）。"""
    rate = stats["error_seconds"] / stats["scored_seconds"] if stats["scored_seconds"] else 0.0
    return {"files": stats["files"], "speaker_error_rate": rate, **{k: v for k, v in stats.items() if k != "files"}}


def eval_speakers(root, cfg: dict, methods: dict | None = None, files: list[str] | None = None,
                  progress: Callable[[str], None] | None = None, use_cache: bool = True,
                  num_speakers: int | str | None = None) -> tuple[list[dict], dict]:
    """说话人标错的时长比例测评（第 3 组主指标）。返回 (每段录音一行, 汇总)。

    只评有说话人标注（annotations/speakers/<文件编号>.csv）的录音。methods、files、progress、use_cache 同 eval_cer。
    num_speakers：说话人数。不填就按 cfg["diarize"]["num_speakers"]（config.yaml，-1 表示自动）；
        填正整数就每段录音都按这个人数；填 -1 表示自动；填 "ref" 表示每段录音按它的标注里有几个人（设对人数）。
    流程：识别结果（和字错率共用缓存）→ diarize 做法用原始录音（没降噪的）给段落配说话人 → pipeline.metrics.speaker_error_details 和标注比。
    配成"未知"的段落不算工具分出的说话人。
    每行：stem、group、condition、speaker_error_rate、annotated_seconds（标注覆盖的秒数，重叠只算一次）、
    scored_seconds、error_seconds、ref_speakers（标注的人数）、hyp_speakers（分出的人数）、num_speakers（人数设置）、seconds。
    汇总同 eval_cer（全体、按录音条件、按组），标错比例 = 标错的时长之和 ÷ 比对的时长之和。
    """
    from pipeline.annotations import annotated_seconds
    from pipeline.audio import SR, read_wav
    from pipeline.metrics import speaker_error_details

    cfg = with_methods(cfg, methods)
    cfg["diarize"] = dict(cfg.get("diarize") or {})
    if num_speakers is not None:
        num_speakers = parse_num_speakers(num_speakers)  # 也接受 "auto"、"4"；写错时报中文说明
    by_annotation = num_speakers == SPEAKERS_BY_ANNOTATION
    if num_speakers is not None and not by_annotation:
        cfg["diarize"]["num_speakers"] = int(num_speakers) if int(num_speakers) > 0 else -1
    items = annotated_items(root, "speakers", files)
    hotword = _get_method(cfg, "hotword")
    diarize = _get_method(cfg, "diarize")

    def score(item: dict, segments: list[dict]) -> tuple[dict, str]:
        turns = item["annotation"]
        ref_count = len({speaker for _, _, speaker in turns})
        run_cfg = cfg
        if by_annotation:  # 设对人数：这段录音的标注里有几个人就设几个人
            run_cfg = copy.deepcopy(cfg)
            run_cfg["diarize"]["num_speakers"] = ref_count or -1
        setting = int(run_cfg["diarize"].get("num_speakers") or -1)
        # 和 run_pipeline 测评模式一样：text 里放识别结果（汉字读法），先做热词纠错，再分说话人（做法可能会看文字）
        segments = hotword([dict(seg, text=seg.get("text_raw") or "") for seg in segments], None, run_cfg)
        result = diarize(read_wav(item["wav"]), SR, segments, run_cfg)  # 用原始录音
        hyp = [(seg["start"], seg["end"], seg["speaker"]) for seg in result
               if seg.get("speaker") and seg["speaker"] != UNKNOWN_SPEAKER]
        details = speaker_error_details(turns, hyp)
        values = {
            "speaker_error_rate": details["speaker_error_rate"],
            "annotated_seconds": round(annotated_seconds(turns), 3),
            "scored_seconds": details["scored_seconds"],
            "error_seconds": details["error_seconds"],
            "ref_speakers": ref_count,
            "hyp_speakers": len({speaker for _, _, speaker in hyp}),
            "num_speakers": "自动" if setting <= 0 else str(setting),
        }
        text = (f"说话人标错比例 {values['speaker_error_rate']:.4f}（比对 {values['scored_seconds']:.0f} 秒；"
                f"标注 {values['ref_speakers']} 人，分出 {values['hyp_speakers']} 人）")
        return values, text

    rows = _run_items(root, cfg, items, progress, use_cache, score)
    summary = _summarize(rows, ["annotated_seconds", "scored_seconds", "error_seconds"], _speaker_rates)
    summary["info"] = _audio_info(root, cfg, SPEAKER_SLOTS, ("vad", "asr", "hotword", "diarize"))
    if by_annotation:
        summary["info"]["params"]["diarize.num_speakers"] = "按标注里的人数（设对人数，每段录音不同）"
    summary["tables"] = [{"title": "汇总（全体、按录音条件、按组）", "rows": summary["overview"]}]
    return rows, summary


# ======================== 话术分类（剧本文字） ========================


def _ratio_row(label: str, count: int, n: int) -> dict:
    """对比表、汇总表的一行：比例 = 分子 ÷ 分母（分母为 0 时记 0.0）。"""
    return {"分组项": label, "rate": count / n if n else 0.0, "count": count, "n": n}


def eval_classify_rules(cfg: dict, method: str = "baseline") -> dict:
    """在剧本 1055 句台词上测话术分类（第 6、7、8 组），不用录音。

    method：classify 槽位的做法名（baseline 关键词规则、tf_model、g6、g7……）。
    没装 TensorFlow 或没有训练好的模型时，g6、g7、tf_model 会退回关键词规则并发出提示，提示收进 warnings。
    返回：
        method、n（句数）、correct（分对的句数）、accuracy（7 类正确率）、labels（7 个类别名）、
        confusion（7×7 混淆矩阵，行是标准答案、列是预测）、per_class（各类 precision、recall、tp、n_pred、n_gold）、
        fp_rate、fp_count、normal_count（误报率：全部剧本）、fp_rate_g8、fp_count_g8、normal_count_g8（只看第 8 组剧本）、
        by_group（各组的句数、分对句数、正确率）、warnings（做法发出的提示）、predictions（每句台词一行）、
        overview（比例表：分组项、rate、count、n，对比表用）、tables、info（报告开头用）。
    误报率、准确率、召回率都用类别名算（pipeline.metrics）。做法返回的不是 7 个类别名
    （如返回了"疑似·费用"这样的显示标签）、或者句数不对时抛 ValueError。
    """
    from pipeline.data import FLAG_LABELS, LABEL_NAMES, load_lines
    from pipeline.metrics import NORMAL_LABEL, confusion_matrix, false_positive_rate, per_class_pr

    cfg = with_methods(cfg, {"classify": method})
    classify = _get_method(cfg, "classify")
    lines = load_lines()
    texts = [line["text"] for line in lines]
    pred, warnings_found = _capture_warnings(classify, texts, cfg)
    pred = list(pred)
    if len(pred) != len(texts):
        raise ValueError(f"话术分类做法“{method}”返回了 {len(pred)} 个类别，但台词有 {len(texts)} 句，两者必须一样多")
    unknown = sorted({str(p) for p in pred if p not in LABEL_NAMES})
    if unknown:
        raise ValueError(f"话术分类做法“{method}”返回了不认识的类别：{'、'.join(unknown)}。"
                         f"做法要返回类别名（{'、'.join(LABEL_NAMES)}），不是显示的标签（如“疑似·费用”）")
    gold = [line["label"] for line in lines]

    def fp_counts(indices: list[int]) -> tuple[int, int]:
        """(被预测成疑似类别的"正常讲解"句数, "正常讲解"句数)，只看 indices 这些句子。"""
        normal = [i for i in indices if gold[i] == NORMAL_LABEL]
        return sum(1 for i in normal if pred[i] in FLAG_LABELS), len(normal)

    everything = list(range(len(lines)))
    in_g8 = [i for i in everything if lines[i]["group"] == FP_GROUP]
    fp_count, normal_count = fp_counts(everything)
    fp_count_g8, normal_count_g8 = fp_counts(in_g8)
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    per_class = per_class_pr(gold, pred, LABEL_NAMES)
    by_group = []
    for group in sorted({line["group"] for line in lines}):
        indices = [i for i in everything if lines[i]["group"] == group]
        right = sum(1 for i in indices if gold[i] == pred[i])
        by_group.append({"group": group, "n": len(indices), "correct": right, "accuracy": right / len(indices)})

    predictions = []
    for line, p in zip(lines, pred):
        predictions.append({
            "script_id": line["script_id"], "group": line["group"], "line_no": line["line_no"],
            "speaker": line["speaker"], "text": line["text"], "label": line["label"], "predicted": p,
            "correct": "对" if p == line["label"] else "错",
            "false_positive": "是" if line["label"] == NORMAL_LABEL and p in FLAG_LABELS else "",
        })

    # 比例表：总体 3 行 → 各类准确率、召回率 → 各组正确率（对比表 compare.py 按这个顺序比）
    overview = [
        _ratio_row("7 类正确率", correct, len(lines)),
        _ratio_row("误报率（全部剧本）", fp_count, normal_count),
        _ratio_row(f"误报率（第 {FP_GROUP} 组剧本）", fp_count_g8, normal_count_g8),
    ]
    for label in LABEL_NAMES:
        item = per_class[label]
        overview.append(_ratio_row(f"{label}·准确率", item["tp"], item["n_pred"]))
        overview.append(_ratio_row(f"{label}·召回率", item["tp"], item["n_gold"]))
    overview += [_ratio_row(f"第 {item['group']} 组·7 类正确率", item["correct"], item["n"]) for item in by_group]

    matrix = confusion_matrix(gold, pred, LABEL_NAMES)
    class_rows = [{"类别": label, "标准答案句数": per_class[label]["n_gold"], "预测成该类的句数": per_class[label]["n_pred"],
                   "分对的句数": per_class[label]["tp"], "准确率": per_class[label]["precision"],
                   "召回率": per_class[label]["recall"]} for label in LABEL_NAMES]
    matrix_rows = [{"标准答案 \\ 预测": label, **dict(zip(LABEL_NAMES, row)), "合计": sum(row)}
                   for label, row in zip(LABEL_NAMES, matrix)]
    fp_rows = [{"剧本编号": row["script_id"], "行号": row["line_no"], "台词": row["text"], "预测": row["predicted"]}
               for row in predictions if row["false_positive"]]
    return {
        "method": method,
        "n": len(lines),
        "correct": correct,
        "accuracy": correct / len(lines) if lines else 0.0,
        "labels": list(LABEL_NAMES),
        "confusion": matrix,
        "per_class": per_class,
        "fp_rate": false_positive_rate(gold, pred),
        "fp_count": fp_count,
        "normal_count": normal_count,
        "fp_rate_g8": false_positive_rate([gold[i] for i in in_g8], [pred[i] for i in in_g8]),
        "fp_count_g8": fp_count_g8,
        "normal_count_g8": normal_count_g8,
        "by_group": by_group,
        "warnings": warnings_found,
        "predictions": predictions,
        "overview": overview,
        "tables": [
            {"title": "总体（比例 = 分子 ÷ 分母）", "rows": overview[:3]},
            {"title": "各类准确率与召回率", "rows": class_rows},
            {"title": "混淆矩阵（行是标准答案，列是预测，对角线上是分对的句数）", "rows": matrix_rows},
            {"title": "各组的 7 类正确率", "rows": overview[3 + 2 * len(LABEL_NAMES):]},
            {"title": "误报的句子（标准答案是正常讲解，被预测成疑似类别）", "rows": fp_rows},
        ],
        "rows_title": "每句台词的预测",
        "rows_in_markdown": False,  # 1055 行太长，只写进 CSV
        "info": {"pool": None, "pool_version": NO_POOL, "methods": {"classify": method},
                 "params": {"台词句数": len(lines)}},
    }


# ======================== 疑似片段起止误差 ========================


def clip_details(ref_clips, tool_clips) -> dict:
    """一段录音的片段起止误差：标注片段 [(开始, 结束, 类别名)] 和工具片段 [(开始, 结束, 类别名)] 比。

    先用 pipeline.metrics.clip_boundary_error 按重叠最多一一配对，得到起点、终点平均误差和没配上的个数；再数两样：
        tool_outside    和任何标注片段都不重叠的工具片段（标注人认为没有纠纷的地方被剪成了疑似片段）
        wrong_category  和它重叠最多的标注片段类别不一样的工具片段（时间对上了，类别标错了）
    返回 start_mae、end_mae（起点、终点平均误差，秒；一对也没配上时是 None）、ref_clips（标注片段数）、
    tool_clips（工具片段数）、matched（配上的对数）、unmatched_ref、unmatched_hyp、unmatched、tool_outside、wrong_category。
    """
    from pipeline.metrics import clip_boundary_error

    result = clip_boundary_error(ref_clips, tool_clips)
    outside = wrong = 0
    for start, end, category in tool_clips:
        best, best_overlap = None, 0.0
        for ref_start, ref_end, ref_category in ref_clips:
            overlap = min(float(end), float(ref_end)) - max(float(start), float(ref_start))
            if overlap > best_overlap:
                best, best_overlap = ref_category, overlap
        if best is None:
            outside += 1
        elif best != category:
            wrong += 1
    return {
        "start_mae": result["start_mae"],
        "end_mae": result["end_mae"],
        "ref_clips": result["n_ref"],
        "tool_clips": result["n_hyp"],
        "matched": result["n_matched"],
        "unmatched_ref": result["unmatched_ref"],
        "unmatched_hyp": result["unmatched_hyp"],
        "unmatched": result["unmatched"],
        "tool_outside": outside,
        "wrong_category": wrong,
    }


def _clip_rates(stats: dict) -> dict:
    """汇总：平均误差 = 各段录音误差之和 ÷ 配上的对数之和（一对也没配上时是 None）。"""
    start_total = stats.pop("start_total")
    end_total = stats.pop("end_total")
    matched = stats["matched"]
    return {"files": stats.pop("files"),  # 平均误差放在录音数后面（表格里排第一列数字）
            "start_mae": start_total / matched if matched else None,
            "end_mae": end_total / matched if matched else None,
            **stats}


def eval_clips(root, cfg: dict, methods: dict | None = None, files: list[str] | None = None,
               progress: Callable[[str], None] | None = None, use_cache: bool = True) -> tuple[list[dict], dict]:
    """疑似片段起止误差测评（第 8 组主指标）。返回 (每段录音一行, 汇总)。参数同 eval_cer。

    只评有片段标注（annotations/clips/<文件编号>.csv）的录音。流程和整理录音的测评模式一样：
    识别结果（和字错率共用缓存）→ 热词纠错 → 数字规范化（mode="spoken"）→ 话术分类 → clips 做法定起止
    （用原始录音；只算起止，不剪音频、不写文件）→ clip_details 和标注比。不做说话人分离（省时间）。
    工具片段的类别 = 做法返回的段落序号对应段落的 category。
    每行：stem、group、condition、clip_details 的各项、seconds。
    汇总同 eval_cer，平均误差 = 误差之和 ÷ 配上的对数之和；warnings 是话术分类做法发出的提示（如退回关键词规则）。
    """
    from pipeline.audio import SR, read_wav
    from pipeline.step6_classify import classify_segments

    cfg = with_methods(cfg, methods)
    items = annotated_items(root, "clips", files)
    hotword, normalize, classify, clips = (_get_method(cfg, slot) for slot in ("hotword", "normalize", "classify", "clips"))
    warnings_found: list[str] = []

    def score(item: dict, segments: list[dict]) -> tuple[dict, str]:
        samples = read_wav(item["wav"])  # 原始录音（片段从原始录音剪）
        # 和 run_pipeline 的测评模式一样：text 先设成 text_raw，再热词纠错、数字规范化、分类
        segments = [dict(seg, text=seg.get("text_raw") or "") for seg in segments]
        segments = hotword(segments, None, cfg)
        segments = [normalize(seg, "spoken", cfg) for seg in segments]
        segments, messages = _capture_warnings(classify_segments, segments, cfg, classify)
        _add_new(warnings_found, messages)
        for seg in segments:
            seg["source"] = "asr"
        tool = []
        for start, end, index in clips(segments, samples, SR, cfg):
            if not 0 <= index < len(segments):
                raise ValueError(f"片段做法返回的段落序号 {index} 超出范围（一共 {len(segments)} 段，序号从 0 开始）")
            tool.append((float(start), float(end), segments[index].get("category") or ""))
        details = clip_details(item["annotation"], tool)
        text = (f"标注片段 {details['ref_clips']} 个，工具片段 {details['tool_clips']} 个，"
                f"配上 {details['matched']} 对，和标注都不重叠的工具片段 {details['tool_outside']} 个")
        return details, text

    rows = _run_items(root, cfg, items, progress, use_cache, score)
    # 汇总平均误差要先把误差加起来：每段录音的误差之和 = 平均误差 × 配上的对数
    with_totals = [dict(row, start_total=(row["start_mae"] or 0.0) * row["matched"],
                        end_total=(row["end_mae"] or 0.0) * row["matched"]) for row in rows]
    summary = _summarize(with_totals, ["ref_clips", "tool_clips", "matched", "unmatched_ref", "unmatched_hyp",
                                       "unmatched", "tool_outside", "wrong_category", "start_total", "end_total"],
                         _clip_rates)
    summary["info"] = _audio_info(root, cfg, CLIP_SLOTS, ("vad", "asr", "hotword", "clips"))
    summary["warnings"] = warnings_found
    summary["tables"] = [{"title": "汇总（全体、按录音条件、按组）", "rows": summary["overview"]}]
    return rows, summary


# ======================== 对比 ========================


def compare_rows(metric: str, base_summary: dict, new_summary: dict) -> list[dict]:
    """把基线和改进两次测评的汇总排成对比表：每行 分组项、基线、改进、差值（= 改进 − 基线）。

    分组项的顺序和汇总一样：字错率、说话人、片段是 全体 → Q/N/F → 各组；数字提取是 主指标 → 次要指标 → 各类型；
    话术分类是 7 类正确率 → 误报率（全部剧本、第 8 组剧本）→ 各类准确率、召回率 → 各组正确率。
    没有值的格子（如一对片段也没配上时的平均误差）写空，差值也写空。
    """
    def rounded(value):
        return None if value is None else round(value, 4)

    values = COMPARE_VALUES[metric]
    new_rows = {row["分组项"]: row for row in new_summary["overview"]}
    rows = []
    for base_row in base_summary["overview"]:
        label = base_row["分组项"]
        new_row = new_rows.get(label)
        if new_row is None:
            continue
        for key, title in values:
            base, new = base_row[key], new_row[key]
            diff = None if base is None or new is None else new - base
            rows.append({"分组项": f"{label}·{title}" if len(values) > 1 else label,
                         "基线": rounded(base), "改进": rounded(new), "差值": rounded(diff)})
    return rows


# ======================== 写报告 ========================


def _columns(rows: list[dict]) -> list[str]:
    """表格的列：按各行字段第一次出现的顺序。"""
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def _cell(key: str, value) -> str:
    """表格里一格的写法：小数保留 4 位（时长 seconds、…_seconds 保留 1 位），没有值写空。"""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.1f}" if key == "seconds" or key.endswith("_seconds") else f"{value:.4f}"
    return str(value)


def _markdown_table(rows: list[dict]) -> list[str]:
    if not rows:
        return ["（没有数据）"]
    columns = _columns(rows)
    lines = ["| " + " | ".join(COLUMN_TITLES.get(c, c) for c in columns) + " |",
             "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(_cell(c, row.get(c)) for c in columns) + " |")
    return lines


def _header_lines(info: dict) -> list[str]:
    """报告开头：数据池版本、数据池、做法、关键参数、生成时间、局限说明。"""
    from pipeline.methods import SLOT_TITLES

    methods = "；".join(f"{SLOT_TITLES.get(slot, slot)} {slot}={name}"
                        for slot, name in (info.get("methods") or {}).items())
    params = "；".join(f"{key}={value}" for key, value in (info.get("params") or {}).items())
    lines = [f"- 数据池版本：{info.get('pool_version') or NOT_FROZEN}"]
    if info.get("pool"):
        lines.append(f"- 数据池：{info['pool']}")
    lines += [
        f"- 做法：{methods or '（无）'}",
        f"- 关键参数：{params or '（无）'}",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 局限：{FIXED_LIMITATION}",
    ]
    return lines


def write_report(out_dir, name: str, rows: list[dict], summary: dict, notes: list[str]) -> tuple[Path, Path]:
    """写测评报告：<out_dir>/<name>.csv（rows，UTF-8 带 BOM、中文表头）和 <out_dir>/<name>.md。

    Markdown 的内容依次是：标题；数据池版本、做法、关键参数、生成时间、局限说明（取自 summary["info"]）；
    运行时的提示（summary["warnings"]，如第 6、7 组的模型用不了、退回了关键词规则；没有就不写）；
    说明（notes，一条一行）；summary["tables"] 里的各张表（每张 {"title", "rows"}）；
    最后是 rows 这张表（标题用 summary["rows_title"]，没有就写"明细"）。
    summary["rows_in_markdown"] 为 False 时（如话术分类每句一行，太长）rows 只写进 CSV，Markdown 里只写一句去哪里看。
    返回 (CSV 路径, Markdown 路径)。文件正用 Excel 打开着时，Windows 会报 PermissionError。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    md_path = out_dir / f"{name}.md"

    columns = _columns(rows)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([COLUMN_TITLES.get(c, c) for c in columns])
        for row in rows:
            writer.writerow([_cell(c, row.get(c)) for c in columns])

    title = REPORT_TITLES.get(name) or REPORT_TITLES.get(name.removeprefix("compare_")) or name
    if name.startswith("compare_"):
        title = f"对比 · {title}"
    lines = [f"# 测评报告：{title}", ""]
    lines += _header_lines(summary.get("info") or {})
    warnings = summary.get("warnings") or []
    if warnings:
        lines += ["", "## 运行时的提示（请先看）", ""] + [f"- {message}" for message in warnings]
    if notes:
        lines += ["", "## 说明", ""] + [f"- {note}" for note in notes]
    for table in summary.get("tables") or []:
        lines += ["", f"## {table['title']}", ""] + _markdown_table(table["rows"])
    rows_title = summary.get("rows_title") or "明细"
    if summary.get("rows_in_markdown", True):
        lines += ["", f"## {rows_title}（和 {csv_path.name} 的内容相同）", ""] + _markdown_table(rows)
    else:
        lines += ["", f"## {rows_title}", "",
                  f"共 {len(rows)} 行，太长，没有放进本文件，见同一文件夹里的 {csv_path.name}（用 Excel 打开可以筛选）。"]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path
