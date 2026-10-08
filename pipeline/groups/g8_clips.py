"""第 8 组：疑似片段与误报率（槽位 clips）

你们组负责什么
    流水线第 7 步：按时间把疑似片段从原始录音里截出来，供处理人员点开听原声。
    你们的剧本是"正常讲解"对照组，所以还负责测量：正常讲解被误标成纠纷（误报）的比例。
    基线做法在 pipeline/step7_clips.py：每个有标签的段，前后各多留 1 秒（config.yaml 的 clips.padding）。

在哪里改
    只改这个文件。下面的函数现在直接调用基线；把函数体换成你们的改进做法即可，
    函数名、参数、返回值的格式不要改：返回 [(开始秒, 结束秒, 段落序号), ...]，段落序号从 0 开始；
    剪音频、写文件、写索引表由 export_clips 统一做，你们只决定"从哪剪到哪"。

怎么测（需要人工标注的片段起止，见 docs/guides/annotation.md）
    python tools/evaluate.py clips --pool 数据池路径 --out reports/g8                         （基线）
    python tools/evaluate.py clips --pool 数据池路径 --method clips=g8 --out reports/g8/after （你们的做法）
    python tools/compare.py --slot clips --method g8 --metric clips --pool 数据池路径 --out reports/g8
    python tools/evaluate.py classify --out reports/g8      （看第 8 组剧本的误报率）

注意
    - 片段一定从原始录音剪（不是降噪后的），复核时要听原声。
    - 误报率 = 标准答案为"正常讲解"的句子里，被标成任何一个"疑似"类别的比例；第 8 组剧本里也有标为费用、
      行程变更、购物安排的句子，它们不算误报。
    - 不联网。

可以试的方向（由易到难）
    1. 调前后留白，看片段起止误差怎么变。
    2. 把相邻的、标签相同的疑似段合并成一个片段（中间间隔小于几秒时）。
    3. 按静音调整边界：片段的起点和终点挪到最近的停顿处，不把半句话剪进来。
    4. 只在分类比较有把握时才出片段，减少误报（要和召回率一起看）。
"""
import numpy as np

from pipeline.methods import register
from pipeline.step7_clips import clips_baseline


@register("clips", "g8")
def clips_g8(segments: list[dict], samples: np.ndarray, sr: int, cfg: dict) -> list[tuple[float, float, int]]:
    """第 8 组的片段做法：返回 [(开始秒, 结束秒, 段落序号), ...]。"""
    # TODO 第 8 组：在这里写你们的改进。现在先直接用基线。
    return clips_baseline(segments, samples, sr, cfg)
