"""步骤 7：疑似片段导出（第 8 组）。

把被标成"疑似·…"的段落从录音里剪出来，存成一个个小 WAV 文件，方便工作人员直接点开听原声。
另外写一张索引表 clips_index.csv（序号、开始、结束、说话人、文字、标签、文件），
并在对应段落里写上 clip（片段文件名）。打包成 ZIP 在步骤 8 的 export_bundle 里做。

槽位 clips 的做法：函数 (段落列表, 采样数组, 采样率, 配置) → [(开始秒, 结束秒, 段落序号), ...]
    段落序号从 0 开始，就是这个段落在段落列表里的位置。做法只决定"剪哪几段、从哪里剪到哪里"，
    真正剪音频、写文件由 export_clips 统一做。

基线做法：
    1. 只看标签（label）不为空的段落；
    2. 每段前后各多留 padding 秒（config.yaml 里 clips.padding，默认 1 秒），让人听到上下文；
    3. 不超出录音的开头（0 秒）和结尾（录音时长）；
    4. 一个段落出一个片段，相邻的片段不合并。
    剪的时候按采样点切（开始秒 × 16000 = 第几个采样点），起止时间精确到 1/16000 秒。
可改进方向（第 8 组，pipeline/groups/g8_clips.py）：
    - 相邻或重叠的疑似片段合并成一个，免得同一段话被剪成好几个文件；
    - 按说话人或停顿调整起止点，不把别人的话、半个字剪进来；
    - 和第 6、7 组一起统计误报：正常讲解被标成疑似的有多少。
测评指标：
    片段起止误差：和人工标注的疑似片段起止时间比，开始、结束各差多少秒；
    误报率：标准答案为"正常讲解"的句子中被标成任一疑似类别的比例（句子层面，tools/evaluate.py classify）；
    片段层面看 tools/evaluate.py clips 的"和标注都不重叠的工具片段"。
"""
import csv
import logging
from pathlib import Path
from typing import Callable

import numpy as np

from pipeline.audio import write_wav
from pipeline.methods import register

logger = logging.getLogger(__name__)

# 片段索引表的文件名和表头
CLIPS_INDEX_NAME = "clips_index.csv"
CLIPS_INDEX_HEADERS = ["序号", "开始", "结束", "说话人", "文字", "标签", "文件"]

# 配置里没写 clips.padding 时，前后各留几秒
DEFAULT_PADDING = 1.0


def clip_filename(n: int, start: float, end: float) -> str:
    """片段文件名，只用英文字母和数字（Windows 解压不乱码）。

    例：clip_filename(3, 12.4, 18.9) → "clip_003_000012.4-000018.9.wav"
    （第 3 个片段，从第 12.4 秒到第 18.9 秒；时间补零到 8 位，文件按名字排序就是按时间排序）
    """
    return f"clip_{n:03d}_{start:08.1f}-{end:08.1f}.wav"


@register("clips", "baseline")
def clips_baseline(segments: list[dict], samples: np.ndarray, sr: int, cfg: dict) -> list[tuple[float, float, int]]:
    """基线做法：有标签的段落前后各留 padding 秒，裁到 [0, 录音时长] 之内。"""
    padding = float((cfg.get("clips") or {}).get("padding", DEFAULT_PADDING))
    duration = len(samples) / sr
    clips = []
    for index, segment in enumerate(segments):
        if not segment.get("label"):
            continue  # 没有标签（正常讲解、其他）不出片段
        start = max(0.0, segment["start"] - padding)
        end = min(duration, segment["end"] + padding)
        if end > start:
            clips.append((round(start, 3), round(end, 3), index))
    return clips


def _remove_old_clips(out_dir: Path) -> None:
    """删掉文件夹里上一次导出的片段和索引表（人工改了标签后再导出，旧片段不能混进来）。"""
    for path in out_dir.glob("clip_*.wav"):
        path.unlink()
    index_path = out_dir / CLIPS_INDEX_NAME
    if index_path.exists():
        index_path.unlink()


def export_clips(segments: list[dict], samples: np.ndarray, sr: int, out_dir, cfg: dict,
                 method: Callable | None = None) -> list[dict]:
    """按做法 method 定好的起止时间剪出疑似片段，写到 out_dir，返回写上 clip 字段的新段落列表。

    method：片段做法 (段落列表, 采样数组, 采样率, 配置) → [(开始, 结束, 段落序号)]；不填就用基线。
            整条流程里由 pipeline.methods.resolve(cfg)["clips"] 按配置选好再传进来。
    samples：要剪的录音（应该用步骤 1 转好的原始录音，不要用降噪后的，复核时要听原声）。
    out_dir 里上一次导出的 clip_*.wav 和 clips_index.csv 会先被删掉。传进来的段落不会被修改。
    """
    method = method or clips_baseline
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _remove_old_clips(out_dir)

    # 复制段落，并去掉上一次导出留下的 clip 字段
    result = []
    for segment in segments:
        segment = dict(segment)
        segment.pop("clip", None)
        result.append(segment)

    index_rows = []
    for start, end, index in method(result, samples, sr, cfg):
        if not 0 <= index < len(result):
            raise ValueError(f"片段做法返回的段落序号 {index} 超出范围（一共 {len(result)} 段，序号从 0 开始）")
        first = max(0, int(round(start * sr)))  # 开始秒 → 第几个采样点
        last = min(len(samples), int(round(end * sr)))
        if last <= first:
            logger.warning("片段 (%.2f, %.2f) 没有长度，跳过", start, end)
            continue
        n = len(index_rows) + 1
        name = clip_filename(n, start, end)
        write_wav(out_dir / name, samples[first:last], sr)

        segment = result[index]
        # 同一个段落被剪了不止一个片段时（某些改进做法可能这样），文件名用"；"连起来
        segment["clip"] = f"{segment['clip']}；{name}" if segment.get("clip") else name
        index_rows.append([n, round(start, 2), round(end, 2), segment.get("speaker", ""),
                           segment.get("text", ""), segment.get("label", ""), name])

    # 索引表：UTF-8 带 BOM，Excel 双击打开不乱码
    with open(out_dir / CLIPS_INDEX_NAME, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CLIPS_INDEX_HEADERS)
        writer.writerows(index_rows)

    logger.info("步骤 7 完成：导出 %d 个疑似片段到 %s", len(index_rows), out_dir)
    return result
