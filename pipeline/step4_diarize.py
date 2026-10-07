"""步骤 4：说话人分离。

说话人分离回答的是"谁在什么时候说话"。它分两件事：
  1. diarize_turns()：听整段录音，切成一个个"说话片段"，并把声音像的片段归为同一个人，
     返回 [(开始秒, 结束秒, 说话人编号)]，编号从 0 开始；
  2. attach_speakers()：步骤 2 切出来的每个段落，看它和哪个说话人的片段重叠的时间最长，
     就算这个人说的，写成"说话人1""说话人2"……（编号从 1 开始，给人看）。
分离出来的只是编号，不知道谁是导游、谁是游客。由界面上的人用 apply_speaker_map()
把"说话人1"对应成导游、游客等（可选的角色见 SPEAKER_ROLES）；
speaker_durations() 统计每人说了多久，可以提示"说话时间最长的可能是导游"，但不要自动定。

基线做法：
    sherpa-onnx 离线说话人分离：
      - pyannote 分割模型 3.0：找出每一小段时间里有几个人在说话、在哪里换人；
      - CAM++ 声纹模型：给每个说话片段算一个"声音特征"（声纹）；
      - 聚类：把声纹相近的片段归为同一个人。
    参数在 config.yaml 的 diarize 下面：
      num_speakers      说话人数。-1 表示自动；知道人数时一定要填
                        （实测 4 人的测试音频在"自动"下被分成了 5 人，填 4 后正确）；
      threshold         自动判断人数时用：越小越容易把人分开（人数变多），默认 0.5；
      min_duration_on   短于这么多秒的说话片段丢掉，默认 0.3；
      min_duration_off  同一个人两次说话之间的停顿短于这么多秒就连成一段，默认 0.5。
    录音短于 2 秒时不做分离，整段算一个人。
    模型给的说话人编号不一定连续，按第一次说话的先后重新编成 0、1、2……
    配说话人：按重叠时长最大配对；和所有说话片段都不重叠的段落写"未知"。
可改进方向：
    第 3 组（pipeline/groups/g3_diarize.py）：
      - 已知问题：一个段落里中途换了人（中间没停顿），整段只会算给一个人。
        可以按说话人片段的边界把段落再切开；
      - 换声纹模型（如 ERes2Net）、调聚类阈值，比较效果。
测评指标：
    说话人标错的时长比例（标错的秒数 ÷ 总的说话秒数），越小越好。
"""
import logging
import time

import numpy as np

from pipeline.audio import SR, has_non_ascii
from pipeline.methods import register
from pipeline.models import model_path

logger = logging.getLogger(__name__)

# 说话人映射时可以选的角色（界面下拉框里用）
SPEAKER_ROLES = ["导游", "游客", "店员", "司机", "经理", "未知"]

# 录音短于这么多秒就不做分离，整段算一个人（太短了，分不出几个人）
MIN_SECONDS = 2.0

# 已经做好的分离器：(分割模型, 声纹模型, 人数, 阈值, min_duration_on, min_duration_off) → 分离器。
# 加载模型要花时间，同样的设置只加载一次，以后直接拿来用。
_DIARIZERS: dict[tuple, object] = {}


def get_diarizer(cfg: dict, num_speakers: int, threshold: float):
    """取出（第一次用时先做好）一个 sherpa-onnx 离线说话人分离器。

    num_speakers：说话人数，-1 表示自动；threshold：自动判断人数时用的聚类阈值。
    人数和阈值不同，分离器也不同；同样的设置只做一次。
    """
    import sherpa_onnx

    segmentation = model_path(cfg, "pyannote")
    embedding = model_path(cfg, "campplus")
    for path in (segmentation, embedding):
        if not path.is_file():
            raise FileNotFoundError(f"找不到说话人分离模型：{path}。请先运行 python models/download_models.py 下载模型")

    params = cfg["diarize"]
    min_duration_on = float(params["min_duration_on"])
    min_duration_off = float(params["min_duration_off"])
    key = (str(segmentation), str(embedding), num_speakers, threshold, min_duration_on, min_duration_off)
    if key in _DIARIZERS:
        return _DIARIZERS[key]

    # 只在第一次加载模型时提醒一次，不用每次分离都重复说
    for path in (segmentation, embedding):
        if has_non_ascii(path):
            logger.warning("模型路径里有中文等非英文字符，Windows 上可能加载失败，建议把程序放到纯英文路径：%s", path)

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(segmentation)),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(embedding)),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers, threshold=threshold),
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
    )
    if not config.validate():
        raise ValueError("说话人分离的配置不对，请检查 config.yaml 里 diarize 下面的参数和模型文件")
    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    _DIARIZERS[key] = diarizer
    return diarizer


def _num_speakers(params: dict) -> int:
    """从配置里读出说话人数：没填、填 0 或负数都当作 -1（自动）。"""
    value = params.get("num_speakers")
    if value is None:
        return -1
    value = int(value)
    return value if value > 0 else -1


def diarize_turns(samples: np.ndarray, sr: int, cfg: dict) -> list[tuple[float, float, int]]:
    """说话人分离：返回 [(开始秒, 结束秒, 说话人编号)]，按开始时间排好，时间保留 2 位小数。

    编号从 0 开始、连续，按第一次说话的先后排（第一个开口的人是 0）。

    人数用 cfg["diarize"]["num_speakers"]（-1 表示自动）。
    录音短于 2 秒时不加载模型，直接返回 [(0, 时长, 0)]（整段算一个人）。
    """
    if sr != SR:
        raise ValueError(f"说话人分离需要 {SR} Hz 的录音，现在是 {sr} Hz，请先用步骤 1 转换格式")
    samples = np.asarray(samples, dtype=np.float32)
    duration = len(samples) / sr
    if duration < MIN_SECONDS:
        return [(0.0, round(duration, 2), 0)]

    params = cfg["diarize"]
    diarizer = get_diarizer(cfg, _num_speakers(params), float(params["threshold"]))
    result = diarizer.process(samples).sort_by_start_time()

    # 模型给的编号不一定连续（实测"自动"时出现过 0、1、2、5、7），
    # 这里按第一次说话的先后重新编号为 0、1、2……，第一个开口的人就是"说话人1"
    new_ids: dict[int, int] = {}
    turns = []
    for t in result:
        speaker = new_ids.setdefault(int(t.speaker), len(new_ids))
        turns.append((round(t.start, 2), round(t.end, 2), speaker))
    return turns


def attach_speakers(segments: list[dict], turns: list[tuple[float, float, int]]) -> list[dict]:
    """给每个段落配说话人：和哪个说话片段重叠的时间最长，就算谁说的。

    写入 speaker_id（从 1 开始）和 speaker="说话人N"；和所有片段都不重叠时写"未知"、speaker_id=None。
    返回新的段落列表，不改动传进来的段落。
    """
    result = []
    for seg in segments:
        best_speaker, best_overlap = None, 0.0
        for start, end, speaker in turns:
            overlap = min(seg["end"], end) - max(seg["start"], start)  # 两段时间重叠了多少秒
            if overlap > best_overlap:
                best_speaker, best_overlap = speaker, overlap
        new_seg = dict(seg)
        if best_speaker is None:
            new_seg["speaker"] = "未知"
            new_seg["speaker_id"] = None
        else:
            new_seg["speaker_id"] = int(best_speaker) + 1
            new_seg["speaker"] = f"说话人{new_seg['speaker_id']}"
        result.append(new_seg)
    return result


@register("diarize", "baseline")
def diarize_baseline(samples: np.ndarray, sr: int, segments: list[dict], cfg: dict) -> list[dict]:
    """基线说话人分离：先 diarize_turns 分出说话片段，再 attach_speakers 按重叠时长给每个段落配说话人。"""
    if not segments:
        return []  # 没有人声段落（例如全静音）：不用加载模型
    started = time.perf_counter()
    turns = diarize_turns(samples, sr, cfg)
    result = attach_speakers(segments, turns)

    seconds = time.perf_counter() - started
    duration = len(samples) / sr if sr else 0.0
    rtf = seconds / duration if duration else 0.0
    logger.info("步骤 4 完成：录音 %.1f 秒，分出 %d 个说话人，用时 %.2f 秒（实时率 %.3f）",
                duration, len({speaker for _, _, speaker in turns}), seconds, rtf)
    return result


def apply_speaker_map(segments: list[dict], mapping: dict[str, str]) -> list[dict]:
    """把"说话人1"等编号换成导游、游客等角色名，例如 mapping={"说话人1": "导游"}。

    只改 speaker，保留 speaker_id。只按段落现在显示的说话人（表格里"说话人"那一列）查 mapping，
    不看隐藏的编号 speaker_id——人在表格里手动改过的说话人，不能被编号悄悄改回去。
    映射过以后想改主意，就按现在显示的名字再映射一次，例如 {"导游": "游客"}。
    mapping 里角色为空的项不改。返回新的段落列表，不改动传进来的段落。
    """
    result = []
    for seg in segments:
        new_seg = dict(seg)
        role = mapping.get(seg.get("speaker", ""))
        if role and role.strip():
            new_seg["speaker"] = role.strip()
        result.append(new_seg)
    return result


def speaker_durations(segments: list[dict]) -> dict[str, float]:
    """每个说话人一共说了多少秒（保留 2 位小数），说话时间长的排在前面。"""
    totals: dict[str, float] = {}
    for seg in segments:
        speaker = seg.get("speaker", "未知")
        totals[speaker] = totals.get(speaker, 0.0) + max(0.0, seg["end"] - seg["start"])
    ordered = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return {speaker: round(seconds, 2) for speaker, seconds in ordered}
