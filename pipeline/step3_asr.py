"""步骤 3：语音识别（热词纠错在 pipeline/hotwords.py）。

输入：整段录音的采样（16000 Hz、单声道）和步骤 2 切好的段落（只有开始、结束时间）；
输出：同样的段落，填上识别出来的文字。

基线做法：
    用 sherpa-onnx 运行 SenseVoice 识别模型（int8，2024-07-17 版，不要换成 2025-09-09 版，那是粤语模型）。
    两种识别模式：
      - 显示模式（display，界面默认）：use_itn=True，带标点、数字写成阿拉伯数字，写进 text；
      - 测评模式（eval）：use_itn=False，汉字读法、没有标点，写进 text_raw，和参考文本口径一致，用来算字错率；
      - 两种都要（both）：各识别一遍，时间翻倍。
    模型只加载一次：两种模式各一个识别器，放在模块里的字典 _RECOGNIZERS 里反复用。
    段落按 config.yaml 的 asr.batch_size（默认 20）分批识别，每识别完一批更新一次进度条。
可改进方向：
    热词：SenseVoice 不支持解码时热词，基线在识别后按拼音纠错（第 5 组，见 pipeline/hotwords.py）；
    对照别的识别模型（如 Paraformer 小模型 + 标点模型）；调线程数和每批段数，看速度变化。
测评指标：
    字错率（用测评模式的 text_raw 和参考文本比）；实时率（识别耗时 ÷ 录音时长）。
    用 tools/evaluate.py cer 计算。
"""
import numpy as np

# 导入热词模块：它里面的 @register 会把"热词纠错"的基线做法登记进来
# （methods.load_all() 只导入各步骤模块，不单独导入 pipeline.hotwords）
import pipeline.hotwords  # noqa: F401
from pipeline.models import model_path

# 识别模式 → [(写进哪个字段, 是否 use_itn, 中文名)]
MODES = {
    "display": [("text", True, "显示模式")],
    "eval": [("text_raw", False, "测评模式")],
    "both": [("text", True, "显示模式"), ("text_raw", False, "测评模式")],
}

# 已经加载好的识别器（模块级缓存）：键是 (模型文件, 线程数, 语言, use_itn)
_RECOGNIZERS: dict[tuple, object] = {}


def get_recognizer(cfg: dict, use_itn: bool):
    """取一个 SenseVoice 识别器：第一次调用时加载模型，以后同样的设置直接用缓存里的。

    use_itn=True 是显示模式（阿拉伯数字、带标点），False 是测评模式（汉字读法、没有标点），各缓存一个。
    模型文件不存在时抛 FileNotFoundError，提示先下载模型。
    """
    asr = cfg.get("asr") or {}
    model = model_path(cfg, "sense_voice_model")
    tokens = model_path(cfg, "sense_voice_tokens")
    num_threads = int(asr.get("num_threads", 4))
    language = asr.get("language", "zh")
    key = (str(model), num_threads, language, bool(use_itn))
    if key in _RECOGNIZERS:
        return _RECOGNIZERS[key]

    if not model.is_file() or not tokens.is_file():
        raise FileNotFoundError(
            f"找不到识别模型：{model}。请先运行 python models/download_models.py 下载模型，"
            f"或者把下载好的模型文件夹拷进 models/"
        )
    import sherpa_onnx  # 重依赖，用到时才导入

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model),
        tokens=str(tokens),
        num_threads=num_threads,
        language=language,
        use_itn=bool(use_itn),
    )
    _RECOGNIZERS[key] = recognizer
    return recognizer


def _decode_batch(recognizer, samples: np.ndarray, sr: int, batch: list[dict]) -> list[str]:
    """识别一批段落：按每段的开始、结束时间从整段录音里切出声音，一起解码，返回每段的文字。"""
    streams = []
    for seg in batch:
        begin = max(0, int(round(seg["start"] * sr)))
        end = min(len(samples), int(round(seg["end"] * sr)))
        stream = recognizer.create_stream()
        stream.accept_waveform(sr, samples[begin:end])
        streams.append(stream)
    recognizer.decode_streams(streams)  # 一批一起解码，比一段一段解码快
    return [stream.result.text.strip() for stream in streams]


def recognize(samples, sr: int, segments: list[dict], cfg: dict, mode: str = "display",
              progress=None) -> list[dict]:
    """识别每个段落的文字，返回新的段落列表（不改调用方传进来的段落）。

    mode：
        "display"  显示模式，写 text（use_itn=True，阿拉伯数字、带标点）
        "eval"     测评模式，写 text_raw（use_itn=False，汉字读法、没有标点）
        "both"     两个都写（各识别一遍）
    progress：可以不传；传了的话每识别完一批调用一次 progress(完成比例 0~1, 中文说明)，
        可以直接传 Gradio 的 gr.Progress()。
    段落列表为空时直接返回空列表，不加载模型。
    """
    if mode not in MODES:
        raise ValueError(f"不认识的识别模式“{mode}”，只能是 display（显示模式）、eval（测评模式）或 both（两种都要）")
    if not segments:
        return []

    samples = np.asarray(samples, dtype=np.float32)
    batch_size = max(1, int((cfg.get("asr") or {}).get("batch_size", 20)))
    results = [dict(seg) for seg in segments]  # 复制一份再写文字
    batch_starts = list(range(0, len(results), batch_size))  # 每批第一段的序号
    total = len(MODES[mode]) * len(batch_starts)  # 一共要识别几批
    done = 0

    for field, use_itn, title in MODES[mode]:
        recognizer = get_recognizer(cfg, use_itn)
        for batch_no, first in enumerate(batch_starts, start=1):
            batch = results[first:first + batch_size]
            texts = _decode_batch(recognizer, samples, sr, batch)
            for seg, text in zip(batch, texts):
                seg[field] = text
            done += 1
            if progress is not None:
                progress(done / total, f"语音识别（{title}）：第 {batch_no}/{len(batch_starts)} 批")
    return results
