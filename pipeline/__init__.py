"""旅游纠纷录音材料整理（教学原型）的处理流程。

对外接口：
- run_pipeline(path, options=None, progress=None, cfg=None) -> (segments, meta)：按顺序执行八步中的前七步
  （第 8 步"导出核查初稿"由界面或调用方在人工复核后用 step8_report.export_bundle 完成）。
- 各步骤的公开函数（见 pipeline/step1_ingest.py … step8_report.py）。
- 做法登记（见 pipeline/methods.py）：每组在 pipeline/groups/ 里给某个槽位加自己的做法。

注意：本文件导入时不能加载 sherpa-onnx、gradio、tensorflow 等重依赖，
这样只装了 TensorFlow 的电脑也能运行分类训练脚本。重依赖一律在函数内部导入。
"""

import contextlib
import threading

__version__ = "0.2.0"

NO_SPEECH_MESSAGE = "没有检测到人声，请检查录音"

# 写进 meta["models"] 的模型名称（写进核查初稿的"文件信息"）
MODEL_NAMES = {
    "asr": "sense-voice-int8-2024-07-17",
    "vad": "silero_vad",
    "diarization": "pyannote-3.0 + campplus",
}


# warnings.catch_warnings 改的是整个程序共用的设置：网页工具同时有几个人在用（几个线程）时，
# 一个人的提示可能串到另一个人的摘要里。凡是要收集或屏蔽警告的地方都先拿这把锁（同一线程里可以重复拿）。
_WARNINGS_LOCK = threading.RLock()


def capture_warnings(func, *args, **kwargs):
    """调用 func(*args, **kwargs)，收集它发出的提示，返回 (结果, 提示文字列表（去掉重复）)。

    只收集工具自己发的提示（UserWarning，例如"没有训练好的分类模型，话术分类改用关键词规则"），
    第三方库的 RuntimeWarning 之类不收。
    """
    import warnings

    with _WARNINGS_LOCK, warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = func(*args, **kwargs)
    messages = [str(record.message) for record in records if issubclass(record.category, UserWarning)]
    return result, list(dict.fromkeys(messages))


@contextlib.contextmanager
def quiet_warnings():
    """在 with 里面不显示任何警告（例如 cn2an 转不了"三四十"时的警告），和 capture_warnings 用同一把锁。"""
    import warnings

    with _WARNINGS_LOCK, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def ascii_name(name: str) -> str:
    """文件名里的中文、空格等换成 _，用作输出文件夹名（Windows 上路径最好只有英文）。"""
    safe = "".join(c if (c.isascii() and (c.isalnum() or c in "-_.")) else "_" for c in name)
    return safe.strip("._") or "audio"


def default_out_dir(cfg: dict, path) -> str:
    """默认输出文件夹：outputs/<日期-时间>_<文件名>，重名时加序号。"""
    from datetime import datetime
    from pathlib import Path

    stem = ascii_name(Path(path).stem)
    base = Path(cfg["paths"]["outputs"]) / f"{datetime.now():%Y%m%d-%H%M%S}_{stem}"
    out, n = base, 1
    while out.exists():
        n += 1
        out = base.with_name(f"{base.name}_{n}")
    return str(out)


def run_pipeline(path, options: dict | None = None, progress=None, cfg: dict | None = None):
    """处理一段录音或视频，返回 (段落列表, 元信息)。

    options（都可以不填）：
        num_speakers  说话人数，-1 表示自动（默认用 config.yaml），知道人数时一定要填
        hotword_fix   是否开热词纠错（默认用 config.yaml 的 hotword.enabled）
        hotwords      热词列表；None 表示用 data/hotwords.txt
        methods       各槽位用哪种做法，如 {"denoise": "g1"}，覆盖 config.yaml 的 methods
        asr_mode      "display"（默认，界面用）、"eval"（汉字读法，测评用）或 "both"
        out_dir       输出文件夹，默认 outputs/<日期-时间>_<文件名>
    progress：可选，progress(完成比例 0~1, 中文说明)，可以直接传 Gradio 的 gr.Progress()。

    返回的 meta 里有：file、duration、sha256、processed_at、models、options、methods、notice、
    timings（每步秒数）、rtf（总耗时 ÷ 录音时长）、work_dir、wav、qc、warnings、message（没有人声时的提示）。
    """
    import copy
    import logging
    import time
    from datetime import datetime
    from pathlib import Path

    from pipeline import methods as method_registry
    from pipeline.audio import SR, read_wav
    from pipeline.config import load_config
    from pipeline.schema import NOTICE
    from pipeline.step1_ingest import ingest
    from pipeline.step2_vad import detect_speech
    from pipeline.step3_asr import recognize
    from pipeline.step6_classify import classify_segments
    from pipeline.step7_clips import export_clips

    log = logging.getLogger("pipeline")
    opts = dict(options or {})
    cfg = copy.deepcopy(cfg or load_config())
    if opts.get("methods"):
        cfg["methods"] = {**cfg.get("methods", {}), **opts["methods"]}
    if opts.get("num_speakers") is not None:
        cfg["diarize"]["num_speakers"] = int(opts["num_speakers"])
    if opts.get("hotword_fix") is not None:
        cfg["hotword"]["enabled"] = bool(opts["hotword_fix"])
    asr_mode = opts.get("asr_mode", "display")
    funcs = method_registry.resolve(cfg)
    chosen = {slot: (cfg.get("methods") or {}).get(slot, "baseline") for slot in method_registry.SLOTS}

    out_dir = Path(opts.get("out_dir") or default_out_dir(cfg, path))
    timings: dict[str, float] = {}
    caught_warnings: list[str] = []

    def report(fraction: float, desc: str) -> None:
        if progress is not None:
            progress(min(max(fraction, 0.0), 1.0), desc)

    def timed(name: str, start: float, duration: float) -> None:
        used = time.time() - start
        timings[name] = round(used, 2)
        rtf = used / duration if duration else 0.0
        log.info("%s：用时 %.2f 秒，实时率 %.3f", name, used, rtf)

    t_all = time.time()

    # 1. 上传与格式统一
    report(0.0, "1/7 统一格式")
    t = time.time()
    info = ingest(path, out_dir, cfg)
    duration = info["duration"]
    timed("1 上传与格式统一", t, duration)
    samples = read_wav(info["wav"])  # 原始录音（转好格式），片段从这里剪

    meta = {
        "file": info["file"],
        "duration": duration,
        "sha256": info["sha256"],
        "processed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "models": dict(MODEL_NAMES),
        "options": {"num_speakers": cfg["diarize"]["num_speakers"], "hotword_fix": cfg["hotword"]["enabled"],
                    "asr_mode": asr_mode},
        "methods": chosen,
        "notice": NOTICE,
        "timings": timings,
        "work_dir": str(out_dir),
        "wav": info["wav"],
        "qc": info["qc"],
        "warnings": caught_warnings,
        "message": "",
    }

    # 2. 降噪与端点检测
    report(0.05, "2/7 降噪与端点检测")
    t = time.time()
    processed, segments = detect_speech(samples, SR, cfg, methods=funcs)
    timed("2 降噪与端点检测", t, duration)
    if not segments:
        meta["message"] = NO_SPEECH_MESSAGE
        meta["rtf"] = round((time.time() - t_all) / duration, 3) if duration else 0.0
        report(1.0, NO_SPEECH_MESSAGE)
        return [], meta

    # 3. 语音识别（用降噪/增强后的声音）+ 热词纠错
    t = time.time()
    segments = recognize(processed, SR, segments, cfg, mode=asr_mode,
                         progress=lambda f, d: report(0.15 + 0.5 * f, f"3/7 语音识别 {d}"))
    if asr_mode == "eval":
        for seg in segments:
            seg["text"] = seg.get("text_raw", "")
    segments = funcs["hotword"](segments, opts.get("hotwords"), cfg)
    timed("3 语音识别与热词", t, duration)

    # 4. 说话人分离（用原始录音）
    report(0.65, "4/7 说话人分离")
    t = time.time()
    segments = funcs["diarize"](samples, SR, segments, cfg)
    timed("4 说话人分离", t, duration)

    # 5. 数字规范化
    report(0.85, "5/7 数字规范化")
    t = time.time()
    mode = "spoken" if asr_mode == "eval" else "display"
    segments = [funcs["normalize"](seg, mode, cfg) for seg in segments]
    timed("5 数字规范化", t, duration)

    # 6. 话术分类（模型用不了时会退回规则，把原因记进 meta["warnings"]）
    report(0.9, "6/7 话术分类")
    t = time.time()
    segments, messages = capture_warnings(classify_segments, segments, cfg, method=funcs["classify"])
    for message in messages:
        if message not in caught_warnings:
            caught_warnings.append(message)
    timed("6 话术分类", t, duration)

    # 7. 疑似片段（从原始录音剪，复核时听原声）
    report(0.95, "7/7 疑似片段")
    t = time.time()
    for seg in segments:
        seg["source"] = "asr"
    segments = export_clips(segments, samples, SR, out_dir / "clips", cfg, method=funcs["clips"])
    timed("7 疑似片段", t, duration)

    total = time.time() - t_all
    meta["rtf"] = round(total / duration, 3) if duration else 0.0
    log.info("全部完成：%d 段，用时 %.1f 秒，录音 %.1f 秒，实时率 %.3f", len(segments), total, duration, meta["rtf"])
    report(1.0, "完成")
    return segments, meta
