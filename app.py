"""旅游纠纷录音材料整理（教学原型）· 网页界面。

启动：python app.py（便携包里双击 start.bat），浏览器打开 http://本机IP:7860。
     可选参数：--host、--port、--inbrowser（自动打开浏览器）、--config（换一份配置文件）。
本文件只是一层薄薄的“外壳”：真正的处理都在 pipeline/ 里，这里只负责把按钮、表格和处理函数连起来。

界面有三个标签页（每个标签页一个函数，以后加标签页就加一个函数，再在 build_app 里调用）：
    整理录音      _process_tab：上传 → 开始整理 → 核查表里复核、说话人映射、点行听原声 → 导出核查初稿
    剧本文本演示  _demo_tab：没有录音时直接用剧本台词演示步骤 5—8，并和剧本标注对照
    使用说明      _help_tab
处理函数（process_file、play_row、apply_mapping、export_files、run_demo 等）都写在模块里，
不启动网页也能直接调用和测试（见 tests/test_app.py）。

基线做法：
    Gradio 6 搭界面；主题、样式传给 launch()；同一时间每种操作只处理一个任务（队列）；
    只在本机和局域网提供服务，不开公网分享；关闭 Gradio 的使用统计（GRADIO_ANALYTICS_ENABLED=False）；
    上传的临时文件放在项目里的 tmp/（避开 Windows 中文用户名路径）；启动时清理超过 cleanup_hours 的旧结果；
    导出时把文件复制到 tmp/exports/<随机编号>/ 再给浏览器下载，不对网页开放 outputs 文件夹；
    设置了环境变量 DEMO_USERNAME 和 DEMO_PASSWORD 才要求登录（云端演示用）。
可改进方向：
    这是教师模板，学生一般不改。各组的改进做法写在 pipeline/groups/ 里，会自动出现在“高级设置”的下拉框中。
测评指标：
    不涉及识别效果；由 tests/test_app.py 检查各处理函数、导出内容和服务能否启动。

注意：gradio 在本文件开头导入（界面程序离不开它，处理函数的参数也要用到 gr.Progress、gr.SelectData）；
pipeline 里的模块仍然不导入 gradio。
"""
import argparse
import logging
import os
import re
import shutil
import stat
import sys
import time
import uuid
import warnings
from collections import Counter
from pathlib import Path

from pipeline.config import ROOT, load_config

# 导入 gradio 之前先设好：关闭使用统计；上传的临时文件放在项目里的 tmp/
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / "tmp"))

import gradio as gr

from pipeline import default_out_dir, run_pipeline, ui_text  # default_out_dir：输出文件夹的命名规则
from pipeline import methods as method_registry
from pipeline.audio import SR, has_non_ascii, read_wav, remove_tree
from pipeline.data import DISPLAY_NAMES, FLAG_LABELS, FLAG_OUTPUTS, load_hotwords, load_scripts
from pipeline.methods import SLOT_TITLES, SLOTS
from pipeline.schema import NOTICE
from pipeline.script_demo import run_script_demo
from pipeline.step1_ingest import SUPPORTED_EXTS
from pipeline.step4_diarize import apply_speaker_map, speaker_durations
from pipeline.step7_clips import export_clips
from pipeline.step8_report import CSV_NAME, DOCX_NAME, JSON_NAME, export_bundle, fmt_mmss
from pipeline.table import TABLE_HEADERS, rows_to_segments, segments_to_rows

logger = logging.getLogger("app")

# 说话人数下拉框的选项（“自动”＝ -1）
SPEAKER_CHOICES = ["自动", "2", "3", "4", "5"]
SWITCH_CHOICES = ["关", "开"]

# 核查表每一列的类型；序号、开始、结束、数字这 4 列不让改（改了也不会写回段落）
TABLE_TYPES = ["number", "number", "number", "str", "str", "str", "str", "str", "str"]
TABLE_STATIC_COLUMNS = [0, 1, 2, 6]
TABLE_WIDTHS = ["5%", "7%", "7%", "9%", "34%", "12%", "10%", "8%", "8%"]

# 界面字体：Windows 用微软雅黑，Mac 用苹方，Linux 用思源黑体
FONTS = ["Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "sans-serif"]
CSS = """
#banner { background: #fff4e5; border: 2px solid #d9480f; border-radius: 8px; padding: 10px 14px; }
#banner p { color: #a61e1e; font-weight: bold; margin: 0; }
"""

# build_app() 用的配置；各处理函数从这里取（没调用过 build_app 时读 config.yaml）
_current_cfg: dict | None = None


def _config() -> dict:
    global _current_cfg
    if _current_cfg is None:
        _current_cfg = load_config()
    return _current_cfg


# ---------------- 小工具：把界面上的选项变成程序用的值 ----------------

def is_on(value) -> bool:
    """开关的值：“开”（或 True）算打开，“关”算关闭。"""
    return value is True or value == "开"


def parse_num_speakers(value) -> int | None:
    """说话人数下拉框 → 整数：“自动”→ -1，“4”→ 4；没选（None）→ None（用 config.yaml 里的值）。"""
    if value is None or str(value).strip() == "":
        return None
    if str(value).strip() == "自动":
        return -1
    return int(value)


def parse_hotwords(text) -> list[str] | None:
    """热词文本框 → 热词列表（去重、保持顺序）。可以每行一个，也可以用顿号、逗号、分号、空格隔开。

    全部清空时返回 None，表示用 data/hotwords.txt。
    """
    words = []
    for word in re.split(r"[\s,，、;；]+", str(text or "")):
        if word and word not in words:
            words.append(word)
    return words or None


def parse_mapping(text) -> dict[str, str]:
    """说话人映射文本框 → 字典。每行一条“说话人1=导游”（全角＝也认）；没有等号或右边空着的行跳过。"""
    mapping = {}
    for line in str(text or "").splitlines():
        line = line.replace("＝", "=")
        if "=" not in line:
            continue
        old, new = (part.strip() for part in line.split("=", 1))
        if old and new:
            mapping[old] = new
    return mapping


def denoise_method(switch, chosen=None) -> str:
    """降噪开关 + 高级设置里“降噪”下拉框 → 实际用的降噪做法。

    开关只在 baseline（不降噪）和 noisereduce 之间切换；
    高级设置里选了 baseline 以外的做法（如 noisereduce、第 1 组的 g1）时，以高级设置为准。
    """
    if chosen and chosen != "baseline":
        return chosen
    return "noisereduce" if is_on(switch) else "baseline"


def new_state(segments: list[dict], meta: dict) -> dict:
    """界面“记住”的一次处理结果：段落列表和元信息。整理录音和剧本文本演示用同一种格式。"""
    return {"segments": segments, "meta": meta}


def _require_state(state) -> dict:
    if not state or "segments" not in state:
        raise gr.Error(ui_text.NEED_RESULT)
    return state


# ---------------- 摘要和说明文字 ----------------

def _label_counts(segments: list[dict]) -> str:
    """各类疑似片段的数量，例如“共 3 段（疑似·费用 2、疑似·购物安排 1）”。"""
    counts = Counter(seg.get("label") for seg in segments if seg.get("label"))
    detail = "、".join(f"{label} {counts[label]}" for label in FLAG_OUTPUTS if counts.get(label))
    text = f"共 {sum(counts.values())} 段"
    return f"{text}（{detail}）" if detail else text


def summary_markdown(segments: list[dict], meta: dict) -> str:
    """一次处理的摘要：时长、处理用时、段落和说话人数、各类疑似片段数、做法、质检和其他提示。"""
    lines = []
    if meta.get("demo_notice"):
        lines.append(f"**{meta['demo_notice']}**\n")
    if meta.get("message"):
        lines.append(f"**{meta['message']}**\n")

    duration = float(meta.get("duration") or 0)
    name = meta.get("file", "")
    if meta.get("title"):
        name = f"{name} {meta['title']}"
    lines.append(f"- 文件：{name}")
    estimated = "（估算）" if meta.get("demo_notice") else ""
    lines.append(f"- 时长{estimated}：{fmt_mmss(duration)}（{duration:.1f} 秒）")
    timings = meta.get("timings") or {}
    if timings:
        lines.append(f"- 处理用时：{sum(timings.values()):.1f} 秒（实时率 {meta.get('rtf', 0)}：处理用时 ÷ 录音时长）")
    speakers = {seg.get("speaker") for seg in segments}
    lines.append(f"- 段落：{len(segments)} 段；说话人：{len(speakers)} 个")
    lines.append(f"- 疑似片段：{_label_counts(segments)}")

    changed = [f"{SLOT_TITLES.get(slot, slot)} {method}" for slot, method in (meta.get("methods") or {}).items()
               if method != "baseline"]
    lines.append("- 做法：" + ("、".join(changed) if changed else "全部是基线做法"))

    if "qc" in meta:  # 剧本文本演示没有录音，不做质检
        problems = meta.get("qc") or []
        if problems:
            lines.append("- 录音质检提示（只提示，不影响处理）：")
            lines.extend(f"  - {problem}" for problem in problems)
        else:
            lines.append("- 录音质检：没有发现问题")
    for warning in meta.get("warnings") or []:
        lines.append(f"- 注意：{warning}")

    lines.append(f"\n> {NOTICE}")
    return "\n".join(lines)


def speaker_summary(state) -> str:
    """每个说话人说了多久，提示“说话时间最长的可能是导游，请人工确认”。"""
    if not state or not state.get("segments"):
        return ""
    lines = [f"**各说话人说话时长**（{ui_text.LONGEST_SPEAKER_HINT}）"]
    for speaker, seconds in speaker_durations(state["segments"]).items():
        lines.append(f"- {speaker}：{seconds:.1f} 秒")
    return "\n".join(lines)


def _cell(text) -> str:
    """放进 Markdown 表格的文字：去掉换行，竖线前加反斜杠。"""
    return str(text or "").replace("\n", " ").replace("|", "\\|")


def demo_contrast_markdown(segments: list[dict], stats: dict) -> str:
    """剧本文本演示：和剧本标注对照的统计，并列出误报、漏标、类别标错的台词。"""
    lines = [
        "### 和剧本标注对照",
        f"- 一共 {stats['total']} 句台词；被标疑似 {stats['flagged']} 句，其中类别和剧本一致 {stats['correct_flags']} 句",
        f"- 误报（剧本标“正常讲解”却被标疑似）：{stats['false_positives']} 句；"
        f"误报率 {stats['fp_rate']}（剧本里正常讲解共 {stats.get('normal_total', 0)} 句）",
        f"- 漏标（剧本是会标出的 5 类之一，工具却没标）：{stats['missed']} 句",
        "",
        ui_text.DEMO_CAVEAT,
    ]
    groups = {"误报": [], "漏标": [], "类别不一致": []}
    for n, seg in enumerate(segments, start=1):
        gold = seg.get("gold_label", "")
        if seg.get("label") and gold == "正常讲解":
            groups["误报"].append((n, seg))
        elif not seg.get("label") and gold in FLAG_LABELS:
            groups["漏标"].append((n, seg))
        elif seg.get("label") and gold in FLAG_LABELS and seg.get("category") != gold:
            groups["类别不一致"].append((n, seg))
    for title, items in groups.items():
        if not items:
            continue
        lines += ["", f"**{title}的台词**", "", "| 序号 | 说话人 | 文字 | 剧本类别 | 工具标签 |", "|---|---|---|---|---|"]
        for n, seg in items:
            gold = DISPLAY_NAMES.get(seg.get("gold_label", ""), seg.get("gold_label", ""))
            lines.append(f"| {n} | {_cell(seg.get('speaker'))} | {_cell(seg.get('text'))} | {_cell(gold)} "
                         f"| {_cell(seg.get('label')) or '（未标）'} |")
    return "\n".join(lines)


# ---------------- 处理函数（按钮、表格的事件） ----------------

def process_file(file_path, num_speakers, hotword_on, hotword_text, *method_names, progress=gr.Progress()):
    """“开始整理”：处理一个录音或视频，返回 (摘要 Markdown, 核查表的行, 界面要记住的结果)。

    method_names：高级设置里 8 个下拉框的值，顺序同 pipeline.methods.SLOTS；空着的槽位用 config.yaml。
    """
    if not file_path:
        raise gr.Error(ui_text.NEED_FILE)
    options = {
        "num_speakers": parse_num_speakers(num_speakers),
        "hotword_fix": is_on(hotword_on),
        "hotwords": parse_hotwords(hotword_text) if is_on(hotword_on) else None,
        "methods": {slot: name for slot, name in zip(SLOTS, method_names) if name},
    }
    try:
        segments, meta = run_pipeline(file_path, options, progress=progress, cfg=_config())
    except Exception as exc:  # 把错误原因用中文显示在网页上，详细信息打印在黑色窗口里
        logger.exception("处理失败：%s", file_path)
        raise gr.Error(f"处理失败：{exc}") from exc
    return summary_markdown(segments, meta), segments_to_rows(segments), new_state(segments, meta)


def start_processing(file_path, num_speakers, denoise_on, hotword_on, hotword_text, *method_names,
                     progress=gr.Progress()):
    """“开始整理”按钮实际调用的函数：先按降噪开关定好降噪做法，再交给 process_file。"""
    names = list(method_names) + [None] * (len(SLOTS) - len(method_names))
    denoise_index = SLOTS.index("denoise")
    names[denoise_index] = denoise_method(denoise_on, names[denoise_index])
    return process_file(file_path, num_speakers, hotword_on, hotword_text, *names, progress=progress)


def _selected_index(evt: gr.SelectData, count: int) -> int | None:
    """表格里点的是第几个段落（从 0 开始）。优先看这一行的“序号”（表格排过序也不会对错）。"""
    index = None
    if evt.row_value:
        try:
            index = int(float(str(evt.row_value[0]))) - 1
        except (TypeError, ValueError):
            index = None
    if index is None and isinstance(evt.index, (list, tuple)) and evt.index:
        index = int(evt.index[0])
    return index if index is not None and 0 <= index < count else None


def play_row(state, evt: gr.SelectData):
    """点核查表的某一行：返回这一段的原声 (采样率, 采样数组)；没有录音（如剧本文本演示）时返回 None。"""
    if not state or not (state.get("meta") or {}).get("wav"):
        return None
    segments = state.get("segments") or []
    index = _selected_index(evt, len(segments))
    if index is None:
        return None
    import soundfile as sf  # 只读这一段，长录音也很快

    segment = segments[index]
    first = max(0, int(round(segment["start"] * SR)))
    last = int(round(segment["end"] * SR))
    samples, sr = sf.read(state["meta"]["wav"], start=first, stop=last, dtype="int16")
    return sr, samples


def apply_mapping(state, rows, mapping_text):
    """“应用映射”：先收下表格里的人工修改，再把“说话人1”等换成导游、游客等，返回 (新的表格行, 新的结果)。"""
    state = _require_state(state)
    segments = rows_to_segments(rows if rows is not None else [], state["segments"])
    mapping = parse_mapping(mapping_text)
    if mapping:
        segments = apply_speaker_map(segments, mapping)
    return segments_to_rows(segments), new_state(segments, state["meta"])


def export_files(state, rows) -> list[str]:
    """“导出核查初稿”：用表格里现在的内容导出，返回 [ZIP, Word, CSV, JSON] 四个文件的路径。

    有录音时，按表格里改过的标签重新剪疑似片段（从步骤 1 转好的原始录音里剪，复核时听原声），
    片段做法和整理时用的一样。
    """
    state = _require_state(state)
    cfg = _config()
    meta = state["meta"]
    segments = rows_to_segments(rows if rows is not None else [], state["segments"])
    # 整理录音时 run_pipeline 已经建好了输出文件夹（work_dir）；
    # 剧本文本演示没有，用和整理录音一样的命名规则：outputs/<日期-时间>_<剧本编号>_demo
    out_dir = Path(meta.get("work_dir") or default_out_dir(cfg, f"{meta.get('file', '')}_demo"))

    clips_dir = None
    if meta.get("wav"):
        run_cfg = dict(cfg, methods={**(cfg.get("methods") or {}), **(meta.get("methods") or {})})
        clips_dir = out_dir / "clips"
        segments = export_clips(segments, read_wav(meta["wav"]), SR, clips_dir, run_cfg,
                                method=method_registry.resolve(run_cfg)["clips"])
    zip_path = export_bundle(segments, meta, out_dir, clips_dir=str(clips_dir) if clips_dir else None)
    files = [Path(zip_path), out_dir / DOCX_NAME, out_dir / CSV_NAME, out_dir / JSON_NAME]
    return serve_copies(files, cfg)


def serve_copies(files: list[Path], cfg: dict) -> list[str]:
    """把要下载的文件复制到 tmp/exports/<随机编号>/ 里再交给浏览器。

    原件留在 outputs 里给老师在本机找；浏览器只能拿到这一次导出的副本。
    这样网页不用开放整个 outputs 文件夹（里面还有别人上传的原始录音）。
    tmp 是 Gradio 的临时文件夹（启动时设成 GRADIO_TEMP_DIR），Gradio 允许从这里下载；随机编号猜不出来。
    """
    target = Path(cfg["paths"]["tmp"]) / "exports" / uuid.uuid4().hex
    target.mkdir(parents=True, exist_ok=True)
    copies = []
    for path in files:
        dest = target / path.name
        shutil.copyfile(path, dest)
        copies.append(str(dest))
    return copies


def run_demo_with_state(script_id):
    """剧本文本演示，返回 (摘要, 表格行, 对照说明, 界面要记住的结果)。script_id 可以是“G8-S1”或“G8-S1 标题”。"""
    if not script_id:
        raise gr.Error("请先选一个剧本")
    script_id = str(script_id).split()[0]
    if script_id not in {script["id"] for script in load_scripts()}:
        raise gr.Error(f"没有这个剧本编号：{script_id}（编号形如 G1-S1，从 G1-S1 到 G8-S3）")
    with warnings.catch_warnings(record=True) as records:  # 分类模型用不了时会退回规则，把原因显示出来
        warnings.simplefilter("always")
        try:
            segments, meta, stats = run_script_demo(script_id, cfg=_config())
        except Exception as exc:  # 和 process_file 一样：中文原因显示在网页上，详细信息打印在黑色窗口里
            logger.exception("剧本文本演示失败：%s", script_id)
            raise gr.Error(f"处理失败：{exc}") from exc
    meta["warnings"] = list(dict.fromkeys(str(record.message) for record in records))
    summary = summary_markdown(segments, meta)
    return summary, segments_to_rows(segments), demo_contrast_markdown(segments, stats), new_state(segments, meta)


def run_demo(script_id):
    """剧本文本演示，返回 (摘要, 表格行, 对照说明)。"""
    summary, rows, contrast, _ = run_demo_with_state(script_id)
    return summary, rows, contrast


def clear_old_results():
    """开始整理新录音前，清掉上一次的下载文件、选中段落的原声、说话人映射和各说话人时长，
    返回 (下载, 原声, 映射, 说话人时长)。

    核查初稿、CSV、JSON 每次导出的文件名都一样，不清掉的话，复核人员容易下载到上一段录音的核查初稿；
    上一段录音的“说话人1=导游”也不一定适用于新录音。
    """
    return None, None, "", ""


# ---------------- 搭界面 ----------------

def _table() -> gr.Dataframe:
    return gr.Dataframe(headers=TABLE_HEADERS, datatype=TABLE_TYPES, type="array", interactive=True,
                        static_columns=TABLE_STATIC_COLUMNS, column_widths=TABLE_WIDTHS, wrap=True,
                        label="核查表")


def _process_tab(cfg: dict) -> None:
    """标签页“整理录音”。"""
    with gr.Tab("整理录音"):
        state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(label=ui_text.UPLOAD_LABEL, file_types=sorted(SUPPORTED_EXTS), type="filepath")
                default_n = int(cfg["diarize"].get("num_speakers") or -1)
                speakers = gr.Dropdown(SPEAKER_CHOICES, value=str(default_n) if str(default_n) in SPEAKER_CHOICES
                                       else "自动", label="说话人数", info=ui_text.SPEAKERS_INFO)
                denoise_on = gr.Radio(SWITCH_CHOICES, label="降噪", info=ui_text.DENOISE_INFO,
                                      value="开" if cfg["methods"].get("denoise") == "noisereduce" else "关")
                hotword_on = gr.Radio(SWITCH_CHOICES, label="热词纠错", info=ui_text.HOTWORD_INFO,
                                      value="开" if cfg["hotword"].get("enabled") else "关")
                hotword_text = gr.Textbox("\n".join(load_hotwords()), lines=4, max_lines=8,
                                          label="热词表（可以改）", info=ui_text.HOTWORD_TEXT_INFO)
                with gr.Accordion(ui_text.ADVANCED_TITLE, open=False):
                    gr.Markdown(ui_text.ADVANCED_HINT)
                    method_boxes = [gr.Dropdown(method_registry.available(slot),
                                                value=cfg["methods"].get(slot, "baseline"),
                                                label=f"{SLOT_TITLES[slot]}（{slot}）") for slot in SLOTS]
                start = gr.Button("开始整理", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Markdown(ui_text.PROCESS_PLACEHOLDER)

        gr.Markdown(ui_text.TABLE_HINT)
        table = _table()
        audio = gr.Audio(label=ui_text.AUDIO_LABEL, type="numpy", interactive=False, autoplay=True,
                         buttons=["download"])  # 只留“下载”，不要“分享”按钮
        with gr.Row():
            with gr.Column():
                mapping = gr.Textbox(label=ui_text.MAPPING_LABEL, lines=3, info=ui_text.MAPPING_HINT,
                                     placeholder=ui_text.MAPPING_PLACEHOLDER)
                apply_btn = gr.Button("应用映射")
            speaker_md = gr.Markdown()
        gr.Markdown(ui_text.EXPORT_HINT)
        export_btn = gr.Button("导出核查初稿", variant="primary")
        files_out = gr.File(label="下载", file_count="multiple", interactive=False)

        # 点“开始整理”：先清掉上一段录音的下载文件、原声、映射和说话人时长，再整理，最后显示各说话人时长
        start.click(clear_old_results, None, [files_out, audio, mapping, speaker_md]).then(
            start_processing, [file_in, speakers, denoise_on, hotword_on, hotword_text, *method_boxes],
            [summary, table, state]).success(speaker_summary, state, speaker_md)
        table.select(play_row, state, audio)
        apply_btn.click(apply_mapping, [state, table, mapping], [table, state]).then(speaker_summary, state,
                                                                                     speaker_md)
        export_btn.click(export_files, [state, table], files_out)


def _demo_tab(cfg: dict) -> None:
    """标签页“剧本文本演示”。"""
    with gr.Tab("剧本文本演示"):
        state = gr.State(None)
        gr.Markdown(ui_text.DEMO_INTRO)  # “演示模式……”的提示在运行后的摘要里
        choices = [f"{script['id']} {script['title']}" for script in load_scripts()]
        with gr.Row():
            script = gr.Dropdown(choices, value=choices[0], label="选择剧本", scale=3)
            run_btn = gr.Button("运行演示", variant="primary", scale=1)
        summary = gr.Markdown()
        table = _table()
        contrast = gr.Markdown()
        export_btn = gr.Button("导出核查初稿")
        files_out = gr.File(label="下载", file_count="multiple", interactive=False)

        # 点“运行演示”：先清掉上一个剧本的下载文件，再运行
        run_btn.click(lambda: None, None, files_out).then(run_demo_with_state, script,
                                                          [summary, table, contrast, state])
        export_btn.click(export_files, [state, table], files_out)


def _help_tab() -> None:
    """标签页“使用说明”。"""
    with gr.Tab("使用说明"):
        gr.Markdown(ui_text.USAGE_MD)


def build_app(cfg: dict | None = None) -> gr.Blocks:
    """搭好整个网页界面并返回（不启动服务）。cfg 不填就读 config.yaml。"""
    global _current_cfg
    _current_cfg = cfg if cfg is not None else load_config()
    method_registry.load_all()  # 各组的做法登记进来，才会出现在“高级设置”里

    with gr.Blocks(title=ui_text.APP_TITLE, analytics_enabled=False) as demo:
        gr.Markdown(f"# {ui_text.APP_TITLE}")
        gr.Markdown(ui_text.BANNER, elem_id="banner")
        with gr.Tabs():
            _process_tab(_current_cfg)
            _demo_tab(_current_cfg)
            _help_tab()
    demo.queue(default_concurrency_limit=1)  # 同一时间每种操作只处理一个任务
    return demo


# ---------------- 启动 ----------------

def cleanup_old(folder, hours) -> int:
    """删除 folder 里（只看第一层）超过 hours 小时没改动的文件和文件夹，返回删了几个。folder 不存在时返回 0。"""
    folder = Path(folder)
    if not folder.is_dir():
        return 0
    limit = time.time() - float(hours) * 3600
    removed = 0
    for path in folder.iterdir():
        try:
            if path.stat().st_mtime >= limit:
                continue
            if path.is_dir():
                remove_tree(path)  # 原始录音是只读的，remove_tree 也能删
            else:
                os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
                path.unlink()
            removed += 1
        except OSError as exc:
            print(f"清理旧文件时跳过 {path}：{exc}")
    return removed


def auth_from_env() -> tuple[str, str] | None:
    """环境变量 DEMO_USERNAME 和 DEMO_PASSWORD 都设置了才要求登录（云端演示用），否则返回 None。"""
    user = os.environ.get("DEMO_USERNAME", "").strip()
    password = os.environ.get("DEMO_PASSWORD", "")
    return (user, password) if user and password else None


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=ui_text.APP_TITLE)
    parser.add_argument("--host", help="监听地址，默认用 config.yaml 的 app.host（0.0.0.0 表示局域网都能访问）")
    parser.add_argument("--port", type=int, help="端口，默认用 config.yaml 的 app.port（7860）")
    parser.add_argument("--inbrowser", action="store_true", help="启动后自动打开浏览器")
    parser.add_argument("--config", help="配置文件路径，默认用项目文件夹里的 config.yaml")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")  # 黑色窗口显示不了的字换成问号，不让程序因此出错
    except (AttributeError, ValueError):
        pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")  # 打印每一步的用时和实时率

    cfg = load_config(args.config)
    tmp_dir = Path(cfg["paths"]["tmp"])
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["GRADIO_TEMP_DIR"] = str(tmp_dir)
    if has_non_ascii(ROOT):
        print(ui_text.NON_ASCII_WARNING.format(path=ROOT))

    app_cfg = cfg["app"]
    for folder in (cfg["paths"]["outputs"], cfg["paths"]["tmp"]):
        removed = cleanup_old(folder, app_cfg["cleanup_hours"])
        if removed:
            print(f"已清理 {folder} 里超过 {app_cfg['cleanup_hours']} 小时的 {removed} 项旧内容")

    host = args.host or app_cfg["host"]
    port = args.port or int(app_cfg["port"])
    auth = auth_from_env()
    message = f"{ui_text.APP_TITLE}：本机浏览器打开 http://127.0.0.1:{port}"
    if host == "0.0.0.0":
        message += f"，局域网里其他电脑打开 http://本机IP:{port}"
    if auth:
        message += "（需要登录）"
    print(message)

    demo = build_app(cfg)
    # 下载文件由 serve_copies 放进 tmp（GRADIO_TEMP_DIR）里，所以不用 allowed_paths 开放 outputs 文件夹。
    # quiet=True：不打印 Gradio 自己的英文提示（其中有"怎么开公网分享"的提示，本课程不开公网分享），
    # 访问地址由上面的中文提示给出。
    demo.launch(server_name=host, server_port=port, max_file_size=app_cfg["max_file_size"],
                inbrowser=args.inbrowser, auth=auth, share=False,  # 不开公网分享
                quiet=True, show_error=True, theme=gr.themes.Soft(font=FONTS), css=CSS)


if __name__ == "__main__":
    main()
