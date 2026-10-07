"""旅游纠纷录音材料整理（教学原型）· 网页界面。

启动：python app.py（便携包里双击 start.bat），浏览器打开 http://本机IP:7860。
     可选参数：--host、--port、--inbrowser（自动打开浏览器）、--config（换一份配置文件）。
本文件只是一层薄薄的“外壳”：真正的处理都在 pipeline/ 里，这里只负责把按钮、表格和处理函数连起来。

界面有五个标签页（每个标签页一个函数，以后加标签页就加一个函数，再在 build_app 里调用）：
    整理录音      _process_tab：上传 → 开始整理 → 核查表里复核、说话人映射、点行听原声 → 导出核查初稿
    剧本文本演示  _demo_tab：没有录音时直接用剧本台词演示步骤 5—8，并和剧本标注对照
    数据校对      _proofread_tab：数据池里的录音 → 测评模式识别（有缓存）→ 和参考文本逐段对照、点行听原声
                  → 改参考文本、保存并记下校对人（做法见 docs/guides/proofreading.md）
    录音质检      _qc_tab：上传或从数据池选一段录音 → 波形、语谱图、MFCC 三张图 + 五项质检和中文解释
    使用说明      _help_tab
处理函数（process_file、play_row、apply_mapping、export_files、run_demo、generate_comparison、
save_reference、check_upload 等）都写在模块里，不启动网页也能直接调用和测试（见 tests/test_app.py）。

基线做法：
    Gradio 6 搭界面；主题、样式传给 launch()；同一时间每种操作只处理一个任务（队列）；
    只在本机和局域网提供服务，不开公网分享；关闭 Gradio 的使用统计（GRADIO_ANALYTICS_ENABLED=False）；
    上传的临时文件放在项目里的 tmp/（避开 Windows 中文用户名路径）；启动时清理超过 cleanup_hours 的旧结果；
    导出时把文件复制到 tmp/exports/<随机编号>/ 再给浏览器下载，不对网页开放 outputs 文件夹和数据池；
    数据池里的录音只在服务端读，播放时把采样数组交给浏览器，不给浏览器文件路径；
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
from pipeline import step2_vad, step3_asr  # 数据校对用：按“模块.函数”调用，测试时可以换成假的
from pipeline.align import align_segments, cer_details
from pipeline.audio import SR, convert_to_wav, has_non_ascii, probe, read_wav, remove_tree, sha256_file
from pipeline.data import (DISPLAY_NAMES, FLAG_LABELS, FLAG_OUTPUTS, load_hotwords, load_recording_plan,
                           load_scripts, script_reference_text)
from pipeline.evaluation import read_reference
from pipeline.features import plot_recording
from pipeline.methods import SLOT_TITLES, SLOTS
from pipeline.pool import log_proofread, parse_recording_name, pool_paths, read_csv_rows
from pipeline.schema import NOTICE, read_json, write_json
from pipeline.script_demo import run_script_demo
from pipeline.step1_ingest import SUPPORTED_EXTS, quality_check
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

# “数据校对”页的对照表：每一段的时间、识别结果、参考文本里对应的部分、是否不同（只给人看，不能改）
PROOFREAD_HEADERS = ["序号", "开始", "结束", "识别结果", "参考文本", "是否不同"]
PROOFREAD_TYPES = ["number", "number", "number", "str", "str", "str"]
PROOFREAD_WIDTHS = ["6%", "8%", "8%", "35%", "35%", "8%"]

# 校对人只收 1—12 个英文字母或数字（学号后四位），不收姓名
PROOFREADER_RE = re.compile(r"[A-Za-z0-9]{1,12}")

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


# ---------------- 数据池（“数据校对”和“录音质检”两页共用） ----------------

def pool_recordings(root) -> list[str]:
    """数据池清单（manifest.csv）里的录音编号，按编号排好；数据池不存在或还没有入池的录音时返回空列表。

    只看清单，不去翻 normalized/ 文件夹：没入池、入池失败的文件不会混进来（和测评工具一样）。
    """
    text = str(root or "").strip()
    if not text:
        return []
    rows = read_csv_rows(pool_paths(text)["manifest"])  # 文件不存在时是空列表
    return sorted({(row.get("文件编号") or "").strip() for row in rows} - {""})


def _pool_listing(root) -> tuple[list[str], str]:
    """数据池里的录音编号和一句中文提示。出了问题也只给提示、不报错（搭界面时也要调用它）。"""
    text = str(root or "").strip()
    if not text:
        return [], ui_text.POOL_NO_PATH
    if not Path(text).is_dir():
        return [], ui_text.POOL_MISSING.format(root=text)
    try:
        stems = pool_recordings(text)
    except (OSError, ValueError) as exc:  # 清单打不开、编码认不出来
        return [], f"数据池清单 manifest.csv 读不了：{exc}"
    if not stems:
        return [], ui_text.POOL_EMPTY.format(root=text)
    return stems, ui_text.POOL_READY.format(count=len(stems))


def refresh_recordings(root):
    """“刷新录音列表”：返回 (录音下拉框的新选项，默认选第一个；提示文字)。"""
    stems, hint = _pool_listing(root)
    return gr.update(choices=stems, value=stems[0] if stems else None), hint


def _pool_recording(root, stem) -> tuple[Path, str, dict, Path]:
    """检查数据池路径和选中的录音，返回 (数据池文件夹, 录音编号, 清单里这一行, 转换后的录音)。有问题时报中文错误。"""
    text = str(root or "").strip()
    if not text:
        raise gr.Error(ui_text.POOL_NO_PATH)
    if not Path(text).is_dir():
        raise gr.Error(ui_text.POOL_MISSING.format(root=text))
    if not stem or not str(stem).strip():
        raise gr.Error(ui_text.NEED_RECORDING)
    root, stem = Path(text), str(stem).strip()
    try:
        parse_recording_name(f"{stem}.wav")  # 编号必须是 G1-S1-Q 这样的写法
        rows = read_csv_rows(pool_paths(root)["manifest"])
    except (OSError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    row = next((row for row in rows if (row.get("文件编号") or "").strip() == stem), None)
    if row is None:
        raise gr.Error(f"录音 {stem} 不在数据池清单（manifest.csv）里。请先入池（python tools/ingest_pool.py），"
                       "或检查数据池路径后点“刷新录音列表”")
    # 入池时转换后的录音一律是 normalized/<编号>.wav；只读这个位置，不按表格里写的路径去读别处的文件
    wav = pool_paths(root)["normalized"] / f"{stem}.wav"
    if not wav.is_file():
        raise gr.Error(f"清单里有录音 {stem}，但找不到转换后的录音 {wav}。请检查数据池是否拷全了")
    return root, stem, row, wav


# ---------------- 数据校对 ----------------

def check_proofreader(value) -> str:
    """校对人编号：去掉前后空格，必须是 1—12 个英文字母或数字（学号后四位），否则报错。不收姓名。"""
    who = str(value or "").strip()
    if not PROOFREADER_RE.fullmatch(who):
        raise gr.Error(ui_text.PROOFREADER_ERROR)
    return who


def _recognize_for_proofreading(root: Path, stem: str, wav: Path, progress=None) -> list[dict]:
    """用测评模式识别一段池内录音（汉字读法、没有标点，和参考文本口径一致），返回带 text_raw 的段落。

    结果缓存在 数据池/asr_cache/<编号>.json，里面记着录音的 SHA-256 指纹和降噪、增强、端点检测用的做法；
    录音换了（复录）或做法换了才重新识别，否则直接用缓存，第二次“生成对照”几乎不用等。
    入池工具复录时会删掉 asr_cache/<编号>.*；测评工具的缓存叫 <编号>.eval-<钥匙>.json，和这里的互不影响。
    """
    cfg = _config()
    cache = pool_paths(root)["asr_cache"] / f"{stem}.json"
    fingerprint = sha256_file(wav)
    chosen = cfg.get("methods") or {}
    methods = {slot: chosen.get(slot) or "baseline" for slot in step2_vad.STEP_SLOTS}
    if cache.is_file():
        try:
            segments, meta = read_json(cache)
            if meta.get("wav_sha256") == fingerprint and meta.get("methods") == methods:
                return segments
        except (OSError, ValueError):
            pass  # 缓存文件坏了：重新识别

    if progress is not None:
        progress(0, "端点检测")
    processed, segments = step2_vad.detect_speech(read_wav(wav), SR, cfg)
    segments = step3_asr.recognize(processed, SR, segments, cfg, mode="eval", progress=progress)

    meta = {"stem": stem, "mode": "eval", "wav_sha256": fingerprint, "methods": methods,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = cache.with_name(cache.name + ".tmp")  # 先写临时文件再改名：写到一半出错不会留下半个缓存
    try:
        write_json(tmp, segments, meta)
        os.replace(tmp, cache)
    except OSError as exc:
        logger.warning("识别结果缓存写不进去（不影响这次校对，只是下次要重新识别）：%s", exc)
        if tmp.exists():
            tmp.unlink()
    return segments


def _pool_reference(root: Path, stem: str) -> tuple[str, bool]:
    """读这段录音的参考文本 references/<编号>.txt，返回 (文字, 文件是否存在)。

    文件还不存在时（老师还没运行 tools/export_references.py），先照抄剧本台词，保存时再写进文件。
    """
    path = pool_paths(root)["references"] / f"{stem}.txt"
    if path.is_file():
        return read_reference(path).replace("\r\n", "\n"), True  # 记事本存的换行是 \r\n
    return script_reference_text(parse_recording_name(f"{stem}.wav")["script_id"]), False


def proofread_rows(segments: list[dict], reference: str) -> list[list]:
    """对照表的行：[序号, 开始, 结束, 识别结果, 参考文本里对应的部分, "相同"/"不同"]。"""
    pieces = align_segments([seg.get("text_raw", "") for seg in segments], reference)
    return [[n, round(seg["start"], 2), round(seg["end"], 2), piece["hyp"], piece["ref"],
             "不同" if piece["diff"] else "相同"]
            for n, (seg, piece) in enumerate(zip(segments, pieces), start=1)]


def proofread_summary(stem: str, segments: list[dict], reference: str) -> str:
    """对照的摘要：整体字错率（拆成错字、漏字、多字）、有几段不同。"""
    details = cer_details(reference, "".join(seg.get("text_raw", "") for seg in segments))
    pieces = align_segments([seg.get("text_raw", "") for seg in segments], reference)
    different = sum(1 for piece in pieces if piece["diff"])
    lines = [
        f"### 录音 {stem}",
        f"- 整体字错率：**{details['cer']:.1%}**（参考文本 {details['n_ref']} 字：错字 {details['sub']}、"
        f"漏字 {details['dele']}、多字 {details['ins']}）",
        f"- 不同的段：{different}/{len(segments)}（重点听这些段）",
    ]
    if not segments:
        lines.append("- 没有检测到人声：请到“录音质检”页看看这段录音有没有声音")
    lines.append(f"\n{ui_text.PROOFREAD_CER_NOTE}")
    return "\n".join(lines)


def generate_comparison(root, stem, progress=gr.Progress()):
    """“生成对照”：返回 (摘要 Markdown, 对照表的行, 参考文本, 界面要记住的结果)。

    界面记住的结果和“整理录音”页同一种格式（段落 + 元信息，元信息里有 wav），所以点行播放直接用 play_row。
    """
    root, stem, _, wav = _pool_recording(root, stem)
    try:
        segments = _recognize_for_proofreading(root, stem, wav, progress)
    except Exception as exc:  # 模型没下载、录音读不了等：中文原因显示在网页上，详细信息打印在黑色窗口里
        logger.exception("数据校对识别失败：%s", stem)
        raise gr.Error(f"识别失败：{exc}") from exc
    reference, exists = _pool_reference(root, stem)
    summary = proofread_summary(stem, segments, reference)
    if not exists:
        summary = f"**{ui_text.REFERENCE_MISSING_NOTE}**\n\n{summary}"
    state = new_state(segments, {"root": str(root), "stem": stem, "wav": str(wav)})
    return summary, proofread_rows(segments, reference), reference, state


def clear_comparison():
    """点“生成对照”后、识别之前，清掉上一段录音的结果，返回 (摘要, 对照表, 参考文本, 记住的结果, 原声, 保存提示)。"""
    return ui_text.PROOFREAD_PLACEHOLDER, [], "", None, None, ""


def save_reference(state, text, proofreader, note="") -> str:
    """“保存参考文本”：写回 references/<编号>.txt（UTF-8），在 proofread_log.csv 里记下校对人和时间。返回提示文字。"""
    meta = (state or {}).get("meta") or {}
    if not meta.get("stem") or not meta.get("root"):
        raise gr.Error(ui_text.NEED_COMPARISON)
    who = check_proofreader(proofreader)
    text = str(text or "")
    if not text.strip():
        raise gr.Error("参考文本是空的，没有保存。请按录音里实际说的话填写（可以先点“生成对照”取回原来的参考文本）")

    root, stem = Path(meta["root"]), meta["stem"]
    path = pool_paths(root)["references"] / f"{stem}.txt"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise gr.Error(f"参考文本没保存上：{exc}。请确认数据池所在的 U 盘或共享文件夹还连着、可以写入") from exc
    try:
        log_proofread(root, stem, who, note or "")
    except (OSError, ValueError) as exc:  # 比如 proofread_log.csv 正用 Excel 打开着
        raise gr.Error(f"参考文本已保存，但校对记录没记上：{exc}。如果 proofread_log.csv 正用 Excel 打开着，"
                       "关掉后再点一次“保存参考文本”") from exc
    return (f"已保存 {stem} 的参考文本，并记下校对人 {who} 和时间。"
            "请再点“生成对照”看一遍：“不同”的行应该只剩识别错的。")


# ---------------- 录音质检 ----------------

def _expected_seconds(file_name: str) -> tuple[float | None, str]:
    """按文件名找录音计划里的预计时长（秒）。文件名不是 G1-S1-Q.m4a 这样的写法时返回 (None, 原因)。"""
    try:
        stem = parse_recording_name(file_name)["stem"]
    except ValueError as exc:
        return None, f"没有检查时长（{exc}）"
    for item in load_recording_plan():
        if Path(item["file_name"]).stem == stem:
            return round(item["expected_minutes"] * 60, 1), ""
    return None, f"没有检查时长（录音计划里没有 {stem}）"


def _qc_result(samples, original_sr: int, name: str, file_name: str):
    """画三张图、做五项质检，返回 (图, 质检结果 Markdown)。samples 是 16000 Hz 单声道的采样。"""
    expected, reason = _expected_seconds(file_name)
    problems = quality_check(samples, original_sr, _config(), expected)
    figure = plot_recording(samples, SR, title=name)

    seconds = len(samples) / SR
    lines = [f"### 质检结果：{name}",
             f"- 时长：{fmt_mmss(seconds)}（{seconds:.1f} 秒）；原始采样率：{original_sr or '读不出'} Hz"]
    if expected:
        lines.append(f"- 剧本预计时长：{fmt_mmss(expected)}")
    else:
        lines.append(f"- {reason}")
    if problems:
        lines.append("- **发现的问题**（只提示，不修改录音；每一项的意思和怎么办见本页下方）：")
        lines.extend(f"  - {problem}" for problem in problems)
    else:
        lines.append("- 质检：没有发现问题")
    return figure, "\n".join(lines)


def check_upload(file_path):
    """“检查上传的录音”：返回 (三张图, 质检结果 Markdown)。

    任何格式先用 ffmpeg 转成 16000 Hz 单声道 WAV，放在 tmp/qc/ 里，读完就删掉。
    文件名是 G1-S1-Q.m4a 这样的录音编号时，和录音计划里的预计时长比较。
    """
    if not file_path:
        raise gr.Error("请先上传一个录音或视频文件")
    src = Path(file_path)
    wav = Path(_config()["paths"]["tmp"]) / "qc" / f"{uuid.uuid4().hex}.wav"
    try:
        original_sr = probe(src)["sample_rate"]
        convert_to_wav(src, wav)
        samples = read_wav(wav)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("录音质检读不了文件 %s：%s", src.name, exc)
        raise gr.Error(f"读不了这个文件：{src.name}。文件可能已损坏，或者不是录音、视频文件，"
                       "请重新从手机拷贝后再试") from exc
    finally:
        if wav.exists():
            wav.unlink()
    return _qc_result(samples, original_sr, src.name, src.name)


def check_pool_recording(root, stem):
    """“检查池内录音”：返回 (三张图, 质检结果 Markdown)。原始采样率用清单里记的（转换后的录音都是 16000 Hz）。"""
    root, stem, row, wav = _pool_recording(root, stem)
    try:
        original_sr = int(float(row.get("原始采样率") or 0))
    except ValueError:
        original_sr = 0
    try:
        samples = read_wav(wav)
    except (OSError, RuntimeError, ValueError) as exc:
        raise gr.Error(f"读不了转换后的录音 {wav}：{exc}。文件可能已损坏，请老师检查数据池") from exc
    return _qc_result(samples, original_sr, stem, f"{stem}.wav")


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


def _pool_picker(cfg: dict) -> tuple[gr.Textbox, gr.Dropdown]:
    """数据池路径框、录音下拉框、“刷新录音列表”按钮和提示（“数据校对”“录音质检”两页各放一套），返回路径框和下拉框。

    打开网页时按 config.yaml 的 paths.data_pool 先列一次；数据池不存在或是空的，只显示中文提示。
    """
    stems, hint = _pool_listing(cfg["paths"]["data_pool"])
    pool = gr.Textbox(str(cfg["paths"]["data_pool"]), label=ui_text.POOL_PATH_LABEL, info=ui_text.POOL_PATH_INFO)
    with gr.Row():
        recording = gr.Dropdown(stems, value=stems[0] if stems else None, label=ui_text.RECORDING_LABEL, scale=3)
        refresh_btn = gr.Button("刷新录音列表", scale=1)
    hint_md = gr.Markdown(hint)
    refresh_btn.click(refresh_recordings, pool, [recording, hint_md])
    pool.submit(refresh_recordings, pool, [recording, hint_md])  # 在路径框里按回车也刷新
    return pool, recording


def _proofread_tab(cfg: dict) -> None:
    """标签页“数据校对”。"""
    with gr.Tab("数据校对"):
        state = gr.State(None)
        gr.Markdown(ui_text.PROOFREAD_INTRO)
        with gr.Row():
            with gr.Column(scale=3):
                pool, recording = _pool_picker(cfg)
            with gr.Column(scale=1):
                proofreader = gr.Textbox(label=ui_text.PROOFREADER_LABEL, info=ui_text.PROOFREADER_INFO,
                                         max_length=12)
                run_btn = gr.Button("生成对照", variant="primary")
        summary = gr.Markdown(ui_text.PROOFREAD_PLACEHOLDER)
        gr.Markdown(ui_text.PROOFREAD_TABLE_HINT)
        table = gr.Dataframe(headers=PROOFREAD_HEADERS, datatype=PROOFREAD_TYPES, type="array", interactive=False,
                             column_widths=PROOFREAD_WIDTHS, wrap=True, label="对照表")
        audio = gr.Audio(label=ui_text.AUDIO_LABEL, type="numpy", interactive=False, autoplay=True,
                         buttons=["download"])  # 只留“下载”，不要“分享”按钮
        reference = gr.Textbox(label=ui_text.REFERENCE_LABEL, info=ui_text.REFERENCE_INFO, lines=10, max_lines=30,
                               interactive=True)
        note = gr.Textbox(label=ui_text.PROOFREAD_NOTE_LABEL, info=ui_text.PROOFREAD_NOTE_INFO)
        save_btn = gr.Button("保存参考文本", variant="primary")
        save_msg = gr.Markdown()

        # 点“生成对照”：先清掉上一段录音的对照、参考文本、原声和保存提示，再识别（或用缓存）、对照。
        # 这样出错时页面上不会留着上一段录音的结果，也不会把它的参考文本错存到别的录音里
        run_btn.click(clear_comparison, None, [summary, table, reference, state, audio, save_msg]).then(
            generate_comparison, [pool, recording], [summary, table, reference, state])
        table.select(play_row, state, audio)
        save_btn.click(save_reference, [state, reference, proofreader, note], save_msg)


def _qc_tab(cfg: dict) -> None:
    """标签页“录音质检”。"""
    with gr.Tab("录音质检"):
        gr.Markdown(ui_text.QC_INTRO)
        with gr.Row():
            with gr.Column():
                file_in = gr.File(label=ui_text.QC_UPLOAD_LABEL, file_types=sorted(SUPPORTED_EXTS), type="filepath")
                upload_btn = gr.Button("检查上传的录音", variant="primary")
            with gr.Column():
                pool, recording = _pool_picker(cfg)
                pool_btn = gr.Button("检查池内录音", variant="primary")
        result = gr.Markdown()
        figure = gr.Plot(label=ui_text.QC_PLOT_LABEL, format="png")
        gr.Markdown(ui_text.QC_EXPLAIN)

        # 每次检查前先清掉上一次的图和结果：出错时不会让人误以为显示的是这一次的结果
        upload_btn.click(lambda: (None, ""), None, [figure, result]).then(check_upload, file_in, [figure, result])
        pool_btn.click(lambda: (None, ""), None, [figure, result]).then(check_pool_recording, [pool, recording],
                                                                         [figure, result])


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
            _proofread_tab(_current_cfg)
            _qc_tab(_current_cfg)
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
