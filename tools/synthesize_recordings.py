"""按剧本合成 72 段 AI 测评录音和标准答案（老师或助教运行一次；见 docs/adr/0006-ai-synthesized-recordings.md）。

用法：
    python models/download_models.py --optional kokoro_tts        先下载语音合成模型（约 365 MB）
    python tools/synthesize_recordings.py                         合成全部 72 段，放在 ai_recordings/
    python tools/synthesize_recordings.py --scripts G1-S1 --conditions Q    只合成一段（先试听）
    python tools/synthesize_recordings.py --make-voice-plan       重新给每个角色挑声音（改写 data/tts_voice_plan.csv）
    python tools/synthesize_recordings.py --digits                选做数字实验的录音（100 个声音 × 零到九 × 3 遍）

输出（--out，默认项目里的 ai_recordings/，不进 git）：
    README.txt                                 说明：AI 合成、内容虚构、怎么用
    recordings/G1-S1-Q-TTS.m4a ……              录音：开头念一句 AI 合成声明，文件名带 -TTS，文件信息里写明 AI 合成
    answer_key/references/G1-S1-Q.txt          参考文本（第一行是声明）
    answer_key/annotations/speakers/G1-S1-Q.csv 说话人时间（开头的声明算一个说话人"声明"）
    answer_key/annotations/clips/G1-S1-Q.csv   疑似片段起止（连续几句同一个疑似类别的台词连成一个片段）
    answer_key/timeline/G1-S1-Q.csv            每句台词的起止、说话人、类别、声音编号、舞台说明
    answer_key/summary.csv                     每段录音一行：时长、F 条件模拟的是口袋还是远距离、句数……
已经合成好的录音默认跳过（中断后再运行会接着做）；加 --overwrite 全部重做。

全部在本机运行：离线语音合成（sherpa-onnx + Kokoro），不联网、不调用任何云服务，不做声音克隆。
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONDITION_CODES = ("Q", "N", "F")
F_VARIANT_NAMES = {"pocket": "口袋（模拟）", "far": "2—3 米外（模拟）", "": ""}
TIMELINE_COLUMNS = ["index", "speaker", "start", "end", "label", "sid", "direction", "text"]
SUMMARY_COLUMNS = ["file", "script_id", "condition", "f_variant", "duration_s", "expected_s", "lines", "speakers",
                   "clips", "voices"]

# 写进每个录音文件信息（标题、注释、作者）的文字
META_TITLE = "AI合成·虚构教学录音 {stem}"
META_COMMENT = ("由人工智能技术（Kokoro v1.1 离线语音合成）按虚构剧本自动生成，不是真实录音；"
                "人物、单位、电话全部虚构，仅用于《语音识别技术及应用》实训课教学，不得剪掉开头的声明或当作真实录音使用。")
META_ARTIST = "AI合成（非真人录音）"

README = """旅游纠纷录音材料整理 · AI 合成测评录音（虚构教学材料）

这里的录音全部由人工智能技术（Kokoro v1.1 离线语音合成，sherpa-onnx）按仓库 data/scripts 里的虚构剧本
自动生成，不是真实录音，也没有用任何真人的声音做参考（不是声音克隆）。
每段录音开头都念一句声明："{disclaimer}"文件名带 -TTS，文件信息（标题、注释）里也写明 AI 合成。
剧本里的人物、旅行社、店铺、电话号码全部虚构。只用于《语音识别技术及应用》实训课；
不得剪掉开头的声明，不得当作真实录音使用或传播。

文件名：G组号-S剧本号-条件-TTS.m4a。条件：Q 安静；N 嘈杂教室（叠加别的剧本的合成人声，模拟）；
F 口袋或远距离（组号 + 剧本号是单数的模拟"手机在口袋里"，双数的模拟"手机在 2—3 米外"）。
台词的舞台说明（重叠、打断、停顿、扩音器、电话外放、离手机远、提示音、发动机声……）已经做成了声音效果。

怎么用：把 recordings 文件夹里的录音拷进数据池的 raw 文件夹，运行 tools/ingest_pool.py 入池，
再运行 tools/export_references.py 生成参考文本初稿（见 docs/teacher/data_pool.md）。

测评提醒：合成语音比真人干净、发音标准，嘈杂和远距离是模拟出来的，测出来的字错率、说话人分离结果会偏乐观，
只能当作方法练习，不代表工具在真实录音上的效果；报告里要写明"测评数据是 AI 合成的"。
"""

ANSWER_README = """标准答案（合成时同时得到的，给老师检查同学的校对和标注用）

references/G1-S1-Q.txt            参考文本：第一行是开头的声明，后面是剧本台词（合成时照着念的就是这些文字）
annotations/speakers/G1-S1-Q.csv  说话人时间（start,end,speaker），开头的声明算一个说话人"声明"
annotations/clips/G1-S1-Q.csv     疑似片段起止（start,end,label），连续几句同一个疑似类别的台词连成一个片段
audacity_labels/G1-S1-Q.speakers.txt、G1-S1-Q.clips.txt
                                  同样的说话人和片段，Audacity 标签格式：在 Audacity 里打开录音，
                                  "文件 → 导入 → 标签"导入，就能和同学标的标签上下对照
timeline/G1-S1-Q.csv              每句台词：序号、说话人、开始、结束（秒）、剧本里的类别、合成声音编号、舞台说明、台词
summary.csv                       每段录音一行：时长、计划时长（秒）、F 条件模拟的方式、句数、说话人数（含开头的声明）、
                                  疑似片段数、每个角色用的合成声音编号
voice_plan.csv                    每个剧本的每个角色用的合成声音（data/tts_voice_plan.csv 的副本）

时间是每句合成语音的起止（去掉了首尾静音），重叠的台词时间会交叠。合成语音偶尔会念错字或停顿不自然，
同学校对时听到的"实际说了什么"以录音为准。
"""


def _seconds(start: float) -> str:
    minutes, secs = divmod(int(time.time() - start), 60)
    return f"{minutes} 分 {secs:02d} 秒"


def load_tts(cfg: dict, threads: int):
    """加载 Kokoro 语音合成模型，返回 tts(text, sid, speed) -> (采样, 采样率)。

    纯中文的台词用 lang="zh"（中文念得最稳）；有英文字母的台词（"A区""HT"，见 has_latin）用不指定语言的
    中英混读设置，否则字母会被吞掉。两个设置第一次用到时才加载。
    """
    import numpy as np
    import sherpa_onnx

    from pipeline.models import MODEL_SPECS, model_path, model_ready, models_dir
    from pipeline.tts_dataset import has_latin

    folder = model_path(cfg, "kokoro_tts")
    spec = next(s for s in MODEL_SPECS if s["name"] == "kokoro_tts")
    if not model_ready(spec, models_dir(cfg)):
        raise SystemExit(f"找不到语音合成模型：{folder}。请先运行 python models/download_models.py --optional kokoro_tts")
    d = str(folder)
    engines = {}

    def engine(lang):
        if lang not in engines:
            engines[lang] = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                        model=f"{d}/model.onnx", voices=f"{d}/voices.bin", tokens=f"{d}/tokens.txt",
                        lexicon=f"{d}/lexicon-us-en.txt,{d}/lexicon-zh.txt", data_dir=f"{d}/espeak-ng-data",
                        dict_dir=f"{d}/dict", lang=lang),
                    num_threads=threads),
                rule_fsts=f"{d}/date-zh.fst,{d}/phone-zh.fst,{d}/number-zh.fst",
                max_num_sentences=1))
        return engines[lang]

    def tts(text, sid, speed):
        audio = engine("" if has_latin(text) else "zh").generate(text, sid=int(sid), speed=float(speed))
        return np.array(audio.samples, dtype=np.float32), audio.sample_rate

    return tts


# ---------------- 挑声音（--make-voice-plan） ----------------

PROFILE_TEXT = "大家好，今天的行程是先去石林，下午三点半回到酒店，有问题随时问我。"


def _f0(x, sr) -> float:
    """粗略的音高（Hz）：每 10 毫秒一帧做自相关，取有声帧的中位数。"""
    import numpy as np

    frame, hop, values = int(0.04 * sr), int(0.01 * sr), []
    for i in range(0, len(x) - frame, hop):
        w = x[i:i + frame]
        if np.sqrt(np.mean(w ** 2)) < 0.02:
            continue
        ac = np.correlate(w, w, "full")[frame - 1:]
        lo, hi = int(sr / 400), int(sr / 70)
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.3 * ac[0]:
            values.append(sr / k)
    return float(np.median(values)) if values else 0.0


def voice_profiles(cfg: dict, tts, cache_path: Path) -> dict:
    """每个中文合成声音（3—102 号）念同一句话，算音高和 CAM++ 声纹（结果存在 tmp/ 里，下次直接用）。"""
    import numpy as np
    import sherpa_onnx
    from scipy.signal import resample_poly

    from pipeline.models import model_path
    from pipeline.tts_dataset import FEMALE_SIDS, MALE_SIDS

    if cache_path.is_file():
        return {int(k): v for k, v in json.loads(cache_path.read_text(encoding="utf-8")).items()}
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model_path(cfg, "campplus")), num_threads=4))
    profiles = {}
    for sid in list(FEMALE_SIDS) + list(MALE_SIDS):
        samples, sr = tts(PROFILE_TEXT, sid, 1.0)
        y = resample_poly(samples, 16000, sr).astype(np.float32)
        stream = extractor.create_stream()
        stream.accept_waveform(16000, y)
        stream.input_finished()
        emb = np.array(extractor.compute(stream))
        profiles[sid] = {"f0": round(_f0(y, 16000), 1), "emb": (emb / np.linalg.norm(emb)).round(5).tolist()}
        print(f"  声音 {sid}：音高约 {profiles[sid]['f0']:.0f} Hz", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(profiles), encoding="utf-8")
    return profiles


def make_plan(cfg: dict, tts) -> None:
    from pipeline.data import load_scripts
    from pipeline.tts_dataset import VOICE_PLAN, VOICE_RULES, make_voice_plan, pick_disclaimer_voice, write_voice_plan

    print("== 给每个角色挑合成声音 ==")
    profiles = voice_profiles(cfg, tts, Path(cfg["paths"]["tmp"]) / "tts_voice_profiles.json")
    disclaimer = pick_disclaimer_voice(profiles)
    others = {sid: p for sid, p in profiles.items() if sid != disclaimer}  # 声明的声音不给角色用
    plan = make_voice_plan(load_scripts(), others, VOICE_RULES)
    write_voice_plan(VOICE_PLAN, plan, disclaimer)
    print(f"开头声明用 {disclaimer} 号声音；{len(plan)} 个角色用了 {len({p['sid'] for p in plan})} 个不同的声音。")
    print(f"已写入 {VOICE_PLAN}（可以用 Excel 打开修改 sid 和 speed，再重新合成）。")


# ---------------- 合成录音 ----------------


def babble_pieces(scripts: list[dict], voices: dict, tts, per_script: int = 2) -> dict:
    """嘈杂声的素材：每个剧本取几句中等长度的台词合成好，返回 {剧本编号: [采样, ...]}。

    合成某个剧本的 N 条件时，只用别的剧本的句子（不会听到和本剧本一样的话）。
    """
    from pipeline.tts_dataset import clean_tts_text, synthesize

    pieces = {}
    for script in scripts:
        lines = [line for line in script["lines"] if 15 <= len(line["text"]) <= 40][:per_script]
        pieces[script["id"]] = [synthesize(tts, clean_tts_text(line["text"]), voices[script["id"]][line["speaker"]]["sid"])
                                for line in lines]
    return pieces


def encode(audio, path: Path, stem: str, tmp: Path, fmt: str) -> None:
    """用 ffmpeg 写成 m4a（AAC 64 kbps）或 wav，文件信息里写明 AI 合成。先写到临时文件，成功了再改名。"""
    from pipeline.audio import _last_lines, _run_ffmpeg, write_wav
    from pipeline.tts_dataset import SR

    tmp.mkdir(parents=True, exist_ok=True)
    source = tmp / f"{stem}.synth.wav"
    write_wav(source, audio, SR)
    partial = path.with_name(path.stem + ".partial" + path.suffix)
    codec = ["-c:a", "aac", "-b:a", "64k"] if fmt == "m4a" else ["-c:a", "pcm_s16le"]
    meta = ["-metadata", f"title={META_TITLE.format(stem=stem)}", "-metadata", f"comment={META_COMMENT}",
            "-metadata", f"artist={META_ARTIST}"]
    code, err = _run_ffmpeg(["-loglevel", "error", "-y", "-i", str(source), "-ac", "1", *codec, *meta, str(partial)])
    source.unlink(missing_ok=True)
    if code != 0 or not partial.is_file():
        raise RuntimeError(f"ffmpeg 写 {path.name} 失败：\n" + _last_lines(err))
    os.replace(partial, path)


def write_answer(truth: dict, stem: str, script_id: str, key: Path) -> None:
    from pipeline.annotations import write_clips_csv, write_turns_csv

    (key / "references").mkdir(parents=True, exist_ok=True)
    (key / "references" / f"{stem}.txt").write_text(truth["reference"], encoding="utf-8")
    write_turns_csv(key / "annotations" / "speakers" / f"{stem}.csv", truth["turns"])
    write_clips_csv(key / "annotations" / "clips" / f"{stem}.csv", truth["clips"])
    labels = key / "audacity_labels"  # Audacity 的"导入标签"能直接打开（制表符分隔：开始、结束、标签）
    labels.mkdir(parents=True, exist_ok=True)
    for kind, rows in (("speakers", truth["turns"]), ("clips", truth["clips"])):
        text = "".join(f"{start:.3f}\t{end:.3f}\t{label}\n" for start, end, label in sorted(rows))
        (labels / f"{stem}.{kind}.txt").write_text(text, encoding="utf-8")
    (key / "timeline").mkdir(parents=True, exist_ok=True)
    with open(key / "timeline" / f"{stem}.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TIMELINE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(truth["timeline"])


def write_summary(out: Path) -> None:
    """按 answer_key/timeline 里已有的录音重写 summary.csv（中断后接着合成，汇总也是完整的）。"""
    from pipeline.annotations import read_clips_csv, read_turns_csv
    from pipeline.audio import probe
    from pipeline.data import load_recording_plan

    key = out / "answer_key"
    plan = {Path(item["file_name"]).stem: item for item in load_recording_plan()}
    rows = []
    for path in sorted((key / "timeline").glob("*.csv")):
        stem = path.stem
        with open(path, encoding="utf-8-sig", newline="") as f:
            timeline = list(csv.DictReader(f))
        turns = read_turns_csv(key / "annotations" / "speakers" / f"{stem}.csv")
        audio = next((out / "recordings").glob(f"{stem}-TTS.*"), None)
        group, script = stem.split("-")[0][1:], stem.split("-")[1][1:]
        variant = ("pocket" if (int(group) + int(script)) % 2 else "far") if stem.endswith("F") else ""
        rows.append({
            "file": f"{stem}-TTS", "script_id": stem[:-2], "condition": stem[-1],
            "f_variant": F_VARIANT_NAMES[variant], "duration_s": round(probe(audio)["duration"], 1) if audio else "",
            "expected_s": round(plan[stem]["expected_minutes"] * 60, 1), "lines": len(timeline),
            "speakers": len({s for _, _, s in turns}),
            "clips": len(read_clips_csv(key / "annotations" / "clips" / f"{stem}.csv")),
            "voices": " ".join(sorted({f"{r['speaker']}={r['sid']}" for r in timeline})),
        })
    with open(key / "summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------- 选做的数字实验（tools/tf_lab） ----------------

DIGITS = "零一二三四五六七八九"
DIGIT_SPEEDS = (0.9, 1.0, 1.15)  # 每个声音念 3 遍：慢、正常、快

DIGITS_README = """AI 合成的数字录音（选做的 TensorFlow 数字实验用，见 tools/tf_lab/README.md）

由人工智能技术（Kokoro v1.1 离线语音合成）自动生成，不是真实录音，没有用任何真人的声音做参考。
每个合成声音把"零"到"九"各念 3 遍（语速慢、正常、快），16000 Hz 单声道 16 位 WAV：
    <数字>/tts<声音编号>_<第几遍>.wav      例如 3/tts23_2.wav 是 23 号声音第 2 遍念的"三"
训练脚本按文件名里"_"前面的部分分人（tts23 算一个"人"），按人留出一部分声音只做测试。
"""


def synthesize_digits(tts, out: Path, sids=None) -> int:
    """每个合成声音把"零"到"九"各念 3 遍，存成 out/digits/<数字>/tts<声音编号>_<第几遍>.wav（16000 Hz）。返回文件数。"""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    from pipeline.tts_dataset import FEMALE_SIDS, MALE_SIDS, SR, synthesize

    sids = list(sids) if sids is not None else list(FEMALE_SIDS) + list(MALE_SIDS)
    folder = out / "digits"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "README.txt").write_text(DIGITS_README, encoding="utf-8")
    pad = np.zeros(int(0.1 * 16000), dtype=np.float32)
    count = 0
    for digit, text in enumerate(DIGITS):
        (folder / str(digit)).mkdir(exist_ok=True)
        for sid in sids:
            for rep, speed in enumerate(DIGIT_SPEEDS, start=1):
                x = resample_poly(synthesize(tts, text, sid, speed), 16000, SR).astype(np.float32)
                x = np.concatenate([pad, x[:int(0.8 * 16000)], pad])  # 前后留 0.1 秒，总长不到 1 秒
                x = np.clip(x * (0.5 / (np.max(np.abs(x)) + 1e-9)), -1, 1)
                with sf.SoundFile(str(folder / str(digit) / f"tts{sid}_{rep}.wav"), "w", 16000, 1, "PCM_16") as f:
                    f.title = f"AI合成·数字 {text}（声音 {sid}）"
                    f.comment = META_COMMENT
                    f.write(x)
                count += 1
        print(f"  \"{text}\"：{len(sids) * len(DIGIT_SPEEDS)} 段", flush=True)
    return count


def seed_for(script_id: str, condition: str, base: int) -> int:
    """每段录音一个固定的随机种子：同样的参数再合成一遍，结果一模一样。"""
    group, script = int(script_id.split("-")[0][1:]), int(script_id.split("-")[1][1:])
    return base * 1000 + group * 100 + script * 10 + CONDITION_CODES.index(condition)


def synthesize_all(cfg: dict, tts, script_ids: list[str], conditions: list[str], out: Path, fmt: str, seed: int,
                   overwrite: bool) -> int:
    import shutil

    from pipeline.data import AI_DISCLAIMER, load_scripts
    from pipeline.tts_dataset import VOICE_PLAN, load_voice_plan, render_recording

    voices, disclaimer = load_voice_plan()
    all_scripts = load_scripts()
    scripts = [s for s in all_scripts if s["id"] in script_ids]
    recordings, key = out / "recordings", out / "answer_key"
    recordings.mkdir(parents=True, exist_ok=True)
    key.mkdir(parents=True, exist_ok=True)
    (out / "README.txt").write_text(README.format(disclaimer=AI_DISCLAIMER), encoding="utf-8")
    (key / "README.txt").write_text(ANSWER_README, encoding="utf-8")
    shutil.copyfile(VOICE_PLAN, key / "voice_plan.csv")
    tmp = Path(cfg["paths"]["tmp"])

    def done(stem):
        return (recordings / f"{stem}-TTS.{fmt}").is_file() and (key / "timeline" / f"{stem}.csv").is_file()

    todo = [(s, c) for s in scripts for c in conditions if overwrite or not done(f"{s['id']}-{c}")]
    skipped = len(scripts) * len(conditions) - len(todo)
    if skipped:
        print(f"已经合成好的 {skipped} 段跳过（要重做加 --overwrite）。")
    if not todo:
        write_summary(out)
        return 0

    start = time.time()
    pieces = {}
    if any(c == "N" for _, c in todo):
        print("先合成嘈杂声素材（每个剧本取 2 句）……", flush=True)
        pieces = babble_pieces(all_scripts, voices, tts)
        print(f"  完成（用时 {_seconds(start)}）", flush=True)

    for number, (script, condition) in enumerate(todo, start=1):
        stem = f"{script['id']}-{condition}"
        if number == 1 or todo[number - 2][0]["id"] != script["id"]:
            cache = {}  # 每个剧本的台词只合成一次，Q、N、F 三种条件共用
        pool = [x for sid, xs in pieces.items() if sid != script["id"] for x in xs]
        audio, truth = render_recording(script, condition, voices[script["id"]], disclaimer, tts,
                                        seed=seed_for(script["id"], condition, seed), babble_pool=pool, cache=cache)
        encode(audio, recordings / f"{stem}-TTS.{fmt}", stem, tmp, fmt)
        write_answer(truth, stem, script["id"], key)
        print(f"[{number}/{len(todo)}] {stem}-TTS.{fmt}：{truth['duration'] / 60:.1f} 分钟，"
              f"{len(truth['timeline'])} 句（已用 {_seconds(start)}）", flush=True)
    write_summary(out)
    print(f"完成：录音在 {recordings}，标准答案在 {key}。")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="按剧本合成 AI 测评录音和标准答案（离线语音合成，不联网）")
    parser.add_argument("--scripts", nargs="+", metavar="ID", help="只合成这些剧本，如 G1-S1 G3-S2（不填就是全部 24 个）")
    parser.add_argument("--conditions", nargs="+", choices=CONDITION_CODES, default=list(CONDITION_CODES),
                        help="只合成这些条件（Q 安静、N 嘈杂教室、F 口袋或远距离），不填就是全部")
    parser.add_argument("--out", default=str(ROOT / "ai_recordings"), help="输出文件夹（默认项目里的 ai_recordings）")
    parser.add_argument("--format", choices=("m4a", "wav"), default="m4a", help="录音格式（默认 m4a，和手机录音一样）")
    parser.add_argument("--seed", type=int, default=1, help="随机种子（停顿、重叠、嘈杂声；同样的种子结果一样）")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1), help="语音合成用几个线程")
    parser.add_argument("--overwrite", action="store_true", help="已经合成好的也重做")
    parser.add_argument("--digits", action="store_true",
                        help="只合成选做数字实验用的录音（100 个声音 × 零到九 × 3 遍），放在 输出文件夹/digits")
    parser.add_argument("--make-voice-plan", action="store_true",
                        help="重新给每个角色挑声音，改写 data/tts_voice_plan.csv（不合成录音）")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from pipeline.config import load_config
    from pipeline.data import load_scripts

    cfg = load_config()
    known = [s["id"] for s in load_scripts()]
    wanted = args.scripts or known
    unknown = [s for s in wanted if s not in known]
    if unknown:
        print(f"没有这些剧本：{'、'.join(unknown)}。剧本编号形如 G1-S1（组号-剧本号）。")
        return 2

    print("== AI 合成测评录音（离线语音合成，内容全部虚构，每段开头念 AI 合成声明） ==", flush=True)
    tts = load_tts(cfg, args.threads)
    if args.make_voice_plan:
        make_plan(cfg, tts)
        return 0
    if args.digits:
        count = synthesize_digits(tts, Path(args.out))
        print(f"完成：{count} 段数字录音在 {Path(args.out) / 'digits'}。")
        return 0
    return synthesize_all(cfg, tts, wanted, args.conditions, Path(args.out), args.format, args.seed, args.overwrite)


if __name__ == "__main__":
    sys.exit(main())
