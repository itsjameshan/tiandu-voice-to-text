"""模型清单与模型文件的位置。

本工具用的模型都是别人训练好的公开模型（sherpa-onnx 在 GitHub 上发布），
不需要我们自己训练。这个文件只做两件事：
1. MODEL_SPECS：列出每个模型从哪里下载、下载下来的文件叫什么、解压后应该有哪些文件；
2. model_path()、missing_models()：告诉其他步骤“模型文件在哪里”“还缺哪些模型”。
真正的下载由 models/download_models.py 完成（本文件不联网，处理录音时也不联网）。

基线做法：
    必需模型 5 项——SenseVoice 识别（2024-07-17 版）、Silero VAD 端点检测、
    pyannote 说话人分割、CAM++ 声纹，以及给自检和冒烟测试用的四人测试音频；
    可选模型 5 项（声纹备选、数字规则、对照识别模型、标点模型、热词实验模型），需要时再下载。
    “模型齐不齐”只看 check 里列的文件在不在。
可改进方向：
    给每个文件记下 SHA-256，下载后核对，能发现下载不完整或被改过的文件；
    加国内镜像地址，GitHub 访问慢时换用镜像。
测评指标：
    不涉及识别效果；由 tests/test_models.py 检查清单完整、地址正确（识别模型必须是 2024-07-17 版），
    以及在已下载模型的电脑上 missing_models 为空。

注意：不要换成 2025-09-09 版的 SenseVoice，那是粤语模型，普通话也按粤语输出。
"""
from pathlib import Path

# 所有模型都从 sherpa-onnx 的 GitHub 发布页下载
RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

# 几个常用的文件夹名、文件名（model_path 里也要用）
SENSE_VOICE_DIR = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
PYANNOTE_DIR = "sherpa-onnx-pyannote-segmentation-3-0"
CAMPPLUS_FILE = "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
SILERO_VAD_FILE = "silero_vad.onnx"
TEST_WAV_FILE = "0-four-speakers-zh.wav"

PARAFORMER_DIR = "sherpa-onnx-paraformer-zh-small-2024-03-09"
PUNCT_DIR = "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12"
CONFORMER_DIR = "sherpa-onnx-conformer-zh-stateless2-2023-05-23"

# 每个模型一项：
#   name      名字（下载脚本的 --optional 后面写这个）
#   title     中文说明（只用于显示）
#   url       下载地址
#   archive   下载下来的文件名（.tar.bz2 是压缩包，下载后要解压）
#   check     解压后必须存在的文件（相对于 models 文件夹），都在才算“已下载”
#   size_mb   下载大小，单位 MB（大约值）
#   required  True 表示必需；False 表示可选，需要时用 --optional 下载
MODEL_SPECS = [
    {
        "name": "sense_voice",
        "title": "语音识别 SenseVoice（2024-07-17 版）",
        "url": f"{RELEASES}/asr-models/{SENSE_VOICE_DIR}.tar.bz2",
        "archive": f"{SENSE_VOICE_DIR}.tar.bz2",
        "check": [
            f"{SENSE_VOICE_DIR}/model.int8.onnx",
            f"{SENSE_VOICE_DIR}/tokens.txt",
            f"{SENSE_VOICE_DIR}/test_wavs/zh.wav",
        ],
        "size_mb": 163,
        "required": True,
    },
    {
        "name": "silero_vad",
        "title": "端点检测 Silero VAD",
        "url": f"{RELEASES}/asr-models/{SILERO_VAD_FILE}",
        "archive": SILERO_VAD_FILE,
        "check": [SILERO_VAD_FILE],
        "size_mb": 0.6,
        "required": True,
    },
    {
        "name": "pyannote",
        "title": "说话人分割 pyannote 3.0",
        "url": f"{RELEASES}/speaker-segmentation-models/{PYANNOTE_DIR}.tar.bz2",
        "archive": f"{PYANNOTE_DIR}.tar.bz2",
        "check": [f"{PYANNOTE_DIR}/model.onnx"],
        "size_mb": 7,
        "required": True,
    },
    {
        "name": "campplus",
        "title": "声纹 CAM++",
        # 地址里的 recongition 拼写就是这样（发布页本身拼错了），不要改
        "url": f"{RELEASES}/speaker-recongition-models/{CAMPPLUS_FILE}",
        "archive": CAMPPLUS_FILE,
        "check": [CAMPPLUS_FILE],
        "size_mb": 28,
        "required": True,
    },
    {
        "name": "test_wav",
        "title": "四人测试音频（只用于自检和自动测试）",
        "url": f"{RELEASES}/speaker-segmentation-models/{TEST_WAV_FILE}",
        "archive": TEST_WAV_FILE,
        "check": [TEST_WAV_FILE],
        "size_mb": 1.8,
        "required": True,
    },
    # ---------- 以下是可选模型 ----------
    {
        "name": "eres2net",
        "title": "声纹备选 ERes2Net（第 3 组对比用）",
        "url": f"{RELEASES}/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        "archive": "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        "check": ["3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"],
        "size_mb": 40,
        "required": False,
    },
    {
        "name": "itn_fst",
        "title": "汉字数字转阿拉伯数字的规则文件（第 4 组对比用）",
        "url": f"{RELEASES}/asr-models/itn_zh_number.fst",
        "archive": "itn_zh_number.fst",
        "check": ["itn_zh_number.fst"],
        "size_mb": 0.03,
        "required": False,
    },
    {
        "name": "paraformer_small",
        "title": "对照识别模型 Paraformer 小模型（不带标点）",
        "url": f"{RELEASES}/asr-models/{PARAFORMER_DIR}.tar.bz2",
        "archive": f"{PARAFORMER_DIR}.tar.bz2",
        "check": [f"{PARAFORMER_DIR}/model.int8.onnx", f"{PARAFORMER_DIR}/tokens.txt"],
        "size_mb": 78,
        "required": False,
    },
    {
        "name": "punct",
        "title": "标点模型（给 Paraformer 加标点）",
        "url": f"{RELEASES}/punctuation-models/{PUNCT_DIR}.tar.bz2",
        "archive": f"{PUNCT_DIR}.tar.bz2",
        "check": [f"{PUNCT_DIR}/model.onnx"],
        "size_mb": 279,
        "required": False,
    },
    {
        "name": "conformer_hotword",
        "title": "支持热词的识别模型（第 5 组热词实验用）",
        "url": f"{RELEASES}/asr-models/{CONFORMER_DIR}.tar.bz2",
        "archive": f"{CONFORMER_DIR}.tar.bz2",
        "check": [
            f"{CONFORMER_DIR}/encoder-epoch-99-avg-1.onnx",
            f"{CONFORMER_DIR}/decoder-epoch-99-avg-1.onnx",
            f"{CONFORMER_DIR}/joiner-epoch-99-avg-1.onnx",
            f"{CONFORMER_DIR}/tokens.txt",
        ],
        "size_mb": 456,
        "required": False,
    },
]

# model_path 能查的文件（相对于 models 文件夹）
_PATHS = {
    "sense_voice_model": f"{SENSE_VOICE_DIR}/model.int8.onnx",
    "sense_voice_tokens": f"{SENSE_VOICE_DIR}/tokens.txt",
    "sense_voice_test_zh": f"{SENSE_VOICE_DIR}/test_wavs/zh.wav",
    "silero_vad": SILERO_VAD_FILE,
    "pyannote": f"{PYANNOTE_DIR}/model.onnx",
    "campplus": CAMPPLUS_FILE,
    "test_wav": TEST_WAV_FILE,
}


def models_dir(cfg: dict) -> Path:
    """模型文件夹：config.yaml 里 paths.models 的值（load_config 已经把它换成了绝对路径）。"""
    return Path(cfg["paths"]["models"])


def model_path(cfg: dict, key: str) -> Path:
    """返回某个模型文件的完整路径，例如 model_path(cfg, "silero_vad")。

    key 可以是：sense_voice_model、sense_voice_tokens、silero_vad、pyannote、campplus、
    test_wav（四人测试音频）、sense_voice_test_zh（识别模型自带的中文测试音频）。
    只拼出路径，不检查文件在不在；要检查请用 missing_models。
    """
    if key not in _PATHS:
        raise KeyError(f"没有叫 {key} 的模型文件。可以用的有：{'、'.join(_PATHS)}")
    return models_dir(cfg) / _PATHS[key]


def model_ready(spec: dict, folder: str | Path) -> bool:
    """某个模型是否已经在 folder 里：check 列出的文件全都在才算。"""
    folder = Path(folder)
    return all((folder / rel).is_file() for rel in spec["check"])


def missing_models(cfg: dict) -> list[str]:
    """返回还没下载的必需模型的名字列表（按 MODEL_SPECS 的顺序）；空列表表示必需模型都在。"""
    folder = models_dir(cfg)
    return [spec["name"] for spec in MODEL_SPECS if spec["required"] and not model_ready(spec, folder)]
