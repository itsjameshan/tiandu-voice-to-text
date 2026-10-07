import sys
import time

import wave

import numpy as np
import sherpa_onnx


def write_wav(path, samples, sr):
    pcm = np.clip(samples, -1, 1)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())

mdir = sys.argv[1]
out = sys.argv[2]
text = sys.argv[3]
speed = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

cfg = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=f"{mdir}/model.onnx",
            lexicon=f"{mdir}/lexicon.txt",
            tokens=f"{mdir}/tokens.txt",
            dict_dir=f"{mdir}/dict",
        ),
        num_threads=2,
    ),
    rule_fsts=f"{mdir}/date.fst,{mdir}/phone.fst,{mdir}/number.fst,{mdir}/new_heteronym.fst",
    max_num_sentences=1,
)
tts = sherpa_onnx.OfflineTts(cfg)
t0 = time.time()
audio = tts.generate(text, sid=0, speed=speed)
el = time.time() - t0
samples = np.array(audio.samples, dtype=np.float32)
write_wav(out, samples, audio.sample_rate)
dur = len(samples) / audio.sample_rate
n = sum(1 for ch in text if ("一" <= ch <= "鿿") or ch.isalnum())
print(f"sr={audio.sample_rate} dur={dur:.2f}s chars={n} rate={n / dur * 60:.0f}/min gen={el:.1f}s peak={np.abs(samples).max():.2f}")
