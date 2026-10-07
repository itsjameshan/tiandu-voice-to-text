import json
import os
import sys
import wave

import numpy as np
import sherpa_onnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from narration import SHOTS  # noqa: E402

mdir, outdir, speed = sys.argv[1], sys.argv[2], float(sys.argv[3])
kind = sys.argv[4] if len(sys.argv) > 4 else "melo"
vocoder = sys.argv[5] if len(sys.argv) > 5 else ""
os.makedirs(outdir, exist_ok=True)

if kind == "matcha":
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=f"{mdir}/model-steps-3.onnx",
                vocoder=vocoder,
                lexicon=f"{mdir}/lexicon.txt",
                tokens=f"{mdir}/tokens.txt",
                dict_dir=f"{mdir}/dict",
            ),
            num_threads=2,
        ),
        rule_fsts=f"{mdir}/date.fst,{mdir}/phone.fst,{mdir}/number.fst",
        max_num_sentences=1,
    )
else:
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


def trim(x, sr, thr=0.01, pad=0.05):
    idx = np.where(np.abs(x) > thr)[0]
    if len(idx) == 0:
        return x
    a = max(0, idx[0] - int(pad * sr))
    b = min(len(x), idx[-1] + int(pad * sr))
    return x[a:b]


meta = []
for si, sentences in enumerate(SHOTS, start=1):
    for k, text in enumerate(sentences, start=1):
        audio = tts.generate(text, sid=0, speed=speed)
        x = trim(np.array(audio.samples, dtype=np.float32), audio.sample_rate)
        path = os.path.join(outdir, f"s{si:02d}_{k:02d}.wav")
        pcm = (np.clip(x / max(1e-6, np.abs(x).max()) * 0.89, -1, 1) * 32767).astype(np.int16)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(audio.sample_rate)
            w.writeframes(pcm.tobytes())
        dur = len(x) / audio.sample_rate
        meta.append({"shot": si, "k": k, "text": text, "wav": path, "dur": round(dur, 3), "sr": audio.sample_rate})
        print(f"s{si:02d}_{k:02d} {dur:5.2f}s {text}")
with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
tot = sum(m["dur"] for m in meta)
print(f"sentences={len(meta)} speech={tot:.1f}s")
