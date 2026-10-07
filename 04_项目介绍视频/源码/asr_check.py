import sys
import wave

import numpy as np
import sherpa_onnx

mdir = sys.argv[1]
files = sys.argv[2:]

rec = sherpa_onnx.OfflineRecognizer.from_paraformer(
    paraformer=f"{mdir}/model.int8.onnx",
    tokens=f"{mdir}/tokens.txt",
    num_threads=2,
)


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return data, sr


for f in files:
    x, sr = read_wav(f)
    s = rec.create_stream()
    s.accept_waveform(sr, x)
    rec.decode_stream(s)
    print(f"{f}: {s.result.text}")
