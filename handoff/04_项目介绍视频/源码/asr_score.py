import json
import sys
import wave

import numpy as np
import sherpa_onnx

mdir, meta_path = sys.argv[1], sys.argv[2]
rec = sherpa_onnx.OfflineRecognizer.from_paraformer(
    paraformer=f"{mdir}/model.int8.onnx", tokens=f"{mdir}/tokens.txt", num_threads=2
)


def norm(s):
    return "".join(ch for ch in s if ("一" <= ch <= "鿿") or ch.isalnum()).lower()


def edit(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[len(b)]


meta = json.load(open(meta_path, encoding="utf-8"))
E = N = 0
for m in meta:
    with wave.open(m["wav"], "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    s = rec.create_stream()
    s.accept_waveform(sr, x)
    rec.decode_stream(s)
    ref, hyp = norm(m["text"]), norm(s.result.text)
    e = edit(ref, hyp)
    E += e
    N += len(ref)
    flag = "  <--" if e / max(1, len(ref)) > 0.15 else ""
    print(f"s{m['shot']:02d}_{m['k']:02d} cer={e / max(1, len(ref)):.2f} | {hyp}{flag}")
print(f"overall CER={E / N:.3f} ({E}/{N})")
