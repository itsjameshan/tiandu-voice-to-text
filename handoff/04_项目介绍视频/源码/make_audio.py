"""Mix narration clips and light sound effects into one track; also write the SRT subtitles."""
import json
import sys
import wave

import numpy as np

SR = 44100
tl = json.load(open(sys.argv[1], encoding="utf-8"))
out_wav, out_srt = sys.argv[2], sys.argv[3]
rng = np.random.default_rng(7)


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    if sr != SR:
        n = int(len(x) * SR / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def env(n, attack, decay_tau):
    t = np.arange(n) / SR
    a = np.minimum(1, t / max(attack, 1e-4))
    return a * np.exp(-t / decay_tau)


def smooth(x, k):
    return np.convolve(x, np.ones(k) / k, mode="same")


def sfx(kind):
    if kind == "beep":
        n = int(0.16 * SR)
        t = np.arange(n) / SR
        return 0.22 * np.sin(2 * np.pi * 1046 * t) * np.minimum(1, t / 0.005) * np.minimum(1, (0.16 - t) / 0.03)
    if kind == "ding":
        n = int(0.9 * SR)
        t = np.arange(n) / SR
        tone = np.sin(2 * np.pi * 1318.5 * t) + 0.45 * np.sin(2 * np.pi * 1975.5 * t) + 0.2 * np.sin(2 * np.pi * 2637 * t)
        return 0.11 * tone * env(n, 0.004, 0.28)
    if kind == "whoosh":
        n = int(0.55 * SR)
        noise = smooth(rng.standard_normal(n), 18)
        t = np.arange(n) / n
        return 0.35 * noise * np.sin(np.pi * t) ** 2
    if kind == "flip":
        n = int(0.12 * SR)
        noise = smooth(rng.standard_normal(n), 6)
        return 0.28 * noise * env(n, 0.002, 0.03)
    if kind == "stamp":
        n = int(0.45 * SR)
        t = np.arange(n) / SR
        thump = np.sin(2 * np.pi * 95 * t) * env(n, 0.003, 0.09)
        click = smooth(rng.standard_normal(n), 3) * env(n, 0.001, 0.012)
        return 0.45 * thump + 0.12 * click
    raise ValueError(kind)


total = int((tl["total"] + 0.5) * SR)
track = np.zeros(total, dtype=np.float32)
for c in tl["clips"]:
    x = read_wav(c["wav"]) * 0.9
    a = int(c["at"] * SR)
    track[a:a + len(x)] += x[: max(0, total - a)]
for e in tl["sfx"]:
    x = sfx(e["kind"]) * e["gain"]
    a = int(e["t"] * SR)
    track[a:a + len(x)] += x[: max(0, total - a)]
peak = np.abs(track).max()
track = track / max(peak, 1e-6) * 0.95
pcm = (track * 32767).astype(np.int16)
with wave.open(out_wav, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())


def ts(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


with open(out_srt, "w", encoding="utf-8") as f:
    for i, s in enumerate(tl["subs"], start=1):
        text = s["text"].rstrip("，、；：。")
        f.write(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n{text}\n\n")
print(f"audio {total / SR:.1f}s peak-normalized; srt {len(tl['subs'])} cues")
