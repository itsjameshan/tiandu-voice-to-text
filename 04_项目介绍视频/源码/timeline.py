"""Build the shot schedule, subtitles, cue times and sound-effect events from the TTS clips."""
import json
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from narration import SHOTS  # noqa: E402

LEAD, GAP, TAIL, END_TAIL = 0.7, 0.35, 0.8, 3.2
SR = 44100

STATION_PHRASES = ["统一格式", "降噪切段", "转成文字", "分清谁在说话", "把数字和证号写规范",
                   "用深度学习模型给每句话分类", "按时间截出证据片段", "最后生成核查初稿"]


def eff(s):
    return sum(1 for ch in s if ("一" <= ch <= "鿿") or ch.isalnum())


def split_cards(text, max_chars=34):
    """Split a sentence into subtitle cards at punctuation, each at most max_chars."""
    if eff(text) <= max_chars:
        return [text]
    parts, cur = [], ""
    for ch in text:
        cur += ch
        if ch in "，、；：。？！" and eff(cur) >= 10:
            parts.append(cur)
            cur = ""
    if cur:
        parts.append(cur)
    cards, buf = [], ""
    for p in parts:
        if buf and eff(buf + p) > max_chars:
            cards.append(buf)
            buf = p
        else:
            buf += p
    if buf:
        cards.append(buf)
    return cards


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return x, sr


def build(meta_path):
    meta = json.load(open(meta_path, encoding="utf-8"))
    by_shot = {}
    for m in meta:
        by_shot.setdefault(m["shot"], []).append(m)
    shots, subs, clips = [], [], []
    t = 0.0
    for si in range(1, len(SHOTS) + 1):
        items = sorted(by_shot[si], key=lambda m: m["k"])
        local = LEAD
        cues = []
        for j, m in enumerate(items):
            cues.append({"start": round(local, 3), "dur": m["dur"], "text": m["text"]})
            clips.append({"wav": m["wav"], "at": round(t + local, 3)})
            cards = split_cards(m["text"])
            total = sum(eff(c) for c in cards)
            cs = local
            for c in cards:
                cd = m["dur"] * eff(c) / total
                subs.append({"start": round(t + cs, 3), "end": round(t + cs + cd, 3), "text": c})
                cs += cd
            local += m["dur"] + (GAP if j < len(items) - 1 else 0)
        local += END_TAIL if si == len(SHOTS) else TAIL
        shots.append({"shot": si, "start": round(t, 3), "dur": round(local, 3), "cues": cues})
        t += local
    # extend each subtitle slightly so it does not flicker off in short gaps
    for a, b in zip(subs, subs[1:]):
        if b["start"] - a["end"] < 0.5:
            a["end"] = b["start"]
    return shots, subs, clips, t


def station_times(shot):
    """Local times when each pipeline station is named in shot 7's second sentence."""
    c = shot["cues"][1]
    text = c["text"]
    total = eff(text)
    out = []
    for p in STATION_PHRASES:
        idx = text.index(p)
        out.append(round(c["start"] + c["dur"] * eff(text[:idx]) / total, 3))
    return out


def sfx_events(shots):
    ev = []

    def at(si, local, kind, gain=1.0):
        ev.append({"t": round(shots[si - 1]["start"] + local, 3), "kind": kind, "gain": gain})

    s = shots
    at(1, s[0]["cues"][1]["start"] + 0.9, "beep")
    at(2, 1.2, "whoosh", 0.7)
    at(2, s[1]["cues"][2]["start"] + 0.2, "flip", 0.8)
    at(4, s[3]["cues"][1]["start"] + 0.3, "whoosh", 0.7)
    at(5, s[4]["cues"][1]["start"] + s[4]["cues"][1]["dur"] * 0.55, "stamp")
    at(6, s[5]["cues"][2]["start"] + 0.2, "ding", 0.7)
    for st in station_times(s[6]):
        at(7, st, "ding", 0.55)
    at(9, s[8]["cues"][2]["start"] + s[8]["cues"][2]["dur"] * 0.45, "ding", 0.7)
    for c in s[9]["cues"][1:]:
        at(10, c["start"], "flip", 0.8)
    for c in s[10]["cues"][1:]:
        at(11, c["start"], "ding", 0.5)
    return ev


if __name__ == "__main__":
    shots, subs, clips, total = build(sys.argv[1])
    out = {"shots": shots, "subs": subs, "clips": clips, "total": round(total, 3),
           "stations": station_times(shots[6]), "sfx": sfx_events(shots)}
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for s in shots:
        print(f"shot {s['shot']:2d} start {s['start']:6.1f}s dur {s['dur']:5.1f}s")
    print(f"total {total:.1f}s, subtitles {len(subs)}, longest sub {max(eff(x['text']) for x in subs)} chars")
