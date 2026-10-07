"""Render the explainer video.

  python3 render.py sheet <shot> [t1 t2 ...]   -> previews/shotNN.png contact sheet (local times)
  python3 render.py frame <global_t> <out.png>  -> one full-size frame
  python3 render.py seg <shot>                  -> segs/segNN.mp4 (video only)
"""
import json
import os
import subprocess
import sys
import time

import cairo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gfx import CREAM, W, H, col, corner_label, prog, subtitle, text, rrect, INK  # noqa: E402
from scenes_a import SCENES_A  # noqa: E402
from scenes_b import make_scenes_b  # noqa: E402

FPS = 30
TL = json.load(open(os.path.join(HERE, "timeline.json"), encoding="utf-8"))
SHOTS = TL["shots"]
SCENES = SCENES_A + make_scenes_b(TL)
TOTAL = TL["total"]


def shot_at(T):
    for i, s in enumerate(SHOTS):
        if T < s["start"] + s["dur"]:
            return i
    return len(SHOTS) - 1


def draw_frame(ctx, T):
    i = shot_at(T)
    sh = SHOTS[i]
    lt = T - sh["start"]
    ctx.save()
    SCENES[i](ctx, lt, sh)
    ctx.restore()
    if i == 0:
        a = 1 - prog(lt, 0, 0.6)
    else:
        a = 1 - prog(lt, 0, 0.3)
    if i < len(SHOTS) - 1:
        a = max(a, prog(lt, sh["dur"] - 0.25, 0.25))
    else:
        a = max(a, prog(lt, sh["dur"] - 0.7, 0.7))
    if a > 0:
        ctx.rectangle(0, 0, W, H)
        col(ctx, CREAM, a)
        ctx.fill()
    for s in TL["subs"]:
        if s["start"] <= T < s["end"]:
            subtitle(ctx, s["text"])
            break
    corner_label(ctx)


def new_surface():
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surf)
    ctx.set_antialias(cairo.ANTIALIAS_GOOD)
    return surf, ctx


def render_png(T, out):
    surf, ctx = new_surface()
    draw_frame(ctx, T)
    surf.write_to_png(out)


def sheet(si, times, out):
    sh = SHOTS[si]
    if not times:
        n = 9
        times = [round(0.4 + (sh["dur"] - 0.8) * k / (n - 1), 2) for k in range(n)]
    cols = 3
    rows = (len(times) + cols - 1) // cols
    tw, th = 640, 360
    big = cairo.ImageSurface(cairo.FORMAT_ARGB32, tw * cols, th * rows)
    bctx = cairo.Context(big)
    bctx.set_source_rgb(1, 1, 1)
    bctx.paint()
    for k, lt in enumerate(times):
        surf, ctx = new_surface()
        draw_frame(ctx, sh["start"] + lt)
        bctx.save()
        bctx.translate((k % cols) * tw, (k // cols) * th)
        bctx.scale(tw / W, th / H)
        bctx.set_source_surface(surf, 0, 0)
        bctx.paint()
        bctx.restore()
        rrect(bctx, (k % cols) * tw + 4, (k // cols) * th + 4, 92, 30, 6, INK, 0.75)
        text(bctx, f"{lt:.2f}s", (k % cols) * tw + 50, (k // cols) * th + 19, 18, (1, 1, 1), True)
    big.write_to_png(out)


def seg(si):
    sh = SHOTS[si]
    f0 = round(sh["start"] * FPS)
    f1 = round(SHOTS[si + 1]["start"] * FPS) if si < len(SHOTS) - 1 else round(TOTAL * FPS)
    os.makedirs(os.path.join(HERE, "segs"), exist_ok=True)
    out = os.path.join(HERE, "segs", f"seg{si + 1:02d}.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
           "-pix_fmt", "yuv420p", "-threads", "1", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    surf, ctx = new_surface()
    t0 = time.time()
    for f in range(f0, f1):
        ctx.save()
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_rgb(*CREAM)
        ctx.paint()
        ctx.restore()
        draw_frame(ctx, f / FPS)
        surf.flush()
        proc.stdin.write(bytes(surf.get_data()))
    proc.stdin.close()
    proc.wait()
    print(f"seg {si + 1}: frames {f0}-{f1} ({f1 - f0}) in {time.time() - t0:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sheet":
        si = int(sys.argv[2]) - 1
        times = [float(x) for x in sys.argv[3:]]
        os.makedirs(os.path.join(HERE, "previews"), exist_ok=True)
        out = os.path.join(HERE, "previews", f"shot{si + 1:02d}.png")
        sheet(si, times, out)
        print(out)
    elif mode == "frame":
        render_png(float(sys.argv[2]), sys.argv[3])
    elif mode == "seg":
        for a in sys.argv[2:]:
            seg(int(a) - 1)
