"""Helpers shared by the scene modules: cue timing, grouping, headlines, bubbles, small props."""
import math
from contextlib import contextmanager

import cairo

from gfx import (BLUE, BLUE_LL, CREAM, GOOD, GRAY, GRAY_L, INK, INK2, ORANGE, ORANGE_D, ORANGE_L, PI,
                 SKIN, TEAL, TEAL_D, TEAL_L, WHITE, W, H, back, circle, clamp, col, ellipse, eo, hx, lerp,
                 line, paint, person, prog, rrect, text, text_w)


# ---------------------------------------------------------------- timing
def eff(s):
    return sum(1 for ch in s if ("一" <= ch <= "鿿") or ch.isalnum())


def ph(sh, ci, phrase):
    """Local time at which `phrase` starts inside cue `ci` (proportional to spoken characters)."""
    c = sh["cues"][ci]
    txt = c["text"]
    i = txt.index(phrase)
    return c["start"] + c["dur"] * eff(txt[:i]) / max(1, eff(txt))


def cs(sh, ci):
    return sh["cues"][ci]["start"]


def ce(sh, ci):
    c = sh["cues"][ci]
    return c["start"] + c["dur"]


def pop(t, t0, d=0.45):
    if t < t0:
        return 0.0
    return max(0.0, back(prog(t, t0, d)))


def fin(t, t0, d=0.4):
    return eo(prog(t, t0, d))


def bump(t, t0, d=0.5):
    """0 -> 1 -> 0 over d seconds starting at t0."""
    p = prog(t, t0, d)
    return math.sin(PI * p) if 0 < p < 1 else 0.0


def mix(c1, c2, p):
    return tuple(lerp(c1[i], c2[i], p) for i in range(3))


# ---------------------------------------------------------------- context helpers
@contextmanager
def alpha(ctx, a):
    a = clamp(a)
    if a >= 0.999:
        yield
        return
    ctx.push_group()
    try:
        yield
    finally:
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(a)


@contextmanager
def at(ctx, x, y, s=1.0, rot=0.0, sx=None):
    ctx.save()
    ctx.translate(x, y)
    if rot:
        ctx.rotate(rot)
    ctx.scale(s if sx is None else sx, s)
    try:
        yield
    finally:
        ctx.restore()


@contextmanager
def zoom(ctx, fx, fy, s, sy=None):
    ctx.save()
    ctx.translate(fx, fy)
    ctx.scale(max(s, 1e-4), max(s if sy is None else sy, 1e-4))
    ctx.translate(-fx, -fy)
    try:
        yield
    finally:
        ctx.restore()


# ---------------------------------------------------------------- backgrounds and titles
def bg(ctx, t, base=CREAM):
    col(ctx, base)
    ctx.paint()
    for x, y, r, c, sp in [(140, 170, 230, TEAL_L, 0.30), (1820, 960, 270, ORANGE_L, 0.25),
                           (1790, 150, 150, TEAL_L, 0.20), (110, 990, 170, ORANGE_L, 0.35)]:
        circle(ctx, x + 22 * math.sin(t * sp), y + 16 * math.cos(t * sp * 1.3), r, c, 0.55)


def head(ctx, t, items, y=118, size=60, color=INK, accent=TEAL):
    """Sequence of headlines: items = [(t0, text[, size])]; each replaces the previous one."""
    for i, it in enumerate(items):
        t0, s = it[0], it[1]
        sz = it[2] if len(it) > 2 else size
        t_out = items[i + 1][0] if i + 1 < len(items) else None
        if t < t0 or (t_out is not None and t > t_out + 0.3):
            continue
        a_in = eo(prog(t, t0, 0.5))
        a_out = 1.0 if t_out is None else 1 - prog(t, t_out - 0.05, 0.3)
        a = a_in * a_out
        if a <= 0:
            continue
        yy = y - 22 * (1 - a_in)
        w = text(ctx, s, 960, yy, sz, color, True, a=a)
        rrect(ctx, 960 - w * 0.17 * a_in, yy + sz * 0.72, w * 0.34 * a_in, 8, 4, accent, a)


def bubble(ctx, x, y, w, h, tx, ty, fill=WHITE, stroke=INK2, a=1.0, lw=3, r=24, tail=22):
    """Rounded speech bubble whose tail points at (tx, ty)."""
    if ty >= y + h:
        edge = "bottom"
    elif ty <= y:
        edge = "top"
    elif tx <= x:
        edge = "left"
    else:
        edge = "right"
    if edge in ("top", "bottom"):
        c = clamp(tx, x + r + tail, x + w - r - tail)
    else:
        c = clamp(ty, y + r + tail, y + h - r - tail)
    ctx.new_sub_path()
    ctx.arc(x + r, y + r, r, PI, 1.5 * PI)
    if edge == "top":
        ctx.line_to(c - tail, y)
        ctx.line_to(tx, ty)
        ctx.line_to(c + tail, y)
    ctx.arc(x + w - r, y + r, r, 1.5 * PI, 2 * PI)
    if edge == "right":
        ctx.line_to(x + w, c - tail)
        ctx.line_to(tx, ty)
        ctx.line_to(x + w, c + tail)
    ctx.arc(x + w - r, y + h - r, r, 0, 0.5 * PI)
    if edge == "bottom":
        ctx.line_to(c + tail, y + h)
        ctx.line_to(tx, ty)
        ctx.line_to(c - tail, y + h)
    ctx.arc(x + r, y + h - r, r, 0.5 * PI, PI)
    if edge == "left":
        ctx.line_to(x, c + tail)
        ctx.line_to(tx, ty)
        ctx.line_to(x, c - tail)
    ctx.close_path()
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    paint(ctx, fill, a, stroke, lw)


def figure(ctx, x, y, s=1.0, shirt=BLUE, upper=False, arm_l=None, arm_r=None, badge=False, skip_r=False, **kw):
    """person() plus arms; arm targets are absolute points (None = hanging)."""
    info = person(ctx, x, y, s, shirt=shirt, upper=upper, **kw)
    by = -100 if upper else -158
    shy = y + (by + 26) * s
    sleeve = kw.get("vest") or shirt
    for side, tgt in ((-1, arm_l), (1, arm_r)):
        if side == 1 and skip_r:
            continue
        sx_ = x + side * 33 * s
        if tgt is None:
            tgt = (x + side * 50 * s, y + (by + 88) * s)
        line(ctx, sx_, shy, tgt[0], tgt[1], sleeve, 17 * s)
        circle(ctx, tgt[0], tgt[1], 10 * s, SKIN)
    if badge:
        rrect(ctx, x + 6 * s, y + (by + 34) * s, 22 * s, 15 * s, 3 * s, WHITE, 1, GRAY_L, 1.5)
    return info


def sparkle(ctx, x, y, r, c=ORANGE, a=1.0):
    k = 0.22
    ctx.move_to(x, y - r)
    ctx.line_to(x + r * k, y - r * k)
    ctx.line_to(x + r, y)
    ctx.line_to(x + r * k, y + r * k)
    ctx.line_to(x, y + r)
    ctx.line_to(x - r * k, y + r * k)
    ctx.line_to(x - r, y)
    ctx.line_to(x - r * k, y - r * k)
    ctx.close_path()
    col(ctx, c, a)
    ctx.fill()


def megaphone(ctx, x, y, s=1.0, ang=PI):
    with at(ctx, x, y, s, ang):
        ctx.move_to(4, -11)
        ctx.line_to(66, -32)
        ctx.line_to(66, 32)
        ctx.line_to(4, 11)
        ctx.close_path()
        paint(ctx, WHITE, 1, INK2, 3)
        rrect(ctx, 60, -34, 12, 68, 5, ORANGE)
        rrect(ctx, -12, -11, 18, 22, 5, INK2)


def teabox(ctx, x, y, s=1.0):
    with at(ctx, x, y, s):
        rrect(ctx, -42, -52, 84, 104, 10, hx("4E8A55"))
        rrect(ctx, -32, -34, 64, 50, 8, ORANGE_L)
        text(ctx, "茶", 0, -9, 34, hx("2F5E36"), True)
        with at(ctx, 0, 32, 1.0, -0.5):
            ellipse(ctx, 0, 0, 16, 7, hx("9ED38A"))


def shop(ctx, x, y, s=1.0):
    with at(ctx, x, y, s):
        rrect(ctx, -90, -40, 180, 120, 8, WHITE, 1, INK2, 3)
        n = 6
        for i in range(n):
            ctx.rectangle(-100 + i * 200 / n, -84, 200 / n, 44)
            col(ctx, ORANGE if i % 2 == 0 else WHITE)
            ctx.fill()
        for i in range(n):
            circle(ctx, -100 + 200 / n * (i + 0.5), -40, 100 / n, ORANGE if i % 2 == 0 else WHITE)
        rrect(ctx, -100, -84, 200, 44, 6, None, 1, INK2, 3)
        rrect(ctx, -24, 12, 48, 68, 6, TEAL_L, 1, INK2, 2)
        rrect(ctx, -76, 6, 40, 34, 4, BLUE_LL, 1, INK2, 2)
        rrect(ctx, 36, 6, 40, 34, 4, BLUE_LL, 1, INK2, 2)


def cylinder(ctx, x, y, w, h, c=TEAL, top=None, a=1.0):
    rx, ry = w / 2, w * 0.14
    top = top or mix(c, WHITE, 0.45)
    ellipse(ctx, x, y + h / 2, rx, ry, c, a)
    ctx.rectangle(x - rx, y - h / 2, w, h)
    col(ctx, c, a)
    ctx.fill()
    for k in (1 / 3, 2 / 3):
        ctx.save()
        ctx.translate(x, y - h / 2 + h * k)
        ctx.scale(rx, ry)
        ctx.new_sub_path()
        ctx.arc(0, 0, 1, 0.05 * PI, 0.95 * PI)
        ctx.restore()
        col(ctx, WHITE, 0.35 * a)
        ctx.set_line_width(4)
        ctx.stroke()
    ellipse(ctx, x, y - h / 2, rx, ry, top, a)


def rich(ctx, segs, x, y, size, bold=True, a=1.0):
    """Centered single line made of (text, color) segments; returns [(x_left, width)] per segment."""
    widths = [text_w(ctx, s, size, bold) for s, _ in segs]
    xx = x - sum(widths) / 2
    out = []
    for (s, c), w in zip(segs, widths):
        text(ctx, s, xx, y, size, c, bold, "left", a)
        out.append((xx, w))
        xx += w
    return out


def dashed(ctx, x1, y1, x2, y2, c, lw=4, a=1.0, dash=(14, 12), offset=0.0):
    ctx.save()
    ctx.set_dash(list(dash), offset)
    line(ctx, x1, y1, x2, y2, c, lw, a)
    ctx.restore()


def qmark(ctx, x, y, r, a=1.0, c=ORANGE):
    circle(ctx, x, y, r, WHITE, a, INK2, 3)
    text(ctx, "?", x, y - 2, r * 1.15, c, True, a=a)


__all__ = [n for n in dir() if not n.startswith("_")]
