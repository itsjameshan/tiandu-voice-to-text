"""Drawing primitives and characters for the explainer video (pycairo, 1920x1080)."""
import math

import cairo

W, H = 1920, 1080
FONT = "Noto Sans CJK SC"
PI = math.pi


def hx(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) / 255 for i in (0, 2, 4))


CREAM = hx("F6F2E9")
SKY = hx("DDEFF6")
TEAL = hx("1D9E75")
TEAL_D = hx("0F6E56")
TEAL_L = hx("E1F5EE")
TEAL_M = hx("9FE1CB")
ORANGE = hx("EF9F27")
ORANGE_D = hx("BA7517")
ORANGE_L = hx("FAEEDA")
EARTH = hx("C2593A")
EARTH_L = hx("E8A07C")
HILL1 = hx("8CC07A")
HILL2 = hx("6AA65E")
HILL3 = hx("4F8C4A")
INK = hx("2C2C2A")
INK2 = hx("5F5E5A")
GRAY = hx("888780")
GRAY_L = hx("D3D1C7")
GRAY_LL = hx("ECEAE3")
WHITE = (1, 1, 1)
BLUE = hx("378ADD")
BLUE_L = hx("B5D4F4")
BLUE_LL = hx("E6F1FB")
RED = hx("E24B4A")
GOOD = hx("3B9E4A")
HL = hx("FFE08A")
SKIN = hx("F2C8A2")
SCREEN = hx("173B36")
PURPLE = hx("7F77DD")


# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def prog(t, t0, d):
    if d <= 0:
        return 1.0 if t >= t0 else 0.0
    return clamp((t - t0) / d)


def eo(p):
    return 1 - (1 - p) ** 3


def eio(p):
    return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def back(p, s=1.70158):
    p -= 1
    return 1 + (s + 1) * p ** 3 + s * p ** 2


def lerp(a, b, p):
    return a + (b - a) * p


# ---------------------------------------------------------------- basic shapes
def col(ctx, c, a=1.0):
    ctx.set_source_rgba(c[0], c[1], c[2], a)


def rrect_path(ctx, x, y, w, h, r):
    r = max(0.0, min(r, w / 2, h / 2))
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -PI / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, PI / 2)
    ctx.arc(x + r, y + h - r, r, PI / 2, PI)
    ctx.arc(x + r, y + r, r, PI, 3 * PI / 2)
    ctx.close_path()


def paint(ctx, fill=None, a=1.0, stroke=None, lw=3, sa=1.0):
    if fill is not None:
        col(ctx, fill, a)
        if stroke is not None:
            ctx.fill_preserve()
        else:
            ctx.fill()
    if stroke is not None:
        col(ctx, stroke, sa * (a if fill is None else 1))
        ctx.set_line_width(lw)
        ctx.stroke()


def rrect(ctx, x, y, w, h, r, fill=None, a=1.0, stroke=None, lw=3):
    rrect_path(ctx, x, y, w, h, r)
    paint(ctx, fill, a, stroke, lw)


def circle(ctx, x, y, r, fill=None, a=1.0, stroke=None, lw=3):
    ctx.new_sub_path()
    ctx.arc(x, y, r, 0, 2 * PI)
    paint(ctx, fill, a, stroke, lw)


def ellipse(ctx, x, y, rx, ry, fill=None, a=1.0, stroke=None, lw=3):
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(rx, ry)
    ctx.new_sub_path()
    ctx.arc(0, 0, 1, 0, 2 * PI)
    ctx.restore()
    paint(ctx, fill, a, stroke, lw)


def line(ctx, x1, y1, x2, y2, c, lw=4, a=1.0, cap=cairo.LINE_CAP_ROUND):
    ctx.set_line_cap(cap)
    ctx.move_to(x1, y1)
    ctx.line_to(x2, y2)
    col(ctx, c, a)
    ctx.set_line_width(lw)
    ctx.stroke()


def poly(ctx, pts, c, lw=4, a=1.0, close=False, fill=None):
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    if close:
        ctx.close_path()
    if fill is not None:
        col(ctx, fill, a)
        ctx.fill_preserve()
    col(ctx, c, a)
    ctx.set_line_width(lw)
    ctx.stroke()


def arrow(ctx, x1, y1, x2, y2, c, lw=5, a=1.0, head=18):
    line(ctx, x1, y1, x2, y2, c, lw, a)
    ang = math.atan2(y2 - y1, x2 - x1)
    pts = [(x2, y2),
           (x2 - head * math.cos(ang - 0.45), y2 - head * math.sin(ang - 0.45)),
           (x2 - head * math.cos(ang + 0.45), y2 - head * math.sin(ang + 0.45))]
    ctx.move_to(*pts[0])
    ctx.line_to(*pts[1])
    ctx.line_to(*pts[2])
    ctx.close_path()
    col(ctx, c, a)
    ctx.fill()


# ---------------------------------------------------------------- text
def font(ctx, size, bold=False):
    ctx.select_font_face(FONT, cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)


def text_w(ctx, s, size, bold=False):
    font(ctx, size, bold)
    return ctx.text_extents(s).x_advance


def text(ctx, s, x, y, size, c=INK, bold=False, align="center", a=1.0):
    """Draw text with y as the visual centre line."""
    font(ctx, size, bold)
    w = ctx.text_extents(s).x_advance
    x0 = x - w / 2 if align == "center" else (x - w if align == "right" else x)
    col(ctx, c, a)
    ctx.move_to(x0, y + size * 0.36)
    ctx.show_text(s)
    return w


def pill(ctx, s, x, y, size, bg, fg=WHITE, bold=True, a=1.0, padx=None, h=None, stroke=None, lw=3):
    w = text_w(ctx, s, size, bold)
    padx = size * 0.7 if padx is None else padx
    h = size * 1.7 if h is None else h
    rrect(ctx, x - w / 2 - padx, y - h / 2, w + 2 * padx, h, h / 2, bg, a, stroke, lw)
    text(ctx, s, x, y, size, fg, bold, a=a)
    return w + 2 * padx


def headline(ctx, s, t, t0, y=118, size=62, c=INK, accent=TEAL, sub=None):
    """Big title that slides down and fades in at t0."""
    p = eo(prog(t, t0, 0.5))
    if p <= 0:
        return
    yy = y - 24 * (1 - p)
    w = text(ctx, s, 960, yy, size, c, True, a=p)
    rrect(ctx, 960 - w * 0.18 * p, yy + size * 0.72, w * 0.36 * p, 8, 4, accent, p)
    if sub:
        text(ctx, sub, 960, yy + size * 1.35, size * 0.5, INK2, False, a=p)


def subtitle(ctx, s):
    s = s.rstrip("，、；：。")
    size = 44 if len(s) <= 30 else 40
    w = text_w(ctx, s, size, False)
    rrect(ctx, 960 - w / 2 - 34, 986 - 38, w + 68, 76, 18, hx("1E2422"), 0.72)
    text(ctx, s, 960, 986, size, WHITE, False)


def corner_label(ctx):
    pill(ctx, "AI 合成配音", 1790, 42, 20, INK, WHITE, False, a=0.45, padx=16, h=36)


# ---------------------------------------------------------------- props
def ai_ting(ctx, x, y, s=1.0, t=0.0, talk=0.0, wave=0.0, a=1.0, look=0.0):
    """Original narrator robot: round teal body, screen face with a sound wave, headphones."""
    ctx.save()
    ctx.translate(x, y + 4 * math.sin(t * 2.2))
    ctx.scale(s, s)
    if a < 1:
        ctx.push_group()
    ellipse(ctx, 0, 100, 62, 10, INK, 0.10)
    circle(ctx, -82, 26, 18, TEAL_D)
    ra = -0.6 - wave * (0.9 + 0.35 * math.sin(t * 9))
    circle(ctx, 82 + 18 * math.sin(ra + 0.6), 26 + 60 * math.sin(ra) * 0.8, 18, TEAL_D)
    rrect(ctx, -44, 64, 30, 30, 10, TEAL_D)
    rrect(ctx, 14, 64, 30, 30, 10, TEAL_D)
    circle(ctx, 0, 0, 82, TEAL)
    ellipse(ctx, -26, -40, 26, 14, WHITE, 0.18)
    rrect(ctx, -58 + look * 6, -38, 116, 66, 20, SCREEN)
    pts = []
    for i in range(23):
        u = i / 22
        xx = -46 + 92 * u + look * 6
        amp = 3 + talk * 15 * abs(math.sin(t * 11 + i * 0.75)) * math.sin(PI * u)
        pts.append((xx, -5 + amp * math.sin(i * 1.9 + t * 14)))
    poly(ctx, pts, TEAL_M, 4)
    ctx.new_sub_path()
    ctx.arc(0, 0, 96, PI * 1.12, PI * 1.88)
    col(ctx, INK)
    ctx.set_line_width(10)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.stroke()
    rrect(ctx, -106, -26, 26, 50, 10, INK)
    rrect(ctx, 80, -26, 26, 50, 10, INK)
    if a < 1:
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(a)
    ctx.restore()


def person(ctx, x, y, s=1.0, shirt=BLUE, hair=INK, hat=None, vest=None, headphones=False,
           upper=False, skin=SKIN, a=1.0, look=0.0, mouth=0.0):
    """Simple flat person; (x, y) is the feet line (or the seat line when upper=True)."""
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(s, s)
    if a < 1:
        ctx.push_group()
    if not upper:
        rrect(ctx, -24, -62, 20, 62, 8, INK2)
        rrect(ctx, 4, -62, 20, 62, 8, INK2)
        body_y = -158
    else:
        body_y = -100
    rrect(ctx, -40, body_y, 80, 100, 28, shirt)
    if vest is not None:
        rrect(ctx, -40, body_y, 80, 100, 28, vest)
        ctx.move_to(-15, body_y + 2)
        ctx.line_to(15, body_y + 2)
        ctx.line_to(9, body_y + 96)
        ctx.line_to(-9, body_y + 96)
        ctx.close_path()
        col(ctx, shirt)
        ctx.fill()
    hy = body_y - 34
    circle(ctx, 0, hy, 32, skin)
    ctx.new_sub_path()
    ctx.arc(0, hy - 2, 33, PI * 1.02, PI * 1.98)
    ctx.close_path()
    col(ctx, hair)
    ctx.fill()
    circle(ctx, -11 + look * 6, hy + 4, 3.4, INK)
    circle(ctx, 11 + look * 6, hy + 4, 3.4, INK)
    if mouth > 0:
        ellipse(ctx, look * 6, hy + 17, 6, 3 + 4 * mouth, INK2)
    else:
        ctx.new_sub_path()
        ctx.arc(look * 6, hy + 13, 7, 0.2 * PI, 0.8 * PI)
        col(ctx, INK2)
        ctx.set_line_width(2.5)
        ctx.stroke()
    if hat is not None:
        ellipse(ctx, 0, hy - 24, 46, 9, hat)
        rrect(ctx, -28, hy - 50, 56, 30, 12, hat)
    if headphones:
        ctx.new_sub_path()
        ctx.arc(0, hy, 38, PI * 1.08, PI * 1.92)
        col(ctx, INK)
        ctx.set_line_width(7)
        ctx.stroke()
        rrect(ctx, -44, hy - 8, 14, 26, 6, INK)
        rrect(ctx, 30, hy - 8, 14, 26, 6, INK)
    if a < 1:
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(a)
    ctx.restore()
    return {"hand_l": (x - 40 * s, y + (body_y + 40) * s), "hand_r": (x + 40 * s, y + (body_y + 40) * s),
            "head": (x, y + hy * s)}


def arm(ctx, x1, y1, x2, y2, c, s=1.0):
    line(ctx, x1, y1, x2, y2, c, 16 * s)
    circle(ctx, x2, y2, 10 * s, SKIN)


def bus(ctx, x, y, s=1.0, rot=0.0):
    """Side view of a tour bus facing right; (x, y) = bottom-left on the road."""
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(s, s)
    rrect(ctx, 0, -220, 600, 175, 28, WHITE, 1, INK2, 3)
    ctx.rectangle(0, -120, 600, 26)
    col(ctx, TEAL)
    ctx.fill()
    for i in range(6):
        rrect(ctx, 26 + i * 78, -198, 64, 58, 10, BLUE_L)
    rrect(ctx, 520, -198, 62, 74, 12, BLUE_L)
    rrect(ctx, 466, -192, 48, 138, 8, GRAY_LL, 1, INK2, 2)
    rrect(ctx, 472, -184, 36, 56, 5, BLUE_L)
    rrect(ctx, 472, -120, 36, 56, 5, BLUE_L, 0.55)
    line(ctx, 490, -184, 490, -62, INK2, 2)
    circle(ctx, 588, -72, 9, ORANGE)
    for wx in (120, 470):
        circle(ctx, wx, -44, 38, INK)
        circle(ctx, wx, -44, 16, GRAY_L)
        for k in range(3):
            ang = rot + k * 2 * PI / 3
            line(ctx, wx, -44, wx + 28 * math.cos(ang), -44 + 28 * math.sin(ang), INK2, 4)
    ctx.restore()


def phone(ctx, x, y, s=1.0, rec=0.0, t=0.0, a=1.0):
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(s, s)
    rrect(ctx, -36, -66, 72, 132, 14, INK, a)
    rrect(ctx, -29, -56, 58, 108, 7, hx("2E4A46"), a)
    if rec > 0:
        circle(ctx, 0, -18, 14, RED, a * rec * (0.75 + 0.25 * math.sin(t * 6)))
        for i in range(7):
            hgt = 6 + 18 * abs(math.sin(t * 8 + i))
            rrect(ctx, -21 + i * 7, 22 - hgt / 2, 4, hgt, 2, TEAL_M, a * rec)
    ctx.restore()


def wave_rings(ctx, x, y, t, a=1.0, color=TEAL, n=3, spread=140):
    for i in range(n):
        ph = ((t * 0.9 + i / n) % 1.0)
        r = 40 + spread * ph
        ctx.new_sub_path()
        ctx.arc(x, y, r, -0.9, 0.9)
        col(ctx, color, a * (1 - ph))
        ctx.set_line_width(6)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.stroke()
        ctx.new_sub_path()
        ctx.arc(x, y, r, PI - 0.9, PI + 0.9)
        ctx.stroke()


def audio_file(ctx, x, y, s=1.0, c=TEAL, a=1.0, rot=0.0):
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(rot)
    ctx.scale(s, s)
    ctx.move_to(-40, -52)
    ctx.line_to(18, -52)
    ctx.line_to(40, -30)
    ctx.line_to(40, 52)
    ctx.line_to(-40, 52)
    ctx.close_path()
    paint(ctx, WHITE, a, INK2, 3)
    poly(ctx, [(18, -52), (18, -30), (40, -30)], INK2, 3, a)
    for i, hgt in enumerate([14, 30, 44, 24, 36, 16]):
        rrect(ctx, -28 + i * 10, 14 - hgt / 2, 6, hgt, 3, c, a)
    ctx.restore()


def envelope(ctx, x, y, s=1.0, a=1.0, rot=0.0):
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(rot)
    ctx.scale(s, s)
    rrect(ctx, -70, -46, 140, 92, 10, ORANGE_L, a, ORANGE_D, 3)
    poly(ctx, [(-66, -42), (0, 10), (66, -42)], ORANGE_D, 3, a)
    ctx.restore()


def doc_icon(ctx, x, y, s=1.0, c=TEAL, a=1.0):
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(s, s)
    rrect(ctx, -34, -44, 68, 88, 8, WHITE, a, INK2, 3)
    for i in range(4):
        rrect(ctx, -22, -28 + i * 16, 44 - (i % 2) * 12, 6, 3, c, a)
    ctx.restore()


def calendar(ctx, x, y, big, small, a=1.0, s=1.0):
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(s, s)
    rrect(ctx, -110, -120, 220, 240, 18, WHITE, a, INK2, 3)
    rrect(ctx, -110, -120, 220, 64, 18, ORANGE, a)
    ctx.rectangle(-110, -80, 220, 24)
    col(ctx, ORANGE, a)
    ctx.fill()
    circle(ctx, -55, -120, 10, INK2, a)
    circle(ctx, 55, -120, 10, INK2, a)
    text(ctx, big, 0, 18, 92, INK, True, a=a)
    text(ctx, small, 0, 92, 28, INK2, False, a=a)
    ctx.restore()


def clock(ctx, x, y, r, ang):
    circle(ctx, x, y, r, WHITE, 1, INK2, 5)
    for k in range(12):
        aa = k * PI / 6
        line(ctx, x + (r - 14) * math.cos(aa), y + (r - 14) * math.sin(aa),
             x + (r - 6) * math.cos(aa), y + (r - 6) * math.sin(aa), GRAY, 3)
    line(ctx, x, y, x + r * 0.5 * math.cos(ang / 12 - PI / 2), y + r * 0.5 * math.sin(ang / 12 - PI / 2), INK, 7)
    line(ctx, x, y, x + r * 0.78 * math.cos(ang - PI / 2), y + r * 0.78 * math.sin(ang - PI / 2), ORANGE_D, 5)
    circle(ctx, x, y, 7, INK)


def magnifier(ctx, x, y, s=1.0, a=1.0):
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(s, s)
    circle(ctx, 0, 0, 48, WHITE, 0.35 * a)
    circle(ctx, 0, 0, 48, None, a, INK, 10)
    line(ctx, 34, 34, 78, 78, INK, 16, a)
    ctx.restore()


def check(ctx, x, y, s=1.0, c=GOOD, a=1.0, p=1.0):
    pts = [(-30, 0), (-8, 24), (34, -26)]
    pts = [(x + px * s, y + py * s) for px, py in pts]
    if p < 1:
        if p < 0.4:
            q = p / 0.4
            pts = [pts[0], (lerp(pts[0][0], pts[1][0], q), lerp(pts[0][1], pts[1][1], q))]
        else:
            q = (p - 0.4) / 0.6
            pts = [pts[0], pts[1], (lerp(pts[1][0], pts[2][0], q), lerp(pts[1][1], pts[2][1], q))]
    poly(ctx, pts, c, 12 * s, a)


def cross(ctx, x, y, s=1.0, c=RED, a=1.0):
    line(ctx, x - 30 * s, y - 30 * s, x + 30 * s, y + 30 * s, c, 13 * s, a)
    line(ctx, x + 30 * s, y - 30 * s, x - 30 * s, y + 30 * s, c, 13 * s, a)


def cloud(ctx, x, y, s=1.0, c=BLUE_L, a=1.0):
    for cx, cy, r in [(-70, 20, 52), (-10, -18, 70), (62, 6, 58), (110, 34, 40), (0, 40, 56)]:
        circle(ctx, x + cx * s, y + cy * s, r * s, c, a)
    rrect(ctx, x - 120 * s, y + 20 * s, 270 * s, 70 * s, 35 * s, c, a)


def lock(ctx, x, y, s=1.0, a=1.0):
    ctx.new_sub_path()
    ctx.arc(x, y - 18 * s, 26 * s, PI, 2 * PI)
    col(ctx, INK2, a)
    ctx.set_line_width(10 * s)
    ctx.stroke()
    line(ctx, x - 26 * s, y - 18 * s, x - 26 * s, y, INK2, 10 * s, a)
    line(ctx, x + 26 * s, y - 18 * s, x + 26 * s, y, INK2, 10 * s, a)
    rrect(ctx, x - 40 * s, y - 4 * s, 80 * s, 62 * s, 12 * s, ORANGE, a)
    circle(ctx, x, y + 24 * s, 8 * s, INK2, a)


def monitor(ctx, x, y, s=1.0, a=1.0, glow=0.0):
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(s, s)
    rrect(ctx, -150, -110, 300, 190, 16, INK, a)
    rrect(ctx, -136, -96, 272, 162, 8, TEAL_L, a)
    for i in range(4):
        rrect(ctx, -110, -70 + i * 34, 200 - (i % 2) * 70, 14, 7, TEAL if i % 2 == 0 else TEAL_M, a)
    rrect(ctx, -20, 80, 40, 40, 4, INK2, a)
    rrect(ctx, -80, 116, 160, 18, 9, INK2, a)
    if glow > 0:
        rrect(ctx, -160, -120, 320, 210, 22, None, glow, GOOD, 8)
    ctx.restore()


def station(ctx, x, y, w, h, num, title, badge, active=0.0, a=1.0, title2=None):
    fill = tuple(lerp(WHITE[i], TEAL_L[i], active) for i in range(3))
    stroke = tuple(lerp(GRAY_L[i], TEAL[i], active) for i in range(3))
    rrect(ctx, x, y, w, h, 20, fill, a, stroke, 3 + 3 * active)
    circle(ctx, x + 38, y + 38, 22, TEAL if active > 0.5 else GRAY, a)
    text(ctx, str(num), x + 38, y + 38, 26, WHITE, True, a=a)
    if title2:
        text(ctx, title, x + w / 2 + 16, y + 44, 30, INK, True, a=a)
        text(ctx, title2, x + w / 2 + 16, y + 82, 24, INK2, False, a=a)
    else:
        text(ctx, title, x + w / 2 + 16, y + 56, 30, INK, True, a=a)
    bg = ORANGE if badge.startswith("第") else (TEAL_D if badge == "教师模板" else INK2)
    pill(ctx, badge, x + w / 2, y + h - 30, 22, bg, WHITE, True, a=a, padx=16, h=40)


def draft_card(ctx, x, y, w, rows=4.0, stamp=0.0, t=0.0, a=1.0, hl_pulse=0.0, review=0.0, foot=0):
    """The 'core check draft' card: timestamps, speakers, suspected segment, amount."""
    h = 92 + 4 * 78 + 24 + foot
    rrect(ctx, x + 10, y + 12, w, h, 22, INK, 0.08 * a)
    rrect(ctx, x, y, w, h, 22, WHITE, a, GRAY_L, 3)
    rrect(ctx, x, y, w, 70, 22, TEAL, a)
    ctx.rectangle(x, y + 40, w, 30)
    col(ctx, TEAL, a)
    ctx.fill()
    text(ctx, "核查初稿", x + 32, y + 35, 32, WHITE, True, "left", a)
    text(ctx, "自动生成 · 待人工复核", x + w - 28, y + 35, 22, TEAL_L, False, "right", a)
    data = [("00:12", "导游", TEAL, [260, 120], False, None),
            ("00:18", "游客", ORANGE, [200], False, None),
            ("00:25", "导游", TEAL, [230, 150], True, None),
            ("00:31", "游客", ORANGE, [120], False, "2800 元")]
    for i, (ts_, who, c, bars, hl, amt) in enumerate(data):
        ra = clamp(rows - i) * a
        if ra <= 0:
            continue
        ry = y + 92 + i * 78
        if hl:
            rrect(ctx, x + 16, ry - 6, w - 32, 64, 12, HL, ra * (0.75 + 0.25 * hl_pulse))
        text(ctx, ts_, x + 40, ry + 26, 24, GRAY, False, "left", ra)
        pill(ctx, who, x + 168, ry + 26, 22, c, WHITE, True, ra, padx=14, h=40)
        bx = x + 222
        for bw in bars:
            rrect(ctx, bx, ry + 18, bw, 16, 8, GRAY_L, ra)
            bx += bw + 14
        if hl:
            pill(ctx, "疑似", x + w - 80, ry + 26, 22, WHITE, ORANGE_D, True, ra, padx=14, h=40,
                 stroke=ORANGE_D, lw=3)
        if amt:
            aw = text_w(ctx, amt, 26, True)
            ax = bx + 30 + aw / 2
            text(ctx, amt, ax, ry + 26, 26, ORANGE_D, True, a=ra)
            ellipse(ctx, ax, ry + 26, aw / 2 + 22, 30, None, ra, ORANGE, 4)
        if review > 0 and hl:
            check(ctx, x + w - 150, ry + 28, 0.8, GOOD, ra, review)
    if stamp > 0:
        p = eo(clamp(stamp))
        sc = lerp(1.7, 1.0, p)
        ctx.save()
        if foot > 0:
            ctx.translate(x + w - 170, y + h - foot / 2 - 8)
        else:
            ctx.translate(x + w - 150, y + h - 70)
        ctx.rotate(-0.12 if foot > 0 else -0.2)
        ctx.scale(sc, sc)
        sw = text_w(ctx, "疑似 · 待核查", 28, True)
        rrect(ctx, -sw / 2 - 20, -30, sw + 40, 60, 10, None, p * a, ORANGE_D, 5)
        text(ctx, "疑似 · 待核查", 0, 0, 28, ORANGE_D, True, a=p * a)
        ctx.restore()
    return h


def tag(ctx, s, x, y, size=28, bg=WHITE, fg=INK, stroke=TEAL, a=1.0):
    return pill(ctx, s, x, y, size, bg, fg, True, a, padx=18, h=size * 1.75, stroke=stroke, lw=3)


def hills(ctx, t=0.0, scroll=0.0):
    for (bx, by, rx, ry, c) in [(200, 820, 520, 240, HILL1), (900, 840, 640, 260, HILL2),
                                 (1650, 830, 560, 250, HILL1), (2300, 840, 600, 260, HILL2)]:
        xx = ((bx - scroll * 0.3) % 2600) - 300
        ellipse(ctx, xx, by, rx, ry, c)


def road(ctx, y=840, h=120, scroll=0.0):
    ctx.rectangle(0, y, W, h)
    col(ctx, EARTH)
    ctx.fill()
    ctx.rectangle(0, y, W, 10)
    col(ctx, EARTH_L)
    ctx.fill()
    for i in range(-1, 12):
        xx = i * 200 - (scroll % 200)
        rrect(ctx, xx, y + h / 2 - 5, 110, 10, 5, ORANGE_L, 0.8)
