"""Shots 1-6."""
import math

import cairo

from gfx import *  # noqa: F401,F403
from scene_util import *  # noqa: F401,F403

# ================================================================ shot 1 — on the bus
BUS_X, BUS_Y, BUS_S = 520, 892, 1.15
WIN_C = (BUS_X + (26 + 78 + 32) * BUS_S, BUS_Y + (-198 + 29) * BUS_S)
SEAT_Y, PAX_S = 770, 1.45
TOURIST_X = 330


def mountains(ctx, t):
    off = (t * 30) % 1200
    pts = [(0, 650), (140, 470), (300, 560), (470, 410), (650, 560), (820, 480), (1000, 590), (1200, 650)]
    for k in range(3):
        bx = k * 1200 - off
        ctx.move_to(bx + pts[0][0], pts[0][1])
        for px, py in pts[1:]:
            ctx.line_to(bx + px, py)
        ctx.line_to(bx + 1200, 800)
        ctx.line_to(bx, 800)
        ctx.close_path()
        col(ctx, hx("A9CDBB"))
        ctx.fill()


def exterior(ctx, t):
    g = cairo.LinearGradient(0, 0, 0, 800)
    g.add_color_stop_rgb(0, *hx("BCE2F2"))
    g.add_color_stop_rgb(1, *hx("EEF7F2"))
    ctx.rectangle(0, 0, W, H)
    ctx.set_source(g)
    ctx.fill()
    circle(ctx, 1600, 170, 100, hx("FFD47E"), 0.25)
    circle(ctx, 1600, 170, 66, hx("FFD47E"))
    for x0, y0, s in [(320, 170, 0.55), (1080, 115, 0.45), (1900, 250, 0.4)]:
        xx = (x0 - t * 25) % 2300 - 200
        cloud(ctx, xx, y0, s, WHITE, 0.95)
    mountains(ctx, t)
    hills(ctx, t, scroll=t * 260)
    ctx.rectangle(0, 780, W, 300)
    col(ctx, HILL3)
    ctx.fill()
    road(ctx, y=800, h=112, scroll=t * 520)
    sx = 2050 - t * 520
    if -200 < sx < 2150:
        rrect(ctx, sx - 8, 560, 16, 240, 6, GRAY)
        rrect(ctx, sx - 120, 470, 240, 120, 14, hx("2E7D5B"), 1, WHITE, 5)
        text(ctx, "大理", sx, 512, 46, WHITE, True)
        text(ctx, "120 km", sx, 562, 26, WHITE, False)
    by = BUS_Y + 2.5 * math.sin(t * 13)
    bus(ctx, BUS_X, by, BUS_S, rot=t * 9)
    for i in range(1, 6):
        cx = BUS_X + (26 + i * 78 + 32) * BUS_S
        cy = by + (-198 + 40) * BUS_S
        circle(ctx, cx, cy, 13 * BUS_S, SKIN)
        ctx.new_sub_path()
        ctx.arc(cx, cy - 2, 13.5 * BUS_S, PI, 2 * PI)
        col(ctx, INK if i % 2 else hx("6B4A2E"))
        ctx.fill()
    text(ctx, "旅游专线", BUS_X + 250 * BUS_S, by - 107 * BUS_S, 20, WHITE, True)


def seat_back(ctx, x, sy, s):
    rrect(ctx, x - 82 * s, sy - 205 * s, 164 * s, 215 * s, 30 * s, TEAL_D)
    rrect(ctx, x - 58 * s, sy - 198 * s, 116 * s, 36 * s, 14 * s, TEAL)


def seat_front(ctx, x, sy, s):
    rrect(ctx, x - 92 * s, sy - 8 * s, 184 * s, 44 * s, 16 * s, TEAL)
    leg_h = 862 - (sy + 36 * s)
    rrect(ctx, x - 70 * s, sy + 36 * s, 14 * s, leg_h, 4, INK2)
    rrect(ctx, x + 56 * s, sy + 36 * s, 14 * s, leg_h, 4, INK2)


def tourist_phone(t, sh):
    p = eo(prog(t, cs(sh, 1) + 0.15, 0.5))
    return TOURIST_X + 30 * PAX_S, lerp(SEAT_Y + 40, SEAT_Y - 62 * PAX_S, p)


def interior(ctx, t, sh, t_in, rec_t):
    col(ctx, hx("E8F1ED"))
    ctx.paint()
    ctx.rectangle(0, 0, W, 132)
    col(ctx, hx("D2E2DB"))
    ctx.fill()
    rrect(ctx, -20, 150, W + 40, 18, 9, hx("BFD0C9"))
    wins = [(90, 190), (540, 190), (990, 190), (1440, 190)]
    ctx.save()
    for wx, wy in wins:
        rrect_path(ctx, wx, wy, 390, 230, 30)
    ctx.clip()
    ctx.rectangle(0, 190, W, 230)
    col(ctx, hx("CDEAF4"))
    ctx.fill()
    for k in range(-1, 7):
        xx = k * 420 - ((t * 240) % 420)
        ellipse(ctx, xx, 455, 260, 120, HILL1 if k % 2 else HILL2)
    for k in range(-1, 6):
        xx = k * 520 - ((t * 700) % 520)
        rrect(ctx, xx, 300, 14, 140, 6, hx("6E8F62"))
        circle(ctx, xx + 7, 300, 34, HILL3)
    ctx.restore()
    for wx, wy in wins:
        rrect(ctx, wx, wy, 390, 230, 30, None, 1, hx("B5C9C0"), 10)
    ctx.rectangle(0, 860, W, 220)
    col(ctx, hx("D8D0BF"))
    ctx.fill()
    ctx.rectangle(0, 860, W, 8)
    col(ctx, hx("C2B9A6"))
    ctx.fill()

    # two passengers in the middle rows
    for x, shirt, hair in [(720, GOOD, hx("6B4A2E")), (1080, PURPLE, INK)]:
        seat_back(ctx, x, SEAT_Y, PAX_S)
        figure(ctx, x, SEAT_Y, PAX_S, shirt=shirt, hair=hair, upper=True, look=1.0)
        seat_front(ctx, x, SEAT_Y, PAX_S)

    # the tourist in the back row quietly records
    px, py = tourist_phone(t, sh)
    seat_back(ctx, TOURIST_X, SEAT_Y, PAX_S)
    figure(ctx, TOURIST_X, SEAT_Y, PAX_S, shirt=BLUE, hair=INK, upper=True, look=1.0, skip_r=True)
    ctx.save()
    ctx.rectangle(0, 0, W, SEAT_Y - 8 * PAX_S)
    ctx.clip()
    line(ctx, TOURIST_X + 33 * PAX_S, SEAT_Y - 74 * PAX_S, px - 6, py + 30, BLUE, 17 * PAX_S)
    phone(ctx, px, py, 0.75, rec=fin(t, rec_t, 0.15), t=t)
    circle(ctx, px - 8, py + 36, 10 * PAX_S, SKIN)
    ctx.restore()
    seat_front(ctx, TOURIST_X, SEAT_Y, PAX_S)
    if t >= rec_t:
        wave_rings(ctx, px, py, t, a=0.85 * fin(t, rec_t, 0.3), color=RED, n=3, spread=80)
        pr = pop(t, rec_t + 0.05)
        with zoom(ctx, px + 120, py - 70, pr):
            pill(ctx, "录音中", px + 120, py - 70, 26, RED, WHITE, True, padx=18, h=46)

    # the guide at the front
    gx = 1560
    talk = 0.5 + 0.5 * math.sin(t * 15)
    figure(ctx, gx, 878, PAX_S, shirt=EARTH, hat=ORANGE, look=-1.0, mouth=talk,
           arm_l=(gx - 102, 655), arm_r=(gx + 82, 724))
    megaphone(ctx, gx - 102, 655, 1.0, ang=PI + 0.3)
    teabox(ctx, gx + 98, 700, 0.9)
    pb = pop(t, t_in + 0.35)
    if pb > 0:
        with zoom(ctx, 1380, 470, pb):
            bubble(ctx, 1100, 290, 370, 170, 1440, 548)
            teabox(ctx, 1185, 375, 0.8)
            text(ctx, "特价好茶", 1350, 345, 42, INK, True)
            text(ctx, "¥ ¥ ¥", 1350, 408, 36, ORANGE_D, True)


def phone_close(ctx, t, sh):
    bg(ctx, t)
    rg = cairo.RadialGradient(960, 560, 60, 960, 560, 720)
    rg.add_color_stop_rgba(0, *TEAL_L, 1)
    rg.add_color_stop_rgba(1, *TEAL_L, 0)
    ctx.rectangle(0, 0, W, H)
    ctx.set_source(rg)
    ctx.fill()
    X, Y = 960, 580
    c2 = cs(sh, 2)
    # dashed paths to the question marks
    qs = [(520, 400), (1400, 360), (1420, 740)]
    for k, (qx, qy) in enumerate(qs):
        p = pop(t, c2 + 0.15 + 0.3 * k)
        if p > 0:
            fx = X - 175 if qx < X else X + 175
            dashed(ctx, fx, Y - 60 + 60 * k, qx, qy, ORANGE, 4, min(1, p), offset=-t * 40)
    rrect(ctx, X - 175 + 10, Y - 345 + 14, 350, 680, 48, INK, 0.12)
    rrect(ctx, X - 175, Y - 345, 350, 680, 48, INK)
    rrect(ctx, X - 156, Y - 318, 312, 626, 32, hx("1D2D2A"))
    rrect(ctx, X - 40, Y - 334, 80, 10, 5, hx("3A3A38"))
    circle(ctx, X - 58, Y - 252, 12, RED, 0.6 + 0.4 * math.sin(t * 6))
    text(ctx, "录音中", X + 10, Y - 252, 30, WHITE, True)
    el = 2 + 3598 * eio(prog(t, c2 + 0.3, 2.9))
    hh, mm, ss = int(el // 3600), int(el % 3600 // 60), int(el % 60)
    text(ctx, f"{hh:02d}:{mm:02d}:{ss:02d}", X, Y - 168, 64, WHITE, True)
    for i in range(24):
        bx = X - 134 + i * 11.6
        hgt = 10 + 74 * abs(math.sin(t * 7 + i * 0.7)) * (0.35 + 0.65 * abs(math.sin(i * 1.3 + t)))
        rrect(ctx, bx, Y + 10 - hgt / 2, 6, hgt, 3, TEAL_M)
    circle(ctx, X, Y + 205, 50, RED)
    rrect(ctx, X - 18, Y + 187, 36, 36, 7, WHITE)
    for k, (qx, qy) in enumerate(qs):
        p = pop(t, c2 + 0.15 + 0.3 * k)
        if p > 0:
            qmark(ctx, qx, qy + 6 * math.sin(t * 3 + k), 54 * p, a=min(1, p))
    head(ctx, t, [(cs(sh, 3), "这段录音，能帮到谁？")])


def scene1(ctx, t, sh):
    T_Z0, T_Z1, T_IN = 2.3, 3.1, 2.75
    t_p0 = cs(sh, 2) - 0.45
    rec_t = cs(sh, 1) + 0.9
    if t < T_Z1 + 0.05:
        z = eio(prog(t, T_Z0, T_Z1 - T_Z0))
        with zoom(ctx, WIN_C[0], WIN_C[1], 1 + 6 * z):
            exterior(ctx, t)
    if T_IN <= t < t_p0 + 0.65:
        s1 = lerp(1.12, 1.0, eo(prog(t, T_IN, 0.9)))
        s2 = lerp(1.0, 1.35, eio(prog(t, cs(sh, 1), 1.1)))
        f2 = (430, 600)
        px, py = tourist_phone(t, sh)
        p1 = (960 + s1 * (px - 960), 540 + s1 * (py - 540))
        p2 = (f2[0] + s2 * (p1[0] - f2[0]), f2[1] + s2 * (p1[1] - f2[1]))
        s3 = lerp(1.0, 3.4, eio(prog(t, t_p0, 0.6)))
        with alpha(ctx, fin(t, T_IN, 0.45)):
            with zoom(ctx, p2[0], p2[1], s3):
                with zoom(ctx, f2[0], f2[1], s2):
                    with zoom(ctx, 960, 540, s1):
                        interior(ctx, t, sh, T_IN, rec_t)
    if t >= t_p0:
        with alpha(ctx, fin(t, t_p0, 0.5)):
            phone_close(ctx, t, sh)


# ================================================================ shot 2 — the complaint and the new rules
def law_doc(ctx):
    rrect(ctx, -130 + 8, -170 + 10, 260, 340, 16, INK, 0.08)
    rrect(ctx, -130, -170, 260, 340, 16, WHITE, 1, INK2, 3)
    rrect(ctx, -130, -170, 260, 72, 16, TEAL_D)
    ctx.rectangle(-130, -120, 260, 22)
    col(ctx, TEAL_D)
    ctx.fill()
    text(ctx, "旅游投诉处理办法", 0, -133, 26, WHITE, True)
    for i in range(7):
        w = 190 if i % 3 else 150
        rrect(ctx, -95, -66 + i * 30, w, 10, 5, GRAY_L)


def flip_calendar(ctx, x, y, p_flip):
    with at(ctx, x, y):
        rrect(ctx, -120 + 8, -130 + 10, 240, 260, 18, INK, 0.08)
        rrect(ctx, -120, -130, 240, 260, 18, WHITE, 1, INK2, 3)
        rrect(ctx, -120, -130, 240, 72, 18, ORANGE)
        ctx.rectangle(-120, -80, 240, 22)
        col(ctx, ORANGE)
        ctx.fill()
        text(ctx, "收到投诉后", 0, -94, 28, WHITE, True)
        circle(ctx, -62, -130, 10, INK2)
        circle(ctx, 62, -130, 10, INK2)
        text(ctx, "2", 0, 4, 104, INK, True)
        text(ctx, "个工作日", 0, 88, 28, INK2, False)
        if p_flip < 1:
            ctx.save()
            ctx.translate(0, -58)
            ctx.scale(1, max(1e-3, 1 - p_flip))
            ctx.translate(0, 58)
            ctx.rectangle(-114, -58, 228, 182)
            col(ctx, mix(WHITE, GRAY_L, 0.6 * p_flip))
            ctx.fill()
            text(ctx, "1", 0, 4, 104, INK, True)
            text(ctx, "个工作日", 0, 88, 28, INK2, False)
            ctx.restore()


def scene2(ctx, t, sh):
    bg(ctx, t)
    t_b = cs(sh, 1) - 0.3
    a_a = 1 - prog(t, t_b - 0.1, 0.4)
    if a_a > 0:
        with alpha(ctx, a_a):
            head(ctx, t, [(0.7, "云南 · 旅游大省")])
            pd = fin(t, 0.25, 0.5)
            with alpha(ctx, pd):
                figure(ctx, 1420, 690, 1.25, shirt=BLUE, upper=True, badge=True, look=-0.6)
                rrect(ctx, 1060, 676, 700, 34, 10, hx("B07D4F"))
                rrect(ctx, 1090, 708, 640, 190, 8, hx("D7AE82"))
                rrect(ctx, 1220, 752, 380, 76, 14, WHITE)
                text(ctx, "旅游投诉受理", 1410, 790, 36, TEAL_D, True)
                rrect(ctx, 1125, 646, 180, 32, 8, GRAY_L)
                rrect(ctx, 1135, 640, 160, 10, 4, hx("BDBBB2"))
            if t < 1.25:
                p = eio(prog(t, 0.15, 1.1))
                audio_file(ctx, lerp(-90, 700, p), 540 - 170 * math.sin(PI * p), 1.1, rot=lerp(-0.6, 0, p))
            elif t < 2.35:
                p = eio(prog(t, 1.25, 1.05))
                sp = min(1.0, pop(t, 1.25, 0.3))
                envelope(ctx, lerp(700, 1215, p), lerp(420, 628, p) - 130 * math.sin(PI * p),
                         lerp(1.0, 0.55, p) * sp, rot=0.25 * math.sin(PI * p))
            else:
                envelope(ctx, 1215, 628, 0.55)
            t_nb = ph(sh, 0, "文旅部门")
            p = pop(t, t_nb - 0.2)
            if p > 0:
                with at(ctx, 500, 540, p):
                    rrect(ctx, -330 + 8, -190 + 10, 660, 380, 26, INK, 0.08)
                    rrect(ctx, -330, -190, 660, 380, 26, WHITE, 1, GRAY_L, 3)
                    rrect(ctx, -330, -190, 660, 92, 26, ORANGE)
                    ctx.rectangle(-330, -130, 660, 32)
                    col(ctx, ORANGE)
                    ctx.fill()
                    text(ctx, "文旅部门 · 集中整治", 0, -144, 40, WHITE, True)
                    t_it = ph(sh, 0, "整治")
                    for k, label in enumerate(["导游乱象", "强制消费"]):
                        pk = pop(t, t_it + 0.45 * k)
                        if pk > 0:
                            yy = -10 + k * 112
                            with at(ctx, -170, yy, pk):
                                circle(ctx, 0, 0, 34, ORANGE_L, 1, ORANGE, 4)
                                text(ctx, "!", 0, -2, 42, ORANGE_D, True)
                            text(ctx, label, -110, yy, 54, INK, True, "left", min(1, pk))
    if t >= t_b:
        t1 = cs(sh, 1)
        head(ctx, t, [(t1, "《旅游投诉处理办法》")])
        p = pop(t, t1 + 0.25)
        if p > 0:
            with at(ctx, 400, 480, 1.18 * p):
                law_doc(ctx)
        pa = pop(t, ph(sh, 1, "起施行") - 0.2)
        if pa > 0:
            with zoom(ctx, 400, 790, pa):
                pill(ctx, "2026 年 3 月 15 日起施行", 400, 790, 32, TEAL, WHITE, True)
        t2 = cs(sh, 2)
        p = pop(t, t2 - 0.15)
        if p > 0:
            with zoom(ctx, 960, 480, 1.22 * p):
                flip_calendar(ctx, 960, 480, eio(prog(t, t2 + 0.2, 0.45)))
        pa = pop(t, ph(sh, 2, "两个工作日") + 0.2)
        if pa > 0:
            with zoom(ctx, 960, 790, pa):
                pill(ctx, "2 个工作日内决定是否受理", 960, 790, 32, ORANGE, WHITE, True)
        t3 = cs(sh, 3)
        p = pop(t, t3 - 0.15)
        if p > 0:
            with zoom(ctx, 1515, 560, 1.2 * p):
                figure(ctx, 1410, 650, 0.9, shirt=BLUE, look=0.6)
                shop(ctx, 1622, 570, 0.8)
                text(ctx, "游客", 1410, 692, 28, INK2, True)
                text(ctx, "商家", 1622, 692, 28, INK2, True)
        for k, (dx, dy) in enumerate([(1388, 336), (1642, 416)]):
            pd = pop(t, t3 + 0.35 + 0.2 * k)
            if pd > 0:
                with at(ctx, dx, dy - 10 * math.sin(PI * min(1, pd)), 0.75 * pd):
                    doc_icon(ctx, 0, 0, 1.0, ORANGE)
        pa = pop(t, t3 + 0.6)
        if pa > 0:
            with zoom(ctx, 1515, 790, pa):
                pill(ctx, "投诉双方都要提供证据", 1515, 790, 32, BLUE, WHITE, True)
        text(ctx, "以官方文件为准", 1830, 892, 22, GRAY, False, "right", fin(t, t1 + 0.6))


# ================================================================ shot 3 — the long recording problem
def long_wave(ctx, x0, x1, y, h, reveal, play=None):
    rrect(ctx, x0 - 18, y - h / 2 - 16, (x1 - x0) + 36, h + 32, 18, WHITE, 1, GRAY_L, 2)
    xr = x0 + (x1 - x0) * reveal
    i = 0
    while True:
        bx = x0 + i * 9
        if bx > min(xr, x1):
            break
        v = 0.22 + 0.78 * abs(math.sin(i * 0.37) * math.sin(i * 0.11 + 1.3))
        hgt = 6 + (h - 10) * v
        c = TEAL if (play is not None and bx < play) else TEAL_M
        rrect(ctx, bx, y - hgt / 2, 5, hgt, 2.5, c)
        i += 1


def scene3(ctx, t, sh):
    bg(ctx, t)
    c1, c2 = cs(sh, 1), cs(sh, 2)
    head(ctx, t, [(0.7, "录音没法“一眼看完”"), (c1, "1 小时录音，人工听写要好几个小时")])
    rv = eio(prog(t, 0.6, 2.2))
    play = 170 + 9 * max(0.0, t - c1)
    long_wave(ctx, 170, 1750, 300, 92, rv, play if t > c1 else None)
    if rv > 0.05:
        text(ctx, "00:00:00", 170, 384, 24, GRAY, False, "left", fin(t, 0.8))
        text(ctx, "01:00:00", 1750, 384, 24, GRAY, False, "right", fin(t, 2.4))
    if t > c1:
        line(ctx, play, 238, play, 362, RED, 4)
        circle(ctx, play, 238, 9, RED)
    # markers dropped on the strip while searching (shot cue 2)
    for k, (mx, phrase) in enumerate([(640, "关键"), (1120, "多少钱"), (1460, "是谁说的")]):
        pm = pop(t, ph(sh, 2, phrase))
        if pm > 0:
            qmark(ctx, mx, 300 - 4 * math.sin(t * 3 + k), 26 * pm, a=min(1, pm))
    # desk scene
    figure(ctx, 760, 700, 1.35, shirt=hx("F2F2EE"), vest=BLUE, headphones=True, upper=True, badge=True,
           look=0.5 if t < c2 else -0.3, arm_l=(700, 676 + 4 * math.sin(t * 18)),
           arm_r=(830, 672 + 4 * math.sin(t * 18 + 1.5)))
    rrect(ctx, 340, 690, 880, 30, 10, hx("B07D4F"))
    rrect(ctx, 380, 720, 26, 170, 6, hx("8E6440"))
    rrect(ctx, 1154, 720, 26, 170, 6, hx("8E6440"))
    rrect(ctx, 680, 664, 180, 26, 6, GRAY_L, 1, GRAY, 2)
    n_files = 3 + int(clamp((t - c1) * 0.9, 0, 6)) if t > c1 else 3
    for k in range(n_files):
        audio_file(ctx, 470 + (k % 2) * 16, 640 - k * 22, 0.7, TEAL if k % 3 else ORANGE, rot=0.08 * ((k * 37) % 5 - 2))
    if t > c1 + 3.0:
        a = fin(t, c1 + 3.0)
        ctx.move_to(845, 555)
        ctx.curve_to(835, 575, 838, 590, 848, 592)
        ctx.curve_to(858, 590, 860, 575, 845, 555)
        col(ctx, BLUE, 0.85 * a)
        ctx.fill()
    # transcript panel
    rrect(ctx, 1010, 470, 290, 200, 16, WHITE, 1, GRAY_L, 2)
    text(ctx, "听写稿", 1036, 498, 22, INK2, True, "left")
    done = max(0.0, (t - c1) * 0.42) if t > c1 else 0.0
    for i in range(5):
        frac = clamp(done - i)
        if frac > 0:
            rrect(ctx, 1036, 528 + i * 26, 230 * frac, 9, 4.5, GRAY_L)
    if t > c1 and int(t * 2) % 2 == 0:
        li = min(4, int(done))
        rrect(ctx, 1036 + 230 * clamp(done - li) + 4, 522 + li * 26, 3, 20, 1, INK)
    # clock spinning
    spin = max(0.0, t - c1) * 5.0
    clock(ctx, 1570, 600, 118, spin)
    pa = pop(t, ph(sh, 1, "好几倍"))
    if pa > 0:
        with zoom(ctx, 1570, 790, pa):
            pill(ctx, "好几倍的时间", 1570, 790, 30, ORANGE, WHITE, True)
    for k, (phrase, label, bx) in enumerate([("关键", "关键是哪几句？", 470),
                                             ("多少钱", "说了多少钱？", 780),
                                             ("是谁说的", "是谁说的？", 1060)]):
        pb = pop(t, ph(sh, 2, phrase) - 0.05)
        if pb > 0:
            w = text_w(ctx, label, 30, True) + 44
            with zoom(ctx, bx, 440, pb):
                bubble(ctx, bx - w / 2, 396, w, 72, 760 + (bx - 760) * 0.2, 492, WHITE, INK2, 1, 3, 22, 16)
                text(ctx, label, bx, 432, 30, ORANGE_D if k == 1 else INK, True)


# ================================================================ shot 4 — meet the assistant
def scene4(ctx, t, sh):
    bg(ctx, t)
    c1, c2 = cs(sh, 1), cs(sh, 2)
    m = eio(prog(t, c1 - 0.2, 0.7))
    t_title = ph(sh, 0, "旅游投诉录音")
    pa = fin(t, t_title - 0.25, 0.5)
    if pa > 0:
        y = lerp(262, 118, m) + 20 * (1 - pa)
        size = lerp(68, 56, m)
        w = text(ctx, "旅游投诉录音证据智能整理助手", 960, y, size, INK, True, a=pa)
        rrect(ctx, 960 - w * 0.17 * pa, y + size * 0.72, w * 0.34 * pa, 8, 4, TEAL, pa)
    pp = pop(t, ph(sh, 0, "我们要做的项目") - 0.1) * (1 - m)
    if pp > 0:
        with zoom(ctx, 960, 168, pp):
            pill(ctx, "本课程项目", 960, 168, 26, ORANGE, WHITE, True)
    rise = back(prog(t, 0.3, 0.8))
    if t > 0.3:
        ax = lerp(960, 400, m)
        ay = lerp(610, 520, m) + 420 * (1 - rise)
        s = lerp(1.55, 1.2, m)
        talk = 0.8 if (c1 + 0.7 < t < c1 + 1.4) else 0.35
        ai_ting(ctx, ax, ay, s, t, talk=talk, wave=bump(t, 1.4, 1.4))
        for k in range(5):
            sp = bump(t, 0.9 + 0.12 * k, 0.9)
            if sp > 0:
                ang = -PI / 2 + (k - 2) * 0.6
                sparkle(ctx, ax + 210 * math.cos(ang), ay - 40 + 170 * math.sin(ang), 20 * sp, ORANGE)
    pb = pop(t, 1.7) * (1 - prog(t, c1 - 0.6, 0.3))
    if pb > 0:
        with zoom(ctx, 1150, 560, pb):
            bubble(ctx, 1130, 470, 360, 92, 1080, 560)
            text(ctx, "你好，我是阿听！", 1310, 516, 34, TEAL_D, True)
    # recording flies in
    if c1 - 0.1 < t < c1 + 0.8:
        p = eio(prog(t, c1, 0.7))
        audio_file(ctx, lerp(-80, 400, p), lerp(760, 520, p) - 120 * math.sin(PI * p), lerp(1.1, 0.3, p),
                   rot=lerp(-0.5, 0.4, p))
    # draft card appears
    cx0, cy0, cw = 760, 250, 900
    pc = eo(prog(t, c1 + 1.0, 0.6))
    if pc > 0:
        ar = fin(t, c1 + 0.8)
        arrow(ctx, 545, 470, lerp(560, 735, ar), 470, TEAL, 6, ar)
        text(ctx, "生成", 645, 438, 24, TEAL_D, True, a=ar)
        rows = max(0.0, (t - c1 - 1.3) * 3.0)
        hl_t = ph(sh, 2, "哪几段")
        am_t = ph(sh, 2, "提到了")
        hp = 0.5 + 0.5 * math.sin((t - hl_t) * 7) if t > hl_t else 0.0
        with alpha(ctx, pc):
            draft_card(ctx, cx0 + 60 * (1 - pc), cy0, cw, rows=rows, t=t, hl_pulse=hp)
            # column highlight: timestamps, speakers
            for k, (x_off, w_off, cc) in enumerate([(18, 116, INK2), (124, 92, TEAL)]):
                tk = c2 + 0.55 * k
                a_k = fin(t, tk, 0.3) * (1 - prog(t, tk + 1.9, 0.4))
                if a_k > 0:
                    rrect(ctx, cx0 + x_off, cy0 + 80, w_off, 4 * 78 + 4, 14, None, a_k, cc, 4)
            ra = bump(t, am_t, 1.6)
            if ra > 0:
                ellipse(ctx, cx0 + 433, cy0 + 92 + 3 * 78 + 26, 92 + 10 * ra, 38 + 6 * ra, None, ra, ORANGE, 5)
    # legend chips under the card
    chips = [("时间", INK2, c2), ("说话人", TEAL, c2 + 0.55), ("疑似片段", ORANGE, ph(sh, 2, "哪几段")),
             ("金额号码", ORANGE_D, ph(sh, 2, "提到了"))]
    t_all = ph(sh, 2, "一目了然")
    if t > c2 - 0.3:
        for k, (lab, cc, tk) in enumerate(chips):
            x = 960 + (k - 1.5) * 250
            on = fin(t, tk, 0.3)
            glow = bump(t, t_all + 0.1 * k, 0.6)
            a = fin(t, c2 - 0.3 + 0.08 * k)
            with zoom(ctx, x, 800, 1 + 0.12 * glow):
                pill(ctx, lab, x, 800, 30, mix(WHITE, cc, on), mix(GRAY, WHITE, on), True, a=a,
                     padx=26, h=56, stroke=mix(GRAY_L, cc, on), lw=3)


# ================================================================ shot 5 — it only flags, people decide
def scene5(ctx, t, sh):
    bg(ctx, t)
    c1 = cs(sh, 1)
    t_final = ph(sh, 1, "最后一定")
    head(ctx, t, [(0.7, "只提示，不判定"), (t_final, "最后一定人工复核")])
    ai_ting(ctx, 330, 640, 1.05, t, talk=0.5 if t < ce(sh, 0) else 0.2)
    ps = pop(t, 1.0)
    if ps > 0:
        with zoom(ctx, 330, 520, ps):
            line(ctx, 416, 667, 416, 440, hx("8E6440"), 12)
            rrect(ctx, 150, 330, 360, 120, 18, WHITE, 1, TEAL, 5)
            text(ctx, "我只提示", 330, 368, 36, TEAL_D, True)
            text(ctx, "不判定", 330, 414, 36, ORANGE_D, True)
    cx0, cy0, cw = 640, 262, 760
    pc = fin(t, 0.4, 0.5)
    stamp_t = c1 + sh["cues"][1]["dur"] * 0.55
    t_check = t_final + 1.5
    hp = 0.5 + 0.5 * math.sin((t - c1) * 7) if t > c1 else 0.0
    with alpha(ctx, pc):
        draft_card(ctx, cx0, cy0, cw, rows=4, t=t, hl_pulse=hp, stamp=prog(t, stamp_t - 0.25, 0.25),
                   review=prog(t, t_check, 0.6), foot=96)
    # reviewer walks in with a magnifier
    wp = eo(prog(t, t_final - 0.1, 0.9))
    if wp > 0:
        x = lerp(2080, 1530, wp)
        bob = 6 * abs(math.sin(t * 10)) * (1 - wp)
        hp_ = eio(prog(t, t_final + 0.8, 0.6))
        hand = (lerp(x - 70, 1402, hp_), lerp(880 - 150, 594, hp_) - bob)
        figure(ctx, x, 902 - bob, 1.3, shirt=hx("F2F2EE"), vest=BLUE, headphones=True, badge=True,
               look=-1.0, arm_l=hand)
        magnifier(ctx, hand[0] - 52, hand[1] - 52, 0.9)
        pr = pop(t, t_check + 0.4)
        if pr > 0:
            with zoom(ctx, 1600, 470, pr):
                rrect(ctx, 1470, 442, 260, 58, 29, GOOD)
                check(ctx, 1510, 471, 0.55, WHITE)
                text(ctx, "人工复核", 1625, 471, 30, WHITE, True)


# ================================================================ shot 6 — why not a public chatbot
def scene6(ctx, t, sh):
    bg(ctx, t)
    c1, c2 = cs(sh, 1), cs(sh, 2)
    head(ctx, t, [(0.7, "为什么不直接用通用 AI？"), (c2, "本地运行 · 听懂本地专名")])
    pdv = fin(t, c1 - 0.2, 0.5)
    if pdv > 0:
        dashed(ctx, 960, 230, 960, lerp(230, 890, pdv), GRAY_L, 4, 1, (12, 12))
    # left: public cloud model
    pl = pop(t, c1 - 0.1)
    if pl > 0:
        with zoom(ctx, 480, 350, pl):
            cloud(ctx, 465, 310, 1.45, BLUE_LL)
            text(ctx, "公共大模型", 488, 362, 46, hx("185FA5"), True)
        audio_file(ctx, 480, 700, 1.45 * min(1, pl), ORANGE)
    for k, (lab, x, y) in enumerate([("姓名", 300, 680), ("电话", 660, 680), ("证件号", 480, 852)]):
        pk = pop(t, ph(sh, 1, "个人信息") + 0.12 * k)
        if pk > 0:
            with zoom(ctx, x, y, pk):
                pill(ctx, lab, x, y, 30, ORANGE_L, ORANGE_D, True, stroke=ORANGE, lw=2)
    pu = prog(t, ph(sh, 1, "不适合") - 0.6, 0.5)
    if pu > 0:
        y_tip = lerp(615, 470, eo(pu))
        dashed(ctx, 480, 615, 480, y_tip, GRAY, 6, 1, (14, 10), offset=-t * 30)
    px_ = pop(t, ph(sh, 1, "不适合"))
    if px_ > 0:
        with zoom(ctx, 480, 538, px_):
            circle(ctx, 480, 538, 50, WHITE, 1, RED, 6)
            cross(ctx, 480, 538, 0.85, RED)
        with zoom(ctx, 700, 538, px_):
            pill(ctx, "不适合上传", 700, 538, 32, RED, WHITE, True)
    # right: local computer + hot words
    pr = pop(t, c2 - 0.1)
    if pr > 0:
        with zoom(ctx, 1290, 560, pr):
            monitor(ctx, 1290, 530, 1.25, glow=fin(t, c2 + 0.2, 0.3))
    pg = pop(t, c2 + 0.2)
    if pg > 0:
        with zoom(ctx, 1290, 776, pg):
            rrect(ctx, 1150, 746, 280, 60, 30, GOOD)
            check(ctx, 1192, 776, 0.55, WHITE)
            text(ctx, "本地运行", 1312, 776, 32, WHITE, True)
    pa = pop(t, ph(sh, 2, "还要专门") - 0.2)
    if pa > 0:
        ai_ting(ctx, 1665, 560, 1.0 * min(1.05, pa), t, talk=0.6, look=-0.5)
        with zoom(ctx, 1665, 776, pa):
            pill(ctx, "热词增强", 1665, 776, 30, PURPLE, WHITE, True)
    tags = [("石林", 1140, 300, TEAL, "地名"), ("大理", 1320, 280, TEAL, "地名"),
            ("雾隐行舟旅行社", 1590, 330, ORANGE, "店名"), ("鲜花饼", 1820, 400, ORANGE, "店名"),
            ("翡翠", 1150, 870, PURPLE, "行话"), ("冰种", 1330, 870, PURPLE, "行话")]
    t_tags = ph(sh, 2, "听懂")
    for k, (lab, x, y, cc, key) in enumerate(tags):
        pk = pop(t, t_tags + 0.3 * k)
        if pk > 0:
            yy = y + 5 * math.sin(t * 2.5 + k)
            with zoom(ctx, x, yy, pk):
                tag(ctx, lab, x, yy, 28, WHITE, INK, cc)


SCENES_A = [scene1, scene2, scene3, scene4, scene5, scene6]
