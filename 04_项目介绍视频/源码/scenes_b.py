"""Shots 7-12."""
import math
import random

import cairo

from gfx import *  # noqa: F401,F403
from scene_util import *  # noqa: F401,F403

# ================================================================ shot 7 — the eight-station pipeline
ST_W, ST_H = 390, 170
ST_X = [120, 555, 990, 1425]
ST_Y = [272, 592]
STATIONS = [
    ("统一格式", "采样率 · 声道 · 命名", "教师模板"),
    ("降噪切段", "降噪 · 端点检测", "第 1、2 组"),
    ("转成文字", "语音识别 · 热词", "第 5 组"),
    ("分清说话人", "说话人分离", "第 3 组"),
    ("数字证号规范化", "金额 · 电话 · 证号", "第 4 组"),
    ("话术分类", "TensorFlow 深度学习", "第 6、7 组"),
    ("截取证据片段", "时间戳 · 片段导出", "第 8 组"),
    ("核查初稿", "人工复核后才能使用", "处理人员"),
]


def st_pos(i):
    if i < 4:
        return ST_X[i], ST_Y[0]
    return ST_X[7 - i], ST_Y[1]


def badge_color(b):
    return TEAL_D if b == "教师模板" else (INK2 if b == "处理人员" else ORANGE)


def draw_station(ctx, i, act, pulse):
    x, y = st_pos(i)
    title, sub, badge = STATIONS[i]
    rrect(ctx, x + 6, y + 8, ST_W, ST_H, 22, INK, 0.06)
    rrect(ctx, x, y, ST_W, ST_H, 22, mix(WHITE, TEAL_L, act), 1, mix(GRAY_L, TEAL, act), 3 + 3 * act)
    circle(ctx, x + 42, y + 44, 25, mix(GRAY, TEAL, act))
    text(ctx, str(i + 1), x + 42, y + 44, 28, WHITE, True)
    text(ctx, title, x + ST_W / 2 + 22, y + 48, 32, INK, True)
    text(ctx, sub, x + ST_W / 2 + 22, y + 90, 23, INK2, False)
    bx, by = x + ST_W / 2, y + ST_H - 34
    with zoom(ctx, bx, by, 1 + 0.15 * pulse):
        pill(ctx, badge, bx, by, 23, badge_color(badge), WHITE, True, padx=18, h=42)


def connector(i):
    """Start/end points of the connector feeding station i (i >= 1)."""
    x0, y0 = st_pos(i - 1)
    x1, y1 = st_pos(i)
    if i < 4:
        return (x0 + ST_W + 4, y0 + ST_H / 2), (x1 - 4, y1 + ST_H / 2)
    if i == 4:
        return (x0 + ST_W / 2, y0 + ST_H + 4), (x1 + ST_W / 2, y1 - 4)
    return (x0 - 4, y0 + ST_H / 2), (x1 + ST_W + 4, y1 + ST_H / 2)


def scene7(ctx, t, sh, stations):
    bg(ctx, t)
    c2 = cs(sh, 2)
    head(ctx, t, [(0.7, "八步流水线"), (c2, "八步流水线 · 每组守一个工位")])
    # feed line into station 1
    p0 = prog(t, stations[0] - 0.4, 0.4)
    if p0 > 0:
        y = ST_Y[0] + ST_H / 2
        line(ctx, 0, y, 116, y, TEAL, 6)
        if p0 < 1:
            audio_file(ctx, lerp(-40, 90, eo(p0)), y, 0.45, ORANGE)
    for i in range(1, 8):
        (x1, y1), (x2, y2) = connector(i)
        lit = t >= stations[i] - 0.35
        pa = fin(t, 0.9 + 0.25 * i)
        if pa <= 0:
            continue
        arrow(ctx, x1, y1, x2, y2, TEAL if lit else GRAY_L, 6, pa, head=16)
        p = prog(t, stations[i] - 0.35, 0.35)
        if 0 < p < 1:
            circle(ctx, lerp(x1, x2, p), lerp(y1, y2, p), 12, ORANGE)
    for i in range(8):
        ps = pop(t, 0.9 + 0.25 * i)
        if ps <= 0:
            continue
        x, y = st_pos(i)
        act = fin(t, stations[i], 0.3)
        flash = bump(t, stations[i], 0.5)
        pulse = bump(t, c2 + 0.12 * i, 0.6)
        with zoom(ctx, x + ST_W / 2, y + ST_H / 2, ps * (1 + 0.05 * flash)):
            draw_station(ctx, i, act, pulse)
    pl = fin(t, c2 + 0.2)
    if pl > 0:
        items = [("教师模板", TEAL_D), ("学生小组", ORANGE), ("处理人员（人工）", INK2)]
        x = 960 - 380
        for lab, cc in items:
            rrect(ctx, x, 845, 26, 26, 6, cc, pl)
            w = text(ctx, lab, x + 40, 858, 26, INK2, False, "left", pl)
            x += 40 + w + 60


# ================================================================ shot 8 — where the data comes from
def cond_card(ctx, y, kind, t):
    rrect(ctx, 1000 + 6, y - 72 + 8, 400, 144, 20, INK, 0.06)
    rrect(ctx, 1000, y - 72, 400, 144, 20, WHITE, 1, GRAY_L, 3)
    phone(ctx, 1062, y, 0.82, rec=1.0, t=t)
    labels = {"quiet": "安静", "noisy": "嘈杂教室", "pocket": "口袋 · 远距离"}
    text(ctx, labels[kind], 1118, y - 26, 32, INK, True, "left")
    pts = []
    for i in range(41):
        xx = 1120 + i * 6.5
        if kind == "quiet":
            yy = 22 * math.sin(i * 0.5 + t * 6)
        elif kind == "noisy":
            yy = 16 * math.sin(i * 0.5 + t * 6) + 12 * math.sin(i * 2.7 + t * 23) * math.sin(i * 1.3)
        else:
            yy = 6 * math.sin(i * 0.5 + t * 6)
        pts.append((xx, y + 34 + yy))
    poly(ctx, pts, {"quiet": TEAL, "noisy": ORANGE, "pocket": GRAY}[kind], 4)
    if kind == "pocket":
        rrect(ctx, 1062 - 46, y + 2, 92, 66, 12, hx("4A6FA5"))
        line(ctx, 1062 - 34, y + 14, 1062 + 34, y + 14, hx("7F9CC8"), 3)


def scene8(ctx, t, sh):
    bg(ctx, t)
    c1, c2, c3 = cs(sh, 1), cs(sh, 2), cs(sh, 3)
    head(ctx, t, [(0.7, "数据从哪儿来？"), (c3, "24 个剧本 × 3 种条件 = 72 段录音", 54)])
    a1 = fin(t, c1 - 0.15) * (1 - prog(t, c2 - 0.35, 0.4))
    if a1 > 0:
        with alpha(ctx, a1):
            with zoom(ctx, 960, 520, pop(t, c1 - 0.15)):
                audio_file(ctx, 960, 480, 2.0, GRAY)
                text(ctx, "真实投诉录音", 960, 650, 38, INK2, True)
            px_ = pop(t, ph(sh, 1, "真实") + 0.1)
            if px_ > 0:
                with zoom(ctx, 960, 480, px_):
                    circle(ctx, 960, 480, 130, None, 1, RED, 12)
                    line(ctx, 868, 572, 1052, 388, RED, 12)
                with zoom(ctx, 960, 740, px_):
                    pill(ctx, "一律不用", 960, 740, 32, RED, WHITE, True)
    a2 = fin(t, c2 - 0.25, 0.5)
    if a2 > 0:
        with alpha(ctx, a2):
            rrect(ctx, 130, 812, 740, 34, 12, hx("C9A57E"))
            rrect(ctx, 130, 846, 740, 20, 8, hx("A9845C"))
            talk_a = 0.5 + 0.5 * math.sin(t * 13) if int(t * 1.2) % 2 == 0 else 0.0
            talk_b = 0.5 + 0.5 * math.sin(t * 13) if int(t * 1.2) % 2 == 1 else 0.0
            figure(ctx, 270, 812, 1.12, shirt=EARTH, hat=ORANGE, look=1.0, mouth=talk_a,
                   arm_r=(330, 640 + 10 * math.sin(t * 4)))
            figure(ctx, 490, 812, 1.12, shirt=BLUE, hair=hx("6B4A2E"), look=-1.0, mouth=talk_b,
                   arm_l=(430, 660), arm_r=(552, 640))
            phone(ctx, 556, 622, 0.5, rec=1.0, t=t)
            figure(ctx, 720, 812, 1.12, shirt=PURPLE, look=-0.6, arm_l=(684, 712), arm_r=(762, 712))
            with at(ctx, 724, 690, 0.85, 0.06):
                rrect(ctx, -58, -70, 116, 140, 8, WHITE, 1, INK2, 3)
                text(ctx, "剧本", 0, -38, 28, INK, True)
                for k in range(3):
                    rrect(ctx, -40, -6 + k * 22, 80 - (k % 2) * 20, 8, 4, GRAY_L)
            for k, (bx, by_, who) in enumerate([(160, 430, talk_a), (420, 430, talk_b)]):
                if who > 0:
                    bubble(ctx, bx, by_, 120, 62, bx + 80 + 60 * k, 520, WHITE, INK2, 1, 3, 20, 12)
                    for d in range(3):
                        circle(ctx, bx + 36 + d * 24, by_ + 31, 6, INK2)
            pill(ctx, "课堂情景剧", 500, 320, 32, TEAL_L, TEAL_D, True, stroke=TEAL, lw=2)
    pb = pop(t, ph(sh, 2, "虚构剧本"))
    if pb > 0:
        with zoom(ctx, 830, 500, pb):
            pill(ctx, "全部虚构", 830, 500, 26, TEAL, WHITE, True)
    for k, (key, kind) in enumerate([("安静", "quiet"), ("嘈杂教室", "noisy"), ("手机放口袋", "pocket")]):
        pk = pop(t, ph(sh, 2, key) - 0.1)
        if pk > 0:
            y = 330 + 195 * k
            with zoom(ctx, 1200, y, pk):
                cond_card(ctx, y, kind, t)
    # shared data pool
    pp = pop(t, c3 - 0.1)
    if pp > 0:
        with zoom(ctx, 1650, 600, pp):
            cylinder(ctx, 1650, 590, 250, 210, TEAL)
            n = int(round(72 * eio(prog(t, c3 + 0.3, 3.0))))
            text(ctx, f"{n} 段", 1650, 612, 46, WHITE, True)
            text(ctx, "共享数据池", 1650, 752, 32, INK, True)
    pd = pop(t, ph(sh, 3, "同一份标准答案"))
    if pd > 0:
        with zoom(ctx, 1650, 350, pd):
            doc_icon(ctx, 1650, 330, 0.9, TEAL)
            text(ctx, "同一份标准答案", 1650, 410, 26, INK2, True)
    if t > c3:
        for k in range(12):
            t0 = c3 + 0.3 + 0.22 * k
            p = prog(t, t0, 0.6)
            if 0 < p < 1:
                sy = 330 + 195 * (k % 3)
                x = lerp(1400, 1650, eio(p))
                y = lerp(sy, 500, eio(p)) - 90 * math.sin(PI * p)
                audio_file(ctx, x, y, 0.42, [TEAL, ORANGE, GRAY][k % 3])


# ================================================================ shot 9 — measuring with character error rate
REF = "雾隐行舟旅行社"
HYP = "雾影行走旅行社"


def char_row(ctx, label, chars, y, t, t0, err_t, is_hyp):
    a = fin(t, t0, 0.3)
    if a <= 0:
        return
    text(ctx, label, 415, y, 32, INK2, True, "right", a)
    for i, ch in enumerate(chars):
        pk = pop(t, t0 + 0.06 * i, 0.35)
        if pk <= 0:
            continue
        x = 486 + i * 106
        wrong = REF[i] != HYP[i]
        e = fin(t, err_t, 0.25) if wrong else 0.0
        shake = 8 * math.sin(t * 50) * bump(t, err_t, 0.45) if (wrong and is_hyp) else 0.0
        with zoom(ctx, x, y, pk):
            if is_hyp:
                fill, stroke, fg = mix(WHITE, ORANGE, e), mix(GRAY_L, ORANGE_D, e), mix(INK, WHITE, e)
            else:
                fill, stroke, fg = WHITE, mix(GRAY_L, ORANGE, e), INK
            rrect(ctx, x - 46 + shake, y - 46, 92, 92, 16, fill, 1, stroke, 3 + 2 * e)
            text(ctx, ch, x + shake, y - 2, 50, fg, True)


def scene9(ctx, t, sh):
    bg(ctx, t)
    c1, c2, c3 = cs(sh, 1), cs(sh, 2), cs(sh, 3)
    head(ctx, t, [(0.7, "识别得准不准？看字错率")])
    pf = pop(t, c1 - 0.1)
    if pf > 0:
        with zoom(ctx, 960, 272, pf):
            rrect(ctx, 960 - 600, 212, 1200, 120, 24, WHITE, 1, GRAY_L, 3)
            segs = [("字错率 CER =（", INK), ("错字", ORANGE_D), (" + ", INK), ("漏字", BLUE), (" + ", INK),
                    ("多字", PURPLE), ("）÷ 总字数", INK)]
            pos = rich(ctx, segs, 960, 272, 46)
            for idx, key, cc in [(1, "错字", ORANGE), (3, "漏字", BLUE), (5, "多字", PURPLE)]:
                tk = ph(sh, 1, key)
                u = fin(t, tk, 0.3)
                if u > 0:
                    x0, w = pos[idx]
                    rrect(ctx, x0, 306, w * u, 8, 4, cc)
    t_cmp = ph(sh, 1, "把识别结果")
    t_sdi = ph(sh, 1, "错字")
    a_cmp = fin(t, t_cmp, 0.4) * (1 - prog(t, t_sdi - 0.45, 0.35))
    if a_cmp > 0:
        with alpha(ctx, a_cmp):
            pill(ctx, "识别结果", 690, 560, 40, BLUE_LL, hx("185FA5"), True, stroke=BLUE, lw=3)
            pill(ctx, "标准答案", 1230, 560, 40, TEAL_L, TEAL_D, True, stroke=TEAL, lw=3)
            arrow(ctx, 830, 560, 1088, 560, INK2, 5, head=16)
            arrow(ctx, 1088, 560, 830, 560, INK2, 5, head=16)
            text(ctx, "逐字对比", 960, 512, 32, INK2, True)
            sweep = (t - t_cmp) * 1.4 % 1.0
            circle(ctx, lerp(840, 1078, sweep), 560, 10, ORANGE)
    a_ex = 1 - prog(t, c2 - 0.35, 0.35)
    if a_ex > 0 and t > c1:
        exs = [("错字", "隐 → 影", ORANGE), ("漏字", "旅行社 → 旅社", BLUE), ("多字", "石林 → 石林啊", PURPLE)]
        for k, (title, ex, cc) in enumerate(exs):
            pk = pop(t, ph(sh, 1, title))
            if pk > 0:
                x = 960 + (k - 1) * 440
                with alpha(ctx, a_ex):
                    with zoom(ctx, x, 540, pk):
                        rrect(ctx, x - 190 + 6, 450 + 8, 380, 180, 24, INK, 0.06)
                        rrect(ctx, x - 190, 450, 380, 180, 24, WHITE, 1, cc, 4)
                        text(ctx, title, x, 496, 34, cc, True)
                        text(ctx, ex, x, 572, 44, INK, True)
    t_err = c2 + sh["cues"][2]["dur"] * 0.45
    char_row(ctx, "标准答案", REF, 470, t, c2, t_err, False)
    char_row(ctx, "识别结果", HYP, 600, t, ph(sh, 2, "被听成了"), t_err, True)
    pr = pop(t, ph(sh, 2, "七个字"))
    if pr > 0:
        with zoom(ctx, 800, 738, pr):
            rich(ctx, [("错 2 字 ÷ 共 7 字 ≈ ", INK), ("28.6%", ORANGE_D)], 800, 738, 48)
    # before / after chart
    if t > c3 - 0.2:
        a = fin(t, c3 - 0.2)
        line(ctx, 1340, 790, 1790, 790, INK2, 4, a)
        text(ctx, "（示意数据）", 1565, 870, 22, GRAY, False, "center", a)
        bars = [(1460, 28.6, ORANGE, "改进前", c3 + 0.1), (1670, 9.5, TEAL, "改进后", ph(sh, 3, "改进后"))]
        for x, v, cc, lab, tb in bars:
            g = eo(prog(t, tb, 0.7))
            hgt = v * 10 * g
            if g > 0:
                rrect(ctx, x - 60, 790 - hgt, 120, hgt, 10, cc)
                text(ctx, f"{v * g:.1f}%", x, 790 - hgt - 30, 34, INK, True, a=min(1, g * 2))
            text(ctx, lab, x, 826, 28, INK2, True, a=a)
        pt = pop(t, ph(sh, 3, "让数字说话"))
        if pt > 0:
            with zoom(ctx, 1565, 392, pt):
                pill(ctx, "让数字说话", 1565, 392, 30, BLUE, WHITE, True)


# ================================================================ shot 10 — four ground rules
RULES = [
    ("名字全部虚构", "人名 · 店名 · 旅行社名", "mask"),
    ("只写商业纠纷", "购物 · 费用 · 行程 · 服务态度", "scale"),
    ("只标疑似", "必须人工复核", "lens"),
    ("签同意书", "不录真实个人信息", "pen"),
]


def rule_icon(ctx, kind, x, y):
    if kind == "mask":
        line(ctx, x - 110, y - 6, x - 86, y, PURPLE, 6)
        line(ctx, x + 110, y - 6, x + 86, y, PURPLE, 6)
        rrect(ctx, x - 92, y - 42, 184, 84, 42, PURPLE)
        ellipse(ctx, x - 40, y - 4, 26, 16, WHITE)
        ellipse(ctx, x + 40, y - 4, 26, 16, WHITE)
        text(ctx, "虚构", x, y + 72, 26, PURPLE, True)
    elif kind == "scale":
        line(ctx, x, y - 70, x, y + 62, INK2, 8)
        rrect(ctx, x - 52, y + 58, 104, 16, 8, INK2)
        line(ctx, x - 92, y - 50, x + 92, y - 50, INK2, 8)
        circle(ctx, x, y - 72, 11, ORANGE)
        for sx in (-92, 92):
            line(ctx, x + sx, y - 50, x + sx - 32, y + 8, GRAY, 3)
            line(ctx, x + sx, y - 50, x + sx + 32, y + 8, GRAY, 3)
            ctx.new_sub_path()
            ctx.arc(x + sx, y + 8, 38, 0, PI)
            ctx.close_path()
            paint(ctx, ORANGE_L, 1, INK2, 3)
    elif kind == "lens":
        magnifier(ctx, x - 14, y - 14, 1.05)
        check(ctx, x - 14, y - 12, 0.7, GOOD)
    else:
        doc_icon(ctx, x - 18, y, 1.35, TEAL)
        with at(ctx, x + 42, y + 18, 1.0, -0.7):
            rrect(ctx, -12, -64, 24, 110, 6, ORANGE)
            ctx.move_to(-12, 46)
            ctx.line_to(12, 46)
            ctx.line_to(0, 72)
            ctx.close_path()
            col(ctx, INK2)
            ctx.fill()


def scene10(ctx, t, sh):
    bg(ctx, t)
    head(ctx, t, [(0.7, "四条规矩")])
    cw, chh, gap = 390, 500, 44
    x0 = (W - (4 * cw + 3 * gap)) / 2
    for i, (title, sub, kind) in enumerate(RULES):
        cx = x0 + i * (cw + gap) + cw / 2
        cy = 500
        deal = pop(t, 0.9 + 0.15 * i)
        if deal <= 0:
            continue
        fp = eio(prog(t, cs(sh, i + 1) - 0.1, 0.5))
        sx = abs(math.cos(PI * fp))
        front = fp >= 0.5
        with zoom(ctx, cx, cy, deal * max(sx, 0.02), deal):
            rrect(ctx, cx - cw / 2 + 8, cy - chh / 2 + 10, cw, chh, 28, INK, 0.07)
            if not front:
                rrect(ctx, cx - cw / 2, cy - chh / 2, cw, chh, 28, TEAL, 1, TEAL_D, 4)
                rrect(ctx, cx - cw / 2 + 22, cy - chh / 2 + 22, cw - 44, chh - 44, 18, None, 1, TEAL_M, 3)
                text(ctx, "规矩", cx, cy - 50, 40, TEAL_L, True)
                text(ctx, str(i + 1), cx, cy + 40, 120, WHITE, True)
            else:
                rrect(ctx, cx - cw / 2, cy - chh / 2, cw, chh, 28, WHITE, 1, TEAL, 4)
                circle(ctx, cx - cw / 2 + 50, cy - chh / 2 + 50, 28, TEAL)
                text(ctx, str(i + 1), cx - cw / 2 + 50, cy - chh / 2 + 50, 30, WHITE, True)
                rule_icon(ctx, kind, cx, cy - 90)
                text(ctx, title, cx, cy + 96, 40, INK, True)
                rrect(ctx, cx - 40, cy + 136, 80, 6, 3, ORANGE)
                text(ctx, sub, cx, cy + 180, 24, INK2, False)


# ================================================================ shot 11 — ten-week roadmap
ROAD = [(70, 770), (300, 690), (600, 440), (900, 690), (1200, 440), (1500, 690), (1730, 440), (1880, 360)]
MILESTONES = [
    ("第 1—2 周", "跑通识别 · 算字错率"),
    ("第 3—4 周", "原理 · TensorFlow 小实验"),
    ("第 5 周", "写剧本 · 录音"),
    ("第 6—8 周", "分组专项改进"),
    ("第 9 周", "八个工位串联"),
    ("第 10 周", "上台演示"),
]


def road_samples():
    pts = []
    P = ROAD
    idx = []
    for k in range(len(P) - 1):
        p0 = P[max(0, k - 1)]
        p1, p2 = P[k], P[k + 1]
        p3 = P[min(len(P) - 1, k + 2)]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        idx.append(len(pts))
        for j in range(40):
            u = j / 40
            a, b, c, d = (1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u * u * (1 - u), u ** 3
            pts.append((a * p1[0] + b * c1[0] + c * c2[0] + d * p2[0], a * p1[1] + b * c1[1] + c * c2[1] + d * p2[1]))
    idx.append(len(pts))
    pts.append(P[-1])
    return pts, idx


ROAD_PTS, ROAD_IDX = road_samples()


def scene11(ctx, t, sh):
    bg(ctx, t)
    head(ctx, t, [(0.7, "十周路线图")])
    rv = eio(prog(t, 0.2, 1.4))
    n = max(2, int(len(ROAD_PTS) * rv))
    for lw, c, dash in [(56, EARTH, None), (6, ORANGE_L, (22, 18))]:
        ctx.save()
        if dash:
            ctx.set_dash(list(dash))
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.move_to(*ROAD_PTS[0])
        for p in ROAD_PTS[1:n]:
            ctx.line_to(*p)
        col(ctx, c)
        ctx.set_line_width(lw)
        ctx.stroke()
        ctx.restore()
    lit_times = [cs(sh, k + 1) for k in range(6)]
    for k, (week, desc) in enumerate(MILESTONES):
        mx, my = ROAD[k + 1]
        if rv < (ROAD_IDX[k + 1] / len(ROAD_PTS)):
            continue
        lit = fin(t, lit_times[k], 0.3)
        ring = bump(t, lit_times[k], 0.7)
        if ring > 0:
            circle(ctx, mx, my, 34 + 24 * ring, None, ring, TEAL, 5)
        circle(ctx, mx, my, 30, mix(GRAY_L, TEAL, lit), 1, WHITE, 5)
        text(ctx, str(k + 1), mx, my, 28, WHITE, True)
        below = (k % 2 == 0)
        ly = my + 112 if below else my - 152
        pl = pop(t, lit_times[k])
        if pl > 0:
            w = max(text_w(ctx, week, 28, True), text_w(ctx, desc, 24, False)) + 44
            with zoom(ctx, mx, ly, pl):
                line(ctx, mx, my + (34 if below else -34), mx, ly + (-50 if below else 50), GRAY_L, 4)
                rrect(ctx, mx - w / 2, ly - 50, w, 100, 18, WHITE, 1, mix(GRAY_L, TEAL, lit), 3)
                text(ctx, week, mx, ly - 18, 28, ORANGE_D, True)
                text(ctx, desc, mx, ly + 22, 24, INK, False)
        if k == 5 and t > lit_times[k]:
            pf = pop(t, lit_times[k] + 0.2)
            fx, fy = ROAD[-1][0] - 6, ROAD[-1][1] - 4
            with zoom(ctx, fx, fy, pf):
                line(ctx, fx, fy, fx, fy - 96, INK2, 6)
                ctx.move_to(fx - 3, fy - 96)
                ctx.line_to(fx - 52, fy - 78 + 5 * math.sin(t * 6))
                ctx.line_to(fx - 3, fy - 60)
                ctx.close_path()
                col(ctx, ORANGE)
                ctx.fill()
    # Ah-Ting walks the road
    if t > 0.7:
        k_now = sum(1 for lt in lit_times if t >= lt - 0.8)
        if k_now == 0:
            j = 0
            moving = 0.0
        else:
            k = k_now
            t_arr = lit_times[k - 1]
            p = eio(prog(t, t_arr - 0.8, 0.8))
            j0 = ROAD_IDX[k - 1] - 12 if k - 1 > 0 else 0
            j1 = ROAD_IDX[k] - 12
            j = int(lerp(j0, j1, p))
            moving = 1.0 if 0 < p < 1 else 0.0
        px, py = ROAD_PTS[min(j, len(ROAD_PTS) - 1)]
        hop = -abs(math.sin(t * 12)) * 14 * moving
        ai_ting(ctx, px, py - 50 + hop, 0.42 * pop(t, 0.7), t, talk=0.3)


# ================================================================ shot 12 — wrap-up
CONFETTI = []
_rng = random.Random(11)
for _ in range(90):
    CONFETTI.append((_rng.uniform(0, W), _rng.uniform(-400, 0), _rng.uniform(160, 320),
                     _rng.uniform(-4, 4), _rng.choice([TEAL, ORANGE, BLUE, PURPLE, GOOD, RED]),
                     _rng.uniform(0, 6.28)))


def scene12(ctx, t, sh):
    bg(ctx, t)
    c1, c2 = cs(sh, 1), cs(sh, 2)
    a0 = 1 - prog(t, c1 - 0.3, 0.4)
    if a0 > 0:
        with alpha(ctx, a0):
            y = 410
            steps = [(0.7, "bus"), (1.3, "pipe"), (1.9, "card"), (2.5, "check")]
            xs = [250, 720, 1180, 1620]
            ext = [(85, 415), (510, 930), (1120, 1240), (1545, 1700)]
            for k, (tk, kind) in enumerate(steps):
                p = pop(t, tk)
                if p <= 0:
                    continue
                x = xs[k]
                with zoom(ctx, x, y, p):
                    if kind == "bus":
                        bus(ctx, x - 165, y + 62, 0.55, rot=t * 6)
                        label = "一段录音"
                    elif kind == "pipe":
                        rrect(ctx, x - 210, y - 52, 420, 104, 52, WHITE, 1, TEAL, 4)
                        for d in range(8):
                            on = 0.5 + 0.5 * math.sin(t * 5 - d * 0.8)
                            circle(ctx, x - 168 + d * 48, y, 16, mix(TEAL_L, TEAL, on))
                        label = "八步流水线"
                    elif kind == "card":
                        doc_icon(ctx, x, y, 1.6, ORANGE)
                        label = "核查初稿"
                    else:
                        figure(ctx, x - 36, y + 98, 0.78, shirt=hx("F2F2EE"), vest=BLUE, headphones=True)
                        circle(ctx, x + 58, y - 36, 38, GOOD)
                        check(ctx, x + 58, y - 34, 0.6, WHITE)
                        label = "人工复核"
                    text(ctx, label, x, y + 128, 32, INK2, True)
                if k > 0:
                    pa = fin(t, tk - 0.1)
                    arrow(ctx, ext[k - 1][1] + 14, y, ext[k][0] - 14, y, TEAL, 6, pa, head=16)
            for k, (key, lab, x) in enumerate([("听不清", "听不清", 560), ("找不到", "找不到", 860)]):
                pk = pop(t, ph(sh, 0, key))
                if pk > 0:
                    with zoom(ctx, x, 700, pk):
                        text(ctx, lab, x, 700, 46, GRAY, True)
                        st = eo(prog(t, ph(sh, 0, "有据可查"), 0.35))
                        if st > 0:
                            line(ctx, x - 80, 702, x - 80 + 160 * st, 702, RED, 6)
            pz = pop(t, ph(sh, 0, "有据可查"))
            if pz > 0:
                arrow(ctx, 980, 700, 1080, 700, TEAL, 6, min(1, pz), head=16)
                with zoom(ctx, 1250, 700, pz):
                    pill(ctx, "有据可查", 1250, 700, 46, TEAL, WHITE, True)
    if t > c1 - 0.3:
        pa = pop(t, c1 - 0.2)
        wave = bump(t, c2 + 0.2, 2.2) if t > c2 else 0.0
        if pa > 0.01:
            ai_ting(ctx, 960, 600, 1.2 * min(1.05, pa), t, talk=0.5, wave=wave)
        studs = [(430, BLUE, "phone"), (650, EARTH, "doc"), (1270, PURPLE, "phone"), (1490, GOOD, "doc")]
        for k, (x, shirt, prop) in enumerate(studs):
            ps = pop(t, c1 + 0.1 * k)
            if ps <= 0:
                continue
            with zoom(ctx, x, 880, ps):
                hand = (x + 45, 700 + 6 * math.sin(t * 3 + k)) if x < 960 else (x - 45, 700 + 6 * math.sin(t * 3 + k))
                if x < 960:
                    figure(ctx, x, 880, 0.95, shirt=shirt, look=0.8, arm_r=hand)
                else:
                    figure(ctx, x, 880, 0.95, shirt=shirt, look=-0.8, arm_l=hand)
                if prop == "phone":
                    phone(ctx, hand[0], hand[1] - 22, 0.42, rec=1.0, t=t)
                else:
                    doc_icon(ctx, hand[0], hand[1] - 24, 0.5, TEAL)
        a_chip = 1 - prog(t, c2 - 0.3, 0.3)
        for key, lab, x in [("每一次录音", "每一次录音", 600), ("每一次测量", "每一次测量", 1320)]:
            pk = pop(t, ph(sh, 1, key))
            if pk > 0 and a_chip > 0:
                with zoom(ctx, x, 320, pk):
                    pill(ctx, lab, x, 320, 46, ORANGE, WHITE, True, a=a_chip)
    if t > c2 - 0.1:
        pt = pop(t, c2 - 0.1)
        with zoom(ctx, 960, 200, pt):
            text(ctx, "旅游投诉录音证据智能整理助手", 960, 200, 56, INK, True)
        pk = pop(t, ph(sh, 2, "我们开工"), 0.5)
        if pk > 0:
            with zoom(ctx, 960, 330, pk):
                text(ctx, "开工！", 960, 330, 120, ORANGE, True)
        tc = c2 + 0.3
        if t > tc:
            for x0, y0, v, rs, cc, ph0 in CONFETTI:
                y = y0 + (t - tc) * v
                if y > H + 20:
                    continue
                x = x0 + 30 * math.sin((t - tc) * 2 + ph0)
                with at(ctx, x, y, 1.0, ph0 + rs * (t - tc)):
                    rrect(ctx, -9, -5, 18, 10, 2, cc)
        pe = fin(t, ce(sh, 2) + 0.4, 0.6)
        if pe > 0:
            rrect(ctx, 960 - 560, 958, 1120, 54, 27, WHITE, 0.85 * pe)
            text(ctx, "本视频由 AI 辅助制作，配音为 AI 合成 · 剧本、人物与机构均为虚构", 960, 985, 24, INK2,
                 False, a=pe)


def make_scenes_b(tl):
    st = tl["stations"]
    return [lambda ctx, t, sh: scene7(ctx, t, sh, st), scene8, scene9, scene10, scene11, scene12]
