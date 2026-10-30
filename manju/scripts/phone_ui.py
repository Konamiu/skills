#!/usr/bin/env python3
"""phone_ui.py - animated phone-screen renderer for vertical short dramas.

Usage:
  uv run --with pillow --with numpy python phone_ui.py <ui.json> <outdir>
      [--only ID[,ID]] [--stills t1,t2,...] [--no-video]

For every screen in the json it writes <outdir>/<id>.mp4 (1080x1920, 24 fps,
H.264, duration = seconds), <outdir>/<id>.png (last frame) and, once all
screens are done, <outdir>/预览-<EPxx>.jpg (contact sheet).

Screen types: chat, mail, post, notifications, call, views.
No real app names / logos are used anywhere.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
from functools import lru_cache

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1080, 1920, 24
RS = Image.Resampling

KO_FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
ZH_FONTS = [
    os.path.expanduser("~/Library/Fonts/LXGWWenKai-Medium.ttf"),
    "/Library/Fonts/LXGWWenKai-Medium.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
]

# ----------------------------------------------------------------- palette
INK = (24, 27, 32)
INK2 = (86, 94, 106)
INK3 = (138, 145, 156)
LINE = (228, 231, 236)
ZH_L = (58, 104, 168)  # translation colour on light backgrounds
ZH_D = (240, 200, 130)  # translation colour on dark backgrounds
CHAT_BG = (186, 206, 224)

SB_H = 130  # status bar height（真 iPhone 的状态栏更高；也给左上角“AI生成”标识留出位置，别压返回箭头）


# ----------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def ease_out_cubic(p):
    p = clamp(p)
    return 1 - (1 - p) ** 3


def ease_out_quint(p):
    p = clamp(p)
    return 1 - (1 - p) ** 5


def ease_out_quad(p):
    p = clamp(p)
    return 1 - (1 - p) ** 2


def ease_in_out(p):
    p = clamp(p)
    return 0.5 - 0.5 * math.cos(math.pi * p)


def ease_out_back(p, s=1.70158):
    p = clamp(p)
    c3 = s + 1
    return 1 + c3 * (p - 1) ** 3 + s * (p - 1) ** 2


# ------------------------------------------------------------------- fonts
@lru_cache(None)
def _ko_index():
    idx = {}
    for i in range(64):
        try:
            f = ImageFont.truetype(KO_FONT, 20, index=i)
        except Exception:
            break
        fam, sty = f.getname()
        if fam == "Apple SD Gothic Neo" and sty not in idx:
            idx[sty] = i
    return idx


@lru_cache(None)
def ko(size, weight="Regular"):
    idx = _ko_index()
    i = idx.get(weight, idx.get("Regular", 0))
    return ImageFont.truetype(KO_FONT, size, index=i)


@lru_cache(None)
def zh(size):
    for p in ZH_FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    raise RuntimeError("no Chinese font found")


def tlen(font, s):
    return font.getlength(s)


# ------------------------------------------------------------ text helpers
def paste_rgba(dst, src, x, y):
    """alpha_composite src onto RGBA dst at (x, y) with clipping."""
    x = int(round(x))
    y = int(round(y))
    sw, sh = src.size
    dw, dh = dst.size
    sx0, sy0 = max(0, -x), max(0, -y)
    sx1, sy1 = min(sw, dw - x), min(sh, dh - y)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    if (sx0, sy0, sx1, sy1) != (0, 0, sw, sh):
        src = src.crop((sx0, sy0, sx1, sy1))
    dst.alpha_composite(src, (x + sx0, y + sy0))


def text(dst, x, y, s, font, fill, anchor="ls", alpha=1.0):
    """Draw text at baseline-left (x, y) onto an RGBA image without fringes."""
    if not s:
        return
    if len(fill) == 4:
        alpha *= fill[3] / 255.0
    l, t, r, b = font.getbbox(s, anchor=anchor)
    pad = 3
    mw, mh = int(r - l + 2 * pad + 1), int(b - t + 2 * pad + 1)
    mask = Image.new("L", (mw, mh), 0)
    ImageDraw.Draw(mask).text((pad - l, pad - t), s, font=font, fill=255, anchor=anchor)
    if alpha < 1:
        mask = mask.point([int(v * alpha) for v in range(256)])
    ov = Image.new("RGBA", (mw, mh), tuple(fill[:3]) + (0,))
    ov.putalpha(mask)
    paste_rgba(dst, ov, x + l - pad, y + t - pad)


@lru_cache(None)
def digit_w(font):
    return max(font.getlength(str(d)) for d in range(10))


def text_tab(dst, x, y, s, font, fill, alpha=1.0):
    """Text with tabular digits (no jitter while counters change). Returns end x."""
    dw = digit_w(font)
    for ch in s:
        if ch.isdigit():
            cw = font.getlength(ch)
            text(dst, x + (dw - cw) / 2, y, ch, font, fill, alpha=alpha)
            x += dw
        else:
            text(dst, x, y, ch, font, fill, alpha=alpha)
            x += font.getlength(ch)
    return x



def draw_num(dst, x, y, s, font, fill, rolling):
    """tabular digits only while a counter is rolling, natural spacing at rest."""
    if rolling:
        return text_tab(dst, x, y, s, font, fill)
    text(dst, x, y, s, font, fill)
    return x + font.getlength(s)


def tab_len(font, s):
    dw = digit_w(font)
    return sum(dw if ch.isdigit() else font.getlength(ch) for ch in s)


def wrap_ko(s, font, maxw):
    out = []
    for para in str(s).split("\n"):
        words = para.split(" ")
        cur = ""
        for w in words:
            trial = (cur + " " + w) if cur else w
            if tlen(font, trial) <= maxw:
                cur = trial
                continue
            if cur:
                out.append(cur)
                cur = ""
            if tlen(font, w) <= maxw:
                cur = w
            else:  # break an over-long word by characters
                part = ""
                for ch in w:
                    if tlen(font, part + ch) > maxw and part:
                        out.append(part)
                        part = ch
                    else:
                        part += ch
                cur = part
        out.append(cur)
    return out or [""]


_NO_START = set("，。、！？；：）」』】》”’…·,.!?;:)%")
_NO_END = set("（「『【《“‘(")


def wrap_zh(s, font, maxw):
    units = re.findall(r"[A-Za-z0-9][A-Za-z0-9.,%:/+\-]*|\s|.", str(s))
    lines, cur = [], ""
    for u in units:
        if cur and tlen(font, cur + u) > maxw and u not in _NO_START and not u.isspace():
            # kinsoku: do not leave an opening bracket at the end of a line
            if cur and cur[-1] in _NO_END:
                lines.append(cur[:-1])
                cur = cur[-1] + u
            else:
                lines.append(cur)
                cur = u
        else:
            cur += u
    if cur:
        lines.append(cur)
    return [l.strip() for l in lines] or [""]


def base_off(size, lh, kind="ko"):
    """baseline offset from the top of a line box of height lh."""
    return lh / 2 + (0.35 if kind == "ko" else 0.34) * size


def draw_lines(dst, x, top, lines, font, fill, lh, size, kind="ko", align="l", width=0, alpha=1.0):
    y = top
    for ln in lines:
        bx = x
        if align == "r":
            bx = x + width - tlen(font, ln)
        elif align == "c":
            bx = x + (width - tlen(font, ln)) / 2
        text(dst, bx, y + base_off(size, lh, kind), ln, font, fill, alpha=alpha)
        y += lh
    return y


# ------------------------------------------------------------ shape helpers
def to_img(im, w, h):
    return im.resize((w, h), RS.LANCZOS)


def solid_mask_layer(size, color, mask):
    lay = Image.new("RGBA", size, tuple(color[:3]) + (0,))
    lay.putalpha(mask)
    return lay


def rr_layer(w, h, r, fill, ss=3, outline=None, ow=0, corners=None):
    im = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle(
        (0, 0, w * ss - 1, h * ss - 1),
        radius=r * ss,
        fill=fill,
        outline=outline,
        width=int(ow * ss),
        corners=corners,
    )
    return to_img(im, w, h)


def circle_layer(d, fill, ss=3):
    im = Image.new("RGBA", (d * ss, d * ss), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse((0, 0, d * ss - 1, d * ss - 1), fill=fill)
    return to_img(im, d, d)


SHADOW_PAD = 20


@lru_cache(None)
def bubble_img(w, h, r, fill, tail=False, shadow=0.16):
    """RGBA bubble with soft shadow; the bubble's top-left sits at (SHADOW_PAD, SHADOW_PAD)."""
    ss = 3
    SP = SHADOW_PAD
    size = ((w + 2 * SP) * ss, (h + 2 * SP) * ss)
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    x0, y0 = SP * ss, SP * ss
    x1, y1 = x0 + w * ss - 1, y0 + h * ss - 1
    d.rounded_rectangle((x0, y0, x1, y1), radius=r * ss, fill=255)
    if tail:
        d.rectangle((x0, y0, x0 + r * ss, y0 + r * ss), fill=255)  # square corner
        d.polygon(
            [(x0 + 4 * ss, y0), (x0 - 15 * ss, y0 + 1 * ss), (x0 - 2 * ss, y0 + 15 * ss),
             (x0 + 4 * ss, y0 + 34 * ss)],
            fill=255,
        )
    base = Image.new("RGBA", size, (0, 0, 0, 0))
    if shadow:
        sh = mask.filter(ImageFilter.GaussianBlur(7 * ss))
        sh = ImageChops.offset(sh, 0, 4 * ss)
        sh = sh.point([int(v * shadow) for v in range(256)])
        base.alpha_composite(solid_mask_layer(size, (36, 56, 88), sh))
    base.alpha_composite(solid_mask_layer(size, fill, mask))
    return to_img(base, w + 2 * SP, h + 2 * SP)


# ------------------------------------------------------------------- icons
@lru_cache(None)
def icon(name, size, color=(24, 27, 32), lw=7.0):
    ss = 4
    S = size * ss
    u = S / 100.0
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = tuple(color[:3]) + ((color[3],) if len(color) == 4 else (255,))

    def P(pts):
        return [(x * u, y * u) for x, y in pts]

    def ln(*pts, w=None):
        w = lw if w is None else w
        p = P(pts)
        d.line(p, fill=c, width=max(1, int(w * u)), joint="curve")
        r = w * u / 2
        for x, y in (p[0], p[-1]):
            d.ellipse((x - r, y - r, x + r, y + r), fill=c)

    def dot(cx, cy, r):
        d.ellipse(((cx - r) * u, (cy - r) * u, (cx + r) * u, (cy + r) * u), fill=c)

    def ring(cx, cy, r, w=None):
        w = lw if w is None else w
        d.ellipse(((cx - r) * u, (cy - r) * u, (cx + r) * u, (cy + r) * u), outline=c, width=int(w * u))

    def rrect(x0, y0, x1, y1, r, fill=False, w=None):
        w = lw if w is None else w
        d.rounded_rectangle((x0 * u, y0 * u, x1 * u, y1 * u), radius=r * u,
                            fill=c if fill else None, outline=None if fill else c,
                            width=0 if fill else int(w * u))

    def arc(box, a0, a1, w=None):
        w = lw if w is None else w
        d.arc(tuple(v * u for v in box), a0, a1, fill=c, width=int(w * u))

    n = name
    if n == "back":
        ln((64, 16), (30, 50), (64, 84))
    elif n == "chev_down":
        ln((28, 38), (50, 62), (72, 38))
    elif n == "search":
        ring(43, 43, 25)
        ln((62, 62), (84, 84))
    elif n == "menu3":
        for y in (28, 50, 72):
            ln((18, y), (82, y))
    elif n == "kebab":
        for y in (18, 50, 82):
            dot(50, y, 7.5)
    elif n == "plus":
        ln((50, 18), (50, 82))
        ln((18, 50), (82, 50))
    elif n == "smile":
        ring(50, 50, 36, w=lw * 0.85)
        dot(37, 42, 4.5)
        dot(63, 42, 4.5)
        arc((27, 28, 73, 76), 30, 150, w=lw * 0.85)
    elif n == "hash":
        ln((40, 20), (32, 80))
        ln((68, 20), (60, 80))
        ln((18, 38), (84, 38))
        ln((16, 62), (82, 62))
    elif n == "trash":
        ln((18, 28), (82, 28))
        ln((38, 28), (38, 17), (62, 17), (62, 28))
        ln((26, 28), (32, 86), (68, 86), (74, 28))
        ln((43, 44), (44, 70), w=lw * 0.8)
        ln((57, 44), (56, 70), w=lw * 0.8)
    elif n == "mail":
        rrect(12, 22, 88, 78, 8)
        ln((16, 28), (50, 56), (84, 28), w=lw * 0.9)
    elif n == "heart":
        pts = []
        for i in range(120):
            t = i / 120 * 2 * math.pi
            x = 16 * math.sin(t) ** 3
            y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
            pts.append((50 + x * 2.55, 46 - y * 2.55))
        p = P(pts + [pts[0]])
        d.line(p, fill=c, width=int(lw * u), joint="curve")
    elif n == "comment":
        rrect(10, 14, 90, 70, 20)
        ln((32, 69), (26, 88), (50, 70), w=lw * 0.9)
    elif n == "share":
        ln((34, 42), (18, 42), (18, 86), (82, 86), (82, 42), (66, 42))
        ln((50, 66), (50, 14))
        ln((33, 30), (50, 13), (67, 30))
    elif n == "eye":
        d.ellipse((8 * u, 26 * u, 92 * u, 74 * u), outline=c, width=int(lw * 0.85 * u))
        dot(50, 50, 13)
    elif n == "mic":
        rrect(37, 8, 63, 56, 13)
        arc((24, 30, 76, 76), 0, 180)
        ln((50, 76), (50, 90))
        ln((36, 91), (64, 91))
    elif n == "mic_off":
        rrect(37, 8, 63, 56, 13)
        arc((24, 30, 76, 76), 0, 180)
        ln((50, 76), (50, 90))
        ln((36, 91), (64, 91))
        ln((16, 12), (84, 88), w=lw * 1.1)
    elif n == "keypad":
        for yy in (24, 50, 76):
            for xx in (24, 50, 76):
                dot(xx, yy, 7)
    elif n == "speaker":
        d.polygon(P([(12, 38), (32, 38), (54, 18), (54, 82), (32, 62), (12, 62)]), fill=c)
        arc((34, 34, 74, 66), -55, 55, w=lw * 0.95)
        arc((30, 20, 92, 80), -55, 55, w=lw * 0.95)
    elif n == "hangup":
        arc((6, 18, 94, 108), 212, 328, w=17)
        for cx in (17, 83):
            d.rounded_rectangle(((cx - 12) * u, 46 * u, (cx + 12) * u, 72 * u), radius=7 * u, fill=c)
    elif n == "lock":
        rrect(26, 44, 74, 86, 8, fill=True)
        arc((33, 12, 67, 62), 180, 360, w=lw * 0.9)
        ln((34.5, 36), (34.5, 46), w=lw * 0.9)
        ln((65.5, 36), (65.5, 46), w=lw * 0.9)
    elif n == "camera":
        rrect(10, 28, 90, 80, 11)
        ring(50, 54, 15)
        ln((34, 28), (40, 18), (60, 18), (66, 28), w=lw * 0.9)
    elif n == "flash":
        ln((30, 10), (70, 10), (62, 36), (38, 36), (30, 10))
        rrect(38, 36, 62, 92, 7)
        ln((50, 50), (50, 62), w=lw * 0.9)
    elif n == "house":
        d.polygon(P([(50, 12), (88, 46), (76, 46), (76, 86), (24, 86), (24, 46), (12, 46)]), fill=c)
        # door cut-out
        d.rectangle((43 * u, 60 * u, 57 * u, 86 * u), fill=(0, 0, 0, 0))
    elif n == "play":
        d.polygon(P([(26, 12), (26, 88), (90, 50)]), fill=c)
    elif n == "reply":
        ln((42, 22), (18, 46), (42, 70))
        ln((20, 46), (62, 46), (84, 62), (84, 82))
    elif n == "forward":
        ln((58, 22), (82, 46), (58, 70))
        ln((80, 46), (38, 46), (16, 62), (16, 82))
    elif n == "reply_all":
        ln((34, 24), (12, 46), (34, 68))
        ln((58, 24), (36, 46), (58, 68))
        ln((38, 46), (70, 46), (88, 62), (88, 80))
    return to_img(im, size, size)


@lru_cache(None)
def avatar_img(size, bg=(170, 177, 188), fg=(236, 239, 244)):
    ss = 4
    S = size * ss
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse((0, 0, S - 1, S - 1), fill=tuple(bg) + (255,))
    lay = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    dl = ImageDraw.Draw(lay)
    cx = S / 2
    dl.ellipse((cx - 0.185 * S, 0.215 * S, cx + 0.185 * S, 0.585 * S), fill=tuple(fg) + (255,))
    dl.ellipse((cx - 0.355 * S, 0.645 * S, cx + 0.355 * S, 1.27 * S), fill=tuple(fg) + (255,))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, S - 1, S - 1), fill=255)
    lay.putalpha(ImageChops.multiply(lay.getchannel("A"), mask))
    im.alpha_composite(lay)
    return to_img(im, size, size)


# ------------------------------------------------------------ status bar etc.
def fmt_clock_ko(clock, minus=0):
    h, m = [int(x) for x in str(clock).split(":")[:2]]
    tot = (h * 60 + m - minus) % (24 * 60)
    h, m = divmod(tot, 60)
    ap = "오전" if h < 12 else "오후"
    return f"{ap} {(h % 12) or 12}:{m:02d}"


@lru_cache(None)
def status_icons(color):
    """signal / wifi / battery cluster, right aligned inside a W x SB_H RGBA."""
    ss = 4
    im = Image.new("RGBA", (W * ss, SB_H * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = tuple(color) + (255,)
    cd = tuple(color) + (110,)
    cy = 50
    # battery
    bx1 = W - 44
    bw, bh = 58, 29
    bx0 = bx1 - bw - 5
    d.rounded_rectangle((bx0 * ss, (cy - bh / 2) * ss, (bx0 + bw) * ss, (cy + bh / 2) * ss),
                        radius=8 * ss, outline=cd, width=int(3 * ss))
    fillw = (bw - 9) * 0.78
    d.rounded_rectangle(((bx0 + 4.5) * ss, (cy - bh / 2 + 4.5) * ss, (bx0 + 4.5 + fillw) * ss,
                         (cy + bh / 2 - 4.5) * ss), radius=4 * ss, fill=c)
    d.rounded_rectangle(((bx0 + bw + 2) * ss, (cy - 6) * ss, (bx0 + bw + 6) * ss, (cy + 6) * ss),
                        radius=2 * ss, fill=cd)
    # wifi
    wx = bx0 - 30
    wy = cy + 15
    for R, inner in ((32, 25), (22, 15), (11, 0)):
        box = ((wx - R) * ss, (wy - R) * ss, (wx + R) * ss, (wy + R) * ss)
        d.pieslice(box, 227, 313, fill=c)
        if inner:
            bi = ((wx - inner) * ss, (wy - inner) * ss, (wx + inner) * ss, (wy + inner) * ss)
            d.pieslice(bi, 220, 320, fill=(0, 0, 0, 0))
    # signal
    sx = wx - 32 - 4 * 11 - 22
    hs = [11, 18, 25, 32]
    for i, hh in enumerate(hs):
        x0 = sx + i * 13
        d.rounded_rectangle((x0 * ss, (cy + 16 - hh) * ss, (x0 + 8) * ss, (cy + 16) * ss),
                            radius=2 * ss, fill=c if i < 3 else cd)
    return to_img(im, W, SB_H)


def status_bar(clock, color=INK, bg=None):
    im = Image.new("RGBA", (W, SB_H), tuple(bg) + (255,) if bg else (0, 0, 0, 0))
    text(im, 54, 62, str(clock), ko(36, "SemiBold"), color)
    im.alpha_composite(status_icons(tuple(color)))
    return im


def home_bar(color=INK, alpha=190):
    im = Image.new("RGBA", (W, 40), (0, 0, 0, 0))
    im.alpha_composite(rr_layer(330, 12, 6, tuple(color) + (alpha,)), (375, 12))
    return im


def gradient_bg(top, bottom, blobs=(), noise=0.9, seed=3):
    yv = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    arr = (np.array(top, np.float32) * (1 - yv) + np.array(bottom, np.float32) * yv)
    arr = np.broadcast_to(arr, (H, W, 3)).copy()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for cx, cy, r, color, strength in blobs:
        dd = np.sqrt((xx - cx * W) ** 2 + (yy - cy * H) ** 2) / (r * W)
        a = (np.exp(-dd * dd * 1.6) * strength)[..., None]
        arr += a * (np.array(color, np.float32) - arr)
    if noise:
        arr += np.random.default_rng(seed).normal(0, noise, (H, W, 1)).astype(np.float32)
    return Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8), "RGB")


# --------------------------------------------------------------- sprites
class Sprite:
    """RGBA bitmap cropped to its content, with an anchor for scaling."""

    def __init__(self, img, ox, oy, ax=None, ay=None):
        bb = img.getchannel("A").getbbox()
        if bb is None:
            bb = (0, 0, 1, 1)
        self.img = img.crop(bb)
        self.ox = ox + bb[0]
        self.oy = oy + bb[1]
        self.ax = self.ox if ax is None else ax
        self.ay = self.oy if ay is None else ay
        self.w, self.h = self.img.size


_ALPHA_LUT = {}


def _lut(a):
    k = int(round(a * 64))
    if k not in _ALPHA_LUT:
        _ALPHA_LUT[k] = [int(v * k / 64.0) for v in range(256)]
    return _ALPHA_LUT[k]


def blit(canvas, sp, X, Y, scale=1.0, alpha=1.0):
    """paste sprite on RGB canvas; (X, Y) is the item origin in screen coordinates."""
    if alpha <= 0.003:
        return
    im = sp.img
    if abs(scale - 1.0) > 1e-3:
        nw = max(1, int(round(im.width * scale)))
        nh = max(1, int(round(im.height * scale)))
        im = im.resize((nw, nh), RS.BICUBIC)
    px = X + sp.ax + (sp.ox - sp.ax) * scale
    py = Y + sp.ay + (sp.oy - sp.ay) * scale
    if alpha < 0.997:
        im = im.copy()
        im.putalpha(im.getchannel("A").point(_lut(alpha)))
    canvas.paste(im, (int(round(px)), int(round(py))), im)


POP = 0.18


def pop_params(p):
    """(scale, alpha) for a bubble popping in; p = seconds since appearance / POP."""
    p = clamp(p)
    scale = 0.9 + 0.1 * ease_out_back(p)
    alpha = min(1.0, 0.28 + 0.72 * (p / 0.55))
    return scale, alpha


# ------------------------------------------------------------ base screen
class Base:
    def __init__(self, spec):
        self.spec = spec
        self.id = spec["id"]
        self.seconds = float(spec["seconds"])
        self.nframes = max(1, int(round(self.seconds * FPS)))
        self.clock = spec.get("clock", "09:00")

    def events(self):
        return []

    def frame(self, t):
        raise NotImplementedError

    def key_times(self):
        """4 preview moments: start, up to two 'after new item appears' moments, end."""
        end = (self.nframes - 1) / FPS
        evs = sorted(set(round(e + 0.45, 3) for e in self.events() if e + 0.45 < end - 0.05))
        n = len(evs)
        if n > 2:
            evs = [evs[int(round((n - 1) / 3.0))], evs[int(round(2 * (n - 1) / 3.0))]]
        for c in (self.seconds * 0.35, self.seconds * 0.65):
            if len(evs) < 2 and all(abs(c - e) > 0.3 for e in evs):
                evs.append(round(c, 3))
        return [0.04] + sorted(evs)[:2] + [end]


def make_backdrop(color):
    return Image.new("RGB", (W, H), color)


def dim(img, k):
    return Image.blend(img, Image.new("RGB", img.size, (0, 0, 0)), k)


# ------------------------------------------------------------------- chat
AV = 96
AV_X = 34
BUB_X = AV_X + AV + 18
NAME_H = 56
PADX = 34
PADY = 20
LH_KO = 66
LH_ZH = 52
BUB_MAX = 776
ZH_MAX = 806
KO_SIZE = 48
ZH_SIZE = 38
TYPING_W, TYPING_H = 150, 106
GAP_SAME = 22
GAP_GROUP = 38
CHAT_TOP = SB_H + 128  # header bottom
CHAT_LIMIT = int(0.70 * H)


class Item:
    pass


class ChatScreen(Base):
    def __init__(self, spec):
        super().__init__(spec)
        self.bg = CHAT_BG
        self.contact = spec.get("contact", {})
        self.cname = self.contact.get("name", "")
        self.cname_zh = self.contact.get("name_zh", "")
        self.bg_img = Image.new("RGB", (W, H), self.bg)
        self._build_chrome()
        self._build_items(spec["messages"])
        self._build_scroll()

    # ---- static chrome
    def _build_chrome(self):
        hdr = Image.new("RGBA", (W, CHAT_TOP), self.bg + (255,))
        hdr.alpha_composite(status_bar(self.clock, INK))
        cy = SB_H + 64
        hdr.alpha_composite(icon("back", 54, (30, 36, 44), 7), (30, cy - 27))
        x = 112
        f1, f2 = ko(46, "SemiBold"), zh(34)
        text(hdr, x, cy + 17, self.cname, f1, (22, 28, 36))
        x2 = x + tlen(f1, self.cname) + 22
        text(hdr, x2, cy + 16, self.cname_zh, f2, ZH_L)
        hdr.alpha_composite(icon("search", 52, (30, 36, 44), 7), (866, cy - 26))
        hdr.alpha_composite(icon("menu3", 52, (30, 36, 44), 7), (968, cy - 26))
        self.header = hdr
        # soft fade below header
        fade = Image.new("RGBA", (W, 30), self.bg + (0,))
        a = np.tile(np.linspace(235, 0, 30, dtype=np.float32)[:, None], (1, W))
        fade.putalpha(Image.fromarray(a.astype(np.uint8), "L"))
        self.fade = fade
        # footer (input bar)
        fh = H - 1722
        ft = Image.new("RGBA", (W, fh), (251, 252, 253, 255))
        ft.paste((222, 227, 233), (0, 0, W, 2))
        cyf = 78
        ft.alpha_composite(icon("plus", 46, (110, 118, 130), 6.5), (38, cyf - 23))
        ft.alpha_composite(rr_layer(742, 88, 44, (238, 241, 245, 255)), (110, cyf - 44))
        ft.alpha_composite(icon("smile", 50, (130, 138, 150), 6), (770, cyf - 25))
        ft.alpha_composite(circle_layer(88, (238, 241, 245, 255)), (876, cyf - 44))
        ft.alpha_composite(icon("hash", 42, (110, 118, 130), 7), (899, cyf - 21))
        ft.alpha_composite(home_bar(INK), (0, fh - 44))
        self.footer = ft
        self.footer_y = 1722
        # date chip
        f = ko(28, "Medium")
        fz = zh(26)
        t1, t2 = "오늘", "今天"
        w1, w2 = tlen(f, t1), tlen(fz, t2)
        cw = int(w1 + w2 + 22 + 56)
        chip = Image.new("RGBA", (cw, 58), (0, 0, 0, 0))
        chip.alpha_composite(rr_layer(cw, 58, 29, (64, 86, 116, 120)))
        text(chip, 28, 40, t1, f, (255, 255, 255))
        text(chip, 28 + w1 + 22, 39, t2, fz, ZH_D)
        self.chip = Sprite(chip, (W - cw) // 2, 0)
        self.chip_y = CHAT_TOP + 26

    # ---- items
    def _build_items(self, msgs):
        items = []
        y = self.chip_y + 58 + 34
        for i, m in enumerate(msgs):
            it = Item()
            it.m = m
            it.side = "out" if (m.get("me") or m.get("from") in ("me", "나")) else "in"
            it.at = m.get("at")
            it.typing = m.get("typing") if it.at is not None else None
            it.first = i == 0 or items[-1].side != it.side
            it.has_head = it.side == "in" and it.first
            it.time = m.get("time") or fmt_clock_ko(self.clock, 1 if it.at is None else 0)
            items.append(it)
        for i, it in enumerate(items):
            nxt = items[i + 1] if i + 1 < len(items) else None
            it.show_time = nxt is None or nxt.side != it.side or nxt.time != it.time
        for i, it in enumerate(items):
            if i:
                y += GAP_GROUP if it.first else GAP_SAME
            it.y = y
            btop = NAME_H if it.has_head else 0
            it.btop = btop
            self._build_body(it)
            if it.has_head:
                self._build_head(it)
            if it.at is not None and it.typing is not None:
                self._build_typing_base(it)
            it.bottom = y + it.body_bottom
            it.typing_bottom = y + btop + TYPING_H
            y = it.bottom
        self.items = items

    def _build_head(self, it):
        im = Image.new("RGBA", (BUB_X + 700, AV + 4), (0, 0, 0, 0))
        im.alpha_composite(avatar_img(AV), (AV_X, 0))
        f1, f2 = ko(32, "Medium"), zh(28)
        text(im, BUB_X + 4, 30, self.cname, f1, (52, 62, 76))
        text(im, BUB_X + 4 + tlen(f1, self.cname) + 16, 30, self.cname_zh, f2, ZH_L)
        it.head = Sprite(im, 0, 0, AV_X + AV / 2, AV / 2)

    def _build_body(self, it):
        m = it.m
        SP = SHADOW_PAD
        in_side = it.side == "in"
        fill = (255, 255, 255, 255) if in_side else (255, 232, 18, 255)
        if m.get("image"):
            self._build_image_body(it, fill)
            return
        kf = ko(KO_SIZE, "Regular")
        lines = wrap_ko(m.get("ko", ""), kf, BUB_MAX - 2 * PADX)
        tw = max(tlen(kf, l) for l in lines)
        bw = int(math.ceil(tw)) + 2 * PADX
        bw = max(bw, 110)
        bh = len(lines) * LH_KO + 2 * PADY
        bx = BUB_X if in_side else W - 36 - bw
        zf = zh(ZH_SIZE)
        zlines = wrap_zh(m.get("zh", ""), zf, ZH_MAX) if m.get("zh") else []
        zh_h = (10 + len(zlines) * LH_ZH) if zlines else 0
        total_h = bh + zh_h + 2 * SP
        im = Image.new("RGBA", (W, total_h), (0, 0, 0, 0))
        bimg = bubble_img(bw, bh, 38, fill, in_side and it.first)
        im.alpha_composite(bimg, (bx - SP, 0))
        draw_lines(im, bx + PADX, SP + PADY, lines, kf, (22, 24, 28), LH_KO, KO_SIZE)
        if zlines:
            zx = bx if in_side else bx + bw - max(tlen(zf, l) for l in zlines)
            draw_lines(im, zx, SP + bh + 10, zlines, zf, ZH_L, LH_ZH, ZH_SIZE, kind="zh")
        self._time_label(im, it, bx, bw, SP, bh, in_side)
        it.body = Sprite(im, 0, it.btop - SP, bx, it.btop)
        it.body_bottom = it.btop + bh + zh_h
        it.bubble_geom = (bx, bw, bh)

    def _time_label(self, im, it, bx, bw, SP, bh, in_side):
        if not it.show_time:
            return
        tf = ko(25, "Regular")
        tcol = (84, 100, 120)
        if in_side:
            text(im, bx + bw + 14, SP + bh - 12, it.time, tf, tcol)
        else:
            text(im, bx - 14 - tlen(tf, it.time), SP + bh - 12, it.time, tf, tcol)

    def _build_image_body(self, it, fill):
        m = it.m
        SP = SHADOW_PAD
        in_side = it.side == "in"
        src = Image.open(m["image"]).convert("RGB")
        c = m.get("image_crop") or [0, 0, 1, 1]
        sw, sh = src.size
        box = (int(c[0] * sw), int(c[1] * sh), int(c[2] * sw), int(c[3] * sh))
        crop = src.crop(box)
        bw = int(round(W * 0.62))
        bh = int(round(bw * crop.height / crop.width))
        bx = BUB_X if in_side else W - 36 - bw
        photo = crop.resize((bw, bh), RS.LANCZOS)
        # rounded mask supersampled
        ss = 3
        mk = Image.new("L", (bw * ss, bh * ss), 0)
        ImageDraw.Draw(mk).rounded_rectangle((0, 0, bw * ss - 1, bh * ss - 1), radius=34 * ss, fill=255)
        mk = mk.resize((bw, bh), RS.LANCZOS)
        zf = zh(ZH_SIZE)
        zlines = wrap_zh(m.get("zh", ""), zf, ZH_MAX) if m.get("zh") else []
        zh_h = (10 + len(zlines) * LH_ZH) if zlines else 0
        total_h = bh + zh_h + 2 * SP
        im = Image.new("RGBA", (W, total_h), (0, 0, 0, 0))
        # shadow
        shm = Image.new("L", (bw + 2 * SP, bh + 2 * SP), 0)
        shm.paste(mk, (SP, SP))
        shm = shm.filter(ImageFilter.GaussianBlur(7))
        shm = ImageChops.offset(shm, 0, 4).point([int(v * 0.22) for v in range(256)])
        im.alpha_composite(solid_mask_layer(shm.size, (36, 56, 88), shm), (bx - SP, 0))
        ph = photo.convert("RGBA")
        ph.putalpha(mk)
        im.alpha_composite(ph, (bx, SP))
        if zlines:
            zx = bx if in_side else bx + bw - max(tlen(zf, l) for l in zlines)
            draw_lines(im, zx, SP + bh + 10, zlines, zf, ZH_L, LH_ZH, ZH_SIZE, kind="zh")
        self._time_label(im, it, bx, bw, SP, bh, in_side)
        it.body = Sprite(im, 0, it.btop - SP, bx, it.btop)
        it.body_bottom = it.btop + bh + zh_h
        it.bubble_geom = (bx, bw, bh)

    def _build_typing_base(self, it):
        SP = SHADOW_PAD
        in_side = it.side == "in"
        bx = BUB_X if in_side else W - 36 - TYPING_W
        it.typing_x = bx
        it.typing_img = bubble_img(TYPING_W, TYPING_H, 38, (255, 255, 255, 255) if in_side else (255, 232, 18, 255),
                                   in_side and it.first)

    def _typing_sprite(self, it, t):
        SP = SHADOW_PAD
        base = it.typing_img.copy()
        ss = 4
        w, h = TYPING_W, TYPING_H
        lay = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        for i in range(3):
            ph = t / 0.95 - i * 0.17
            ph = ph - math.floor(ph)
            b = math.sin(math.pi * ph / 0.55) if ph < 0.55 else 0.0
            cx = (w / 2 + (i - 1) * 32) * ss
            cy = (h / 2 - 11 * b) * ss
            g = int(176 - 74 * b)
            r = 9.5 * ss
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(g, g + 4, g + 12, 255))
        lay = to_img(lay, w, h)
        base.alpha_composite(lay, (SP, SP))
        return Sprite(base, it.typing_x - SP, it.btop - SP, it.typing_x, it.btop)

    # ---- scroll model
    def _build_scroll(self):
        static_bottoms = [it.bottom for it in self.items if it.at is None]
        self.s0 = max(0.0, max(static_bottoms) - CHAT_LIMIT) if static_bottoms else 0.0
        evs = []
        for it in self.items:
            if it.at is None:
                continue
            if it.typing is not None:
                evs.append((it.typing, it.typing_bottom))
            evs.append((it.at, it.bottom))
        evs.sort(key=lambda e: e[0])
        out = []
        cur = self.s0
        for te, bottom in evs:
            tgt = max(cur, bottom - CHAT_LIMIT)
            out.append((te, tgt))
            cur = tgt
        self.scroll_events = out

    def scroll_at(self, t):
        evs = self.scroll_events

        def f(k, tt):
            if k < 0:
                return self.s0
            te, tgt = evs[k]
            if tt < te:
                return f(k - 1, tt)
            start = f(k - 1, te)
            return start + (tgt - start) * ease_out_cubic((tt - te) / 0.25)

        return f(len(evs) - 1, t)

    def events(self):
        return [it.at for it in self.items if it.at is not None]

    # ---- frame
    def frame(self, t):
        img = self.bg_img.copy()
        sc = self.scroll_at(t)
        blit(img, self.chip, 0, self.chip_y - sc)
        for it in self.items:
            Y = it.y - sc
            if Y > H or Y + (it.bottom - it.y) < -40:
                continue
            if it.at is None:
                if it.has_head:
                    blit(img, it.head, 0, Y)
                blit(img, it.body, 0, Y)
                continue
            t_head = it.typing if it.typing is not None else it.at
            if it.has_head and t >= t_head:
                s, a = pop_params((t - t_head) / POP)
                blit(img, it.head, 0, Y, s, a)
            if it.typing is not None and it.typing <= t < it.at + 0.14:
                s, a = pop_params((t - it.typing) / POP)
                if t >= it.at:
                    a *= max(0.0, 1 - (t - it.at) / 0.12)
                blit(img, self._typing_sprite(it, t), 0, Y, s, a)
            if t >= it.at:
                s, a = pop_params((t - it.at) / POP)
                blit(img, it.body, 0, Y, s, a)
        img.paste(self.fade, (0, CHAT_TOP), self.fade)
        img.paste(self.header, (0, 0))
        img.paste(self.footer, (0, self.footer_y), self.footer)
        return img


# ------------------------------------------------------------------- mail
class MailScreen(Base):
    def __init__(self, spec):
        super().__init__(spec)
        self.open_at = float(spec.get("open_at", 0.0))
        self.page = self._build()
        self.back = make_backdrop((30, 33, 40))

    def _build(self):
        s = self.spec
        pg = Image.new("RGBA", (W, H), (255, 255, 255, 255))
        pg.alpha_composite(status_bar(self.clock, INK))
        # app bar
        cy = SB_H + 60
        pg.alpha_composite(icon("back", 54, INK, 7), (34, cy - 27))
        for i, nm in enumerate(("trash", "mail", "kebab")):
            pg.alpha_composite(icon(nm, 52, (60, 66, 78), 6.5), (760 + i * 100, cy - 26))
        y = SB_H + 120 + 30
        # subject
        fs, fz = ko(56, "Bold"), zh(40)
        sl = wrap_ko(s.get("subject", ""), fs, 984)
        y = draw_lines(pg, 48, y, sl, fs, INK, 74, 56)
        if s.get("subject_zh"):
            zl = wrap_zh(s["subject_zh"], fz, 984)
            y = draw_lines(pg, 48, y + 2, zl, fz, ZH_L, 56, 40, kind="zh")
        # inbox chip
        y += 20
        chip_f = ko(26, "Medium")
        chip_t = "받은편지함"
        cw = int(tlen(chip_f, chip_t) + 40)
        pg.alpha_composite(rr_layer(cw, 48, 24, (232, 238, 247, 255)), (48, int(y)))
        text(pg, 68, int(y) + 33, chip_t, chip_f, (74, 98, 140))
        y += 48 + 38
        # sender row
        pg.alpha_composite(avatar_img(116, (92, 118, 156), (232, 238, 248)), (48, int(y)))
        nf, nz = ko(44, "Bold"), zh(34)
        text(pg, 188, int(y) + 50, s.get("from", ""), nf, INK)
        if s.get("from_zh"):
            text(pg, 188 + tlen(nf, s.get("from", "")) + 20, int(y) + 49, s["from_zh"], nz, ZH_L)
        sub = ko(30, "Regular")
        text(pg, 188, int(y) + 100, "나에게", sub, INK3)
        pg.alpha_composite(icon("chev_down", 30, INK3, 9), (188 + int(tlen(sub, "나에게")) + 8, int(y) + 76))
        tm = fmt_clock_ko(self.clock)
        text(pg, W - 48 - tlen(sub, tm), int(y) + 50, tm, sub, INK3)
        y += 116 + 34
        pg.paste(LINE, (48, int(y), W - 48, int(y) + 2))
        y += 2 + 38
        # body
        bf, bz = ko(48, "Regular"), zh(40)
        bodies = s.get("body")
        zhs = s.get("body_zh")
        if isinstance(bodies, str):
            bodies = [bodies]
            zhs = [zhs] if zhs else [None]
        for i, b in enumerate(bodies or []):
            y = draw_lines(pg, 48, y, wrap_ko(b, bf, 984), bf, (28, 30, 34), 68, 48)
            z = (zhs or [None] * 9)[i] if zhs else None
            if z:
                y = draw_lines(pg, 48, y + 4, wrap_zh(z, bz, 984), bz, ZH_L, 56, 40, kind="zh")
            y += 34
        # action buttons
        y += 26
        bx = 48
        for nm, lab in (("reply", "답장"), ("reply_all", "전체답장"), ("forward", "전달")):
            lf = ko(32, "Medium")
            wbtn = int(tlen(lf, lab) + 96)
            pg.alpha_composite(rr_layer(wbtn, 88, 44, (255, 255, 255, 255), outline=(206, 212, 222, 255), ow=2.5),
                               (bx, int(y)))
            pg.alpha_composite(icon(nm, 40, (84, 92, 106), 8), (bx + 24, int(y) + 24))
            text(pg, bx + 24 + 40 + 12, int(y) + 57, lab, lf, (60, 68, 82))
            bx += wbtn + 20
        self.content_bottom = y + 88
        pg.alpha_composite(home_bar(INK), (0, H - 44))
        return pg

    def frame(self, t):
        p = (t - self.open_at) / 0.25
        e = ease_out_cubic(p)
        off = int(round((1 - e) * 0.22 * H))
        if off <= 0:
            return self.page.convert("RGB")
        img = self.back.copy()
        # rounded top corners while sliding
        pg = self.page
        mask = rr_layer(W, H, 44, (255, 255, 255, 255))
        sheet = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sheet.alpha_composite(pg)
        sheet.putalpha(ImageChops.multiply(pg.getchannel("A"), mask.getchannel("A")))
        img.paste(sheet, (0, off), sheet)
        return img


# ------------------------------------------------------------------- post
def fmt_wan(v, rolling=False):
    x = v / 10000.0
    if rolling:
        x = math.floor(x * 10 + 1e-6) / 10.0
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x))}"
    return f"{x:.1f}"


class PostScreen(Base):
    def __init__(self, spec):
        super().__init__(spec)
        self.vf = spec.get("views_from")
        self.vt = spec.get("views")
        self.has_views = self.vt is not None
        if self.vf is None:
            self.vf = self.vt
        self.views_zh = spec.get("views_zh") or "浏览 {n} 万"
        self.back = make_backdrop((206, 210, 217))
        self._build()

    def _build(self):
        s = self.spec
        pg = Image.new("RGBA", (W, H), (255, 255, 255, 255))
        pg.alpha_composite(status_bar(self.clock, INK))
        cy = SB_H + 60
        pg.alpha_composite(icon("back", 54, INK, 7), (34, cy - 27))
        f1, f2 = ko(44, "SemiBold"), zh(32)
        board = s.get("board", "")
        text(pg, 110, cy + 16, board, f1, INK)
        text(pg, 110 + tlen(f1, board) + 20, cy + 15, s.get("board_zh", ""), f2, ZH_L)
        for i, nm in enumerate(("search", "share", "kebab")):
            pg.alpha_composite(icon(nm, 50, (60, 66, 78), 6.5), (800 + i * 90, cy - 25))
        pg.paste((226, 230, 236), (0, SB_H + 120, W, SB_H + 122))
        y = SB_H + 122 + 38
        # title
        tf, tz = ko(54, "Bold"), zh(40)
        y = draw_lines(pg, 48, y, wrap_ko(s.get("title", ""), tf, 984), tf, INK, 72, 54)
        if s.get("title_zh"):
            y = draw_lines(pg, 48, y + 2, wrap_zh(s["title_zh"], tz, 984), tz, ZH_L, 56, 40, kind="zh")
        y += 26
        # author row
        pg.alpha_composite(avatar_img(66, (200, 205, 214), (244, 246, 250)), (48, int(y)))
        af, az = ko(34, "Medium"), zh(28)
        text(pg, 128, int(y) + 44, "익명", af, INK2)
        text(pg, 128 + tlen(af, "익명") + 14, int(y) + 43, "匿名", az, ZH_L)
        y += 66 + 24
        self.views_slot = None
        if self.has_views:
            pg.alpha_composite(icon("eye", 38, INK3, 7), (48, int(y) + 6))
            vf = ko(36, "Medium")
            vz = zh(32)
            ko_label = "조회수 "
            digits = max(len(str(int(max(self.vf, self.vt) // 10000))), 1)
            maxw = tab_len(vf, ko_label + "8" * digits + ".8만")
            self.views_slot = dict(x=100, y=int(y) + 38, vf=vf, vz=vz, zx=100 + maxw + 22, label=ko_label)
            y += 52
        y += 18
        pg.paste((234, 237, 242), (48, int(y), W - 48, int(y) + 2))
        y += 2 + 36
        # body
        bf, bz = ko(46, "Regular"), zh(38)
        body = s.get("body", [])
        for blk in body:
            if isinstance(blk, str):
                blk = {"ko": blk}
            y = draw_lines(pg, 48, y, wrap_ko(blk.get("ko", ""), bf, 984), bf, (30, 32, 36), 64, 46)
            if blk.get("zh"):
                y = draw_lines(pg, 48, y + 4, wrap_zh(blk["zh"], bz, 984), bz, ZH_L, 52, 38, kind="zh")
            y += 34
        self.content_bottom = y
        # comment bar
        fh = H - 1722
        pg.paste((247, 248, 250), (0, 1722, W, H))
        pg.paste((226, 230, 236), (0, 1722, W, 1724))
        pg.alpha_composite(rr_layer(820, 84, 42, (235, 238, 243, 255)), (40, 1722 + 34))
        text(pg, 80, 1722 + 34 + 55, "댓글을 입력하세요", ko(32, "Regular"), (150, 158, 170))
        pg.alpha_composite(icon("comment", 54, (96, 104, 118), 6.5), (900, 1722 + 34 + 15))
        pg.alpha_composite(icon("heart", 54, (96, 104, 118), 6.5), (980, 1722 + 34 + 15))
        pg.alpha_composite(home_bar(INK), (0, H - 44))
        self.page = pg

    def _views_text(self, t):
        if not self.has_views:
            return None
        dur = 1.5
        p = clamp(t / dur)
        if p >= 1:
            return fmt_wan(self.vt)
        v = self.vf + (self.vt - self.vf) * ease_out_quad(p)
        if p <= 0:
            return fmt_wan(self.vf)
        return fmt_wan(v, rolling=True)

    def frame(self, t):
        pg = self.page.copy()
        vt = self._views_text(t)
        if vt is not None:
            sl = self.views_slot
            rolling = 0 < t < 1.5
            draw_num(pg, sl["x"], sl["y"], sl["label"] + vt + "만", sl["vf"], INK2, rolling)
            draw_num(pg, sl["zx"], sl["y"] - 1, self.views_zh.replace("{n}", vt), sl["vz"], ZH_L, rolling)
        p = (t - 0.0) / 0.25
        e = ease_out_cubic(p)
        off = int(round((1 - e) * 0.34 * W))
        if off <= 0:
            return pg.convert("RGB")
        img = self.back.copy()
        # slight parallax dim
        img = dim(img, 0.0)
        # edge shadow
        sh = Image.new("RGBA", (60, H), (0, 0, 0, 0))
        grad = np.tile(np.linspace(0, 70 * (1 - e) + 20, 60, dtype=np.float32)[None, :], (H, 1))
        sh.putalpha(Image.fromarray(grad.astype(np.uint8), "L"))
        img.paste(sh, (off - 60, 0), sh)
        img.paste(pg.convert("RGB"), (off, 0))
        return img


# ----------------------------------------------------------- notifications
class NotifScreen(Base):
    CARD_W = W - 72
    CARD_H = 178
    GAP = 12
    LIST_Y = 492
    SLIDE = 0.32
    MAXN = 5

    def __init__(self, spec):
        super().__init__(spec)
        self.bg = gradient_bg(
            (12, 18, 40), (10, 12, 26),
            blobs=[(0.15, 0.78, 0.55, (24, 96, 118), 0.55), (0.92, 0.12, 0.6, (84, 52, 120), 0.5),
                   (0.6, 0.45, 0.5, (30, 52, 104), 0.35)],
        )
        self._chrome()
        self.items = []
        for it in spec.get("items", []):
            self.items.append(dict(it, sprite=self._card(it)))
        self.items.sort(key=lambda d: d.get("at", 0.0))

    def _chrome(self):
        bg = self.bg.convert("RGBA")
        bg.alpha_composite(status_bar(self.clock, (255, 255, 255)))
        bg.alpha_composite(icon("lock", 44, (255, 255, 255, 235), 8), (W // 2 - 22, 142))
        date = self.spec.get("date", "7월 18일")
        df = ko(40, "Medium")
        text(bg, (W - tlen(df, date)) / 2, 252, date, df, (255, 255, 255, 235))
        cf = ko(212, "Light")
        cs = self.clock
        text(bg, (W - tlen(cf, cs)) / 2, 456 - 20, cs, cf, (255, 255, 255))
        # bottom shortcuts
        for cx, nm in ((120, "flash"), (W - 120, "camera")):
            bg.alpha_composite(circle_layer(112, (255, 255, 255, 40)), (cx - 56, 1730))
            bg.alpha_composite(icon(nm, 46, (255, 255, 255, 240), 7), (cx - 23, 1730 + 33))
        bg.alpha_composite(home_bar((255, 255, 255), 200), (0, H - 44))
        self.base = bg.convert("RGB")

    def _card(self, it):
        w, h = self.CARD_W, self.CARD_H
        SP = 26
        im = Image.new("RGBA", (w + 2 * SP, h + 2 * SP), (0, 0, 0, 0))
        ss = 3
        size = (w * ss, h * ss)
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), radius=40 * ss, fill=255)
        mask = mask.resize((w, h), RS.LANCZOS)
        shm = Image.new("L", im.size, 0)
        shm.paste(mask, (SP, SP))
        shm = shm.filter(ImageFilter.GaussianBlur(12))
        shm = ImageChops.offset(shm, 0, 6).point([int(v * 0.30) for v in range(256)])
        im.alpha_composite(solid_mask_layer(im.size, (0, 0, 10), shm))
        card = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        card.alpha_composite(solid_mask_layer((w, h), (246, 248, 252), mask.point([int(v * 0.88) for v in range(256)])))
        # app icon
        ic = Image.new("RGBA", (46, 46), (0, 0, 0, 0))
        ic.alpha_composite(rr_layer(46, 46, 12, (236, 108, 76, 255)))
        ic.alpha_composite(icon("house", 30, (255, 255, 255), 6), (8, 8))
        card.alpha_composite(ic, (26, 16))
        af, azf, tf = ko(26, "SemiBold"), zh(24), ko(25, "Regular")
        app = self.spec.get("app", "")
        text(card, 86, 16 + 33, app, af, (74, 82, 96))
        text(card, 86 + tlen(af, app) + 14, 16 + 32, self.spec.get("app_zh", ""), azf, ZH_L)
        text(card, w - 28 - tlen(tf, "지금"), 16 + 33, "지금", tf, (124, 132, 146))
        text(card, 28, 108, it.get("ko", ""), ko(40, "SemiBold"), (20, 22, 28))
        text(card, 28, 154, it.get("zh", ""), zh(34), ZH_L)
        im.alpha_composite(card, (SP, SP))
        return Sprite(im, -SP, -SP, w / 2 - SP, h / 2 - SP)

    def events(self):
        return [it.get("at", 0.0) for it in self.items]

    def frame(self, t):
        img = self.base.copy()
        step = self.CARD_H + self.GAP
        shown = [it for it in self.items if t >= it.get("at", 0.0)]
        n = len(shown)
        # newest first
        for rank, it in enumerate(reversed(shown)):
            at = it.get("at", 0.0)
            e = ease_out_quint((t - at) / self.SLIDE)
            if rank >= self.MAXN:
                continue
            # position: slot index counting newer cards, each animated by its own progress
            pos = 0.0
            for j in range(rank):
                newer = shown[n - 1 - j]
                pos += ease_out_quint((t - newer.get("at", 0.0)) / self.SLIDE)
            y_slot = self.LIST_Y + pos * step
            if rank == 0 and e < 1:
                y = -self.CARD_H + (y_slot + self.CARD_H) * e
                a = min(1.0, e * 2.2)
            else:
                y = y_slot
                a = 1.0
            if rank == self.MAXN - 1:
                # card about to be pushed out fades quickly
                a *= 1.0
            blit(img, it["sprite"], 36, y, 1.0, a)
        return img


# ------------------------------------------------------------------- call
class CallScreen(Base):
    def __init__(self, spec):
        super().__init__(spec)
        self.contact = spec.get("contact", {})
        self.timer_from = int(spec.get("timer_from", 0))
        self.bg = gradient_bg(
            (24, 32, 48), (8, 11, 18),
            blobs=[(0.5, 0.18, 0.7, (46, 74, 112), 0.45), (0.1, 0.8, 0.5, (34, 54, 86), 0.3)],
            noise=0.8,
        )
        self._chrome()

    def _chrome(self):
        c = self.spec
        bg = self.bg.convert("RGBA")
        bg.alpha_composite(status_bar(self.clock, (255, 255, 255)))
        self.av_c = (W // 2, 352)
        bg.alpha_composite(avatar_img(260, (120, 132, 150), (222, 228, 238)), (W // 2 - 130, 352 - 130))
        nf = ko(66, "Bold")
        name = self.contact.get("name", "")
        text(bg, (W - tlen(nf, name)) / 2, 590, name, nf, (255, 255, 255))
        self.timer_y = 672
        zl = (self.contact.get("name_zh", "") + " · " + c.get("label_zh", "通话中")).strip(" ·")
        zf = zh(40)
        text(bg, (W - tlen(zf, zl)) / 2, 742, zl, zf, ZH_D)
        # buttons
        labels = [("mic_off", "음소거", "静音", False), ("keypad", "키패드", "拨号盘", False),
                  ("speaker", "스피커", "免提", bool(c.get("speaker_on", False)))]
        cy = 972
        xs = [W // 2 - 270, W // 2, W // 2 + 270]
        lf, lz = ko(30, "Medium"), zh(28)
        for (nm, k, z, on), x in zip(labels, xs):
            r = 88
            if on:
                bg.alpha_composite(circle_layer(2 * r, (255, 255, 255, 255)), (x - r, cy - r))
                bg.alpha_composite(icon(nm, 84, (30, 34, 44), 7), (x - 42, cy - 42))
            else:
                bg.alpha_composite(circle_layer(2 * r, (255, 255, 255, 40)), (x - r, cy - r))
                bg.alpha_composite(icon(nm, 84, (255, 255, 255, 245), 7), (x - 42, cy - 42))
            text(bg, x - tlen(lf, k) / 2, cy + r + 46, k, lf, (255, 255, 255, 215))
            text(bg, x - tlen(lz, z) / 2, cy + r + 84, z, lz, ZH_D)
        # hang up
        hy = 1288
        bg.alpha_composite(circle_layer(184, (234, 62, 72, 255)), (W // 2 - 92, hy - 92))
        bg.alpha_composite(icon("hangup", 100, (255, 255, 255), 7), (W // 2 - 50, hy - 50))
        bg.alpha_composite(home_bar((255, 255, 255), 200), (0, H - 44))
        self.base = bg.convert("RGB")
        self.timer_font = ko(46, "Regular")

    def frame(self, t):
        img = self.base.copy().convert("RGBA")
        # pulsing rings around avatar
        cx, cy = self.av_c
        ring = Image.new("RGBA", (560, 560), (0, 0, 0, 0))
        ss = 2
        big = Image.new("RGBA", (560 * ss, 560 * ss), (0, 0, 0, 0))
        d = ImageDraw.Draw(big)
        for k in range(2):
            ph = ((t / 2.4) + k * 0.5) % 1.0
            r = (136 + 62 * ease_out_quad(ph)) * ss
            a = int(80 * (1 - ph) ** 1.3)
            d.ellipse((280 * ss - r, 280 * ss - r, 280 * ss + r, 280 * ss + r),
                      outline=(180, 205, 240, a), width=int(3 * ss))
        ring = to_img(big, 560, 560)
        img.alpha_composite(ring, (cx - 280, cy - 280))
        secs = self.timer_from + int(math.floor(t + 1e-6))
        mm, ss_ = divmod(secs, 60)
        s = f"통화 중 {mm:02d}:{ss_:02d}"
        f = self.timer_font
        wtot = tab_len(f, s)
        text_tab(img, (W - wtot) / 2, self.timer_y, s, f, (255, 255, 255, 225))
        return img.convert("RGB")


# ------------------------------------------------------------------ views
def inpaint_white_triangle(crop):
    """The source thumbnails have a baked-in white play triangle in the middle.
    Locate it (pure white blob near the centre), remove it and return
    (image, (cx, cy) centroid fraction)."""
    a = np.asarray(crop).astype(np.int16)
    h, w, _ = a.shape
    x0, x1 = int(w * 0.35), int(w * 0.65)
    y0, y1 = int(h * 0.30), int(h * 0.62)
    sub = a[y0:y1, x0:x1]
    m = sub.min(axis=2) >= 250
    area = int(m.sum())
    if not (1200 <= area <= 12000):
        return crop, None
    ys, xs = np.nonzero(m)
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    if not (30 <= bw <= 220 and 30 <= bh <= 220) or area / float(bw * bh) > 0.75 or area / float(bw * bh) < 0.25:
        return crop, None
    cx, cy = xs.mean() + x0, ys.mean() + y0
    # dilate mask a little, then diffuse the surrounding colours into it
    pad = 14
    bx0, by0 = max(0, int(xs.min()) + x0 - pad), max(0, int(ys.min()) + y0 - pad)
    bx1, by1 = min(w, int(xs.max()) + x0 + pad + 1), min(h, int(ys.max()) + y0 + pad + 1)
    reg = a[by0:by1, bx0:bx1].astype(np.float32)
    mk = reg.min(axis=2) >= 244
    for _ in range(4):
        mk = mk | np.roll(mk, 1, 0) | np.roll(mk, -1, 0) | np.roll(mk, 1, 1) | np.roll(mk, -1, 1)
    mk[0, :] = mk[-1, :] = False
    mk[:, 0] = mk[:, -1] = False
    known = ~mk
    cur = reg.copy()
    mean = reg[known].mean(axis=0)
    cur[mk] = mean
    for _ in range(600):
        nb = (np.roll(cur, 1, 0) + np.roll(cur, -1, 0) + np.roll(cur, 1, 1) + np.roll(cur, -1, 1)) / 4.0
        cur[mk] = nb[mk]
    out = a.copy()
    out[by0:by1, bx0:bx1] = np.clip(cur + 0.5, 0, 255).astype(np.int16)
    return Image.fromarray(out.astype(np.uint8), "RGB"), (cx / w, cy / h)


class ViewsScreen(Base):
    PH = 1010

    def __init__(self, spec):
        super().__init__(spec)
        self.vf = spec.get("views_from", spec.get("views"))
        self.vt = spec.get("views")
        self.t0, self.t1 = spec.get("count_at", [0.0, 1.0])
        self.views_zh = spec.get("views_zh", "{n} 万次播放")
        self._build()

    def _build(self):
        s = self.spec
        src = Image.open(s["thumb"]).convert("RGB")
        c = s.get("thumb_crop") or [0, 0, 1, 1]
        sw, sh = src.size
        crop = src.crop((int(c[0] * sw), int(c[1] * sh), int(c[2] * sw), int(c[3] * sh)))
        crop, focus = inpaint_white_triangle(crop)
        fy = focus[1] if focus else 0.5
        scale = W / crop.width
        full = crop.resize((W, int(round(crop.height * scale))), RS.LANCZOS)
        ph = min(self.PH, full.height)
        cy = fy * full.height
        top = int(round(clamp(cy - ph / 2, 0, full.height - ph)))
        player = full.crop((0, top, W, top + ph)).convert("RGBA")
        pcy = cy - top if focus else ph / 2
        # play button
        pb = Image.new("RGBA", (260, 260), (0, 0, 0, 0))
        pb.alpha_composite(circle_layer(200, (0, 0, 0, 105)), (30, 30))
        pb.alpha_composite(icon("play", 100, (255, 255, 255, 238), 7), (30 + 100 - 50 + 6, 30 + 100 - 50))
        player.alpha_composite(pb, (int(W / 2 - 130), int(pcy - 130)))
        self.player_h = ph
        self.player = player
        pg = Image.new("RGBA", (W, H), (255, 255, 255, 255))
        pg.alpha_composite(status_bar(self.clock, INK))
        pg.alpha_composite(player, (0, SB_H))
        self.info_y = SB_H + ph
        off = 0
        if s.get("title"):  # 视频标题：韩文一行、中文一行，播放数跟着往下挪
            text(pg, 48, self.info_y + 82, s["title"], ko(52, "Bold"), INK)
            if s.get("title_zh"):
                text(pg, 48, self.info_y + 138, s["title_zh"], zh(40), ZH_L)
            off = 130
        self.vslot = dict(x=48, y=self.info_y + 98 + off)
        self.vfont = ko(62, "SemiBold")
        self.zfont = zh(42)
        y = self.info_y + 98 + 70 + off
        self.zslot = (48, y)
        yy = y + 36
        pg.paste((228, 232, 238), (48, yy, W - 48, yy + 2))
        xs = [int(W * k / 8) for k in (1, 3, 5, 7)]
        for x, nm in zip(xs, ("heart", "comment", "share", "kebab")):
            pg.alpha_composite(icon(nm, 66, (70, 78, 92), 6), (x - 33, yy + 40))
        self.icon_xs, self.cnt_y, self.cfont = xs, yy + 150, ko(34, "Medium")  # 点赞、评论数写在图标下面（可选 likes / comments）
        self.bar_y = SB_H + ph - 10
        pg.alpha_composite(home_bar(INK), (0, H - 44))
        self.page = pg

    def _roll(self, t, a, b):
        p = clamp((t - self.t0) / max(1e-6, self.t1 - self.t0))
        return a + (b - a) * ease_out_quad(p)

    def _count(self, t):
        if t <= self.t0:
            return fmt_wan(self.vf)
        if t >= self.t1:
            return fmt_wan(self.vt)
        return fmt_wan(self._roll(t, self.vf, self.vt), rolling=True)

    def events(self):
        return [self.t0, self.t1]

    def key_times(self):
        end = (self.nframes - 1) / FPS
        return [0.04, self.t0 + (self.t1 - self.t0) * 0.45, self.t1 + 0.25, end]

    def frame(self, t):
        pg = self.page.copy()
        # video progress bar
        prog = 0.34 + 0.012 * t
        pg.alpha_composite(Image.new("RGBA", (W, 8), (255, 255, 255, 90)), (0, self.bar_y + 2))
        pg.alpha_composite(Image.new("RGBA", (int(W * prog), 8), (255, 62, 62, 255)), (0, self.bar_y + 2))
        n = self._count(t)
        vx, vy = self.vslot["x"], self.vslot["y"]
        rolling = self.t0 < t < self.t1
        draw_num(pg, vx, vy, "조회수 " + n + "만", self.vfont, INK, rolling)
        zx, zy = self.zslot
        draw_num(pg, zx, zy, self.views_zh.replace("{n}", n), self.zfont, ZH_L, rolling)
        for i, key in ((0, "likes"), (1, "comments")):
            end = self.spec.get(key)
            if end is None:
                continue
            v = self._roll(t, self.spec.get(key + "_from", end), end)
            lab = (fmt_wan(v, rolling) + "만") if end >= 10000 else f"{int(v):,}"
            text(pg, self.icon_xs[i] - tlen(self.cfont, lab) / 2, self.cnt_y, lab, self.cfont, (70, 78, 92))
        return pg.convert("RGB")


# ------------------------------------------------------------ driver
SCREEN_TYPES = {
    "chat": ChatScreen,
    "mail": MailScreen,
    "post": PostScreen,
    "notifications": NotifScreen,
    "call": CallScreen,
    "views": ViewsScreen,
}


def build_screen(spec):
    cls = SCREEN_TYPES.get(spec["type"])
    if cls is None:
        raise ValueError(f"unknown screen type: {spec['type']}")
    return cls(spec)


def encode(screen, path):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-an",
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "16",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
        "-r", str(FPS), "-movflags", "+faststart", path,
    ]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    last = None
    try:
        for i in range(screen.nframes):
            fr = screen.frame(i / FPS)
            if fr.size != (W, H) or fr.mode != "RGB":
                fr = fr.convert("RGB").resize((W, H))
            pr.stdin.write(fr.tobytes())
            last = fr
    finally:
        pr.stdin.close()
        rc = pr.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")
    return last


def contact_sheet(screens, path, scale=0.30):
    cw, chh = int(W * scale), int(H * scale)
    gap, lab_h = 14, 44
    cols = 4
    sw = cols * cw + (cols + 1) * gap
    sh = len(screens) * (chh + lab_h + gap) + gap
    sheet = Image.new("RGB", (sw, sh), (34, 36, 42))
    d = ImageDraw.Draw(sheet)
    lf = ko(24, "Medium")
    y = gap
    for sc in screens:
        d.text((gap, y + 6), f"{sc.id}   {sc.spec['type']}   {sc.seconds}s", font=lf, fill=(235, 238, 244))
        times = sc.key_times()
        for i, tt in enumerate(times[:cols]):
            fr = sc.frame(tt).resize((cw, chh), RS.LANCZOS)
            x = gap + i * (cw + gap)
            sheet.paste(fr, (x, y + lab_h))
            d.text((x + 6, y + lab_h + chh - 28), f"t={tt:.2f}s", font=ko(20, "Medium"), fill=(255, 255, 255),
                   stroke_width=2, stroke_fill=(0, 0, 0))
        y += chh + lab_h + gap
    sheet.save(path, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("outdir")
    ap.add_argument("--only", default="")
    ap.add_argument("--stills", default="")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    with open(a.json, encoding="utf-8") as f:
        data = json.load(f)
    os.makedirs(a.outdir, exist_ok=True)
    only = set(x for x in a.only.split(",") if x)
    screens = []
    for spec in data["screens"]:
        if only and spec["id"] not in only:
            continue
        sc = build_screen(spec)
        screens.append(sc)
        if a.stills:
            for tt in [float(x) for x in a.stills.split(",")]:
                sc.frame(tt).save(os.path.join(a.outdir, f"{sc.id}_t{tt:.2f}.png"))
            continue
        if a.no_video:
            sc.frame((sc.nframes - 1) / FPS).save(os.path.join(a.outdir, sc.id + ".png"))
            continue
        print(f"[{sc.id}] {spec['type']} {sc.seconds}s -> {sc.nframes} frames", flush=True)
        last = encode(sc, os.path.join(a.outdir, sc.id + ".mp4"))
        last.save(os.path.join(a.outdir, sc.id + ".png"))
    if screens and not a.stills:
        ep = None
        for s in (os.path.basename(os.path.normpath(a.outdir)), os.path.splitext(os.path.basename(a.json))[0]):
            m = re.search(r"EP\d+", s, re.I)
            if m:
                ep = m.group(0).upper()
                break
        ep = ep or os.path.basename(os.path.normpath(a.outdir))
        contact_sheet(screens, os.path.join(a.outdir, f"预览-{ep}.jpg"))


if __name__ == "__main__":
    main()
