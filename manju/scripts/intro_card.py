# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=11"]
# ///
"""
人物出场介绍卡：竖屏古风短剧里人物第一次出场时的字幕卡。

版式（克制，不花哨）：
  大号宋体名字竖排（米白 + 柔和深色阴影） ｜ 一条细线 ｜ 小字身份说明竖排（淡一点）
  名字下方一枚小朱红印章，里面是名字第一个字（白字）。
  side="left" 靠左、side="right" 靠右，离边约 90px；名字总是在靠边的一侧，
  身份说明在靠画面中心的一侧，左右镜像。放在画面中上部，下方 1/4 留给对白字幕。

用法：
  uv run intro_card.py 沈夜 "死了十三年的人" out.png [--side left|right]
      [--width 1080 --height 1920]

出来的是一张和画面同尺寸的透明 PNG；叠到视频上用 ffmpeg overlay，
淡入淡出用 fade=t=in:alpha=1 / fade=t=out:alpha=1（见 mj.py 或下面的建议值）。
建议：出现 2.2 秒，淡入 0.3 秒，淡出 0.4 秒。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_CANDIDATES = [
    # (路径, {角色: ttc 里的字重序号})  Songti.ttc: 0 Black, 1 Bold, 3 Light, 6 Regular
    ("/System/Library/Fonts/Supplemental/Songti.ttc", {"name": 1, "tag": 6, "seal": 0}),
    ("/System/Library/Fonts/STHeiti Medium.ttc", {"name": 0, "tag": 0, "seal": 0}),
    ("/System/Library/Fonts/PingFang.ttc", {"name": 0, "tag": 0, "seal": 0}),
]

CREAM = (240, 236, 228)
TAG_COLOR = (214, 208, 196)
SEAL_RED = (176, 36, 32)
SHADOW = (8, 6, 6)

# 竖排时标点没法直接叠，画成半格空白
_PUNCT = set("，。、；：！,.;:!")


def _font(role: str, size: int) -> ImageFont.FreeTypeFont:
    for path, idx in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size, index=idx[role])
            except OSError:
                continue
    raise RuntimeError("找不到可用的中文字体（Songti / STHeiti / PingFang）")


def _glyph_box(font: ImageFont.FreeTypeFont, ch: str) -> tuple[int, int, int, int]:
    return font.getbbox(ch, anchor="ls")  # 相对基线原点


def _vertical_run(draw: ImageDraw.ImageDraw, font, text: str, cx: float, top: float,
                  step: int, fill, stroke: int = 0, stroke_fill=None) -> float:
    """竖排画一列，字符水平居中在 cx，从 top 往下，返回列底部 y。"""
    y = top
    for ch in text:
        if ch in _PUNCT:
            y += step * 0.5
            continue
        if ch == " ":
            y += step * 0.5
            continue
        l, t, r, b = font.getbbox(ch, anchor="ls")
        w = font.getlength(ch)
        # 用 em 方格居中（而不是墨迹居中），竖排才整齐；窄字符（？、一）按墨迹居中
        x = cx - w / 2 if w >= font.size * 0.8 else cx - (l + r) / 2
        # 基线：em 方格内大致让字心落在格子中央
        base = y + step * 0.5 + font.size * 0.36
        draw.text((x, base), ch, font=font, fill=fill, anchor="ls",
                  stroke_width=stroke, stroke_fill=stroke_fill)
        y += step
    return y


def _column_height(text: str, step: int) -> float:
    h = 0.0
    for ch in text:
        h += step * 0.5 if (ch in _PUNCT or ch == " ") else step
    return h


def render_intro(name: str, tagline: str, out: Path, width: int = 1080, height: int = 1920,
                 side: str = "left") -> tuple[int, int, int, int]:
    """渲染出场介绍卡，输出和画面同尺寸的透明 PNG，返回卡片矩形 (x, y, w, h)。"""
    assert side in ("left", "right"), "side 只能是 left 或 right"
    name, tagline = name.strip(), tagline.strip()
    if not name:
        raise ValueError("名字不能为空")
    s = width / 1080  # 以 1080 宽为基准按比例缩放

    name_size = round((124 if len(name) <= 2 else 114 if len(name) == 3 else 106) * s)
    name_step = round(name_size * 1.1)
    # 身份说明只是问号（神秘人物）：画成一个大一点的问号，否则全角问号在宋体里太小
    mystery = bool(tagline) and all(c in "？?" for c in tagline)
    if mystery:
        tagline = "?" * len(tagline)
    tag_size = round((84 if mystery else 36) * s)
    tag_step = round(tag_size * (1.1 if mystery else 1.32))
    margin = round(90 * s)
    gap = round(30 * s)       # 名字与细线、细线与说明之间
    line_w = max(2, round(2 * s))
    seal = round(66 * s)
    seal_gap = round(34 * s)
    top = round(height * 0.10)

    f_name = _font("name", name_size)
    f_tag = _font("name" if mystery else "tag", tag_size)

    name_h = _column_height(name, name_step)
    tag_h = _column_height(tagline, tag_step) if tagline else 0
    name_w = name_size
    tag_w = tag_size if tagline else 0
    body_h = max(name_h, tag_h + round(26 * s))
    card_w = name_w + (gap * 2 + line_w + tag_w if tagline else 0)
    card_h = body_h + seal_gap + seal

    x0 = margin if side == "left" else width - margin - card_w
    y0 = top

    # 列的位置：名字靠边，说明靠中心
    if side == "left":
        name_cx = x0 + name_w / 2
        line_x = x0 + name_w + gap
        tag_cx = line_x + line_w + gap + tag_w / 2
    else:
        name_cx = x0 + card_w - name_w / 2
        line_x = x0 + card_w - name_w - gap - line_w
        tag_cx = line_x - gap - tag_w / 2

    # ---------- 文字层（实心，用来生成阴影，也用来最终合成）----------
    text_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    td = ImageDraw.Draw(text_layer)
    _vertical_run(td, f_name, name, name_cx, y0, name_step, CREAM + (255,))
    if tagline:
        _vertical_run(td, f_tag, tagline, tag_cx, y0 + round(26 * s), tag_step, TAG_COLOR + (255,))
        # 细线：上下渐隐，不拖泥带水
        line = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        ld = ImageDraw.Draw(line)
        steps = 40
        for i in range(steps):
            a = int(210 * min(1.0, i / 8, (steps - 1 - i) / 8))
            ya = y0 + body_h * i / steps
            yb = y0 + body_h * (i + 1) / steps + 1
            ld.rectangle([line_x, ya, line_x + line_w - 1, yb], fill=CREAM + (a,))
        text_layer = Image.alpha_composite(text_layer, line)

    # ---------- 印章 ----------
    seal_x = (name_cx - seal / 2)
    seal_y = y0 + body_h + seal_gap
    seal_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    sd = ImageDraw.Draw(seal_layer)
    r = max(3, round(5 * s))
    sd.rounded_rectangle([seal_x, seal_y, seal_x + seal, seal_y + seal], radius=r, fill=SEAL_RED + (240,))
    inset = max(4, round(5 * s))
    sd.rounded_rectangle([seal_x + inset, seal_y + inset, seal_x + seal - inset, seal_y + seal - inset],
                         radius=max(1, r - 2), outline=(255, 255, 255, 70), width=max(1, round(1.5 * s)))
    f_seal = _font("seal", round(seal * 0.60))
    sd.text((seal_x + seal / 2, seal_y + seal / 2), name[0], font=f_seal, fill=(255, 255, 255, 255), anchor="mm")

    # ---------- 阴影：整体一层大范围柔光暗晕 + 一层贴字的小投影 ----------
    combined = Image.alpha_composite(text_layer, seal_layer)
    alpha = combined.getchannel("A")
    halo = alpha.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(26 * s))
    halo = halo.point(lambda v: int(v * 0.55))
    drop = alpha.filter(ImageFilter.GaussianBlur(5 * s)).point(lambda v: min(255, int(v * 1.0)))
    drop_off = Image.new("L", (width, height), 0)
    drop_off.paste(drop, (round(2 * s), round(4 * s)))

    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    canvas = Image.alpha_composite(canvas, Image.merge("RGBA", (
        *[Image.new("L", (width, height), c) for c in SHADOW], halo)))
    canvas = Image.alpha_composite(canvas, Image.merge("RGBA", (
        *[Image.new("L", (width, height), c) for c in SHADOW], drop_off.point(lambda v: int(v * 0.85)))))
    canvas = Image.alpha_composite(canvas, combined)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return (round(x0), round(y0), round(card_w), round(card_h))


def main() -> None:
    ap = argparse.ArgumentParser(description="渲染人物出场介绍卡（透明 PNG）")
    ap.add_argument("name", help="人物名，竖排")
    ap.add_argument("tagline", help="身份说明，一行小字")
    ap.add_argument("out", type=Path, help="输出 PNG 路径")
    ap.add_argument("--side", choices=["left", "right"], default="left")
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1920)
    a = ap.parse_args()
    rect = render_intro(a.name, a.tagline, a.out, a.width, a.height, a.side)
    print(f"{a.out}  rect(x,y,w,h)={rect}")


if __name__ == "__main__":
    main()
