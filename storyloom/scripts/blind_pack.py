# /// script
# requires-python = ">=3.10"
# ///
"""切出盲测包：把剧本正文按段切成一段一个文件，给盲测观众和推理迷读。

只取每集“## 正文”之后的内容（一句话、本集人物这些会泄底或剧透的部分不要），
段落按“**第 N 段 …**”切开。【新增】【改】这类修改标记会去掉，免得读的人注意到改过哪里。

用法：
  uv run blind_pack.py <剧的文件夹> --from 1 --to 1 --out <目录>      # 只要第 1 集
  uv run blind_pack.py <剧的文件夹> --from 1 --to 3:7 --out <目录>    # 第 1 集到第 3 集第 7 段之前
  uv run blind_pack.py --selftest
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

# 段标题前后可能带【新增】【改】标记
SEG_RE = re.compile(r"^(?:【(?:新增|改|删)】\s*)?\*\*第\s*(\d+)\s*段.*$", re.M)
MARK_RE = re.compile(r"【(?:新增|改|删)】")


def episode_file(show: Path, n: int) -> Path:
    for name in (f"EP{n:02d}", f"EP{n}"):
        p = show / name / "剧本.md"
        if p.exists():
            return p
    raise SystemExit(f"✗ 找不到第 {n} 集的剧本：{show}/EP{n:02d}/剧本.md")


def body_of(text: str) -> str:
    m = re.search(r"^##\s*正文\s*$", text, re.M)
    if not m:
        raise ValueError("剧本里没有“## 正文”这一节")
    rest = text[m.end():]
    nxt = re.search(r"^##\s+\S", rest, re.M)  # 正文之后如果还有别的二级标题，不要
    return rest[: nxt.start()] if nxt else rest


def segments(text: str) -> list[tuple[int, str]]:
    body = body_of(text)
    heads = list(SEG_RE.finditer(body))
    if not heads:
        raise ValueError("正文里没有“**第 N 段 …**”这样的段标题")
    out = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        chunk = MARK_RE.sub("", body[h.start():end]).strip()
        chunk = re.sub(r"\n{3,}", "\n\n", chunk)
        out.append((int(h.group(1)), chunk + "\n"))
    return out


def parse_to(s: str) -> tuple[int, int | None]:
    if ":" in s:
        ep, seg = s.split(":", 1)
        return int(ep), int(seg)
    return int(s), None


def build(show: Path, ep_from: int, to: str, out: Path) -> list[Path]:
    ep_to, seg_stop = parse_to(to)
    if ep_to < ep_from:
        raise SystemExit("✗ --to 不能在 --from 前面")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    multi = ep_to > ep_from
    files = []
    for ep in range(ep_from, ep_to + 1):
        segs = segments(episode_file(show, ep).read_text(encoding="utf-8"))
        for n, chunk in segs:
            if ep == ep_to and seg_stop is not None and n >= seg_stop:
                break
            name = f"ep{ep:02d}_seg{n:02d}.md" if multi else f"seg{n:02d}.md"
            head = f"（第 {ep} 集）\n\n" if multi else ""
            p = out / name
            p.write_text(head + chunk, encoding="utf-8")
            files.append(p)
    if not files:
        raise SystemExit("✗ 没切出任何段，检查 --from / --to")
    return files


def selftest() -> None:
    sample = """# 测试剧 · 第 1 集 · 测试（约 30 秒，3 段）

## 一句话
凶手是房东。（这句不能进盲测包）

## 本集人物
- 甲：主角

## 正文

**第 1 段 · 前台 · 傍晚（约 10 秒）**
- 镜头 1（全景）：甲推门进来。
  甲（慌）：안녕하세요 / 你好

**第 2 段 · 走廊 · 夜（约 10 秒）**【新增】
- 镜头 1（中景）：【改】走廊尽头有一件白雨衣。

【改】**第 3 段 · 天台 · 凌晨（约 10 秒）**
- 镜头 1（全景）：房东摘下兜帽。

## 附：给 manju 的备注
不该进盲测包。
"""
    with tempfile.TemporaryDirectory() as d:
        show = Path(d) / "剧"
        (show / "EP01").mkdir(parents=True)
        (show / "EP02").mkdir(parents=True)
        (show / "EP01" / "剧本.md").write_text(sample, encoding="utf-8")
        (show / "EP02" / "剧本.md").write_text(sample.replace("第 1 集", "第 2 集"), encoding="utf-8")

        files = build(show, 1, "1", Path(d) / "out1")
        assert [f.name for f in files] == ["seg01.md", "seg02.md", "seg03.md"], files
        all_text = "".join(f.read_text(encoding="utf-8") for f in files)
        assert "凶手是房东" not in all_text, "一句话泄进了盲测包"
        assert "给 manju 的备注" not in all_text, "正文后面的附录泄进了盲测包"
        assert "【" not in all_text, "修改标记没去掉"
        assert "白雨衣" in all_text

        files = build(show, 1, "2:3", Path(d) / "out2")
        assert [f.name for f in files] == [
            "ep01_seg01.md", "ep01_seg02.md", "ep01_seg03.md",
            "ep02_seg01.md", "ep02_seg02.md",
        ], files
        assert "摘下兜帽" not in files[-1].read_text(encoding="utf-8")
        assert files[0].read_text(encoding="utf-8").startswith("（第 1 集）")

        try:
            segments("# 没有正文\n")
            raise AssertionError("没有正文时应该报错")
        except ValueError:
            pass
    print("✓ 自测通过")


def main() -> None:
    ap = argparse.ArgumentParser(description="切出盲测包（一段一个文件）")
    ap.add_argument("show", nargs="?", type=Path, help="剧的文件夹，里面有 EP01/剧本.md 这样的文件")
    ap.add_argument("--from", dest="ep_from", type=int, default=1, help="从第几集开始（默认 1）")
    ap.add_argument("--to", default=None, help="到第几集为止；写 3:7 表示到第 3 集第 7 段之前")
    ap.add_argument("--out", type=Path, help="输出目录（会先清空）")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.show or not a.out:
        ap.error("要给剧的文件夹和 --out")
    to = a.to or str(a.ep_from)
    try:
        files = build(a.show.expanduser(), a.ep_from, to, a.out.expanduser())
    except ValueError as e:
        sys.exit(f"✗ {e}")
    print(f"✓ 切出 {len(files)} 段 → {a.out}")
    for f in files:
        print("  ", f.name)


if __name__ == "__main__":
    main()
