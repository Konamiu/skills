# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=11", "edge-tts>=7", "requests>=2.31", "numpy>=1.26"]
# ///
"""
AI 漫剧工作流的执行器。一集一个文件夹，分镜.json 是唯一依据，每一步都可以单独重跑。

  uv run mj.py init      <集目录> --title 剧名 [--ratio 9:16]
  uv run mj.py check     <集目录>                 检查分镜.json（台词放不放得下、角色有没有设定）
  uv run mj.py sheets    <集目录> [--only lin,shop] 用 Codex 出定妆照 / 场景图（免费）
  uv run mj.py frames    <集目录> [--shots 1,3-5] [--force]  用 Codex 出每镜首帧（免费）
  uv run mj.py handoff   <集目录>                 生成「图转视频清单.md」，拿去 AI Studio / Flow 出视频
  uv run mj.py tts       <集目录> [--force]        配音（edge-tts，按角色固定声线）
  —— 真人模式（mode: live，分镜按“场”写在 segments 里）——
  uv run mj.py voices    <集目录> [--force]        给每个角色做一段参考声音（之后每场都用它，声线不变）
  uv run mj.py vo        <集目录> [--segments 2,3] [--force]  单独生成旁白 / 内心独白（Qwen3-TTS）
  uv run mj.py live      <集目录> [--segments 1,3] [--force]  在远程 5090（ComfyUI + MiniMax-H3）一场一场拍：画面、对白、口型一次出
  uv run mj.py music     <集目录> [--seed 2]       在 5090 上生成原创配乐（MiniMax Music 3），长度对齐成片
  uv run mj.py ambience  <集目录> rain|wind|city|night|room|sea  本机合成一条环境音，铺满全片
  uv run mj.py assemble  <集目录>                 剪辑合成：视频/首帧 + 配音 + 字幕 + 配乐 + 调色 → 成片
  uv run mj.py qc        <集目录>                 自检：时长、分辨率、响度、黑场、台词识别、缩略图
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures as cf
import difflib
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")


def _config() -> dict:
    """本机设置 ~/.config/manju/config.json：ComfyUI 地址、AutoDL 实例、各种模型放在哪。不进仓库。"""
    try:
        return json.loads((Path.home() / ".config" / "manju" / "config.json").read_text("utf-8"))
    except (OSError, ValueError):
        return {}


CFG = _config()
FPS = 24
CANVAS = {"9:16": (1080, 1920), "16:9": (1920, 1080), "1:1": (1080, 1080), "3:4": (1080, 1440)}
SHAPE = {"9:16": "9:16 竖图", "16:9": "16:9 横图", "1:1": "1:1 方图", "3:4": "3:4 竖图"}

# 情绪 → edge-tts 语速、音调（同一个人换情绪，声线不变）
EMOTION = {
    "平静": ("+0%", "+0Hz"), "温柔": ("-8%", "-2Hz"), "开心": ("+8%", "+6Hz"), "激动": ("+18%", "+8Hz"),
    "紧张": ("+22%", "+10Hz"), "愤怒": ("+12%", "+4Hz"), "冷漠": ("-5%", "-6Hz"), "悲伤": ("-15%", "-6Hz"),
    "低语": ("-12%", "-4Hz"), "命令": ("-25%", "-5Hz"), "嘲讽": ("+5%", "+3Hz"),
}
DEFAULT_VOICES = {"narrator": "zh-CN-YunjianNeural"}
CHARS_PER_SEC = 4.0  # 中文台词大约每秒 4 个字
KO_SYL_PER_SEC = 5.5  # 韩语台词每秒音节数（量过第一版实拍：中位 5.8，四分位 4.9–6.3）


# ---------------------------------------------------------------- 基础

def die(msg: str):
    print(f"✗ {msg}", file=sys.stderr)
    sys.exit(1)


def run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} 失败：{(p.stderr or p.stdout)[-800:]}")
    return p


def probe(path: Path) -> dict:
    out = run([FFPROBE, "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height", "-of", "json", str(path)]).stdout
    j = json.loads(out)
    v = next((s for s in j.get("streams", []) if s.get("codec_type") == "video"), {})
    return {
        "duration": float(j.get("format", {}).get("duration") or 0),
        "width": v.get("width", 0), "height": v.get("height", 0),
        "audio": any(s.get("codec_type") == "audio" for s in j.get("streams", [])),
    }


def keep_ranges(ep: "Ep", shot: dict, src: Path) -> Path:
    """"keep": [[1.1, 5.6], [7.95, 12.1]] —— 一条视频里只留这几段（剪掉穿帮帧、叠影、拖沓），画面硬切，声音接缝处 30 毫秒淡入淡出。
    某一段写成 [2.85, 5.4, "videos/seg01-种子7.mp4"] 就从另一条（同一场换种子重拍的版本）取，用来拼各版本里最好的镜头。
    剪好的放在 build/，原视频不动；之后所有步骤用剪好的这条。"""
    keep = shot.get("keep")
    if not keep:
        return src
    out = ep.root / "build" / f"{ep.sid(shot)}.keep.mp4"
    files = [src] + [ep.root / k[2] for k in keep if len(k) > 2]
    files = list(dict.fromkeys(files))
    tag = hashlib.sha1(json.dumps([keep] + [[p.name, p.stat().st_mtime] for p in files]).encode()).hexdigest()[:10]
    stamp = out.with_suffix(".txt")
    if out.exists() and stamp.exists() and stamp.read_text() == tag:
        return out
    parts, f = [], []
    for i, k in enumerate(keep):
        a, b = k[0], k[1]
        n = files.index(ep.root / k[2]) if len(k) > 2 else 0
        d = float(b) - float(a)
        f.append(f"[{n}:v]trim=start={a}:end={b},setpts=PTS-STARTPTS,scale={ep.W}:{ep.H}:force_original_aspect_ratio=increase,crop={ep.W}:{ep.H},setsar=1,fps=24[v{i}]")
        f.append(f"[{n}:a]atrim=start={a}:end={b},asetpts=PTS-STARTPTS,aresample=48000,afade=t=in:d=0.03,afade=t=out:st={max(0, d - 0.03):.3f}:d=0.03[a{i}]")
        parts.append(f"[v{i}][a{i}]")
    f.append(f"{''.join(parts)}concat=n={len(keep)}:v=1:a=1[v][a]")
    out.parent.mkdir(exist_ok=True)
    run([FFMPEG, "-y", "-v", "error", *[x for p in files for x in ("-i", str(p))], "-filter_complex", ";".join(f), "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-c:a", "aac", "-b:a", "192k", str(out)])
    stamp.write_text(tag)
    return out


class Ep:
    """一集：目录、分镜.json、各类素材的位置。"""

    def __init__(self, root: str):
        self.root = Path(root).expanduser().resolve()
        self.sb_path = self.root / "分镜.json"
        if not self.sb_path.exists():
            die(f"{self.sb_path} 不存在，先运行 init")
        self.sb = json.loads(self.sb_path.read_text("utf-8"))
        # 一个案子几集共用一套人物、场景、道具和声音：这一集的分镜.json 写 "assets_from": "../EP01/分镜.json"，
        # 只写自己的场；人物、场景、道具合并进来（这一集写了同名的就用这一集的），定妆照、场景图、参考声音都从那一集的 assets 找。
        self.assets_root = self.root
        base = self.sb.get("assets_from")
        if base:
            bp = (self.root / base).resolve()
            if not bp.exists():
                die(f"assets_from 指的 {bp} 不存在")
            b = json.loads(bp.read_text("utf-8"))
            self.assets_root = bp.parent
            for k in ("characters", "scenes", "props"):
                self.sb[k] = {**b.get(k, {}), **self.sb.get(k, {})}
            for k, v in b.items():
                if k not in ("segments", "shots", "episode", "note", "bgm", "ambience", "bgm_cues", "bgm_note", "cover") and k not in self.sb:
                    self.sb[k] = v
        self.ratio = self.sb.get("ratio", "9:16")
        self.W, self.H = CANVAS.get(self.ratio, CANVAS["9:16"])
        for d in ("assets/角色", "assets/场景", "assets/道具", "assets/声音", "frames", "videos", "audio", "build", "out", ".codex"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    @property
    def shots(self) -> list[dict]:
        return self.sb.get("shots", [])

    @property
    def live(self) -> bool:
        return self.sb.get("mode") == "live"

    @property
    def segments(self) -> list[dict]:
        return self.sb.get("segments", [])

    def sid(self, shot: dict) -> str:
        # 真人模式的单位是“场”（segment，里面有多个镜头），文件叫 seg01；动态漫的单位是镜头，叫 shot01
        return f"{'seg' if 'shots' in shot else 'shot'}{int(shot['id']):02d}"

    def voice_char(self, key: str) -> dict:
        """说话用的那份角色设定：换装造型写了 "voice_of" 就取原角色的（voice_design、voice_sample 都在那里）。"""
        chars = self.sb.get("characters", {})
        return chars.get(chars.get(key, {}).get("voice_of", key), {})

    def voice_ref(self, key: str) -> Path:
        # 同一个人换了一套造型（睡衣、戏服）另起一个角色 id，写 "voice_of": "原角色" 共用原来的参考声音
        key = self.sb.get("characters", {}).get(key, {}).get("voice_of", key)
        return self.assets_root / "assets" / "声音" / f"{key}.wav"

    def frame(self, shot: dict) -> Path:
        return self.root / "frames" / f"{self.sid(shot)}.png"

    def video(self, shot: dict) -> Path | None:
        """视频素材：videos/shot03.mp4，或 shot03 开头的任意 mp4/mov（多条时用 shot03.mp4 或名字最短的）。"""
        if shot.get("from"):
            # 这一场不单独拍，从别的场剪几段拼起来（比如冷开场用后面高潮的片段）："from": [[0.5, 3.2, "videos/seg17.mp4"], …]
            srcs = [self.root / k[2] for k in shot["from"]]
            if not all(x.exists() for x in srcs):
                return None
            return keep_ranges(self, {**shot, "keep": shot["from"]}, srcs[0])
        exact = self.root / "videos" / f"{self.sid(shot)}.mp4"
        if exact.exists():
            return keep_ranges(self, shot, exact)
        found = sorted((p for p in (self.root / "videos").glob(f"{self.sid(shot)}*") if p.suffix.lower() in (".mp4", ".mov")), key=lambda p: len(p.name))
        return keep_ranges(self, shot, found[0]) if found else None

    def sheet(self, key: str) -> Path:
        kind = "角色" if key in self.sb.get("characters", {}) else "道具" if key in self.sb.get("props", {}) else "场景"
        return self.assets_root / "assets" / kind / f"{key}.png"

    def out_name(self) -> str:
        return f"{self.sb.get('title', '漫剧')}-EP{int(self.sb.get('episode', 1)):02d}"


def parse_ids(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    ids: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            ids.update(range(int(a), int(b) + 1))
        elif part.strip():
            ids.add(int(part))
    return ids


# ---------------------------------------------------------------- init / check

def cmd_init(a):
    root = Path(a.dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    sb = root / "分镜.json"
    if sb.exists() and not a.force:
        die(f"{sb} 已存在（要覆盖加 --force）")
    tpl = json.loads((Path(__file__).parent.parent / "assets" / "分镜模板.json").read_text("utf-8"))
    tpl["title"] = a.title or tpl["title"]
    tpl["ratio"] = a.ratio
    sb.write_text(json.dumps(tpl, ensure_ascii=False, indent=2), "utf-8")
    for name in ("剧本.md",):
        p = root / name
        if not p.exists():
            p.write_text(f"# {tpl['title']} · 第 1 集\n\n", "utf-8")
    Ep(str(root))
    print(f"✓ 已建好 {root}\n  下一步：写 剧本.md，再把分镜填进 分镜.json")


def line_seconds(text: str, lang: str = "zh") -> float:
    if lang == "ko":  # 韩语按音节算：第一版实拍里 H3 念韩语中位 5.8 音节/秒，留点余量按 5.5；数字读出来更长
        return (len(re.findall(r"[\uac00-\ud7a3]", text)) + 1.5 * len(re.findall(r"\d", text))) / KO_SYL_PER_SEC
    n = len(re.sub(r"[\s，。！？、…“”‘’,.!?\-—]", "", text))
    return n / CHARS_PER_SEC


def cmd_check(a):
    ep = Ep(a.dir)
    chars = ep.sb.get("characters", {})
    scenes = ep.sb.get("scenes", {})
    problems, warns = [], []
    total = 0.0
    seen = set()
    for s in ep.shots:
        sid = s.get("id")
        if sid in seen:
            problems.append(f"第 {sid} 镜：id 重复")
        seen.add(sid)
        for c in s.get("characters", []):
            if c not in chars:
                problems.append(f"第 {sid} 镜：角色 {c} 没在 characters 里设定")
        for k in s.get("props", []):
            if k not in ep.sb.get("props", {}):
                problems.append(f"第 {sid} 镜：道具 {k} 没在 props 里设定")
        if s.get("scene") and s["scene"] not in scenes:
            problems.append(f"第 {sid} 镜：场景 {s['scene']} 没在 scenes 里设定")
        for ln in s.get("lines", []):
            if ln.get("who") not in chars and ln.get("who") != "narrator":
                problems.append(f"第 {sid} 镜：说话人 {ln.get('who')} 没有设定")
        need = sum(line_seconds(l["text"]) for l in s.get("lines", [])) + 0.3 * len(s.get("lines", [])) + 0.5
        dur = float(s.get("duration", 3))
        if s.get("lines") and need > dur + 0.2:
            warns.append(f"第 {sid} 镜：台词大约要 {need:.1f} 秒，镜头只有 {dur} 秒（合成时会自动拉长，或者拆成两镜）")
        if not s.get("image"):
            problems.append(f"第 {sid} 镜：没有首帧描述 image")
        if len(re.findall(r"然后|接着|随后|再", s.get("video", ""))) >= 2:
            warns.append(f"第 {sid} 镜：视频描述里有好几个连续动作，视频模型一镜只能做好一个动作，考虑拆镜")
        total += max(dur, need if s.get("lines") else dur)
    for k, c in chars.items():
        if not c.get("lock"):
            problems.append(f"角色 {k}：缺少 lock（外貌锁定描述）")
        if not c.get("voice") and any(l.get("who") == k for s in ep.shots for l in s.get("lines", [])):
            warns.append(f"角色 {k}：有台词但没指定 voice，会用默认声线")
    if ep.live:
        total = 0.0
        for g in ep.segments:
            gid = g.get("id")
            for c in g.get("characters", []):
                if c not in chars:
                    problems.append(f"第 {gid} 场：角色 {c} 没有设定")
            if g.get("scene") and g["scene"] not in scenes:
                problems.append(f"第 {gid} 场：场景 {g['scene']} 没有设定")
            sec = float(g.get("seconds", 10))
            if not 4 <= sec <= 15:
                problems.append(f"第 {gid} 场：{sec} 秒，一场要在 4–15 秒之间（H3 一次最长 15 秒）")
            talk = sum(line_seconds(l["text"], live_lang(ep)[0]) + 0.4 for sh in g.get("shots", []) for l in sh.get("lines", []))
            if talk > sec - 1:
                warns.append(f"第 {gid} 场：台词大约要 {talk:.1f} 秒，这场只有 {sec} 秒，会说得很赶，加长或拆场")
            speakers = {l["who"] for sh in g.get("shots", []) for l in sh.get("lines", [])}
            if len(speakers) > 3:
                problems.append(f"第 {gid} 场：{len(speakers)} 个人说话，一场最多 3 个人带参考声音")
            if len(g.get("shots", [])) > 5:
                warns.append(f"第 {gid} 场：{len(g['shots'])} 个镜头，一场 2–4 个镜头最稳")
            print(f"  · 第 {gid} 场：VO {len(g.get('vo', []))} 句")
            problems.extend(vo_errors(ep, g))
            if g.get("vo"):
                # check 不触发 keep 转码；已有视频用实长，没有视频用分镜长度。
                if g.get("keep"):
                    duration = sum(float(k[1]) - float(k[0]) for k in g["keep"])
                else:
                    video = ep.video(g)
                    duration = probe(video)["duration"] if video else sec
                duration -= float(g.get("video_in", 0))
                for i, ln in enumerate(g["vo"], 1):
                    audio = vo_file(ep, g, i)
                    if not audio.exists():
                        warns.append(f"第 {gid} 场 VO {i}：尚未生成音频，先跑 vo")
                    elif isinstance(ln.get("at"), (int, float)) and ln["at"] + probe(audio)["duration"] > duration + 0.01:
                        warns.append(f"第 {gid} 场 VO {i}：结束于 {ln['at'] + probe(audio)['duration']:.2f} 秒，超出剪后场长 {duration:.2f} 秒（合成会在场尾截断）")
            total += sec
        for b in live_untranslated(ep):
            problems.append(f"{b} 还是中文：真人模式除了台词都要写英文，中文会被模型当成台词念出来")
        print(f"《{ep.sb.get('title')}》第 {ep.sb.get('episode', 1)} 集 · 真人模式 · {len(ep.segments)} 场 · 预计 {total:.0f} 秒 · {ep.ratio}")
    else:
        print(f"《{ep.sb.get('title')}》第 {ep.sb.get('episode', 1)} 集 · {len(ep.shots)} 镜 · 预计 {total:.0f} 秒 · {ep.ratio}")
    for p in problems:
        print(f"  ✗ {p}")
    for w in warns:
        print(f"  ⚠ {w}")
    if not problems and not warns:
        print("  ✓ 没发现问题")
    if problems:
        sys.exit(1)


# ---------------------------------------------------------------- Codex 出图

def codex_bin() -> str:
    for c in (os.environ.get("MJ_CODEX_BIN"), shutil.which("codex"), str(Path.home() / ".npm-global/bin/codex")):
        if c and Path(c).exists():
            return c
    die("找不到 codex 命令（先安装 Codex CLI 并登录 ChatGPT）")


def codex_image(ep: Ep, name: str, prompt: str, refs: list[tuple[Path, str]], out: Path, stall: int = 600, limit: int = 1200, ratio: str | None = None) -> tuple[bool, str]:
    """一张图 = 一次 codex exec。卡住（日志 stall 秒不动）或超时就停掉；被安全审核拒绝会返回 'safety'。"""
    job = ep.root / ".codex" / f"{name}-{int(time.time())}"
    job.mkdir(parents=True, exist_ok=True)
    refs = [(p, label) for p, label in refs if p.exists()][:6]
    ref_text = ("附带的图片按顺序是：" + "、".join(f"图{i + 1}（{label}）" for i, (_, label) in enumerate(refs)) + "，画面里的人物、场景、道具要和附图保持一致。") if refs else ""
    text = "\n".join(x for x in [
        f"用你的图片生成工具画一张 {SHAPE.get(ratio or ep.ratio, ratio or ep.ratio)}，用工具能给的最高分辨率和最高画质（长边尽量 2048 像素以上），画面清晰锐利、细节丰富，不要模糊。"
        "然后把生成的图片文件复制保存为当前目录下的 out.png，保存好后只回复“完成”。",
        ref_text,
        "不要修改当前目录以外的任何文件，也不要做别的事。",
        "画面要求：",
        prompt,
    ] if x)
    args = [codex_bin(), "exec", "--skip-git-repo-check", "-s", "workspace-write", "-m", "gpt-6.1-sol", "-c", "model_reasoning_effort=low", "-C", str(job)]  # 出图用 Sol（默认是贵 5 倍的 Astra）
    for p, _ in refs:
        args += ["-i", str(p)]
    args += ["--", text]  # “--” 必须有，否则提示词会被 -i 吃掉
    log = job / "codex.log"
    with open(log, "w") as fh:
        proc = subprocess.Popen(args, cwd=job, stdout=fh, stderr=fh, stdin=subprocess.DEVNULL, start_new_session=True)
        start = time.time()
        while proc.poll() is None:
            time.sleep(3)
            idle = time.time() - log.stat().st_mtime
            if (job / "out.png").exists() and idle > 5:
                break
            if idle > stall or time.time() - start > limit:
                os.killpg(proc.pid, signal.SIGTERM)
                return False, f"卡住了（{int(idle)} 秒没有进展），已停掉"
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
    if (job / "out.png").exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(job / "out.png", out)
        return True, "ok"
    tail = log.read_text("utf-8", errors="ignore")[-600:]
    if re.search(r"安全|拒绝|safety|policy|violat", tail, re.I):
        return False, "safety：被安全审核拒绝了，换个说法（只写静止的画面，不写坠落、伤害、血）再出"
    return False, "没出图：" + tail.strip().replace("\n", " / ")[-300:]


def style_block(ep: Ep) -> str:
    return ep.sb.get("style", "").strip()


def negative_block(ep: Ep) -> str:
    neg = ep.sb.get("negative", "").strip()
    return f"不要：{neg}" if neg else ""


def sheet_refs(ep: Ep, item: dict, name: str) -> list[tuple[Path, str]] | None:
    """设定图的参考图：分镜.json 里写 "refs"，每项是路径（相对集目录），或 {"path": ..., "label": 说明这张图怎么用}。
    这一集没有的图，到 assets_from 那一集找（第 2 集起的贴图常要照第 1 集的道具图）。
    参考图还没出（比如夜景照着同一批里的白天图）就返回 None，这张先不出，免得出来和参考对不上。"""
    refs = []
    for r in item.get("refs", []):
        path, label = (r["path"], r.get("label")) if isinstance(r, dict) else (r, None)
        p = ep.root / path
        if not p.exists() and (ep.assets_root / path).exists():
            p = ep.assets_root / path
        if not p.exists():
            print(f"  ⚠ {name}：参考图 {path} 还没有，这张先不出（参考图出好后再跑一次）")
            return None
        refs.append((p, label or Path(path).stem))
    return refs


def sheet_text(ep: Ep, item: dict, required: str = "") -> list[str]:
    """要写字的设定图（门牌、标签、名片）：写清楚哪些字，其余照样不要文字。"""
    text = item.get("text")
    if not text:
        return [negative_block(ep)]
    return [f"{required}{text}；除此之外不要其他文字、水印和边框。", negative_block(ep).replace("任何文字", "其他文字")]


def cmd_sheets(a):
    ep = Ep(a.dir)
    only = set(a.only.split(",")) if a.only else None
    tasks = []
    for k, c in ep.sb.get("characters", {}).items():
        if only and k not in only:
            continue
        out = ep.sheet(k)
        if out.exists() and not a.force:
            continue
        p = "\n".join(x for x in [style_block(ep),
            f"角色设定图（定妆照）：{c.get('name', k)}。同一个人，左边一张正面全身，中间一张侧面全身，右边一张脸部特写，纯浅灰色背景，光线均匀，没有任何文字。",
            f"外貌：{c['lock']}", negative_block(ep)] if x)
        refs = sheet_refs(ep, c, k)
        if refs is not None:
            tasks.append((k, p, refs, out))
    for k, s in ep.sb.get("scenes", {}).items():
        if only and k not in only:
            continue
        out = ep.sheet(k)
        if out.exists() and not a.force:
            continue
        p = "\n".join(x for x in [style_block(ep), f"场景设定图（空镜，画面里没有人）：{s.get('name', k)}。", s["lock"],
            *sheet_text(ep, s, "画面里要清楚地出现这些字：")] if x)
        refs = sheet_refs(ep, s, k)
        if refs is not None:
            tasks.append((k, p, refs, out))
    for k, pr in ep.sb.get("props", {}).items():
        if only and k not in only:
            continue
        out = ep.sheet(k)
        if out.exists() and not a.force:
            continue
        p = "\n".join(x for x in [style_block(ep), f"道具设定图：{pr.get('name', k)}，放在纯浅灰色背景正中，光线均匀，画面里没有人。", pr["lock"],
            *sheet_text(ep, pr, "道具上清楚地写着：")] if x)
        refs = sheet_refs(ep, pr, k)
        if refs is not None:
            tasks.append((k, p, refs, out))
    # 设定图用横图：三视图并排放得下，不会被挤
    _run_images(ep, tasks, a.jobs, ratio="16:9")


def shot_prompt(ep: Ep, s: dict) -> tuple[str, list[tuple[Path, str]]]:
    chars = ep.sb.get("characters", {})
    scenes = ep.sb.get("scenes", {})
    parts = [style_block(ep), f"景别和机位：{s.get('size', '')} {s.get('camera', '')}".strip()]
    refs: list[tuple[Path, str]] = []
    if s.get("scene") in scenes:
        sc = scenes[s["scene"]]
        parts.append(f"场景：{sc.get('name', s['scene'])}——{sc['lock']}")
        refs.append((ep.sheet(s["scene"]), f"{sc.get('name', s['scene'])}的场景图"))
    for c in s.get("characters", []):
        ch = chars[c]
        parts.append(f"人物 {ch.get('name', c)}：{ch['lock']}")
        refs.insert(0, (ep.sheet(c), f"{ch.get('name', c)}的定妆照"))
    props = ep.sb.get("props", {})
    for k in s.get("props", []):
        pr = props[k]
        parts.append(f"道具 {pr.get('name', k)}：{pr['lock']}")
        refs.append((ep.sheet(k), f"{pr.get('name', k)}的道具图"))
    # 额外参考：别的镜头的首帧（shot03）或者集目录里的图片路径
    for r in s.get("refs", []):
        if re.fullmatch(r"shot\d+", r):
            refs.append((ep.root / "frames" / f"{r}.png", f"第 {int(r[4:])} 镜的画面"))
        else:
            refs.append((ep.root / r, Path(r).stem))
    parts.append(f"这一镜的第一帧：{s['image']}")
    text = s.get("text_in_frame")
    if text:
        parts.append(f"只画这个镜头开始时的那一瞬间。画面里要清楚地出现这些字：{text}；除此之外不要其他文字、字幕、水印和边框。")
        parts.append(negative_block(ep).replace("任何文字", "其他文字"))
    else:
        parts.append("只画这个镜头开始时的那一瞬间，画面里不要任何文字、字幕、水印和边框。")
        parts.append(negative_block(ep))
    return "\n".join(x for x in parts if x), refs


def cmd_frames(a):
    ep = Ep(a.dir)
    want = parse_ids(a.shots)
    tasks = []
    for s in ep.shots:
        if want and int(s["id"]) not in want:
            continue
        out = ep.frame(s)
        if out.exists() and not a.force:
            continue
        p, refs = shot_prompt(ep, s)
        tasks.append((ep.sid(s), p, refs, out))
    missing = {str(p) for _, _, refs, _ in tasks for p, _ in refs if not p.exists()}
    if missing:
        print("⚠ 这些参考图还没有，先跑 sheets（没有参考图人物会不一致）：\n  " + "\n  ".join(sorted(missing)))
    _run_images(ep, tasks, a.jobs)


def _run_images(ep: Ep, tasks, jobs: int, ratio: str | None = None):
    if not tasks:
        print("✓ 都已经有了（要重出加 --force）")
        return
    print(f"Codex 出图 {len(tasks)} 张，同时 {jobs} 张，每张约 1 分钟…")
    fails = []
    with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(codex_image, ep, name, p, refs, out, ratio=ratio): (name, out) for name, p, refs, out in tasks}
        for f in cf.as_completed(futs):
            name, out = futs[f]
            ok, msg = f.result()
            # 用了 assets_from 的集，设定图写在第 1 集的 assets 里，不在这一集目录下
            shown = os.path.relpath(out, ep.root) if ok else msg
            print(f"  {'✓' if ok else '✗'} {name}  {shown}")
            if not ok:
                fails.append(name)
    if fails:
        print(f"失败 {len(fails)} 张：{', '.join(fails)}（改了描述后用 --shots / --only 单独重出）")
        sys.exit(2)


def cmd_inserts(a):
    """出整屏贴图：每场 "inserts" 里还没有图的，按 desc 和 text 用 Codex 出一张和成片同画幅的图，存到 inserts/<id>.png。"""
    ep = Ep(a.dir)
    tasks = []
    for g in ep.segments:
        for ins in g.get("inserts", []):
            if ins.get("video"):  # 盖视频的不用出图
                continue
            out = ep.root / "inserts" / f"{ins['id']}.png"
            if a.only and ins["id"] not in a.only.split(","):
                continue
            if out.exists() and not a.force:
                continue
            text = ins.get("text")  # 只写图里真出现的字（韩文照原样）；中文翻译写在 "sub"，合成时当字幕
            rows = [r.strip() for r in re.split(r"\s+/\s+|\n", text) if r.strip()] if text else []  # 分行可以写 “ / ” 也可以直接换行
            show = "；".join(f"第 {i} 行“{r}”" for i, r in enumerate(rows, 1)) if len(rows) > 1 else (f"“{rows[0]}”" if rows else "")
            p = "\n".join(x for x in [
                f"{'竖屏' if ep.H > ep.W else '横屏'}整屏画面：{ins['desc']}",
                f"画面里要清楚、准确地出现这些字（照抄，韩文一个字母都不能错；分行的按顺序分开排，不要画出“第几行”和引号）：{show}；除此之外不要别的字、水印和边框（界面上的图标、头像、按钮、分隔线可以有，让它像真的手机界面，但上面不要字）。" if text else "画面里不要任何文字、水印和边框。",
                "像真实的手机截图、屏幕画面或实物特写，干净、清楚、一眼看得懂；不要真实品牌的标志。"] if x)
            refs = sheet_refs(ep, ins, ins["id"])
            if refs is not None:
                tasks.append((ins["id"], p, refs, out))
    _run_images(ep, tasks, a.jobs, ratio=ep.ratio)  # 贴图跟成片同一个画幅，横屏剧出横图，合成时才不会被裁掉一大半


# ---------------------------------------------------------------- 图转视频清单

def cmd_handoff(a):
    ep = Ep(a.dir)
    chars = ep.sb.get("characters", {})
    rows = [f"# 图转视频清单 · 《{ep.sb.get('title')}》第 {ep.sb.get('episode', 1)} 集", "",
            f"- 比例选 **{ep.ratio}**；每镜**只放首帧，不放尾帧**（首尾帧一起给容易穿模、变形）。",
            "- 推荐：Google AI Studio 的 Veo 3.1 Fast（Pro 会员在网页里出不花钱）或 Flow；每镜出 2 条挑一条。",
            f"- 下载后改名放进 `{ep.root / 'videos'}`：第 3 镜就叫 `shot03.mp4`（多条可以叫 shot03a.mp4、shot03b.mp4，合成时用 shot03.mp4 或名字最短的那条）。",
            "- 没放视频的镜头，合成时会用首帧做缓推（动态漫），照样能出片。", ""]
    style = ep.sb.get("video_style") or ep.sb.get("style", "")
    for s in ep.shots:
        who = "、".join(chars[c].get("name", c) for c in s.get("characters", []) if c in chars)
        lines = "；".join(f"{chars.get(l['who'], {}).get('name', '旁白')}：“{l['text']}”" for l in s.get("lines", []))
        video = s.get("video", "")
        framing = s.get("size", "") if "镜头" in video else f"{s.get('size', '')}，{s.get('camera', '')}"
        no_text = (f"画面里的“{s['text_in_frame']}”这些字保持清楚不变；除此之外不要字幕、文字、水印和胶片边框。" if s.get("text_in_frame")
                   else "画面里不要字幕、文字、水印和胶片边框。")
        prompt = "\n".join(x for x in [style, f"{framing.strip('，')}。", video,
                                        "人物不说话，嘴巴自然闭合（配音后期加）。" if not s.get("lip_sync") else (f"台词：{lines}" if lines else ""),
                                        no_text] if x)
        have = "✓ 已有视频" if ep.video(s) else ""
        rows += [f"## 第 {s['id']} 镜 · {s.get('duration', 3)} 秒 {have}", "",
                 f"- 首帧：`{ep.frame(s)}`" + ("" if ep.frame(s).exists() else "（还没出）"),
                 f"- 人物：{who or '无'}" + (f"；台词（后期配音）：{lines}" if lines else ""), "", "```", prompt, "```", ""]
    path = ep.root / "图转视频清单.md"
    path.write_text("\n".join(rows), "utf-8")
    print(f"✓ {path}")


# ---------------------------------------------------------------- 配音

async def _tts_one(text: str, voice: str, rate: str, pitch: str, out: Path):
    import edge_tts
    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(str(out))


def tts_line(text: str, voice: str, emotion: str, out: Path) -> str:
    rate, pitch = EMOTION.get(emotion, EMOTION["平静"])
    try:
        asyncio.run(_tts_one(text, voice, rate, pitch, out))
        return "edge-tts"
    except Exception as e:  # 没网或被墙时退回 macOS 自带的 say（声音一般，但能出）
        aiff = out.with_suffix(".aiff")
        run(["say", "-v", os.environ.get("MJ_SAY_VOICE", "Tingting"), "-o", str(aiff), text])
        run([FFMPEG, "-y", "-v", "error", "-i", str(aiff), str(out)])
        aiff.unlink(missing_ok=True)
        return f"say（edge-tts 失败：{str(e)[:80]}）"


def trim_silence(path: Path):
    """edge-tts 的音频前后带着静音（结尾约 1 秒），不剪掉的话每句都会多占 1 秒、字幕也挂得太久。"""
    tmp = path.with_name(path.stem + ".trim" + path.suffix)
    trim = "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.03"
    run([FFMPEG, "-y", "-v", "error", "-i", str(path), "-af", f"{trim},areverse,{trim},areverse,apad=pad_dur=0.08", str(tmp)])
    if probe(tmp)["duration"] > 0.2:
        tmp.replace(path)
    else:
        tmp.unlink(missing_ok=True)


def cmd_tts(a):
    ep = Ep(a.dir)
    chars = ep.sb.get("characters", {})
    voices = {**DEFAULT_VOICES, **{k: c.get("voice") for k, c in chars.items() if c.get("voice")}}
    index = {}
    n = 0
    for s in ep.shots:
        for i, ln in enumerate(s.get("lines", []), 1):
            out = ep.root / "audio" / f"{ep.sid(s)}_{i}.mp3"
            voice = voices.get(ln["who"]) or "zh-CN-XiaoxiaoNeural"
            if not out.exists() or a.force:
                how = tts_line(ln["text"], voice, ln.get("emotion", "平静"), out)
                trim_silence(out)
                n += 1
                print(f"  ✓ {out.name}  {voice} · {ln.get('emotion', '平静')} · {how}  “{ln['text']}”")
            index[out.name] = {"shot": s["id"], "who": ln["who"], "text": ln["text"], "seconds": round(probe(out)["duration"], 2)}
    (ep.root / "audio" / "lines.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), "utf-8")
    print(f"✓ 配音 {len(index)} 句（新生成 {n} 句）")


# ---------------------------------------------------------------- 真人模式：参考声音 + 远程 5090 拍场

LIVE_LANG = {"Chinese": ("zh", "Chinese"), "Korean": ("ko", "Korean"), "Japanese": ("ja", "Japanese"), "English": ("en", "English")}


def live_lang(ep: "Ep") -> tuple[str, str]:
    """真人模式台词用什么语言说：(whisper 语言代码, H3 的 <d>[语言] 标记)。字幕用每句的 sub（不写就用台词本身）。"""
    return LIVE_LANG.get(ep.sb.get("live_lang", "Chinese"), LIVE_LANG["Chinese"])


def vo_file(ep: Ep, segment: dict, index: int) -> Path:
    return ep.root / "audio" / f"vo_{ep.sid(segment)}_{index}.wav"


def vo_errors(ep: Ep, segment: dict) -> list[str]:
    errors = []
    for i, ln in enumerate(segment.get("vo", []), 1):
        label = f"第 {segment['id']} 场 VO {i}"
        if not isinstance(ln.get("text"), str) or not ln["text"].strip():
            errors.append(f"{label}：缺少 text")
        at = ln.get("at")
        if isinstance(at, bool) or not isinstance(at, (int, float)) or not math.isfinite(at) or at < 0:
            errors.append(f"{label}：at 必须是非负秒数")
        if ln.get("kind") not in ("narration", "inner"):
            errors.append(f"{label}：kind 必须是 narration 或 inner")
        who = ln.get("who")
        if who == "narrator":
            if not ep.sb.get("narrator", {}).get("voice_design"):
                errors.append(f"{label}：缺少 narrator.voice_design")
        elif who not in ep.sb.get("characters", {}):
            errors.append(f"{label}：角色 {who} 没有设定")
        else:
            if not ep.voice_ref(who).exists():
                errors.append(f"{label}：缺少角色参考声音，先跑 voices")
            if not ep.voice_char(who).get("voice_sample"):
                errors.append(f"{label}：缺少角色 voice_sample（须与参考声音一致）")
    return errors


def cmd_vo(a):
    """旁白设计声线，人物内心独白克隆参考声线；音频按句缓存。"""
    ep = Ep(a.dir)
    if not ep.live:
        die("vo 仅用于真人模式（mode: live）")
    ids = parse_ids(a.segments)
    segments = [g for g in ep.segments if ids is None or int(g["id"]) in ids]
    errors = [e for g in segments for e in vo_errors(ep, g)]
    if errors:
        die("\n  ".join(errors))
    import comfy
    comfy.requests.packages.urllib3.disable_warnings()
    uploaded, count, generated = {}, 0, 0
    for g in segments:
        for i, ln in enumerate(g.get("vo", []), 1):
            count += 1
            out = vo_file(ep, g, i)
            if out.exists() and not a.force:
                print(f"  · {out.name} 已有配音", flush=True)
                continue
            who, prefix = ln["who"], f"manju/{ep.out_name()}-vo-{ep.sid(g)}-{i}"
            if who == "narrator":
                graph = comfy.tts_design_graph(ln["text"], ep.sb["narrator"]["voice_design"], live_lang(ep)[1], prefix)
            else:
                if who not in uploaded:
                    uploaded[who] = comfy.upload(ep.voice_ref(who))
                graph = comfy.tts_clone_graph(ln["text"], uploaded[who], ep.voice_char(who)["voice_sample"], live_lang(ep)[1], prefix)
            print(f"  生成 {out.name}：{ln['text']}", flush=True)
            raw = ep.root / "build" / f"{out.stem}.raw.flac"
            comfy.download(comfy.wait(comfy.submit(graph)), raw)
            tmp = ep.root / "build" / out.name
            run([FFMPEG, "-y", "-v", "error", "-i", str(raw), "-ar", "48000", "-ac", "2", str(tmp)])
            trim_silence(tmp)
            if probe(tmp)["duration"] <= 0.2:
                die(f"{out.name} 生成的声音太短，未覆盖已有配音")
            tmp.replace(out)
            generated += 1
            print(f"  ✓ {out.name} · {probe(out)['duration']:.2f} 秒", flush=True)
    print(f"✓ VO {count} 句（新生成 {generated} 句）")


DEMUCS_PY = os.environ.get("MJ_DEMUCS_PY") or CFG.get("demucs_python") or "python3"  # 装了 demucs 的那个 python


def cmd_redub(a):
    """把拍好的场里的台词换成定好的声线，口型不动：
    Demucs 分出人声 → whisper 找到每句台词的起止 → Qwen3 用角色参考声音克隆念这句 → 拉伸到原句一样长 → 盖回原位。
    原视频存到 videos/原声/，再跑一遍会从原声重做。某句不想换（尖叫、喘气）就在台词上写 "redub": false。"""
    ep = Ep(a.dir)
    if not ep.live:
        die("redub 只用于真人模式")
    try:
        has_demucs = subprocess.run([DEMUCS_PY, "-c", "import demucs"], capture_output=True).returncode == 0
    except OSError:
        has_demucs = False
    if not has_demucs:
        die("redub 要用 demucs 分出人声：找一个 python 装上 demucs（pip install demucs），"
            "再把它的路径写进 ~/.config/manju/config.json 的 demucs_python")
    import comfy
    comfy.requests.packages.urllib3.disable_warnings()
    ids = parse_ids(a.segments)
    keep_dir = ep.root / "videos" / "原声"
    keep_dir.mkdir(parents=True, exist_ok=True)
    uploaded = {}
    for g in ep.segments:
        if (ids and int(g["id"]) not in ids) or g.get("from"):
            continue
        script = [l for sh in g.get("shots", []) for l in sh.get("lines", [])]
        if not any(l.get("redub", True) and l["who"] in ep.sb.get("characters", {}) for l in script):
            continue
        # 一句台词里有停顿（“哎哟，夏俊来啦！……卖不卖随你。”），按小句分开对时间、分开念，停顿才对得上嘴
        script = [{**l, "text": part} for l in script for part in re.split(r"(?<=[.!?…])\s+", spoken(l["text"])) if part]
        cur = ep.root / "videos" / f"{ep.sid(g)}.mp4"
        src = keep_dir / f"{ep.sid(g)}.mp4"
        if not src.exists():
            if not cur.exists():
                print(f"  · 第 {g['id']} 场还没拍")
                continue
            shutil.copy2(cur, src)
        elif cur.exists() and cur.stat().st_mtime > src.stat().st_mtime + 1 and not a.force:
            print(f"  · 第 {g['id']} 场已换过声音（重做加 --force）")
            continue
        timed = align_lines([spoken(l["text"]) for l in script], whisper_words(ep, src, live_lang(ep)[0]))
        build = ep.root / "build" / "redub" / ep.sid(g)
        build.mkdir(parents=True, exist_ok=True)
        wav = build / "mix.wav"
        run([FFMPEG, "-y", "-v", "error", "-i", str(src), "-vn", "-ar", "44100", "-ac", "2", str(wav)])
        run([DEMUCS_PY, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "-d", "cpu", "-o", str(build), str(wav)], timeout=900)
        stems = build / "htdemucs" / "mix"
        inputs, filters, mutes, mixed = ["-i", str(src), "-i", str(stems / "no_vocals.wav"), "-i", str(stems / "vocals.wav")], [], [], ["[bg]", "[vox]"]
        n = 3
        for j, (l, tm) in enumerate(zip(script, timed), 1):
            who = l["who"]
            if not tm or not l.get("redub", True) or who not in ep.sb.get("characters", {}):
                print(f"  · 第 {g['id']} 场第 {j} 句保留原声：{l['text']}" + ("（没听到这句）" if not tm else ""))
                continue
            if who not in uploaded:
                uploaded[who] = comfy.upload(ep.voice_ref(who))
            raw = build / f"line{j}.flac"
            graph = comfy.tts_clone_graph(spoken(l["text"]), uploaded[who], ep.voice_char(who)["voice_sample"], live_lang(ep)[1], f"manju/{ep.out_name()}-redub-{ep.sid(g)}-{j}")
            comfy.download(comfy.wait(comfy.submit(graph, front=True)), raw)
            line = build / f"line{j}.wav"
            run([FFMPEG, "-y", "-v", "error", "-i", str(raw), "-ar", "44100", "-ac", "2", str(line)])
            trim_silence(line)
            a0, a1 = tm
            want = max(0.3, a1 - a0)
            speed = probe(line)["duration"] / want
            tempo = min(1.35, max(0.8, speed))  # 拉得太狠会变调发怪，超出的部分宁可略长略短
            filters.append(f"[{n}:a]atempo={tempo:.4f},adelay={int(a0 * 1000)}|{int(a0 * 1000)}[l{j}]")
            inputs += ["-i", str(line)]
            mixed.append(f"[l{j}]")
            mutes.append(f"between(t,{max(0, a0 - 0.05):.3f},{a1 + 0.1:.3f})")
            n += 1
            print(f"  ✓ 第 {g['id']} 场第 {j} 句（{ep.sb['characters'][who].get('name', who)}）：{l['text']} · 拉伸 {tempo:.2f}" + ("（长度差得多，听一下）" if tempo != speed else ""))
        if not mutes:
            continue
        filters.insert(0, f"[2:a]volume=0:enable='{'+'.join(mutes)}'[vox]")
        filters.insert(0, "[1:a]anull[bg]")
        filters.append(f"{''.join(mixed)}amix=inputs={len(mixed)}:normalize=0:duration=first[a]")
        out = build / "out.mp4"
        run([FFMPEG, "-y", "-v", "error", *inputs, "-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out)])
        out.replace(cur)
        print(f"  ✓ videos/{cur.name} 换好声音（原声在 videos/原声/）")


def cmd_voices(a):
    """每个角色一段 6–9 秒的参考声音。之后每一场都把它当 <Audio n> 传给 H3，声线就固定了。
    角色写了 voice_ref（自己录的或克隆好的音频）就用它，否则用 edge-tts 的 voice 读一段 voice_sample。"""
    ep = Ep(a.dir)
    for k, c in ep.sb.get("characters", {}).items():
        if c.get("voice_of"):
            continue  # 换装造型，声音跟着原角色
        out = ep.voice_ref(k)
        if out.exists() and not a.force:
            print(f"  · {c.get('name', k)} 已有参考声音")
            continue
        if c.get("voice_ref"):
            src = (ep.root / c["voice_ref"]) if not Path(c["voice_ref"]).is_absolute() else Path(c["voice_ref"])
            run([FFMPEG, "-y", "-v", "error", "-i", str(src), "-t", "12", "-ar", "44100", "-ac", "2", str(out)])
            print(f"  ✓ {c.get('name', k)}：用 {src.name}")
            continue
        own = [l["text"] for g in ep.segments for sh in g.get("shots", []) for l in sh.get("lines", []) if l["who"] == k]
        if c.get("voice_design"):
            # 在远程 5090 上用 Qwen3-TTS“设计”一个声音：按描述生成，比 edge-tts 自然得多，支持韩语、日语等
            sys.path.insert(0, str(Path(__file__).parent))
            import comfy
            comfy.requests.packages.urllib3.disable_warnings()
            text = c.get("voice_sample") or " ".join(own)
            lang = c.get("voice_lang") or ep.sb.get("live_lang", "Chinese")
            raw = out.with_suffix(".flac")
            comfy.download(comfy.wait(comfy.submit(comfy.tts_design_graph(text, c["voice_design"], lang, f"manju/voice-{k}"), front=True)), raw)
            run([FFMPEG, "-y", "-v", "error", "-i", str(raw), "-ar", "44100", "-ac", "2", str(out)])
            raw.unlink(missing_ok=True)
            trim_silence(out)
            print(f"  ✓ {c.get('name', k)}：Qwen3 声音设计 · {probe(out)['duration']:.1f} 秒 · “{c['voice_design'][:40]}…”")
            continue
        text = c.get("voice_sample") or ("。".join(re.sub(r"[，。！？、…,.!?]+$", "", t) for t in own) + "。今天的雨下得真大，我们改天再说吧。")[:60]
        mp3 = out.with_suffix(".mp3")
        tts_line(text, c.get("voice") or "zh-CN-XiaoxiaoNeural", c.get("voice_emotion", "平静"), mp3)
        trim_silence(mp3)
        run([FFMPEG, "-y", "-v", "error", "-i", str(mp3), "-ar", "44100", "-ac", "2", str(out)])
        mp3.unlink(missing_ok=True)
        print(f"  ✓ {c.get('name', k)}：{c.get('voice')} · {probe(out)['duration']:.1f} 秒 “{text}”")


def segment_media(ep: Ep, g: dict) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    """这一场带哪些参考图（人物、场景、道具、额外参考、上一场最后一帧）和参考声音（说话的人）。"""
    chars, scenes, props = ep.sb.get("characters", {}), ep.sb.get("scenes", {}), ep.sb.get("props", {})
    pics: list[tuple[Path, str]] = []
    for c in g.get("characters", []):
        pics.append((ep.sheet(c), f"char:{c}"))
    if g.get("scene") in scenes:
        pics.append((ep.sheet(g["scene"]), f"scene:{g['scene']}"))
    for k in g.get("props", []):
        if k in props:
            pics.append((ep.sheet(k), f"prop:{k}"))
    for r in g.get("refs", []):
        pics.append(((ep.root / "frames" / f"{r}.png") if re.fullmatch(r"shot\d+", r) else ep.root / r, f"ref:{r}"))
    if g.get("continue"):
        prev = next((x for x in ep.segments if int(x["id"]) == int(g["id"]) - 1), None)
        pv = ep.video(prev) if prev else None
        if pv:
            last = ep.root / "build" / f"{ep.sid(g)}_from_prev.png"
            run([FFMPEG, "-y", "-v", "error", "-sseof", "-0.1", "-i", str(pv), "-frames:v", "1", "-update", "1", str(last)])
            pics.append((last, "prev:last"))
    speakers = []
    for sh in g.get("shots", []):
        for l in sh.get("lines", []):
            if l["who"] in chars and l["who"] not in speakers:
                speakers.append(l["who"])
    auds = [(ep.voice_ref(k), f"voice:{k}") for k in speakers[:3]]
    return pics[:9], auds


TAIL_FRAMES = 22  # H3 能钉住的片段长度是 5/22/39… 帧；22 帧 ≈ 0.92 秒


def segment_guides(ep: Ep, g: dict, comfy) -> tuple[list[dict], float]:
    """接续拍（H3-B 的做法）和关键帧（H3-A 的做法），接在多角色参考后面，人物参考和声线不丢。
    "extend": true —— 把上一场最后 22 帧画面和声音钉在这一场开头，动作、光线、声音无缝接上；
    "keyframes": [{"image": "assets/xx.png", "at": 0}] —— 在第 at 秒钉一张画面（比如“闭着眼”的开头）。
    返回 (guides, 要从开头去掉的秒数)。"""
    guides, overlap = [], 0.0
    if g.get("extend"):
        prev = next((x for x in ep.segments if int(x["id"]) == int(g["id"]) - 1), None)
        pv = ep.video(prev) if prev else None
        if not pv:
            die(f"第 {g['id']} 场要接着上一场拍（extend），但上一场还没有视频")
        tail = ep.root / "build" / f"{ep.sid(g)}_tail.mp4"
        sec = TAIL_FRAMES / 24
        run([FFMPEG, "-y", "-v", "error", "-sseof", f"-{sec + 0.3:.3f}", "-i", str(pv), "-vf", f"fps=24,reverse,trim=end_frame={TAIL_FRAMES},reverse,setpts=PTS-STARTPTS",
             "-af", f"areverse,atrim=0:{sec:.4f},areverse,asetpts=N/SR/TB", "-c:v", "libx264", "-crf", "12", "-c:a", "aac", str(tail)])
        guides.append({"video": comfy.upload(tail), "frame": 0})
        overlap = sec
    for k in g.get("keyframes", []):
        guides.append({"image": comfy.upload(ep.root / k["image"]), "frame": round(float(k.get("at", 0)) * 24)})
    return guides, overlap


EMOTION_EN = {
    "平静": "calm", "温柔": "gentle", "开心": "happy", "激动": "excited, trembling", "紧张": "nervous", "愤怒": "angry",
    "冷漠": "cold, flat", "悲伤": "sad, sighing", "低语": "soft, whispering", "命令": "commanding", "嘲讽": "mocking",
}
HAS_CJK = re.compile(r"[\u4e00-\u9fff]")


def en(d: dict, key: str) -> str:
    """真人模式的提示词除了台词全部用英文：H3 会把提示词里的中文当成台词念出来。优先用 xxx_en 字段。"""
    return (d.get(f"{key}_en") or d.get(key) or "").strip()


# 台词里括号中的是动作提示，不念出来：(딸꾹) 会被模型当字念成 “ttal-kkuk”。认得的提示自动写成英文动作，其余的在这句写 "cue_en"
STAGE_EN = {"딸꾹": "hiccups once: a short, loud, involuntary 'hic!' that jerks the shoulders up",
            "嗝": "hiccups once: a short, loud, involuntary 'hic!' that jerks the shoulders up"}


def spoken(text: str) -> str:
    """真正要念出来的字：去掉括号里的动作提示。"""
    return re.sub(r"\s*[(（][^)）]*[)）]", "", text).strip()


def stage_cue(line: dict, who: str) -> str:
    """这句念完以后的动作或声音（英文）：先用 cue_en，没有就按括号里认得的提示生成。"""
    if en(line, "cue"):
        return en(line, "cue")
    cues = [STAGE_EN[c.strip()] for c in re.findall(r"[(（]([^)）]*)[)）]", line["text"]) if c.strip() in STAGE_EN]
    if not cues:
        return ""
    lead = f"Right after the line, {who}" if spoken(line["text"]) else who
    return f"{lead} " + ", then ".join(cues) + "."


def live_untranslated(ep: Ep) -> list[str]:
    """真人模式里还写着中文、会被模型念出来的描述。"""
    bad = []
    for k, c in ep.sb.get("characters", {}).items():
        for f in ("name", "lock"):
            if HAS_CJK.search(en(c, f)):
                bad.append(f"角色 {k}.{f}_en")
    for k, sc in ep.sb.get("scenes", {}).items():
        if HAS_CJK.search(en(sc, "lock")):
            bad.append(f"场景 {k}.lock_en")
    for k, pr in ep.sb.get("props", {}).items():
        for f in ("name", "lock"):
            if HAS_CJK.search(en(pr, f)):
                bad.append(f"道具 {k}.{f}_en")
    for g in ep.segments:
        if HAS_CJK.search(en(g, "summary")) or HAS_CJK.search(en(g, "sound")) or HAS_CJK.search(g.get("lighting_en", "") + g.get("live_style_en", "")):
            bad.append(f"第 {g['id']} 场 summary_en / sound_en")
        for j, sh in enumerate(g.get("shots", []), 1):
            if HAS_CJK.search(en(sh, "camera")) or HAS_CJK.search(en(sh, "action")):
                bad.append(f"第 {g['id']} 场第 {j} 镜 camera_en / action_en")
    if HAS_CJK.search(ep.sb.get("live_style", "")) or HAS_CJK.search(ep.sb.get("live_sound", "")):
        bad.append("live_style / live_sound")
    return bad


def segment_prompt(ep: Ep, g: dict, pics, auds, timing=None) -> str:
    """MiniMax-H3 参考生视频的六段式提示词（官方格式）：先定义每张参考图/参考声音是谁，再按 [Shot n] 写镜头、动作和台词。
    台词放在 <d>[Chinese] …</d> 里，其余全部英文。
    说话的人按开口顺序编 (S1)/(S2)（官方 ref 指南的写法）。timing 是先配音后拍时每句的 (起, 止, 谁)，有它就按真实时间写镜头和台词。"""
    chars, scenes, props = ep.sb.get("characters", {}), ep.sb.get("scenes", {}), ep.sb.get("props", {})
    defs, keep, subj = [], [], {}
    aud_of = {tag.split(":", 1)[1]: i + 1 for i, (_, tag) in enumerate(auds)}
    # 场景图都是照亮了画的；闹鬼、吓人的场要换灯光时写 "lighting_en"：地方和布局照参考图，灯光不照
    relit = g.get("lighting_en")
    place = (lambda i: f"the place and layout exactly as in <Picture {i}>, but NOT its lighting: {relit}") if relit else \
            (lambda i: f"the place, layout and lighting exactly as in <Picture {i}>")
    n = 0
    for i, (_, tag) in enumerate(pics, 1):
        kind, key = tag.split(":", 1)
        if kind in ("char", "scene", "prop"):
            n += 1
        if kind == "char":
            c = chars[key]
            name = en(c, "name")
            voice = f" {name[:1].upper() + name[1:]}'s voice is the voice in <Audio {aud_of[key]}>." if key in aud_of else ""
            defs.append(f"<Subject {n}> is {name}, the person in <Picture {i}>: {en(c, 'lock')}.{voice}")
            keep.append(f"<Subject {n}>: fully_preserved - identity, face, hair and clothing exactly as in <Picture {i}>.")
            subj[key] = (n, name)
        elif kind == "scene":
            defs.append(f"<Subject {n}> is the environment in <Picture {i}>: {en(scenes[key], 'lock')}.")
            keep.append(f"<Subject {n}>: fully_preserved - {place(i)}.")
        elif kind == "prop":
            pr = props[key]
            pname = en(pr, "name")
            art = "" if pname.lower().startswith("the ") or pname[:1].isupper() else "the "  # 名字自带 the 或是专名（Gyul the cat）就不再加
            defs.append(f"<Subject {n}> is {art}{pname} in <Picture {i}>: {en(pr, 'lock')}.")
            keep.append(f"<Subject {n}>: fully_preserved - shape, color, material and any carved marks exactly as in <Picture {i}>.")
        elif kind == "prev":
            defs.append(f"<Picture {i}> is the last frame of the previous clip; [Shot 1] starts exactly from this frame and continues the same moment without a cut.")
        else:
            rkey = Path(key).stem
            if rkey in scenes:  # 额外参考是另一个场景的图：写明是这场里的另一个地点，镜头里说到它，模型才知道照哪张
                n += 1
                defs.append(f"<Subject {n}> is another location used in this clip, shown in <Picture {i}>: {en(scenes[rkey], 'lock')}.")
                keep.append(f"<Subject {n}>: fully_preserved - {place(i)}.")
            elif rkey in props:
                n += 1
                pname = en(props[rkey], "name")
                art = "" if pname.lower().startswith("the ") or pname[:1].isupper() else "the "
                defs.append(f"<Subject {n}> is {art}{pname} in <Picture {i}>: {en(props[rkey], 'lock')}.")
                keep.append(f"<Subject {n}>: fully_preserved - shape, color, material and any carved marks exactly as in <Picture {i}>.")
            else:
                defs.append(f"<Picture {i}> is an extra visual reference for continuity.")
    for k, (_, tag) in enumerate(auds, 1):  # 只闻其声的人（电话里的中介）：没有照片，也要说清哪段参考声音是他的
        kind, key = tag.split(":", 1)
        if kind == "voice" and key not in subj:
            name = en(chars.get(key, {}), "name") or key
            who = en(chars.get(key, {}), "lock")
            defs.append((f"{name[:1].upper() + name[1:]} is {who}. " if who else "") + f"{name[:1].upper() + name[1:]}'s voice is the voice in <Audio {k}>.")
    shots = g.get("shots", [])
    sec = float(g.get("seconds", 10))
    sid = {}
    for sh in shots:
        for l in sh.get("lines", []):
            sid.setdefault(l["who"], len(sid) + 1)
    for _, tag in auds:
        if tag.startswith("drive:"):
            names = ", ".join(f"S{sid[k]} is <Subject {subj[k][0]}> ({subj[k][1]})" for k in sid if k in subj)
            defs.append(f"<Audio 1> is the finished pre-recorded dialogue track of this clip and is copied exactly as the clip's audio (fully_copy); {names}. "
                        "Each line is spoken only by its own speaker: only that person's lips move, in sync with the track; whoever is listening keeps their mouth closed and reacts silently.")
    weights = [max(2.0, sum(line_seconds(l["text"]) + 0.8 for l in sh.get("lines", [])) + 1.0) for sh in shots] or [1.0]
    # 镜头写了 "seconds" 就给它这么长（没台词的动作镜头按字数分只能分到一秒多），其余镜头按台词多少分剩下的时间
    fixed = {j: float(sh["seconds"]) for j, sh in enumerate(shots) if sh.get("seconds")}
    left = sec - sum(fixed.values())
    free = sum(w for j, w in enumerate(weights) if j not in fixed)
    if fixed and len(fixed) < len(weights) and left >= len(weights) - len(fixed):
        durs = [fixed.get(j, w / free * left) for j, w in enumerate(weights)]
    elif fixed and len(fixed) == len(weights):
        durs = [fixed[j] * sec / sum(fixed.values()) for j in range(len(weights))]
    else:
        durs = [w * sec / sum(weights) for w in weights]
    starts = []
    for j in range(len(shots)):
        starts.append(sum(durs[:j]))
    times = iter(timing or [])
    if timing:
        # 有真实台词时间：每个镜头从它第一句开口前 0.3 秒开始；没台词的镜头接在上一个镜头最后一句后面
        k, last_end = 0, 0.0
        for j, sh in enumerate(shots):
            n_l = len(sh.get("lines", []))
            if j and n_l:
                starts[j] = max(starts[j - 1] + 1.0, timing[k][0] - 0.3)
            elif j:
                starts[j] = max(starts[j - 1] + (fixed.get(j - 1, 1.0) if not shots[j - 1].get("lines") else 1.0), last_end + 0.3)
            if n_l:
                last_end = timing[k + n_l - 1][1]
            k += n_l
    parts = []
    for j, sh in enumerate(shots, 1):
        action = en(sh, "action")
        for key, (num, name) in subj.items():
            action = action.replace(name, f"<Subject {num}> ({name})", 1)
        say = []
        for l in sh.get("lines", []):
            num, name = subj.get(l["who"], (None, en(chars.get(l["who"], {}), "name") or l["who"]))
            who = f"<Subject {num}> ({name}) (S{sid[l['who']]})" if num else f"{name} (S{sid[l['who']]})"
            at = next(times, None)
            when = f" from 00:{at[0]:06.3f} to 00:{at[1]:06.3f}" if at else ""
            cue = stage_cue(l, f"<Subject {num}> ({name})" if num else name)
            if spoken(l["text"]):
                say.append(f"{who} says{when} in a {EMOTION_EN.get(l.get('emotion', ''), 'natural')} voice, <d>[{live_lang(ep)[1]}] {spoken(l['text'])}</d>" + (f" {cue}" if cue else ""))
            elif cue:  # 整句只有动作提示（比如只打一个嗝）：不给台词标记，写成动作
                say.append(cue)
        head = f"[Shot {j}]" if j == 1 else f"[Shot {j}] At 00:{starts[j - 1]:06.3f},"
        parts.append(f"{head} {en(sh, 'camera')}. {action}{' ' + ' '.join(say) if say else ''}")
    style = g.get("live_style_en") or ep.sb.get("live_style", "")  # 闹鬼的场可以换掉全剧“明亮、不要恐怖感”的统一画风
    # 触发词默认不写进提示词：H3 会把 r34l1sm 这种词当台词念出来（实测成片里有一句“234 LLSM”）。LoRA 不靠它也生效。
    trigger = " ".join(lo["trigger"] for lo in ep.sb.get("live_loras", []) if lo.get("trigger") and lo.get("trigger_in_prompt"))
    if trigger:
        style = f"{trigger}, {style}"
    prompt = {
        "subject_definitions": defs,
        "summary": f"[reference generation] {en(g, 'summary')} A {sec:g}-second {'vertical' if ep.H > ep.W else 'horizontal widescreen'} {ep.ratio} clip with {len(shots)} shots, realistic live-action look.",
        "retention_analysis": keep,
        "detailed_description": f"{style} " + " ".join(parts),
        "overall_soundscape": en(g, "sound") or ep.sb.get("live_sound") or "Natural ambient sound of the scene. No background music.",
        "non_diegetic_music": "N/A",
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def drive_track(ep: Ep, g: dict, comfy, force: bool = False):
    """先配音后对口型：用每个角色定好的声音（Qwen3 克隆）把这一场的台词按顺序念好，排成一条音轨，拍的时候锁进 H3。
    可以几个人对话：每句用说话人自己的声音。返回 (音轨, 说话的人列表, 音轨秒数, 每句 (起, 止, 谁))。"""
    script = [l for sh in g.get("shots", []) for l in sh.get("lines", [])]
    if not script:
        die(f"第 {g['id']} 场要先配音后拍（drive），但没有台词")
    refs, whos = {}, []
    parts, timing, t = [], [], float(g.get("drive_lead", 0.4))
    for j, l in enumerate(script, 1):
        who = l["who"]
        c = ep.voice_char(who)
        if who not in whos:
            whos.append(who)
        wav = ep.root / "audio" / f"drive_{ep.sid(g)}_{j}.wav"
        stamp = wav.with_suffix(".txt")
        tag = hashlib.sha1(json.dumps([who, l["text"], c.get("voice_design"), ep.voice_ref(who).stat().st_mtime]).encode()).hexdigest()[:10]
        if force or not wav.exists() or not stamp.exists() or stamp.read_text() != tag:
            refs[who] = refs.get(who) or comfy.upload(ep.voice_ref(who))
            raw = wav.with_suffix(".flac")
            wav.parent.mkdir(exist_ok=True)
            comfy.download(comfy.wait(comfy.submit(comfy.tts_clone_graph(spoken(l["text"]) or "…", refs[who], c["voice_sample"], live_lang(ep)[1], f"manju/{ep.out_name()}-drive-{ep.sid(g)}-{j}"), front=True)), raw)
            run([FFMPEG, "-y", "-v", "error", "-i", str(raw), "-ar", "48000", "-ac", "2", str(wav)])
            raw.unlink(missing_ok=True)
            trim_silence(wav)
            stamp.write_text(tag)
            print(f"  ✓ 配好第 {j} 句（{c.get('name', who)}）：{l['text']} · {probe(wav)['duration']:.1f} 秒")
        d = probe(wav)["duration"]
        parts.append((t, wav))
        timing.append((round(t, 3), round(t + d, 3), who))
        t += d + float(l.get("pause", 0.35))
    seconds = max(float(g.get("seconds", 10)), round(t + 0.5, 2))
    out = ep.root / "audio" / f"drive_{ep.sid(g)}.wav"
    inputs, f = [], []
    for i, (at, w) in enumerate(parts):
        inputs += ["-i", str(w)]
        f.append(f"[{i}:a]adelay={int(at * 1000)}|{int(at * 1000)}[p{i}]")
    f.append(f"{''.join(f'[p{i}]' for i in range(len(parts)))}amix=inputs={len(parts)}:normalize=0,apad=whole_dur={seconds}[a]")
    run([FFMPEG, "-y", "-v", "error", *inputs, "-filter_complex", ";".join(f), "-map", "[a]", "-ar", "48000", "-ac", "2", "-t", str(seconds), str(out)])
    return out, whos, seconds, timing


def cmd_live(a):
    sys.path.insert(0, str(Path(__file__).parent))
    import comfy  # 同目录的远程 ComfyUI 客户端
    comfy.requests.packages.urllib3.disable_warnings()
    ep = Ep(a.dir)
    if not ep.live:
        die("分镜.json 里没写 \"mode\": \"live\"，这是动态漫项目（用 frames / handoff）")
    bad = live_untranslated(ep)
    if bad:
        die("这些描述还是中文（模型会把中文当台词念出来），先补上英文 xxx_en：\n  " + "\n  ".join(bad))
    want = parse_ids(a.segments)
    w, h = (int(x) for x in str(a.size or ep.sb.get("live_size", "480x864")).split("x"))
    steps = int(a.steps or ep.sb.get("live_steps", 25))
    for g in ep.segments:
        if want and int(g["id"]) not in want:
            continue
        out = ep.root / "videos" / f"{ep.sid(g)}.mp4"
        if g.get("from"):
            print(f"  · 第 {g['id']} 场从别的场剪出来（from），不用拍")
            continue
        if out.exists() and not a.force:
            print(f"  · 第 {g['id']} 场已有 {out.name}（重拍加 --force）")
            continue
        pics, auds = segment_media(ep, g)
        lacking = [str(p) for p, _ in pics + auds if not p.exists()]
        if lacking:
            die("这些参考还没有（先跑 sheets / voices）：\n  " + "\n  ".join(lacking))
        drive, timing, seconds = None, None, float(g.get("seconds", 10))
        if g.get("drive", ep.sb.get("drive_audio", False)) and any(l for sh in g.get("shots", []) for l in sh.get("lines", [])):
            drive, whos, seconds, timing = drive_track(ep, g, comfy)
            # 参考声音就用这条配好的音轨，提示词里的 <Audio 1> 指它；两个人以上就声明成整条对白音轨，按 S1/S2 分人
            auds = [(drive, f"voice:{whos[0]}" if len(whos) == 1 else "drive:" + ",".join(whos))]
            if g.get("extend"):
                timing = [(a0 + TAIL_FRAMES / 24, a1 + TAIL_FRAMES / 24, w) for a0, a1, w in timing]
        prompt = segment_prompt(ep, g, pics, auds, timing)
        (ep.root / "build" / f"{ep.sid(g)}.prompt.json").write_text(prompt, "utf-8")
        refs = [comfy.upload(p) for p, _ in pics]
        voices = [comfy.upload(p) for p, _ in auds]
        seed = int(g.get("seed") or a.seed or 7)
        guides, overlap = segment_guides(ep, g, comfy)
        gw, gh = w, h
        if guides and (w % 32 or h % 32):
            # 钉画面（extend / keyframes）时宽高必须是 32 的倍数，否则 H3 报 shape invalid（720 不行，704 可以）；合成时照样放大到竖屏
            gw, gh = w // 32 * 32, h // 32 * 32
            print(f"  · 这一场钉了画面，尺寸改成 {gw}×{gh}（宽高要是 32 的倍数）")
        if drive and overlap:
            # 接续拍时开头有上一场的尾巴，配音往后挪同样的时长
            shifted = drive.with_name(drive.stem + ".shift.wav")
            run([FFMPEG, "-y", "-v", "error", "-i", str(drive), "-af", f"adelay={int(overlap * 1000)}|{int(overlap * 1000)}", str(shifted)])
            drive = shifted
        graph = comfy.h3_graph(prompt, refs, voices, gw, gh, seconds + overlap, seed, steps, f"manju/{ep.out_name()}-{ep.sid(g)}",
                               loras=ep.sb.get("live_loras"), guides=guides, drive_audio=comfy.upload(drive) if drive else None)
        t0 = time.time()
        # 提交过的任务记下来：网络断了重跑时接着等同一个任务，不在显卡上重复拍
        job = ep.root / "build" / f"{ep.sid(g)}.job.json"
        key = hashlib.sha1(json.dumps([prompt, gw, gh, steps, seed, seconds, guides, drive and hashlib.sha1(drive.read_bytes()).hexdigest()], ensure_ascii=False).encode()).hexdigest()
        prev = json.loads(job.read_text("utf-8")) if job.exists() else {}
        if prev.get("key") == key and not a.force and comfy.known(prev["pid"]):
            pid = prev["pid"]
            print(f"第 {g['id']} 场：接着等之前提交的任务 {pid}")
        else:  # 没提交过、内容改了，或者机器关过机、以前的任务已经没了：重新提交
            pid = comfy.submit(graph)
            job.write_text(json.dumps({"key": key, "pid": pid}), "utf-8")
        print(f"第 {g['id']} 场：{g.get('seconds', 10)} 秒 · {len(g.get('shots', []))} 个镜头 · {w}×{h} · {steps} 步 · 种子 {seed} · 参考图 {len(pics)} 张 · 参考声音 {len(auds)} 段")
        if overlap:
            raw = out.with_suffix(".raw.mp4")
            comfy.download(comfy.wait(pid), raw)
            # 开头那 0.92 秒是上一场的尾巴（钉住用的），去掉后正好接在上一场后面
            run([FFMPEG, "-y", "-v", "error", "-ss", f"{overlap:.4f}", "-i", str(raw), "-c:v", "libx264", "-crf", "16", "-preset", "medium",
                 "-c:a", "aac", "-b:a", "192k", str(out)])
        else:
            comfy.download(comfy.wait(pid), out)
        print(f"  ✓ {out.relative_to(ep.root)}（{int(time.time() - t0)} 秒）")


def cmd_music(a):
    """在 5090 上用 MiniMax Music 3 生成原创配乐（纯音乐），长度对齐成片，写进 bgm。编排写在分镜.json 的 music_prompt（英文）。"""
    sys.path.insert(0, str(Path(__file__).parent))
    import comfy
    comfy.requests.packages.urllib3.disable_warnings()
    ep = Ep(a.dir)
    caption = ep.sb.get("music_prompt")
    if not caption:
        die("分镜.json 里还没有 music_prompt（英文写：风格、速度、配器、按秒的起伏，结尾怎么收）")
    plan = live_plan(ep) if ep.live else plan_timeline(ep)
    total = round(sum(p["duration"] for p in plan), 2)
    lyrics = ep.sb.get("music_structure") or "[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Outro]"
    out = ep.root / "assets" / f"配乐-{a.seed or 1}.flac"
    pid = comfy.submit(comfy.music_graph(caption, lyrics, total + 1.0, int(a.seed or 1), f"manju/{ep.out_name()}-music"))
    print(f"生成配乐 {total + 1:.0f} 秒（种子 {a.seed or 1}）…")
    comfy.download(comfy.wait(pid), out)
    # 生成要好几分钟，期间分镜.json 可能被改过：重新读一遍再只改 bgm，别把别人的改动覆盖掉
    sb = json.loads(ep.sb_path.read_text("utf-8"))
    sb["bgm"] = str(out.relative_to(ep.root))
    ep.sb_path.write_text(json.dumps(sb, ensure_ascii=False, indent=2), "utf-8")
    print(f"✓ {out}（{probe(out)['duration']:.1f} 秒，已写进 bgm；不满意换 --seed 再来）")


def whisper_words(ep: Ep, video: Path, lang: str = "zh") -> list[tuple[float, float, str]]:
    """听一遍这一场，拿到逐词的起止时间（有 Silero VAD 就用，时间更准）。缓存在 build/。"""
    cache = ep.root / "build" / f"{video.stem}.words.json"
    if cache.exists() and cache.stat().st_mtime > video.stat().st_mtime:
        return [tuple(x) for x in json.loads(cache.read_text("utf-8"))]
    model, cli = whisper_model(), shutil.which("whisper-cli")
    if not (model and cli):
        return []
    wav = ep.root / "build" / f"{video.stem}.16k.wav"
    run([FFMPEG, "-y", "-v", "error", "-i", str(video), "-ar", "16000", "-ac", "1", str(wav)])
    prefix = ep.root / "build" / f"{video.stem}.w"
    vad = next(iter(sorted(Path(model).parent.glob("ggml-silero*.bin"))), None)
    args = [cli, "-m", model, "-l", lang, "-np", "-ml", "1", "-sow", "-oj", "-of", str(prefix), "-f", str(wav)]
    if vad:
        args += ["--vad", "-vm", str(vad)]
    run(args, timeout=600)
    j = json.loads(Path(f"{prefix}.json").read_text("utf-8"))
    words = [(x["offsets"]["from"] / 1000, x["offsets"]["to"] / 1000, x["text"].strip()) for x in j.get("transcription", []) if _plain(x["text"])]
    cache.write_text(json.dumps(words, ensure_ascii=False), "utf-8")
    return words


def _plain(t: str) -> str:
    return re.sub(r"[\s，。！？、…“”‘’,.!?\-—~]", "", t)


def align_lines(script: list[str], words: list[tuple[float, float, str]]) -> list[tuple[float, float] | None]:
    """把剧本台词对到 whisper 听到的逐词时间上：两边都拼成一串字，按最长公共片段把每个字对过去。"""
    heard, times = "", []
    for t0, t1, w in words:
        pw = _plain(w)
        for k, ch in enumerate(pw):
            heard += ch
            times.append(t0 + (t1 - t0) * k / max(1, len(pw)))
        if pw:
            times[-1] = t1
    joined, spans = "", []
    for line in script:
        a = len(joined)
        joined += _plain(line)
        spans.append((a, len(joined)))
    pos = {}
    for blk in difflib.SequenceMatcher(None, joined, heard, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            pos[blk.a + k] = blk.b + k
    out = []
    for a, b in spans:
        hits = [pos[i] for i in range(a, b) if i in pos]
        if len(hits) >= max(1, (b - a) // 3):
            out.append((times[min(hits)], times[max(hits)] + 0.25))
        else:
            out.append(None)
    return out


def live_plan(ep: Ep) -> list[dict]:
    """真人模式的时间线：每场一段视频，台词时间按 whisper 逐词听到的位置（对不上就按字数估）。"""
    plan, t = [], 0.0
    for g in ep.segments:
        errors = vo_errors(ep, g)
        if errors:
            die("\n  ".join(errors))
        for i, _ in enumerate(g.get("vo", []), 1):
            if not vo_file(ep, g, i).exists():
                die(f"{vo_file(ep, g, i).name} 不存在（先跑 vo）")
        v = ep.video(g)
        if not v:
            die(f"第 {g['id']} 场还没有视频（先跑 live）")
        dur = round(probe(v)["duration"] - float(g.get("video_in", 0)), 2)
        script = [l for sh in g.get("shots", []) for l in sh.get("lines", [])]
        timed = align_lines([spoken(l["text"]) for l in script], whisper_words(ep, v, live_lang(ep)[0]))
        skip = float(g.get("video_in", 0))  # 剪掉开头几秒时，台词时间跟着往前挪
        lines, cursor = [], 0.3
        for l, tm in zip(script, timed):
            a0, a1 = (tm[0] - skip, tm[1] - skip) if tm else (cursor, cursor + line_seconds(l["text"], live_lang(ep)[0]))
            if "at" in l:  # whisper 听岔了（比如把 귤이야 听成乱码）：台词里写 "at": 场内开口的秒数，手动钉字幕
                a0 = float(l["at"]) - skip
                a1 = a0 + float(l.get("seconds") or line_seconds(l["text"], live_lang(ep)[0]))
                tm = None
            a0 = max(0.0, a0)
            a1 = min(a1, dur)
            # whisper 有时把句子前后的静音也算进去：字幕最多挂 “字数×0.3+1 秒”，短句不拖着
            cap = 1.0 + 0.3 * len(re.sub(r"[\W_]", "", l.get("sub") or l["text"]))
            if tm and a1 - a0 > cap:
                a0 = max(a0, a1 - cap)  # 句尾更准（说完那一刻），从句尾往前算
            lines.append({"file": None, "text": l.get("sub") or l["text"], "spoken": spoken(l["text"]), "at": a0, "seconds": max(0.6, a1 - a0), "who": l["who"]})
            cursor = a1 + 0.3
        for a, b in zip(lines, lines[1:]):  # 前一句字幕在后一句出现前收掉
            a["seconds"] = max(0.5, min(a["seconds"], b["at"] - a["at"] - 0.2))
        vo = []
        for i, ln in enumerate(g.get("vo", []), 1):
            audio = vo_file(ep, g, i)
            at, seconds = float(ln["at"]), probe(audio)["duration"]
            if at + seconds > dur + 0.01:
                print(f"  ⚠ 第 {g['id']} 场 VO {i} 超出场长，合成会在场尾截断")
            seconds = min(seconds, max(0.0, dur - at))
            if seconds <= 0:
                continue
            text = ln.get("sub") or ln["text"]
            if ln["kind"] == "inner":
                text = f"（{text}）"
            # 字幕至少挂到看得完（字数×0.3+1 秒），声音照原长；不超出这一场
            readable = 1.0 + 0.3 * len(re.sub(r"[\W_]", "", text))
            vo.append({"file": audio, "text": text, "spoken": ln["text"], "who": ln["who"],
                       "at": at, "seconds": seconds, "sub_seconds": max(seconds, min(readable, dur - at)), "kind": ln["kind"]})
        for a, b in zip(vo, vo[1:]):  # 旁白字幕也在下一句出现前收掉，否则两句叠在同一个位置
            a["sub_seconds"] = max(a["seconds"], min(a["sub_seconds"], b["at"] - a["at"] - 0.2))
        plan.append({"shot": g, "start": round(t, 2), "duration": dur, "lines": lines, "vo": vo, "own_audio": v})
        t += dur
    return plan


# ---------------------------------------------------------------- 合成

def font_path() -> str:
    for f in (os.environ.get("MJ_FONT"), "/System/Library/Fonts/STHeiti Medium.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc",
              "/System/Library/Fonts/PingFang.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
        if f and Path(f).exists():
            return f
    die("找不到中文字体（用 MJ_FONT 指定一个 .ttf/.ttc）")


SUB_FONTS = {  # 分镜.json 的 "sub_font"：字幕字体。古装剧用宋体，现代剧用普惠体/黑体
    "songti": ("/System/Library/Fonts/Supplemental/Songti.ttc", 1),        # 宋体 粗
    "songti-black": ("/System/Library/Fonts/Supplemental/Songti.ttc", 0),  # 宋体 特粗
    "puhuiti": (str(Path.home() / "Library/Fonts/AlibabaPuHuiTi-3-65-Medium.ttf"), 0),
    "puhuiti-bold": (str(Path.home() / "Library/Fonts/AlibabaPuHuiTi-3-85-Bold.ttf"), 0),
    # 下面三款是 OFL 开源字体，可以商用：资源圆体（韩剧字幕组常用的圆体样子）、霞鹜文楷（温润、有书卷气）、得意黑（年轻、斜体）
    "rounded": (str(Path.home() / "Library/Fonts/ResourceHanRoundedCN-Bold.ttf"), 0),
    "rounded-medium": (str(Path.home() / "Library/Fonts/ResourceHanRoundedCN-Medium.ttf"), 0),
    "wenkai": (str(Path.home() / "Library/Fonts/LXGWWenKai-Medium.ttf"), 0),
    "smiley": (str(Path.home() / "Library/Fonts/SmileySans-Oblique.ttf"), 0),
    "heiti": ("/System/Library/Fonts/STHeiti Medium.ttc", 0),
}


def load_font(size: int, name: str | None = None):
    from PIL import ImageFont
    path, index = SUB_FONTS.get(name or "", (None, 0))
    if name and not path:  # 也可以直接写字体文件路径，"xxx.ttc#1" 指定第几个字重
        path, _, i = name.partition("#")
        index = int(i or 0)
    if not path or not Path(path).exists():
        path, index = font_path(), 0
    return ImageFont.truetype(path, size, index=index)


NO_LINE_START = set("，。、！？；：,.!?;:…）」』”’》〉】)]%")  # 这些字不放在一行开头


def wrap_text(text: str, font, width: int, draw) -> list[str]:
    """按宽度换行。两行时尽量对半分，不让标点、省略号或一两个字单独挂到下一行开头。"""
    if draw.textlength(text, font=font) <= width:
        return [text]
    lines, cur = [], ""
    for ch in text:  # 先按宽度贪心排，得到行数
        if draw.textlength(cur + ch, font=font) > width and cur:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    lines.append(cur)
    if len(lines) == 2:  # 两行：在中间附近找一个两行都放得下、下一行不以标点开头的断点，挑最均匀的
        best = None
        for i in range(2, len(text) - 1):
            a, b = text[:i], text[i:]
            if b[0] in NO_LINE_START or max(draw.textlength(a, font=font), draw.textlength(b, font=font)) > width:
                continue
            score = abs(draw.textlength(a, font=font) - draw.textlength(b, font=font)) - (width * 0.15 if a[-1] in NO_LINE_START else 0)
            if best is None or score < best[0]:
                best = (score, [a, b])
        if best:
            return best[1]
    for i in range(1, len(lines)):  # 多行：把下一行开头的标点挪回上一行
        while lines[i] and lines[i][0] in NO_LINE_START:
            lines[i - 1] += lines[i][0]
            lines[i] = lines[i][1:]
    return [l for l in lines if l]


def text_png(text: str, width: int, size: int, out: Path, stroke: int = 4, font_name: str | None = None, soft: bool = False,
             color: tuple[int, int, int] | None = None, style: str | None = None):
    """字幕渲染成透明 PNG（不依赖 ffmpeg 的 drawtext/libass）。自动按宽度换行。
    style="soft"（或 soft=True）：电视剧字幕的样子——暖白字、细描边、柔和的阴影，不用粗黑边。
    style="outline"：白字、细而实的深色描边，加一层很淡的小投影；不糊，亮画面、白屏上都清楚。"""
    from PIL import Image, ImageDraw, ImageFilter
    style = style or ("soft" if soft else "hard")
    font = load_font(size, font_name)
    draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    lines = wrap_text(text, font, width, draw)
    lh = int(size * 1.35)
    sw = max(2, size // 15)
    pad = stroke * 2 + 8 + (int(size * 0.5) if style in ("soft", "outline") else 0)
    w = int(max(draw.textlength(l, font=font) for l in lines)) + pad * 2
    h = lh * len(lines) + pad * 2
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if style in ("soft", "outline"):
        # soft：一层模糊的暗影托底（亮画面上也看得清）；outline：只是往右下错开一点的淡投影
        shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        for i, l in enumerate(lines):
            x = (w - sd.textlength(l, font=font)) / 2
            if style == "soft":
                sd.text((x, pad + i * lh + size * 0.06), l, font=font, fill=(0, 0, 0, 210), stroke_width=max(2, size // 12), stroke_fill=(0, 0, 0, 210))
            else:
                sd.text((x + size * 0.04, pad + i * lh + size * 0.06), l, font=font, fill=(0, 0, 0, 150), stroke_width=sw, stroke_fill=(0, 0, 0, 150))
        img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(size * (0.18 if style == "soft" else 0.06))))
    d = ImageDraw.Draw(img)
    for i, l in enumerate(lines):
        x = (w - d.textlength(l, font=font)) / 2
        if style == "soft":
            d.text((x, pad + i * lh), l, font=font, fill=(*(color or (250, 247, 240)), 255), stroke_width=max(1, size // 30), stroke_fill=(20, 18, 16, 200))
        elif style == "outline":
            d.text((x, pad + i * lh), l, font=font, fill=(*(color or (255, 255, 255)), 255), stroke_width=sw, stroke_fill=(22, 22, 22, 240))
        else:
            d.text((x, pad + i * lh), l, font=font, fill=(*(color or (255, 255, 255)), 255), stroke_width=stroke, stroke_fill=(0, 0, 0, 230))
    img.save(out)
    return img.size


GRADES = {
    "none": "",
    "cinema": "eq=contrast=1.06:saturation=0.9:gamma=0.98,colorbalance=rs=-0.03:bs=0.05:rh=0.04:bh=-0.03",
    "cool": "eq=contrast=1.04:saturation=0.86,colorbalance=rs=-0.05:bs=0.08",
    "warm": "eq=contrast=1.03:saturation=0.98,colorbalance=rs=0.05:bs=-0.05:rh=0.04",
    "anime": "eq=contrast=1.05:saturation=1.08",
    # 夜里闹鬼：亮度基本不动，只把暖黄灯色压成偏冷的月光色、降一点饱和，还像真实的夜晚但不温馨
    # （2026-10-25 用户从轻/中/重三档里选了“中”；压暗加暗角那版被嫌太黑、不真实）。场里写 "grade": "horror_night"
    "horror_night": "eq=saturation=0.82:contrast=1.07:gamma=0.97,"
                    "colorbalance=rs=-0.07:gs=-0.01:bs=0.10:rm=-0.05:bm=0.07:rh=0.02:bh=-0.01",
}


def plan_timeline(ep: Ep) -> list[dict]:
    """每镜的长度：分镜写的时长，台词放不下就自动拉长；台词在镜头里的位置。"""
    t = 0.0
    plan = []
    for s in ep.shots:
        lines = []
        cursor = float(s.get("line_start", 0.3))
        for i, ln in enumerate(s.get("lines", []), 1):
            name = f"{ep.sid(s)}_{i}.mp3"
            f = ep.root / "audio" / name
            # 自己放进来的配音（克隆声线等）按实际长度排；还没配音的按字数估
            sec = probe(f)["duration"] if f.exists() else line_seconds(ln["text"])
            lines.append({"file": ep.root / "audio" / name, "text": ln["text"], "at": cursor, "seconds": sec, "who": ln["who"]})
            cursor += sec + 0.25
        need = cursor + 0.35 if lines else 0
        dur = round(max(float(s.get("duration", 3)), need), 2)
        plan.append({"shot": s, "start": round(t, 2), "duration": dur, "lines": lines})
        t += dur
    return plan


def render_shot(ep: Ep, item: dict) -> Path:
    s, dur = item["shot"], item["duration"]
    out = ep.root / "build" / f"{ep.sid(s)}.mp4"
    W, H = ep.W, ep.H
    fit = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1"
    vid = ep.video(s)
    if vid:
        src_in = float(s.get("video_in", 0))
        speed = float(s.get("speed", 1))
        g = s.get("grade")  # 这一场单独调色：GRADES 里的名字，或直接写 ffmpeg 滤镜
        look = f"{GRADES.get(g, g)}," if g else ""
        if g and s.get("grade_at"):  # 只调这场里的一段（比如白天戏中间的夜里闪回）："grade_at": [起, 止]，按这场的秒数
            a0, a1 = s["grade_at"]
            look = "".join(f"{f}:enable='between(t,{a0},{a1})'," for f in GRADES.get(g, g).split(","))
        vf = f"trim=start={src_in},setpts=(PTS-STARTPTS)/{speed},{fit},{look}fps={FPS},tpad=stop_mode=clone:stop_duration={dur},trim=duration={dur},format=yuv420p"
        args = [FFMPEG, "-y", "-v", "error", "-i", str(vid), "-vf", vf, "-an"]
    else:
        img = ep.frame(s)
        if not img.exists():
            raise RuntimeError(f"第 {s['id']} 镜既没有视频也没有首帧")
        frames = int(dur * FPS)
        cam = s.get("camera", "")
        # 动态漫：静帧上加一点运镜
        if "拉" in cam:
            z, x, y = f"1.10-0.10*on/{frames}", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
        elif "左摇" in cam or "左移" in cam:
            z, x, y = "1.10", f"(iw-iw/zoom)*(1-on/{frames})", "ih/2-(ih/zoom/2)"
        elif "摇" in cam or "移" in cam:
            z, x, y = "1.10", f"(iw-iw/zoom)*on/{frames}", "ih/2-(ih/zoom/2)"
        elif "固定" in cam and "推" not in cam:
            z, x, y = f"1+0.03*on/{frames}", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
        else:
            z, x, y = f"1+0.08*on/{frames}", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
        vf = f"{fit},scale={W * 3 // 2}:{H * 3 // 2},zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={W}x{H}:fps={FPS},trim=duration={dur},format=yuv420p"
        args = [FFMPEG, "-y", "-v", "error", "-loop", "1", "-i", str(img), "-vf", vf, "-t", str(dur)]
    args += ["-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(out)]
    run(args)
    return out


def integrated_lufs(path: Path, audio_filter: str = "") -> float | None:
    af = f"{audio_filter},ebur128=framelog=quiet" if audio_filter else "ebur128=framelog=quiet"
    r = subprocess.run([FFMPEG, "-v", "info", "-i", str(path), "-vn", "-af", af, "-f", "null", "-"], capture_output=True, text=True)
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)
    return float(m[-1]) if m else None


def normalize_loudness(path: Path, target: float = -14.0):
    """两遍做响度：先量，再整体加减音量、限幅。短片里静音多，单遍 loudnorm 拉不到位。"""
    lufs = integrated_lufs(path)
    if lufs is None or abs(lufs - target) < 1.0:
        return
    tmp = path.with_name(path.stem + ".tmp.mp4")
    run([FFMPEG, "-y", "-v", "error", "-i", str(path), "-c:v", "copy", "-af", f"volume={target - lufs:.2f}dB,alimiter=limit=0.89:level=false",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(tmp)])
    tmp.replace(path)


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))  # 先整体取整到毫秒，不然 2.9996 秒会写成 “02,1000”
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def vo_duck(lines: list[dict]) -> str:
    """场内 VO 两侧 150 ms 渐变，重叠句也只压到 50%。"""
    envelope = "0"
    for ln in lines:
        start, end = ln["at"], ln["at"] + ln["seconds"]
        ramp = f"clip(min(t-({start - 0.15:.3f})\\,{end + 0.15:.3f}-t)/0.15\\,0\\,1)"
        envelope = f"max({envelope}\\,{ramp})"
    return f"1-0.5*({envelope})"


def translated_insert(ep: "Ep", ins: dict, png: Path) -> Path:
    """手机截图、帖子上的韩文，把中文翻译直接贴在旁边（像字幕组翻手机截图），不放到画面最下面。
    "tr": [{"y": 0.18, "x": 0.045, "text": "老板，民宿的事想跟您说一下。"}, …]，y、x 是原图高、宽的比例：
      默认在 y 这条线下面插一条和背景同色的窄条写中文，插多高就从下面最长的一段空白里抽掉多高，图的大小不变；
      "overlay": true 直接写在 (x, y)（那里本来就空着），"align": "center" 时 x 是中心；
      "chip": true 在 (x, y) 贴一个半透明深色小标签，x 是中心（照片类的贴图用）。"""
    from PIL import Image, ImageDraw
    import numpy as np
    out = ep.root / "build" / f"{ins['id']}.tr.png"
    size = float(ins.get("tr_size", 0.044))  # 和截图里的韩文差不多大，手机上看得清
    tag = hashlib.sha1(json.dumps([ins["tr"], size, png.stat().st_mtime, ep.sb.get("sub_font"), Path(__file__).stat().st_mtime]).encode()).hexdigest()[:10]
    stamp = out.with_suffix(".txt")
    if out.exists() and stamp.exists() and stamp.read_text() == tag:
        return out
    img = Image.open(png).convert("RGB")
    iw, ih = img.size
    fs = int(iw * size)
    font = load_font(fs, ep.sb.get("sub_font"))
    probe_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lh = int(fs * 1.4)

    def ink(rgb):  # 浅色界面用沉稳的蓝，深色界面用暖琥珀
        return (58, 104, 168) if sum(rgb) / 3 > 128 else (240, 200, 130)

    bands = sorted([t for t in ins["tr"] if not t.get("overlay") and not t.get("chip")], key=lambda t: t["y"])
    if bands:
        a = np.asarray(img).astype(int)
        pieces, texts, prev, extra = [], [], 0, 0
        for t in bands:
            y = int(t["y"] * ih)
            row = a[min(y, ih - 1)]
            bg = tuple(int(v) for v in np.median(row, axis=0))
            x = int(t.get("x", 0.045) * iw)
            lines = wrap_text(t["text"], font, iw - x - int(iw * 0.05), probe_draw)
            bh = lh * len(lines) + int(fs * 0.35)
            pieces.append(img.crop((0, prev, iw, y)))
            band = Image.new("RGB", (iw, bh), bg)
            texts.append((sum(pc.height for pc in pieces), x, lines, ink(bg)))
            pieces.append(band)
            prev, extra = y, extra + bh
        pieces.append(img.crop((0, prev, iw, ih)))
        tall = Image.new("RGB", (iw, ih + extra))
        yy = 0
        for pc in pieces:
            tall.paste(pc, (0, yy)); yy += pc.height
        d = ImageDraw.Draw(tall)
        for top, x, lines, color in texts:
            for i, l in enumerate(lines):
                d.text((x, top + int(fs * 0.1) + i * lh), l, font=font, fill=color)
        # 从最后一条中文下面找最长的一段平的空白（每行颜色几乎一样），抽掉多出来的高度
        b = np.asarray(tall).astype(int)
        flat = (b.std(axis=1).max(axis=1) < 2.5)
        last = texts[-1][0] + lh * len(texts[-1][2]) + fs
        best, start = (0, 0), None
        for y in range(last, tall.height + 1):
            if y < tall.height and flat[y]:
                start = y if start is None else start
            elif start is not None:
                best = max(best, (y - start, start)); start = None
        run, at = best
        if run >= extra + 10:
            cut = at + (run - extra) // 2
            img = Image.new("RGB", (iw, ih))
            img.paste(tall.crop((0, 0, iw, cut)), (0, 0))
            img.paste(tall.crop((0, cut + extra, iw, tall.height)), (0, cut))
        else:  # 没有够长的空白：整张缩回原高，两边补背景色
            small = tall.resize((int(iw * ih / tall.height), ih), Image.LANCZOS)
            img = Image.new("RGB", (iw, ih), tuple(int(v) for v in np.median(b[0], axis=0)))
            img.paste(small, ((iw - small.width) // 2, 0))
    d = ImageDraw.Draw(img, "RGBA")
    for t in ins["tr"]:
        if not (t.get("overlay") or t.get("chip")):
            continue
        cx, y = int(t.get("x", 0.5) * iw), int(t["y"] * ih)
        tw = probe_draw.textlength(t["text"], font=font)
        if t.get("chip"):
            px, py = int(fs * 0.55), int(fs * 0.3)
            x0 = int(cx - tw / 2) - px
            d.rounded_rectangle([x0, y, x0 + tw + px * 2, y + fs + py * 2 + int(fs * 0.2)], radius=int(fs * 0.45), fill=(12, 16, 24, 185))
            d.text((x0 + px, y + py), t["text"], font=font, fill=(250, 244, 230, 255))
        else:
            x = int(cx - tw / 2) if t.get("align") == "center" else cx
            color = ink(img.getpixel((max(0, min(iw - 1, cx)), max(0, min(ih - 1, y)))))
            d.text((x, y), t["text"], font=font, fill=(*color, 255))
    out.parent.mkdir(exist_ok=True)
    img.save(out)
    stamp.write_text(tag)
    return out


def card_png(kind: str, text: str, W: int, H: int, out: Path, font_name: str | None = None) -> tuple[int, int]:
    """现代剧的字卡，渲染成透明 PNG，返回贴图左上角的位置。
    time：右上角的小号时间牌（"02:00:45"、"01:59 · 第二夜"）；place：画面上部居中的地名卡；
    name：左下的人物字卡，写成"名字｜身份"或"名字｜身份｜韩文名"，背后渐隐的暗色托底，合成时从左边轻轻滑进来。time、place 是暖白字加一层柔和的暗影。"""
    from PIL import Image, ImageDraw, ImageFilter
    base = int(min(W, H) * 0.050)
    probe_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

    def render(size: tuple[int, int], draw_fn) -> Image.Image:
        shadow = Image.new("RGBA", size, (0, 0, 0, 0))
        draw_fn(ImageDraw.Draw(shadow), (0, 0, 0, 190))
        img = Image.alpha_composite(Image.new("RGBA", size, (0, 0, 0, 0)), shadow.filter(ImageFilter.GaussianBlur(base * 0.16)))
        draw_fn(ImageDraw.Draw(img), (250, 247, 240, 255))
        return img

    pad = int(base * 0.5)
    if kind == "name":
        # 人物字卡："名字｜身份" 或 "名字｜身份｜韩文名"。大名字 + 旁边小号琥珀色韩文名，下面一条往右渐隐的细线，再下面是身份；
        # 背后一层往右渐隐的暗色托底，亮画面上也清楚
        name, role, native = (text.split("｜") + ["", ""])[:3]
        f1, f2, f3 = load_font(int(base * 1.3), font_name), load_font(int(base * 0.7), font_name), load_font(int(base * 0.66), font_name)
        cream, amber, soft = (252, 248, 238, 255), (240, 196, 118, 255), (236, 231, 222, 255)
        n_w = probe_draw.textlength(name, font=f1)
        k_gap = int(base * 0.4)
        k_w = probe_draw.textlength(native, font=f3) if native else 0
        r_w = probe_draw.textlength(role, font=f2) if role else 0
        content_w = int(max(n_w + (k_gap + k_w if native else 0), r_w))
        l1, rule_gap, l2 = int(base * 1.3 * 1.25), int(base * 0.32), int(base * 0.66 * 1.5) if role else 0
        px, py = int(base * 0.9), int(base * 0.55)
        w, h = content_w + px * 2 + int(base * 1.2), l1 + rule_gap * 2 + l2 + py * 2
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        # 托底：一团左边深、往右渐隐的暗影，四周留足边再糊开，看不出方框
        m = int(base * 0.5)
        panel = Image.new("L", (w, h), 0)
        pd = ImageDraw.Draw(panel)
        for x in range(m, w - m):
            pd.line([(x, m), (x, h - m)], fill=int(130 * max(0.0, 1 - (x - m) / ((w - 2 * m) * 0.9)) ** 0.7))
        panel = panel.filter(ImageFilter.GaussianBlur(base * 0.45))
        img.paste((10, 14, 22, 255), (0, 0), panel)
        text_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        td = ImageDraw.Draw(text_layer)
        sw = max(1, base // 22)
        td.text((px, py), name, font=f1, fill=cream, stroke_width=sw, stroke_fill=(20, 20, 20, 200))
        if native:  # 韩文名和名字底边对齐
            td.text((px + n_w + k_gap, py + int(base * 1.3) - int(base * 0.66) - int(base * 0.02)), native, font=f3, fill=amber, stroke_width=sw, stroke_fill=(20, 20, 20, 180))
        ry = py + l1 + rule_gap
        rule_len = content_w + int(base * 0.6)
        for x in range(rule_len):  # 细线：琥珀色，往右渐隐
            td.line([(px + x, ry), (px + x, ry + max(2, base // 22))], fill=(*amber[:3], int(235 * (1 - x / rule_len) ** 1.3)))
        if role:
            td.text((px, ry + rule_gap), role, font=f2, fill=soft, stroke_width=sw, stroke_fill=(20, 20, 20, 190))
        shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        shadow.putalpha(text_layer.getchannel("A").point(lambda v: int(v * 0.55)))
        img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(base * 0.08)))
        img = Image.alpha_composite(img, text_layer)
        img.save(out)
        return int(W * 0.045), int(H * 0.60)
    size = int(base * 0.72) if kind == "time" else int(base * 1.15)
    font = load_font(size, font_name)
    w, h = int(probe_draw.textlength(text, font=font)) + pad * 2, int(size * 1.4) + pad * 2
    render((w, h), lambda d, fill: d.text((pad, pad), text, font=font, fill=fill)).save(out)
    if kind == "time":
        return W - w - int(W * 0.04), int(H * 0.055)
    return (W - w) // 2, int(H * 0.16)


def cmd_assemble(a):
    ep = Ep(a.dir)
    plan = live_plan(ep) if ep.live else plan_timeline(ep)
    # "end_black": 片尾黑场几秒（剧本写“黑场”收尾时用）：画面切黑，配乐和音效（比如结尾的心跳）可以在黑场里延续
    tail = float(ep.sb.get("end_black", 0))
    total = round(sum(p["duration"] for p in plan) + tail, 2)
    print(f"合成 {len(plan)} 镜，共 {total} 秒{f'（含片尾黑场 {tail} 秒）' if tail else ''}…")
    with cf.ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) // 3)) as pool:
        for item, _ in zip(plan, pool.map(lambda it: render_shot(ep, it), plan)):
            src = "视频" if ep.video(item["shot"]) else "首帧缓推"
            print(f"  ✓ 第 {item['shot']['id']} {'场' if ep.live else '镜'} {item['duration']} 秒（{src}）")
    lst = ep.root / "build" / "list.txt"
    lst.write_text("\n".join(f"file '{ep.root / 'build' / (ep.sid(p['shot']) + '.mp4')}'" for p in plan), "utf-8")
    visual = ep.root / "build" / "visual.mp4"
    run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(visual)])

    W, H = ep.W, ep.H
    inputs = ["-i", str(visual)]
    filters, idx = [], 1
    # 字幕：每句台词出现在说话的那段时间；镜头的画面字（caption）整镜显示
    subs, srt = [], []
    # 字幕在画面下方（竖屏 0.80，抖音底部的标题和按钮大约占最下面 15%）；"sub_y" 可改
    sub_y = int(H * float(ep.sb.get("sub_y", 0.80 if H > W else 0.86)))
    sub_size = int(min(W, H) * float(ep.sb.get("sub_size", 0.050)))
    for item in plan:
        for ln in item["lines"]:
            a0 = item["start"] + ln["at"]
            subs.append((ln["text"], a0, a0 + ln["seconds"] + 0.15, sub_y, None))
        def clear_of_lines(at: float, seconds: float) -> int:
            # 和对白重叠时挪到对白上方；多行对白预留其完整高度。
            overlap = [line for line in item["lines"] if line["at"] < at + seconds
                       and line["at"] + line["seconds"] + 0.15 > at]
            rows = max((math.ceil(load_font(sub_size, ep.sb.get("sub_font")).getlength(line["text"]) / (W * 0.84)) for line in overlap), default=0)
            return sub_y - int(sub_size * 1.35 * max(1, rows) + sub_size * 1.4) if overlap else sub_y
        for ln in item.get("vo", []):
            a0 = item["start"] + ln["at"]
            color = (232, 220, 190) if ln["kind"] == "narration" else (225, 225, 225)
            subs.append((ln["text"], a0, a0 + ln.get("sub_seconds", ln["seconds"]), clear_of_lines(ln["at"], ln["seconds"]), color))
        for ins in item["shot"].get("inserts", []):
            # 贴图上的外文（手机消息、帖子、名片）配一行中文翻译，跟贴图同进同出；写了 "tr" 的翻译直接贴在图上，这里不再出
            if ins.get("sub") and not ins.get("tr"):
                at, seconds = float(ins.get("at", 0)), float(ins.get("seconds", 2.5))
                subs.append((ins["sub"], item["start"] + at, item["start"] + at + seconds, clear_of_lines(at, seconds), None))
        cap = item["shot"].get("caption")
        if cap:
            # 真人模式一场很长，画面字（时间地点）只在开头停 3 秒
            end = item["start"] + (min(3.0, item["duration"]) if ep.live else item["duration"] - 0.1)
            subs.append((cap, item["start"] + 0.2, end, int(H * (0.18 if H > W else 0.12)), None))
    last = "0:v"
    grade = GRADES.get(ep.sb.get("grade", "cinema"), "")
    grain = int(ep.sb.get("grain", 6))
    look = ",".join(x for x in [grade, f"noise=c0s={grain}:c0f=t+u" if grain else "", "vignette=a=PI/6" if ep.sb.get("vignette", True) else ""] if x)
    if look:
        filters.append(f"[0:v]{look}[g]")
        last = "g"
    # 整屏贴图：场里写 "inserts": [{"id": ..., "at": 场内秒数, "seconds": 时长, "desc": ..., "text": ...}]，图在 inserts/<id>.png（mj inserts 出图）。
    # 手机屏幕、帖子、名片这类要写韩文、画界面的画面，视频模型画不准，后期整屏盖上去；声音照常走
    n = 0
    for item in plan:
        for ins in item["shot"].get("inserts", []):
            if ins.get("video"):
                # 盖一段别的视频（补拍的特写盖住穿帮的那几帧）："video": "补拍/videos/seg01.mp4", "video_in": 从第几秒取。
                # 只换画面、硬切、同样调色；声音照常走这一场的，台词可以从补拍画面底下先开口
                src = ep.root / ins["video"]
                if not src.exists():
                    print(f"  ⚠ 盖画面用的 {ins['video']} 还没有，这次先不盖")
                    continue
                t0, dur = item["start"] + float(ins.get("at", 0)), float(ins.get("seconds", 1.0))
                # 手机界面动画（phone_ui.py 出的）写 "look": false：屏幕不加颗粒、暗角、调色，才像真手机
                inputs += ["-ss", f"{float(ins.get('video_in', 0)):.3f}", "-t", f"{dur:.3f}", "-i", str(src)]
                vlook = look if ins.get("look", True) else ""
                filters.append(f"[{idx}:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,fps=24{',' + vlook if vlook else ''},"
                               f"setpts=PTS-STARTPTS+{t0:.3f}/TB[ins{n}]")
                filters.append(f"[{last}][ins{n}]overlay=0:0:eof_action=pass[iv{n}]")
                last, idx, n = f"iv{n}", idx + 1, n + 1
                continue
            png = ep.root / "inserts" / f"{ins['id']}.png"
            if not png.exists():
                print(f"  ⚠ 贴图 inserts/{png.name} 还没有（先跑 mj inserts），这次先不贴")
                continue
            if ins.get("tr"):  # 中文翻译直接贴在韩文旁边
                png = translated_insert(ep, ins, png)
            t0, dur = item["start"] + float(ins.get("at", 0)), float(ins.get("seconds", 2.5))
            inputs += ["-loop", "1", "-t", f"{dur:.2f}", "-i", str(png)]
            filters.append(f"[{idx}:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},format=rgba,"
                           f"fade=t=in:st=0:d=0.12:alpha=1,fade=t=out:st={max(0.0, dur - 0.12):.2f}:d=0.12:alpha=1,setpts=PTS-STARTPTS+{t0:.2f}/TB[ins{n}]")
            filters.append(f"[{last}][ins{n}]overlay=0:0:eof_action=pass[iv{n}]")
            last, idx, n = f"iv{n}", idx + 1, n + 1
    # 字卡：场里写 "cards": [{"kind": "time"|"place"|"name", "text": ..., "at": 场内秒数, "seconds": 时长}]
    n = 0
    for item in plan:
        for c in item["shot"].get("cards", []):
            png = ep.root / "build" / f"card{n:03d}.png"
            x, y = card_png(c.get("kind", "time"), c["text"], W, H, png, ep.sb.get("sub_font"))
            t0, dur = item["start"] + float(c.get("at", 0.2)), float(c.get("seconds", 2.5))
            inputs += ["-loop", "1", "-t", f"{dur:.2f}", "-i", str(png)]
            filters.append(f"[{idx}:v]format=rgba,fade=t=in:st=0:d=0.25:alpha=1,fade=t=out:st={max(0.0, dur - 0.3):.2f}:d=0.3:alpha=1,setpts=PTS-STARTPTS+{t0:.2f}/TB[cd{n}]")
            ox = f"'{x}-{int(W * 0.035)}*pow(max(0,1-(t-{t0:.2f})/0.5),2)'" if c.get("kind") == "name" else x  # 人物字卡从左边轻轻滑进来
            filters.append(f"[{last}][cd{n}]overlay=x={ox}:y={y}:eof_action=pass[cv{n}]")
            last, idx, n = f"cv{n}", idx + 1, n + 1
    if ep.sb.get("subtitles", True):
        for i, (text, t0, t1, y, color) in enumerate(subs):
            png = ep.root / "build" / f"sub{i:03d}.png"
            text_png(text, int(W * 0.84), sub_size, png,
                     font_name=ep.sb.get("sub_font"), style=ep.sb.get("sub_style", "soft"), color=color)
            inputs += ["-i", str(png)]
            filters.append(f"[{last}][{idx}:v]overlay=x=(W-w)/2:y={y}:enable='between(t,{t0:.2f},{t1:.2f})'[s{i}]")
            last, idx = f"s{i}", idx + 1
    # .srt 放对白和 VO，画面字不重复导出。
    for item in plan:
        for ln in item["lines"]:
            a0 = item["start"] + ln["at"]
            srt.append((a0, a0 + ln["seconds"] + 0.15, ln["text"]))
        for ln in item.get("vo", []):
            a0 = item["start"] + ln["at"]
            srt.append((a0, a0 + ln.get("sub_seconds", ln["seconds"]), ln["text"]))
        for ins in item["shot"].get("inserts", []):
            if ins.get("sub"):
                a0 = item["start"] + float(ins.get("at", 0))
                srt.append((a0, a0 + float(ins.get("seconds", 2.5)), ins["sub"]))
    # 人物出场介绍卡：场里写 "intros": [{"who": "shen", "at": 3.0, "tagline": "死了十三年的人", "side": "left"}]
    # at 是这一场里（剪好之后）的秒数；名字取角色的 name。卡片淡入 0.3 秒、停留、淡出 0.4 秒
    intro_n = 0
    for item in plan:
        for it in item["shot"].get("intros", []):
            sys.path.insert(0, str(Path(__file__).parent))
            from intro_card import render_intro
            name = it.get("name") or ep.sb.get("characters", {}).get(it.get("who"), {}).get("name", it.get("who", ""))
            png = ep.root / "build" / f"intro{intro_n:02d}.png"
            render_intro(name, it.get("tagline", ""), png, W, H, it.get("side", "left"))
            t0, dur = item["start"] + float(it.get("at", 0.5)), float(it.get("seconds", 2.2))
            inputs += ["-loop", "1", "-t", f"{dur:.2f}", "-i", str(png)]
            filters.append(f"[{idx}:v]format=rgba,fade=t=in:st=0:d=0.3:alpha=1,fade=t=out:st={dur - 0.4:.2f}:d=0.4:alpha=1,setpts=PTS-STARTPTS+{t0:.2f}/TB[ic{intro_n}]")
            filters.append(f"[{last}][ic{intro_n}]overlay=0:0:eof_action=pass[in{intro_n}]")
            last, idx, intro_n = f"in{intro_n}", idx + 1, intro_n + 1
    # AI 生成标识（平台和备案要求）
    if ep.sb.get("ai_label", True):
        png = ep.root / "build" / "ailabel.png"
        text_png(ep.sb.get("ai_label_text", "AI生成"), W, int(min(W, H) * 0.028), png, stroke=2)
        inputs += ["-i", str(png)]
        # ai_label_y：标识离顶边的比例。贴了手机截图的剧往下挪一点，别压在截图的返回箭头上
        filters.append(f"[{last}][{idx}:v]overlay=x=W*0.04:y=H*{float(ep.sb.get('ai_label_y', 0.035))}[lab]")
        last, idx = "lab", idx + 1
    if tail:  # 最后一句说完很快切黑，再补上黑场
        fade_in = "fade=t=in:st=0:d=0.5," if ep.sb.get("fade", True) else ""
        filters.append(f"[{last}]{fade_in}fade=t=out:st={max(0, total - tail - 0.25):.2f}:d=0.25,tpad=stop_mode=add:stop_duration={tail}:color=black[vout]")
    elif ep.sb.get("fade", True):
        filters.append(f"[{last}]fade=t=in:st=0:d=0.5,fade=t=out:st={max(0, total - 0.8):.2f}:d=0.8[vout]")
    else:
        filters.append(f"[{last}]null[vout]")

    # 声音：台词按时间点摆好；配乐循环垫底，台词时压低；音效按镜头时间点
    n_video_filters = len(filters)
    layers = []
    speech = []
    # VO 统一到本集对白片段的中位响度；没有可测对白时用 -18 LUFS。
    dialogue_levels = []
    if any(item.get("vo") for item in plan):
        for item in plan:
            if not item.get("own_audio") or not item["lines"] or not probe(item["own_audio"])["audio"]:
                continue
            skip = float(item["shot"].get("video_in", 0))
            select = "+".join(f"between(t\\,{skip + line['at']:.3f}\\,{skip + line['at'] + line['seconds']:.3f})" for line in item["lines"])
            level = integrated_lufs(item["own_audio"], f"aselect='{select}',asetpts=N/SR/TB,volume={item['shot'].get('volume', 1.0)}")
            if level is not None and level > -60:
                dialogue_levels.append(level)
    target = sorted(dialogue_levels)[len(dialogue_levels) // 2] if dialogue_levels else -18.0
    if any(item.get("vo") for item in plan):
        print(f"  · VO 统一响度：{target:.1f} LUFS（按本集对白）")
    for item in plan:  # noqa: B007
        if item.get("own_audio") and probe(item["own_audio"])["audio"]:
            # 真人模式：用这一场自己生成的声音（对白、口型、动作声都在里面）
            ms = int(item["start"] * 1000)
            src_in = float(item["shot"].get("video_in", 0))
            inputs += ["-i", str(item["own_audio"])]
            dip = vo_duck(item.get("vo", []))
            dn = ""
            if scene_denoise(ep, item["shot"]):  # 降噪：FFT 降噪压底噪，检测到持续啸叫再在那一带轻轻压 8 dB
                try:
                    hums = detect_hum(item["own_audio"], src_in, item["duration"])
                except Exception:
                    hums = []
                dn = "afftdn=nr=12:nf=-40," + "".join(f"equalizer=f={hz:.0f}:width_type=h:w=500:g=-8," for hz, _ in hums)
                print(f"  · 第 {item['shot']['id']} 场降噪" + (f"（并压 {'、'.join(f'{hz:.0f}' for hz, _ in hums)} Hz 一带）" if hums else ""))
            filters.append(f"[{idx}:a]atrim=start={src_in}:duration={item['duration']},asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo,{dn}"
                           f"volume={item['shot'].get('volume', 1.0)},volume='{dip}':eval=frame,afade=t=in:d=0.05,afade=t=out:st={max(0, item['duration'] - 0.08):.2f}:d=0.08,adelay={ms}|{ms},asetpts=N/SR/TB,apad=whole_dur={total}[o{idx}]")
            layers.append(f"o{idx}")
            idx += 1
            for ln in item["lines"]:
                speech.append((item["start"] + ln["at"], item["start"] + ln["at"] + ln["seconds"]))
        for ln in item["lines"]:
            if not ln["file"] or not ln["file"].exists():
                continue
            ms = int((item["start"] + ln["at"]) * 1000)
            inputs += ["-i", str(ln["file"])]
            filters.append(f"[{idx}:a]aresample=48000,aformat=channel_layouts=stereo,volume=1.0,adelay={ms}|{ms},asetpts=N/SR/TB,apad=whole_dur={total}[d{idx}]")
            layers.append(f"d{idx}")
            speech.append((item["start"] + ln["at"], item["start"] + ln["at"] + ln["seconds"]))
            idx += 1
        for ln in item.get("vo", []):
            level = integrated_lufs(ln["file"])
            gain = target - level if level is not None and level > -60 else 0.0
            ms = round((item["start"] + ln["at"]) * 1000)
            inputs += ["-i", str(ln["file"])]
            filters.append(f"[{idx}:a]atrim=duration={ln['seconds']},asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo,"
                           f"volume={gain:.2f}dB,alimiter=limit=0.89:level=false,afade=t=in:d=0.02,afade=t=out:st={max(0, ln['seconds'] - 0.04):.3f}:d=0.04,"
                           f"adelay={ms}|{ms},asetpts=N/SR/TB,apad=whole_dur={total}[vo{idx}]")
            layers.append(f"vo{idx}")
            speech.append((item["start"] + ln["at"], item["start"] + ln["at"] + ln["seconds"]))
            idx += 1
    for item in plan:
        for fx in item["shot"].get("sfx", []):
            f = (ep.root / fx["file"]) if not Path(fx["file"]).is_absolute() else Path(fx["file"])
            if not f.exists():
                print(f"  ⚠ 音效不存在：{f}")
                continue
            ms = int((item["start"] + float(fx.get("at", 0))) * 1000)
            inputs += ["-i", str(f)]
            filters.append(f"[{idx}:a]aresample=48000,aformat=channel_layouts=stereo,volume={fx.get('volume', 0.8)},adelay={ms}|{ms},asetpts=N/SR/TB,apad=whole_dur={total}[x{idx}]")
            layers.append(f"x{idx}")
            idx += 1
    amb = ep.sb.get("ambience")
    if amb:
        f = (ep.root / amb) if not Path(amb).is_absolute() else Path(amb)
        if f.exists():
            vol = float(ep.sb.get("ambience_volume", 0.5))
            dip = "*".join(f"(1-0.4*clip(min(t-{a0 - 0.3:.2f}\\,{a1 + 0.3:.2f}-t)/0.3\\,0\\,1))" for a0, a1 in speech) or "1"
            inputs += ["-stream_loop", "-1", "-i", str(f)]
            filters.append(f"[{idx}:a]atrim=duration={total},asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo,volume={vol},volume='{dip}':eval=frame,afade=t=in:d=0.8,afade=t=out:st={max(0, total - 1.5):.2f}:d=1.5[amb]")
            layers.append("amb")
            idx += 1
        else:
            print(f"  ⚠ 环境音文件不存在：{f}")
    bgm = ep.sb.get("bgm")
    if bgm:
        f = (ep.root / bgm) if not Path(bgm).is_absolute() else Path(bgm)
        if f.exists():
            vol = float(ep.sb.get("bgm_volume", 0.35))
            # 配乐骤停：从标了 music_out 的那一镜开始，配乐 0.06 秒内停掉到片尾
            stop = next((p["start"] for p in plan if p["shot"].get("music_out")), None)
            cut = f",volume='1-clip((t-{stop:.2f})/0.06\\,0\\,1)':eval=frame" if stop is not None else ""
            duck = "*".join(f"(1-0.65*clip(min(t-{a0 - 0.3:.2f}\\,{a1 + 0.3:.2f}-t)/0.3\\,0\\,1))" for a0, a1 in speech) or "1"
            inputs += ["-stream_loop", "-1", "-i", str(f)]
            filters.append(f"[{idx}:a]atrim=duration={total},asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo,volume={vol},volume='{duck}':eval=frame{cut},afade=t=in:d=1,afade=t=out:st={max(0, total - 2):.2f}:d=2[bgm]")
            layers.append("bgm")
            idx += 1
        else:
            print(f"  ⚠ 配乐文件不存在：{f}")
    # 分段配乐："bgm_cues": [{"file": "assets/配乐/x.mp3", "from": [2, 0.0], "to": [4, 3.0], "in": 0.0, "volume": 0.3,
    #   "fade_in": 0.8, "fade_out": 1.2, "cut": false}]。from/to 是 [场号, 这一场里（剪好之后）的秒数]，to 可以写 "end"；
    #   in 是从曲子第几秒开始用；cut=true 时在 to 那一刻 0.06 秒内骤停（比如尖叫那一下）。台词处一样自动压低
    starts = {p["shot"]["id"]: p["start"] for p in plan}
    def cue_time(ref):
        if ref == "end":
            return total
        return min(total, starts[ref[0]] + float(ref[1]))
    duck_all = "*".join(f"(1-0.65*clip(min(t-{a0 - 0.3:.2f}\\,{a1 + 0.3:.2f}-t)/0.3\\,0\\,1))" for a0, a1 in speech) or "1"
    for k, cue in enumerate(ep.sb.get("bgm_cues", [])):
        f = (ep.root / cue["file"]) if not Path(cue["file"]).is_absolute() else Path(cue["file"])
        if not f.exists():
            print(f"  ⚠ 配乐文件不存在：{f}")
            continue
        t0, t1 = cue_time(cue["from"]), cue_time(cue.get("to", "end"))
        dur = t1 - t0
        if dur <= 0.1:
            continue
        fi = float(cue.get("fade_in", 0.8))
        fo = 0.06 if cue.get("cut") else float(cue.get("fade_out", 1.2))
        ms = int(t0 * 1000)
        inputs += ["-stream_loop", "-1", "-ss", f"{float(cue.get('in', 0)):.3f}", "-i", str(f)]
        filters.append(f"[{idx}:a]atrim=duration={dur:.3f},asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo,"
                       f"volume={float(cue.get('volume', ep.sb.get('bgm_volume', 0.3)))},afade=t=in:d={fi},afade=t=out:st={max(0, dur - fo):.3f}:d={fo},"
                       f"adelay={ms}|{ms},asetpts=N/SR/TB,apad=whole_dur={total},volume='{duck_all}':eval=frame[cue{k}]")
        layers.append(f"cue{k}")
        idx += 1
    if layers:
        mix = "".join(f"[{l}]" for l in layers)
        filters.append(f"{mix}amix=inputs={len(layers)}:duration=longest:normalize=0,aresample=48000,atrim=duration={total}[aout]")
    out = ep.root / "out" / f"{ep.out_name()}.mp4"
    # 声音单独混成一条再和画面合：画面和声音放在同一个 filter_complex 里时，ffmpeg 8.1 的 amix 会多出一段（2026-10-27 实测
    # 第 2 场起声音整体晚了一场的长度），只混声音时是准的
    vfilters, afilters = filters[:n_video_filters], filters[n_video_filters:]
    mixwav = ep.root / "build" / "mix.wav"
    n_inputs = inputs.count("-i")
    if layers:
        aargs = [FFMPEG, "-y", "-v", "error", *inputs, "-filter_complex", ";".join(afilters), "-map", "[aout]", "-c:a", "pcm_f32le", str(mixwav)]
        (ep.root / "build" / "ffmpeg-audio.txt").write_text(" ".join(aargs), "utf-8")
        run(aargs)
        got = probe(mixwav)["duration"]
        if abs(got - total) > 0.1:
            raise SystemExit(f"✗ 混好的声音 {got:.2f} 秒，应是 {total} 秒，声音会错位，先别发")
    args = [FFMPEG, "-y", "-v", "error", *inputs, *(["-i", str(mixwav)] if layers else []), "-filter_complex", ";".join(vfilters), "-map", "[vout]"]
    args += ["-map", f"{n_inputs}:a", "-c:a", "aac", "-b:a", "192k"] if layers else ["-an"]
    args += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS), "-t", str(total), "-movflags", "+faststart", str(out)]
    (ep.root / "build" / "ffmpeg.txt").write_text(" ".join(args), "utf-8")
    run(args)
    if layers:
        normalize_loudness(out)
    lead = add_cover(ep, out)
    srt = [(t0 + lead, t1 + lead, tx) for t0, t1, tx in srt]
    srt_path = out.with_suffix(".srt")
    srt_path.write_text("\n".join(f"{i}\n{srt_time(t0)} --> {srt_time(t1)}\n{tx}\n" for i, (t0, t1, tx) in enumerate(sorted(srt), 1)), "utf-8")
    print(f"✓ 成片 {out}\n✓ 字幕 {srt_path}")


def add_cover(ep: Ep, out: Path) -> float:
    """"cover": "out/封面/封面-带标题.jpg" —— 封面放在片头停 cover_seconds 秒（默认 0.4，平台截首帧当封面时用得上，
    又短到不拖开头的钩子），同时写进 mp4 的封面图。返回片头加了多少秒（字幕要跟着往后挪）。"""
    cover = ep.sb.get("cover")
    if not cover:
        return 0.0
    img = ep.root / cover
    if not img.exists():
        print(f"  ⚠ 封面 {img} 不存在，跳过")
        return 0.0
    secs = float(ep.sb.get("cover_seconds", 0.4))
    W, H = ep.W, ep.H
    tmp = out.with_name(out.stem + ".cover.mp4")
    fit = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,fps={FPS},format=yuv420p"
    f = (f"[0:v]{fit},trim=duration={secs}[cv];[1:v]{fit}[mv];"
         f"anullsrc=r=48000:cl=stereo,atrim=duration={secs}[ca];[1:a]aresample=48000,aformat=channel_layouts=stereo[ma];"
         f"[cv][ca][mv][ma]concat=n=2:v=1:a=1[v][a]")
    run([FFMPEG, "-y", "-v", "error", "-loop", "1", "-t", f"{secs + 0.1}", "-i", str(img), "-i", str(out), "-i", str(img),
         "-filter_complex", f, "-map", "[v]", "-map", "[a]", "-map", "2:v", "-c:v:0", "libx264", "-preset", "medium", "-crf", "18",
         "-c:v:1", "mjpeg", "-disposition:v:1", "attached_pic", "-c:a", "aac", "-b:a", "192k", "-r", str(FPS), "-movflags", "+faststart", str(tmp)])
    tmp.replace(out)
    print(f"  ✓ 封面放在片头 {secs} 秒，并写进了视频文件的封面图")
    return secs


# ---------------------------------------------------------------- 环境音

def _pink_pair(a: int, b: int, shape: str, label: str, d: int) -> str:
    return (f"anoisesrc=c=pink:r=48000:d={d}:seed={a},{shape}[{label}L];"
            f"anoisesrc=c=pink:r=48000:d={d}:seed={b},{shape}[{label}R];[{label}L][{label}R]join=inputs=2:channel_layout=stereo[{label}]")


def ambience_graph(kind: str, d: int = 30) -> str:
    g = lambda a, b, depth=0.25: f"({1 - depth:.2f}+{depth}*sin(2*PI*{a}*t)*sin(2*PI*{b}*t+1.3))"
    mono = lambda l: f"pan=stereo|c0=c0|c1=c0[{l}]"
    if kind == "rain":
        return ";".join([_pink_pair(31, 37, f"highpass=f=400,lowpass=f=7000,volume='0.35*{g(0.09, 0.031, 0.15)}':eval=frame", "fall", d),
                         f"aevalsrc='if(gt(random(0),0.9993),(random(1)*2-1),0)':s=48000:d={d},bandpass=f=3200:width_type=o:w=1.5,volume=3,{mono('dl')}",
                         f"aevalsrc='if(gt(random(2),0.9994),(random(3)*2-1),0)':s=48000:d={d},bandpass=f=1800:width_type=o:w=1.5,volume=3,{mono('dr')}",
                         f"anoisesrc=c=brown:r=48000:d={d}:seed=41,lowpass=f=120,volume=0.3,{mono('low')}",
                         "[fall][dl][dr][low]amix=inputs=4:normalize=0,volume=0.6,alimiter=limit=0.9[out]"])
    if kind == "wind":
        return ";".join([_pink_pair(11, 29, f"lowpass=f=900,highpass=f=70,volume='0.5*{g(0.21, 0.083)}':eval=frame", "w", d),
                         f"anoisesrc=c=brown:r=48000:d={d}:seed=5,lowpass=f=140,volume=0.5,{mono('rumble')}",
                         f"anoisesrc=c=white:r=48000:d={d}:seed=7,bandpass=f=2300:width_type=h:w=120,volume='0.22*(0.5+0.5*sin(2*PI*0.17*t))':eval=frame,{mono('wh')}",
                         "[w][rumble][wh]amix=inputs=3:normalize=0,volume=0.55,alimiter=limit=0.9[out]"])
    if kind == "city":
        return ";".join([_pink_pair(3, 17, f"lowpass=f=400,highpass=f=60,volume='0.6*{g(0.11, 0.047, 0.2)}':eval=frame", "tr", d),
                         f"anoisesrc=c=brown:r=48000:d={d}:seed=9,lowpass=f=90,volume=0.5,{mono('deep')}",
                         f"anoisesrc=c=pink:r=48000:d={d}:seed=23,highpass=f=1500,lowpass=f=5000,volume='0.08*{g(0.3, 0.07, 0.4)}':eval=frame,{mono('hiss')}",
                         "[tr][deep][hiss]amix=inputs=3:normalize=0,volume=0.5,alimiter=limit=0.9[out]"])
    if kind == "night":
        return ";".join([f"aevalsrc='0.08*sin(2*PI*4200*t)*pow(max(0,sin(2*PI*5.2*t)),8)*(0.7+0.3*sin(2*PI*0.13*t))':s=48000:d={d},{mono('a')}",
                         f"aevalsrc='0.06*sin(2*PI*3600*t+0.5)*pow(max(0,sin(2*PI*3.7*t+2)),10)':s=48000:d={d},pan=stereo|c0=0.3*c0|c1=c0[b]",
                         _pink_pair(13, 19, f"lowpass=f=500,highpass=f=80,volume='0.14*{g(0.15, 0.05)}':eval=frame", "br", d),
                         "[a][b][br]amix=inputs=3:normalize=0,alimiter=limit=0.9[out]"])
    if kind == "room":
        return ";".join([_pink_pair(43, 47, "lowpass=f=600,highpass=f=100,volume=0.07", "air", d),
                         f"aevalsrc='0.012*sin(2*PI*50*t)+0.005*sin(2*PI*100*t)':s=48000:d={d},{mono('hum')}",
                         "[air][hum]amix=inputs=2:normalize=0,alimiter=limit=0.9[out]"])
    if kind == "sea":
        return ";".join([_pink_pair(61, 67, "lowpass=f=1800,highpass=f=90,volume='0.6*pow(max(0,sin(2*PI*0.125*t)),1.6)+0.1':eval=frame", "wave", d),
                         _pink_pair(73, 79, f"lowpass=f=700,highpass=f=70,volume='0.25*{g(0.19, 0.06)}':eval=frame", "sw", d),
                         "[wave][sw]amix=inputs=2:normalize=0,volume=0.6,alimiter=limit=0.9[out]"])
    die(f"没有这种环境音：{kind}（可选 rain / wind / city / night / room / sea）")


def cmd_ambience(a):
    ep = Ep(a.dir)
    out = ep.root / "assets" / f"环境音-{a.kind}.m4a"
    run([FFMPEG, "-y", "-v", "error", "-filter_complex", ambience_graph(a.kind), "-map", "[out]", "-c:a", "aac", "-b:a", "160k", str(out)], timeout=180)
    # 只改这一集自己的文件：ep.sb 里可能合并了 assets_from 的人物和场景，不能整个写回去
    sb = json.loads(ep.sb_path.read_text("utf-8"))
    sb["ambience"] = str(out.relative_to(ep.root))
    sb.setdefault("ambience_volume", 0.5)
    ep.sb_path.write_text(json.dumps(sb, ensure_ascii=False, indent=2), "utf-8")
    print(f"✓ {out}（已写进分镜.json 的 ambience，30 秒循环铺满全片，台词时让开一点）")


# ---------------------------------------------------------------- 自检

def whisper_model() -> str | None:
    for c in (os.environ.get("MJ_WHISPER_MODEL"), CFG.get("whisper_model"),
              *sorted(Path.home().glob(".cache/whisper/ggml-large*.bin")),
              *sorted(Path.home().glob(".cache/whisper/ggml-*.bin"))):
        if c and Path(c).exists():
            return str(c)
    return None


# ---- 自检：对口型检查表 / 啸叫 / 暗 / 台本外人声（真人模式）

HUM_BAND = (1000, 16000)   # 只看 1k–16k：人声和环境音的主体在下面，电流声/啸叫在这一段
HUM_DB = 7.5               # 持续的那一层频谱，比左右各 1200 Hz 的中位数高出这么多分贝才算（真实素材调出来的：坏场 8–10，干净场 ≤7）
HUM_PERSIST = 0.6          # 在 60% 以上的 0.5 秒片段里都在（取每个频点 40% 分位的电平，说话/动作声压不住它）
DARK_YAVG = 55             # 一场的平均亮度（0–255）低于这个算偏暗
STRAY_SKIP = ("字幕", "独播", "Television", "订阅", "谢谢观看", "优优")  # whisper 在静音上常见的幻觉


def detect_hum(video: Path, start: float = 0.0, dur: float | None = None) -> list[tuple[float, float]]:
    """找持续的窄带啸叫（电流声）。返回 [(中心频率 Hz, 高出多少 dB)]，最多 2 个。
    做法：按 0.5 秒切片算频谱 → 每个频点取“60% 的片段都超过”的电平（说话声是一阵一阵的，被它压下去；持续的啸叫压不下去）
    → 平滑到约 100 Hz → 比左右 ±1200 Hz 的中位数高出 HUM_DB 的就是。AI 生成的啸叫是一条约 500 Hz 宽的带，不是单根线，所以邻域要取宽。"""
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
    sr, win = 44100, 22050
    cmd = [FFMPEG, "-v", "error", "-ss", f"{start}"] + (["-t", f"{dur}"] if dur else []) + ["-i", str(video), "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    x = np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, dtype=np.int16).astype(np.float32) / 32768
    n = len(x) // win
    if n < 4:
        return []
    mag = np.abs(np.fft.rfft(x[: n * win].reshape(n, win) * np.hanning(win).astype(np.float32), axis=1)) / win   # 每格 2 Hz
    db = 20 * np.log10(mag + 1e-9)
    floor = np.percentile(db, (1 - HUM_PERSIST) * 100, axis=0)
    sm = np.convolve(floor, np.ones(51) / 51, mode="same")
    base = np.median(sliding_window_view(np.pad(sm, 600, mode="edge"), 1201), axis=1)
    excess = sm - base
    lo, hi = int(HUM_BAND[0] / 2), int(HUM_BAND[1] / 2)
    picked: list[tuple[float, float]] = []
    for k in np.argsort(-excess[lo:hi]) + lo:
        if excess[k] < HUM_DB or len(picked) >= 2:
            break
        if all(abs(k * 2 - p[0]) > 600 for p in picked):
            picked.append((float(round(k * 2, -2)), float(excess[k])))
    return picked


def scene_yavg(video: Path) -> list[tuple[float, float]]:
    """整条视频每秒抽 2 帧，量平均亮度 YAVG（0–255）。返回 [(时间, 亮度)]。"""
    p = subprocess.run([FFMPEG, "-v", "error", "-i", str(video), "-vf", "fps=2,signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-", "-an", "-f", "null", "-"],
                       capture_output=True, text=True)
    pts = [float(x) for x in re.findall(r"pts_time:([\d.]+)", p.stdout)]
    ys = [float(x) for x in re.findall(r"YAVG=([\d.]+)", p.stdout)]
    return list(zip(pts, ys))


def stray_speech(ep: Ep, item: dict) -> list[tuple[float, float, str]]:
    """这一场里台本以外的人声：whisper 听到、但不在任何一句台词的时间段里（重叠不超过 0.3 秒）的片段。
    时间是剪好之后这一场里的秒数。"""
    video = item["own_audio"]
    words = whisper_words(ep, video, live_lang(ep)[0])
    if not words:
        return []
    skip = float(item["shot"].get("video_in", 0))
    spans = [(ln["at"] - 0.15, ln["at"] + ln["seconds"] + 0.15) for ln in item["lines"]]
    script = [l["text"] for sh in item["shot"].get("shots", []) for l in sh.get("lines", [])]
    spans += [(t0 - skip, t1 - skip) for tm in align_lines(script, words) if tm for t0, t1 in [tm]]  # 没有被裁短过的原始听到的位置
    def inside(t0: float, t1: float) -> bool:  # 这个词落在台词时间段里的部分超过一半（或超过 0.3 秒）
        ov = max((min(t1, s1) - max(t0, s0) for s0, s1 in spans), default=0.0)
        return ov > 0.3 or ov >= (t1 - t0) * 0.5

    plain_script = [_plain(t) for t in script]

    def like_script(word: str) -> bool:  # 听错了的台词（“有脉”听成“有麦”）：有一半以上的字和某句台词对得上，就不算外人
        w = _plain(word)
        return any(difflib.SequenceMatcher(None, w, t).find_longest_match().size >= max(1, (len(w) + 1) // 2) for t in plain_script if w)

    groups: list[list] = []
    for w in words:                                            # 只收台词时间段之外的词；相隔不到 0.6 秒的连成一段
        t0, t1 = w[0] - skip, w[1] - skip
        if t1 <= 0 or t0 >= item["duration"] or inside(t0, t1) or like_script(w[2]):
            continue
        if groups and t0 - groups[-1][-1][1] < 0.6:
            groups[-1].append((t0, t1, w[2]))
        else:
            groups.append([(t0, t1, w[2])])
    out = []
    for g in groups:
        text = "".join(w[2] for w in g).strip()
        if len(_plain(text)) < 2 or any(s in text for s in STRAY_SKIP) or re.fullmatch(r"[\[\(（【].*[\]\)）】]", text):
            continue
        out.append((max(0.0, g[0][0]), g[-1][1], text))
    return out


def lipsync_sheet(ep: Ep, plan: list[dict], video: Path, lead: float, dest: Path) -> int:
    """对口型检查表：每句台词中间那一刻在成片里截一帧，拼成一张图，每格下面写“说话人｜台词开头”。返回格数。"""
    from PIL import Image, ImageDraw
    names = {k: v.get("name", k) for k, v in ep.sb.get("characters", {}).items()}
    tiles = []
    for item in plan:
        for ln in item["lines"]:
            t = lead + item["start"] + ln["at"] + ln["seconds"] / 2
            png = ep.root / "build" / f"lip{len(tiles):02d}.jpg"
            try:
                run([FFMPEG, "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", "-vf", "scale=240:-2", str(png)])
                tiles.append((Image.open(png).convert("RGB"), names.get(ln["who"], ln["who"]), re.sub(r"\s", "", ln["text"])[:10], item["shot"]["id"]))
            except Exception:
                continue
    if not tiles:
        return 0
    cols = min(6, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    w, h = 240, max(im.height for im, *_ in tiles)
    label_h = 72
    sheet = Image.new("RGB", (cols * w, rows * (h + label_h)), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    font = load_font(17)
    for i, (im, name, text, sid) in enumerate(tiles):
        x, y = (i % cols) * w, (i // cols) * (h + label_h)
        sheet.paste(im, (x, y))
        d.text((x + 6, y + h + 4), f"第{sid}场 说话人：{name}", fill=(255, 214, 120), font=font)
        d.text((x + 6, y + h + 28), f"｜{text}", fill=(235, 235, 235), font=font)
    sheet.save(dest, quality=90)
    return len(tiles)


def scene_denoise(ep: Ep, g: dict) -> bool:
    """"denoise": true —— 这一场（或写在最外层，整集）自带的声音先过一遍降噪，用来压 AI 视频里常见的电流声/啸叫。默认关。"""
    return bool(g.get("denoise", ep.sb.get("denoise", False)))


def live_checks(ep: Ep, plan: list[dict], out: Path, report: list[str], ok) -> None:
    """真人模式的自检：对口型检查表、啸叫、画面太暗、台本外的人声。每一项出错都只跳过，不拖垮整个 qc。"""
    cover = ep.sb.get("cover")
    lead = float(ep.sb.get("cover_seconds", 0.4)) if cover and (ep.root / cover).exists() else 0.0  # 和 add_cover 一致：片头多出来的秒数

    # 1. 对口型检查表：口型对没对上要人眼看，这里只负责把每句台词那一刻的画面摆出来
    try:
        dest = ep.root / "out" / "对口型检查.jpg"
        n = lipsync_sheet(ep, plan, out, lead, dest)
        if n:
            report.append(f"- 🖼 对口型检查表：`{dest}`（{n} 格）——逐格看：张嘴的是不是这个人")
    except Exception as e:
        report.append(f"- ℹ️ 对口型检查表没做成：{e}")

    # 2. 电流声/啸叫
    try:
        hums = 0
        for item in plan:
            if not item.get("own_audio") or not probe(item["own_audio"])["audio"]:
                continue
            found = detect_hum(item["own_audio"], float(item["shot"].get("video_in", 0)), item["duration"])
            for hz, _db in found[:1]:
                hums += 1
                if scene_denoise(ep, item["shot"]):
                    report.append(f"- ✅ 第 {item['shot']['id']} 场在 {hz:.0f} Hz 附近有持续啸叫，已开 denoise")
                else:
                    ok(False, "", f"第 {item['shot']['id']} 场在 {hz:.0f} Hz 附近有持续啸叫（像电流声），建议加 \"denoise\": true")
        if not hums:
            ok(True, "每场声音里没有持续的啸叫", "")
    except Exception as e:
        report.append(f"- ℹ️ 啸叫检查没做成：{e}")

    # 3. 画面太暗：成片里每秒 2 帧量平均亮度（0–255）
    try:
        ys = scene_yavg(out)
        if ys:
            dark = 0
            for item in plan:
                a, b = lead + item["start"], lead + item["start"] + item["duration"]
                vals = [y for t, y in ys if a <= t < b]
                if vals and sum(vals) / len(vals) < DARK_YAVG:
                    dark += 1
                    ok(False, "", f"第 {item['shot']['id']} 场偏暗（平均亮度 {sum(vals) / len(vals):.0f}）")
            avg = sum(y for _, y in ys) / len(ys)
            if dark:
                report.append(f"- ℹ️ 全片平均亮度 {avg:.0f}（0–255；{DARK_YAVG} 以下算偏暗）")
            else:
                ok(True, f"画面亮度够：全片平均 {avg:.0f}（0–255）", "")
    except Exception as e:
        report.append(f"- ℹ️ 亮度检查没做成：{e}")

    # 4. 台本外的人声（模型念出触发词、多出来的话）：whisper 听到、但不在任何一句台词时间里的
    try:
        strays = 0
        for item in plan:
            if not item.get("own_audio"):
                continue
            for t0, t1, text in stray_speech(ep, item):
                strays += 1
                ok(False, "", f"第 {item['shot']['id']} 场 {t0:.1f}–{t1:.1f} 秒有台本外的人声：“{text}”")
        if not strays and whisper_model() and shutil.which("whisper-cli"):
            ok(True, "没有台本外的人声", "")
    except Exception as e:
        report.append(f"- ℹ️ 台本外人声检查没做成：{e}")


def cmd_qc(a):
    ep = Ep(a.dir)
    out = ep.root / "out" / f"{ep.out_name()}.mp4"
    if not out.exists():
        die(f"还没有成片：{out}（先跑 assemble）")
    report, bad = [], 0

    def ok(cond: bool, good: str, warn: str):
        nonlocal bad
        report.append(f"- {'✅' if cond else '⚠️'} {good if cond else warn}")
        bad += 0 if cond else 1

    info = probe(out)
    plan = live_plan(ep) if ep.live else plan_timeline(ep)
    ok((info["width"], info["height"]) == (ep.W, ep.H), f"画幅 {info['width']}×{info['height']}（{ep.ratio}）", f"画幅是 {info['width']}×{info['height']}，应该是 {ep.W}×{ep.H}")
    ok(info["audio"], "有声音", "成片没有声音")
    ok(abs(info["duration"] - sum(p["duration"] for p in plan)) < 0.5, f"时长 {info['duration']:.1f} 秒", f"时长 {info['duration']:.1f} 秒，和分镜对不上")
    if not ep.live:
        stills = [p["shot"]["id"] for p in plan if not ep.video(p["shot"])]
        ok(not stills, "每镜都有视频", f"这些镜头用的是首帧缓推（还没放视频）：{stills}")
    lufs = integrated_lufs(out)
    if lufs is not None:
        ok(-16.5 <= lufs <= -11.5, f"响度 {lufs} LUFS（平台常用 -14）", f"响度 {lufs} LUFS，偏离 -14 太多")
    r = subprocess.run([FFMPEG, "-v", "info", "-i", str(out), "-vf", "blackdetect=d=0.4:pix_th=0.08", "-an", "-f", "null", "-"], capture_output=True, text=True)
    blacks = [(float(x), float(y)) for x, y in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", r.stderr)]
    blacks = [b for b in blacks if b[0] > 0.6 and b[1] < info["duration"] - 1.0]
    ok(not blacks, "片中没有意外的黑场", f"片中有黑场：{blacks}")
    # 台词识别：听一遍成片，和剧本台词比对
    model, cli = whisper_model(), shutil.which("whisper-cli")
    script_lines = [ln.get("spoken") or ln["text"] for p in plan for ln in p["lines"]]
    if model and cli and script_lines:
        wav = ep.root / "build" / "qc.wav"
        run([FFMPEG, "-y", "-v", "error", "-i", str(out), "-ar", "16000", "-ac", "1", str(wav)])
        heard = run([cli, "-m", model, "-l", live_lang(ep)[0] if ep.live else "zh", "-np", "-nt", "-f", str(wav)], timeout=600).stdout
        heard_plain = re.sub(r"[\s，。！？、…“”,.!?]", "", heard)
        misses = [t for t in script_lines if difflib.SequenceMatcher(None, re.sub(r"[\s，。！？、…“”,.!?]", "", t), heard_plain).find_longest_match().size < max(2, len(re.sub(r"[\s，。！？、…“”,.!?]", "", t)) * 0.5)]
        ok(not misses, f"台词 {len(script_lines)} 句都听得清", f"这些台词识别不出来（可能被配乐盖住或配音出错）：{misses}")
    if ep.live:
        live_checks(ep, plan, out, report, ok)
    # 缩略图
    sheet = ep.root / "out" / "缩略图.jpg"
    n = max(1, int(info["duration"] // 2))
    cols = 6 if ep.H > ep.W else 5
    rows_n = (n + cols - 1) // cols
    run([FFMPEG, "-y", "-v", "error", "-i", str(out), "-vf", f"fps=1/2,scale=240:-2,tile={cols}x{rows_n}", "-frames:v", "1", str(sheet)])
    report.append(f"- 🖼 缩略图：`{sheet}`")
    report += ["", "发布前人工过一遍：", "- [ ] 静音看一遍：只看画面能不能看懂", "- [ ] 闭眼听一遍：台词、配乐、音效有没有打架",
               "- [ ] 手机上看一遍：字幕会不会被抖音的按钮挡住", "- [ ] 标题、封面、话题；发布时勾选「AI 生成内容」",
               "- [ ] 用到的音乐、人物形象都有授权；正式上线的漫剧按广电要求备案"]
    text = f"# 自检报告 · {ep.out_name()}\n\n" + "\n".join(report) + "\n"
    (ep.root / "out" / "自检报告.md").write_text(text, "utf-8")
    print(text)
    print(f"{'✓ 全部通过' if not bad else f'⚠ {bad} 项要看一下'}")


# ---------------------------------------------------------------- 入口

def main():
    ap = argparse.ArgumentParser(description="AI 漫剧工作流")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init"); p.add_argument("dir"); p.add_argument("--title"); p.add_argument("--ratio", default="9:16", choices=list(CANVAS)); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_init)
    p = sub.add_parser("check"); p.add_argument("dir"); p.set_defaults(fn=cmd_check)
    p = sub.add_parser("sheets"); p.add_argument("dir"); p.add_argument("--only"); p.add_argument("--force", action="store_true"); p.add_argument("--jobs", type=int, default=3); p.set_defaults(fn=cmd_sheets)
    p = sub.add_parser("inserts"); p.add_argument("dir"); p.add_argument("--only"); p.add_argument("--force", action="store_true"); p.add_argument("--jobs", type=int, default=3); p.set_defaults(fn=cmd_inserts)
    p = sub.add_parser("frames"); p.add_argument("dir"); p.add_argument("--shots"); p.add_argument("--force", action="store_true"); p.add_argument("--jobs", type=int, default=3); p.set_defaults(fn=cmd_frames)
    p = sub.add_parser("handoff"); p.add_argument("dir"); p.set_defaults(fn=cmd_handoff)
    p = sub.add_parser("tts"); p.add_argument("dir"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_tts)
    p = sub.add_parser("voices"); p.add_argument("dir"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_voices)
    p = sub.add_parser("vo")
    p.add_argument("dir")
    p.add_argument("--segments")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_vo)
    p = sub.add_parser("redub"); p.add_argument("dir"); p.add_argument("--segments"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_redub)
    p = sub.add_parser("live"); p.add_argument("dir"); p.add_argument("--segments"); p.add_argument("--force", action="store_true"); p.add_argument("--size"); p.add_argument("--steps", type=int); p.add_argument("--seed", type=int); p.set_defaults(fn=cmd_live)
    p = sub.add_parser("music"); p.add_argument("dir"); p.add_argument("--seed", type=int); p.set_defaults(fn=cmd_music)
    p = sub.add_parser("ambience"); p.add_argument("dir"); p.add_argument("kind", choices=["rain", "wind", "city", "night", "room", "sea"]); p.set_defaults(fn=cmd_ambience)
    p = sub.add_parser("assemble"); p.add_argument("dir"); p.set_defaults(fn=cmd_assemble)
    p = sub.add_parser("qc"); p.add_argument("dir"); p.set_defaults(fn=cmd_qc)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
