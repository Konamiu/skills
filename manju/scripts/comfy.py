# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.31"]
# ///
"""
远程 ComfyUI（租的 5090）客户端：上传参考图/参考声音 → 提交 MiniMax-H3 参考生视频（画面+声音+对白一次生成，
一条里可以有多个镜头）→ 等结果 → 下载 mp4。

  uv run comfy.py status                     看显卡和队列
  uv run comfy.py h3 --prompt prompt.json --ref a.png --ref b.png --audio v1.wav \
        --seconds 10 --size 480x864 --out out.mp4 [--steps 25] [--seed 1]

ComfyUI 地址用环境变量 MJ_COMFY_URL（默认是 manju 配置里的地址）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path

import requests

def _config_url() -> str:
    cfg = Path.home() / ".config" / "manju" / "config.json"
    try:
        return json.loads(cfg.read_text("utf-8")).get("comfy_url", "")
    except (OSError, ValueError):
        return ""


URL = (os.environ.get("MJ_COMFY_URL") or _config_url()).rstrip("/")
H3_UNET = os.environ.get("MJ_H3_UNET", "minimax/minimax_h3_ref2va_pruned_int8_convrot.safetensors")
H3_CLIP = os.environ.get("MJ_H3_CLIP", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
H3_VAE = "minimax_h3_video_vae_fp16.safetensors"
H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"


def base() -> str:
    if not URL:
        sys.exit("还没配置 ComfyUI 地址：写进 ~/.config/manju/config.json 的 comfy_url，或者设置环境变量 MJ_COMFY_URL")
    return URL


def get(path: str, **kw):
    """网络偶尔会断（SSL 断开、超时）：重试几次再放弃，服务器上的任务不受影响。"""
    for attempt in range(8):
        try:
            r = requests.get(base() + path, timeout=60, verify=False, **kw)
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:  # 下载视频到一半断线也重试
            if attempt == 7:
                raise
            print(f"  网络断了一下，{min(60, 5 * 2 ** attempt)} 秒后重试（{type(e).__name__}）", flush=True)
            time.sleep(min(60, 5 * 2 ** attempt))


def upload(path: Path) -> str:
    """传到 ComfyUI 的 input 目录（图片、声音都走这个接口），返回它在服务器上的文件名。"""
    with open(path, "rb") as f:
        r = requests.post(base() + "/upload/image", files={"image": (path.name, f)}, data={"overwrite": "true"}, timeout=300, verify=False)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def frames_for(seconds: float) -> int:
    """H3 的帧数要落在 17k+5 的网格上（24fps）。"""
    n = max(5, round(seconds * 24))
    return n + (5 - n % 17) % 17


def h3_graph(prompt: str, refs: list[str], audios: list[str], width: int, height: int, seconds: float, seed: int, steps: int, prefix: str,
             loras: list[dict] | None = None, guides: list[dict] | None = None, drive_audio: str | None = None) -> dict:
    """guides：把画面钉在某一帧（H3-B 接续 / H3-A 关键帧的做法，接在多角色参考后面）。
    {"video": 上传的短视频, "frame": 0} 把上一场的尾巴（画面+声音）钉在开头；{"image": 上传的图, "frame": n} 钉一张关键帧。"""
    g: dict[str, dict] = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": H3_UNET, "weight_dtype": "default"}},
        "attn": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["unet", 0], "attention": "comfy kitchen attention"}},
        "lowvram": {"class_type": "MiniMaxLowVRAMAttention", "inputs": {"model": ["attn", 0], "head_chunks": 4}},
        "ff": {"class_type": "MiniMaxChunkFeedForward", "inputs": {"model": ["lowvram", 0], "chunks": 4, "seq_threshold": 4096}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": H3_CLIP, "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VAE}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": H3_AUDIO_VAE}},
        "cond": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["clip", 0], "vae": ["vae", 0], "audio_vae": ["avae", 0], "prompt": prompt,
            "width": width, "height": height, "length": frames_for(seconds), "ref_image_size": "match"}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "sched": {"class_type": "BasicScheduler", "inputs": {"model": ["ff", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "guider": {"class_type": "BasicGuider", "inputs": {"model": ["ff", 0], "conditioning": ["cond", 0]}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0], "sigmas": ["sched", 0], "latent_image": ["cond", 1]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "adecode": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]}},
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "fps": 24.0, "audio": ["adecode", 0]}},
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": prefix, "format": "mp4", "format.codec": "h264"}},
    }
    # LoRA（真实感、电影质感等）串在模型和注意力补丁之间
    prev = "unet"
    for i, lo in enumerate(loras or []):
        g[f"lora{i}"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": [prev, 0], "lora_name": lo["name"], "strength_model": float(lo.get("strength", 0.7))}}
        prev = f"lora{i}"
    g["attn"]["inputs"]["model"] = [prev, 0]
    for i, name in enumerate(refs):
        g[f"img{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g["cond"]["inputs"][f"ref_images.ref_image_{i}"] = [f"img{i}", 0]
    for i, name in enumerate(audios):
        g[f"aud{i}"] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
        g["cond"]["inputs"][f"ref_audios.ref_audio_{i}"] = [f"aud{i}", 0]
    pos = ["cond", 0]
    for i, gd in enumerate(guides or []):
        inp = {"positive": pos, "latent": ["cond", 1], "frame_idx": int(gd.get("frame", 0)), "vae": ["vae", 0]}
        if gd.get("video"):
            g[f"gv{i}"] = {"class_type": "LoadVideo", "inputs": {"file": gd["video"]}}
            g[f"gc{i}"] = {"class_type": "GetVideoComponents", "inputs": {"video": [f"gv{i}", 0]}}
            g[f"gs{i}"] = {"class_type": "ImageScale", "inputs": {"image": [f"gc{i}", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "center"}}
            inp |= {"image": [f"gs{i}", 0], "audio_vae": ["avae", 0], "audio": [f"gc{i}", 1]}
        else:
            g[f"gi{i}"] = {"class_type": "LoadImage", "inputs": {"image": gd["image"]}}
            g[f"gs{i}"] = {"class_type": "ImageScale", "inputs": {"image": [f"gi{i}", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "center"}}
            inp["image"] = [f"gs{i}", 0]
        g[f"guide{i}"] = {"class_type": "MiniMaxH3AddGuide", "inputs": inp}
        pos = [f"guide{i}", 0]
    g["guider"]["inputs"]["conditioning"] = pos
    if drive_audio:
        # 先配音后对口型：把配好的台词音轨锁进 H3 的音频潜空间（去噪强度 0），画面照着它对嘴，成片里的声音就是这条音轨
        g["drive_a"] = {"class_type": "LoadAudio", "inputs": {"audio": drive_audio}}
        g["drive"] = {"class_type": "VRGDG_MiniMaxH3AudioDrive", "inputs": {"av_latent": ["cond", 1], "source_audio": ["drive_a", 0], "audio_vae": ["avae", 0]}}
        g["sample"]["inputs"]["latent_image"] = ["drive", 0]
        g["video"]["inputs"]["audio"] = ["drive", 1]
    return g


_OPTIONS: dict | None = None
MODEL_EXT = (".safetensors", ".gguf", ".pth", ".bin", ".pt", ".sft", ".ckpt")


def node_options() -> dict:
    global _OPTIONS
    if _OPTIONS is None:
        _OPTIONS = get("/object_info").json()
    return _OPTIONS


def _choices(info: dict, node: str, name: str) -> list[str] | None:
    spec = info.get(node, {}).get("input", {})
    v = (spec.get("required", {}) | spec.get("optional", {})).get(name)
    if not v:
        return None
    if isinstance(v[0], list):
        return v[0]
    if v[0] == "COMBO" and len(v) > 1:
        return v[1].get("options")
    return None


def fix_models(graph: dict) -> list[str]:
    """工作流里写的模型文件，在这台服务器上换了目录（比如多了 minimax/ 前缀）时，按文件名找到并改过来。"""
    info = node_options()
    changes, missing = [], []
    for nid, node in graph.items():
        ct = node.get("class_type")
        for k, v in list(node.get("inputs", {}).items()):
            if not (isinstance(v, str) and v.endswith(MODEL_EXT)):
                continue
            opts = _choices(info, ct, k)
            if opts is None or v in opts:
                continue
            name = v.replace("\\", "/").split("/")[-1]
            hit = next((o for o in opts if o.replace("\\", "/").split("/")[-1] == name), None)
            if hit:
                node["inputs"][k] = hit
                changes.append(f"{v} → {hit}")
            else:
                missing.append(f"{ct}.{k} = {v}")
    if missing:
        sys.exit("这台服务器上没有这些模型：\n  " + "\n  ".join(missing))
    return changes


def tts_design_graph(text: str, instruct: str, language: str, prefix: str) -> dict:
    """Qwen3-TTS 声音设计：用一句话描述想要的声音（年龄、性别、音色、情绪），按 text 念出来。支持中英日韩等 10 种语言。"""
    return {
        "m": {"class_type": "TDQwen3TTSModelLoader", "inputs": {"model_path": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "precision": "bf16", "device": "cuda",
                                                            "attn_implementation": "sdpa", "auto_download": True, "download_source": "ModelScope"}},
        "t": {"class_type": "TDQwen3TTSVoiceDesign", "inputs": {"model": ["m", 0], "text": text, "instruct": instruct, "language": language}},
        "s": {"class_type": "SaveAudio", "inputs": {"audio": ["t", 0], "filename_prefix": prefix}},
    }


def tts_clone_graph(text: str, ref: str, ref_text: str, language: str, prefix: str) -> dict:
    """Qwen3-TTS 克隆：ref 是上传后的参考音文件名，ref_text 是参考音对应的原文。
    可用角色 voice_sample / assets/声音 的参考声线生成内心独白，也可补录对白。"""
    return {
        "m": {"class_type": "TDQwen3TTSModelLoader", "inputs": {"model_path": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "precision": "bf16", "device": "cuda",
                                                            "attn_implementation": "sdpa", "auto_download": True, "download_source": "ModelScope"}},
        "r": {"class_type": "LoadAudio", "inputs": {"audio": ref}},
        "t": {"class_type": "TDQwen3TTSVoiceClone", "inputs": {"model": ["m", 0], "text": text, "ref_audio": ["r", 0], "language": language, "ref_text": ref_text}},
        "s": {"class_type": "SaveAudio", "inputs": {"audio": ["t", 0], "filename_prefix": prefix}},
    }


def music_graph(caption: str, lyrics: str, seconds: float, seed: int, prefix: str) -> dict:
    """MiniMax Music 3：caption 写风格、配器、按秒的编排；lyrics 只放 [Intro] [Verse] 这类结构标签就是纯音乐。"""
    return {
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "minimax_music3_text_encoder_bf16.safetensors", "type": "minimax", "device": "default"}},
        "enc": {"class_type": "MiniMaxMusic3TextEncode", "inputs": {"clip": ["clip", 0], "caption": caption, "lyrics": lyrics, "seed": seed,
                                                              "max_duration": float(seconds), "cfg_scale": 1.5, "top_k": 50}},
        "zero": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["enc", 0]}},
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_music3/minimax_music3_dit_fp32.safetensors", "weight_dtype": "default"}},
        "lat": {"class_type": "EmptyMiniMaxMusic3LatentAudio", "inputs": {"seconds": ["enc", 1], "batch_size": 1}},
        "ks": {"class_type": "KSampler", "inputs": {"model": ["unet", 0], "positive": ["enc", 0], "negative": ["zero", 0], "latent_image": ["lat", 0],
                                                    "seed": seed, "steps": 30, "cfg": 1.7, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_music3_dav.safetensors"}},
        "dec": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveAudio", "inputs": {"audio": ["dec", 0], "filename_prefix": prefix}},
    }


def submit(graph: dict, front: bool = False) -> str:
    """front=True：插到队伍最前面（配音这种几十秒的小活，不用等前面排着的长视频）。"""
    for c in fix_models(graph):
        print(f"  模型路径：{c}")
    r = requests.post(base() + "/prompt", json={"prompt": graph, "client_id": uuid.uuid4().hex, **({"front": True} if front else {})}, timeout=120, verify=False)
    if r.status_code != 200:
        sys.exit(f"提交失败（{r.status_code}）：{r.text[:2000]}")
    return r.json()["prompt_id"]


def known(pid: str) -> bool:
    """这个任务 ComfyUI 还认不认：做完了、正在跑或在排队都算；机器关过机重开后，以前的任务就都没了。"""
    if pid in get(f"/history/{pid}").json():
        return True
    q = get("/queue").json()
    return any(item[1] == pid for item in q.get("queue_running", []) + q.get("queue_pending", []))


def wait(pid: str, limit: int = 3600) -> dict:
    start = time.time()
    last = ""
    while time.time() - start < limit:
        # 带上时间戳、不许缓存：2026-10-25 第 3 集第 7 场明明 08:14 就生成完了，这里却一直没看到结果，直到一小时超时
        h = get(f"/history/{pid}", params={"_": int(time.time())}, headers={"Cache-Control": "no-cache"}).json()
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
                sys.exit(f"ComfyUI 出错：{json.dumps(msgs, ensure_ascii=False)[:1500]}")
            if h[pid].get("outputs"):
                return h[pid]
        q = get("/queue").json()
        pos = next((i for i, item in enumerate(q.get("queue_pending", [])) if item[1] == pid), None)
        msg = "正在生成" if any(item[1] == pid for item in q.get("queue_running", [])) else (f"排队中（前面 {pos + 1} 个）" if pos is not None else "等待")
        if msg != last:
            print(f"  {msg}…（{int(time.time() - start)} 秒）", flush=True)
            last = msg
        # 轮询太勤会拖慢采样（社区在 5090 上测到每条多卡 18–43 秒）：头一分钟 5 秒一问，之后 20 秒一问
        time.sleep(5 if time.time() - start < 60 else 20)
    sys.exit("等太久了，去 ComfyUI 页面看看队列（任务要是其实已经做完，重跑同一条 mj live 会直接把结果取回来）")


def download(hist: dict, out: Path) -> Path:
    """下载视频或音频原始字节；SaveAudio 默认是 FLAC，改扩展名不会转码。"""
    exts = (".mp4", ".webm", ".mkv", ".mov") if out.suffix.lower() in (".mp4", ".mov", ".webm") else (".flac", ".wav", ".mp3", ".opus", ".ogg")
    files = [f for node in hist["outputs"].values() for key in ("videos", "images", "gifs", "audio")
             for f in node.get(key, []) if f.get("filename", "").endswith(exts)]
    # LoadVideo（接续拍钉的上一场尾巴）也会出一条 type=input 的预览，不能拿错：先要真正保存的 output
    files.sort(key=lambda f: f.get("type", "output") != "output")
    if files:
        f = files[0]
        r = get("/view", params={"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")})
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(r.content)
        return out
    sys.exit(f"没找到{'视频' if out.suffix.lower() in ('.mp4', '.mov', '.webm') else '音频'}输出：{json.dumps(hist['outputs'], ensure_ascii=False)[:800]}")


def cmd_run(a):
    """跑任意 API 格式的工作流（比如 onething-comfy-api/workflows/*.api.json）：--set 节点.输入=值，--file 节点.输入=本地文件（自动上传）。"""
    g = json.loads(Path(a.workflow).read_text("utf-8"))
    g = g.get("prompt", g) if isinstance(g.get("prompt"), dict) else g
    for item in a.file:
        key, path = item.split("=", 1)
        nid, inp = key.split(".", 1)
        g[nid]["inputs"][inp] = upload(Path(path))
    for item in a.set:
        key, val = item.split("=", 1)
        nid, inp = key.split(".", 1)
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            pass
        g[nid]["inputs"][inp] = val
    t0 = time.time()
    pid = submit(g)
    print(f"已提交 {Path(a.workflow).name}，任务 {pid}")
    out = download(wait(pid), Path(a.out))
    print(f"✓ {out}（用时 {int(time.time() - t0)} 秒）")


def cmd_status(_a):
    s = get("/system_stats").json()
    for d in s.get("devices", []):
        print(f"{d['name']}  显存空闲 {d['vram_free'] / 2**30:.1f} / {d['vram_total'] / 2**30:.1f} GB")
    q = get("/queue").json()
    print(f"队列：正在跑 {len(q['queue_running'])} 个，排队 {len(q['queue_pending'])} 个")


def cmd_h3(a):
    prompt = Path(a.prompt).read_text("utf-8") if Path(a.prompt).exists() else a.prompt
    refs = [upload(Path(p)) for p in a.ref]
    audios = [upload(Path(p)) for p in a.audio]
    w, h = (int(x) for x in a.size.split("x"))
    seed = a.seed if a.seed is not None else random.randint(1, 2**31)
    graph = h3_graph(prompt, refs, audios, w, h, a.seconds, seed, a.steps, f"manju/{Path(a.out).stem}")
    t0 = time.time()
    pid = submit(graph)
    print(f"已提交 H3（{a.seconds} 秒，{w}×{h}，{a.steps} 步，种子 {seed}），任务 {pid}")
    hist = wait(pid)
    out = download(hist, Path(a.out))
    print(f"✓ {out}（用时 {int(time.time() - t0)} 秒）")


def main():
    requests.packages.urllib3.disable_warnings()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("run", help="跑任意 API 格式工作流")
    p.add_argument("workflow")
    p.add_argument("--set", action="append", default=[], help="节点ID.输入名=值（值可以是 JSON）")
    p.add_argument("--file", action="append", default=[], help="节点ID.输入名=本地文件，会先上传")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("h3")
    p.add_argument("--prompt", required=True, help="提示词文件或文本")
    p.add_argument("--ref", action="append", default=[], help="参考图（按顺序对应 <Picture 1>…）")
    p.add_argument("--audio", action="append", default=[], help="参考声音（按顺序对应 <Audio 1>…）")
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--size", default="480x864")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--seed", type=int)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_h3)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
