# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.31"]
# ///
"""
租的 5090（AutoDL 容器实例 Pro）开机、关机、查状态，开机后等 ComfyUI 能用。

  uv run autodl.py status     看机器是开着还是关着（不花钱）
  uv run autodl.py balance    看账户余额（不花钱）
  uv run autodl.py on         开机（有卡模式，按量计费），等到 ComfyUI 能用为止
  uv run autodl.py off        关机，等到真的关了为止

开发者口令放在 ~/.config/manju/autodl_token（只有自己能读），实例编号写在 ~/.config/manju/config.json
的 autodl_instance。脚本不打印口令，也不打印 ComfyUI 的地址。开机要花钱：先问过用户再开。卡难抢：拿到卡后活干完也别自动关机，用户说关再关。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

CFG_DIR = Path.home() / ".config" / "manju"
STOCK_WAIT = int(__import__("os").environ.get("MJ_STOCK_WAIT", "60"))  # 没有空卡时最多等几分钟
# 两套接口都管 pro- 开头的实例：autodl.art 是“应用实例 API”（我们这台 5090 用它），api.autodl.com 是“容器实例 Pro API”。哪个认这台机器就用哪个。
BASES = [
    ("https://www.autodl.art", "/api/v1/adl_dev/dev/instance/pro"),
    ("https://api.autodl.com", "/api/v1/dev/instance/pro"),
]


def die(msg: str):
    sys.exit(msg)


def config() -> dict:
    try:
        return json.loads((CFG_DIR / "config.json").read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def token() -> str:
    p = CFG_DIR / "autodl_token"
    if not p.exists():
        die(f"还没有开发者口令：在 AutoDL 控制台 → 账号 → 设置 → 开发者Token 复制口令，然后运行\n"
            f"  pbpaste | tr -d '[:space:]' > {p} && chmod 600 {p}")
    t = p.read_text("utf-8").strip()
    if not t:
        die(f"{p} 是空的")
    return t


def instance() -> str:
    i = config().get("autodl_instance")
    if not i:
        die("config.json 里没有 autodl_instance（实例编号，形如 pro-xxxxxxxxxxxx）")
    return i


def call(method: str, base: str, prefix: str, path: str, body: dict) -> dict:
    # 文档写 GET 也带 JSON，实际 GET 要把参数放在网址里，否则报“请求参数错误”
    kw = {"params": body} if method == "GET" else {"json": body}
    r = requests.request(method, base + prefix + path, headers={"Authorization": token()}, timeout=30, **kw)
    try:
        return r.json()
    except ValueError:
        return {"code": f"HTTP {r.status_code}", "msg": r.text[:200]}


_picked: tuple[str, str] | None = None


def api(method: str, path: str, body: dict, soft: bool = False) -> dict:
    """第一次用时两套接口都试一下，哪个认这台机器就一直用哪个。soft=True 时失败不退出，把出错的回复交回去。"""
    global _picked
    tries = [_picked] if _picked else BASES
    last, net_err = {}, None
    for base, prefix in tries:
        try:
            res = call(method, base, prefix, path, body)
        except requests.RequestException as e:
            net_err = {"code": type(e).__name__, "msg": str(e)[:200]}
            continue
        if res.get("code") == "Success":
            _picked = (base, prefix)
            return res
        last = res
    last = net_err or last  # 有一套没连上时，别把另一套的“没权限”当成结论
    if soft:
        return last
    die(f"AutoDL 接口没认这个请求：{last.get('code')} {last.get('msg', '')}")


NET_ERRORS = ("ConnectionError", "Timeout", "ReadTimeout", "ConnectTimeout", "SSLError", "ChunkedEncodingError")


def api_retry(method: str, path: str, body: dict, tries: int = 20) -> dict:
    """本机网络偶尔会断（连不上、域名解析失败）：隔 15 秒再试，最多 tries 次。"""
    for _ in range(tries):
        res = api(method, path, body, soft=True)
        if res.get("code") not in NET_ERRORS:
            return res
        time.sleep(15)
    die(f"连不上 AutoDL（{res.get('code')}），检查一下网络")


def status() -> str:
    res = api_retry("GET", "/status", {"instance_uuid": instance()})
    if res.get("code") != "Success":
        die(f"查不到机器状态：{res.get('code')} {res.get('msg', '')}")
    return str(res.get("data"))


def comfy_ready() -> bool:
    url = (config().get("comfy_url") or "").rstrip("/")
    if not url:
        return False
    try:
        return requests.get(url + "/system_stats", timeout=10, verify=False).ok
    except requests.RequestException:
        return False


def wait_for(cond, what: str, limit: int) -> float:
    t0 = time.time()
    while time.time() - t0 < limit:
        if cond():
            return time.time() - t0
        time.sleep(10)
    die(f"等了 {limit // 60} 分钟，{what} 还没好，去 AutoDL 控制台看一眼")


def cmd_status():
    print(f"机器：{status()}；ComfyUI：{'能用' if comfy_ready() else '连不上'}")


def cmd_balance():
    try:  # 余额只有 api.autodl.com 有
        r = requests.post("https://api.autodl.com/api/v1/dev/wallet/balance", json={}, headers={"Authorization": token()}, timeout=30).json()
    except (requests.RequestException, ValueError) as e:
        die(f"查余额失败：{type(e).__name__}")
    if r.get("code") != "Success":
        die(f"查余额失败：{r.get('code')} {r.get('msg', '')}")
    d = r.get("data") or {}
    # 接口返回的金额单位是千分之一元
    for k in ("assets", "voucher_balance", "accumulate"):
        if k in d:
            print(f"{k}: {d[k] / 1000:.2f} 元")


def cmd_on():
    """开机。机器只要不是“已关机”（用户在控制台手动开了、正在开、正在关），就不去碰它，等它变成 running 直接用。"""
    t0 = time.time()
    while True:
        s = status()
        if s == "shutting_down":  # 正在关：等关完再开
            time.sleep(10)
            continue
        if s != "shutdown":  # running / starting 等：已经有人在开或开着了，直接等它开好
            print(f"机器已经是 {s}，直接用", flush=True)
            break
        # 5090 被别人租光时会报“暂无库存”或“机器空闲GPU不足”：每分钟再试一次，最多等 STOCK_WAIT 分钟
        res = api_retry("POST", "/power_on", {"instance_uuid": instance(), "payload": "gpu"})
        if res.get("code") == "Success":
            print("已发开机指令，等机器开起来…", flush=True)
            break
        if status() not in ("shutdown", "shutting_down"):
            continue  # 同一时刻用户在控制台开了：下一轮会走“直接用”
        busy = any(k in str(res.get("msg")) for k in ("库存", "GPU不足", "空闲"))
        if not busy or time.time() - t0 > STOCK_WAIT * 60:
            die(f"开机失败：{res.get('code')} {res.get('msg', '')}")
        print(f"  暂时没有空的 5090，1 分钟后再试（已等 {(time.time() - t0) / 60:.0f} 分钟）", flush=True)
        time.sleep(60)
    t = wait_for(lambda: status() == "running", "机器", 600)
    print(f"机器开好了（{t:.0f} 秒），等 ComfyUI…", flush=True)
    t1 = time.time()
    while not comfy_ready():
        if time.time() - t1 > 900:
            # 卡难抢：不自动关机，告诉用户，由用户决定关不关
            die("开机 15 分钟了 ComfyUI 还连不上。机器还开着（每小时约 2.8 元），先告诉用户，由他决定关不关")
        time.sleep(10)
    print(f"ComfyUI 能用了（又等了 {time.time() - t1:.0f} 秒）")


def cmd_off():
    if status() == "running":
        res = api_retry("POST", "/power_off", {"instance_uuid": instance()})
        if res.get("code") != "Success":
            die(f"关机失败：{res.get('code')} {res.get('msg', '')}——马上去 AutoDL 控制台手动关机")
        print("已发关机指令…", flush=True)
    t = wait_for(lambda: status() == "shutdown", "关机", 600)  # shutting_down 还在关，等到 shutdown 才算关好
    print(f"已关机（状态：{status()}，{t:.0f} 秒）")


def main():
    import urllib3
    urllib3.disable_warnings()
    cmds = {"status": cmd_status, "balance": cmd_balance, "on": cmd_on, "off": cmd_off}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        die(__doc__)
    cmds[sys.argv[1]]()


if __name__ == "__main__":
    main()
