"""高清修复后的皮肤去斑：只动皮肤区域里“中等大小、对比不强”的暗点，
眉毛、睫毛、鼻孔、嘴唇这类强边缘和最细的皮肤纹理都保留。
用法：skin_clean.py <输入.mp4> <输出.mp4> [--frame 秒 --png 输出.png] [--t1 12 --t2 30 --keep 0.25]"""
import argparse, subprocess, json
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--frame", type=float); ap.add_argument("--png")
ap.add_argument("--t1", type=float, default=12)    # 小于这个对比的中频细节压到 keep
ap.add_argument("--t2", type=float, default=30)    # 大于这个的完全保留（五官边缘）
ap.add_argument("--keep", type=float, default=0.25)
ap.add_argument("--sigma", type=float, default=6)  # 斑点尺度
ap.add_argument("--lift", type=float, default=1.0)  # 中间调提亮（gamma，1.0 不动；1.15 左右是轻提），黑白两端不动，窗外不会更爆
a = ap.parse_args()
LIFT = np.clip(255 * (np.arange(256) / 255) ** (1 / a.lift), 0, 255).astype(np.uint8)


def skin_mask(img):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycc, (60, 133, 77), (255, 178, 130))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)   # 把斑点本身也圈进来
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 6)[..., None]


def clean(img):
    f = img.astype(np.float32)
    g1 = cv2.GaussianBlur(f, (0, 0), 1.2)
    g6 = cv2.GaussianBlur(f, (0, 0), a.sigma)
    fine, mid = f - g1, g1 - g6
    lum = np.abs(mid @ np.float32([0.114, 0.587, 0.299]))[..., None]
    gain = a.keep + (1 - a.keep) * np.clip((lum - a.t1) / (a.t2 - a.t1), 0, 1)
    signed = mid @ np.float32([0.114, 0.587, 0.299])
    gain = np.where(signed[..., None] < 0, gain, 1.0)  # 只去比周围暗的点（斑），比周围亮的高光留着，脸才不发闷
    out = g6 + mid * gain + fine
    m = skin_mask(img)
    res = np.clip(f * (1 - m) + out * m, 0, 255).astype(np.uint8)
    return cv2.LUT(res, LIFT) if a.lift != 1.0 else res


def probe(p):
    s = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                            "stream=width,height,r_frame_rate", "-of", "json", p]))["streams"][0]
    return s["width"], s["height"], s["r_frame_rate"]


w, h, fps = probe(a.src)
if a.frame is not None:
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", str(a.frame), "-i", a.src, "-frames:v", "1",
                                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
    img = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
    cv2.imwrite(a.png, np.hstack([img, clean(img)]))
    raise SystemExit

dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", a.src, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", fps,
                        "-i", "-", "-i", a.src, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "14",
                        "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "copy", a.out], stdin=subprocess.PIPE)
n = 0
while (buf := dec.stdout.read(w * h * 3)) and len(buf) == w * h * 3:
    enc.stdin.write(clean(np.frombuffer(buf, np.uint8).reshape(h, w, 3)).tobytes()); n += 1
enc.stdin.close(); enc.wait(); dec.wait()
print(f"✓ {a.out}（{n} 帧）")
