"""Кэш для настройки устойчивости к ракурсу: кадры frames/<fid> искажаются одной перспективой на ролик
(как другая установка камеры), детектор D-FINE (onnx) прогоняется по искажённым кадрам → cache_<tag>/<fid>.json + frames_<tag>/<fid>/<k>.jpg.
  python runs/warp_cache.py A 0      # набор A (seed = fid)       — для настройки порогов
  python runs/warp_cache.py B 100    # набор B (seed = fid + 100) — для проверки"""
import json, pathlib, sys
import cv2, numpy as np
HERE = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(HERE))
from pipeline import Detector
tag, off = sys.argv[1], int(sys.argv[2]); FLIP = len(sys.argv) > 3 and sys.argv[3] == "flip"   # зеркальный вид: камера с другой стороны
C, F = HERE / f"cache_w{tag}", HERE / f"frames_w{tag}"; C.mkdir(exist_ok=True)
def warp_matrix(seed, w, h):
    r = np.random.default_rng(seed); s = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    d = (s + r.uniform(-0.14, 0.14, (4, 2)) * [w, h]).astype(np.float32); M = cv2.getPerspectiveTransform(s, d)
    R = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), r.uniform(-6, 6), r.uniform(1.0, 1.12)), [0, 0, 1]]); return R @ M
det = Detector()
for cf in sorted((HERE / "cache_adapt").glob("*.json")):
    fid = cf.stem; out = C / f"{fid}.json"
    if out.exists(): continue
    c = json.load(open(cf)); src = HERE / "frames" / fid; dst = F / fid; dst.mkdir(parents=True, exist_ok=True); M = None
    for k, f in enumerate(c["frames"]):
        p = src / f"{k:04d}.jpg"
        if not p.exists(): f["dets"] = []; continue
        im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR); h, w = im.shape[:2]
        if M is None:
            M = warp_matrix(int(fid) + off, w, h)
            if FLIP: M = np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]], float) @ M
        wi = cv2.warpPerspective(im, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        cv2.imencode(".jpg", wi, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(dst / f"{k:04d}.jpg"))
        f["dets"] = [[l, round(s, 3), [round(v, 1) for v in b]] for l, s, b in det(wi)]
        for key in ("H", "inl", "feat"): f.pop(key, None)
    json.dump(c, open(out, "w")); print(tag, fid, len(c["frames"]), flush=True)
print("done", tag)
