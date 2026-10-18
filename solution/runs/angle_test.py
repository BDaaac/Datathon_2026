"""Тест нового угла камеры на реальных видео: каждое видео перспективно искажается (другой угол/положение камеры),
затем идёт через тот же сервисный конвейер (Analyzer). Сравниваем с исходными видео по оценщику жюри."""
import glob, json, os, re, sys, time, pathlib
import cv2, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from inference import Analyzer
from evaluate import load_labels, score
V = r"C:/Users/Димаш/Downloads/open (1)/open/videos"
TMP = pathlib.Path(os.environ["TMPDIR_ANGLE"]); TMP.mkdir(parents=True, exist_ok=True)
OUT = pathlib.Path(__file__).resolve().parent / "angle_test"; OUT.mkdir(exist_ok=True)
gt = {**load_labels("dev"), **load_labels("lockbox")}
rng = np.random.default_rng(7)
def warp_video(src, dst, seed):
    r = np.random.default_rng(seed); cap = cv2.VideoCapture(src); fps = cap.get(5) or 30
    w, h = int(cap.get(3)), int(cap.get(4))
    # другой угол: сдвиг углов кадра до 14% (наклон/поворот камеры) + поворот до ±6° + приближение до 12%
    s = np.float32([[0, 0], [w, 0], [w, h], [0, h]]); d = (s + r.uniform(-0.14, 0.14, (4, 2)) * [w, h]).astype(np.float32)
    M = cv2.getPerspectiveTransform(s, d); ang = r.uniform(-6, 6); zoom = r.uniform(1.0, 1.12)
    R = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), ang, zoom), [0, 0, 1]]); M = R @ M
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    while True:
        ok, f = cap.read()
        if not ok: break
        out.write(cv2.warpPerspective(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE))
    out.release()
a = Analyzer(); preds = {}
for p in sorted(glob.glob(V + "/*.mov")):
    fid = "%02d" % int(re.search(r"-(\d+)\.mov$", p).group(1))
    if fid not in gt: continue
    dst = str(TMP / f"{fid}.mp4"); warp_video(p, dst, int(fid)); t = time.time()
    r = a.run(dst); os.remove(dst)
    preds[fid] = {"verdict": r["verdict"], "violations": [{"type_code": next((k for k, v in json.load(open("rules.json", encoding="utf-8")).items() if isinstance(v, dict) and v.get("title") == x["type"]), "procedure"), "timecode_sec": x["timecode_sec"]} for x in r["violations"]]}
    json.dump(preds, open(OUT / f"preds_{os.environ.get('FALLBACK_VISIT_S', '10')}.json", "w"), ensure_ascii=False)
    print(fid, r["verdict"], [v["type_code"] for v in preds[fid]["violations"]], f"{time.time() - t:.0f}s", flush=True)
for sp in ("dev", "lockbox"):
    g = load_labels(sp); s = score(g, {k: preds[k] for k in g}); print(sp, f"найдено {s['tp']}, ложных {s['fp']}, баллы {s['pts'] / s['pts_max']:.2f}")
