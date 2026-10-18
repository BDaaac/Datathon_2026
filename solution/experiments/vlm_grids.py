"""Сетки 3x2 кадров визита (полное разрешение, подпись времени) для VLM-верификатора на Kaggle. Без модели — только cv2."""
import json, pathlib, sys
import cv2, numpy as np
HERE = pathlib.Path(__file__).resolve().parent.parent
OUT = HERE.parent / "research/kaggle" / "vlm_grids" / "grids"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
from evaluate import load_labels
ids = sorted({**load_labels("dev"), **load_labels("lockbox")})
meta = {}
for fid in ids:
    c = json.load(open(HERE / "cache_adapt" / f"{fid}.json"))["frames"]
    n = min(len(c), len(list((HERE / "frames" / fid).glob("*.jpg"))))
    two = [k for k in range(n) if sum(1 for d in c[k]["dets"] if d[0] == "person" and d[1] >= .5) >= 2]
    pool = two if len(two) >= 2 else list(range(n))
    idx = sorted({pool[int(round(i * (len(pool) - 1) / 5))] for i in range(6)})
    tiles = []
    for k in idx:
        im = cv2.imdecode(np.fromfile(str(HERE / "frames" / fid / f"{k:04d}.jpg"), np.uint8), cv2.IMREAD_COLOR)
        s = f"t={c[k]['t']:.0f}s"
        cv2.putText(im, s, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6); cv2.putText(im, s, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        tiles.append(im)
    while len(tiles) < 6: tiles.append(np.zeros_like(tiles[0]))
    g = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(OUT / f"{fid}.jpg"))
    meta[fid] = {"frames": idx, "times": [c[k]["t"] for k in idx], "visit_frames": len(two)}
json.dump(meta, open(OUT / "meta.json", "w"))
print(len(ids), "сеток", g.shape)
