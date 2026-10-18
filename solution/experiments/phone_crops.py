"""Кропы-кандидаты «мелкий предмет у человека» для классификатора телефон / купюры / карта / другое.

Источник — детекции D-FINE из cache_adapt (cell phone, remote, book ≥ 0.35) внутри рамки любого человека.
Только dev-ролики. Не чаще 1 кропа в секунду на ролик и класс-ячейку 40 px (без почти-дублей).
  python phone_crops.py  → crops/phone/{fid}_{t}_{label}.jpg + crops/phone/index.csv + сетки для разметки
"""
import csv, json, pathlib, sys
import cv2, numpy as np
from pipeline import sample_frames

HERE = pathlib.Path(__file__).resolve().parent
V = pathlib.Path(r"C:/Users/Димаш/Downloads/open (1)/open/videos")
OUT = HERE / "crops" / "phone"; OUT.mkdir(parents=True, exist_ok=True)
LOCKBOX = {"08", "10", "11", "12", "13", "14", "15"}
CLS = {"cell phone", "remote", "book"}


def crop(fr, b, size=96):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; s = max(48, 2.2 * max(b[2] - b[0], b[3] - b[1])) / 2
    x0, y0, x1, y1 = int(max(0, cx - s)), int(max(0, cy - s)), int(min(fr.shape[1], cx + s)), int(min(fr.shape[0], cy + s))
    return cv2.resize(fr[y0:y1, x0:x1], (size, size))


def main():
    rows = []
    for cf in sorted((HERE / "cache_adapt").glob("*.json")):
        fid = cf.stem
        if fid in LOCKBOX: continue
        vid = next((p for p in V.glob(f"*-{int(fid)}.mov")), None)
        if vid is None: continue
        frames = json.load(open(cf))["frames"]; seen = {}
        for k, (t, fr) in enumerate(sample_frames(vid)):
            if k >= len(frames): break
            dets = frames[k]["dets"]; people = [b for l, s, b in dets if l == "person" and s >= 0.5]
            for l, s, b in dets:
                if l not in CLS: continue
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if not any(p[0] <= cx <= p[2] and p[1] <= cy <= p[3] for p in people): continue
                key = (round(cx / 40), round(cy / 40))
                if t - seen.get(key, -9) < 1.0: continue
                seen[key] = t
                name = f"{fid}_{t:06.2f}_{l.replace(' ', '')}.jpg"
                cv2.imencode(".jpg", crop(fr, b))[1].tofile(str(OUT / name))
                rows.append({"file": name, "fid": fid, "t": round(t, 2), "det": l, "score": s, "box": [round(v) for v in b]})
        print(fid, len(rows), flush=True)
    with open(OUT / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # сетки 8×6 с номерами для разметки глазами
    for g in range(0, len(rows), 48):
        tiles = []
        for j, r in enumerate(rows[g:g + 48]):
            im = cv2.imdecode(np.fromfile(str(OUT / r["file"]), np.uint8), 1); cv2.putText(im, str(g + j), (2, 14), 0, 0.45, (0, 0, 255), 1); tiles.append(im)
        tiles += [np.zeros((96, 96, 3), np.uint8)] * (48 - len(tiles))
        grid = np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, 48, 8)])
        cv2.imencode(".jpg", grid)[1].tofile(str(OUT / f"grid_{g // 48:02d}.jpg"))
    print("всего", len(rows))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    main()
