"""Тяжёлая часть, считается один раз на видео: ракурс, гомография к эталону, детекции D-FINE на кадрах 2 fps.
Результат — cache/<file_id>.json; правила (decide.py) потом гоняются по кэшу за секунды, что и нужно для A/B.

  python extract.py --videos "<папка>" [--ids 02,26]
"""
import argparse, glob, json, os, pathlib, re, sys, time
from pipeline import Geometry, Detector, GeoState, sample_frames
from features import frame_features

OUT = pathlib.Path(os.environ.get("CACHE_DIR", pathlib.Path(__file__).resolve().parent / "cache"))


def main():
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--videos", required=True); ap.add_argument("--ids", default="")
    a = ap.parse_args(); OUT.mkdir(exist_ok=True)
    want = {x.zfill(2) for x in a.ids.split(",") if x}
    geo, det = Geometry(), Detector()
    for p in sorted(glob.glob(os.path.join(a.videos, "*.mov"))):
        fid = re.sub(r"\D", "", os.path.basename(p)).zfill(2)
        if fid in ("47", "48") or (want and fid not in want) or (OUT / f"{fid}.json").exists(): continue
        t0 = time.time(); rows = []; gs = GeoState(geo)
        warp = None
        if os.environ.get("WARP_SEED"):                       # стресс-тест: случайный «новый ракурс» (перспектива + поворот + масштаб)
            import numpy as np, cv2
            rng = np.random.default_rng(int(os.environ["WARP_SEED"]) + int(fid)); h, w = 848, 464; d = float(os.environ.get("WARP_STRENGTH", 0.12))
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]]); dst = (src + rng.uniform(-d, d, (4, 2)) * [w, h]).astype(np.float32)
            warp = cv2.getPerspectiveTransform(src, dst)
        for i, (t, fr) in enumerate(sample_frames(p)):
            if warp is not None: fr = cv2.warpPerspective(fr, warp, (464, 848), borderMode=cv2.BORDER_REFLECT)
            view, H, inl = gs.update(fr, t)
            dets = [[l, round(s, 3), [round(v, 1) for v in b]] for l, s, b in det(fr)]
            rows.append({"t": round(t, 2), "view": view, "inl": inl, "H": None if H is None else H.tolist(), "dets": dets,
                         "feat": frame_features(fr, H, view, dets, geo)})
        json.dump({"file_id": fid, "frames": rows, "extract_sec": round(time.time() - t0, 1)}, open(OUT / f"{fid}.json", "w"))
        print(fid, len(rows), f"{time.time() - t0:.0f} c", flush=True)


if __name__ == "__main__":
    main()
