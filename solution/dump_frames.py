"""Кадры dev-роликов в моменты кэша (cache_adapt) → frames/{fid}/{k:04d}.jpg — общий вход для датчиков без знания комнаты."""
import json, os, pathlib, sys
import cv2

HERE = pathlib.Path(__file__).resolve().parent
V = pathlib.Path(r"C:/Users/Димаш/Downloads/open (1)/open/videos")
LOCKBOX = {"08", "10", "11", "12", "13", "14", "15"}

for cf in sorted((HERE / "cache_adapt").glob("*.json")):
    fid = cf.stem; out = HERE / "frames" / fid
    if (fid in LOCKBOX and not os.environ.get("WITH_LOCKBOX")) or (out / "done").exists(): continue
    out.mkdir(parents=True, exist_ok=True); fr = json.load(open(cf))["frames"]
    vid = next(V.glob(f"*-{int(fid)}.mov")); cap = cv2.VideoCapture(str(vid)); fps = cap.get(cv2.CAP_PROP_FPS) or 30; i = q = 0
    while q < len(fr):
        ok, im = cap.read()
        if not ok: break
        t = i / fps; i += 1
        while q < len(fr) and t >= fr[q]["t"] - 0.5 / fps:
            cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(out / f"{q:04d}.jpg")); q += 1
    (out / "done").write_text(str(q)); print(fid, q, len(fr), flush=True)
