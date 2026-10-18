"""Досчитывает в кэше признак feat.blob_self (пятно на столе против эталона из самого ролика). Детекции и H — из кэша.
  python blob_self.py --cache cache_adapt [--warp-seed 7]
"""
import argparse, json, pathlib, sys
import cv2, numpy as np
from features import desk_self_blobs
from decide import MASK

HERE = pathlib.Path(__file__).resolve().parent
V = pathlib.Path(r"C:/Users/Димаш/Downloads/open (1)/open/videos")


def person_mask(fr, H):
    m = np.zeros((848, 464), bool)
    for l, s, b in fr["dets"]:
        if l != "person" or s < 0.5: continue
        pts = cv2.perspectiveTransform(np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]).reshape(-1, 1, 2), H).reshape(-1, 2)
        m |= cv2.fillPoly(np.zeros((848, 464), np.uint8), [np.int32(pts)], 1).astype(bool)
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True); ap.add_argument("--warp-seed", type=int); a = ap.parse_args()
    for cf in sorted((HERE / a.cache).glob("*.json")):
        c = json.load(open(cf)); fr = c["frames"]
        if not fr or "blob_self" in (fr[0].get("feat") or {}): continue
        fid = cf.stem; vid = next(V.glob(f"*-{int(fid)}.mov"), None)
        if vid is None: continue
        warp = None
        if a.warp_seed is not None:
            rng = np.random.default_rng(a.warp_seed + int(fid)); h, w = 848, 464
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]]); dst = (src + rng.uniform(-0.12, 0.12, (4, 2)) * [w, h]).astype(np.float32)
            warp = cv2.getPerspectiveTransform(src, dst)
        cap = cv2.VideoCapture(str(vid)); fps = cap.get(cv2.CAP_PROP_FPS) or 30; i = 0; q = 0; imgs = {}
        while q < len(fr):
            ok, im = cap.read()
            if not ok: break
            t = i / fps; i += 1
            while q < len(fr) and t >= fr[q]["t"] - 0.5 / fps:
                if warp is not None: im2 = cv2.warpPerspective(im, warp, (464, 848), borderMode=cv2.BORDER_REFLECT)
                else: im2 = im
                imgs[q] = im2; q += 1
        # по ракурсам: эталон строится отдельно для каждого ракурса ролика
        res = [None] * len(fr)
        for v in {f["view"] for f in fr if f["H"] is not None}:
            idx = [k for k, f in enumerate(fr) if f["H"] is not None and f["view"] == v and k in imgs]
            if v not in MASK or not idx: continue
            Hs = [np.float32(fr[k]["H"]) for k in idx]
            wl = [cv2.warpPerspective(imgs[k], H, (464, 848)) for k, H in zip(idx, Hs)]
            pm = [person_mask(fr[k], H) for k, H in zip(idx, Hs)]
            for k, val in zip(idx, desk_self_blobs(wl, pm, MASK[v]["desk"])): res[k] = val
        for k, f in enumerate(fr): f.setdefault("feat", {})["blob_self"] = res[k]
        json.dump(c, open(cf, "w"))
        print(a.cache, fid, "ok", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    main()
