"""Leave-one-view-out: ролик ракурса V выравнивается БЕЗ эталона V (только к чужим ракурсам) — имитация нового угла с настоящим параллаксом.
Детекции берутся из cache_adapt (детектор не перезапускается), пересчитываются ракурс, гомография и датчики.
  python lovo_extract.py  → cache_lovo/{fid}.json
"""
import json, pathlib, sys, time
from features import frame_features
from pipeline import Geometry, GeoState, sample_frames

HERE = pathlib.Path(__file__).resolve().parent
V = pathlib.Path(r"C:/Users/Димаш/Downloads/open (1)/open/videos")
OUT = HERE / "cache_lovo"; OUT.mkdir(exist_ok=True)
LOCKBOX = {"08", "10", "11", "12", "13", "14", "15"}


def main():
    base = Geometry()
    for cf in sorted((HERE / "cache_adapt").glob("*.json")):
        fid = cf.stem
        if fid in LOCKBOX or (OUT / cf.name).exists(): continue
        c = json.load(open(cf)); own = c["frames"][0]["view"]
        geo = Geometry.__new__(Geometry); geo.__dict__.update(base.__dict__)
        geo.refs = {v: r for v, r in base.refs.items() if v != own}; geo.kd = {v: k for v, k in base.kd.items() if v != own}
        vid = next(V.glob(f"*-{int(fid)}.mov")); gs = GeoState(geo); rows = []; t0 = time.time()
        for k, (t, fr) in enumerate(sample_frames(vid)):
            if k >= len(c["frames"]): break
            view, H, inl = gs.update(fr, t); dets = c["frames"][k]["dets"]
            rows.append({"t": round(t, 2), "view": view, "inl": inl, "H": None if H is None else H.tolist(), "dets": dets,
                         "feat": frame_features(fr, H, view, dets, geo)})
        json.dump({"file_id": fid, "frames": rows, "extract_sec": c.get("extract_sec"), "own_view": own}, open(OUT / cf.name, "w"))
        inl = sorted(r["inl"] for r in rows)
        print(fid, own, "->", rows[0]["view"], "inl median", inl[len(inl) // 2], f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    main()
