# «Увеличение»: D-FINE-S на полосе кадра со столом и людьми (y 230..650), координаты возвращаются в исходный кадр.
import json, pathlib, re, time, cv2, torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
inp = pathlib.Path("/kaggle/input"); vids = sorted(next(p.parent for p in inp.rglob("v26.mov")).glob("*.mov"))
out = pathlib.Path("/kaggle/working/crop"); out.mkdir(exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)
R = "ustc-community/dfine-small-coco"; P = AutoImageProcessor.from_pretrained(R); M = AutoModelForObjectDetection.from_pretrained(R).to(dev).eval()
Y0, Y1 = 230, 650
for v in vids:
    fid = re.sub(r"\D", "", v.name).zfill(2); cap = cv2.VideoCapture(str(v)); fps = cap.get(5) or 30; step = max(1, int(round(fps / 2))); i = 0; rows = []; t0 = time.time()
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i % step == 0:
            crop = cv2.cvtColor(fr[Y0:Y1], cv2.COLOR_BGR2RGB); x = P(images=crop, return_tensors="pt").to(dev)
            with torch.no_grad(): o = M(**x)
            r = P.post_process_object_detection(o, target_sizes=[crop.shape[:2]], threshold=0.2)[0]
            rows.append({"t": round(i / fps, 2), "dets": [[M.config.id2label[int(l)], round(float(s), 3), [round(float(b[0]), 1), round(float(b[1]) + Y0, 1), round(float(b[2]), 1), round(float(b[3]) + Y0, 1)]]
                                                       for s, l, b in zip(r["scores"], r["labels"], r["boxes"])]})
        i += 1
    json.dump({"file_id": fid, "frames": rows}, open(out / f"{fid}.json", "w")); print(fid, len(rows), f"{time.time()-t0:.0f}c", flush=True)
print("done")
