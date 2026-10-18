# Open-vocab кэш: Grounding DINO tiny (Apache-2.0) по кадрам 1 fps для всех роликов; результат /kaggle/working/ov/<id>.json
import json, os, pathlib, re, time, cv2, torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
inp = pathlib.Path("/kaggle/input"); vids = sorted(next(p.parent for p in inp.rglob("v26.mov")).glob("*.mov"))
out = pathlib.Path("/kaggle/working/ov"); out.mkdir(exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)
R = "IDEA-Research/grounding-dino-tiny"; P = AutoProcessor.from_pretrained(R); M = AutoModelForZeroShotObjectDetection.from_pretrained(R).to(dev).eval()
TEXT = "mobile phone. banknotes. stack of money. money bag. laptop screen."
for v in vids:
    fid = re.sub(r"\D", "", v.name).zfill(2); cap = cv2.VideoCapture(str(v)); fps = cap.get(5) or 30; step = int(round(fps)); i = 0; rows = []; t0 = time.time()
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i % step == 0:
            im = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB); x = P(images=im, text=TEXT, return_tensors="pt").to(dev)
            with torch.no_grad(): o = M(**x)
            try: r = P.post_process_grounded_object_detection(o, x.input_ids, threshold=0.25, text_threshold=0.2, target_sizes=[im.shape[:2]])[0]
            except TypeError: r = P.post_process_grounded_object_detection(o, x.input_ids, box_threshold=0.25, text_threshold=0.2, target_sizes=[im.shape[:2]])[0]
            labs = r.get("text_labels", r.get("labels"))
            rows.append({"t": round(i / fps, 2), "dets": [[str(l), round(float(s), 3), [round(float(q), 1) for q in b]] for l, s, b in zip(labs, r["scores"], r["boxes"])]})
        i += 1
    json.dump({"file_id": fid, "frames": rows}, open(out / f"{fid}.json", "w")); print(fid, len(rows), f"{time.time()-t0:.0f}c", flush=True)
print("done")
