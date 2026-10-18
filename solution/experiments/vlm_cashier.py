"""Проба VLM (Qwen3-VL-2B, Apache-2.0) на кропе кассира: «держит ли кассир телефон?»
Нарушение «телефон»: #02, #20, #44; чистые: #05 (телефон у клиента), #27, #35. 5 кадров на ролик, CPU, время на кадр.
  python vlm_cashier.py  → runs/vlm_cashier.jsonl
"""
import json, os, pathlib, sys, time
import cv2, numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import decide as D
from pipeline import sample_frames

if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(max(1, os.cpu_count() - 2))
HERE = pathlib.Path(__file__).resolve().parent
V = pathlib.Path(r"C:/Users/Димаш/Downloads/open (1)/open/videos")
MASK = os.environ.get("VLM_MASK") == "1"; TAG = "_mask" if MASK else ""
SPEC = {"02": (True, [9, 10, 11, 12, 13]), "20": (True, [4, 5, 6, 7, 8]), "44": (True, [14, 15, 16, 17, 18]),
        "05": (False, [8, 12, 20, 30, 40]), "27": (False, [0, 2, 4, 6, 8]), "35": (False, [10, 18, 26, 34, 42])}
Q = ("This is a crop of a bank cashier from a CCTV camera. Is the cashier holding or using a mobile phone (smartphone)? "
     "Banknotes, plastic cards, papers, a calculator or a desk telephone are NOT a mobile phone. "
     'Answer ONLY JSON: {"phone": true|false, "object_in_hands": "mobile phone"|"banknotes"|"card or paper"|"nothing"|"other"}')


def crops(fid, ts):
    cache = json.load(open(HERE / "cache_adapt" / f"{fid}.json")); cid, per = D.cashier_track(cache)
    vid = next(V.glob(f"*-{int(fid)}.mov")); want = {}; out = []
    for k, (t, fr) in enumerate(sample_frames(vid)):
        if k >= len(per): break
        for q in ts:
            if q not in want and t >= q:
                b = next((b for i, b in per[k] if i == cid), None)
                if b is None: continue
                w, h = b[2] - b[0], b[3] - b[1]
                x0, y0, x1, y1 = int(max(0, b[0] - .3 * w)), int(max(0, b[1] - .1 * h)), int(min(fr.shape[1], b[2] + .3 * w)), int(min(fr.shape[0], b[3]))
                c = fr[y0:y1, x0:x1].copy()
                if MASK:                                                   # всё вне рамки кассира — серое (чужие руки и телефоны не в кадре)
                    m = np.full_like(c, 128); bx0, by0 = int(b[0]) - x0, int(b[1]) - y0
                    m[max(0, by0):int(b[3]) - y0, max(0, bx0):int(b[2]) - x0] = c[max(0, by0):int(b[3]) - y0, max(0, bx0):int(b[2]) - x0]; c = m
                s = 448 / max(c.shape[:2]); c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
                want[q] = 1; out.append((round(t, 1), Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))))
    return out


def main():
    t0 = time.time(); R = "Qwen/Qwen3-VL-2B-Instruct"
    P = AutoProcessor.from_pretrained(R); M = AutoModelForImageTextToText.from_pretrained(R, torch_dtype=torch.float32).eval()
    print("модель загружена", round(time.time() - t0), "с", flush=True)
    log = open(HERE / "runs" / f"vlm_cashier{TAG}.jsonl", "w", encoding="utf-8")
    for fid, (pos, ts) in SPEC.items():
        for t, img in crops(fid, ts):
            img.save(HERE / "runs" / f"vlm{TAG}_{fid}_{t}.jpg")
            msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": Q}]}]
            x = P.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            s = time.time()
            with torch.no_grad(): g = M.generate(**x, max_new_tokens=40, do_sample=False)
            ans = P.batch_decode(g[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            r = {"fid": fid, "t": t, "truth_phone_video": pos, "answer": ans, "sec": round(time.time() - s, 1)}
            print(json.dumps(r, ensure_ascii=False), flush=True); log.write(json.dumps(r, ensure_ascii=False) + "\n"); log.flush()


if __name__ == "__main__":
    main()
