"""VLM-верификатор открытого класса: Qwen3-VL-2B смотрит сетку 3x2 кадров визита → JSON violation/норма."""
import json, os, pathlib, sys, time, re
import cv2, numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(max(1, os.cpu_count() - 2))
HERE = pathlib.Path(__file__).resolve().parent.parent
OUT = HERE / "experiments" / "vlm_verify"; OUT.mkdir(exist_ok=True)
VIOL = ["01", "02", "04", "17", "38", "44", "13"]
CLEAN = ["03", "05", "16", "19", "31", "35", "37", "39", "45", "43", "21", "23", "25", "27", "29", "08", "10", "12", "14"]
MAXSIDE = int(os.environ.get("GRID", 700))
PROMPT = ("This is a contact sheet of 6 CCTV frames (time-stamped, in order) of one customer visit at a bank cashier desk. "
 "The cashier is the bank employee sitting behind the desk. Decide whether the visit contains a violation of cashier discipline. "
 "Violation signs: the cashier talks on or holds a mobile phone; an outsider stands at the cashier's desk or reaches for the money; "
 "a visitor leaves a bag or money and walks away without completing a transaction; money or valuables are left on the desk unattended; "
 "cash is counted without the client present; a second visitor enters while the previous client's money is still on the desk; "
 "the cashier leaves the workplace during service. If none of these happen and it is an ordinary service, it is not a violation. "
 'Answer ONLY JSON: {"violation": true|false, "confidence": 0..1, "type": "...", "reason": "..."}')

def imread(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)

def pick(fid):
    c = json.load(open(HERE / "cache_adapt" / f"{fid}.json"))["frames"]
    n = min(len(c), len(list((HERE / "frames" / fid).glob("*.jpg"))))
    two = [k for k in range(n) if sum(1 for d in c[k]["dets"] if d[0] == "person" and d[1] >= .5) >= 2]
    pool = two if len(two) >= 2 else list(range(n))
    idx = sorted({pool[int(round(i * (len(pool) - 1) / 5))] for i in range(6)})
    return idx, [c[k]["t"] for k in idx], len(two)

def grid(fid, idx, ts):
    tiles = []
    for k, t in zip(idx, ts):
        im = imread(HERE / "frames" / fid / f"{k:04d}.jpg")
        cv2.putText(im, f"t={t:.0f}s", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6); cv2.putText(im, f"t={t:.0f}s", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 2)
        tiles.append(im)
    while len(tiles) < 6: tiles.append(np.zeros_like(tiles[0]))
    g = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    s = MAXSIDE / max(g.shape[:2]); g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(g, cv2.COLOR_BGR2RGB))

def parse(a):
    v = re.search(r'"violation"\s*:\s*(true|false)', a, re.I); c = re.search(r'"confidence"\s*:\s*([0-9.]+)', a)
    return (v is not None and v.group(1).lower() == "true"), (float(c.group(1)) if c else .5), {}

def auc(y, s):
    p = [a for a, b in zip(s, y) if b]; n = [a for a, b in zip(s, y) if not b]
    return sum((a > b) + .5 * (a == b) for a in p for b in n) / (len(p) * len(n))

def main():
    R = "Qwen/Qwen3-VL-2B-Instruct"; t0 = time.time()
    P = AutoProcessor.from_pretrained(R); M = AutoModelForImageTextToText.from_pretrained(R, torch_dtype=torch.float32).eval()
    print("load", round(time.time() - t0), flush=True)
    done = {}
    rp = OUT / "results.jsonl"
    if rp.exists(): done = {json.loads(l)["fid"]: json.loads(l) for l in open(rp, encoding="utf-8")}
    log = open(rp, "a", encoding="utf-8")
    for fid in VIOL + CLEAN:
        if fid in done: continue
        idx, ts, n2 = pick(fid); img = grid(fid, idx, ts); img.save(OUT / f"grid_{fid}.jpg")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
        x = P.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        s = time.time()
        with torch.no_grad(): g = M.generate(**x, max_new_tokens=70, do_sample=False)
        ans = P.batch_decode(g[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        v, c, j = parse(ans)
        r = {"fid": fid, "truth": fid in VIOL, "answer": ans, "violation": v, "conf": c, "frames": idx, "n_two": n2, "sec": round(time.time() - s, 1)}
        print(json.dumps(r, ensure_ascii=False), flush=True); log.write(json.dumps(r, ensure_ascii=False) + "\n"); log.flush()
    rs = [json.loads(l) for l in open(rp, encoding="utf-8")]
    y = [r["truth"] for r in rs]; sc = [r["conf"] if r["violation"] else 1 - r["conf"] for r in rs]
    print("viol found", sum(r["violation"] for r in rs if r["truth"]), "/", sum(y),
          "clean ok", sum(not r["violation"] for r in rs if not r["truth"]), "/", len(y) - sum(y),
          "AUC", round(auc(y, sc), 3), "sec/video", round(np.mean([r["sec"] for r in rs]), 1))

if __name__ == "__main__": main()
