# Qwen3-VL-2B-Instruct (Apache-2.0): чек-лист по кадрам раз в 2 с; результат /kaggle/working/vlm/<id>.json
import json, pathlib, re, time, cv2, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
inp = pathlib.Path("/kaggle/input"); vids = sorted(next(p.parent for p in inp.rglob("v26.mov")).glob("*.mov"))
out = pathlib.Path("/kaggle/working/vlm"); out.mkdir(exist_ok=True)
R = "Qwen/Qwen3-VL-2B-Instruct"
P = AutoProcessor.from_pretrained(R); M = AutoModelForImageTextToText.from_pretrained(R, torch_dtype=torch.float16, device_map="cuda").eval()
Q = ("This is a CCTV frame from a bank cashier room. The cashier is the bank employee working at the desk. "
     "Answer ONLY with compact JSON with these keys: "
     '{"cashier_holds_phone": true/false, "client_holds_phone": true/false, "banknotes_visible": true/false, '
     '"money_bag_visible": true/false, "cashier_present": true/false, "people_in_room": integer, '
     '"room_door_open": true/false, "laptop_screen_shows_open_app": true/false}')
def ask(img):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": Q}]}]
    x = P.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")
    with torch.no_grad(): g = M.generate(**x, max_new_tokens=120, do_sample=False)
    return P.batch_decode(g[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0]
for v in vids:
    fid = re.sub(r"\D", "", v.name).zfill(2); cap = cv2.VideoCapture(str(v)); fps = cap.get(5) or 30; step = int(round(fps * 2)); i = 0; rows = []; t0 = time.time()
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i % step == 0:
            img = Image.fromarray(cv2.cvtColor(fr[0:700], cv2.COLOR_BGR2RGB))
            txt = ask(img); m = re.search(r"\{.*\}", txt, re.S)
            try: ans = json.loads(m.group(0)) if m else {}
            except Exception: ans = {}
            rows.append({"t": round(i / fps, 2), "ans": ans, "raw": txt[:300]})
        i += 1
    json.dump({"file_id": fid, "frames": rows}, open(out / f"{fid}.json", "w")); print(fid, len(rows), f"{time.time()-t0:.0f}c", flush=True)
print("done")
