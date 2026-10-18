# Быстрая проверка: различает ли Qwen3-VL-2B телефон / купюры / карточку на спорных кадрах (CPU).
import glob, json, os, subprocess, sys, time, cv2, numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
V = r"C:/Users/Димаш/Downloads/open (1)/open/videos"
def frame(n, t):
    f = [p for p in glob.glob(V + "/*.mov") if os.path.basename(p)[2:-4] == str(int(n))][0]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", f, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    return Image.fromarray(cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), 1)[0:700], cv2.COLOR_BGR2RGB))
R = "Qwen/Qwen3-VL-2B-Instruct"; t0 = time.time()
P = AutoProcessor.from_pretrained(R); M = AutoModelForImageTextToText.from_pretrained(R, torch_dtype=torch.float32).eval()
print("модель загружена", round(time.time() - t0), "с", flush=True)
Q = ("This is a CCTV frame from a bank cashier room. The cashier is the bank employee sitting at the desk; a client may sit across. "
     "What small object is being held or handed over near the desk? Answer ONLY JSON: "
     '{"object": "mobile phone"|"banknotes"|"card or document"|"other"|"none", "held_by": "cashier"|"client"|"nobody"|"both", "cashier_uses_phone": true/false}')
for n, t, truth in [("05", 20, "клиент держит телефон (чисто)"), ("43", 21, "купюры (чисто)"), ("19", 21, "карточка/бумага (чисто)"),
                    ("44", 19, "нарушение, метка 16 с"), ("02", 10, "нарушение, метка 10 с"), ("20", 6, "кассир держит телефон (нарушение)")]:
    img = frame(n, t); msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": Q}]}]
    x = P.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    s = time.time()
    with torch.no_grad(): g = M.generate(**x, max_new_tokens=60, do_sample=False)
    ans = P.batch_decode(g[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    print(f"#{n} {t}с [{truth}] → {ans.strip()}  ({time.time()-s:.0f} с)", flush=True)
