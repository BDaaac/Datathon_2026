# VLM-верификатор визитов: Qwen3-VL-4B и -8B (Apache-2.0) по сеткам 3x2 кадров. Результат: /kaggle/working/vlm_<модель>.jsonl
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.57", "accelerate", "bitsandbytes"], check=False)
import gc, json, pathlib, re, time, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
print("GPU:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
G = next(p.parent for p in pathlib.Path("/kaggle/input").rglob("meta.json"))
ids = sorted(p.stem for p in G.glob("*.jpg"))
PROMPT = ("This is a contact sheet of 6 CCTV frames (time-stamped, in time order) of one video at a bank cashier desk. "
 "The cashier is the bank employee sitting behind the desk; visitors come to the other side of the desk. "
 "Decide whether the video shows a violation of cashier discipline. Violation signs include: the cashier talks on or holds a mobile phone; "
 "an outsider stands at the cashier's desk, comes behind it or reaches for money; a visitor leaves a bag or money and walks away without completing a transaction; "
 "money or valuables are left on the desk unattended; cash is counted or handled without the client present; a second visitor enters while the previous client's money is still on the desk; "
 "the cashier leaves the workplace during service or leaves the room door open. An ordinary service (client sits, hands over money/documents, cashier processes, client leaves) is NOT a violation. "
 'Be strict: say violation only if you can actually see a sign. Answer ONLY JSON: {"suspicion": 0-10, "violation": true|false, "type": "...", "reason": "<=25 words"}')
def run(repo, quant):
    t0 = time.time(); P = AutoProcessor.from_pretrained(repo)
    kw = dict(device_map="auto", torch_dtype=torch.float16)
    if quant: kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    M = AutoModelForImageTextToText.from_pretrained(repo, **kw).eval(); print(repo, "load", round(time.time() - t0), "s", flush=True)
    out = open(f"/kaggle/working/vlm_{repo.split('/')[-1]}.jsonl", "w")
    for fid in ids:
        img = Image.open(G / f"{fid}.jpg").convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
        x = P.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(M.device)
        s = time.time()
        with torch.no_grad(): g = M.generate(**x, max_new_tokens=120, do_sample=False)
        a = P.batch_decode(g[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        sus = re.search(r'"suspicion"\s*:\s*([0-9.]+)', a); v = re.search(r'"violation"\s*:\s*(true|false)', a, re.I)
        r = {"fid": fid, "suspicion": float(sus.group(1)) if sus else None, "violation": (v.group(1).lower() == "true") if v else None, "answer": a[:400], "sec": round(time.time() - s, 1)}
        print(json.dumps(r, ensure_ascii=False), flush=True); out.write(json.dumps(r, ensure_ascii=False) + "\n"); out.flush()
    del M; gc.collect(); torch.cuda.empty_cache()
vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 2**30
for repo, quant in [("Qwen/Qwen3-VL-4B-Instruct", False), ("Qwen/Qwen3-VL-8B-Instruct", vram < 24)]:
    try: run(repo, quant)
    except Exception as e: print("FAIL", repo, repr(e)[:500], flush=True)
print("done")
