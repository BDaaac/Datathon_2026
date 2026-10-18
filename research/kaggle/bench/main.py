"""Замер детекторов и open-vocab моделей на кадрах ForteBank: что ловят и сколько секунд на кадр (GPU и CPU)."""
import json, os, pathlib, time, traceback, sys
import torch
from PIL import Image, ImageDraw

INP = next(p.parent for p in pathlib.Path("/kaggle/input").rglob("manifest.json"))
OUT = pathlib.Path("/kaggle/working"); OUT.mkdir(exist_ok=True)
meta = json.load(open(INP / "manifest.json"))
imgs = {m["file"]: Image.open(INP / m["file"]).convert("RGB") for m in meta}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "", flush=True)

PROMPTS = ["person", "mobile phone", "handbag", "cup", "bottle", "laptop", "banknotes", "money bag", "safe", "open door"]
results, timing, errors = {}, {}, {}


def bench(name, load, infer, cpu_frames=3):
    try:
        t0 = time.time(); m = load(DEV); tl = time.time() - t0
        first = list(imgs)[0]; infer(m, imgs[first], DEV)                     # прогрев
        if DEV == "cuda": torch.cuda.synchronize()
        res, ts = {}, []
        for f, im in imgs.items():
            t = time.time(); res[f] = infer(m, im, DEV)
            if DEV == "cuda": torch.cuda.synchronize()
            ts.append(time.time() - t)
        results[name] = res
        timing[name] = {"load_s": round(tl, 1), "gpu_s_per_frame": round(sum(ts) / len(ts), 3)}
        del m; torch.cuda.empty_cache()
        mc = load("cpu"); ts = []
        for f in list(imgs)[:cpu_frames]:
            t = time.time(); infer(mc, imgs[f], "cpu"); ts.append(time.time() - t)
        timing[name]["cpu_s_per_frame"] = round(sum(ts) / len(ts), 2)
        del mc
        print("OK", name, timing[name], flush=True)
    except Exception as e:
        errors[name] = traceback.format_exc()[-1500:]; print("FAIL", name, repr(e), flush=True)


def boxes(scores, labels, bxs, names=None):
    out = []
    for s, l, b in zip(scores, labels, bxs):
        lab = names[int(l)] if names is not None and not isinstance(l, str) else str(l)
        out.append({"label": lab, "score": round(float(s), 3), "box": [round(float(x), 1) for x in b]})
    return out


# ---------- COCO-детекторы (transformers) ----------
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def coco_model(repo_ids):
    def load(dev):
        last = None
        for r in repo_ids:
            try:
                p = AutoImageProcessor.from_pretrained(r); m = AutoModelForObjectDetection.from_pretrained(r).to(dev).eval()
                return (p, m, r)
            except Exception as e:
                last = e
        raise last

    def infer(pm, im, dev):
        p, m, _ = pm
        x = p(images=im, return_tensors="pt").to(dev)
        with torch.no_grad(): o = m(**x)
        r = p.post_process_object_detection(o, target_sizes=[im.size[::-1]], threshold=0.3)[0]
        return boxes(r["scores"], r["labels"], r["boxes"], m.config.id2label)
    return load, infer


bench("rtdetr_v2_r18", *coco_model(["PekingU/rtdetr_v2_r18vd", "PekingU/rtdetr_r18vd"]))
bench("dfine_s_coco", *coco_model(["ustc-community/dfine-small-coco", "ustc-community/dfine_s_coco"]))

# ---------- OWLv2 ----------
from transformers import Owlv2Processor, Owlv2ForObjectDetection


def owl_load(dev):
    r = "google/owlv2-base-patch16-ensemble"
    return Owlv2Processor.from_pretrained(r), Owlv2ForObjectDetection.from_pretrained(r).to(dev).eval()


def owl_infer(pm, im, dev):
    p, m = pm
    x = p(text=[PROMPTS], images=im, return_tensors="pt").to(dev)
    with torch.no_grad(): o = m(**x)
    s = max(im.size)                                   # OWLv2 дополняет кадр до квадрата
    try:
        r = p.post_process_grounded_object_detection(o, target_sizes=torch.tensor([[s, s]]), threshold=0.2)[0]
    except Exception:
        r = p.post_process_object_detection(o, target_sizes=torch.tensor([[s, s]]), threshold=0.2)[0]
    return boxes(r["scores"], r["labels"], r["boxes"], PROMPTS)


bench("owlv2_base", owl_load, owl_infer)

# ---------- Grounding DINO ----------
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def gd_load(dev):
    r = "IDEA-Research/grounding-dino-tiny"
    return AutoProcessor.from_pretrained(r), AutoModelForZeroShotObjectDetection.from_pretrained(r).to(dev).eval()


def gd_infer(pm, im, dev):
    p, m = pm
    text = ". ".join(PROMPTS) + "."
    x = p(images=im, text=text, return_tensors="pt").to(dev)
    with torch.no_grad(): o = m(**x)
    try:
        r = p.post_process_grounded_object_detection(o, x.input_ids, threshold=0.3, text_threshold=0.25, target_sizes=[im.size[::-1]])[0]
    except TypeError:
        r = p.post_process_grounded_object_detection(o, x.input_ids, box_threshold=0.3, text_threshold=0.25, target_sizes=[im.size[::-1]])[0]
    labs = r.get("text_labels", r.get("labels"))
    return boxes(r["scores"], labs, r["boxes"])


bench("grounding_dino_tiny", gd_load, gd_infer)

# ---------- Florence-2 (MIT) ----------
from transformers import AutoModelForCausalLM
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports


def _no_flash(f):
    imps = get_imports(f)
    return [i for i in imps if i != "flash_attn"]


def fl_load(dev):
    r = "microsoft/Florence-2-base"
    with patch("transformers.dynamic_module_utils.get_imports", _no_flash):
        m = AutoModelForCausalLM.from_pretrained(r, trust_remote_code=True, torch_dtype=torch.float32).to(dev).eval()
    return AutoProcessor.from_pretrained(r, trust_remote_code=True), m


def fl_run(p, m, im, dev, task, text=""):
    x = p(text=task + text, images=im, return_tensors="pt").to(dev)
    with torch.no_grad():
        g = m.generate(input_ids=x["input_ids"], pixel_values=x["pixel_values"], max_new_tokens=256, num_beams=3)
    s = p.batch_decode(g, skip_special_tokens=False)[0]
    return p.post_process_generation(s, task=task, image_size=im.size)[task]


def fl_infer(pm, im, dev):
    p, m = pm
    od = fl_run(p, m, im, dev, "<OD>")
    gr = fl_run(p, m, im, dev, "<CAPTION_TO_PHRASE_GROUNDING>", "a mobile phone. a cup. a handbag. banknotes. a money bag.")
    out = [{"label": l, "score": 1.0, "box": [round(float(v), 1) for v in b], "task": "OD"} for l, b in zip(od["labels"], od["bboxes"])]
    out += [{"label": l, "score": 1.0, "box": [round(float(v), 1) for v in b], "task": "grounding"} for l, b in zip(gr["labels"], gr["bboxes"])]
    return out


bench("florence2_base", fl_load, fl_infer, cpu_frames=2)

# ---------- сохранение и картинки ----------
json.dump({"timing": timing, "errors": errors, "results": results, "manifest": meta},
          open(OUT / "bench.json", "w"), ensure_ascii=False)
show = ["v02_t0115.jpg", "v46_t0120.jpg", "v40_t0170.jpg", "v26_t0150.jpg", "v17_t0080.jpg", "v32_t0080.jpg"]
for name, res in results.items():
    for f in show:
        if f not in res: continue
        im = imgs[f].copy(); d = ImageDraw.Draw(im)
        for b in res[f]:
            if b["label"] in ("dining table", "chair", "tv", "keyboard"): continue
            d.rectangle(b["box"], outline=(255, 0, 0), width=2); d.text((b["box"][0] + 2, b["box"][1] + 2), f'{b["label"]} {b["score"]:.2f}', fill=(255, 255, 0))
        im.save(OUT / f"vis_{name}_{f}")
print("TIMING", json.dumps(timing)); print("ERRORS", list(errors))
