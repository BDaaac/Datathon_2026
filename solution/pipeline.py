"""Бейзлайн v0: видео → кадры 2 fps → D-FINE (люди/предметы) → SIFT-ракурс и зоны → правила → JSON по схеме API.

Правила v0 (минимум для сквозного пути): телефон в руках, личные вещи на столе, ценности на столе после ухода.
Дверь, посторонние, ПК, процедуры — следующие итерации.

  python pipeline.py --videos "<папка с .mov>" --out runs/v0 [--ids 02,26]
"""
import argparse, base64, glob, json, os, pathlib, re, sys, time
import cv2, numpy as np, torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

HERE = pathlib.Path(__file__).resolve().parent
A = HERE / "assets"
CFG = json.load(open(A / "views.json", encoding="utf-8"))
RULES = json.load(open(HERE / "rules.json", encoding="utf-8"))
FPS_SAMPLE = 2.0
DET_REPO = "ustc-community/dfine-small-coco"           # Apache-2.0, только COCO-веса
ITEMS = {"handbag", "backpack", "cup", "bottle", "wine glass"}


def imread(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
def prep(im): return cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))


class Geometry:
    """Определяет ракурс по SIFT и переводит точки кадра в координаты эталона ракурса."""
    def __init__(self):
        self.sift = cv2.SIFT_create(2000); self.bf = cv2.BFMatcher()
        self.refs = {v: imread(A / f"ref_{v}.jpg") for v in CFG["zones"]}
        self.kd = {v: self.sift.detectAndCompute(prep(r), None) for v, r in self.refs.items()}
        self.empty = {v: imread(A / f"empty_{v}.jpg") for v in CFG["empty_room"]}
        self.cfg_zones = CFG["zones"]
        self.masks = {v: {k: cv2.fillPoly(np.zeros((848, 464), np.uint8), [np.int32(z)], 1).astype(bool)
                          for k, z in zs.items()} for v, zs in CFG["zones"].items()}

    def homography(self, frame, view):
        k2, d2 = self.sift.detectAndCompute(prep(frame), None)
        if d2 is None: return None, 0
        k1, d1 = self.kd[view]
        m = [a for a, b in self.bf.knnMatch(d2, d1, k=2) if a.distance < 0.8 * b.distance]
        if len(m) < 8: return None, len(m)
        H, mk = cv2.findHomography(np.float32([k2[a.queryIdx].pt for a in m]),
                                   np.float32([k1[a.trainIdx].pt for a in m]), cv2.USAC_MAGSAC, 4.0)
        inl = int(mk.sum()) if mk is not None else 0
        return (H if inl >= 30 else None), inl

    def best_view(self, frame):
        sc = {v: self.homography(frame, v)[1] for v in self.refs}
        v = max(sc, key=sc.get)
        return v, sc[v]

    @staticmethod
    def to_ref(H, x, y):
        p = cv2.perspectiveTransform(np.float32([[[x, y]]]), H)[0, 0]
        return float(p[0]), float(p[1])

    def in_zone(self, view, zone, x, y):
        m = self.masks[view][zone]; xi, yi = int(round(x)), int(round(y))
        return 0 <= yi < m.shape[0] and 0 <= xi < m.shape[1] and bool(m[yi, xi])

    def desk_blob(self, frame, H, view):
        """Наибольшее цветовое пятно на столе относительно пустой комнаты (px в координатах эталона)."""
        if view not in self.empty: return None
        w = cv2.warpPerspective(frame, H, (464, 848))
        lab = lambda im: cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32), (7, 7), 0)
        d = np.sqrt(((lab(w) - lab(self.empty[view])) ** 2).sum(2))
        b = ((d > 18) & self.masks[view]["desk"]).astype(np.uint8)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(b)
        return int(st[1:, 4].max()) if n > 1 else 0


ONNX_PATH = pathlib.Path(__file__).resolve().parent / "assets" / "onnx" / "dfine_s.onnx"   # export_onnx.py; те же веса D-FINE-S


class Detector:
    """D-FINE-S. Бэкенд DET_BACKEND (по умолчанию onnx): auto (OpenVINO GPU fp32 → onnxruntime CPU → torch), ov-gpu, ov-cpu, onnx, torch.
    Все бэкенды дают те же детекции, что torch (проверено: 438/438 на 30 кадрах, Δscore 0); ускорение CPU ×1.5, iGPU ×8."""
    def __init__(self):
        threads = int(os.environ.get("OMP_NUM_THREADS") or max(1, os.cpu_count() - 2))
        torch.set_num_threads(threads)
        self.p = AutoImageProcessor.from_pretrained(DET_REPO)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = AutoModelForObjectDetection.from_pretrained(DET_REPO).to(self.dev).eval()
        self.id2label = self.m.config.id2label; self.run_np = None; self.backend = "torch"
        want = os.environ.get("DET_BACKEND", "onnx")       # onnx: 5 параллельных ≤107 с, 0.7 ГБ/процесс; ov-gpu быстрее, но ест всю RAM ноутбука
        if want != "torch" and self.dev == "cpu" and ONNX_PATH.exists():
            if want in ("auto", "ov-gpu", "ov-cpu"):
                try:
                    import openvino as ov
                    core = ov.Core(); dev = "GPU" if want != "ov-cpu" and "GPU" in core.available_devices else None
                    if want == "ov-cpu": dev = "CPU"
                    if dev:
                        cfg = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"}
                        if dev == "CPU": cfg["INFERENCE_NUM_THREADS"] = threads
                        c = core.compile_model(str(ONNX_PATH), dev, cfg)
                        self.run_np = lambda x: (lambda o: (o[0], o[1]))(c([x])); self.backend = f"openvino-{dev.lower()}"
                except Exception as e: print("[detector] openvino недоступен:", repr(e)[:160])
            if self.run_np is None and want in ("auto", "onnx"):
                import onnxruntime as ort
                so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
                sess = ort.InferenceSession(str(ONNX_PATH), so, providers=["CPUExecutionProvider"])
                self.run_np = lambda x: tuple(sess.run(None, {"pixel_values": x})); self.backend = "onnxruntime"
        if self.run_np is not None: del self.m; self.m = None
        print("[detector] бэкенд:", self.backend, flush=True)

    def __call__(self, frame_bgr):
        im = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.run_np is None:
            x = self.p(images=im, return_tensors="pt").to(self.dev)
            with torch.no_grad(): o = self.m(**x)
        else:
            from transformers.models.d_fine.modeling_d_fine import DFineObjectDetectionOutput
            lg, bx = self.run_np(self.p(images=im, return_tensors="np")["pixel_values"].astype(np.float32))
            o = DFineObjectDetectionOutput(logits=torch.from_numpy(np.asarray(lg)), pred_boxes=torch.from_numpy(np.asarray(bx)))
        r = self.p.post_process_object_detection(o, target_sizes=[im.shape[:2]], threshold=0.35)[0]
        return [(self.id2label[int(l)], float(s), [float(v) for v in b])
                for s, l, b in zip(r["scores"], r["labels"], r["boxes"])]



H_EVERY_S = 3.0          # гомографию пересчитываем раз в 3 с: камера неподвижна
CUT_CORR = 0.6           # резкое падение похожести кадра = склейка камер → пересчёт ракурса


class GeoState:
    """Держит ракурс и гомографию между пересчётами; SIFT только раз в H_EVERY_S или при склейке."""
    def __init__(self, geo):
        self.geo, self.view, self.H, self.inl, self.t_last, self.thumb = geo, None, None, 0, -1e9, None

    @staticmethod
    def _thumb(fr):
        g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (48, 88)).astype(np.float32).ravel()
        return (g - g.mean()) / (g.std() + 1e-6)

    def update(self, fr, t):
        th = self._thumb(fr)
        cut = self.thumb is not None and float(np.dot(th, self.thumb) / len(th)) < CUT_CORR
        if self.view is None or cut:
            self.view, _ = self.geo.best_view(fr); self.t_last = -1e9
        if t - self.t_last >= H_EVERY_S:
            H, inl = self.geo.homography(fr, self.view)
            if inl < 60 and not cut:                              # ракурс мог смениться плавно — перепроверяем
                v, _ = self.geo.best_view(fr)
                if v != self.view: self.view = v; H, inl = self.geo.homography(fr, v)
            self.H, self.inl, self.t_last, self.thumb = H, inl, t, th
        return self.view, self.H, self.inl

LONG_VIDEO_S = 60.0      # длинные ролики — 1 кадр/с (скорость), короткие — 2 кадра/с (точнее таймкод)


def sample_frames(path):
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    rate = 1.0 if dur > LONG_VIDEO_S else FPS_SAMPLE
    step = max(1, int(round(fps / rate))); i = 0
    while True:                                   # grab() без декодирования в BGR для пропускаемых кадров (×2 быстрее чтение)
        if i % step == 0:
            ok, fr = cap.read()
            if not ok: break
            yield i / fps, fr
        elif not cap.grab(): break
        i += 1


def analyze(path, geo, det, opt=None):
    """Сервисный путь = тот же, что в экспериментах: кадры → детекции + гомография + признаки → decide (A8) → JSON."""
    import decide as D
    from features import frame_features
    policy = os.environ.get("ML_POLICY", "sensors")      # sensors — эластичное ядро без знания комнаты; h5 — старая модель с зонами
    t0 = time.time(); rows = []; jpgs = []; gs = GeoState(geo) if policy == "h5" else None
    for t, fr in sample_frames(path):
        dets = [[l, round(s, 3), [round(v, 1) for v in b]] for l, s, b in det(fr)]
        if gs is not None:
            view, H, inl = gs.update(fr, t)
            rows.append({"t": round(t, 2), "view": view, "inl": inl, "H": None if H is None else H.tolist(), "dets": dets,
                         "feat": frame_features(fr, H, view, dets, geo)})
        else:
            rows.append({"t": round(t, 2), "view": None, "dets": dets})
        jpgs.append(cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
    if policy == "h5":
        o = dict(D.DEFAULT); o.update(opt or D.BEST)
        res = D.decide({"file_id": pathlib.Path(path).stem, "frames": rows}, o)
    else:
        from sensors_core import run_core
        ev = run_core(rows, jpgs, pathlib.Path(path).stem)
        res = {"violations": [{"type_code": e["type"], "timecode_sec": e["t"], "why_seen": e.get("why", ""), "fallback": e.get("fallback"), "t0": e.get("t0"), "t1": e.get("t1")} for e in ev]}
    viol = []
    mmss = lambda x: f"{int(x) // 60:02d}:{x % 60:04.1f}"
    WHO = {"2+": "у рабочего места кассира одновременно находятся два посетителя или больше",
           "вплотную": "рядом с кассиром стоит человек — вплотную к его рабочему месту",
           "не садится": "к столу кассира подходит человек и не садится на место клиента"}
    near = lambda t: min(range(len(rows)), key=lambda q: abs(rows[q]["t"] - t))
    for v in res["violations"]:
        r = RULES[v["type_code"]]; k = near(v["timecode_sec"]); t = rows[k]["t"]
        fb, sig = v.get("fallback"), v.get("why_seen", "")
        t0 = v["t0"] if v.get("t0") is not None else t; t1 = v["t1"] if v.get("t1") is not None else t
        f = {"t": mmss(t), "t0": mmss(t0), "t1": mmss(t1),
             "who": next((w for key, w in WHO.items() if key in sig), "у рабочего места кассира находится человек, не являющийся кассиром")}
        if fb == "geometry":
            desc = (f"Что видно: на {f['t']} ракурс камеры не удалось надёжно сопоставить с планом помещения, зональные проверки отключены.\n"
                    f"Что проверить: просмотрите запись целиком.\nНорма: {r['rule']}")
        else:
            desc = (f"Что видно: {r['seen'].format(**f)}\nПочему это важно: {r['risk']}\nЧто проверить: {r['check'].format(**f)}\n"
                    f"Норма: {r['rule']}")
        if fb == "visit":     # кадры охватывают визит: начало, завершение операции, конец
            ks = sorted({near(t0 + 1.0), k, near(t1 - 0.5)})
        else:
            ks = sorted({near(t + d) for d in (-1.5, 0.0, 1.5)})
        viol.append({"type": r["title"], "type_code": v["type_code"], "timecode_sec": t, "description": desc, "signal": sig,
                     "evidence_frames": [{"timestamp_sec": rows[q]["t"], "image_base64": base64.b64encode(jpgs[q]).decode()} for q in ks]})
    return {"status": "done", "verdict": "violation" if viol else "no_violation", "violations": viol,
            "processing_sec": round(time.time() - t0, 1), "debug": {"view": rows[0]["view"] if rows else None, "frames": len(rows)}}


def main():
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--videos", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default=""); a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    want = {x.zfill(2) for x in a.ids.split(",") if x}
    geo, det = Geometry(), Detector()
    for p in sorted(glob.glob(os.path.join(a.videos, "*.mov"))):
        fid = re.sub(r"\D", "", os.path.basename(p)).zfill(2)
        if fid in ("47", "48") or (want and fid not in want): continue    # дубликаты Х-4 и Х-5
        res = analyze(p, geo, det); res["job_id"] = fid
        json.dump(res, open(out / f"{fid}.json", "w", encoding="utf-8"), ensure_ascii=False)
        print(fid, res["verdict"], [(v["type_code"], v["timecode_sec"]) for v in res["violations"]], f"{res['processing_sec']} c", flush=True)


if __name__ == "__main__":
    main()
