"""Стенд датчиков «без знания комнаты».

Датчик — модуль sensors/<name>.py с функцией
    run(cache, frames, ctx) -> list[{"type": <код из rules.json>, "t": сек, "score": float, "why": str}]
  cache  — кэш ролика БЕЗ геометрии (cache_nogeo/<fid>.json): frames[k] = {t, dets: [[label, score, [x0,y0,x1,y1]]...]}, координаты кадра 464×848
  frames — FrameStore: frames.get(k) → BGR-кадр (np.uint8, 848×464) в момент cache["frames"][k]["t"]; len(frames)
  ctx    — {"cid": id трека кассира, "per": [[(track_id, box), ...] по кадрам], "fid": "02"}  (трекер — decide.cashier_track)
Камера неподвижна: всё считается в пикселях кадра и относительно людей/предметов — без зон, эталонов и гомографии.

  python bench.py --sensor door_tracks                 # поштучная оценка датчика по его типу (только dev)
  python bench.py --sensor door_tracks --integrate     # сборка с H5 в трёх режимах (своя / LOVO / чужая комната)
  python bench.py --sensor a,b,c --integrate --no-fallback   # без запасного вердикта «визит»
  python bench.py --sensor a,b,c --integrate --sensors-first  # датчики главнее правил с зонами (правила — только если датчики молчат)
"""
import argparse, importlib, json, os, pathlib, sys
import cv2, numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import decide as D
from evaluate import load_labels, score, report

MODES = {"своя": "cache_adapt", "LOVO": "cache_lovo", "чужая": "cache_nogeo"}
PRIO = ["door_open", "pc_unlocked", "valuables_on_desk", "outsider", "phone", "personal_items", "procedure"]


class FrameStore:
    def __init__(self, fid):
        self.d = HERE / os.environ.get("SENSOR_FRAMES", "frames") / fid; self.n = len(list(self.d.glob("*.jpg"))); self._c = {}
    def __len__(self): return self.n
    def get(self, k):
        if k not in self._c:
            p = self.d / f"{k:04d}.jpg"
            self._c[k] = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR) if p.exists() else None
            if len(self._c) > 400: self._c.pop(next(iter(self._c)))
        return self._c[k]


def nogeo(c):
    c = json.loads(json.dumps(c))
    for f in c["frames"]: f["H"] = None; f["inl"] = 0; f["feat"] = {"door": None, "blob": None, "scr": None}
    return c


def run_sensors(names, fid):
    c = nogeo(json.load(open(HERE / os.environ.get("SENSOR_CACHE", "cache_adapt") / f"{fid}.json"))); c["file_id"] = fid
    D.CASHIER_RULE = "earliest"; cid, per = D.cashier_track(c)
    fs = FrameStore(fid); ctx = {"cid": cid, "per": per, "fid": fid}; out = []
    for n in names:
        try: out += [dict(e, sensor=n) for e in importlib.import_module(f"sensors.{n}").run(c, fs, ctx)]
        except Exception as ex: print(f"[{n}] {fid}: ошибка {ex!r}", file=sys.stderr)
    return out


def sensor_report(names, gt):
    targets = {}
    for fid in sorted(gt):
        ev = run_sensors(names, fid); g = gt[fid]; gtypes = {t for t, _ in g["events"]}
        fired = sorted({e["type"] for e in ev})
        for T in fired | gtypes if False else set(fired) | gtypes:
            s = targets.setdefault(T, {"tp": [], "fn": [], "fp_clean": [], "fp_other": [], "dt": []})
            hit = [e for e in ev if e["type"] == T]
            if T in gtypes and hit:
                gs = [sec for t, sec in g["events"] if t == T][0]; e = min(hit, key=lambda e: abs(e["t"] - gs))
                s["tp"].append(f"{fid}@{e['t']:.0f}(ист {gs:.0f})"); s["dt"].append(abs(e["t"] - gs))
            elif T in gtypes: s["fn"].append(fid)
            elif hit: (s["fp_clean"] if not g["violation"] else s["fp_other"]).append(f"{fid}@{hit[0]['t']:.0f}")
    for T, s in sorted(targets.items()):
        if not s["tp"] and not s["fp_clean"] and not s["fp_other"]: continue
        dt = s["dt"]
        print(f"== {T}: найдено {len(s['tp'])}/{len(s['tp']) + len(s['fn'])}, ±2с {sum(x <= 2 for x in dt)}/{len(dt)} | "
              f"ложно на чистых {len(s['fp_clean'])}/15, на чужих нарушениях {len(s['fp_other'])}")
        print("   найдено:", s["tp"]); print("   пропуск:", s["fn"]); print("   ложно (чистые):", s["fp_clean"]); print("   ложно (другие нарушения):", s["fp_other"])


def pick(ev, rank, med):
    """Главное событие: prio — по фиксированному приоритету типов; score — по силе доказательства (score / типичный score датчика); time — самое раннее."""
    pr = lambda e: PRIO.index(e["type"]) if e["type"] in PRIO else 99
    if rank == "score": return max(ev, key=lambda e: (float(e.get("score", 0)) / med.get(e.get("sensor"), 1.0), -pr(e)))
    if rank == "time": return min(ev, key=lambda e: (e["t"], pr(e)))
    return sorted(ev, key=lambda e: (pr(e), -e.get("score", 0)))[0]


def visitors(f):
    """Сколько разных людей в кадре (рамки person ≥0.5 без дублей) — без знания комнаты."""
    out = []
    for l, sc, b in f["dets"]:
        if l == "person" and sc >= 0.5 and all(D.iou(b, q) < 0.3 for q in out): out.append(b)
    return len(out)


def integrate(names, gt, no_fallback, quiet, sensors_first=False, pure=False, visit_s=None, rank="prio"):
    opt = dict(D.DEFAULT); opt.update(D.BEST)
    if visit_s is not None: opt["visit_s"] = str(visit_s)
    sens = {fid: run_sensors(names, fid) for fid in gt} if names else {fid: [] for fid in gt}
    n_clean = sum(not g["violation"] for g in gt.values())
    allsc = {}
    for ev in sens.values():
        for e in ev: allsc.setdefault(e["sensor"], []).append(float(e.get("score", 0)))
    med = {k: float(np.median(v)) or 1.0 for k, v in allsc.items()}      # типичная сила срабатывания каждого датчика
    for mode, cdir in MODES.items():
        if not all((HERE / cdir / f"{fid}.json").exists() for fid in gt): continue
        preds = {}
        for fid in gt:
            c = json.load(open(HERE / cdir / f"{fid}.json")); c["file_id"] = fid
            base = D.decide(c, opt); v = [] if pure else base["violations"]
            fb = not v or v[0].get("fallback") or pure
            if (fb or sensors_first) and sens[fid]:
                e = pick(sens[fid], rank, med)
                v = [{"type_code": e["type"], "timecode_sec": e["t"]}]
            elif fb and no_fallback: v = []
            elif pure and visit_s is not None:                         # только датчики + запасной «визит ≥ visit_s» если датчики молчат
                import sensors_core
                e = sensors_core.visit_event(c["frames"], visit_s)          # та же логика, что в сервисе
                v = [{"type_code": "procedure", "timecode_sec": e["t"]}] if e else []
            preds[fid] = {"verdict": "violation" if v else "no_violation", "violations": v}
        s = score(gt, preds); report(mode, s, n_clean)
        if not quiet and mode == "чужая": print("\n".join("   " + r for r in s["rows"]))


def main():
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--sensor", default=""); ap.add_argument("--integrate", action="store_true")
    ap.add_argument("--no-fallback", action="store_true"); ap.add_argument("--quiet", action="store_true"); ap.add_argument("--sensors-first", action="store_true"); ap.add_argument("--split", default="dev"); ap.add_argument("--pure", action="store_true"); ap.add_argument("--visit-s", type=float); ap.add_argument("--rank", default="prio"); a = ap.parse_args()
    names = [x for x in a.sensor.split(",") if x]; gt = load_labels(a.split)
    if not (HERE / "cache_nogeo").exists() or not any((HERE / "cache_nogeo").glob("*.json")):
        (HERE / "cache_nogeo").mkdir(exist_ok=True)
        for f in (HERE / "cache_adapt").glob("*.json"): json.dump(nogeo(json.load(open(f))), open(HERE / "cache_nogeo" / f.name, "w"))
    if a.integrate: integrate(names, gt, a.no_fallback, a.quiet, a.sensors_first, a.pure, a.visit_s, a.rank)
    else: sensor_report(names, gt)


if __name__ == "__main__":
    main()
