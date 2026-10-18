"""Эластичное ядро: датчики состояний БЕЗ знания комнаты (пиксели кадра, относительно людей и предметов).
Та же логика, что в bench.py: трек кассира → датчики sensors/* → главное событие по приоритету типов.
dev 0.65 (ложных 0/15), отложенные V2 0.58 — против 0.39/0.34 у H5 в незнакомой комнате.
Открытый класс (FALLBACK_VISIT_S, по умолчанию 10 с; 0 — выключить): если датчики молчат, но был визит — «требуется проверка».
"""
import importlib, os
import cv2, numpy as np
import decide as D

SENSORS = ["door_tracks", "laptop_screen", "outsider_rel", "desk_left", "items_new"]
PRIO = ["door_open", "pc_unlocked", "valuables_on_desk", "outsider", "phone", "personal_items", "procedure"]


class JpegFrames:
    """Кадры по запросу из JPEG-байтов (интерфейс FrameStore из bench.py)."""
    def __init__(self, jpgs): self.j = jpgs; self._c = {}
    def __len__(self): return len(self.j)
    def get(self, k):
        if not 0 <= k < len(self.j): return None
        if k not in self._c:
            self._c[k] = cv2.imdecode(np.frombuffer(self.j[k], np.uint8), cv2.IMREAD_COLOR)
            if len(self._c) > 400: self._c.pop(next(iter(self._c)))
        return self._c[k]


def run_core(rows, jpgs, fid="video"):
    """rows: [{t, dets}] → события [{type, t, score, why, sensor}], главное первым."""
    cache = {"file_id": fid, "frames": [{"t": r["t"], "dets": r["dets"], "H": None, "inl": 0, "view": None,
                                         "feat": {"door": None, "blob": None, "scr": None}} for r in rows]}
    D.CASHIER_RULE = "earliest"; cid, per = D.cashier_track(cache)
    frames = JpegFrames(jpgs); ctx = {"cid": cid, "per": per, "fid": fid}; ev = []
    for n in SENSORS:
        # A missing or broken sensor must fail the job, never produce a false clean result.
        ev += [dict(e, sensor=n) for e in importlib.import_module(f"sensors.{n}").run(cache, frames, ctx)]
    ev.sort(key=lambda e: (PRIO.index(e["type"]) if e["type"] in PRIO else 99, -float(e.get("score", 0))))
    out, seen = [], set()                       # по одному событию на тип, главное — первым
    for e in ev:
        if e["type"] not in seen: seen.add(e["type"]); out.append(e)
    if not out:                                 # открытый класс: датчики молчат, но был визит с операцией → «требуется проверка»
        e = visit_event(cache["frames"], float(os.environ.get("FALLBACK_VISIT_S", "10")))
        if e: out = [e]
    return out


def visit_event(fr, visit_s, rule=None):
    """Окно визита (кадры с ≥2 людьми). Таймкод: rule=end5 — завершение операции (конец визита − 5 с), mid — середина."""
    rule = rule or os.environ.get("FALLBACK_T", "end5")
    if visit_s <= 0 or not fr: return None
    dt = (fr[-1]["t"] - fr[0]["t"]) / max(1, len(fr) - 1) if len(fr) > 1 else 0.5
    two = [k for k, f in enumerate(fr) if _people(f) >= 2]
    if len(two) * dt < visit_s: return None
    t0, t1 = fr[two[0]]["t"], fr[two[-1]]["t"]
    tc = max(t0, t1 - 5.0) if rule == "end5" else fr[two[len(two) // 2]]["t"]
    return {"type": "procedure", "t": tc, "t0": t0, "t1": t1, "score": 0.0, "why": f"визит {len(two) * dt:.0f} с", "fallback": "visit"}


def _people(f):
    """Число людей в кадре (рамки person ≥0.5 без дублей) — без знания комнаты."""
    out = []
    for l, s, b in f["dets"]:
        if l == "person" and s >= 0.5 and all(D.iou(b, q) < 0.3 for q in out): out.append(b)
    return len(out)
