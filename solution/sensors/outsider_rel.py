"""Посторонний у рабочего места кассира — относительно людей (без зон/гомографии).
Визитёр = человек не-кассир, не дубль кассира, не дубль другого визитёра, на расстоянии < DMAX ростов кассира от кассира
(отсекает отражения в стекле и людей за дверью). Событие: >= 2 визитёра одновременно >= MIN_S секунд,
либо стоящий (рамка >= TALL от роста кассира) визитёр вплотную к кассиру (< NEAR) >= MIN_S секунд."""
import numpy as np

DMAX, NEAR, TALL, MIN_S = 0.95, 0.55, 0.85, 2.0
TALL_FAR, FAR = 0.95, 1.5
SV_MIN, SV_STAT = 3.5, 4.5
MIN_VIS_H = 0.35      # визитёр не ниже этой доли роста кассира (отсекает ложные мелкие рамки «человека»)


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def analyse(cache, ctx):
    cid, per = ctx["cid"], ctx["per"]
    cbs = [b for p in per for i, b in p if i == cid]
    if not cbs: return []
    # якорь кассира: медианная рамка первых 30% его кадров (кассир сидит на месте; трекер может терять его при вставании)
    a = np.median(np.array(cbs[: max(5, len(cbs) * 3 // 10)]), 0); H = float(a[3] - a[1])
    ac = np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2]); rows = []
    for k, p in enumerate(per):
        cand = [(np.linalg.norm(np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) - ac) / H, b) for i, b in p]
        cand.sort(key=lambda x: x[0])
        if not cand or cand[0][0] > 0.6: rows.append((cache["frames"][k]["t"], 0, 0, H)); continue
        cb = [cand[0][1]]; vs = []
        for d, b in cand[1:]:
            if any(_iou(b, q) > 0.3 for q in cb + vs): continue
            h = (b[3] - b[1]) / H
            if h < MIN_VIS_H: continue
            if not (d < DMAX or (h >= TALL_FAR and d < FAR)): continue
            vs.append(b)
        near = 0
        for b in vs:
            d = np.linalg.norm(np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) - ac) / H
            if (b[3] - b[1]) >= TALL * H and d < NEAR: near += 1
        rows.append((cache["frames"][k]["t"], len(vs), near, H))
    return rows


def _first(rows, col, pred):
    s = None
    for r in rows:
        if pred(r[col]):
            if s is None: s = r[0]
            if r[0] - s >= MIN_S: return s
        else: s = None
    return None


def _first_mid(rows, col, pred):
    """Первый период (>= MIN_S), где pred(rows[col]); таймкод — середина периода."""
    s = None
    for j, r in enumerate(rows):
        ok = pred(r[col])
        if ok and s is None: s = r[0]
        if s is not None and (not ok or j == len(rows) - 1):
            e = r[0]
            if e - s >= MIN_S: return (s + e) / 2
            s = None
    return None


def short_visit(cache, ctx):
    """Визитёр у кассира, который не садится: в зоне < DMAX_S ростов >= SV_MIN с, пригнут (hmin < 0.75 роста кассира),
    но неподвижного отрезка (сидение) короче SV_STAT с."""
    cid, per = ctx["cid"], ctx["per"]
    cbs = [b for p in per for i, b in p if i == cid]
    if not cbs: return None
    a = np.median(np.array(cbs[: max(5, len(cbs) * 3 // 10)]), 0); H = float(a[3] - a[1])
    ac = np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2]); tr = {}
    for k, p in enumerate(per):
        for i, b in p:
            if i != cid: tr.setdefault(i, []).append((cache["frames"][k]["t"], np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]), (b[3] - b[1]) / H))
    best = None
    for l in tr.values():
        l = [x for x in l if np.linalg.norm(x[1] - ac) / H < DMAX]
        if len(l) < 4 or l[-1][0] - l[0][0] < SV_MIN or min(x[2] for x in l) >= 0.75 or l[-1][0] > cache["frames"][-1]["t"] - 2: continue
        st = 0; j = 0
        for m in range(len(l)):
            while np.linalg.norm(l[m][1] - l[j][1]) > 0.15 * H: j += 1
            st = max(st, l[m][0] - l[j][0])
        if st < SV_STAT and (best is None or l[0][0] < best): best = l[-1][0] + 1.5
    return best


def run(cache, frames, ctx):
    rows = analyse(cache, ctx); ev = []
    t = _first_mid(rows, 1, lambda n: n >= 2)
    if t is not None: ev.append({"type": "outsider", "t": t, "score": 1.0, "why": "2+ посетителей у кассира"})
    t = _first_mid(rows, 2, lambda n: n >= 1)
    if t is not None: ev.append({"type": "outsider", "t": t, "score": 0.6, "why": "стоящий вплотную к кассиру"})
    t = short_visit(cache, ctx) if rows else None
    if t is not None: ev.append({"type": "outsider", "t": t, "score": 0.5, "why": "визитёр у стола не садится"})
    return sorted(ev, key=lambda e: -e["score"])[:1]
