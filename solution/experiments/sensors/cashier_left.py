"""Кассир покинул рабочее место во время обслуживания посетителя (пост. НБРК № 231, п.22) — без знания комнаты.
Место кассира = медианная рамка его трека в первые 30% кадров (кассир сидит). Кадр «ушёл при посетителе»:
у места кассира никого нет (ближайший человек дальше 0.6 роста кассира), а в кадре есть посетитель
(другой человек, не дубль рамки, не дальше 1 роста от места кассира). Событие — такая ситуация держится >= MIN_S секунд
после того, как кассир и посетитель провели вместе >= TOGETHER_S секунд. Таймкод — момент ухода."""
import numpy as np

NEAR_SEAT, VIS_MAX, MIN_S, TOGETHER_S = 0.6, 1.0, 2.0, 3.0


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def run(cache, frames, ctx):
    cid, per = ctx["cid"], ctx["per"]
    cbs = [b for p in per for i, b in p if i == cid]
    if not cbs: return []
    a = np.median(np.array(cbs[: max(5, len(cbs) * 3 // 10)]), 0); H = float(a[3] - a[1])
    if H <= 0: return []
    ac = np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2])
    ctr = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    tog_t, start = 0.0, None; dt = (cache["frames"][-1]["t"] - cache["frames"][0]["t"]) / max(1, len(cache["frames"]) - 1)
    for k, f in enumerate(cache["frames"]):
        ppl = []
        for l, s, b in f["dets"]:
            if l == "person" and s >= 0.5 and all(_iou(b, q) < 0.3 for q in ppl): ppl.append(b)
        d = sorted(np.linalg.norm(ctr(b) - ac) / H for b in ppl)
        seat = bool(d) and d[0] <= NEAR_SEAT
        vis = sum(1 for x in (d[1:] if seat else d) if x <= VIS_MAX)
        if seat and vis: tog_t += dt
        if tog_t >= TOGETHER_S and not seat and vis:
            if start is None: start = f["t"]
            if f["t"] - start >= MIN_S:
                return [{"type": "cashier_left", "t": float(start), "score": 1.0, "why": "кассира нет на месте, посетитель остаётся"}]
        else:
            start = None
    return []
