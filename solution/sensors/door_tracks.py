"""Датчик door_open: после появления людей на месте входа остаётся устойчивое изменение кадра (дверь распахнута).

Без зон/эталонов. Базис = кадры ДО первого появления человека (нужен хотя бы один пустой кадр).
Для каждого следующего кадра: маска изменений (после коррекции общей яркости), люди вырезаются.
Дверь = самый большой компонент изменений, который (а) не мал относительно кадра,
(б) начинается выше верха первой рамки человека (дальняя от камеры стена/вход),
(в) держится до конца ролика (медиана площади во второй половине >= порога) и ролик идёт >= MIN_SPAN с после появления.
t = момент возникновения изменения + LAG.
"""
import cv2, numpy as np

SC = 0.5            # даунскейл
MIN_AREA = 0.006    # доля площади кадра (0.010 → 0.006: на другом ракурсе дверь занимает меньшую часть кадра; без потерь на исходных)
MIN_SPAN = 4.0      # с после onset
LAG = 2.0
THR = 30
PCONF = 0.3


def run(cache, frames, ctx):
    F = cache["frames"]; n = len(F)
    if n < 6: return []
    P = [[b for l, s, b in f["dets"] if l == "person" and s > PCONF] for f in F]
    k0 = next((k for k, p in enumerate(P) if p), None)
    if k0 is None or k0 < 1: return []
    b0 = P[k0][0]
    def g(k):
        im = frames.get(k)
        if im is None: return None
        im = cv2.resize(im, None, fx=SC, fy=SC, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.int16)
    bl = [x for x in (g(k) for k in range(max(0, k0 - 5), k0)) if x is not None]
    if not bl: return []
    base = np.median(bl, 0); H, W = base.shape; area = H * W
    ker = np.ones((3, 3), np.uint8)
    res = []  # (t, area_frac)
    for k in range(k0, n):
        x = g(k)
        if x is None: continue
        m = np.zeros(base.shape, bool)
        for q in P[k]:
            m[max(0, int((q[1] - 10) * SC)):int((q[3] + 10) * SC), max(0, int((q[0] - 10) * SC)):int((q[2] + 10) * SC)] = True
        if m.mean() > 0.5: continue
        d = x - base; gain = np.median(d[~m][::5])
        dd = ((np.abs(d - gain) > THR) & ~m).astype(np.uint8)
        dd = cv2.morphologyEx(dd, cv2.MORPH_OPEN, ker)
        nc, lab, st, _ = cv2.connectedComponentsWithStats(dd)
        best = 0
        for i in range(1, nc):
            xx, yy, ww, hh, a = st[i]
            if yy <= b0[1] * SC and a > best: best = a
        res.append((F[k]["t"], best / area))
    if len(res) < 4: return []
    ts = np.array([r[0] for r in res]); av = np.array([r[1] for r in res])
    on = np.where(av >= MIN_AREA)[0]
    if not len(on): return []
    for i in on:
        t_on = ts[i]
        tail = av[(ts >= t_on)]
        if ts[-1] - t_on < MIN_SPAN: break
        half = av[ts >= (t_on + ts[-1]) / 2]
        if len(half) and np.median(half) >= MIN_AREA and (tail >= MIN_AREA * 0.6).mean() >= 0.8:
            return [{"type": "door_open", "t": float(t_on + LAG), "score": float(np.median(half)), "why": f"устойчивое изменение у входа с t={t_on:.1f}"}]
    return []
