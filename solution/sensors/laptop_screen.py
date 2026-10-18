"""Датчик pc_unlocked: кассир отошёл от ноутбука, а экран остаётся светлым (открытое приложение).
Без зон/гомографии: рамка ноутбука — медиана лучших детекций laptop; «у места» = любой person в пределах 0.4 ширины ноутбука от рамки."""
import cv2, numpy as np

DEBUG = False


def _gap(a, b):
    dx = max(0, a[0] - b[2], b[0] - a[2]); dy = max(0, a[1] - b[3], b[1] - a[3])
    return (dx * dx + dy * dy) ** .5


def _laptop(cache):
    cand = []
    for fr in cache["frames"]:
        L = [(s, b) for l, s, b in fr["dets"] if l == "laptop" and s >= 0.5]
        if L: cand.append(max(L, key=lambda q: q[0]))
    if len(cand) < 3: return None
    # кластеризация по центру: берём центр с наибольшей суммой score
    cs = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for s, b in cand]); ss = np.array([s for s, b in cand])
    best = max(range(len(cand)), key=lambda i: ss[(np.abs(cs - cs[i]) < 30).all(1)].sum())
    m = (np.abs(cs - cs[best]) < 30).all(1)
    return np.median(np.array([b for (s, b), k in zip(cand, m) if k]), 0)


def run(cache, frames, ctx):
    lap = _laptop(cache)
    if lap is None: return []
    n = len(cache["frames"]); lw = lap[2] - lap[0]; lh = lap[3] - lap[1]
    near = []
    for fr in cache["frames"]:
        P = [b for l, s, b in fr["dets"] if l == "person" and s >= 0.5]
        near.append(any(_gap(lap, b) < 0.4 * lw for b in P))
    # экран: верх рамки
    x0, x1 = int(lap[0]), int(lap[2]); y0, y1 = int(lap[1]), int(lap[1] + 0.55 * lh)
    wh = []; mn = []
    for k in range(n):
        im = frames.get(k)
        if im is None: wh.append(None); mn.append(None); continue
        g = cv2.cvtColor(im[max(0, y0):y1, max(0, x0):x1], cv2.COLOR_BGR2GRAY)
        wh.append(float((g > 170).mean())); mn.append(float(g.mean()))
    t = [fr["t"] for fr in cache["frames"]]
    out = []
    W = max(3, int(round(3.0 * n / max(1e-6, t[-1] - t[0] + 1e-6))))      # кадров в 3 с
    for j in range(n - 1):
        if not (near[j] and not near[j + 1]) or sum(near[max(0, j - 2):j + 1]) < 2: continue
        away = near[j + 1:j + 1 + W]
        if len(away) < 0.6 * W or np.mean(away) > 0.35: continue
        pre = [wh[k] for k in range(max(0, j - 9), j + 1) if near[k] and wh[k] is not None]
        post = [wh[k] for k in range(j + 2, min(n, j + 1 + 2 * W)) if wh[k] is not None and not near[k]]
        if not pre or not post: continue
        pre, post = float(np.median(pre)), float(np.median(post))
        if DEBUG: print(f"  leave j={j} t={t[j]:.0f} pre={pre:.2f} post={post:.2f}")
        if pre >= 0.25 and post >= 0.25 and post >= 0.8 * pre:
            out.append({"type": "pc_unlocked", "t": float(t[j]), "score": post, "why": f"экран светлый {post:.2f} (был {pre:.2f}) после ухода"})
        break
    if DEBUG:
        print([int(x) for x in lap]); print("".join("N" if v else "." for v in near))
    return out
