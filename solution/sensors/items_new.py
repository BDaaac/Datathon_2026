"""Датчик personal_items: рядом с кассиром (в его рабочей зоне) появляется бытовой предмет (сумка, стакан, бутылка, еда),
которого не было в начале ролика, и держится >= MIN_SEC. Без зон/эталонов: зона = рамка кассира, расширенная относительно
его размера; «новизна» = участок в первых кадрах выглядит иначе / предмет не детектируется в первых кадрах;
вещи посетителя отсекаются (ближе к другому человеку, чем к кассиру)."""
import numpy as np, cv2

ITEM = {"handbag", "backpack", "suitcase", "cup", "bottle", "bowl", "sandwich", "banana", "apple", "donut", "cake",
        "wine glass", "vase", "orange", "pizza", "hot dog", "broccoli", "carrot"}
P = dict(score=0.30, min_sec=5.0, near=1.0, diff=20.0, new_after=1.5, gap=8, min_n=4, zl=0.9, zr=1.3)   # gap/min_sec подобраны на искажённых ракурсах A/B без потерь на исходных
DEBUG = False


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def _patch(fr, b, pad=3):
    x0, y0, x1, y1 = [int(v) for v in b]; H, W = fr.shape[:2]
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    return cv2.GaussianBlur(cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)


def _pdiff(a, b):
    if a.shape != b.shape or min(a.shape) < 6: return 0.0
    best = 1e9
    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            ys, xs = slice(max(0, dy), a.shape[0] + min(0, dy)), slice(max(0, dx), a.shape[1] + min(0, dx))
            yb, xb = slice(max(0, -dy), b.shape[0] + min(0, -dy)), slice(max(0, -dx), b.shape[1] + min(0, -dx))
            x, y = a[ys, xs], b[yb, xb]
            best = min(best, float(np.abs((x - x.mean()) - (y - y.mean())).mean()))
    return best


def run(cache, frames, ctx):
    F = cache["frames"]; n = len(F); cid = ctx["cid"]; per = ctx["per"]
    if n < 8: return []
    cb = {k: np.array(b, float) for k in range(n) for i, b in per[k] if i == cid}
    if len(cb) < 4: return []
    seen = sorted(cb)
    # «рабочая поза» кассира: медиана рамок с самым низким нижним краем
    ys = np.array([cb[k][3] for k in seen]); sel = [k for k in seen if cb[k][3] >= np.percentile(ys, 50)]
    seat = np.median([cb[k] for k in sel], 0); w, h = seat[2] - seat[0], seat[3] - seat[1]
    zone = [seat[0] - P["zl"] * w, seat[1] + .25 * h, seat[2] + P["zr"] * w, seat[3] + .35 * h]
    dt = (F[-1]["t"] - F[0]["t"]) / max(1, n - 1)
    t_end = F[-1]["t"]
    tracks = []
    for k in range(n):
        for l, s, b in F[k]["dets"]:
            if l not in ITEM or s < P["score"]: continue
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if not (zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]): continue
            for t in tracks:
                if iou(b, t["b"]) > .4: t["k"].append(k); t["b"] = b; t["s"].append(s); break
            else: tracks.append({"b": b, "k": [k], "s": [s]})
    ref = [frames.get(k) for k in range(min(2, n))]; ref = [r for r in ref if r is not None]
    if not ref: return []
    out = []
    for t in tracks:
        ks = t["k"]; fb = t["b"]
        # устойчивый отрезок (разрыв <=4 кадра) — самый длинный
        runs = [[ks[0]]]
        for k in ks[1:]:
            if k - runs[-1][-1] <= P["gap"]: runs[-1].append(k)
            else: runs.append([k])
        r = max(runs, key=lambda r: r[-1] - r[0])
        dur = F[r[-1]]["t"] - F[r[0]]["t"]; nn = len(r)
        newt = F[r[0]]["t"]
        # вещи посетителя: ближе к другому человеку, чем к кассиру
        cx, cy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
        dc, do = [], []
        for k in r:
            for i, b in per[k]:
                d = np.hypot((b[0] + b[2]) / 2 - cx, (b[1] + b[3]) / 2 - cy)
                (dc if i == cid else do).append((k, d))
        mine = sum(1 for k in r if k in cb)
        theirs = 0
        for k in r:
            dd = {i: np.hypot((b[0] + b[2]) / 2 - cx, (b[1] + b[3]) / 2 - cy) for i, b in per[k]}
            if dd and min(dd, key=dd.get) != cid: theirs += 1
        if theirs > .4 * len(r) or ks[0] <= 2: continue   # вещь посетителя или была с начала ролика
        # новизна: участок в начале ролика выглядит иначе (или предмет там не детектируется)
        last = frames.get(r[-1])
        if last is None: continue
        d = min(_pdiff(_patch(x, fb), _patch(last, fb)) for x in ref)
        if DEBUG: print(ctx["fid"], t["b"].__class__ and [int(v) for v in fb], "run", r[0], r[-1], "dur", round(dur, 1), "n", nn, "diff", round(d, 1), "theirs", theirs)
        if dur < P["min_sec"] or nn < P["min_n"] or d < P["diff"]: continue
        out.append({"type": "personal_items", "t": float(newt), "score": float(np.mean(t["s"])) * min(dur, 10), "why": f"предмет у кассира {dur:.0f}с d={d:.0f}"})
    out.sort(key=lambda e: -e["score"])
    return out[:1]
