"""Датчик valuables_on_desk: кассир ушёл (пропал в конце ролика), а у его места на столе осталась новая вещь.
Без зон/эталонов: «место» = медианная рамка кассира в позах с самым низким нижним краем; вещь = трек детекций мелких
предметов вблизи места, который (1) держится до конца ролика, (2) виден после ухода кассира, (3) его участок в
начале ролика выглядит иначе (не стационарная принадлежность комнаты)."""
import numpy as np, cv2

ITEM = {"book", "cell phone", "remote", "keyboard", "handbag", "backpack", "suitcase", "laptop", "mouse", "cup", "bottle",
        "bowl", "sandwich", "donut", "cake", "banana", "apple", "scissors", "knife", "vase"}
MIN_GONE_FR = 2        # кадров без кассира в конце
P = dict(score=0.30, min_cnt=4, diff=18.0, diff_base=35.0, post_frac=0.6, zl=0.8, zr=1.2)   # zl/zr: зона стола влево/вправо в ширинах кассира
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
    """разность участков, инвариантная к яркости и сдвигу на пару пикселей"""
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
    cb = {k: np.array(b) for k in range(n) for i, b in per[k] if i == cid}
    if len(cb) < 4: return []
    seen = sorted(cb); L = seen[-1]; post = list(range(L + 1, n))
    if len(post) < MIN_GONE_FR: return []
    ys = np.array([cb[k][3] for k in seen]); sel = [k for k in seen if cb[k][3] >= np.percentile(ys, 60)]
    seat = np.median([cb[k] for k in sel], 0); w, h = seat[2] - seat[0], seat[3] - seat[1]
    zone = [seat[0] - P["zl"] * w, seat[1] + .45 * h, seat[2] + P["zr"] * w, seat[3] + .25 * h]
    # конечные объекты: детекции в последних кадрах
    cand = []
    for l, sc, b in [d for k in range(max(0, L - 2), n) for d in F[k]["dets"]]:
        if l not in ITEM or sc < P["score"]: continue
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if not (zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]): continue
        if any(iou(b, c[0]) > .5 for c in cand): continue
        cand.append((b, l, sc))
    ref = [frames.get(k) for k in range(min(2, n))]; ref = [r for r in ref if r is not None]; last = frames.get(n - 1)
    if not ref or last is None: return []
    out = []
    for fb, lab, sc in cand:
        pres = [k for k in range(n) if any(l == lab or True for l, s_, b in F[k]["dets"] if l in ITEM and s_ >= P["score"] and iou(b, fb) > .5)]
        npost = sum(k in post for k in pres)
        if len(pres) < P["min_cnt"] or npost < 1 and pres[-1] < L: continue
        # первое устойчивое появление (разрыв <=2 кадров), считая с конца
        k0 = pres[-1]
        for k in reversed(pres[:-1]):
            if k0 - k <= 5: k0 = k
            else: break
        labs = [l for k in pres for l, s_, b in F[k]["dets"] if iou(b, fb) > .5]
        if max(labs.count("laptop"), labs.count("keyboard")) > .6 * len(labs): continue   # принадлежности ноутбука/клавиатуры
        pl = _patch(last, fb); d = min(_pdiff(_patch(r, fb), pl) for r in ref)
        if DEBUG and fb[0]>150 and fb[0]<235: print(pres, ctx["fid"], [int(v) for v in fb], lab, "cnt", len(pres), "first", k0, "post", npost, "diff", round(d, 1))
        if k0 >= L - 1: continue                      # появилось только после ухода — не «оставлено»
        if d < (P["diff_base"] if k0 <= 2 else P["diff"]): continue
        out.append({"type": "valuables_on_desk", "t": float(F[k0]["t"]), "score": float(sc) * d, "why": f"предмет {lab} у места кассира, d={d:.0f}"})
    out.sort(key=lambda e: -e["score"])
    return out[:1]
