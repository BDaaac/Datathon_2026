"""Признаки состояния по кадру, общие для сервиса (pipeline.analyze) и экспериментов (extract → decide).
Считаются из кадра, гомографии к эталону ракурса и детекций — без заранее посчитанных выровненных кадров.
"""
import cv2, numpy as np

W, Hh = 464, 848


def door_profile(warped, zone):
    """Профиль вертикальных краёв верхней половины створки (выше голов) — для датчика «дверь не вернулась»."""
    x0, x1 = min(p[0] for p in zone), max(p[0] for p in zone); y0, y1 = min(p[1] for p in zone), max(p[1] for p in zone)
    y1 = y0 + (y1 - y0) // 2
    g = cv2.cvtColor(warped[y0:y1, max(0, x0 - 10):x1 + 25], cv2.COLOR_BGR2GRAY).astype(np.float32)
    sx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)); sy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    p = ((sx > sy * 2) * sx).sum(0); p = np.convolve(p, np.ones(5) / 5, "same")
    return (p - p.mean()) / (p.std() + 1e-6)


def desk_blob(warped, empty, desk_mask):
    """Наибольшее цветовое пятно на столе относительно пустой комнаты (px в координатах эталона)."""
    if empty is None: return None
    lab = lambda im: cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32), (7, 7), 0)
    d = np.sqrt(((lab(warped) - lab(empty)) ** 2).sum(2))
    b = ((d > 18) & desk_mask).astype(np.uint8); b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(b)
    return int(st[1:, 4].max()) if n > 1 else 0


def screen_white(warped, H, dets):
    """Доля белого в верхней части рамки ноутбука (открытое приложение → светлый экран)."""
    L = [b for l, s, b in dets if l == "laptop" and s >= 0.4]
    if not L or H is None: return None
    b = max(L, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
    pts = np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[1] + 0.55 * (b[3] - b[1])], [b[0], b[1] + 0.55 * (b[3] - b[1])]]).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(pts, np.float32(H)).reshape(-1, 2); x0, y0 = q.min(0).astype(int); x1, y1 = q.max(0).astype(int)
    im = warped[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    return round(float((cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) > 170).mean()), 3) if im.size else None


def frame_features(frame, H, view, dets, geo):
    """Все признаки кадра одним вызовом. При H=None (геометрия ненадёжна) зонные признаки = None."""
    if H is None or view is None: return {"door": None, "blob": None, "scr": None}
    w = cv2.warpPerspective(frame, np.float32(H), (W, Hh))
    return {"door": [round(float(x), 2) for x in door_profile(w, geo.cfg_zones[view]["door"])],
            "blob": desk_blob(w, geo.empty.get(view), geo.masks[view]["desk"]),
            "scr": screen_white(w, H, dets)}


def desk_self_blobs(warped, person_masks, desk_mask, thr=18.0, ref_frac=0.25, ref_min=6):
    """Пятно на столе относительно эталона ИЗ САМОГО РОЛИКА (не нужен эталон пустой комнаты своего ракурса).
    Эталон пикселя — медиана по первой четверти ролика (кадры, где пиксель не закрыт человеком); пиксели, которые почти всегда закрыты, не сравниваются.
    warped: список кадров в координатах эталона (BGR), person_masks: список bool-масок людей. → список px наибольшего пятна."""
    ys, xs = np.where(desk_mask)
    if not len(warped) or not len(ys): return [None] * len(warped)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    lab = lambda im: cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32), (7, 7), 0)[y0:y1, x0:x1]
    L = np.stack([lab(w) for w in warped]); occ = np.stack([m[y0:y1, x0:x1] for m in person_masks])
    n0 = min(len(warped), max(ref_min, int(ref_frac * len(warped))))          # эталон — начало ролика, пока на столе ничего не оставили
    Lm = np.where(occ[:n0, ..., None], np.nan, L[:n0])
    ref = np.nanmedian(Lm, 0); seen = (~occ[:n0]).sum(0) >= 2
    dm = desk_mask[y0:y1, x0:x1] & seen & ~np.isnan(ref).any(2)
    out = []
    for k in range(len(warped)):
        d = np.sqrt(((L[k] - np.nan_to_num(ref)) ** 2).sum(2))
        b = ((d > thr) & dm & ~occ[k]).astype(np.uint8); b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(b)
        out.append(int(st[1:, 4].max()) if n > 1 else 0)
    return out
