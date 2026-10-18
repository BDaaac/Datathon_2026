"""Гигиена гомографии: H (кадр → эталон) правдоподобна, если нет отражения, анизотропия умеренная,
углы кадра переходят в выпуклый четырёхугольник, площадь меняется не более чем в 2 раза."""
import cv2, numpy as np

C = np.float32([[0, 0], [464, 0], [464, 848], [0, 848]]).reshape(-1, 1, 2)


def h_stats(H):
    if H is None: return None
    H = np.float32(H); A = H[:2, :2] / H[2, 2]
    s = np.linalg.svd(A, compute_uv=False); q = cv2.perspectiveTransform(C, H).reshape(-1, 2)
    return {"det": float(np.linalg.det(A)), "aniso": float(s[0] / max(s[1], 1e-9)), "convex": bool(cv2.isContourConvex(q.astype(np.float32))),
            "area": float(abs(cv2.contourArea(q)) / (464 * 848))}


def h_ok(H, aniso=3.0, area=(0.5, 2.0)):
    st = h_stats(H)
    return st is not None and st["det"] > 0 and st["aniso"] <= aniso and st["convex"] and area[0] <= st["area"] <= area[1]
