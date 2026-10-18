"""Лёгкая часть: правила поверх кэша extract.py. Каждый флаг — одно изменение для A/B.

  python decide.py --variant items=new,door=on --out runs/ab_items_new
Флаги (по умолчанию = v0):
  items  = v0 | new        v0: любой «предмет» в зоне стола 3 кадра подряд; new: предмет, которого не было в начале ролика
  phone  = v0 | off
  valu   = v0 | off        ценности на столе после ухода (пятно относительно пустой комнаты)
  door   = off | on        створка двери не вернулась в исходное состояние ≥ door_s секунд
  door_s = 8
"""
import argparse, json, os, pathlib, sys
import cv2, numpy as np

HERE = pathlib.Path(__file__).resolve().parent
CFG = json.load(open(HERE / "assets" / "views.json", encoding="utf-8"))
ALIGNED = pathlib.Path(os.environ.get("ALIGNED_DIR", ""))
OV = pathlib.Path(os.environ.get("OV_DIR", ""))                # кэш Grounding DINO (1 fps)   # кадры, выровненные к эталону ракурса (для датчиков по изображению)
ITEMS = {"handbag", "backpack", "cup", "bottle", "wine glass"}
CASHIER_RULE = "earliest"
DEFAULT = {"items": "v0", "phone": "v0", "valu": "v0", "door": "off", "door_s": "8"}
BEST = {"items": "new", "door": "on", "valu_k": "2", "owner": "cashier", "outsider": "on", "screen": "on", "leave_tc": "on", "policy": "hybrid", "fb_tc": "mid", "valu_shift": "-4.5", "door_shift": "2", "fallback_bag": "off", "fallback_visit": "on", "visit_s": "5", "outsider2": "on", "out2_prio": "high"}   # H5 = H4 + таймкод двери +2 с: dev 0.62, стресс 0.63
MASK = {v: {k: cv2.fillPoly(np.zeros((848, 464), np.uint8), [np.int32(z)], 1).astype(bool) for k, z in zs.items()}
        for v, zs in CFG["zones"].items()}


def imread(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def to_ref(H, x, y):
    if H is None: return x, y
    p = cv2.perspectiveTransform(np.float32([[[x, y]]]), np.float32(H))[0, 0]; return float(p[0]), float(p[1])


def in_zone(view, zone, x, y):
    m = MASK[view][zone]; xi, yi = int(round(x)), int(round(y))
    return 0 <= yi < 848 and 0 <= xi < 464 and bool(m[yi, xi])


def first_run(flags, k):
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        if run == k: return i - k + 1
    return None



def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


FLOOR_MIN_COV = float(os.environ.get("FLOOR_MIN_COV", "0.5"))
REL_MIN_H = float(os.environ.get("REL_MIN_H", "0.45"))   # без калибровки: человек ниже этой доли «крупных» людей ролика — за стеклом/вдали


def _ref_height(cache):
    hs = [b[3] - b[1] for fr in cache["frames"] for l, s, b in fr["dets"] if l == "person" and s >= 0.5]
    return float(np.percentile(hs, 90)) if hs else None


def _track(cache, room):
    tracks, per = [], []                         # [{id, last_box, frames:set}]
    for i, fr in enumerate(cache["frames"]):
        boxes = [b for l, s, b in fr["dets"] if l == "person" and s >= 0.5 and room(fr, b)]
        ids = []
        for b in boxes:
            best = max(tracks, key=lambda t: iou(t["box"], b), default=None)
            if best is not None and iou(best["box"], b) > 0.2 and best["last"] >= i - 6:
                best["box"], best["last"] = b, i; best["frames"].add(i); ids.append(best["id"])
            else:
                tracks.append({"id": len(tracks), "box": b, "last": i, "first": i, "frames": {i}}); ids.append(len(tracks) - 1)
        per.append(list(zip(ids, boxes)))
    n = max(1, len(cache["frames"]))
    if CASHIER_RULE == "longest_early":                  # кассир = самый «долгий» трек среди появившихся в первые 3 с
        early = [t for t in tracks if t["first"] <= 6] or tracks
        cid = max(early, key=lambda t: len(t["frames"]))["id"] if early else None
    else:
        cands = [t for t in tracks if len(t["frames"]) >= 0.2 * n] or tracks
        cid = min(cands, key=lambda t: t["first"])["id"] if cands else None
    cov = len(next((t["frames"] for t in tracks if t["id"] == cid), ()))
    return cid, per, cov


def cashier_track(cache):
    """Простой трекер по IoU; кассир = самый ранний трек в комнате, прожив. ≥ 20% ролика (клиенты входят позже).
    Людей за стеклом отсекает линия пола (CFG floor_y; без калибровки — в пикселях исходной камеры). Если при этом
    кассир виден меньше чем в FLOOR_MIN_COV кадров с людьми (камера стоит иначе, линия «съела» кассира), трекер
    переключается на относительное правило: рамка ниже REL_MIN_H от 90-го перцентиля роста людей ролика — за стеклом."""
    floor = lambda fr, b: to_ref(fr.get("H"), (b[0] + b[2]) / 2, b[3])[1] > CFG["floor_y"]
    cid, per, cov = _track(cache, floor)
    with_people = sum(1 for fr in cache["frames"] if any(l == "person" and s >= 0.5 for l, s, b in fr["dets"]))
    if any(fr.get("H") is None for fr in cache["frames"]) and with_people and cov < FLOOR_MIN_COV * with_people:
        href = _ref_height(cache)
        cid, per, _ = _track(cache, lambda fr, b: href is None or (b[3] - b[1]) >= REL_MIN_H * href)
    return cid, per


def leave_index(cache, before, radius=40.0):
    """Момент, когда кассир ушёл со своего места: последний кадр до `before`, где его ноги были у «своего» места."""
    cid, per = cashier_track(cache)
    pos = []
    for k in range(min(before, len(per))):
        fr = cache["frames"][k]
        for i, b in per[k]:
            if i == cid: pos.append((k, np.array(to_ref(fr["H"], (b[0] + b[2]) / 2, b[3]))))
    if len(pos) < 3: return None
    seat = np.median(np.stack([p for _, p in pos[: max(3, len(pos) // 3)]]), 0)
    at_seat = [k for k, p in pos if np.linalg.norm(p - seat) <= radius]
    return (at_seat[-1] + 1) if at_seat else None

def per_frame(fr, cashier_boxes=None):
    """cashier_boxes=None — как в v0 (любой человек в комнате); иначе телефон/предметы засчитываются только рядом с кассиром."""
    view, H = fr["view"], fr["H"]
    people = [d for d in fr["dets"] if d[0] == "person" and d[1] >= 0.5]
    room = [b for _, _, b in people if to_ref(H, (b[0] + b[2]) / 2, b[3])[1] > CFG["floor_y"]]
    owners = room if cashier_boxes is None else cashier_boxes
    inside = lambda d, p: p[0] <= (d[2][0] + d[2][2]) / 2 <= p[2] and p[1] <= (d[2][1] + d[2][3]) / 2 <= p[3]
    phone = any(d[0] == "cell phone" and d[1] >= 0.4 and any(inside(d, p) for p in owners) for d in fr["dets"])
    items = []
    for l, s, b in fr["dets"]:
        if l in ITEMS and s >= 0.4 and H is not None:
            cx, cy = to_ref(H, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            if not in_zone(view, "desk", cx, cy): continue
            if cashier_boxes is not None and room:                   # предмет ближе к клиенту/инкассатору — не кассира
                bc = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                dist = lambda p: ((p[0] + p[2]) / 2 - bc[0]) ** 2 + ((p[1] + p[3]) / 2 - bc[1]) ** 2
                nearest = min(room, key=dist)
                if not any(nearest is c for c in cashier_boxes): continue
            items.append((l, round(cx / 25), round(cy / 25)))     # класс + грубая ячейка 25 px
    return {"t": fr["t"], "view": view, "n_room": len(room), "phone": phone, "items": items}


def door_profile(im, z):
    x0, x1 = min(p[0] for p in z), max(p[0] for p in z); y0, y1 = min(p[1] for p in z), max(p[1] for p in z)
    y1 = y0 + (y1 - y0) // 2                                                  # верхняя половина створки (выше голов)
    g = cv2.cvtColor(im[y0:y1, max(0, x0 - 10):x1 + 25], cv2.COLOR_BGR2GRAY).astype(np.float32)
    sx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)); sy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    p = ((sx > sy * 2) * sx).sum(0); p = np.convolve(p, np.ones(5) / 5, "same"); return (p - p.mean()) / (p.std() + 1e-6)


def decide(cache, opt):
    ts_ = [f["t"] for f in cache["frames"]]
    fps_c = (len(ts_) - 1) / (ts_[-1] - ts_[0]) if len(ts_) > 2 and ts_[-1] > ts_[0] else 2.0
    K = lambda k2: max(1, int(round(float(k2))))   # окна в кадрах: на 1 fps они вдвое длиннее по времени — это проверено и лучше (A/B)
    global CASHIER_RULE; CASHIER_RULE = opt.get("cashier", "earliest")
    if opt.get("owner") == "cashier":
        cid, per = cashier_track(cache)
        F = [per_frame(f, [b for i, b in per[k] if i == cid]) for k, f in enumerate(cache["frames"])]
    else:
        F = [per_frame(f) for f in cache["frames"]]
    ev = []
    if opt["phone"] == "v0":
        i = first_run([f["phone"] for f in F], K(opt.get("phone_k", 2)))
        if i is not None: ev.append(("phone", i))
    if opt["items"] == "v0":
        i = first_run([bool(f["items"]) for f in F], 3)
        if i is not None: ev.append(("personal_items", i))
    elif opt["items"] == "new":
        base = set().union(*[set(f["items"]) for f in F[:8]]) if F else set()   # что лежало на столе первые 4 с
        if opt.get("items_static") == "on":                                     # место (без класса, ±1 ячейка), занятое в начале или в ≥ static_frac кадров — предмет комнаты
            from collections import Counter
            cnt = Counter(c for f in F for c in {(x, y) for _, x, y in f["items"]})
            st = {(x, y) for _, x, y in base} | {c for c, m in cnt.items() if m >= float(opt.get("static_frac", 0.3)) * len(F)}
            st |= {(x + dx, y + dy) for x, y in st for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
            newf = lambda f: [it for it in f["items"] if (it[1], it[2]) not in st]
        else:
            newf = lambda f: list(set(f["items"]) - base)
        if opt.get("items_side") == "on":                                       # только на половине стола кассира (ось стола, как в A6)
            cid_, per_ = cashier_track(cache)
            z = np.float32(CFG["zones"][F[0]["view"]]["desk"]) if F else None
            if z is not None and cid_ is not None:
                c0 = z.mean(0); ax = np.linalg.svd(z - c0)[2][0]; nrm = np.array([-ax[1], ax[0]])
                cs = [np.dot(np.array(to_ref(cache["frames"][k]["H"], (b[0] + b[2]) / 2, b[3])) - c0, nrm) for k in range(len(per_)) for i_, b in per_[k] if i_ == cid_]
                if cs:
                    sg = np.sign(np.median(cs)); nf0 = newf
                    newf = lambda f: [it for it in nf0(f) if np.sign(np.dot(np.array([it[1] * 25, it[2] * 25]) - c0, nrm)) == sg]
        i = first_run([bool(newf(f)) for f in F], K(3))
        if i is not None: ev.append(("personal_items", i))
    feat = bool(cache["frames"]) and "feat" in cache["frames"][0]
    valu_ok = True
    if opt.get("valu_gate") == "on" and feat and cache["frames"]:
        # эталон пустой комнаты не подходит (чужой ракурс): уже в первые 3 с стол «отличается» на большую долю площади
        f0 = cache["frames"]; v0 = f0[0]["view"]; area = float(MASK[v0]["desk"].sum()) if v0 in MASK else 0.0
        early = [f["feat"]["blob"] for f in f0 if f["t"] - f0[0]["t"] <= 3 and f["feat"]["blob"] is not None]
        if not area or not early or float(np.median(early)) > float(opt.get("valu_gate_frac", 0.35)) * area: valu_ok = False
    if opt["valu"] == "v0" and feat and valu_ok:
        was = False; k = K(opt.get("valu_k", 6))
        for j, f in enumerate(F):
            if f["n_room"] > 0: was = True; continue
            win = F[j:j + k]
            if not (was and len(win) >= min(2, k) and all(x["n_room"] == 0 for x in win)): continue
            bl = [cache["frames"][j + q]["feat"]["blob"] for q in range(len(win))][min(2, len(win) - 1):]
            if all(b is not None and b >= 40 for b in bl): ev.append(("valuables_on_desk", j)); break
    if opt["valu"] == "v0" and ALIGNED.exists() and not feat:
        fid = cache["file_id"]; d = ALIGNED / fid; v = F[0]["view"] if F else None
        empty = CFG["empty_room"].get(v)
        if d.exists() and empty:
            e = imread(HERE / "assets" / f"empty_{v}.jpg")
            lab = lambda im: cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32), (7, 7), 0); le = lab(e)
            files = sorted(os.listdir(d)); was = False
            for j, f in enumerate(F):
                if f["n_room"] > 0: was = True; continue
                k = int(opt.get("valu_k", 6)); win = F[j:j + k]
                if not (was and len(win) >= min(2, k) and all(x["n_room"] == 0 for x in win)): continue   # ролик может кончиться сразу после ухода
                ok = True
                for x in win[min(2, len(win) - 1):]:
                    fn = min(files, key=lambda q: abs(int(q[:4]) / 10 - x["t"]))
                    dd = np.sqrt(((lab(imread(d / fn)) - le) ** 2).sum(2)); b = ((dd > 18) & MASK[v]["desk"]).astype(np.uint8)
                    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)); n, _, st, _ = cv2.connectedComponentsWithStats(b)
                    if (int(st[1:, 4].max()) if n > 1 else 0) < 40: ok = False; break
                if ok: ev.append(("valuables_on_desk", j)); break
    if opt["door"] == "on" and feat:
        P = [f["feat"]["door"] for f in cache["frames"]]; idx = [q for q, x in enumerate(P) if x is not None]
        if len(idx) >= 4:
            ref = np.median(np.array([P[q] for q in idx[:4]]), 0)
            flags = [False] * len(P)
            for q in idx: flags[q] = float(np.dot(np.array(P[q]), ref) / len(ref)) < 0.8
            if opt.get("door_occl") == "on":                 # человек перед створкой — состояние двери «не видно»: кадр не считается и не рвёт серию
                for q, fr in enumerate(cache["frames"]):
                    if fr["H"] is None: continue
                    dz = MASK[fr["view"]]["door"]
                    for l, s_, b in fr["dets"]:
                        if l != "person" or s_ < 0.5: continue
                        pts = cv2.perspectiveTransform(np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]).reshape(-1, 1, 2), np.float32(fr["H"])).reshape(-1, 2)
                        m = cv2.fillPoly(np.zeros((848, 464), np.uint8), [np.int32(pts)], 1).astype(bool)
                        if (m & dz).sum() > float(opt.get("occl_frac", 0.05)) * dz.sum(): flags[q] = None; break
                run, i, kk = 0, None, K(float(opt["door_s"]) * 2)
                for q, f in enumerate(flags):
                    if f is None: continue
                    run = run + 1 if f else 0
                    if run == 1: st = q
                    if run == kk: i = st; break
            else:
                i = first_run(flags, K(float(opt["door_s"]) * 2))
            if i is not None: ev.append(("door_open", i))
    if opt["door"] == "on" and ALIGNED.exists() and not feat:
        d = ALIGNED / cache["file_id"]
        if d.exists() and F:
            v = F[0]["view"]; z = CFG["zones"][v]["door"]; files = sorted(os.listdir(d))
            P = [door_profile(imread(d / fn), z) for fn in files]; ts = [int(fn[:4]) / 10 for fn in files]
            occl = [False] * len(P)
            if opt.get("door_occl") == "on":                 # человек перед створкой — состояние двери «не видно»
                dz = cv2.fillPoly(np.zeros((848, 464), np.uint8), [np.int32(z)], 1).astype(bool)
                for q, tq in enumerate(ts):
                    fr = min(cache["frames"], key=lambda f: abs(f["t"] - tq))
                    for l, s_, b in fr["dets"]:
                        if l != "person" or s_ < 0.5 or fr["H"] is None: continue
                        pts = cv2.perspectiveTransform(np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]).reshape(-1, 1, 2), np.float32(fr["H"])).reshape(-1, 2)
                        m = cv2.fillPoly(np.zeros((848, 464), np.uint8), [np.int32(pts)], 1).astype(bool)
                        if (m & dz).sum() > 0.05 * dz.sum(): occl[q] = True; break
            free = [q for q in range(len(P)) if not occl[q]][:4] or list(range(min(4, len(P))))
            ref = np.median(np.stack([P[q] for q in free]), 0); c = [float(np.dot(p, ref) / len(p)) for p in P]
            flags = [None if occl[q] else (x < 0.8) for q, x in enumerate(c)]
            if opt.get("door_occl") == "on":
                for q, tq in []:
                    fr = min(cache["frames"], key=lambda f: abs(f["t"] - tq))
                    for l, s_, b in fr["dets"]:
                        if l != "person" or s_ < 0.5 or fr["H"] is None: continue
                        pts = cv2.perspectiveTransform(np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]).reshape(-1, 1, 2), np.float32(fr["H"])).reshape(-1, 2)
                        m = cv2.fillPoly(np.zeros((848, 464), np.uint8), [np.int32(pts)], 1).astype(bool)
                        if (m & dz).sum() > 0.15 * dz.sum(): flags[q] = None; break
                run = 0; i = None                            # «не видно» не прерывает и не продлевает серию
                for q, f in enumerate(flags):
                    if f is None: continue
                    run = run + 1 if f else 0
                    if run == int(float(opt["door_s"]) * 2): i = q - run + 1; break
            else:
                k = int(float(opt["door_s"]) * 2); i = first_run(flags, k)
            if i is not None:
                j = min(range(len(F)), key=lambda q: abs(F[q]["t"] - ts[i])); ev.append(("door_open", j))

    if opt.get("outsider") == "on":
        # линия стола (главная ось полигона стола в эталоне) делит комнату; кассир на одной стороне
        cid, per = cashier_track(cache)
        if cid is not None:
            v = F[0]["view"]; z = np.float32(CFG["zones"][v]["desk"]); c0 = z.mean(0)
            u, sv, vt = np.linalg.svd(z - c0); axis = vt[0]; nrm = np.array([-axis[1], axis[0]])
            side = lambda pt: float(np.dot(np.array(pt) - c0, nrm))
            foot = lambda f, b: to_ref(f["H"], (b[0] + b[2]) / 2, b[3])
            cs = [side(foot(cache["frames"][k], b)) for k in range(len(per)) for i, b in per[k] if i == cid]
            if cs:
                cside = np.sign(np.median(cs)); flags = []
                for k in range(len(per)):
                    fr = cache["frames"][k]
                    flags.append(any(i != cid and np.sign(side(foot(fr, b))) == cside and abs(side(foot(fr, b))) > float(opt.get("out_margin", 15))
                                     for i, b in per[k]))
                i0 = first_run(flags, K(opt.get("out_k", 2)))
                if i0 is not None: ev.append(("outsider", i0))

    if opt.get("outsider2") == "on":
        # посторонний: у стола одновременно ≥2 посетителя (не кассир), рамки не дублируются; таймкод — середина периода
        cid, per = cashier_track(cache); R = float(opt.get("out2_r", -60)); flags = []
        for k in range(len(per)):
            fr = cache["frames"][k]; near = []
            if fr["H"] is None: flags.append(False); continue
            zd = np.float32(CFG["zones"][fr["view"]]["desk"]).reshape(-1, 1, 2)
            for i, b in per[k]:
                if i == cid: continue
                if cv2.pointPolygonTest(zd, to_ref(fr["H"], (b[0] + b[2]) / 2, b[3]), True) > R and all(iou(b, q) < 0.3 for q in near): near.append(b)
            flags.append(len(near) >= 2)
        i0 = first_run(flags, K(opt.get("out2_k", 3)))
        if i0 is not None:
            i1 = i0
            while i1 + 1 < len(flags) and flags[i1 + 1]: i1 += 1
            ev = [e for e in ev if e[0] != "outsider"] + [("outsider", (i0 + i1) // 2 if opt.get("out2_tc", "mid") == "mid" else i0)]

    if opt.get("screen") == "on" and feat:
        was = False
        for j, f in enumerate(F):
            if f["n_room"] > 0: was = True; continue
            if not was: continue
            sw = cache["frames"][j]["feat"]["scr"]
            if sw is not None and sw > float(opt.get("screen_white", 0.25)):
                k = j
                while k > 0 and F[k - 1]["n_room"] == 0: k -= 1
                ev.append(("pc_unlocked", max(0, k - 1)))
            break
    if opt.get("screen") == "on" and ALIGNED.exists() and not feat:
        # ПК не заблокирован: кассир ушёл (комната пуста), а экран ноутбука остаётся светлым (открытое приложение)
        d = ALIGNED / cache["file_id"]
        if d.exists():
            files = sorted(os.listdir(d)); was = False
            for j, f in enumerate(F):
                if f["n_room"] > 0: was = True; continue
                if not was: continue
                fr = cache["frames"][j]; L = [b for l, s_, b in fr["dets"] if l == "laptop" and s_ >= 0.4]
                if not L or fr["H"] is None: break
                b = max(L, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
                pts = np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[1] + 0.55 * (b[3] - b[1])], [b[0], b[1] + 0.55 * (b[3] - b[1])]]).reshape(-1, 1, 2)
                q = cv2.perspectiveTransform(pts, np.float32(fr["H"])).reshape(-1, 2); x0, y0 = q.min(0).astype(int); x1, y1 = q.max(0).astype(int)
                im = imread(d / min(files, key=lambda n: abs(int(n[:4]) / 10 - f["t"])))[max(0, y0):y1, max(0, x0):x1]
                if im.size and float((cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) > 170).mean()) > float(opt.get("screen_white", 0.25)):
                    k = j
                    while k > 0 and F[k - 1]["n_room"] == 0: k -= 1
                    ev.append(("pc_unlocked", max(0, k - 1)))
                break

    if opt.get("phone_ov") == "on" and (OV / f"{cache['file_id']}.json").exists():
        # телефон по Grounding DINO: «mobile phone» внутри рамки кассира ≥ k кадров подряд (1 fps)
        ov = json.load(open(OV / f"{cache['file_id']}.json"))["frames"]
        cid, per = cashier_track(cache); thr = float(opt.get("ov_thr", 0.35)); flags = []
        for o in ov:
            k = min(range(len(cache["frames"])), key=lambda q: abs(cache["frames"][q]["t"] - o["t"]))
            cb = [b for i, b in per[k] if i == cid]
            hit = False
            for l, s_, b in o["dets"]:
                if "phone" in l and s_ >= thr and (b[2] - b[0]) * (b[3] - b[1]) < 0.02 * 464 * 848:
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    if any(p[0] <= cx <= p[2] and p[1] <= cy <= p[3] for p in cb): hit = True
            flags.append(hit)
        i = first_run(flags, int(opt.get("ov_k", 2)))
        if i is not None:
            j = min(range(len(F)), key=lambda q: abs(F[q]["t"] - ov[i]["t"]))
            ev = [e for e in ev if e[0] != "phone"] + [("phone", j)]
    if opt.get("leave_tc") == "on":                       # событие «ушёл и оставил» — таймкод на момент ухода с места
        ev = [(c, (leave_index(cache, i) or i) if c in ("valuables_on_desk", "pc_unlocked") else i) for c, i in ev]
        ev = [(c, min(i, len(F) - 1)) for c, i in ev]
    if opt.get("policy") == "hybrid":
        PRIO = ["door_open", "pc_unlocked", "valuables_on_desk", "phone", "personal_items", "outsider", "procedure"]
        if opt.get("out2_prio") == "high": PRIO = ["door_open", "pc_unlocked", "valuables_on_desk", "outsider", "phone", "personal_items", "procedure"]
        ev = sorted(ev, key=lambda e: PRIO.index(e[0]) if e[0] in PRIO else 99)       # главное событие = самое надёжное правило
        fr = cache["frames"]; n = max(1, len(fr))
        geo_ok = sum(f["H"] is not None for f in fr) / n >= 0.7 and float(np.median([f["inl"] for f in fr])) >= 60
        had_ev = bool(ev); fb_reason = None
        if not ev:
            two = [i for i, f in enumerate(F) if f["n_room"] >= 2]
            if opt.get("fb_tc") == "mid":
                cand = two[len(two) // 2] if two else len(F) // 2                      # середина визита: передача обычно посреди него
            else:
                cand = two[0] if two else len(F) // 2                                  # первый момент «двое у стола»
            if not geo_ok:                                                              # новый ракурс, зонам не верим → по умолчанию «нарушение»
                ev = [("procedure", cand)]; fb_reason = "geometry"
            elif opt.get("fallback_visit") == "on":                                     # визит с передачей: кассир и посетитель вместе ≥ visit_s секунд
                if len(two) / max(fps_c, 1e-6) >= float(opt.get("visit_s", 10)): ev = [("procedure", cand)]; fb_reason = "visit"
            elif opt.get("fallback_bag") == "on":
                isbag = lambda l, s_: l in ("handbag", "backpack", "suitcase") and s_ >= 0.4
                if opt.get("bag_mode") == "new":                                          # только сумки, которых не было в первые 3 с (статичные предметы комнаты не в счёт)
                    cell = lambda f, b: (round(to_ref(f["H"], (b[0] + b[2]) / 2, (b[1] + b[3]) / 2)[0] / 40), round(to_ref(f["H"], (b[0] + b[2]) / 2, (b[1] + b[3]) / 2)[1] / 40))
                    static = {cell(f, b) for f in fr if f["t"] - fr[0]["t"] <= 3 for l, s_, b in f["dets"] if isbag(l, s_)}
                    static |= {(x + dx, y + dy) for x, y in static for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
                    bag = sum(any(isbag(l, s_) and cell(f, b) not in static for l, s_, b in f["dets"]) for f in fr) / n
                else:
                    bag = sum(any(isbag(l, s_) for l, s_, b in f["dets"]) for f in fr) / n
                if bag >= float(opt.get("bag_thr", 0.85)): ev = [("procedure", cand)]; fb_reason = "visit"
        shift = {"valuables_on_desk": float(opt.get("valu_shift", 0)), "door_open": float(opt.get("door_shift", 0))}               # поправка систематического сдвига таймкода
        viol = [{"type_code": c, "type": c, "timecode_sec": max(0.0, round(F[i]["t"] + shift.get(c, 0.0), 2)), "description": "", "evidence_frames": [], "fallback": None if had_ev else fb_reason} for c, i in ev]
        return {"verdict": "violation" if viol else "no_violation", "violations": viol, "processing_sec": cache.get("extract_sec"),
                "geometry_ok": geo_ok}
    viol = [{"type_code": c, "type": c, "timecode_sec": F[i]["t"], "description": "", "evidence_frames": []} for c, i in ev]
    return {"verdict": "violation" if viol else "no_violation", "violations": sorted(viol, key=lambda v: v["timecode_sec"]),
            "processing_sec": cache.get("extract_sec")}


def main():
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default=""); ap.add_argument("--out", required=True)
    a = ap.parse_args(); opt = dict(DEFAULT); opt.update(dict(kv.split("=") for kv in a.variant.split(",") if kv))
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for f in sorted((HERE / os.environ.get("CACHE_SUBDIR", "cache")).glob("*.json")):
        c = json.load(open(f)); json.dump(decide(c, opt), open(out / f.name, "w"))
    print("variant", opt)


if __name__ == "__main__":
    main()
