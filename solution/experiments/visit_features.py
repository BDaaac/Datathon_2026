"""Признаки уровня ролика для модели «подозрительности визита». Без знания комнаты, без разметки.
python visit_features.py -> runs/visit_features.csv"""
import sys, json, pathlib, time
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench, decide as D

SENS = ["door_tracks", "desk_left", "items_new", "laptop_screen", "outsider_rel"]
SMALL = ["book", "cell phone", "remote", "cup", "handbag", "suitcase", "backpack"]
CONF = ["cell phone", "book", "remote", "handbag", "cup"]


def iou(a, b): return D.iou(a, b)
def area(b): return max(0, b[2] - b[0]) * max(0, b[3] - b[1])
def inter(a, b): return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
def ctr(b): return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
def foot(b): return np.array([(b[0] + b[2]) / 2, b[3]])
def hgt(b): return b[3] - b[1]
def mx(l): return max(l) if l else 0.0
def mean(l): return float(np.mean(l)) if l else 0.0


def between_region(a, v, shape):
    """область между двумя людьми: полоса между ближайшими краями по x, общая вертикальная часть."""
    x0, x1 = min(a[2], v[2]), max(a[0], v[0])
    if x0 > x1: x0, x1 = x1, x0
    if x1 - x0 < 10:
        cx = (ctr(a)[0] + ctr(v)[0]) / 2; x0, x1 = cx - 15, cx + 15
    y0, y1 = max(a[1], v[1]), min(a[3], v[3])
    if y1 - y0 < 10: y0, y1 = min(a[1], v[1]), max(a[3], v[3])
    return int(max(0, y0)), int(min(shape[0], y1)), int(max(0, x0)), int(min(shape[1], x1))


def feats(fid):
    c = bench.nogeo(json.load(open(HERE / "cache_adapt" / f"{fid}.json"))); c["file_id"] = fid
    D.CASHIER_RULE = "earliest"; cid, per = D.cashier_track(c)
    fr = c["frames"]; n = len(fr); T = [f["t"] for f in fr]
    dt = (T[-1] - T[0]) / max(1, n - 1) if n > 1 else 0.5
    r = {"fid": fid, "n_frames": n, "dur_s": T[-1] - T[0] + dt, "dt": dt}
    ppl = []
    for f in fr:
        out = []
        for b in sorted([d[2] for d in f["dets"] if d[0] == "person" and d[1] >= 0.5], key=lambda b: -area(b)):
            if all(iou(b, q) < 0.3 for q in out): out.append(b)
        ppl.append(out)
    cnt = np.array([len(p) for p in ppl])
    r["frac_2plus"] = float((cnt >= 2).mean()); r["sec_2plus"] = float((cnt >= 2).sum() * dt); r["max_people"] = int(cnt.max())
    cb = [None] * n
    for k in range(n):
        for i, b in per[k]:
            if i == cid: cb[k] = b
    vb = [None] * n
    for k in range(n):
        cands = [b for b in ppl[k] if cb[k] is None or iou(b, cb[k]) < 0.5]
        if cands: vb[k] = max(cands, key=area)
    vis = np.array([v is not None for v in vb]); gap = max(1, int(round(3 / max(dt, 1e-3))))
    ep = 0; last = -10**9
    for k in range(n):
        if vis[k]:
            if k - last > gap: ep += 1
            last = k
    r["n_visits"] = ep; r["visitor_frac"] = float(vis.mean()); r["visitor_sec"] = float(vis.sum() * dt)
    has = np.array([b is not None for b in cb])
    r["cash_absent_frac"] = float(1 - has.mean())
    hs = [hgt(b) for b in cb if b is not None]
    H0 = max(float(np.percentile(hs, 90)) if hs else 200.0, 50.0)
    r["cash_h"] = H0
    keys = ["cash_far_frac", "cash_max_disp", "cash_stood_up", "cash_h_range", "cash_left_during_visit", "cash_left_end", "cash_gone_while_visitor"]
    if has.sum() >= 3:
        idx = np.where(has)[0]; seat = np.median(np.stack([foot(cb[k]) for k in idx[:max(3, len(idx) // 3)]]), 0)
        far = np.array([np.linalg.norm(foot(cb[k]) - seat) / H0 > 0.4 for k in idx])
        r["cash_far_frac"] = float(far.sum() / n + (1 - has.mean()))   # отсутствие = не на месте
        r["cash_max_disp"] = float(max(np.linalg.norm(foot(cb[k]) - seat) for k in idx) / H0)
        hmed = np.median(hs[:max(3, len(hs) // 3)])
        r["cash_stood_up"] = float(max(hs) / hmed) if hmed > 0 else 1.0
        r["cash_h_range"] = float((max(hs) - min(hs)) / H0)
        lv = np.where(vis)[0]
        r["cash_left_during_visit"] = float(len(lv) > 0 and any(not has[k] for k in range(lv[0], lv[-1] + 1)))
        r["cash_left_end"] = float(not has[-max(2, n // 10):].any())
        r["cash_gone_while_visitor"] = float(np.mean([not has[k] for k in lv])) if len(lv) else 0.0
    else:
        for k_ in keys: r[k_] = 0.0
        r["cash_far_frac"] = 1.0; r["cash_stood_up"] = 1.0
    vh = [hgt(b) for b in vb if b is not None]
    if vh:
        vmax = np.percentile(vh, 95); st = float(np.mean([h >= 0.85 * vmax for h in vh]))
        r["vis_stand_frac"] = st; r["vis_sit_frac"] = 1 - st; r["vis_h_ratio_cash"] = float(np.median(vh) / H0)
    else: r["vis_stand_frac"] = r["vis_sit_frac"] = r["vis_h_ratio_cash"] = 0.0
    dists = [np.linalg.norm(ctr(cb[k]) - ctr(vb[k])) / H0 for k in range(n) if cb[k] is not None and vb[k] is not None]
    r["min_dist_cv"] = float(min(dists)) if dists else 5.0; r["mean_dist_cv"] = mean(dists) if dists else 5.0
    tb = []
    for f in fr:
        t = [d[2] for d in f["dets"] if d[0] == "diningtable" and d[1] >= 0.3]
        tb.append(max(t, key=area) if t else None)
    tl = [t for t in tb if t is not None]
    if tl:
        tbl = np.median(np.stack(tl), 0)
        r["table_w_rel"] = float((tbl[2] - tbl[0]) / H0)
        r["vis_at_table_sec"] = float(sum(vb[k] is not None and inter(vb[k], tbl) / max(1, area(vb[k])) > 0.1 for k in range(n)) * dt)
        r["cash_at_table_frac"] = float(np.mean([cb[k] is not None and inter(cb[k], tbl) / max(1, area(cb[k])) > 0.1 for k in range(n)]))
    else: r["table_w_rel"] = r["vis_at_table_sec"] = r["cash_at_table_frac"] = 0.0
    touch = touch_f = hand = hand_f = 0; was = wh = False
    for k in range(n):
        if cb[k] is None or vb[k] is None: was = wh = False; continue
        a, b = cb[k], vb[k]
        t = inter(a, b) / max(1, min(area(a), area(b))) > 0.05
        touch_f += t; touch += (t and not was); was = t
        u = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
        hit = False
        for l, s, bb in fr[k]["dets"]:
            if l in SMALL and s >= 0.25:
                cc = ctr(bb)
                if u[0] <= cc[0] <= u[2] and u[1] <= cc[1] <= u[3] and np.linalg.norm(cc - ctr(a)) < 1.2 * H0 and np.linalg.norm(cc - ctr(b)) < 1.2 * H0 \
                        and (inter(bb, a) / max(1, area(bb)) < 0.9 or inter(bb, b) / max(1, area(bb)) < 0.9):
                    hit = True; break
        hand_f += hit; hand += (hit and not wh); wh = hit
    r["person_touch_events"] = touch; r["person_touch_sec"] = touch_f * dt
    r["item_between_events"] = hand; r["item_between_sec"] = hand_f * dt

    def near_box(bb, b, m=0.35):
        e = m * hgt(b); return inter(bb, [b[0] - e, b[1] - e, b[2] + e, b[3] + e]) > 0
    for who, B in (("cash", cb), ("vis", vb)):
        for cl in CONF:
            r[f"conf_{cl.replace(' ', '_')}_{who}"] = mx([s for k in range(n) if B[k] is not None for l, s, bb in fr[k]["dets"] if l == cl and near_box(bb, B[k])])
    for cl in CONF: r[f"conf_{cl.replace(' ', '_')}_any"] = mx([s for f in fr for l, s, bb in f["dets"] if l == cl])
    fs = bench.FrameStore(fid); prev = None; ec = []; eb = []
    try:
        for k in range(n):
            g = fs.get(k)
            if g is None: prev = None; continue
            g = g.astype(np.float32).mean(2)
            if prev is not None and cb[k] is not None:
                d = np.abs(g - prev)
                b = [int(max(0, v)) for v in cb[k]]; reg = d[b[1]:b[3], b[0]:b[2]]
                if reg.size: ec.append(float(reg.mean()))
                if vb[k] is not None:
                    y0, y1, x0, x1 = between_region(cb[k], vb[k], d.shape); reg2 = d[y0:y1, x0:x1]
                    if reg2.size: eb.append(float(reg2.mean()))
            prev = g
    except Exception as ex: print("motion err", fid, repr(ex), file=sys.stderr)
    r["motion_cash_mean"] = mean(ec); r["motion_cash_max"] = mx(ec)
    r["motion_between_mean"] = mean(eb); r["motion_between_max"] = mx(eb)
    ev = bench.run_sensors(SENS, fid)
    for s in SENS:
        e = [x for x in ev if x["sensor"] == s]
        r[f"sens_{s}"] = float(bool(e)); r[f"sens_{s}_score"] = mx([float(x.get("score", 0)) for x in e])
    return r


if __name__ == "__main__":
    t0 = time.time(); rows = []
    for p in sorted((HERE / "cache_adapt").glob("*.json")):
        rows.append(feats(p.stem)); print(p.stem, f"{time.time()-t0:.0f}s", file=sys.stderr)
    df = pd.DataFrame(rows).fillna(0)
    df.to_csv(HERE / "runs" / "visit_features.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(df.shape, "NaN:", int(df.isna().sum().sum())); print(df.describe().T)
    print(f"time {time.time()-t0:.0f}s")
