"""Модель «подозрительности визита» (стекинг слабых признаков) поверх датчиков ядра.

Политика: датчик сработал → его событие; датчики молчат и p(нарушение) ≥ порога → «procedure» (требуется проверка) в середине визита;
иначе «нарушений нет». Логистическая регрессия на numpy (стандартизация + L2).
Честность: leave-one-out на dev (ролик не участвует в обучении своей модели), затем обучение на всём dev → отложенные V2.

  python visit_model.py [--feats a,b,c] [--l2 1.0]
"""
import argparse, csv, json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench
from evaluate import load_labels, score

SENS = ["door_tracks", "laptop_screen", "outsider_rel", "desk_left", "items_new"]


def fit(X, y, l2):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd; w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(3000):                                  # градиентный спуск, задача маленькая
        p = 1 / (1 + np.exp(-(Z @ w + b))); g = p - y
        w -= 0.1 * (Z.T @ g / len(y) + l2 * w / len(y)); b -= 0.1 * g.mean()
    return lambda Xn: 1 / (1 + np.exp(-(((Xn - mu) / sd) @ w + b))), w


def load_feats(cols):
    rows = {r["fid"].zfill(2): r for r in csv.DictReader(open(HERE / "runs" / "visit_features.csv", encoding="utf-8"))}
    cols = cols or [c for c in next(iter(rows.values())) if c != "fid"]
    return rows, cols, lambda fid: np.array([float(rows[fid][c] or 0) for c in cols])


def preds_for(gt, sens, prob, thr):
    out = {}
    for fid in gt:
        ev = sens[fid]
        if ev:
            e = sorted(ev, key=lambda e: (bench.PRIO.index(e["type"]) if e["type"] in bench.PRIO else 99, -e.get("score", 0)))[0]
            out[fid] = {"verdict": "violation", "violations": [{"type_code": e["type"], "timecode_sec": e["t"]}]}
        elif prob[fid] >= thr:
            c = json.load(open(HERE / "cache_adapt" / f"{fid}.json"))["frames"]
            two = [k for k, f in enumerate(c) if bench.visitors(f) >= 2] or [len(c) // 2]
            out[fid] = {"verdict": "violation", "violations": [{"type_code": "procedure", "timecode_sec": c[two[len(two) // 2]]["t"]}]}
        else:
            out[fid] = {"verdict": "no_violation", "violations": []}
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--feats", default=""); ap.add_argument("--l2", type=float, default=1.0); a = ap.parse_args()
    rows, cols, vec = load_feats([c for c in a.feats.split(",") if c])
    dev, lock = load_labels("dev"), load_labels("lockbox")
    sens = {fid: bench.run_sensors(SENS, fid) for fid in list(dev) + list(lock)}
    ids = sorted(dev); X = np.stack([vec(f) for f in ids]); y = np.array([dev[f]["violation"] for f in ids], float)
    loo = {}
    for i, fid in enumerate(ids):                          # leave-one-out
        m = np.arange(len(ids)) != i; pr, _ = fit(X[m], y[m], a.l2); loo[fid] = float(pr(X[i:i + 1])[0])
    silent = [f for f in ids if not sens[f]]
    print("датчики молчат на:", " ".join(f"{f}{'V' if dev[f]['violation'] else '-'}:{loo[f]:.2f}" for f in sorted(silent, key=lambda f: -loo[f])))
    n_clean = sum(not g["violation"] for g in dev.values()); best = None
    for thr in [1.01, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        s = score(dev, preds_for(dev, sens, loo, thr))
        print(f"LOO порог {thr:.2f}: найдено {s['tp']}/18, ложных {s['fp']}/{n_clean}, баллы {s['pts'] / s['pts_max']:.3f}")
        if s["fp"] <= 2 and (best is None or s["pts"] > best[1]): best = (thr, s["pts"])
    pr, w = fit(X, y, a.l2)
    print("веса (стандартизованные):", ", ".join(f"{c}={v:+.2f}" for c, v in sorted(zip(cols, w), key=lambda x: -abs(x[1]))[:10]))
    thr = best[0] if best else 1.01; lp = {f: float(pr(vec(f)[None])[0]) for f in lock}
    s = score(lock, preds_for(lock, sens, lp, thr))
    print(f"ОТЛОЖЕННЫЕ (порог {thr:.2f}, выбран по LOO при ложных ≤2): найдено {s['tp']}/3, ложных {s['fp']}/4, баллы {s['pts'] / s['pts_max']:.3f};",
          "p:", {f: round(v, 2) for f, v in lp.items()})


if __name__ == "__main__":
    main()
