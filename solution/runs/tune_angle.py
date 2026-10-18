"""Оценка датчиков (режим «только датчики», как сервис с FALLBACK_VISIT_S=0) на трёх наборах:
исходные кадры, искажение A (для настройки), искажение B (для проверки). Пороги задаются через --set модуль.ПАРАМ=значение.
  python runs/tune_angle.py                                  # текущие пороги
  python runs/tune_angle.py --set outsider_rel.NEAR=0.5 --set items_new.P.diff=16
Печатает баллы по каждому набору (dev+lockbox вместе и раздельно) и список изменений относительно исходных порогов."""
import argparse, importlib, json, os, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(HERE)); os.chdir(HERE)
import bench
from evaluate import load_labels, score

SENS = ["door_tracks", "laptop_screen", "outsider_rel", "desk_left", "items_new"]
SETS = {"orig": ("cache_adapt", "frames"), "A": ("cache_wA", "frames_wA"), "B": ("cache_wB", "frames_wB"), "C": ("cache_wC", "frames_wC")}


def apply(sets):
    for s in sets:
        k, v = s.split("="); parts = k.split("."); m = importlib.import_module("sensors." + parts[0]); v = float(v)
        if len(parts) == 2: setattr(m, parts[1], v)
        else: getattr(m, parts[1])[parts[2]] = v


FALLBACK = 0.0


def evaluate(gt, cache, frames):
    os.environ["SENSOR_CACHE"], os.environ["SENSOR_FRAMES"] = cache, frames
    preds, raw = {}, {}
    for fid in gt:
        ev = bench.run_sensors(SENS, fid); raw[fid] = [(e["sensor"], e["type"], round(e["t"], 1)) for e in ev]
        if ev:
            e = bench.pick(ev, "prio", {}); preds[fid] = {"verdict": "violation", "violations": [{"type_code": e["type"], "timecode_sec": e["t"]}]}
        else:
            e = None
            if FALLBACK > 0:
                import sensors_core
                e = sensors_core.visit_event(json.load(open(HERE / cache / f"{fid}.json"))["frames"], FALLBACK)
            preds[fid] = {"verdict": "violation", "violations": [{"type_code": "procedure", "timecode_sec": e["t"]}]} if e else {"verdict": "no_violation", "violations": []}
    return preds, raw


def main():
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--set", action="append", default=[]); ap.add_argument("--only", default="orig,A,B")
    ap.add_argument("--dump"); ap.add_argument("--fallback", type=float, default=0.0); a = ap.parse_args(); apply(a.set)
    global FALLBACK; FALLBACK = a.fallback
    dev, lock = load_labels("dev"), load_labels("lockbox"); allg = {**dev, **lock}; dump = {}
    tot = []
    for name in a.only.split(","):
        preds, raw = evaluate(allg, *SETS[name]); dump[name] = raw
        r = []
        for sp, g in (("dev", dev), ("lock", lock), ("all", allg)):
            s = score(g, {k: preds[k] for k in g}); r.append(f"{sp} {s['pts'] / s['pts_max']:.3f} ({s['tp']}/{s['tp'] + s['fn']} найд, {s['fp']} лож)")
            if sp == "all": tot.append(s["pts"] / s["pts_max"])
        print(f"{name:5s} | " + " | ".join(r), flush=True)
    print("ИТОГ min/mean:", f"{min(tot):.3f} / {sum(tot) / len(tot):.3f}", a.set)
    if a.dump: json.dump(dump, open(a.dump, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
