"""Локальный оценщик по метрикам жюри (v2, после ревью).

Главное событие = первое в списке violations (сервис кладёт самое уверенное первым) — тип и таймкод
считаются ТОЛЬКО по нему: лишние события больше не дают бесплатных бонусов.

  python evaluate.py --preds runs/v0 [--split dev|lockbox|all]
Баллы — грубая модель оценки жюри (допущение): видео с нарушением: 25 за вердикт + 15 за таймкод ±2 с
+ 15 за тип + 15 за норму (= верный тип); чистое видео: 25 за верный вердикт.
"""
import argparse, csv, json, pathlib, sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
LABELS = ROOT / "research" / "notes" / "labels_typed.csv"
TITLE2CODE = {v["title"]: k for k, v in json.load(open(pathlib.Path(__file__).resolve().parent / "rules.json", encoding="utf-8")).items() if not k.startswith("_")}
LOCKBOX = {"08", "10", "11", "12", "13", "14", "15"}          # засвечен ревью: использован один раз
VIEW = {**{f: "V1" for f in ["33", "34", "35", "37", "38", "39", "40", "43", "44", "45", "46"]},
        **{f: "V2" for f in ["08", "10", "11", "12", "13", "14", "15"]},
        **{f: "V3" for f in ["01", "02", "03", "04", "05", "16", "17", "18", "19"]},
        **{f: "V4" for f in ["30", "31", "32"]}, **{f: "V5" for f in ["20", "21", "22", "23", "24", "25", "26", "27", "28", "29"]}}


def load_labels(split):
    out = {}
    for r in csv.DictReader(open(LABELS, encoding="utf-8")):
        fid = r["file_id"]
        if split == "dev" and fid in LOCKBOX: continue
        if split == "lockbox" and fid not in LOCKBOX: continue
        ev = [(e.split("@")[0], float(e.split("@")[1])) for e in r["events"].split(";") if e]
        out[fid] = {"violation": r["verdict"] == "violation", "events": ev}
    return out


def fbeta(tp, fp, fn, b=2.0):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return (1 + b * b) * p * r / (b * b * p + r) if p + r else 0.0


def score(gt, preds):
    tp = fp = fn = tn = 0; dts = []; type_ok = type_n = 0; pts = 0.0; pts_max = 0.0; per_view = defaultdict(lambda: [0, 0, 0, 0]); rows = []
    for fid, g in sorted(gt.items()):
        pred = preds.get(fid, {"verdict": "no_violation", "violations": []})
        pv = pred.get("verdict") == "violation"; v = per_view[VIEW.get(fid, "?")]
        if g["violation"]:
            pts_max += 70
            if pv:
                tp += 1; v[0] += 1; pts += 25
                main = pred["violations"][0] if pred.get("violations") else None
                if main:
                    gtype, gsec = min(g["events"], key=lambda e: abs(e[1] - main["timecode_sec"]))
                    dt = abs(main["timecode_sec"] - gsec); dts.append(dt); pts += 15 * (dt <= 2)
                    code = main.get("type_code") or TITLE2CODE.get(main.get("type"), main.get("type"))
                    if gtype != "unknown":
                        type_n += 1; ok = code == gtype; type_ok += ok; pts += 30 * ok
                    rows.append(f"{fid}: V→V главное={code}@{main['timecode_sec']:g}  истина={g['events']}")
            else:
                fn += 1; v[1] += 1; rows.append(f"{fid}: ПРОПУСК  истина={g['events']}")
        else:
            pts_max += 25
            if pv: fp += 1; v[2] += 1; rows.append(f"{fid}: ЛОЖНАЯ  {[(x.get('type_code'), x['timecode_sec']) for x in pred.get('violations', [])][:2]}")
            else: tn += 1; v[3] += 1; pts += 25
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, dts=dts, type_ok=type_ok, type_n=type_n, pts=pts, pts_max=pts_max, per_view=dict(per_view), rows=rows)


def report(name, s, n_clean):
    viol = s["tp"] + s["fn"]; n = viol + n_clean
    d = sorted(s["dts"]); med = d[len(d) // 2] if d else float("nan")
    print(f"[{name}] recall {s['tp']}/{viol} | ложные {s['fp']}/{n_clean} | acc {(s['tp'] + s['tn']) / n:.2f} | F2 {fbeta(s['tp'], s['fp'], s['fn']):.2f} | "
          f"тип верно {s['type_ok']}/{s['type_n']} (recall с верным типом {s['type_ok']}/{viol}) | таймкод ±2с {sum(x <= 2 for x in d)}/{len(d)}, медиана {med:.1f}с | "
          f"баллы {s['pts']:.0f}/{s['pts_max']:.0f} = {s['pts'] / max(1, s['pts_max']):.2f}")


def main():
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--preds", required=True); ap.add_argument("--split", default="dev")
    ap.add_argument("--quiet", action="store_true"); a = ap.parse_args()
    gt = load_labels(a.split); P = pathlib.Path(a.preds)
    preds = {f.stem: json.load(open(f, encoding="utf-8")) for f in P.glob("*.json")}
    s = score(gt, preds); n_clean = sum(not g["violation"] for g in gt.values())
    if not a.quiet: print("\n".join(s["rows"]))
    report(P.name, s, n_clean)
    triv = {f: {"verdict": "violation", "violations": []} for f in gt}       # «всегда нарушение», без типа и таймкода
    report("всегда нарушение", score(gt, triv), n_clean)
    print("по ракурсам [найдено, пропущено, ложных, верно чистых]:", s["per_view"])
    t = [p.get("processing_sec") for p in preds.values() if p.get("processing_sec")]
    if t: print(f"время: макс {max(t):.0f} с, среднее {sum(t) / len(t):.0f} с (лимит 300)")


if __name__ == "__main__":
    main()
