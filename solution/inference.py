"""Стабильный интерфейс ML-части для сервиса. Сервис вызывает только это; внутренности (pipeline/decide) меняются в A/B.

    from solution.inference import Analyzer
    an = Analyzer()                       # грузит модели один раз (≈5–10 с)
    result = an.run("/path/video.mov")    # dict строго по схеме API (без job_id/status — их добавляет сервис)

Контракт результата:
    {"verdict": "violation" | "no_violation",
     "violations": [{"type": str, "timecode_sec": float, "description": str,
                     "evidence_frames": [{"timestamp_sec": float, "image_base64": str}]}],   # 1–3 JPEG
     "processing_sec": float}
Исключения пробрасываются наружу — сервис превращает их в status=failed.
"""
import os, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pipeline import Geometry, Detector, analyze  # noqa: E402

SCHEMA_KEYS = ("type", "timecode_sec", "description", "evidence_frames")


class Analyzer:
    def __init__(self):
        self.geo = Geometry() if os.environ.get("ML_POLICY", "sensors") == "h5" else None
        self.det = Detector()

    def run(self, video_path: str) -> dict:
        r = analyze(video_path, self.geo, self.det)
        viol = [{k: v[k] for k in SCHEMA_KEYS} for v in r["violations"]]
        return {"verdict": "violation" if viol else "no_violation", "violations": viol,
                "processing_sec": r.get("processing_sec")}
