import base64
import io
import os
import time

from PIL import Image


class NoViolation:
    def run(self, path):
        return {"verdict": "no_violation", "violations": [], "processing_sec": 0.1}


class Violation:
    def run(self, path):
        out = io.BytesIO()
        Image.new("RGB", (1280, 960), "white").save(out, "JPEG")
        return {"verdict": "violation", "violations": [{
            "type": "Пример", "timecode_sec": 0.2, "description": "Видимый признак. Норма: источник ML.",
            "evidence_frames": [{"timestamp_sec": 0.2, "image_base64": base64.b64encode(out.getvalue()).decode()}]
        }], "processing_sec": 0.1}


class Broken:
    def run(self, path):
        raise ValueError("Controlled inference failure")


class Slow:
    def run(self, path):
        time.sleep(60)
        return {"verdict": "no_violation", "violations": []}


class Crashed:
    def run(self, path):
        os._exit(7)

