import base64
import io
import math
import os
from pathlib import Path

from PIL import Image

from service.schemas import Analysis


def load_analyzer(threads):
    # Set before imports; the constructor currently overrides torch's thread count.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    # cv2.setNumThreads does NOT control FFmpeg's separate per-video decoder pool.
    os.environ["OPENCV_FFMPEG_THREADS"] = "1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(threads)
    import cv2
    import torch
    from solution.inference import Analyzer
    cv2.setNumThreads(1)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    analyzer = Analyzer()
    torch.set_num_threads(threads)
    return analyzer


def video_duration(path: Path):
    import cv2
    with path.open("rb") as file:
        header = file.read(32)
    if len(header) < 8 or header[4:8] not in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip"):
        raise ValueError("File is not a MOV/QuickTime video")
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not ok or frame is None or not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError("Video is empty, damaged or uses an unsupported codec")
        duration = count / fps
        if not math.isfinite(duration):
            raise ValueError("Invalid video duration")
        return duration
    finally:
        cap.release()


def normalize_result(raw, duration):
    result = Analysis.model_validate({"verdict": raw.get("verdict"), "violations": raw.get("violations")})
    for violation in result.violations:
        if violation.timecode_sec > duration + 0.1:
            raise ValueError("Violation timestamp is outside the video")
        for evidence in violation.evidence_frames:
            if evidence.timestamp_sec > duration + 0.1:
                raise ValueError("Evidence timestamp is outside the video")
            data = base64.b64decode(evidence.image_base64, validate=True)
            with Image.open(io.BytesIO(data)) as source:
                if source.format != "JPEG" or source.width * source.height > 40_000_000:
                    raise ValueError("Evidence must be a reasonably sized JPEG")
                image = source.convert("RGB")
                image.thumbnail((640, 640))
                out = io.BytesIO()
                image.save(out, format="JPEG", quality=80, optimize=True)
                evidence.image_base64 = base64.b64encode(out.getvalue()).decode("ascii")
    return result.model_dump()
