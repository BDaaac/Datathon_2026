"""Запуск сервиса на ноутбуке без Docker: читает service/.env (ключ не печатает), 5 ML-процессов, детектор onnxruntime.
    .venv/Scripts/python tools/serve.py        (из корня проекта)"""
import os, pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
for line in (ROOT / "service" / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip("'\""))
os.environ.setdefault("DET_BACKEND", "onnx"); os.environ.setdefault("OPENCV_FFMPEG_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1"); os.environ["PYTHONIOENCODING"] = "utf-8"
port = os.environ.get("SERVICE_PORT", "8765")
py = ROOT / "service" / ".venv" / "Scripts" / "python.exe"
print(f"сервис: http://127.0.0.1:{port}  workers={os.environ.get('SERVICE_WORKERS')} threads={os.environ.get('SERVICE_THREADS_PER_WORKER')} det={os.environ['DET_BACKEND']}", flush=True)
sys.exit(subprocess.call([str(py), "-m", "uvicorn", "service.app:create_app", "--factory", "--host", "127.0.0.1", "--port", port, "--workers", "1", "--no-access-log"], cwd=ROOT))
