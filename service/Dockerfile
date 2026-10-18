FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 HF_HOME=/opt/huggingface HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app
COPY service/requirements.txt service/requirements-ml.txt service/requirements-export.txt /app/service/
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<3" "torchvision>=0.21,<1"
RUN pip install -r service/requirements.txt -r service/requirements-ml.txt -r service/requirements-export.txt
COPY solution/ /app/solution/
COPY service/prepare_weights.py /app/service/prepare_weights.py
ARG DFINE_REVISION=f79e65b5fbb33ceb9d3ebba042955d7410c608f8
RUN DFINE_REVISION=${DFINE_REVISION} python -m service.prepare_weights
RUN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python solution/export_onnx.py
COPY service/ /app/service/
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENCV_FFMPEG_THREADS=1 DET_BACKEND=onnx SERVICE_DATA_DIR=/data SERVICE_WORKERS=5 SERVICE_THREADS_PER_WORKER=2
# Fail the build if the packaged assets/weights cannot initialize offline.
RUN python -c "import importlib; from service.adapter import load_analyzer; a=load_analyzer(1); from sensors_core import SENSORS; [importlib.import_module('sensors.'+n) for n in SENSORS]; assert a.det.backend == 'onnxruntime'; print('Offline ONNX model and all sensors initialized')"
RUN useradd --uid 10001 --create-home appuser && mkdir -p /data && chown -R appuser:appuser /data /opt/huggingface
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2)"
CMD ["python", "-m", "uvicorn", "service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
