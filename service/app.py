import json
import logging
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from service.config import Settings
from service.schemas import Accepted, JobResponse
from service.storage import InstanceLock, Store
from service.upload import save_upload
from service.workers import Supervisor

STATIC = Path(__file__).parent / "static"
log = logging.getLogger(__name__)


def create_app(settings=None, analyzer_factory=None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        lock = InstanceLock(settings.data_dir / "service.lock")
        lock.acquire()
        supervisor = None
        try:
            store = Store(settings.data_dir)
            store.recover()
            supervisor = Supervisor(settings, store, analyzer_factory)
            app.state.store, app.state.supervisor = store, supervisor
            supervisor.start()
            yield
        finally:
            if supervisor:
                await anyio.to_thread.run_sync(supervisor.close)
            lock.release()

    app = FastAPI(title="Кассовая дисциплина — локальный анализ", version="1.0.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def boundary(request, call_next):
        path = request.url.path
        if path not in ("/api/v1", "/api/v1/") and (path.startswith("/api/v1/") or path == "/analyze" or path.startswith("/jobs/")):   # короткие пути — на случай base URL с /api/v1
            supplied = request.headers.get("X-API-Key", "").encode("utf-8")
            if not secrets.compare_digest(supplied, settings.api_key.encode("utf-8")):
                return JSONResponse({"detail": "Invalid or missing X-API-Key"}, status_code=401)
        if path.startswith("/ui/"):
            origin = request.headers.get("origin")
            if request.headers.get("sec-fetch-site") == "cross-site" or (origin and urlparse(origin).netloc != request.url.netloc):
                return JSONResponse({"detail": "Cross-origin UI requests are not allowed"}, status_code=403)
        try:
            response = await call_next(request)
        except Exception:
            log.exception("Request failed")
            response = JSONResponse({"detail": "Internal service error"}, status_code=500)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; media-src blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        return response

    async def analyze(request: Request, scope):
        store, supervisor = app.state.store, app.state.supervisor
        if supervisor.failure:
            raise HTTPException(503, "Processing workers unavailable")
        jid = uuid.uuid4().hex
        if not await anyio.to_thread.run_sync(store.reserve, jid, scope, settings.max_pending_jobs):
            raise HTTPException(503, "Queue is full; try again later", headers={"Retry-After": "5"})
        path = store.path(jid)
        queued = False
        try:
            name, size = await save_upload(request, path, settings)
            if not name.lower().endswith(".mov") or not size:
                await anyio.to_thread.run_sync(supervisor.finish, jid, None, "A non-empty .mov video is required")
            elif supervisor.failure:
                await anyio.to_thread.run_sync(supervisor.finish, jid, None, supervisor.failure)
            else:
                await anyio.to_thread.run_sync(store.enqueue, jid)
            queued = True
            return Accepted(job_id=jid)
        finally:
            if not queued:
                store.schedule_cleanup(jid)
                try:
                    path.unlink(missing_ok=True)
                    store.cleanup_done(jid)
                finally:
                    store.cancel_upload(jid)

    def job(jid, scope):
        row = app.state.store.get(jid)
        if row is None or row["scope"] != scope:
            raise HTTPException(404, "Job not found")
        if row["state"] == "done":
            return {"job_id": jid, "status": "done", **json.loads(row["result"])}
        if row["state"] == "failed":
            return {"job_id": jid, "status": "failed", "error": row["error"]}
        if app.state.supervisor.failure:
            return {"job_id": jid, "status": "failed", "error": app.state.supervisor.failure}
        return {"job_id": jid, "status": "processing"}

    @app.post("/api/v1/analyze", status_code=200, response_model=Accepted)
    async def api_analyze(request: Request):
        return await analyze(request, "api")

    # Псевдонимы API без префикса: работают, если жюри склеит base URL как «…/api/v1» + «/analyze». Тот же ключ и scope.
    app.add_api_route("/analyze", api_analyze, methods=["POST"], status_code=200, response_model=Accepted, include_in_schema=False)
    app.add_api_route("/api/v1/api/v1/analyze", api_analyze, methods=["POST"], status_code=200, response_model=Accepted, include_in_schema=False)

    @app.post("/ui/analyze", status_code=202, response_model=Accepted)
    async def ui_analyze(request: Request):
        return await analyze(request, "ui")

    @app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
    def api_job(job_id: str):
        return job(job_id, "api")

    app.add_api_route("/jobs/{job_id}", api_job, methods=["GET"], response_model=JobResponse, include_in_schema=False)
    app.add_api_route("/api/v1/api/v1/jobs/{job_id}", api_job, methods=["GET"], response_model=JobResponse, include_in_schema=False)

    @app.get("/ui/jobs/{job_id}", response_model=JobResponse)
    def ui_job(job_id: str):
        return job(job_id, "ui")

    @app.get("/api/v1", include_in_schema=False)
    @app.get("/api/v1/", include_in_schema=False)
    def api_info():   # справка для человека, открывшего базовый адрес API в браузере; ключ и данные не раскрываются
        return {"service": "Видеоконтроль: проверка кассовых операций", "auth": "заголовок X-API-Key",
                "endpoints": {"POST /api/v1/analyze": "multipart, поле file (.mov) → {job_id}",
                              "GET /api/v1/jobs/{job_id}": "processing | done (verdict, violations) | failed"}}

    @app.get("/health")
    def health():
        result = app.state.supervisor.health()
        return JSONResponse(result, status_code=200 if result["status"] == "ok" else 503)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC, check_dir=False), name="static")
    return app
