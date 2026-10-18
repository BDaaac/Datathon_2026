import base64
import io
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from service.app import create_app
from service.config import Settings
from service.tests.fakes import Broken, Crashed, NoViolation, Slow, Violation


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "sample.mov"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.release()
    return path.read_bytes()


def settings(tmp_path, **kwargs):
    return Settings(api_key="test-secret", data_dir=tmp_path / "data", workers=2, **kwargs)


def wait_job(client, jid, timeout=20, prefix="/api/v1", headers=None):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        response = client.get(f"{prefix}/jobs/{jid}", headers=headers or {"X-API-Key": "test-secret"})
        assert response.status_code == 200, response.text
        result = response.json()
        if result["status"] != "processing":
            return result
        time.sleep(0.05)
    pytest.fail(f"job {jid} did not finish")


def submit(client, video, name="sample.mov", prefix="/api/v1"):
    response = client.post(f"{prefix}/analyze", files={"file": (name, video, "video/quicktime")},
                           headers={"X-API-Key": "test-secret"} if prefix == "/api/v1" else {})
    assert response.status_code == (200 if prefix == "/api/v1" else 202), response.text   # API жюри — 200, UI — 202
    assert set(response.json()) == {"job_id"}
    return response.json()["job_id"]


def test_auth_before_upload_and_unknown_job(tmp_path):
    with TestClient(create_app(settings(tmp_path), NoViolation)) as client:
        for headers in ({}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}):
            assert client.post("/api/v1/analyze", content=b"bad", headers=headers).status_code == 401
            assert client.get("/api/v1/jobs/unknown", headers=headers).status_code == 401
        assert client.get("/api/v1/jobs/unknown", headers={"X-API-Key": "test-secret"}).status_code == 404


def test_results_survive_restart_and_videos_are_deleted(tmp_path, video):
    cfg = settings(tmp_path)
    with TestClient(create_app(cfg, NoViolation)) as client:
        jid = submit(client, video)
        result = wait_job(client, jid)
        assert result == {"job_id": jid, "status": "done", "verdict": "no_violation", "violations": []}
        assert not list((cfg.data_dir / "uploads").iterdir())
    with TestClient(create_app(cfg, NoViolation)) as client:
        assert wait_job(client, jid) == result


def test_evidence_is_jpeg_and_resized(tmp_path, video):
    with TestClient(create_app(settings(tmp_path), Violation)) as client:
        result = wait_job(client, submit(client, video))
        assert result["status"] == "done", result
        evidence = result["violations"][0]["evidence_frames"][0]
        assert evidence["timestamp_sec"] == 0.2
        with Image.open(io.BytesIO(base64.b64decode(evidence["image_base64"], validate=True))) as image:
            assert image.format == "JPEG"
            assert max(image.size) == 640


@pytest.mark.parametrize("name,data", [("x.txt", b"x"), ("broken.mov", b"not video"), ("empty.mov", b"")])
def test_invalid_input_is_failed_job(tmp_path, name, data):
    cfg = settings(tmp_path)
    with TestClient(create_app(cfg, NoViolation)) as client:
        result = wait_job(client, submit(client, data, name))
        assert result["status"] == "failed"
        assert result["error"]
        assert "verdict" not in result
        assert not list((cfg.data_dir / "uploads").iterdir())


@pytest.mark.parametrize("factory", [Broken, Crashed])
def test_inference_failure_and_worker_crash_are_terminal(tmp_path, video, factory):
    cfg = settings(tmp_path)
    with TestClient(create_app(cfg, factory)) as client:
        result = wait_job(client, submit(client, video))
        assert result["status"] == "failed", result
        assert result["error"]
        assert not list((cfg.data_dir / "uploads").iterdir())


def test_timeout_kills_worker_and_cleans_input(tmp_path, video):
    cfg = settings(tmp_path, job_timeout=2)
    with TestClient(create_app(cfg, Slow)) as client:
        start = time.monotonic()
        jid = submit(client, video)
        assert time.monotonic() - start < 1
        result = wait_job(client, jid)
        assert result["status"] == "failed"
        assert "time" in result["error"].lower()
        assert not list((cfg.data_dir / "uploads").iterdir())


def test_ui_routes_and_api_results_are_separate(tmp_path, video):
    with TestClient(create_app(settings(tmp_path), NoViolation)) as client:
        jid = submit(client, video, prefix="/ui")
        assert wait_job(client, jid, prefix="/ui")["status"] == "done"
        api_job = submit(client, video)
        assert client.get(f"/ui/jobs/{api_job}").status_code == 404
        assert client.get("/").status_code == 200
        assert client.get("/health").status_code in (200, 503)


def test_upload_limit_and_missing_field_are_json(tmp_path, video):
    with TestClient(create_app(settings(tmp_path, max_upload_bytes=100), NoViolation)) as client:
        response = client.post("/ui/analyze", files={"file": ("large.mov", video)})
        assert response.status_code == 413
        assert response.json()["detail"]
        response = client.post("/ui/analyze", files={"other": ("sample.mov", b"abc")})
        assert response.status_code == 422


def test_interrupted_job_is_failed_on_restart(tmp_path, video):
    cfg = settings(tmp_path)
    with TestClient(create_app(cfg, Slow)) as client:
        jid = submit(client, video)
    with TestClient(create_app(cfg, NoViolation)) as client:
        assert wait_job(client, jid)["status"] == "failed"
        assert not list((cfg.data_dir / "uploads").iterdir())
