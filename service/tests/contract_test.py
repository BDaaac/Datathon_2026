"""Real HTTP acceptance test. No mock mode: use the actual service and local video."""
import argparse
import base64
import concurrent.futures
import io
import json
import os
import time
from pathlib import Path

import httpx
import jsonschema
from PIL import Image

FRAME = {"type": "object", "additionalProperties": False,
         "required": ["timestamp_sec", "image_base64"], "properties": {
             "timestamp_sec": {"type": "number", "minimum": 0}, "image_base64": {"type": "string", "minLength": 1}}}
VIOLATION = {"type": "object", "additionalProperties": False,
             "required": ["type", "timecode_sec", "description", "evidence_frames"], "properties": {
                 "type": {"type": "string", "minLength": 1}, "timecode_sec": {"type": "number", "minimum": 0},
                 "description": {"type": "string", "minLength": 1},
                 "evidence_frames": {"type": "array", "minItems": 1, "maxItems": 3, "items": FRAME}}}
DONE_SCHEMA = {"type": "object", "additionalProperties": False,
               "required": ["job_id", "status", "verdict", "violations"], "properties": {
                   "job_id": {"type": "string", "minLength": 1}, "status": {"const": "done"},
                   "verdict": {"enum": ["violation", "no_violation"]},
                   "violations": {"type": "array", "items": VIOLATION}}}


def validate_done(result):
    jsonschema.validate(result, DONE_SCHEMA)
    assert bool(result["violations"]) == (result["verdict"] == "violation"), result
    for violation in result["violations"]:
        for frame in violation["evidence_frames"]:
            data = base64.b64decode(frame["image_base64"], validate=True)
            with Image.open(io.BytesIO(data)) as image:
                assert image.format == "JPEG" and max(image.size) <= 640
                image.verify()


def run_job(url, key, filename, body, deadline, expected="done", prefix="/api/v1"):
    start = time.monotonic()
    with httpx.Client(base_url=url, headers={"X-API-Key": key},
                      timeout=httpx.Timeout(deadline, connect=min(10, deadline)), trust_env=False) as client:
        response = client.post(f"{prefix}/analyze", files={"file": (filename, body, "video/quicktime")})
        assert response.status_code in (200, 202), (response.status_code, response.text)
        accepted = response.json()
        assert set(accepted) == {"job_id"} and isinstance(accepted["job_id"], str) and accepted["job_id"]
        jid = accepted["job_id"]
        post_sec = time.monotonic() - start
        while time.monotonic() - start < deadline:
            response = client.get(f"{prefix}/jobs/{jid}", timeout=max(0.1, deadline - (time.monotonic() - start)))
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["job_id"] == jid
            assert result["status"] in ("processing", "done", "failed")
            if result["status"] != "processing":
                assert result["status"] == expected, result
                if expected == "done":
                    validate_done(result)
                else:
                    assert set(result) == {"job_id", "status", "error"} and result["error"]
                elapsed = time.monotonic() - start
                assert elapsed < deadline
                return {"job_id": jid, "status": expected, "post_sec": round(post_sec, 2), "total_sec": round(elapsed, 2)}
            assert set(result) == {"job_id", "status"}, result
            time.sleep(0.5)
        raise AssertionError(f"{jid}: no result within {deadline}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--video", type=Path, required=True, help="Local longest test video, e.g. Х-14.mov")
    parser.add_argument("--parallel", type=int, default=5)
    parser.add_argument("--deadline", type=float, default=300)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    key = os.environ.get("SERVICE_API_KEY")
    if not key:
        parser.error("Set SERVICE_API_KEY (do not put the key on the command line)")
    assert args.parallel >= 5, "Acceptance requires at least five concurrent uploads"
    with httpx.Client(base_url=args.url, timeout=10, trust_env=False) as client:
        for headers in ({}, {"X-API-Key": key + "-wrong"}):
            for method, path in (("POST", "/api/v1/analyze"), ("GET", "/api/v1/jobs/missing")):
                response = client.request(method, path, headers=headers)
                assert response.status_code == 401, response.text
        end = time.monotonic() + 120
        while client.get("/health").status_code != 200:
            assert time.monotonic() < end, "Workers did not become ready"
            time.sleep(1)
    def attempt(*arguments):
        started = time.monotonic()
        try:
            result = run_job(*arguments)
        except (AssertionError, httpx.HTTPError, ValueError) as exc:
            result = {"status": "test_failed", "error": str(exc), "total_sec": round(time.monotonic() - started, 2)}
        print(json.dumps(result), flush=True)
        return result

    checks = []
    for filename, body in (("bad.txt", b"bad"), ("broken.mov", b"not a video"), ("empty.mov", b"")):
        checks.append(attempt(args.url, key, filename, body, 30, "failed"))
    payload = args.video.read_bytes()
    checks.append(attempt(args.url, key, args.video.name, payload, args.deadline))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(attempt, args.url, key, args.video.name, payload, args.deadline) for _ in range(args.parallel)]
        concurrent_results = [future.result() for future in futures]
    passed = all(x["status"] != "test_failed" for x in checks + concurrent_results)
    report = {"passed": passed, "checks": checks, "parallel": concurrent_results, "max_parallel_sec": max(x["total_sec"] for x in concurrent_results)}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
