"""Regression checks for organizer requirements; no model download needed."""
import importlib
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


@pytest.fixture
def ml(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "solution"))
    return importlib.import_module("pipeline"), importlib.import_module("sensors_core")


def test_sensor_failure_must_not_be_reported_as_clean_video(ml, monkeypatch):
    _, core = ml
    monkeypatch.setattr(core.D, "cashier_track", lambda _: (None, []))
    monkeypatch.setattr(core, "SENSORS", ["broken"])
    def fail(*args):
        raise RuntimeError("sensor unavailable")
    monkeypatch.setattr(core.importlib, "import_module", lambda _: SimpleNamespace(run=fail))
    with pytest.raises(RuntimeError, match="sensor unavailable"):
        core.run_core([], [])


def test_all_detected_types_reach_api_output(ml, monkeypatch):
    pipeline, core = ml
    frame = np.zeros((32, 32, 3), np.uint8)
    monkeypatch.setattr(pipeline, "sample_frames", lambda _: iter([(0.0, frame), (1.0, frame)]))
    types = ["door_open", "pc_unlocked", "valuables_on_desk", "personal_items", "outsider"]
    monkeypatch.setattr(core, "run_core", lambda *args: [{"type": code, "t": 0.0} for code in types])
    monkeypatch.setenv("ML_POLICY", "sensors")
    result = pipeline.analyze("sample.mov", None, lambda _: [])
    assert len(result["violations"]) == len(types)
    assert all(1 <= len(v["evidence_frames"]) <= 3 for v in result["violations"])


def test_default_analyzer_does_not_require_private_reference_frames(ml, monkeypatch):
    inference = importlib.import_module("solution.inference")
    monkeypatch.setenv("ML_POLICY", "sensors")
    def unavailable_geometry():
        raise FileNotFoundError("private reference images not distributed")
    monkeypatch.setattr(inference, "Geometry", unavailable_geometry)
    monkeypatch.setattr(inference, "Detector", lambda: object())
    assert inference.Analyzer().geo is None
