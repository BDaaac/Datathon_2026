import math

import pytest

from service.adapter import normalize_result
from service.storage import InstanceLock
from service.tests.fakes import Violation


@pytest.mark.parametrize("change", ["contradiction", "no_frames", "too_many_frames", "nan", "bad_jpeg", "out_of_range"])
def test_invalid_model_output_cannot_become_done(change):
    result = Violation().run(None)
    violation = result["violations"][0]
    if change == "contradiction":
        result["verdict"] = "no_violation"
    elif change == "no_frames":
        violation["evidence_frames"] = []
    elif change == "too_many_frames":
        violation["evidence_frames"] *= 4
    elif change == "nan":
        violation["timecode_sec"] = math.nan
    elif change == "bad_jpeg":
        violation["evidence_frames"][0]["image_base64"] = "aGVsbG8="
    else:
        violation["timecode_sec"] = 100
    with pytest.raises((ValueError, OSError)):
        normalize_result(result, duration=1)


def test_second_server_cannot_recover_live_jobs(tmp_path):
    first, second = InstanceLock(tmp_path / "service.lock"), InstanceLock(tmp_path / "service.lock")
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="already in use"):
            second.acquire()
    finally:
        first.release()
    second.acquire()
    second.release()
