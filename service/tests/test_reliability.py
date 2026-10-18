from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from service.config import Settings
from service.storage import Store
from service.workers import Supervisor


def test_supervisor_fault_finishes_jobs_and_deletes_inputs(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.add("job", "api")
    store.path("job").write_bytes(b"video")
    supervisor = Supervisor(Settings(api_key="key", data_dir=tmp_path), store)
    def fail():
        raise RuntimeError("injected supervisor fault")
    monkeypatch.setattr(supervisor, "tick", fail)
    supervisor.run()
    assert store.get("job")["state"] == "failed"
    assert not store.path("job").exists()


def test_failed_cleanup_is_retried_after_terminal_result(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.add("job", "api")
    store.path("job").write_bytes(b"video")
    supervisor = Supervisor(Settings(api_key="key", data_dir=tmp_path), store)
    original = store.delete_video
    attempts = []
    def delete(jid):
        attempts.append(jid)
        if len(attempts) == 1:
            raise PermissionError("transient file lock")
        original(jid)
    monkeypatch.setattr(store, "delete_video", delete)
    supervisor.finish("job", {"verdict": "no_violation", "violations": []})
    supervisor.tick()
    assert not store.path("job").exists()


def test_capacity_reservation_is_atomic(tmp_path):
    store = Store(tmp_path)
    def reserve(number):
        return store.reserve(str(number), "api", 1)
    with ThreadPoolExecutor(max_workers=10) as pool:
        admitted = list(pool.map(reserve, range(10)))
    assert sum(admitted) == 1


def test_reserved_upload_cannot_be_claimed_until_validated(tmp_path):
    store = Store(tmp_path)
    assert store.reserve("job", "api", 1)
    assert store.claim() is None
    store.finish("job", error="Invalid extension")
    assert store.claim() is None
    assert store.get("job")["state"] == "failed"


def test_pinned_weights_have_offline_main_alias(tmp_path):
    from service import prepare_weights
    snapshot = tmp_path / "models--example" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    prepare_weights.pin_default_revision(snapshot)
    assert (snapshot.parent.parent / "refs" / "main").read_text() == "a" * 40
