import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


class Store:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.uploads = data_dir / "uploads"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.db = data_dir / "jobs.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, scope TEXT NOT NULL, state TEXT NOT NULL,
                created REAL NOT NULL, result TEXT, error TEXT)""")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(state, created)")
            db.execute("CREATE TABLE IF NOT EXISTS cleanup (id TEXT PRIMARY KEY)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def add(self, jid, scope):
        with self.connect() as db:
            db.execute("INSERT INTO jobs(id,scope,state,created) VALUES(?,?,?,?)", (jid, scope, "queued", time.time()))

    def reserve(self, jid, scope, capacity):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT count(*) FROM jobs WHERE state IN ('uploading','queued','running')").fetchone()[0]
            if count >= capacity:
                return False
            db.execute("INSERT INTO jobs(id,scope,state,created) VALUES(?,?,?,?)", (jid, scope, "uploading", time.time()))
            return True

    def enqueue(self, jid):
        with self.connect() as db:
            db.execute("UPDATE jobs SET state='queued', created=? WHERE id=? AND state='uploading'", (time.time(), jid))

    def cancel_upload(self, jid):
        with self.connect() as db:
            db.execute("DELETE FROM jobs WHERE id=? AND state='uploading'", (jid,))

    def schedule_cleanup(self, jid):
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO cleanup(id) VALUES(?)", (jid,))

    def cleanup_pending(self):
        with self.connect() as db:
            return [row[0] for row in db.execute("SELECT id FROM cleanup")]

    def cleanup_done(self, jid):
        with self.connect() as db:
            db.execute("DELETE FROM cleanup WHERE id=?", (jid,))

    def get(self, jid):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
            return dict(row) if row else None

    def pending(self, include_uploads=False):
        with self.connect() as db:
            states = "('uploading','queued','running')" if include_uploads else "('queued','running')"
            return [dict(row) for row in db.execute(f"SELECT * FROM jobs WHERE state IN {states} ORDER BY created")]

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY created LIMIT 1").fetchone()
            if row:
                db.execute("UPDATE jobs SET state='running' WHERE id=?", (row["id"],))
                return dict(row)

    def finish(self, jid, result=None, error=None):
        with self.connect() as db:
            db.execute("UPDATE jobs SET state=?, result=?, error=? WHERE id=? AND state IN ('uploading','queued','running')",
                       ("failed" if error else "done", json.dumps(result, ensure_ascii=False) if result else None, error, jid))

    def path(self, jid):
        # IDs are generated internally; never use an uploaded filename as a path.
        return self.uploads / f"{jid}.mov"

    def delete_video(self, jid):
        self.path(jid).unlink(missing_ok=True)

    def recover(self):
        for row in self.pending(include_uploads=True):
            self.delete_video(row["id"])
            self.finish(row["id"], error="Service restarted before processing completed")
        # Includes interrupted uploads which never reached the queue.
        for path in self.uploads.iterdir():
            if path.is_file():
                path.unlink()
        with self.connect() as db:
            db.execute("DELETE FROM cleanup")


class InstanceLock:
    """One supervisor per data directory, including across Uvicorn processes."""
    def __init__(self, path):
        self.path = path
        self.handle = None

    def acquire(self):
        import os
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "a+b")
        try:
            self.handle.seek(0)
            if self.handle.read(1) == b"":
                self.handle.write(b"0")
                self.handle.flush()
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            self.handle = None
            raise RuntimeError("Data directory already in use. Run Uvicorn with ONE server worker.") from None

    def release(self):
        if self.handle:
            self.handle.close()
            self.handle = None
