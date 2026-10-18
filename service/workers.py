import logging
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass

from service.adapter import load_analyzer, normalize_result, video_duration

log = logging.getLogger(__name__)


def worker_main(conn, threads, factory):
    try:
        try:
            analyzer = factory() if factory else load_analyzer(threads)
            init_error = None
        except Exception as exc:
            init_error = f"Model initialization failed: {type(exc).__name__}"
            log.exception("Model initialization failed")
        conn.send(("ready", init_error))
        while True:
            jid, path = conn.recv()
            result, error = None, None
            try:
                if init_error:
                    raise RuntimeError(init_error)
                duration = video_duration(path)
                result = normalize_result(analyzer.run(str(path)), duration)
            except Exception as exc:
                # Do not leak local paths, credentials or model internals into public API errors.
                error = f"Analysis failed ({type(exc).__name__}): invalid video, model unavailable or invalid model output"
                log.warning("Job %s failed (%s)", jid, type(exc).__name__)
            conn.send(("result", jid, result, error))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        conn.close()


@dataclass
class Slot:
    process: object
    conn: object
    ready: bool = False
    init_error: str | None = None
    job: str | None = None
    started: float = 0


class Supervisor:
    def __init__(self, settings, store, factory=None):
        self.settings, self.store, self.factory = settings, store, factory
        self.context = mp.get_context("spawn")
        self.stop_event = threading.Event()
        self.slots = []
        self.thread = None
        self.failure = None

    def spawn(self):
        parent, child = self.context.Pipe()
        process = self.context.Process(target=worker_main, args=(child, self.settings.threads_per_worker, self.factory), daemon=True)
        process.start()
        child.close()
        return Slot(process, parent, started=time.monotonic())

    def start(self):
        self.slots = [self.spawn() for _ in range(self.settings.workers)]
        self.thread = threading.Thread(target=self.run, name="job-supervisor", daemon=True)
        self.thread.start()

    @staticmethod
    def terminate(slot):
        if slot.process.is_alive():
            slot.process.terminate()
        slot.process.join(timeout=3)
        if slot.process.is_alive():
            slot.process.kill()
            slot.process.join(timeout=3)
        slot.conn.close()

    def finish(self, jid, result=None, error=None):
        # Delete before exposing a terminal status; restart also cleans interrupted inputs.
        self.store.schedule_cleanup(jid)
        try:
            self.store.delete_video(jid)
            self.store.cleanup_done(jid)
        except OSError:
            log.exception("Unable to delete input %s", jid)
            error = "Input cleanup failed"
        self.store.finish(jid, result, error)

    def retry_cleanup(self):
        for jid in self.store.cleanup_pending():
            try:
                self.store.delete_video(jid)
                self.store.cleanup_done(jid)
            except OSError:
                pass  # Persisted obligation is retried on every tick and on restart.

    def tick(self):
        self.retry_cleanup()
        for index, slot in enumerate(self.slots):
            try:
                if slot.conn.poll():
                    message = slot.conn.recv()
                    if message[0] == "ready":
                        slot.ready, slot.init_error = True, message[1]
                    elif message[0] == "result":
                        _, jid, result, error = message
                        if slot.job == jid:
                            self.finish(jid, result, error)
                            slot.job = None
            except (EOFError, OSError):
                pass
            if not slot.process.is_alive() or (not slot.ready and time.monotonic() - slot.started > self.settings.job_timeout):
                self.terminate(slot)
                if slot.job:
                    self.finish(slot.job, error="Worker exited unexpectedly")
                self.slots[index] = self.spawn()

        now = time.time()
        for row in self.store.pending():
            if now - row["created"] >= self.settings.job_timeout:
                for index, slot in enumerate(self.slots):
                    if slot.job == row["id"]:
                        self.terminate(slot)
                        self.slots[index] = self.spawn()
                self.finish(row["id"], error="Processing time limit exceeded (including queue wait)")

        for slot in self.slots:
            if slot.ready and slot.job is None:
                row = self.store.claim()
                if row:
                    slot.job = row["id"]
                    try:
                        slot.conn.send((slot.job, self.store.path(slot.job)))
                    except (OSError, EOFError):
                        self.terminate(slot)

    def run(self):
        try:
            while not self.stop_event.is_set():
                self.tick()
                self.stop_event.wait(0.05)
        except Exception:
            self.failure = "Job supervisor unavailable"
            log.exception("Supervisor failed")
            for slot in self.slots:
                self.terminate(slot)
            # Fail closed; accepted jobs must not remain processing after the watchdog exits.
            try:
                for row in self.store.pending():
                    self.finish(row["id"], error=self.failure)
                self.retry_cleanup()
            except Exception:
                log.exception("Supervisor failure cleanup failed; recovery will retry on restart")

    def health(self):
        ready = sum(slot.ready and not slot.init_error and slot.process.is_alive() for slot in self.slots)
        cleanup = len(self.store.cleanup_pending())
        healthy = ready == self.settings.workers and not self.failure and not cleanup
        return {"status": "ok" if healthy else "starting" if not self.failure else "failed",
                "ready_workers": ready, "workers": self.settings.workers, "cleanup_pending": cleanup}

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=10)
        for slot in self.slots:
            self.terminate(slot)
        for row in self.store.pending():
            self.finish(row["id"], error="Service stopped before processing completed")
        self.retry_cleanup()
