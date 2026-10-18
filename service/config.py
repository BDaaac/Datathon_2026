import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    api_key: str
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data")
    workers: int = 5
    threads_per_worker: int = 1
    job_timeout: float = 285
    max_upload_bytes: int = 100 * 1024 * 1024
    max_pending_jobs: int = 25
    upload_timeout: float = 240

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("SERVICE_API_KEY is required")
        for name in ("workers", "threads_per_worker", "job_timeout", "max_upload_bytes", "max_pending_jobs", "upload_timeout"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def from_env(cls):
        workers = int(os.getenv("SERVICE_WORKERS", "5"))
        return cls(
            api_key=os.environ.get("SERVICE_API_KEY", ""),
            data_dir=Path(os.getenv("SERVICE_DATA_DIR", str(Path(__file__).parent / "data"))).resolve(),
            workers=workers,
            threads_per_worker=int(os.getenv("SERVICE_THREADS_PER_WORKER", str(max(1, (os.cpu_count() or 2) // max(workers, 1))))),
            job_timeout=float(os.getenv("SERVICE_JOB_TIMEOUT", "285")),
            max_upload_bytes=int(os.getenv("SERVICE_MAX_UPLOAD_MB", "100")) * 1024 * 1024,
            max_pending_jobs=int(os.getenv("SERVICE_MAX_PENDING_JOBS", "25")),
            upload_timeout=float(os.getenv("SERVICE_UPLOAD_TIMEOUT", "240")),   # медленный канал жюри: не обрывать загрузку 30 МБ
        )
