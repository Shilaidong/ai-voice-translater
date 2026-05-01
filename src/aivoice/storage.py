from __future__ import annotations

import json
import uuid
from pathlib import Path

from .models import JobRecord, utc_now


class JobStore:
    def __init__(self, jobs_dir: Path) -> None:
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def new_id(self) -> str:
        return uuid.uuid4().hex

    def job_dir(self, job_id: str) -> Path:
        path = self.jobs_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def metadata_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def save(self, job: JobRecord) -> None:
        job.updated_at = utc_now()
        path = self.metadata_path(job.id)
        temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        temp_path.write_text(
            json.dumps(job.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)

    def load(self, job_id: str) -> JobRecord:
        path = self.metadata_path(job_id)
        return JobRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def exists(self, job_id: str) -> bool:
        return self.metadata_path(job_id).exists()

    def list_recent(self, limit: int = 20) -> list[JobRecord]:
        records: list[JobRecord] = []
        for metadata_path in self.jobs_dir.glob("*/job.json"):
            try:
                records.append(JobRecord.from_dict(json.loads(metadata_path.read_text(encoding="utf-8"))))
            except Exception:
                continue
        records.sort(key=lambda job: job.created_at, reverse=True)
        return records[:limit]
