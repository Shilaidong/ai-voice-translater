from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path

from .pipeline import OfflinePipeline

logger = logging.getLogger(__name__)


class JobQueue:
    def __init__(self, pipeline: OfflinePipeline, worker_count: int = 1) -> None:
        self.pipeline = pipeline
        self.worker_count = max(worker_count, 1)
        self._queue: queue.Queue[tuple[str, Path]] = queue.Queue()
        self._started = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            for index in range(self.worker_count):
                thread = threading.Thread(
                    target=self._worker,
                    name=f"aivoice-job-worker-{index + 1}",
                    daemon=True,
                )
                thread.start()
            self._started = True

    def enqueue(self, job_id: str, video_path: Path) -> None:
        self.start()
        self._queue.put((job_id, video_path))

    def _worker(self) -> None:
        while True:
            job_id, video_path = self._queue.get()
            try:
                self.pipeline.process(video_path, job_id=job_id)
            except Exception:
                logger.exception("background job failed", extra={"job_id": job_id, "stage": "job"})
            finally:
                self._queue.task_done()
