from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (
            "job_id",
            "stage",
            "duration_ms",
            "video_path",
            "output_path",
            "model_name",
            "model_size",
            "device",
            "compute_type",
            "backend",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class MergingLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg: object, kwargs: dict[str, Any]) -> tuple[object, dict[str, Any]]:
        extra = dict(self.extra)
        extra.update(kwargs.get("extra", {}))
        kwargs["extra"] = extra
        return msg, kwargs


def setup_logging(logs_dir: Path, level: str = "INFO") -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    formatter = JsonFormatter()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        logs_dir / "app.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    for noisy_logger in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "huggingface_hub.utils._http",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_job_logger(job_id: str, logs_dir: Path) -> MergingLoggerAdapter:
    logger = logging.getLogger(f"aivoice.job.{job_id}")
    logger.setLevel(logging.INFO)
    log_path = logs_dir / "jobs" / f"{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not any(
        isinstance(handler, RotatingFileHandler)
        and Path(handler.baseFilename) == log_path
        for handler in logger.handlers
    ):
        handler = RotatingFileHandler(
            log_path,
            maxBytes=2 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    return MergingLoggerAdapter(logger, {"job_id": job_id})
