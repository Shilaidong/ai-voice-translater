from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

JobStatus = Literal["queued", "running", "succeeded", "failed"]


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    speaker_id: str | None = None
    words: list[dict[str, Any]] = field(default_factory=list)
    confidence: float | None = None


@dataclass(frozen=True)
class SubtitleCue:
    index: int
    start: float
    end: float
    source_text: str
    translated_text: str
    speaker_id: str | None = None
    source_words: list[dict[str, Any]] = field(default_factory=list)
    confidence: float | None = None
    duration_budget: float | None = None
    duration_tolerance: float = 0.08


@dataclass
class JobRecord:
    id: str
    video_path: str
    status: JobStatus = "queued"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    error: str | None = None
    outputs: dict[str, str] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)
    cues: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobRecord":
        known_fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in known_fields})

    @property
    def job_dir_name(self) -> str:
        return self.id


def stringify_outputs(outputs: dict[str, Path]) -> dict[str, str]:
    return {key: str(value) for key, value in outputs.items()}
