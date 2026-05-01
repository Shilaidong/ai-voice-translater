from __future__ import annotations

from pathlib import Path

from aivoice.models import Segment

from .base import AlignmentBackend


class NoopAlignmentBackend(AlignmentBackend):
    def align(self, audio_path: Path, segments: list[Segment]) -> list[Segment]:
        return segments
