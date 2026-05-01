from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from aivoice.models import Segment


class AlignmentBackend(ABC):
    @abstractmethod
    def align(self, audio_path: Path, segments: list[Segment]) -> list[Segment]:
        raise NotImplementedError
