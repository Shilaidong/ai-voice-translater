from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from aivoice.models import Segment


class AsrBackend(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path) -> list[Segment]:
        raise NotImplementedError
