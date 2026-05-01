from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpeechRegion:
    start: float
    end: float


class VadBackend(ABC):
    @abstractmethod
    def detect(self, audio_path: Path) -> list[SpeechRegion]:
        raise NotImplementedError
