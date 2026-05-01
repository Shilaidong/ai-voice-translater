from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class TtsBackend(ABC):
    @abstractmethod
    def synthesize(self, text: str, output_path: Path) -> None:
        raise NotImplementedError
