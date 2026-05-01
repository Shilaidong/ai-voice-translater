from __future__ import annotations

from pathlib import Path

from .base import AsrBackend
from aivoice.models import Segment


class WhisperXBackend(AsrBackend):
    def __init__(self) -> None:
        raise RuntimeError(
            "WhisperX backend is not wired yet. Install the ml extra and implement "
            "model loading here once the target hardware profile is fixed."
        )

    def transcribe(self, audio_path: Path) -> list[Segment]:
        raise NotImplementedError
