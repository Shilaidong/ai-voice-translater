from __future__ import annotations

from pathlib import Path

from .base import AsrBackend
from aivoice.models import Segment


class MockAsrBackend(AsrBackend):
    def transcribe(self, audio_path: Path) -> list[Segment]:
        name = audio_path.stem.replace("_", " ")
        return [
            Segment(
                start=0.0,
                end=4.0,
                text=f"This is a placeholder transcription for {name}.",
            ),
            Segment(
                start=4.0,
                end=8.0,
                text="Replace the mock backend with WhisperX or faster-whisper for real output.",
            ),
        ]
