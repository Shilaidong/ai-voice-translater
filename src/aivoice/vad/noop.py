from __future__ import annotations

from pathlib import Path

from aivoice.media import probe_duration

from .base import SpeechRegion, VadBackend


class NoopVadBackend(VadBackend):
    def detect(self, audio_path: Path) -> list[SpeechRegion]:
        return [SpeechRegion(start=0.0, end=probe_duration(audio_path))]
