from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SeparatedAudio:
    original_audio: Path
    vocals_audio: Path
    background_audio: Path


class AudioSeparationBackend(Protocol):
    name: str

    def separate(self, audio_path: Path, work_dir: Path, ffmpeg_path: str) -> SeparatedAudio:
        ...
