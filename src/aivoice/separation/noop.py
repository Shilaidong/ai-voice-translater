from __future__ import annotations

import shutil
from pathlib import Path

from .base import SeparatedAudio
from ..media import mix_timed_audio, probe_duration


class NoopAudioSeparationBackend:
    name = "off"

    def separate(self, audio_path: Path, work_dir: Path, ffmpeg_path: str) -> SeparatedAudio:
        work_dir.mkdir(parents=True, exist_ok=True)
        vocals_path = work_dir / "vocals.wav"
        background_path = work_dir / "background.wav"

        shutil.copyfile(audio_path, vocals_path)
        duration = max(probe_duration(audio_path, ffmpeg_path), 0.1)
        result = mix_timed_audio(ffmpeg_path, [], background_path, duration)
        if result.returncode != 0:
            raise RuntimeError(
                "ffmpeg silent background generation failed with exit code "
                f"{result.returncode}: {result.stderr[-1000:]}"
            )

        return SeparatedAudio(
            original_audio=audio_path,
            vocals_audio=vocals_path,
            background_audio=background_path,
        )
