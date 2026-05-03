from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .base import SeparatedAudio
from ..media import normalize_audio


@dataclass(frozen=True)
class DemucsAudioSeparationBackend:
    model_name: str = "htdemucs_ft"
    device: str = "cpu"
    name: str = "demucs"

    def separate(self, audio_path: Path, work_dir: Path, ffmpeg_path: str) -> SeparatedAudio:
        if importlib.util.find_spec("demucs") is None:
            raise RuntimeError(
                "Demucs audio separation is configured but the demucs package is not installed. "
                "Install it separately or set AIVT_AUDIO_SEPARATION_BACKEND=off."
            )

        work_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = work_dir / "demucs_raw"
        command = [
            sys.executable,
            "-m",
            "demucs",
            "-n",
            self.model_name,
            "--two-stems",
            "vocals",
            "-d",
            self.device,
            "-o",
            str(raw_dir),
            str(audio_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"demucs failed with exit code {result.returncode}: {result.stderr[-1000:]}")

        demucs_output_dir = raw_dir / self.model_name / audio_path.stem
        vocals_source = demucs_output_dir / "vocals.wav"
        background_source = demucs_output_dir / "no_vocals.wav"
        if not vocals_source.exists() or not background_source.exists():
            raise RuntimeError(
                "demucs completed but expected vocals.wav/no_vocals.wav were not found under "
                f"{demucs_output_dir}"
            )

        vocals_path = work_dir / "vocals.wav"
        background_path = work_dir / "background.wav"
        self._normalize(ffmpeg_path, vocals_source, vocals_path)
        self._normalize(ffmpeg_path, background_source, background_path)

        return SeparatedAudio(
            original_audio=audio_path,
            vocals_audio=vocals_path,
            background_audio=background_path,
        )

    @staticmethod
    def _normalize(ffmpeg_path: str, input_path: Path, output_path: Path) -> None:
        result = normalize_audio(ffmpeg_path, input_path, output_path)
        if result.returncode != 0:
            raise RuntimeError(
                "ffmpeg audio normalization failed with exit code "
                f"{result.returncode}: {result.stderr[-1000:]}"
            )
