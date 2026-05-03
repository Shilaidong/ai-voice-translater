from __future__ import annotations

import math
import wave
from pathlib import Path

from aivoice.media import probe_duration
from aivoice.separation import NoopAudioSeparationBackend


def create_wav(path: Path, seconds: float = 1.0) -> None:
    rate = 16000
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        frames = bytearray()
        for index in range(int(rate * seconds)):
            value = int(12000 * math.sin(2 * math.pi * 440 * index / rate))
            frames += value.to_bytes(2, "little", signed=True)
        wav.writeframes(frames)


def test_noop_audio_separation_creates_future_dubbing_lanes(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    create_wav(source, seconds=1.2)

    result = NoopAudioSeparationBackend().separate(
        audio_path=source,
        work_dir=tmp_path / "lanes",
        ffmpeg_path="ffmpeg",
    )

    assert result.original_audio == source
    assert result.vocals_audio.exists()
    assert result.background_audio.exists()
    assert abs(probe_duration(result.background_audio) - probe_duration(source)) < 0.05
