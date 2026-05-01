from __future__ import annotations

import math
import wave
from pathlib import Path

from .base import TtsBackend


class MockTtsBackend(TtsBackend):
    def synthesize(self, text: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = 16000
        duration = min(max(len(text) * 0.035, 0.35), 1.5)
        frame_count = int(sample_rate * duration)
        with wave.open(str(output_path), "w") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            frames = bytearray()
            for index in range(frame_count):
                value = int(3000 * math.sin(2 * math.pi * 520 * index / sample_rate))
                frames += value.to_bytes(2, "little", signed=True)
            wav.writeframes(frames)
