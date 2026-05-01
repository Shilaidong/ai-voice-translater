from __future__ import annotations

from pathlib import Path

from .base import SpeechRegion, VadBackend


class SileroVadBackend(VadBackend):
    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
    ) -> None:
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self._model = None
        self._utils = None

    def detect(self, audio_path: Path) -> list[SpeechRegion]:
        model, utils = self._load()
        get_speech_timestamps, _, read_audio, _, _ = utils
        wav = read_audio(str(audio_path), sampling_rate=16000)
        timestamps = get_speech_timestamps(
            wav,
            model,
            sampling_rate=16000,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            return_seconds=True,
        )
        return [
            SpeechRegion(start=float(item["start"]), end=float(item["end"]))
            for item in timestamps
            if float(item["end"]) > float(item["start"])
        ]

    def _load(self):
        if self._model is None or self._utils is None:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("Silero VAD requires torch. Install the ml dependencies first.") from exc

            self._model, self._utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
        return self._model, self._utils
