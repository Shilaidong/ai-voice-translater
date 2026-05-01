from __future__ import annotations

import logging
from pathlib import Path

from .base import AsrBackend
from aivoice.models import Segment

logger = logging.getLogger(__name__)


class FasterWhisperBackend(AsrBackend):
    def __init__(self, model_size: str, device: str = "cpu", compute_type: str = "int8") -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install with `pip install -e .[asr]` "
                "or switch AIVT_ASR_BACKEND=mock."
            ) from exc

        logger.info(
            "loading faster-whisper model",
            extra={
                "stage": "asr_model_load",
                "model_size": model_size,
                "device": device,
                "compute_type": compute_type,
            },
        )
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: Path) -> list[Segment]:
        segments, _info = self.model.transcribe(
            str(audio_path),
            vad_filter=True,
            beam_size=5,
        )
        output: list[Segment] = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                output.append(Segment(start=float(segment.start), end=float(segment.end), text=text))
        return output
