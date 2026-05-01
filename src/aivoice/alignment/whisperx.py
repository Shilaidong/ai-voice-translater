from __future__ import annotations

from pathlib import Path
from typing import Any

from aivoice.models import Segment

from .base import AlignmentBackend


class WhisperXAlignmentBackend(AlignmentBackend):
    def __init__(self, language_code: str = "en", device: str = "cpu") -> None:
        if language_code not in {"en", "eng", "eng_Latn"}:
            raise RuntimeError(
                "WhisperX alignment is currently only enabled for English source audio. "
                "Use AIVT_ALIGNMENT_BACKEND=off for Chinese or mixed-language sources."
            )
        try:
            import whisperx
        except ImportError as exc:
            raise RuntimeError(
                "WhisperX alignment requires whisperx. Install the ml dependencies first."
            ) from exc

        self._whisperx = whisperx
        self.language_code = "en"
        self.device = device
        self._model = None
        self._metadata = None

    def align(self, audio_path: Path, segments: list[Segment]) -> list[Segment]:
        if not segments:
            return []
        model, metadata = self._load_model()
        audio = self._whisperx.load_audio(str(audio_path))
        result = {
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
                for segment in segments
            ],
            "language": self.language_code,
        }
        aligned = self._whisperx.align(
            result["segments"],
            model,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        return _segments_from_whisperx(aligned, fallback=segments)

    def _load_model(self):
        if self._model is None or self._metadata is None:
            self._model, self._metadata = self._whisperx.load_align_model(
                language_code=self.language_code,
                device=self.device,
            )
        return self._model, self._metadata


def _segments_from_whisperx(aligned: dict[str, Any], fallback: list[Segment]) -> list[Segment]:
    output: list[Segment] = []
    aligned_segments = aligned.get("segments", [])
    for index, segment in enumerate(aligned_segments):
        fallback_segment = fallback[index] if index < len(fallback) else None
        text = str(segment.get("text") or (fallback_segment.text if fallback_segment else "")).strip()
        if not text:
            continue
        words = [
            {
                "word": str(word.get("word", "")).strip(),
                "start": float(word["start"]),
                "end": float(word["end"]),
                **({"score": float(word["score"])} if "score" in word else {}),
            }
            for word in segment.get("words", [])
            if "start" in word and "end" in word
        ]
        output.append(
            Segment(
                start=float(segment.get("start", fallback_segment.start if fallback_segment else 0.0)),
                end=float(segment.get("end", fallback_segment.end if fallback_segment else 0.0)),
                text=text,
                speaker_id=fallback_segment.speaker_id if fallback_segment else None,
                words=words,
                confidence=fallback_segment.confidence if fallback_segment else None,
            )
        )
    return output or fallback
