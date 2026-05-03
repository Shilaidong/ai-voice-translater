from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TtsSynthesisRequest:
    text: str
    output_path: Path
    target_duration: float | None = None
    reference_audio: Path | None = None
    reference_text: str | None = None
    style_hint: str | None = None
    speaker_id: str | None = None


class TtsBackend(ABC):
    supports_reference_audio = False
    supports_target_duration = False

    @abstractmethod
    def synthesize(self, text: str, output_path: Path) -> None:
        raise NotImplementedError

    def synthesize_request(self, request: TtsSynthesisRequest) -> None:
        self.synthesize(request.text, request.output_path)
