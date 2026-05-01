from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class GlossaryTerm:
    source: str
    target: str


@dataclass(frozen=True)
class TranslationRequest:
    index: int
    text: str
    duration: float
    context_before: str = ""
    context_after: str = ""


class Translator(ABC):
    @abstractmethod
    def translate_batch(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        raise NotImplementedError

    def translate_segments(
        self,
        requests: list[TranslationRequest],
        source_lang: str,
        target_lang: str,
        glossary: list[GlossaryTerm],
    ) -> list[str]:
        return self.translate_batch([request.text for request in requests], source_lang, target_lang)
