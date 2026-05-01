from __future__ import annotations

from .base import Translator


class MockTranslator(Translator):
    def translate_batch(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        return [f"[{target_lang}] {text}" for text in texts]
