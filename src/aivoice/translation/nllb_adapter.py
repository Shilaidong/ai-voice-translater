from __future__ import annotations

import logging
from itertools import islice

from .base import Translator

logger = logging.getLogger(__name__)


def _chunks(items: list[str], size: int) -> list[list[str]]:
    iterator = iter(items)
    chunks: list[list[str]] = []
    while batch := list(islice(iterator, size)):
        chunks.append(batch)
    return chunks


class NllbTranslator(Translator):
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "cpu",
        batch_size: int = 4,
        max_new_tokens: int = 160,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "NLLB translation dependencies are not installed. Install with "
                "`pip install -e .[translate]` or switch AIVT_TRANSLATOR_BACKEND=mock."
            ) from exc

        if device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(f"Translator device '{device}' requested, but only CPU is available.")

        logger.info(
            "loading nllb translator",
            extra={"stage": "translator_model_load", "model_name": model_name, "device": device},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if device == "cuda" else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self._torch = torch

    def translate_batch(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        if not texts:
            return []

        self.tokenizer.src_lang = source_lang
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(target_lang)
        if forced_bos_token_id is None or forced_bos_token_id < 0:
            raise RuntimeError(f"Unsupported NLLB target language code: {target_lang}")

        translations: list[str] = []
        with self._torch.inference_mode():
            for batch in _chunks(texts, max(1, self.batch_size)):
                encoded = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                generated = self.model.generate(
                    **encoded,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=self.max_new_tokens,
                    max_length=None,
                )
                translations.extend(
                    self._normalize_translation(text)
                    for text in self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                )
        return translations

    def _normalize_translation(self, text: str) -> str:
        return (
            text.strip()
            .replace(" ,", "，")
            .replace(",", "，")
            .replace(" .", "。")
            .replace(".", "。")
            .replace(" ?", "？")
            .replace("?", "？")
            .replace(" !", "！")
            .replace("!", "！")
        )
