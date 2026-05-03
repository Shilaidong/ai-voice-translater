from __future__ import annotations

import json
from typing import Any

from .base import GlossaryTerm, TranslationRequest
from .llm_adapter import OpenAICompatibleTranslator


class LocalQwenTranslator(OpenAICompatibleTranslator):
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_new_tokens: int = 160,
        candidate_count: int = 3,
    ) -> None:
        super().__init__(
            api_base="local-qwen",
            api_key="",
            model=model_name,
            max_tokens=max_new_tokens,
            timeout_seconds=0,
            candidate_count=candidate_count,
        )
        self.device = device
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _complete(self, messages: list[dict[str, str]]) -> str:
        tokenizer, model = self._load_model()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([prompt], return_tensors="pt")
        model_device = getattr(model, "device", None)
        if model_device is not None:
            inputs = inputs.to(model_device)

        import torch

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_length = inputs.input_ids.shape[1]
        generated_ids = generated[0][prompt_length:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def translate_segments(
        self,
        requests: list[TranslationRequest],
        source_lang: str,
        target_lang: str,
        glossary: list[GlossaryTerm],
    ) -> list[str]:
        return super().translate_segments(requests, source_lang, target_lang, glossary)

    def _messages(
        self,
        item: TranslationRequest,
        source_lang: str,
        target_lang: str,
        glossary: list[GlossaryTerm],
    ) -> list[dict[str, str]]:
        min_chars = max(1, round(item.duration * 5))
        max_chars = max(min_chars, round(item.duration * 7))
        glossary_text = "\n".join(f"{term.source} => {term.target}" for term in glossary) or "none"
        user = (
            "Translate the source subtitle into natural Simplified Chinese.\n"
            "Do not translate these instructions. Do not explain.\n"
            f"Source language code: {source_lang}\n"
            f"Target language code: {target_lang}\n"
            f"Context before: {item.context_before or 'none'}\n"
            f"Source text: {item.text}\n"
            f"Context after: {item.context_after or 'none'}\n"
            f"Duration seconds: {item.duration:.2f}\n"
            f"Target Chinese length: {min_chars}-{max_chars} spoken characters.\n"
            f"Glossary:\n{glossary_text}\n"
            f"Return exactly this JSON shape with {self.candidate_count} candidate(s):\n"
            + json.dumps({"candidates": [{"text": "中文译文", "notes": ""}]}, ensure_ascii=False)
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a precise English-to-Simplified-Chinese course subtitle translator. "
                    "Your entire answer must be valid JSON."
                ),
            },
            {"role": "user", "content": user},
        ]

    def _load_model(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local Qwen translation requires transformers and torch. "
                "Install with: python -m pip install -e .[translate]"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=True,
        )
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": "auto",
        }
        if self.device == "auto":
            model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(self.model, **model_kwargs)
        if self.device != "auto":
            model = model.to(self.device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model
