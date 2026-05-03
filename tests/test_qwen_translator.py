from __future__ import annotations

import sys
import types

import torch

from aivoice.translation import GlossaryTerm, LocalQwenTranslator, TranslationRequest


class FakeInputs(dict):
    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):  # noqa: ANN001
        return self


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):  # noqa: ANN001
        assert messages
        assert tokenize is False
        assert add_generation_prompt is True
        return "prompt"

    def __call__(self, prompts, return_tensors):  # noqa: ANN001
        assert prompts == ["prompt"]
        assert return_tensors == "pt"
        return FakeInputs(input_ids=torch.tensor([[1, 2]]))

    def decode(self, token_ids, skip_special_tokens):  # noqa: ANN001
        assert skip_special_tokens is True
        return '{"candidates":[{"text":"编码器会映射输入嵌入。"}]}'


class FakeModel:
    device = torch.device("cpu")

    def to(self, device):  # noqa: ANN001
        assert device == "cpu"
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):  # noqa: ANN003
        assert kwargs["max_new_tokens"] == 64
        assert kwargs["do_sample"] is False
        return torch.tensor([[1, 2, 3, 4]])


def test_local_qwen_translator_uses_transformers_chat_template(monkeypatch) -> None:
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer()),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeModel()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    translator = LocalQwenTranslator(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
        max_new_tokens=64,
        candidate_count=1,
    )

    result = translator.translate_segments(
        [
            TranslationRequest(
                index=1,
                text="The encoder maps input embeddings.",
                duration=3.0,
            )
        ],
        source_lang="eng_Latn",
        target_lang="zho_Hans",
        glossary=[GlossaryTerm("encoder", "编码器")],
    )

    assert result == ["编码器会映射输入嵌入。"]
