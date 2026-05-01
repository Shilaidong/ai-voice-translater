from __future__ import annotations

import os
from pathlib import Path

import pytest

from aivoice.alignment.whisperx import WhisperXAlignmentBackend
from aivoice.models import Segment
from aivoice.translation import GlossaryTerm, TranslationRequest
from aivoice.translation.llm_adapter import OpenAICompatibleTranslator
from aivoice.vad.silero import SileroVadBackend


def test_optional_openai_compatible_llm_smoke() -> None:
    if os.getenv("AIVT_SMOKE_LLM") != "1":
        pytest.skip("Set AIVT_SMOKE_LLM=1 to run the real LLM endpoint smoke test.")

    translator = OpenAICompatibleTranslator(
        api_base=os.getenv("AIVT_TRANSLATOR_API_BASE", "http://127.0.0.1:8000/v1"),
        api_key=os.getenv("AIVT_TRANSLATOR_API_KEY", ""),
        model=os.getenv("AIVT_TRANSLATOR_MODEL", "qwen2.5-7b-instruct"),
        max_tokens=int(os.getenv("AIVT_TRANSLATOR_MAX_NEW_TOKENS", "160")),
        timeout_seconds=int(os.getenv("AIVT_TRANSLATOR_TIMEOUT_SECONDS", "120")),
        candidate_count=3,
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
        glossary=[GlossaryTerm(source="encoder", target="编码器")],
    )

    assert result
    assert "编码器" in result[0]


def test_optional_silero_vad_smoke() -> None:
    audio_path = os.getenv("AIVT_SMOKE_VAD_AUDIO")
    if not audio_path:
        pytest.skip("Set AIVT_SMOKE_VAD_AUDIO to a real speech WAV to run Silero VAD smoke test.")

    path = Path(audio_path)
    assert path.exists()
    regions = SileroVadBackend().detect(path)

    assert regions
    assert all(region.end > region.start for region in regions)


def test_optional_whisperx_alignment_smoke() -> None:
    audio_path = os.getenv("AIVT_SMOKE_ALIGN_AUDIO")
    if not audio_path:
        pytest.skip("Set AIVT_SMOKE_ALIGN_AUDIO to a real English speech WAV to run WhisperX alignment smoke test.")

    path = Path(audio_path)
    assert path.exists()
    aligned = WhisperXAlignmentBackend(language_code="en").align(
        path,
        [
            Segment(
                start=0.0,
                end=float(os.getenv("AIVT_SMOKE_ALIGN_DURATION", "5")),
                text=os.getenv("AIVT_SMOKE_ALIGN_TEXT", "The encoder maps input embeddings."),
            )
        ],
    )

    assert aligned
    assert aligned[0].words
