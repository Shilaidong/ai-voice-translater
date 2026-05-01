from __future__ import annotations

from .alignment import AlignmentBackend, NoopAlignmentBackend
from .config import Settings
from .asr import AsrBackend, MockAsrBackend
from .tts import MockTtsBackend, TtsBackend
from .translation import MockTranslator, Translator
from .vad import NoopVadBackend, VadBackend


def create_asr_backend(settings: Settings) -> AsrBackend:
    name = settings.asr_backend
    if name == "mock":
        return MockAsrBackend()
    if name in {"faster-whisper", "faster_whisper"}:
        from .asr.faster_whisper_adapter import FasterWhisperBackend

        return FasterWhisperBackend(
            model_size=settings.asr_model_size,
            device=settings.asr_device,
            compute_type=settings.asr_compute_type,
        )
    if name == "whisperx":
        from .asr.whisperx_adapter import WhisperXBackend

        return WhisperXBackend()
    raise ValueError(f"Unsupported ASR backend: {name}")


def create_alignment_backend(settings: Settings) -> AlignmentBackend:
    name = settings.alignment_backend
    if name in {"off", "none", "noop", "disabled"}:
        return NoopAlignmentBackend()
    if name == "whisperx":
        from .alignment.whisperx import WhisperXAlignmentBackend

        return WhisperXAlignmentBackend(
            language_code=settings.alignment_language,
            device=settings.alignment_device,
        )
    raise ValueError(f"Unsupported alignment backend: {name}")


def create_translator(settings: Settings) -> Translator:
    name = settings.translator_backend
    if name == "mock":
        return MockTranslator()
    if name == "nllb":
        from .translation.nllb_adapter import NllbTranslator

        return NllbTranslator(
            model_name=settings.translator_model,
            device=settings.translator_device,
            batch_size=settings.translator_batch_size,
            max_new_tokens=settings.translator_max_new_tokens,
        )
    if name in {"llm", "openai", "openai-compatible", "openai_compatible"}:
        from .translation.llm_adapter import OpenAICompatibleTranslator

        return OpenAICompatibleTranslator(
            api_base=settings.translator_api_base,
            api_key=settings.translator_api_key,
            model=settings.translator_model,
            max_tokens=settings.translator_max_new_tokens,
            timeout_seconds=settings.translator_timeout_seconds,
            candidate_count=settings.translator_candidate_count,
        )
    raise ValueError(f"Unsupported translator backend: {name}")


def create_tts_backend(settings: Settings) -> TtsBackend | None:
    name = settings.tts_backend
    if name in {"off", "none", "disabled"}:
        return None
    if name == "mock":
        return MockTtsBackend()
    if name in {"sapi", "windows-sapi", "windows_sapi"}:
        from .tts.sapi import WindowsSapiTtsBackend

        return WindowsSapiTtsBackend(
            voice_hint=settings.tts_voice,
            rate=settings.tts_rate,
            volume=settings.tts_volume,
        )
    raise ValueError(f"Unsupported TTS backend: {name}")


def create_vad_backend(settings: Settings) -> VadBackend:
    name = settings.vad_backend
    if name in {"off", "none", "noop", "disabled"}:
        return NoopVadBackend()
    if name == "silero":
        from .vad.silero import SileroVadBackend

        return SileroVadBackend(
            threshold=settings.vad_threshold,
            min_speech_ms=settings.vad_min_speech_ms,
            min_silence_ms=settings.vad_min_silence_ms,
        )
    raise ValueError(f"Unsupported VAD backend: {name}")
