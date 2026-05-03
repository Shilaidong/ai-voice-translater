from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    log_level: str
    job_worker_count: int
    asr_backend: str
    asr_model_size: str
    asr_device: str
    asr_compute_type: str
    alignment_backend: str
    alignment_language: str
    alignment_device: str
    vad_backend: str
    vad_threshold: float
    vad_min_speech_ms: int
    vad_min_silence_ms: int
    audio_separation_backend: str
    audio_separation_model: str
    audio_separation_device: str
    translator_backend: str
    translator_model: str
    translator_device: str
    translator_batch_size: int
    translator_max_new_tokens: int
    translator_api_base: str
    translator_api_key: str
    translator_timeout_seconds: int
    translator_candidate_count: int
    glossary_path: Path | None
    tts_backend: str
    tts_voice: str
    tts_rate: int
    tts_volume: int
    translation_replacements: tuple[tuple[str, str], ...]
    subtitle_source_max_chars: int
    subtitle_target_max_chars: int
    subtitle_target_max_cps: float
    source_lang: str
    target_lang: str
    ffmpeg_path: str

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    def snapshot(self) -> dict[str, object]:
        data = asdict(self)
        data["data_dir"] = str(self.data_dir)
        data["logs_dir"] = str(self.logs_dir)
        data["jobs_dir"] = str(self.jobs_dir)
        data["glossary_path"] = str(self.glossary_path) if self.glossary_path else None
        data["translation_replacements"] = [list(item) for item in self.translation_replacements]
        return data


def load_settings() -> Settings:
    root = _project_root()
    data_dir = Path(os.getenv("AIVT_DATA_DIR", root / "data")).expanduser()
    translator_backend = os.getenv("AIVT_TRANSLATOR_BACKEND", "mock").lower()
    default_translator_model = (
        "Qwen/Qwen2.5-1.5B-Instruct"
        if translator_backend == "qwen"
        else "facebook/nllb-200-distilled-600M"
    )
    return Settings(
        data_dir=data_dir,
        log_level=os.getenv("AIVT_LOG_LEVEL", "INFO").upper(),
        job_worker_count=int(os.getenv("AIVT_JOB_WORKER_COUNT", "1")),
        asr_backend=os.getenv("AIVT_ASR_BACKEND", "mock").lower(),
        asr_model_size=os.getenv("AIVT_ASR_MODEL_SIZE", "small.en"),
        asr_device=os.getenv("AIVT_ASR_DEVICE", "cpu").lower(),
        asr_compute_type=os.getenv("AIVT_ASR_COMPUTE_TYPE", "int8").lower(),
        alignment_backend=os.getenv("AIVT_ALIGNMENT_BACKEND", "off").lower(),
        alignment_language=os.getenv("AIVT_ALIGNMENT_LANGUAGE", "en"),
        alignment_device=os.getenv("AIVT_ALIGNMENT_DEVICE", "cpu").lower(),
        vad_backend=os.getenv("AIVT_VAD_BACKEND", "off").lower(),
        vad_threshold=float(os.getenv("AIVT_VAD_THRESHOLD", "0.5")),
        vad_min_speech_ms=int(os.getenv("AIVT_VAD_MIN_SPEECH_MS", "250")),
        vad_min_silence_ms=int(os.getenv("AIVT_VAD_MIN_SILENCE_MS", "100")),
        audio_separation_backend=os.getenv("AIVT_AUDIO_SEPARATION_BACKEND", "off").lower(),
        audio_separation_model=os.getenv("AIVT_AUDIO_SEPARATION_MODEL", "htdemucs_ft"),
        audio_separation_device=os.getenv("AIVT_AUDIO_SEPARATION_DEVICE", "cpu").lower(),
        translator_backend=translator_backend,
        translator_model=os.getenv("AIVT_TRANSLATOR_MODEL", default_translator_model),
        translator_device=os.getenv("AIVT_TRANSLATOR_DEVICE", "cpu").lower(),
        translator_batch_size=int(os.getenv("AIVT_TRANSLATOR_BATCH_SIZE", "4")),
        translator_max_new_tokens=int(os.getenv("AIVT_TRANSLATOR_MAX_NEW_TOKENS", "160")),
        translator_api_base=os.getenv("AIVT_TRANSLATOR_API_BASE", "http://127.0.0.1:8000/v1"),
        translator_api_key=os.getenv("AIVT_TRANSLATOR_API_KEY", ""),
        translator_timeout_seconds=int(os.getenv("AIVT_TRANSLATOR_TIMEOUT_SECONDS", "120")),
        translator_candidate_count=int(os.getenv("AIVT_TRANSLATOR_CANDIDATE_COUNT", "3")),
        glossary_path=_optional_path(os.getenv("AIVT_GLOSSARY_PATH", "")),
        tts_backend=os.getenv("AIVT_TTS_BACKEND", "sapi").lower(),
        tts_voice=os.getenv("AIVT_TTS_VOICE", "Chinese"),
        tts_rate=int(os.getenv("AIVT_TTS_RATE", "0")),
        tts_volume=int(os.getenv("AIVT_TTS_VOLUME", "100")),
        translation_replacements=parse_replacements(
            os.getenv(
                "AIVT_TRANSLATION_REPLACEMENTS",
                "\u5f53\u5730\u8bed\u97f3\u8bc6\u522b=\u672c\u5730\u8bed\u97f3\u8bc6\u522b;"
                "\u5730\u65b9\u7ffb\u8bd1=\u672c\u5730\u7ffb\u8bd1",
            )
        ),
        subtitle_source_max_chars=int(os.getenv("AIVT_SUBTITLE_SOURCE_MAX_CHARS", "48")),
        subtitle_target_max_chars=int(os.getenv("AIVT_SUBTITLE_TARGET_MAX_CHARS", "22")),
        subtitle_target_max_cps=float(os.getenv("AIVT_SUBTITLE_TARGET_MAX_CPS", "7.0")),
        source_lang=os.getenv("AIVT_SOURCE_LANG", "eng_Latn"),
        target_lang=os.getenv("AIVT_TARGET_LANG", "zho_Hans"),
        ffmpeg_path=os.getenv("AIVT_FFMPEG_PATH", "ffmpeg"),
    )


def parse_replacements(raw: str) -> tuple[tuple[str, str], ...]:
    replacements: list[tuple[str, str]] = []
    for item in raw.split(";"):
        if not item.strip() or "=" not in item:
            continue
        source, target = item.split("=", 1)
        source = source.strip()
        target = target.strip()
        if source:
            replacements.append((source, target))
    return tuple(replacements)


def _optional_path(raw: str) -> Path | None:
    value = raw.strip()
    return Path(value).expanduser() if value else None
