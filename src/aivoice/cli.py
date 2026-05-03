from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

from .backends import create_translator, create_vad_backend
from .config import load_settings
from .logging_config import setup_logging
from .media import check_ffmpeg
from .pipeline import OfflinePipeline
from .storage import JobStore
from .translation import TranslationRequest
from .translation.glossary import load_glossary


def process_video(video_path: str) -> int:
    settings = load_settings()
    setup_logging(settings.logs_dir, settings.log_level)
    store = JobStore(settings.jobs_dir)
    pipeline = OfflinePipeline(settings, store)
    job = pipeline.process(Path(video_path))
    print(json.dumps(job.to_dict(), ensure_ascii=False, indent=2))
    return 0


def serve(host: str, port: int) -> int:
    import uvicorn
    from .api import create_app

    app = create_app(load_settings())
    uvicorn.run(app, host=host, port=port)
    return 0


def doctor() -> int:
    settings = load_settings()
    ffmpeg = check_ffmpeg(settings.ffmpeg_path)
    report = {
        "python": sys.executable,
        "data_dir": str(settings.data_dir),
        "logs_dir": str(settings.logs_dir),
        "job_worker_count": settings.job_worker_count,
        "asr_backend": settings.asr_backend,
        "asr_model_size": settings.asr_model_size,
        "asr_device": settings.asr_device,
        "asr_compute_type": settings.asr_compute_type,
        "alignment_backend": settings.alignment_backend,
        "alignment_language": settings.alignment_language,
        "alignment_device": settings.alignment_device,
        "vad_backend": settings.vad_backend,
        "vad_threshold": settings.vad_threshold,
        "vad_min_speech_ms": settings.vad_min_speech_ms,
        "vad_min_silence_ms": settings.vad_min_silence_ms,
        "audio_separation_backend": settings.audio_separation_backend,
        "audio_separation_model": settings.audio_separation_model,
        "audio_separation_device": settings.audio_separation_device,
        "translator_backend": settings.translator_backend,
        "translator_model": settings.translator_model,
        "translator_device": settings.translator_device,
        "translator_batch_size": settings.translator_batch_size,
        "translator_api_base": settings.translator_api_base if settings.translator_backend == "llm" else "",
        "translator_candidate_count": settings.translator_candidate_count,
        "glossary_path": str(settings.glossary_path) if settings.glossary_path else "",
        "tts_backend": settings.tts_backend,
        "tts_voice": settings.tts_voice,
        "tts_rate": settings.tts_rate,
        "tts_volume": settings.tts_volume,
        "translation_replacements": list(settings.translation_replacements),
        "subtitle_source_max_chars": settings.subtitle_source_max_chars,
        "subtitle_target_max_chars": settings.subtitle_target_max_chars,
        "subtitle_target_max_cps": settings.subtitle_target_max_cps,
        "source_lang": settings.source_lang,
        "target_lang": settings.target_lang,
        "packages": {
            "fastapi": importlib.util.find_spec("fastapi") is not None,
            "faster_whisper": importlib.util.find_spec("faster_whisper") is not None,
            "imageio_ffmpeg": importlib.util.find_spec("imageio_ffmpeg") is not None,
            "sentencepiece": importlib.util.find_spec("sentencepiece") is not None,
            "torch": importlib.util.find_spec("torch") is not None,
            "transformers": importlib.util.find_spec("transformers") is not None,
            "whisperx": importlib.util.find_spec("whisperx") is not None,
            "demucs": importlib.util.find_spec("demucs") is not None,
        },
        "package_versions": {
            "demucs": _package_version("demucs"),
        },
        "ffmpeg": {
            "ok": ffmpeg.ok,
            "path": ffmpeg.path,
            "detail": ffmpeg.detail,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ffmpeg.ok else 1


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def smoke_llm(text: str, duration: float) -> int:
    settings = load_settings()
    if settings.translator_backend != "llm":
        print(
            "Set AIVT_TRANSLATOR_BACKEND=llm before running smoke-llm.",
            file=sys.stderr,
        )
        return 2
    translator = create_translator(settings)
    glossary = load_glossary(settings.glossary_path, settings.translation_replacements)
    result = translator.translate_segments(
        [
            TranslationRequest(
                index=1,
                text=text,
                duration=duration,
            )
        ],
        source_lang=settings.source_lang,
        target_lang=settings.target_lang,
        glossary=glossary,
    )
    print(json.dumps({"translation": result[0]}, ensure_ascii=False, indent=2))
    return 0


def smoke_vad(audio_path: str) -> int:
    settings = load_settings()
    vad = create_vad_backend(settings)
    regions = vad.detect(Path(audio_path))
    print(
        json.dumps(
            {
                "vad_backend": settings.vad_backend,
                "region_count": len(regions),
                "regions": [region.__dict__ for region in regions],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if regions else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="aivt")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process", help="Process one local video file.")
    process_parser.add_argument("video_path")

    serve_parser = subparsers.add_parser("serve", help="Start the local API service.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)

    subparsers.add_parser("doctor", help="Check local runtime dependencies.")

    smoke_llm_parser = subparsers.add_parser("smoke-llm", help="Smoke test the configured LLM translator.")
    smoke_llm_parser.add_argument("--text", default="The encoder maps input embeddings.")
    smoke_llm_parser.add_argument("--duration", type=float, default=3.0)

    smoke_vad_parser = subparsers.add_parser("smoke-vad", help="Smoke test the configured VAD backend.")
    smoke_vad_parser.add_argument("audio_path")

    args = parser.parse_args()
    if args.command == "process":
        return process_video(args.video_path)
    if args.command == "serve":
        return serve(args.host, args.port)
    if args.command == "doctor":
        return doctor()
    if args.command == "smoke-llm":
        return smoke_llm(args.text, args.duration)
    if args.command == "smoke-vad":
        return smoke_vad(args.audio_path)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
