from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

from aivoice.config import Settings
from aivoice.media import resolve_ffmpeg
from aivoice.pipeline import OfflinePipeline
from aivoice.storage import JobStore


def make_settings(data_dir: Path) -> Settings:
    return Settings(
        data_dir=data_dir,
        log_level="INFO",
        job_worker_count=1,
        asr_backend="mock",
        asr_model_size="tiny.en",
        asr_device="cpu",
        asr_compute_type="int8",
        alignment_backend="off",
        alignment_language="en",
        alignment_device="cpu",
        vad_backend="off",
        vad_threshold=0.5,
        vad_min_speech_ms=250,
        vad_min_silence_ms=100,
        audio_separation_backend="off",
        translator_backend="mock",
        translator_model="facebook/nllb-200-distilled-600M",
        translator_device="cpu",
        translator_batch_size=1,
        translator_max_new_tokens=80,
        translator_api_base="http://127.0.0.1:8000/v1",
        translator_api_key="",
        translator_timeout_seconds=10,
        translator_candidate_count=3,
        glossary_path=None,
        tts_backend="mock",
        tts_voice="Chinese",
        tts_rate=0,
        tts_volume=100,
        translation_replacements=(),
        subtitle_source_max_chars=32,
        subtitle_target_max_chars=24,
        subtitle_target_max_cps=7.0,
        source_lang="eng_Latn",
        target_lang="zho_Hans",
        ffmpeg_path="ffmpeg",
    )


def create_wav(path: Path) -> None:
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        frames = bytearray()
        for index in range(16000):
            value = int(12000 * math.sin(2 * math.pi * 440 * index / 16000))
            frames += value.to_bytes(2, "little", signed=True)
        wav.writeframes(frames)


def create_mp4(path: Path) -> None:
    ffmpeg = resolve_ffmpeg()
    command = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x90:rate=10:duration=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_pipeline_processes_wav_with_mock_backends(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.wav"
    create_wav(video_path)
    settings = make_settings(tmp_path / "data")
    store = JobStore(settings.jobs_dir)

    job = OfflinePipeline(settings, store).process(video_path)

    assert job.status == "succeeded"
    assert Path(job.outputs["audio"]).exists()
    assert Path(job.outputs["original_audio"]).exists()
    assert Path(job.outputs["vocals_audio"]).exists()
    assert Path(job.outputs["background_audio"]).exists()
    assert Path(job.outputs["dubbed_audio"]).exists()
    assert Path(job.outputs["source_srt"]).exists()
    assert Path(job.outputs["zh_srt"]).exists()
    assert Path(job.outputs["bilingual_vtt"]).exists()
    assert "[zho_Hans]" in Path(job.outputs["zh_srt"]).read_text(encoding="utf-8")
    assert job.config_snapshot["vad_backend"] == "off"
    assert job.config_snapshot["audio_separation_backend"] == "off"
    assert job.model_versions["alignment_backend"] == "off"
    assert job.model_versions["audio_separation_backend"] == "off"
    assert job.model_versions["asr_model_size"] == "tiny.en"
    assert job.cues[0]["speaker_id"] is None
    assert job.cues[0]["duration_tolerance"] == 0.08
    assert store.load(job.id).status == "succeeded"


def test_pipeline_processes_video_with_translated_video_output(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    create_mp4(video_path)
    settings = make_settings(tmp_path / "data")
    store = JobStore(settings.jobs_dir)

    job = OfflinePipeline(settings, store).process(video_path)

    assert job.status == "succeeded"
    assert Path(job.outputs["background_audio"]).exists()
    assert Path(job.outputs["source_srt"]).exists()
    assert Path(job.outputs["zh_srt"]).exists()
    assert Path(job.outputs["dubbed_audio"]).exists()
    translated_video = Path(job.outputs["translated_video"])
    assert translated_video.exists()
    assert translated_video.suffix == ".mkv"
    dubbed_video = Path(job.outputs["dubbed_video"])
    assert dubbed_video.exists()
    assert dubbed_video.suffix == ".mkv"
