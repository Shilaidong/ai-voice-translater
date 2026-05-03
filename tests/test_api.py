from __future__ import annotations

import math
import time
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from aivoice.api import create_app
from aivoice.config import Settings


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
        audio_separation_model="htdemucs_ft",
        audio_separation_device="cpu",
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


def wait_for_job(client: TestClient, job_id: str) -> dict[str, object]:
    for _ in range(40):
        response = client.get(f"/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()
        if job["status"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_gui_root_and_runtime(tmp_path: Path) -> None:
    client = TestClient(create_app(make_settings(tmp_path / "data")))

    response = client.get("/")
    assert response.status_code == 200
    assert "AI Voice Translater" in response.text

    runtime = client.get("/runtime").json()
    assert runtime["asr_backend"] == "mock"
    assert runtime["alignment_backend"] == "off"
    assert runtime["job_worker_count"] == 1
    assert runtime["vad_backend"] == "off"
    assert runtime["audio_separation_backend"] == "off"
    assert runtime["audio_separation_model"] == "htdemucs_ft"
    assert runtime["audio_separation_device"] == "cpu"
    assert runtime["tts_backend"] == "mock"
    assert runtime["target_lang"] == "zho_Hans"


def test_upload_job_and_download_output(tmp_path: Path) -> None:
    sample = tmp_path / "sample.wav"
    create_wav(sample)
    client = TestClient(create_app(make_settings(tmp_path / "data")))

    with sample.open("rb") as handle:
        response = client.post("/jobs/upload", files={"file": ("sample.wav", handle, "audio/wav")})
    assert response.status_code == 200
    job_id = response.json()["id"]

    job = wait_for_job(client, job_id)
    assert job["status"] == "succeeded"
    assert job["config_snapshot"]["asr_backend"] == "mock"
    assert job["model_versions"]["vad_backend"] == "off"
    assert job["model_versions"]["audio_separation_backend"] == "off"
    assert job["cues"][0]["duration_budget"] > 0

    output = client.get(f"/jobs/{job_id}/outputs/bilingual_vtt")
    assert output.status_code == 200
    assert "WEBVTT" in output.text

    unknown_output = client.get(f"/jobs/{job_id}/outputs/job_json")
    assert unknown_output.status_code == 404

    path_traversal = client.get(f"/jobs/{job_id}/outputs/..%2F..%2FREADME.md")
    assert path_traversal.status_code == 404

    dubbed_audio = client.get(f"/jobs/{job_id}/outputs/dubbed_audio")
    assert dubbed_audio.status_code == 200

    background_audio = client.get(f"/jobs/{job_id}/outputs/background_audio")
    assert background_audio.status_code == 200

    logs = client.get(f"/jobs/{job_id}/logs")
    assert logs.status_code == 200
    assert "job succeeded" in logs.text
