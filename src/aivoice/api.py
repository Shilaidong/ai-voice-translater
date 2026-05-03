from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings, load_settings
from .job_queue import JobQueue
from .logging_config import setup_logging
from .pipeline import OfflinePipeline
from .storage import JobStore

logger = logging.getLogger(__name__)
ALLOWED_OUTPUT_NAMES = {
    "audio",
    "background_audio",
    "bilingual_vtt",
    "dubbed_audio",
    "dubbed_video",
    "original_audio",
    "source_srt",
    "translated_video",
    "vocals_audio",
    "zh_srt",
}


class CreateJobRequest(BaseModel):
    video_path: str = Field(..., min_length=1)


def _web_dir() -> Path:
    return Path(__file__).resolve().parent / "web"


def _safe_upload_name(filename: str) -> str:
    name = Path(filename).name.strip().replace(" ", "_")
    return "".join(char for char in name if char.isalnum() or char in "._-") or "upload.bin"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    setup_logging(settings.logs_dir, settings.log_level)
    store = JobStore(settings.jobs_dir)
    pipeline = OfflinePipeline(settings, store)
    job_queue = JobQueue(pipeline, worker_count=settings.job_worker_count)

    app = FastAPI(title="AI Voice Translater", version="0.1.0")
    web_dir = _web_dir()
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/runtime")
    def runtime() -> dict[str, object]:
        return {
            "asr_backend": settings.asr_backend,
            "asr_model_size": settings.asr_model_size,
            "alignment_backend": settings.alignment_backend,
            "alignment_language": settings.alignment_language,
            "job_worker_count": settings.job_worker_count,
            "vad_backend": settings.vad_backend,
            "audio_separation_backend": settings.audio_separation_backend,
            "audio_separation_model": settings.audio_separation_model,
            "audio_separation_device": settings.audio_separation_device,
            "translator_backend": settings.translator_backend,
            "translator_model": settings.translator_model,
            "translator_api_base": settings.translator_api_base if settings.translator_backend == "llm" else "",
            "translator_candidate_count": settings.translator_candidate_count,
            "glossary_path": str(settings.glossary_path) if settings.glossary_path else "",
            "tts_backend": settings.tts_backend,
            "tts_voice": settings.tts_voice,
            "source_lang": settings.source_lang,
            "target_lang": settings.target_lang,
            "subtitle_source_max_chars": settings.subtitle_source_max_chars,
            "subtitle_target_max_chars": settings.subtitle_target_max_chars,
            "subtitle_target_max_cps": settings.subtitle_target_max_cps,
        }

    @app.get("/jobs")
    def list_jobs() -> list[dict[str, object]]:
        return [job.to_dict() for job in store.list_recent()]

    @app.post("/jobs")
    def create_job(request: CreateJobRequest) -> dict[str, object]:
        return _submit_job(Path(request.video_path), pipeline, store)

    @app.post("/jobs/upload")
    async def upload_job(file: Annotated[UploadFile, File(...)]) -> dict[str, object]:
        uploads_dir = settings.data_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        upload_path = uploads_dir / f"{store.new_id()}_{_safe_upload_name(file.filename or 'upload.bin')}"
        with upload_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                output.write(chunk)
        return _submit_job(upload_path, pipeline, store)

    def _submit_job(video_path: Path, pipeline: OfflinePipeline, store: JobStore) -> dict[str, object]:
        job = pipeline.create_job(video_path)
        job_queue.enqueue(job.id, video_path)
        return job.to_dict()

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, object]:
        if not store.exists(job_id):
            raise HTTPException(status_code=404, detail="Job not found")
        return store.load(job_id).to_dict()

    @app.get("/jobs/{job_id}/outputs/{name}")
    def get_output(job_id: str, name: str) -> FileResponse:
        if not store.exists(job_id):
            raise HTTPException(status_code=404, detail="Job not found")
        if name not in ALLOWED_OUTPUT_NAMES:
            raise HTTPException(status_code=404, detail="Output not found")
        job = store.load(job_id)
        output_path = job.outputs.get(name)
        if not output_path:
            raise HTTPException(status_code=404, detail="Output not found")
        job_dir = store.job_dir(job.id).resolve()
        path = Path(output_path).resolve()
        if not path.is_relative_to(job_dir):
            raise HTTPException(status_code=404, detail="Output not found")
        if not path.exists():
            raise HTTPException(status_code=404, detail="Output file missing")
        return FileResponse(path, filename=path.name)

    @app.get("/jobs/{job_id}/logs", response_class=PlainTextResponse)
    def get_logs(job_id: str) -> str:
        log_path = settings.logs_dir / "jobs" / f"{job_id}.log"
        if not log_path.exists():
            raise HTTPException(status_code=404, detail="Job log not found")
        return log_path.read_text(encoding="utf-8")

    return app


app = create_app()
