from __future__ import annotations

import logging
import shutil
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from .alignment import AlignmentBackend
from .backends import (
    create_alignment_backend,
    create_asr_backend,
    create_audio_separation_backend,
    create_translator,
    create_tts_backend,
    create_vad_backend,
)
from .config import Settings
from .dubbing import synthesize_dubbed_audio
from .logging_config import get_job_logger
from .media import extract_audio, is_video_file, mux_dubbed_video, mux_subtitle_track, slice_audio
from .models import JobRecord, Segment, SubtitleCue, stringify_outputs
from .quality import check_reading_speed
from .storage import JobStore
from .subtitles import write_bilingual_vtt, write_srt
from .asr import AsrBackend
from .tts import TtsBackend
from .translation import GlossaryTerm, TranslationRequest, Translator
from .translation.glossary import load_glossary
from .translation.postprocess import apply_replacements
from .vad import SpeechRegion, VadBackend
from .separation import AudioSeparationBackend, SeparatedAudio

logger = logging.getLogger(__name__)
DIRECT_AUDIO_SUFFIXES = {".wav"}


class OfflinePipeline:
    def __init__(self, settings: Settings, store: JobStore) -> None:
        self.settings = settings
        self.store = store
        self._alignment: AlignmentBackend | None = None
        self._asr: AsrBackend | None = None
        self._translator: Translator | None = None
        self._glossary: list[GlossaryTerm] | None = None
        self._tts: TtsBackend | None | object = _UNSET
        self._vad: VadBackend | None = None
        self._audio_separation: AudioSeparationBackend | None = None

    def create_job(self, video_path: Path) -> JobRecord:
        job = JobRecord(
            id=uuid.uuid4().hex,
            video_path=str(video_path),
            config_snapshot=self.settings.snapshot(),
            model_versions=self._model_versions(),
        )
        self.store.save(job)
        return job

    def process(self, video_path: Path, job_id: str | None = None) -> JobRecord:
        resolved_video = video_path.expanduser().resolve()
        if not resolved_video.exists():
            raise FileNotFoundError(f"Video file not found: {resolved_video}")

        job = JobRecord(
            id=job_id or uuid.uuid4().hex,
            video_path=str(resolved_video),
            config_snapshot=self.settings.snapshot(),
            model_versions=self._model_versions(),
        )
        self.store.save(job)
        job_logger = get_job_logger(job.id, self.settings.logs_dir)

        job.status = "running"
        self.store.save(job)
        started = time.perf_counter()
        job_logger.info(
            "job started",
            extra={"stage": "job", "video_path": str(resolved_video)},
        )

        try:
            job_dir = self.store.job_dir(job.id)
            audio_path = job_dir / "audio.wav"

            self._extract_audio(resolved_video, audio_path, job_logger)
            separated_audio = self._separate_audio(audio_path, job_dir, job_logger)

            segments = self._transcribe_with_vad(audio_path, job_dir, job_logger)
            segments = self._align_segments(audio_path, segments, job_logger)

            stage_started = time.perf_counter()
            translation_requests = self._translation_requests(segments)
            translations = self.translator.translate_segments(
                translation_requests,
                source_lang=self.settings.source_lang,
                target_lang=self.settings.target_lang,
                glossary=self.glossary,
            )
            translations = [
                apply_replacements(text, self.settings.translation_replacements)
                for text in translations
            ]
            job_logger.info(
                "translation completed",
                extra={
                    "stage": "translation",
                    "duration_ms": int((time.perf_counter() - stage_started) * 1000),
                },
            )

            cues = [
                SubtitleCue(
                    index=index + 1,
                    start=segment.start,
                    end=segment.end,
                    source_text=segment.text,
                    translated_text=translations[index],
                    speaker_id=segment.speaker_id,
                    source_words=segment.words,
                    confidence=segment.confidence,
                    duration_budget=segment.end - segment.start,
                    duration_tolerance=0.08,
                )
                for index, segment in enumerate(segments)
            ]
            job.cues = [asdict(cue) for cue in cues]
            self._log_quality_warnings(cues, job_logger)

            outputs = {
                "audio": audio_path,
                "original_audio": separated_audio.original_audio,
                "vocals_audio": separated_audio.vocals_audio,
                "background_audio": separated_audio.background_audio,
                "source_srt": job_dir / "source.srt",
                "zh_srt": job_dir / "zh.srt",
                "bilingual_vtt": job_dir / "bilingual.vtt",
            }
            write_srt(
                outputs["source_srt"],
                cues,
                field="source",
                max_chars=self.settings.subtitle_source_max_chars,
            )
            write_srt(
                outputs["zh_srt"],
                cues,
                field="translated",
                max_chars=self.settings.subtitle_target_max_chars,
            )
            write_bilingual_vtt(
                outputs["bilingual_vtt"],
                cues,
                source_max_chars=self.settings.subtitle_source_max_chars,
                target_max_chars=self.settings.subtitle_target_max_chars,
            )

            if self.tts is not None:
                outputs["dubbed_audio"] = job_dir / "dubbed.wav"
                self._synthesize_dubbing(cues, job_dir, outputs["dubbed_audio"], job_logger)

            if is_video_file(resolved_video):
                outputs["translated_video"] = job_dir / "translated.mkv"
                self._mux_subtitles(
                    resolved_video,
                    outputs["zh_srt"],
                    outputs["translated_video"],
                    job_logger,
                )
                if self.tts is not None:
                    outputs["dubbed_video"] = job_dir / "dubbed.mkv"
                    self._mux_dubbed_video(
                        resolved_video,
                        outputs["dubbed_audio"],
                        outputs["zh_srt"],
                        outputs["dubbed_video"],
                        job_logger,
                    )

            job.outputs = stringify_outputs(outputs)
            job.status = "succeeded"
            self.store.save(job)
            job_logger.info(
                "job succeeded",
                extra={
                    "stage": "job",
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                    "output_path": str(job_dir),
                },
            )
            return job
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            self.store.save(job)
            job_logger.exception("job failed", extra={"stage": "job"})
            raise

    @property
    def asr(self) -> AsrBackend:
        if self._asr is None:
            self._asr = create_asr_backend(self.settings)
        return self._asr

    @property
    def alignment(self) -> AlignmentBackend:
        if self._alignment is None:
            self._alignment = create_alignment_backend(self.settings)
        return self._alignment

    @property
    def translator(self) -> Translator:
        if self._translator is None:
            self._translator = create_translator(self.settings)
        return self._translator

    @property
    def glossary(self) -> list[GlossaryTerm]:
        if self._glossary is None:
            self._glossary = load_glossary(self.settings.glossary_path, self.settings.translation_replacements)
        return self._glossary

    @property
    def vad(self) -> VadBackend:
        if self._vad is None:
            self._vad = create_vad_backend(self.settings)
        return self._vad

    @property
    def tts(self) -> TtsBackend | None:
        if self._tts is _UNSET:
            self._tts = create_tts_backend(self.settings)
        return None if self._tts is None else self._tts

    @property
    def audio_separation(self) -> AudioSeparationBackend:
        if self._audio_separation is None:
            self._audio_separation = create_audio_separation_backend(self.settings)
        return self._audio_separation

    def _extract_audio(self, video_path: Path, audio_path: Path, job_logger: logging.LoggerAdapter) -> None:
        stage_started = time.perf_counter()
        if video_path.suffix.lower() in DIRECT_AUDIO_SUFFIXES:
            shutil.copyfile(video_path, audio_path)
            job_logger.info(
                "audio copied from direct audio input",
                extra={
                    "stage": "extract_audio",
                    "duration_ms": int((time.perf_counter() - stage_started) * 1000),
                    "output_path": str(audio_path),
                },
            )
            return

        job_logger.info("extracting audio", extra={"stage": "extract_audio"})
        result = extract_audio(self.settings.ffmpeg_path, video_path, audio_path)
        if result.returncode != 0:
            logger.error("ffmpeg failed: %s", result.stderr)
            raise RuntimeError(f"ffmpeg failed with exit code {result.returncode}: {result.stderr[-1000:]}")
        job_logger.info(
            "audio extracted",
            extra={
                "stage": "extract_audio",
                "duration_ms": int((time.perf_counter() - stage_started) * 1000),
                "output_path": str(audio_path),
            },
        )

    def _separate_audio(
        self,
        audio_path: Path,
        job_dir: Path,
        job_logger: logging.LoggerAdapter,
    ) -> SeparatedAudio:
        stage_started = time.perf_counter()
        job_logger.info(
            "preparing audio lanes",
            extra={
                "stage": "audio_separation",
                "backend": self.settings.audio_separation_backend,
            },
        )
        separated = self.audio_separation.separate(
            audio_path=audio_path,
            work_dir=job_dir / "audio_lanes",
            ffmpeg_path=self.settings.ffmpeg_path,
        )
        job_logger.info(
            "audio lanes prepared",
            extra={
                "stage": "audio_separation",
                "duration_ms": int((time.perf_counter() - stage_started) * 1000),
                "backend": self.settings.audio_separation_backend,
                "original_audio": str(separated.original_audio),
                "vocals_audio": str(separated.vocals_audio),
                "background_audio": str(separated.background_audio),
            },
        )
        return separated

    def _transcribe_with_vad(
        self,
        audio_path: Path,
        job_dir: Path,
        job_logger: logging.LoggerAdapter,
    ) -> list[Segment]:
        stage_started = time.perf_counter()
        regions = self.vad.detect(audio_path)
        if not regions:
            regions = [SpeechRegion(start=0.0, end=0.1)]
        job_logger.info(
            "vad completed",
            extra={
                "stage": "vad",
                "duration_ms": int((time.perf_counter() - stage_started) * 1000),
                "region_count": len(regions),
            },
        )

        stage_started = time.perf_counter()
        segments: list[Segment] = []
        if len(regions) == 1 and regions[0].start <= 0.001:
            segments = self.asr.transcribe(audio_path)
        else:
            vad_dir = job_dir / "vad_segments"
            vad_dir.mkdir(parents=True, exist_ok=True)
            for index, region in enumerate(regions, 1):
                region_audio = vad_dir / f"{index:04d}.wav"
                result = slice_audio(
                    self.settings.ffmpeg_path,
                    audio_path,
                    region_audio,
                    region.start,
                    region.end,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg audio slice failed with exit code {result.returncode}: {result.stderr[-1000:]}")
                for segment in self.asr.transcribe(region_audio):
                    segments.append(
                        Segment(
                            start=segment.start + region.start,
                            end=segment.end + region.start,
                            text=segment.text,
                            speaker_id=segment.speaker_id,
                            words=[
                                {
                                    **word,
                                    "start": word.get("start", 0) + region.start,
                                    "end": word.get("end", 0) + region.start,
                                }
                                for word in segment.words
                            ],
                            confidence=segment.confidence,
                        )
                    )
        segments.sort(key=lambda item: (item.start, item.end))
        job_logger.info(
            "asr completed",
            extra={
                "stage": "asr",
                "duration_ms": int((time.perf_counter() - stage_started) * 1000),
                "segment_count": len(segments),
            },
        )
        return segments

    def _model_versions(self) -> dict[str, str]:
        return {
            "asr_backend": self.settings.asr_backend,
            "asr_model_size": self.settings.asr_model_size,
            "alignment_backend": self.settings.alignment_backend,
            "alignment_language": self.settings.alignment_language,
            "translator_backend": self.settings.translator_backend,
            "translator_model": self.settings.translator_model,
            "translator_api_base": self.settings.translator_api_base if self.settings.translator_backend == "llm" else "",
            "tts_backend": self.settings.tts_backend,
            "vad_backend": self.settings.vad_backend,
            "audio_separation_backend": self.settings.audio_separation_backend,
        }

    def _translation_requests(self, segments: list[Segment]) -> list[TranslationRequest]:
        requests: list[TranslationRequest] = []
        for index, segment in enumerate(segments):
            context_before = segments[index - 1].text if index > 0 else ""
            context_after = segments[index + 1].text if index + 1 < len(segments) else ""
            requests.append(
                TranslationRequest(
                    index=index + 1,
                    text=segment.text,
                    duration=max(segment.end - segment.start, 0.1),
                    context_before=context_before,
                    context_after=context_after,
                )
            )
        return requests

    def _align_segments(
        self,
        audio_path: Path,
        segments: list[Segment],
        job_logger: logging.LoggerAdapter,
    ) -> list[Segment]:
        stage_started = time.perf_counter()
        aligned = self.alignment.align(audio_path, segments)
        job_logger.info(
            "alignment completed",
            extra={
                "stage": "alignment",
                "duration_ms": int((time.perf_counter() - stage_started) * 1000),
                "segment_count": len(aligned),
                "word_count": sum(len(segment.words) for segment in aligned),
            },
        )
        return aligned

    def _log_quality_warnings(self, cues: list[SubtitleCue], job_logger: logging.LoggerAdapter) -> None:
        for issue in check_reading_speed(cues, self.settings.subtitle_target_max_cps):
            job_logger.warning(
                "subtitle reading speed exceeds target",
                extra={
                    "stage": "quality",
                    "cue_index": issue.cue_index,
                    "chars_per_second": round(issue.chars_per_second, 3),
                    "max_chars_per_second": issue.max_chars_per_second,
                },
            )

    def _mux_subtitles(
        self,
        video_path: Path,
        subtitle_path: Path,
        output_path: Path,
        job_logger: logging.LoggerAdapter,
    ) -> None:
        stage_started = time.perf_counter()
        job_logger.info("muxing translated subtitle track", extra={"stage": "mux_subtitles"})
        result = mux_subtitle_track(
            self.settings.ffmpeg_path,
            video_path,
            subtitle_path,
            output_path,
        )
        if result.returncode != 0:
            logger.error("ffmpeg subtitle mux failed: %s", result.stderr)
            raise RuntimeError(f"ffmpeg subtitle mux failed with exit code {result.returncode}: {result.stderr[-1000:]}")
        job_logger.info(
            "translated video created",
            extra={
                "stage": "mux_subtitles",
                "duration_ms": int((time.perf_counter() - stage_started) * 1000),
                "output_path": str(output_path),
            },
        )

    def _synthesize_dubbing(
        self,
        cues: list[SubtitleCue],
        job_dir: Path,
        output_path: Path,
        job_logger: logging.LoggerAdapter,
    ) -> None:
        if self.tts is None:
            return
        stage_started = time.perf_counter()
        job_logger.info("synthesizing translated speech", extra={"stage": "tts"})
        synthesize_dubbed_audio(
            cues=cues,
            tts=self.tts,
            work_dir=job_dir,
            output_path=output_path,
            ffmpeg_path=self.settings.ffmpeg_path,
        )
        job_logger.info(
            "translated speech synthesized",
            extra={
                "stage": "tts",
                "duration_ms": int((time.perf_counter() - stage_started) * 1000),
                "output_path": str(output_path),
            },
        )

    def _mux_dubbed_video(
        self,
        video_path: Path,
        dubbed_audio_path: Path,
        subtitle_path: Path,
        output_path: Path,
        job_logger: logging.LoggerAdapter,
    ) -> None:
        stage_started = time.perf_counter()
        job_logger.info("muxing dubbed video", extra={"stage": "mux_dubbed_video"})
        result = mux_dubbed_video(
            self.settings.ffmpeg_path,
            video_path,
            dubbed_audio_path,
            subtitle_path,
            output_path,
        )
        if result.returncode != 0:
            logger.error("ffmpeg dubbed video mux failed: %s", result.stderr)
            raise RuntimeError(f"ffmpeg dubbed video mux failed with exit code {result.returncode}: {result.stderr[-1000:]}")
        job_logger.info(
            "dubbed video created",
            extra={
                "stage": "mux_dubbed_video",
                "duration_ms": int((time.perf_counter() - stage_started) * 1000),
                "output_path": str(output_path),
            },
        )


_UNSET = object()
