from __future__ import annotations

from pathlib import Path

from .media import mix_timed_audio, slice_audio
from .models import SubtitleCue
from .tts import TtsBackend, TtsSynthesisRequest


def synthesize_dubbed_audio(
    cues: list[SubtitleCue],
    tts: TtsBackend,
    work_dir: Path,
    output_path: Path,
    ffmpeg_path: str,
    source_audio_path: Path | None = None,
) -> None:
    segment_dir = work_dir / "tts_segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = work_dir / "tts_references"
    if tts.supports_reference_audio and source_audio_path is not None:
        reference_dir.mkdir(parents=True, exist_ok=True)
    timed_audio_paths: list[tuple[float, Path]] = []
    for cue in cues:
        text = cue.translated_text.strip()
        if not text:
            continue
        segment_path = segment_dir / f"{cue.index:04d}.wav"
        reference_audio = None
        if tts.supports_reference_audio and source_audio_path is not None:
            reference_audio = reference_dir / f"{cue.index:04d}.wav"
            result = slice_audio(
                ffmpeg_path,
                source_audio_path,
                reference_audio,
                cue.start,
                cue.end,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg reference audio slice failed with exit code {result.returncode}: {result.stderr[-1000:]}")
        tts.synthesize_request(
            TtsSynthesisRequest(
                text=text,
                output_path=segment_path,
                target_duration=cue.duration_budget or max(cue.end - cue.start, 0.1),
                reference_audio=reference_audio,
                reference_text=cue.source_text,
                style_hint="Match the source speaker's pace and tone for course dubbing.",
                speaker_id=cue.speaker_id,
            )
        )
        timed_audio_paths.append((cue.start, segment_path))

    duration = max((cue.end for cue in cues), default=0.1)
    result = mix_timed_audio(ffmpeg_path, timed_audio_paths, output_path, duration)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio mix failed with exit code {result.returncode}: {result.stderr[-1000:]}")
