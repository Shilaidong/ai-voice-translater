from __future__ import annotations

from pathlib import Path

from .media import mix_timed_audio
from .models import SubtitleCue
from .tts import TtsBackend


def synthesize_dubbed_audio(
    cues: list[SubtitleCue],
    tts: TtsBackend,
    work_dir: Path,
    output_path: Path,
    ffmpeg_path: str,
) -> None:
    segment_dir = work_dir / "tts_segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    timed_audio_paths: list[tuple[float, Path]] = []
    for cue in cues:
        text = cue.translated_text.strip()
        if not text:
            continue
        segment_path = segment_dir / f"{cue.index:04d}.wav"
        tts.synthesize(text, segment_path)
        timed_audio_paths.append((cue.start, segment_path))

    duration = max((cue.end for cue in cues), default=0.1)
    result = mix_timed_audio(ffmpeg_path, timed_audio_paths, output_path, duration)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio mix failed with exit code {result.returncode}: {result.stderr[-1000:]}")
