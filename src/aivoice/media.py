from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


@dataclass(frozen=True)
class ToolCheck:
    name: str
    path: str | None
    ok: bool
    detail: str


def resolve_ffmpeg(configured_path: str = "ffmpeg") -> str:
    if configured_path and configured_path != "ffmpeg":
        return configured_path

    system_path = shutil.which("ffmpeg")
    if system_path:
        return system_path

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg is required but was not found on PATH, and imageio-ffmpeg "
            f"fallback failed: {exc}"
        ) from exc


def check_ffmpeg(configured_path: str = "ffmpeg") -> ToolCheck:
    try:
        ffmpeg = resolve_ffmpeg(configured_path)
        result = subprocess.run(
            [ffmpeg, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        first_line = (result.stdout or result.stderr).splitlines()[0] if (result.stdout or result.stderr) else ""
        return ToolCheck("ffmpeg", ffmpeg, result.returncode == 0, first_line)
    except Exception as exc:
        return ToolCheck("ffmpeg", None, False, str(exc))


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def extract_audio(ffmpeg_path: str, video_path: Path, audio_path: Path) -> subprocess.CompletedProcess[str]:
    ffmpeg = resolve_ffmpeg(ffmpeg_path)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]
    return subprocess.run(command, capture_output=True, text=True)


def probe_duration(path: Path, ffmpeg_path: str = "ffmpeg") -> float:
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return frames / float(rate)

    ffmpeg = resolve_ffmpeg(ffmpeg_path)
    ffprobe = str(Path(ffmpeg).with_name("ffprobe.exe" if Path(ffmpeg).suffix.lower() == ".exe" else "ffprobe"))
    if not Path(ffprobe).exists():
        ffprobe = shutil.which("ffprobe") or ffprobe
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed with exit code {result.returncode}: {result.stderr[-1000:]}")
    return max(float(result.stdout.strip()), 0.0)


def slice_audio(
    ffmpeg_path: str,
    audio_path: Path,
    output_path: Path,
    start: float,
    end: float,
) -> subprocess.CompletedProcess[str]:
    ffmpeg = resolve_ffmpeg(ffmpeg_path)
    command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(audio_path),
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]
    return subprocess.run(command, capture_output=True, text=True)


def mux_subtitle_track(
    ffmpeg_path: str,
    video_path: Path,
    subtitle_path: Path,
    output_path: Path,
) -> subprocess.CompletedProcess[str]:
    ffmpeg = resolve_ffmpeg(ffmpeg_path)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(subtitle_path),
        "-map",
        "0",
        "-map",
        "1:0",
        "-c",
        "copy",
        "-c:s",
        "srt",
        "-metadata:s:s:0",
        "language=chi",
        str(output_path),
    ]
    return subprocess.run(command, capture_output=True, text=True)


def mix_timed_audio(
    ffmpeg_path: str,
    timed_audio_paths: list[tuple[float, Path]],
    output_path: Path,
    duration: float,
) -> subprocess.CompletedProcess[str]:
    ffmpeg = resolve_ffmpeg(ffmpeg_path)
    if not timed_audio_paths:
        command = [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=mono:sample_rate=16000",
            "-t",
            f"{duration:.3f}",
            str(output_path),
        ]
        return subprocess.run(command, capture_output=True, text=True)

    command = [ffmpeg, "-y"]
    for _, audio_path in timed_audio_paths:
        command.extend(["-i", str(audio_path)])

    delayed_labels = []
    filters = []
    for index, (start, _) in enumerate(timed_audio_paths):
        label = f"a{index}"
        delay_ms = max(int(start * 1000), 0)
        filters.append(f"[{index}:a]adelay={delay_ms}:all=1[{label}]")
        delayed_labels.append(f"[{label}]")
    if len(delayed_labels) == 1:
        filters.append(f"{delayed_labels[0]}atrim=0:{duration:.3f},asetpts=N/SR/TB[mix]")
    else:
        filters.append(
            "".join(delayed_labels)
            + f"amix=inputs={len(delayed_labels)}:duration=longest:dropout_transition=0,"
            + f"atrim=0:{duration:.3f},asetpts=N/SR/TB[mix]"
        )

    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[mix]",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_path),
        ]
    )
    return subprocess.run(command, capture_output=True, text=True)


def mux_dubbed_video(
    ffmpeg_path: str,
    video_path: Path,
    dubbed_audio_path: Path,
    subtitle_path: Path,
    output_path: Path,
) -> subprocess.CompletedProcess[str]:
    ffmpeg = resolve_ffmpeg(ffmpeg_path)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(dubbed_audio_path),
        "-i",
        str(subtitle_path),
        "-map",
        "0:v?",
        "-map",
        "1:a",
        "-map",
        "2:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-c:s",
        "srt",
        "-metadata:s:a:0",
        "language=chi",
        "-metadata:s:a:0",
        "title=Chinese Dub",
        "-metadata:s:s:0",
        "language=chi",
        "-disposition:a:0",
        "default",
        str(output_path),
    ]
    return subprocess.run(command, capture_output=True, text=True)
