from __future__ import annotations

from pathlib import Path

from .models import SubtitleCue

CJK_PUNCTUATION = "，。！？；、,.;!?"


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    sec = total_seconds % 60
    total_minutes = total_seconds // 60
    minute = total_minutes % 60
    hour = total_minutes // 60
    return f"{hour:02}:{minute:02}:{sec:02},{ms:03}"


def format_vtt_timestamp(seconds: float) -> str:
    return format_srt_timestamp(seconds).replace(",", ".")


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def wrap_subtitle_text(text: str, max_chars: int | None) -> str:
    normalized = " ".join(text.strip().split())
    if not normalized or not max_chars or max_chars <= 0 or len(normalized) <= max_chars:
        return normalized
    if contains_cjk(normalized):
        return "\n".join(_wrap_cjk(normalized, max_chars))
    return "\n".join(_wrap_words(normalized, max_chars))


def _wrap_cjk(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        min_break = max(1, int(max_chars * 0.55))
        cut = max_chars
        for index in range(max_chars - 1, min_break - 1, -1):
            if remaining[index] in CJK_PUNCTUATION:
                cut = index + 1
                break
        lines.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        lines.append(remaining)
    return lines


def _wrap_words(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def write_srt(path: Path, cues: list[SubtitleCue], field: str, max_chars: int | None = None) -> None:
    lines: list[str] = []
    for cue in cues:
        text = cue.source_text if field == "source" else cue.translated_text
        lines.extend(
            [
                str(cue.index),
                f"{format_srt_timestamp(cue.start)} --> {format_srt_timestamp(cue.end)}",
                wrap_subtitle_text(text, max_chars),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_bilingual_vtt(
    path: Path,
    cues: list[SubtitleCue],
    source_max_chars: int | None = None,
    target_max_chars: int | None = None,
) -> None:
    lines = ["WEBVTT", ""]
    for cue in cues:
        lines.extend(
            [
                f"{format_vtt_timestamp(cue.start)} --> {format_vtt_timestamp(cue.end)}",
                wrap_subtitle_text(cue.translated_text, target_max_chars),
                wrap_subtitle_text(cue.source_text, source_max_chars),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
