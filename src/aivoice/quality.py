from __future__ import annotations

from dataclasses import dataclass

from .models import SubtitleCue

IGNORED_SPEECH_CHARS = set(" \t\r\n，。！？；：、,.!?;:()（）[]【】\"'“”‘’")


@dataclass(frozen=True)
class ReadingSpeedIssue:
    cue_index: int
    chars_per_second: float
    max_chars_per_second: float
    text: str


def speech_char_count(text: str) -> int:
    return len([char for char in text if char not in IGNORED_SPEECH_CHARS])


def check_reading_speed(cues: list[SubtitleCue], max_chars_per_second: float) -> list[ReadingSpeedIssue]:
    issues: list[ReadingSpeedIssue] = []
    if max_chars_per_second <= 0:
        return issues

    for cue in cues:
        duration = max(cue.end - cue.start, 0.001)
        cps = speech_char_count(cue.translated_text) / duration
        if cps > max_chars_per_second:
            issues.append(
                ReadingSpeedIssue(
                    cue_index=cue.index,
                    chars_per_second=cps,
                    max_chars_per_second=max_chars_per_second,
                    text=cue.translated_text,
                )
            )
    return issues
