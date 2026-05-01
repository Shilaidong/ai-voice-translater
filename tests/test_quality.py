from __future__ import annotations

from aivoice.models import SubtitleCue
from aivoice.quality import check_reading_speed, speech_char_count


def test_speech_char_count_ignores_common_punctuation() -> None:
    assert speech_char_count("编码器，会映射 input。") == 11


def test_check_reading_speed_flags_fast_cues() -> None:
    cues = [
        SubtitleCue(
            index=1,
            start=0.0,
            end=1.0,
            source_text="source",
            translated_text="这是一个明显太长的中文字幕",
        )
    ]

    issues = check_reading_speed(cues, max_chars_per_second=7.0)

    assert len(issues) == 1
    assert issues[0].cue_index == 1
    assert issues[0].chars_per_second > 7.0
