from aivoice.models import SubtitleCue
from aivoice.subtitles import (
    format_srt_timestamp,
    format_vtt_timestamp,
    wrap_subtitle_text,
    write_bilingual_vtt,
    write_srt,
)


def test_srt_timestamp_rounding() -> None:
    assert format_srt_timestamp(0) == "00:00:00,000"
    assert format_srt_timestamp(62.3456) == "00:01:02,346"
    assert format_srt_timestamp(3661.2) == "01:01:01,200"


def test_vtt_timestamp() -> None:
    assert format_vtt_timestamp(1.5) == "00:00:01.500"


def test_wrap_english_by_words() -> None:
    wrapped = wrap_subtitle_text(
        "This is a long subtitle line that should wrap on word boundaries.",
        max_chars=24,
    )
    assert wrapped == "This is a long subtitle\nline that should wrap on\nword boundaries."


def test_wrap_chinese_prefers_punctuation() -> None:
    wrapped = wrap_subtitle_text("这是一个测试，用于验证中文字幕换行是否正常。", max_chars=12)
    assert wrapped == "这是一个测试，\n用于验证中文字幕换行是否\n正常。"
    assert all(len(line) <= 12 for line in wrapped.splitlines())


def test_write_srt_wraps_text(tmp_path) -> None:
    path = tmp_path / "zh.srt"
    cues = [
        SubtitleCue(
            index=1,
            start=0,
            end=2,
            source_text="source",
            translated_text="这是一个测试，用于验证中文字幕换行是否正常。",
        )
    ]
    write_srt(path, cues, field="translated", max_chars=12)
    assert "这是一个测试，\n用于验证中文字幕换行是否\n正常。" in path.read_text(encoding="utf-8")


def test_write_bilingual_vtt_wraps_both_tracks(tmp_path) -> None:
    path = tmp_path / "bilingual.vtt"
    cues = [
        SubtitleCue(
            index=1,
            start=0,
            end=2,
            source_text="This is a long subtitle line that should wrap.",
            translated_text="这是一个测试，用于验证中文字幕换行是否正常。",
        )
    ]
    write_bilingual_vtt(path, cues, source_max_chars=20, target_max_chars=12)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("WEBVTT")
    assert "这是一个测试，\n用于验证中文字幕换行是否\n正常。" in text
    assert "This is a long\nsubtitle line that\nshould wrap." in text
