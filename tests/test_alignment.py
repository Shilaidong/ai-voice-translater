from __future__ import annotations

import pytest

from aivoice.alignment.noop import NoopAlignmentBackend
from aivoice.alignment.whisperx import _segments_from_whisperx
from aivoice.backends import create_alignment_backend
from aivoice.models import Segment

from tests.test_pipeline import make_settings


def test_noop_alignment_returns_segments(tmp_path) -> None:
    backend = create_alignment_backend(make_settings(tmp_path / "data"))
    segments = [Segment(start=0.0, end=1.0, text="hello")]

    assert isinstance(backend, NoopAlignmentBackend)
    assert backend.align(tmp_path / "audio.wav", segments) == segments


def test_whisperx_result_converts_words_to_segments() -> None:
    fallback = [Segment(start=0.0, end=1.0, text="hello world", speaker_id="speaker_1")]
    aligned = {
        "segments": [
            {
                "start": 0.1,
                "end": 0.9,
                "text": "hello world",
                "words": [
                    {"word": "hello", "start": 0.1, "end": 0.4, "score": 0.9},
                    {"word": "world", "start": 0.5, "end": 0.9},
                ],
            }
        ]
    }

    result = _segments_from_whisperx(aligned, fallback)

    assert result[0].start == 0.1
    assert result[0].end == 0.9
    assert result[0].speaker_id == "speaker_1"
    assert result[0].words == [
        {"word": "hello", "start": 0.1, "end": 0.4, "score": 0.9},
        {"word": "world", "start": 0.5, "end": 0.9},
    ]


def test_whisperx_alignment_rejects_non_english_language() -> None:
    from aivoice.alignment.whisperx import WhisperXAlignmentBackend

    with pytest.raises(RuntimeError, match="only enabled for English"):
        WhisperXAlignmentBackend(language_code="zho_Hans")
