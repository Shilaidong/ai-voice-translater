from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.request import Request

from aivoice.tts import TtsSynthesisRequest
from aivoice.tts.openai_speech import OpenAICompatibleSpeechTtsBackend


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False

    def read(self) -> bytes:
        return b"RIFFfake-wav"


def test_openai_speech_tts_sends_reference_audio(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: Request, timeout: int):  # noqa: ANN001
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("aivoice.tts.openai_speech.urlopen", fake_urlopen)
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"reference-audio")
    output_path = tmp_path / "speech.wav"

    backend = OpenAICompatibleSpeechTtsBackend(
        api_base="http://127.0.0.1:8000/v1",
        api_key="secret",
        model="voxcpm2",
        voice="default",
        timeout_seconds=12,
    )
    backend.synthesize_request(
        TtsSynthesisRequest(
            text="编码器会映射输入嵌入。",
            output_path=output_path,
            target_duration=3.2,
            reference_audio=reference_audio,
            reference_text="The encoder maps input embeddings.",
            style_hint="Match the source speaker's pace.",
            speaker_id="speaker-a",
        )
    )

    assert output_path.read_bytes() == b"RIFFfake-wav"
    assert captured["url"] == "http://127.0.0.1:8000/v1/audio/speech"
    assert captured["timeout"] == 12
    assert captured["headers"]["Authorization"] == "Bearer secret"
    payload = captured["payload"]
    assert payload["model"] == "voxcpm2"
    assert payload["input"] == "编码器会映射输入嵌入。"
    assert payload["target_duration"] == 3.2
    assert payload["reference_text"] == "The encoder maps input embeddings."
    assert payload["speaker_id"] == "speaker-a"
    assert payload["reference_audio"] == base64.b64encode(b"reference-audio").decode("ascii")
