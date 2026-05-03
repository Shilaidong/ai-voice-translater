from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import TtsBackend, TtsSynthesisRequest


class OpenAICompatibleSpeechTtsBackend(TtsBackend):
    supports_reference_audio = True
    supports_target_duration = True

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        voice: str,
        timeout_seconds: int = 300,
        response_format: str = "wav",
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.timeout_seconds = timeout_seconds
        self.response_format = response_format

    def synthesize(self, text: str, output_path: Path) -> None:
        self.synthesize_request(TtsSynthesisRequest(text=text, output_path=output_path))

    def synthesize_request(self, request: TtsSynthesisRequest) -> None:
        output_path = request.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "input": request.text,
            "voice": self.voice,
            "response_format": self.response_format,
        }
        if request.target_duration is not None:
            payload["target_duration"] = round(request.target_duration, 3)
        if request.reference_text:
            payload["reference_text"] = request.reference_text
        if request.style_hint:
            payload["style_hint"] = request.style_hint
        if request.speaker_id:
            payload["speaker_id"] = request.speaker_id
        if request.reference_audio:
            payload["reference_audio"] = base64.b64encode(request.reference_audio.read_bytes()).decode("ascii")
            payload["reference_audio_format"] = request.reference_audio.suffix.lstrip(".") or "wav"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = Request(
            f"{self.api_base}/audio/speech",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=self.timeout_seconds) as response:
                output_path.write_bytes(response.read())
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"TTS endpoint HTTP {exc.code}: {detail[-1000:]}") from exc
        except URLError as exc:
            raise RuntimeError(f"TTS endpoint request failed: {exc}") from exc
