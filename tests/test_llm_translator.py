from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from aivoice.translation import GlossaryTerm, TranslationRequest
from aivoice.translation.llm_adapter import Candidate, OpenAICompatibleTranslator


class _Handler(BaseHTTPRequestHandler):
    payloads: list[dict] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.payloads.append(payload)
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "candidates": [
                                    {"text": "短译文"},
                                    {"text": "这是长度更合适的译文"},
                                    {"text": "这是一个明显太长太长的翻译候选"},
                                ]
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }
        data = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        return


def test_openai_compatible_translator_uses_duration_context_and_glossary() -> None:
    _Handler.payloads = []
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        translator = OpenAICompatibleTranslator(
            api_base=f"http://127.0.0.1:{server.server_port}/v1",
            api_key="test-key",
            model="qwen-test",
            max_tokens=128,
            timeout_seconds=5,
            candidate_count=3,
        )
        result = translator.translate_segments(
            [
                TranslationRequest(
                    index=1,
                    text="The encoder maps input embeddings.",
                    duration=2.0,
                    context_before="We discussed tokenization.",
                    context_after="Then the decoder generates logits.",
                )
            ],
            source_lang="eng_Latn",
            target_lang="zho_Hans",
            glossary=[GlossaryTerm(source="encoder", target="编码器")],
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert result == ["这是长度更合适的译文"]
    payload = _Handler.payloads[0]
    assert payload["model"] == "qwen-test"
    assert payload["messages"][0]["role"] == "system"
    user = json.loads(payload["messages"][1]["content"])
    assert user["duration_seconds"] == 2.0
    assert user["target_chinese_character_range"] == [10, 14]
    assert "encoder => 编码器" in user["glossary"]
    assert user["context_before"] == "We discussed tokenization."


def test_candidate_selection_prefers_required_glossary_terms() -> None:
    translator = OpenAICompatibleTranslator(
        api_base="http://127.0.0.1:1/v1",
        api_key="",
        model="qwen-test",
    )
    request = TranslationRequest(
        index=1,
        text="The Transformer encoder maps tokens.",
        duration=2.0,
    )
    selected = translator._select_candidate(
        [
            Candidate(text="它会映射词元"),
            Candidate(text="Transformer 编码器会映射词元"),
        ],
        request,
        [GlossaryTerm(source="encoder", target="编码器")],
    )

    assert selected.text == "Transformer 编码器会映射词元"


def test_parse_candidates_rejects_missing_candidate_list() -> None:
    translator = OpenAICompatibleTranslator(
        api_base="http://127.0.0.1:1/v1",
        api_key="",
        model="qwen-test",
    )

    with pytest.raises(RuntimeError, match="no valid candidates"):
        translator._parse_candidates('{"message":"not a candidate response"}')
