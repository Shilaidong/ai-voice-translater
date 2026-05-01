from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import GlossaryTerm, TranslationRequest, Translator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    text: str
    notes: str = ""


class OpenAICompatibleTranslator(Translator):
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        max_tokens: int = 512,
        timeout_seconds: int = 120,
        candidate_count: int = 3,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.candidate_count = max(candidate_count, 1)

    def translate_batch(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        requests = [
            TranslationRequest(
                index=index + 1,
                text=text,
                duration=max(len(text) / 12.0, 1.0),
            )
            for index, text in enumerate(texts)
        ]
        return self.translate_segments(requests, source_lang, target_lang, glossary=[])

    def translate_segments(
        self,
        requests: list[TranslationRequest],
        source_lang: str,
        target_lang: str,
        glossary: list[GlossaryTerm],
    ) -> list[str]:
        translations: list[str] = []
        for item in requests:
            content = self._complete(self._messages(item, source_lang, target_lang, glossary))
            candidates = self._parse_candidates(content)
            selected = self._select_candidate(candidates, item, glossary)
            self._log_candidate_quality(selected, item, glossary)
            translations.append(selected.text)
        return translations

    def _messages(
        self,
        item: TranslationRequest,
        source_lang: str,
        target_lang: str,
        glossary: list[GlossaryTerm],
    ) -> list[dict[str, str]]:
        min_chars = max(1, round(item.duration * 5))
        max_chars = max(min_chars, round(item.duration * 7))
        glossary_text = "\n".join(f"- {term.source} => {term.target}" for term in glossary) or "- none"
        system = (
            "You are a course-video dubbing translator. Translate accurately into natural Simplified Chinese. "
            "Preserve technical terms according to the glossary. Do not add explanations. "
            "Return strict JSON only."
        )
        user = {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "duration_seconds": round(item.duration, 3),
            "target_chinese_character_range": [min_chars, max_chars],
            "glossary": glossary_text,
            "context_before": item.context_before,
            "source_text": item.text,
            "context_after": item.context_after,
            "required_json_schema": {
                "candidates": [
                    {
                        "text": "Chinese translation candidate",
                        "notes": "short optional note about terms or timing",
                    }
                ]
            },
            "candidate_count": self.candidate_count,
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]

    def _complete(self, messages: list[dict[str, str]]) -> str:
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM translator HTTP {exc.code}: {detail[-1000:]}") from exc
        except URLError as exc:
            raise RuntimeError(f"LLM translator request failed: {exc}") from exc

        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM translator response: {body}") from exc

    def _parse_candidates(self, content: str) -> list[Candidate]:
        data = self._load_json(content)
        raw_candidates: Any
        if isinstance(data, dict):
            raw_candidates = data.get("candidates", data.get("translations", data.get("choices", [])))
        else:
            raw_candidates = data

        candidates: list[Candidate] = []
        if isinstance(raw_candidates, list):
            for item in raw_candidates:
                if isinstance(item, str):
                    text = item
                    notes = ""
                elif isinstance(item, dict):
                    text = str(item.get("text") or item.get("translation") or "").strip()
                    notes = str(item.get("notes") or "").strip()
                else:
                    text = ""
                    notes = ""
                if text:
                    candidates.append(Candidate(text=text, notes=notes))
        elif isinstance(raw_candidates, str):
            candidates.append(Candidate(text=raw_candidates))

        if not candidates:
            raise RuntimeError(f"LLM translator returned no valid candidates: {content[:500]}")
        return candidates

    def _load_json(self, content: str) -> Any:
        text = content.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if match:
                return json.loads(match.group(0))
            raise RuntimeError(f"LLM translator did not return JSON: {content[:500]}")

    def _select_candidate(
        self,
        candidates: list[Candidate],
        request: TranslationRequest,
        glossary: list[GlossaryTerm],
    ) -> Candidate:
        ideal_chars = max(request.duration * 6.0, 1.0)
        relevant_terms = _relevant_terms(request.text, glossary)

        def score(candidate: Candidate) -> tuple[int, float]:
            missing_terms = _missing_target_terms(candidate.text, relevant_terms)
            duration_distance = abs(_speech_char_count(candidate.text) - ideal_chars)
            return (len(missing_terms), duration_distance)

        return min(candidates, key=score)

    def _log_candidate_quality(
        self,
        candidate: Candidate,
        request: TranslationRequest,
        glossary: list[GlossaryTerm],
    ) -> None:
        min_chars = max(1, round(request.duration * 5))
        max_chars = max(min_chars, round(request.duration * 7))
        char_count = _speech_char_count(candidate.text)
        missing_terms = _missing_target_terms(candidate.text, _relevant_terms(request.text, glossary))
        if missing_terms or char_count < min_chars or char_count > max_chars:
            logger.warning(
                "llm translation candidate outside quality targets",
                extra={
                    "stage": "translation",
                    "cue_index": request.index,
                    "duration_seconds": request.duration,
                    "char_count": char_count,
                    "target_min_chars": min_chars,
                    "target_max_chars": max_chars,
                    "missing_terms": [term.source for term in missing_terms],
                },
            )


def _speech_char_count(text: str) -> int:
    ignored = set(" \t\r\n，。！？；：、,.!?;:()（）[]【】\"'“”‘’")
    return len([char for char in text if char not in ignored])


def _relevant_terms(source_text: str, glossary: list[GlossaryTerm]) -> list[GlossaryTerm]:
    return [term for term in glossary if _contains_term(source_text, term.source)]


def _missing_target_terms(text: str, terms: list[GlossaryTerm]) -> list[GlossaryTerm]:
    return [term for term in terms if term.target and term.target not in text]


def _contains_term(text: str, term: str) -> bool:
    if not term:
        return False
    if any(char.isascii() and (char.isalnum() or char == "_") for char in term):
        return re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", text, flags=re.I) is not None
    return term in text
