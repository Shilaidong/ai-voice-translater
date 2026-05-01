from __future__ import annotations

import re


def _has_ascii_word_chars(text: str) -> bool:
    return any(char.isascii() and (char.isalnum() or char == "_") for char in text)


def apply_replacements(text: str, replacements: tuple[tuple[str, str], ...]) -> str:
    result = text
    for source, target in replacements:
        if _has_ascii_word_chars(source):
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(source)}(?![A-Za-z0-9_])"
            result = re.sub(pattern, target, result)
        else:
            result = result.replace(source, target)
    return result
