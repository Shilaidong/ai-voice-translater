from __future__ import annotations

from .base import TtsBackend, TtsSynthesisRequest
from .mock import MockTtsBackend

__all__ = ["MockTtsBackend", "TtsBackend", "TtsSynthesisRequest"]
