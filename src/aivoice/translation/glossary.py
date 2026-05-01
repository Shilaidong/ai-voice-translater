from __future__ import annotations

from pathlib import Path

from .base import GlossaryTerm


def load_glossary(path: Path | None, replacements: tuple[tuple[str, str], ...]) -> list[GlossaryTerm]:
    terms = [GlossaryTerm(source=source, target=target) for source, target in replacements]
    if path is None:
        return terms
    resolved = path.expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Glossary file not found: {resolved}")

    for raw_line in resolved.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            source, target = line.split("=", 1)
        elif "\t" in line:
            source, target = line.split("\t", 1)
        elif "," in line:
            source, target = line.split(",", 1)
        else:
            continue
        source = source.strip()
        target = target.strip()
        if source and target:
            terms.append(GlossaryTerm(source=source, target=target))
    return terms
