from __future__ import annotations

from aivoice.translation.glossary import load_glossary


def test_load_glossary_merges_replacements_and_file_terms(tmp_path) -> None:
    glossary_path = tmp_path / "glossary.txt"
    glossary_path.write_text(
        "\n".join(
            [
                "# course terms",
                "encoder=编码器",
                "decoder\t解码器",
                "embedding,嵌入",
            ]
        ),
        encoding="utf-8",
    )

    terms = load_glossary(glossary_path, (("Transformer", "Transformer 模型"),))

    assert [(term.source, term.target) for term in terms] == [
        ("Transformer", "Transformer 模型"),
        ("encoder", "编码器"),
        ("decoder", "解码器"),
        ("embedding", "嵌入"),
    ]
