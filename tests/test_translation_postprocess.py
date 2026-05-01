from aivoice.config import parse_replacements
from aivoice.translation.postprocess import apply_replacements


def test_parse_replacements() -> None:
    assert parse_replacements("当地语音识别=本地语音识别;地方翻译=本地翻译") == (
        ("当地语音识别", "本地语音识别"),
        ("地方翻译", "本地翻译"),
    )


def test_apply_replacements() -> None:
    replacements = (("当地语音识别", "本地语音识别"),)
    assert apply_replacements("这是当地语音识别测试。", replacements) == "这是本地语音识别测试。"


def test_apply_replacements_respects_ascii_word_boundaries() -> None:
    replacements = (("Transformer", "Transformer 模型"),)
    text = "Transformer is different from Transformers and MyTransformer."

    assert (
        apply_replacements(text, replacements)
        == "Transformer 模型 is different from Transformers and MyTransformer."
    )
