from __future__ import annotations

from pathlib import Path

from aivoice.pipeline import OfflinePipeline
from aivoice.storage import JobStore

from tests.test_pipeline import create_wav, make_settings


GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def test_mock_pipeline_matches_golden_srt_outputs(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    create_wav(wav_path)
    settings = make_settings(tmp_path / "data")
    store = JobStore(settings.jobs_dir)

    job = OfflinePipeline(settings, store).process(wav_path)

    assert _normalized(Path(job.outputs["source_srt"])) == _normalized(GOLDEN_DIR / "mock_source.srt")
    assert _normalized(Path(job.outputs["zh_srt"])) == _normalized(GOLDEN_DIR / "mock_zh.srt")


def _normalized(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip().replace("\r\n", "\n")
