from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

from aivoice.media import probe_duration
from aivoice.separation import DemucsAudioSeparationBackend, NoopAudioSeparationBackend


def create_wav(path: Path, seconds: float = 1.0) -> None:
    rate = 16000
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        frames = bytearray()
        for index in range(int(rate * seconds)):
            value = int(12000 * math.sin(2 * math.pi * 440 * index / rate))
            frames += value.to_bytes(2, "little", signed=True)
        wav.writeframes(frames)


def test_noop_audio_separation_creates_future_dubbing_lanes(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    create_wav(source, seconds=1.2)

    result = NoopAudioSeparationBackend().separate(
        audio_path=source,
        work_dir=tmp_path / "lanes",
        ffmpeg_path="ffmpeg",
    )

    assert result.original_audio == source
    assert result.vocals_audio.exists()
    assert result.background_audio.exists()
    assert abs(probe_duration(result.background_audio) - probe_duration(source)) < 0.05


def test_demucs_audio_separation_runs_expected_command(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    create_wav(source, seconds=0.4)

    class FakeSpec:
        pass

    def fake_find_spec(name: str):
        return FakeSpec() if name == "demucs" else None

    def fake_run(command, capture_output, text):  # noqa: ANN001
        if "-o" not in command:
            create_wav(Path(command[-1]), seconds=0.4)
            return subprocess.CompletedProcess(command, 0, "", "")

        output_root = Path(command[command.index("-o") + 1])
        model_name = command[command.index("-n") + 1]
        audio_path = Path(command[-1])
        output_dir = output_root / model_name / audio_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        create_wav(output_dir / "vocals.wav", seconds=0.4)
        create_wav(output_dir / "no_vocals.wav", seconds=0.4)
        assert command[1:3] == ["-m", "demucs"]
        assert "--two-stems" in command
        assert command[command.index("--two-stems") + 1] == "vocals"
        assert command[command.index("-d") + 1] == "cpu"
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("aivoice.separation.demucs.importlib.util.find_spec", fake_find_spec)
    monkeypatch.setattr("aivoice.separation.demucs.subprocess.run", fake_run)

    result = DemucsAudioSeparationBackend(model_name="htdemucs_ft", device="cpu").separate(
        audio_path=source,
        work_dir=tmp_path / "lanes",
        ffmpeg_path="ffmpeg",
    )

    assert result.vocals_audio.exists()
    assert result.background_audio.exists()
    assert result.original_audio == source
