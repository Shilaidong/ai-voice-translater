from __future__ import annotations

import subprocess
from pathlib import Path

from .base import TtsBackend


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class WindowsSapiTtsBackend(TtsBackend):
    def __init__(self, voice_hint: str = "Chinese", rate: int = 0, volume: int = 100) -> None:
        self.voice_hint = voice_hint
        self.rate = rate
        self.volume = volume

    def synthesize(self, text: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        text_path = output_path.with_suffix(".txt")
        text_path.write_text(text, encoding="utf-8")
        script = f"""
$ErrorActionPreference = 'Stop'
$text = Get-Content -LiteralPath {_ps_quote(str(text_path))} -Raw -Encoding UTF8
$voiceHint = {_ps_quote(self.voice_hint)}
$voice = New-Object -ComObject SAPI.SpVoice
$tokens = @($voice.GetVoices())
if ($voiceHint) {{
  $match = @($tokens | Where-Object {{ $_.GetDescription() -like "*$voiceHint*" }})
  if ($match.Count -eq 0) {{
    $match = @($tokens | Where-Object {{ $_.GetDescription() -like "*Chinese*" }})
  }}
  if ($match.Count -gt 0) {{
    $voice.Voice = $match[0]
  }}
}}
$voice.Rate = {self.rate}
$voice.Volume = {self.volume}
$stream = New-Object -ComObject SAPI.SpFileStream
$format = New-Object -ComObject SAPI.SpAudioFormat
$format.Type = 22
$stream.Format = $format
$stream.Open({_ps_quote(str(output_path))}, 3, $false)
$voice.AudioOutputStream = $stream
[void]$voice.Speak($text, 0)
[void]$voice.WaitUntilDone(90000)
$stream.Close()
"""
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True,
            text=True,
            timeout=90,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Windows SAPI TTS failed: {result.stderr[-1000:]}")
