$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not $env:AIVT_TRANSLATOR_MODEL) {
    $env:AIVT_TRANSLATOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
}

$env:AIVT_TRANSLATOR_BACKEND = "qwen"
$env:AIVT_TRANSLATOR_DEVICE = "cpu"
$env:AIVT_TRANSLATOR_CANDIDATE_COUNT = "3"
$env:AIVT_TRANSLATOR_MAX_NEW_TOKENS = "180"
$env:AIVT_ASR_BACKEND = if ($env:AIVT_ASR_BACKEND) { $env:AIVT_ASR_BACKEND } else { "faster-whisper" }
$env:AIVT_ASR_MODEL_SIZE = if ($env:AIVT_ASR_MODEL_SIZE) { $env:AIVT_ASR_MODEL_SIZE } else { "tiny.en" }
$env:AIVT_ASR_DEVICE = "cpu"
$env:AIVT_ASR_COMPUTE_TYPE = "int8"

& ".\.venv\Scripts\python.exe" -m aivoice.cli serve --host 127.0.0.1 --port 8765
