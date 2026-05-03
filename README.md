# AI Voice Translater

Local-first prototype for turning videos into translated subtitles and Chinese dubbed videos.

## Current Scope

This repository starts with the v1 offline pipeline:

1. Import a local video file.
2. Extract a 16 kHz mono WAV with `ffmpeg`.
3. Run a pluggable ASR backend.
4. Run a pluggable translation backend.
5. Write `source.srt`, `zh.srt`, and `bilingual.vtt`.
6. Prepare audio-lane artifacts for future vocal/background separation.
7. Synthesize translated speech with a pluggable TTS backend and write `dubbed.wav`.
8. For video inputs, create `translated.mkv` with the original audio/video plus a Chinese subtitle track.
9. For video inputs, create `dubbed.mkv` with the original video, Chinese dubbed audio, and Chinese subtitles.
10. Keep app logs and per-job logs for debugging.

The default ASR and translator are mock implementations so the API, job model,
logging, and subtitle output can be tested before installing large ML models.
`faster-whisper` is already wired as the first real ASR backend.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
```

For real ASR:

```powershell
.\.venv\Scripts\python -m pip install -e .[asr]
```

System `ffmpeg` is optional. The app first checks `PATH`, then falls back to the
`imageio-ffmpeg` bundled binary. You can still override with `AIVT_FFMPEG_PATH`.

## CLI

```powershell
aivt process C:\path\to\video.mp4
aivt serve --host 127.0.0.1 --port 8765
aivt doctor
aivt smoke-llm --text "The encoder maps input embeddings." --duration 3
aivt smoke-vad C:\path\to\speech.wav
```

## GUI

Start the local service:

```powershell
.\.venv\Scripts\aivt.exe serve --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

The GUI supports uploading video/audio files, submitting an existing local path,
watching task status, previewing subtitles/logs, and downloading generated
artifacts. For video inputs, the download list includes subtitle-track video,
Chinese dubbed audio, Chinese dubbed video, and audio-lane artifacts. With the
default `AIVT_AUDIO_SEPARATION_BACKEND=off`, the background lane is a silent
placeholder and the vocals lane is the original extracted speech. Real
speech-removed background audio is available through the optional Demucs backend
when `demucs` is installed and `AIVT_AUDIO_SEPARATION_BACKEND=demucs`.

Use real ASR on CPU:

```powershell
$env:AIVT_ASR_BACKEND="faster-whisper"
$env:AIVT_ASR_MODEL_SIZE="tiny.en"
$env:AIVT_ASR_DEVICE="cpu"
$env:AIVT_ASR_COMPUTE_TYPE="int8"
aivt process C:\path\to\speech.wav
```

Use real ASR plus NLLB translation:

```powershell
$env:AIVT_ASR_BACKEND="faster-whisper"
$env:AIVT_ASR_MODEL_SIZE="tiny.en"
$env:AIVT_TRANSLATOR_BACKEND="nllb"
$env:AIVT_TRANSLATOR_MODEL="facebook/nllb-200-distilled-600M"
aivt process C:\path\to\speech.wav
```

The first NLLB run downloads a large model from Hugging Face and is much slower
than later cached runs.

Use an OpenAI-compatible local LLM endpoint for contextual translation:

```powershell
$env:AIVT_ASR_BACKEND="faster-whisper"
$env:AIVT_ASR_MODEL_SIZE="tiny.en"
$env:AIVT_TRANSLATOR_BACKEND="llm"
$env:AIVT_TRANSLATOR_MODEL="qwen2.5-7b-instruct"
$env:AIVT_TRANSLATOR_API_BASE="http://127.0.0.1:8000/v1"
$env:AIVT_TRANSLATOR_API_KEY=""
$env:AIVT_GLOSSARY_PATH="C:\path\to\glossary.txt"
aivt process C:\path\to\speech.wav
```

The LLM backend sends each cue with neighboring context, glossary terms, and a
duration budget. It asks for multiple Chinese candidates and selects the one
closest to the target speech length while preferring candidates that preserve
required glossary terms. The endpoint must return JSON with a `candidates` list,
for example:

```json
{
  "candidates": [
    {"text": "编码器会映射输入嵌入。", "notes": "preserved encoder term"}
  ]
}
```

Malformed responses fail the job instead of being silently accepted.

Before running a full video job against a local LLM, smoke test the endpoint:

```powershell
$env:AIVT_TRANSLATOR_BACKEND="llm"
$env:AIVT_TRANSLATOR_MODEL="qwen2.5-7b-instruct"
$env:AIVT_TRANSLATOR_API_BASE="http://127.0.0.1:8000/v1"
.\.venv\Scripts\aivt.exe smoke-llm --text "The encoder maps input embeddings." --duration 3
```

To smoke test Silero VAD on a real speech WAV:

```powershell
$env:AIVT_VAD_BACKEND="silero"
.\.venv\Scripts\aivt.exe smoke-vad C:\path\to\speech.wav
```

Use optional Demucs vocal/background separation:

```powershell
.\.venv\Scripts\python -m pip install demucs
$env:AIVT_AUDIO_SEPARATION_BACKEND="demucs"
$env:AIVT_AUDIO_SEPARATION_MODEL="htdemucs_ft"
$env:AIVT_AUDIO_SEPARATION_DEVICE="cpu"
.\.venv\Scripts\aivt.exe process C:\path\to\video.mp4
```

Demucs is intentionally not a default dependency because it is large and can be
slow on CPU. The pipeline keeps the same outputs either way:
`original_audio`, `vocals_audio`, and `background_audio`.

WhisperX alignment is wired as an optional English-only path. It is off by
default because Chinese/mixed input needs a different alignment strategy.

On Windows, the current SAPI TTS backend is only a debug fallback. It proves the
audio timeline and video muxing path, but it is not product-quality dubbing.
Product dubbing will use a reference-audio TTS route: VoxCPM2 as the primary
backend, IndexTTS-2 as the hard-duration fallback, and optional OpenVoice-style
conversion when speaker similarity needs post-processing.

Note: `dubbed.mkv` currently replaces the original audio with the Chinese dubbed
track. With the debug SAPI backend it will sound mechanical. The planned product
path is cross-lingual voice cloning, duration-budget translation, per-segment
duration fitting, and vocal/background separation before final mixing.

## API

Start the service:

```powershell
aivt serve
```

Create a job:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/jobs `
  -ContentType application/json `
  -Body '{"video_path":"C:\\path\\to\\video.mp4"}'
```

Inspect a job:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/jobs/<job_id>
```

## Environment

Copy `.env.example` or set variables directly:

- `AIVT_DATA_DIR`: output directory, defaults to `./data`
- `AIVT_LOG_LEVEL`: defaults to `INFO`
- `AIVT_JOB_WORKER_COUNT`: background job worker count, defaults to `1`
- `AIVT_ASR_BACKEND`: `mock` or `faster-whisper`; future `whisperx`
- `AIVT_ASR_MODEL_SIZE`: defaults to `small.en`; use `tiny.en` for smoke tests
- `AIVT_ASR_DEVICE`: defaults to `cpu`
- `AIVT_ASR_COMPUTE_TYPE`: defaults to `int8`
- `AIVT_ALIGNMENT_BACKEND`: `off` or `whisperx`; defaults to `off`
- `AIVT_ALIGNMENT_LANGUAGE`: alignment language, defaults to `en`
- `AIVT_ALIGNMENT_DEVICE`: alignment device, defaults to `cpu`
- `AIVT_VAD_BACKEND`: `off` or `silero`; defaults to `off`
- `AIVT_VAD_THRESHOLD`: Silero speech threshold, defaults to `0.5`
- `AIVT_VAD_MIN_SPEECH_MS`: minimum speech duration, defaults to `250`
- `AIVT_VAD_MIN_SILENCE_MS`: minimum silence duration, defaults to `100`
- `AIVT_AUDIO_SEPARATION_BACKEND`: `off` for placeholder lanes or `demucs`
- `AIVT_AUDIO_SEPARATION_MODEL`: Demucs model name, defaults to `htdemucs_ft`
- `AIVT_AUDIO_SEPARATION_DEVICE`: Demucs device, defaults to `cpu`
- `AIVT_TRANSLATOR_BACKEND`: `mock`, `nllb`, or `llm`
- `AIVT_TRANSLATOR_MODEL`: defaults to `facebook/nllb-200-distilled-600M`
- `AIVT_TRANSLATOR_API_BASE`: OpenAI-compatible `/v1` base URL for `llm`
- `AIVT_TRANSLATOR_API_KEY`: optional API key for `llm`
- `AIVT_TRANSLATOR_TIMEOUT_SECONDS`: request timeout for `llm`, defaults to `120`
- `AIVT_TRANSLATOR_CANDIDATE_COUNT`: candidate count requested from `llm`, defaults to `3`
- `AIVT_GLOSSARY_PATH`: optional glossary file; supports `source=target`, TSV, or CSV lines
- `AIVT_TTS_BACKEND`: `sapi`, `mock`, or `off`; `sapi` is debug-only
- `AIVT_TTS_VOICE`: voice name hint, defaults to `Chinese`
- `AIVT_TTS_RATE`: Windows SAPI speech rate, defaults to `0`
- `AIVT_TTS_VOLUME`: Windows SAPI speech volume, defaults to `100`
- `AIVT_TRANSLATION_REPLACEMENTS`: semicolon-separated post-processing replacements,
  using `source=target;source2=target2`
- `AIVT_SUBTITLE_SOURCE_MAX_CHARS`: source subtitle line width, defaults to `48`
- `AIVT_SUBTITLE_TARGET_MAX_CHARS`: translated subtitle line width, defaults to `22`
- `AIVT_SUBTITLE_TARGET_MAX_CPS`: translated subtitle reading-speed warning threshold,
  defaults to `7.0` chars/second
- `AIVT_SOURCE_LANG`: defaults to `eng_Latn`
- `AIVT_TARGET_LANG`: defaults to `zho_Hans`
- `AIVT_FFMPEG_PATH`: defaults to `ffmpeg`

## Tests

```powershell
.\.venv\Scripts\python -m pip install -e .[dev]
.\.venv\Scripts\python -m pytest -q
```

The test suite includes golden SRT fixtures and subtitle reading-speed checks so
subtitle formatting changes are caught explicitly.

Optional real-model smoke tests are skipped by default:

```powershell
$env:AIVT_SMOKE_LLM="1"
$env:AIVT_TRANSLATOR_API_BASE="http://127.0.0.1:8000/v1"
$env:AIVT_TRANSLATOR_MODEL="qwen2.5-7b-instruct"
.\.venv\Scripts\python -m pytest tests\test_real_smoke_optional.py -q

$env:AIVT_SMOKE_VAD_AUDIO="C:\path\to\speech.wav"
.\.venv\Scripts\python -m pytest tests\test_real_smoke_optional.py -q

$env:AIVT_SMOKE_ALIGN_AUDIO="C:\path\to\english.wav"
$env:AIVT_SMOKE_ALIGN_TEXT="The encoder maps input embeddings."
.\.venv\Scripts\python -m pytest tests\test_real_smoke_optional.py -q
```

## Next Technical Milestones

- V1.1: add Silero VAD, English-only WhisperX alignment path, output safety,
  richer cue schema, and future two-lane audio artifacts.
- V1.5: harden local LLM translation quality, add better structured validation,
  and keep NLLB only as fallback.
- V2.0: browser extension fast path using existing subtitle tracks and local
  WebSocket, with DRM capture limits documented.
- V2.1: realtime ASR path with stable 5-10 second output, not a fragile 3 second target.
- V3.0: integrate VoxCPM2 through an OpenAI-compatible HTTP endpoint for cloned
  TTS.
- V3.1: integrate IndexTTS-2 as the hard-duration fallback for tight cues.
- V3.2+: add diarization, per-speaker reference voices, VoxCPM2 LoRA, and
  optional voice conversion.
- Add task queue cancellation and progress details.
- Add browser extension/native messaging once local offline processing is stable.
