# Implementation Plan

## Product Direction

Build a local-first video translation tool in this order:

1. Reliable offline bilingual subtitles.
2. Higher-quality contextual translation with duration budgets.
3. Browser quasi-realtime subtitles.
4. High-quality offline dubbing with voice cloning and duration control.

Do not optimize the current SAPI/NLLB path as the product path. SAPI is a debug
fallback; NLLB is a fallback translator.

## Current State

Implemented:

- CLI: `aivt process <path>`, `aivt doctor`, `aivt serve`.
- API: jobs, uploads, output downloads, job logs, runtime metadata.
- GUI: local upload/path submission, queue, preview, downloads.
- Outputs: `audio.wav`, `source.srt`, `zh.srt`, `bilingual.vtt`,
  `translated.mkv`, `dubbed.wav`, `dubbed.mkv`, plus audio-lane artifacts.
- ASR backend: `mock`, `faster-whisper`.
- Alignment backend: `off`, English-only `whisperx` optional path.
- Translation backend: `mock`, local Qwen, NLLB.
- Local Qwen and OpenAI-compatible LLM translation backends with context,
  glossary, duration budget, JSON candidates, and duration-aware candidate
  selection.
- TTS backend: `mock`, Windows SAPI debug fallback, OpenAI-compatible speech
  endpoint for VoxCPM2-style reference-audio services.
- ffmpeg fallback through `imageio-ffmpeg`.
- Job-level JSON line logs.
- Fixed local worker queue with default `AIVT_JOB_WORKER_COUNT=1`.
- Optional VAD layer with `off` and `silero` backends.
- Audio separation backends: `off` scaffold and optional `demucs`.
- Cue schema prework: speaker id, source words, confidence, duration budget,
  duration tolerance ratio, duration tolerance seconds.
- Job artifacts store config snapshots and model-version metadata.
- Output downloads are restricted to a whitelist and job-local files.
- Subtitle quality checks include translated reading-speed warnings.
- Golden SRT fixtures lock deterministic subtitle formatting for regression.

Known product limits:

- SAPI sounds mechanical and cannot clone source speakers.
- VoxCPM2-style TTS endpoint is wired, but the external service is not bundled.
- NLLB lacks course context, glossary control, and duration-budget translation.
- Current dubbed output replaces original audio and does not remove source speech.
- Cue schema has speaker fields, but diarization is not implemented yet.

## V1.0: Offline Bilingual Subtitles

Goal: local video/audio import to bilingual subtitle artifacts.

Status: mostly implemented.

Remaining hardening:

- Real ASR golden fixtures with timestamp tolerance.
- WER smoke fixture for selected real ASR model.

## V1.1: Alignment, Safety, And Audio-Track Prework

Goal: make the subtitle and artifact pipeline stable enough for future dubbing.

Required work:

- Enable Silero VAD in real smoke runs and tune thresholds.
- WhisperX forced alignment for English-source videos is wired as an optional
  backend; real smoke and threshold tuning remain.
- For Chinese or mixed-language source, fall back to Whisper word timestamps or
  MFA rather than forcing the default WhisperX wav2vec2 path.
- Extend cue schema now. Implemented fields:
  - `speaker_id`
  - `source_words`
  - `confidence`
  - `duration_budget`
  - `duration_tolerance`
- Preserve two audio lanes in artifacts:
  - original audio / source speech retained,
  - speech-removed background lane for future dubbing.
  Implemented: `off` creates a vocals copy and silent background placeholder;
  `demucs` can produce real vocals/background artifacts when installed.
- Tune subtitle reading-speed limits and Chinese line-breaking rules against real
  course videos.
- Keep local API bound to `127.0.0.1` by default; document that `0.0.0.0`
  exposes uploaded files to the LAN.

## V1.5: LLM Translation And Duration Budget

Goal: make translation quality good enough before investing in dubbing.

This must happen before V3 TTS work. Translation quality is the ceiling for both
subtitles and dubbed audio.

Planned backend:

- Implemented integration paths: local Qwen through Transformers and
  OpenAI-compatible `/v1/chat/completions`.
- Target model: Qwen2.5/3 or DeepSeek distill local backend, preferably
  OpenVINO INT4 or a local server exposing an OpenAI-compatible API.
- Keep NLLB as a fallback.

Prompt contract:

- Input: source text, neighboring context, glossary, source duration.
- Output: 3 Chinese candidates.
- Each candidate must target roughly 5-7 Chinese characters per second.
- Preserve terms through the prompt instead of post-hoc string replacement.
- Return structured metadata:
  - candidate text,
  - estimated speech length,
  - preserved terms,
  - uncertainty notes.

Selection:

- Prefer semantic accuracy first.
- Missing required glossary terms are treated as a higher-priority failure than
  duration mismatch.
- Among equivalent translations, choose the candidate closest to the duration
  budget.
- Log a warning when the selected candidate is outside the 5-7 chars/second
  budget or misses a relevant glossary term.
- Do not repeatedly retranslate just to satisfy timing; that loses information.

Remaining hardening:

- Add stricter schema validation for candidate metadata beyond the required
  `text` field.
- Add quality fixtures comparing NLLB fallback against LLM output.

## V2.0: Browser Extension Fast Path

Goal: translate web playback when source subtitles are already available.

Design:

- Browser extension probes native subtitle/caption tracks first.
- If source captions exist, use caption -> translation path.
- Use local `127.0.0.1` WebSocket to the service, not native messaging.
- Clearly document DRM limits: `chrome.tabCapture` cannot capture many DRM
  protected streams such as Netflix, Disney+, some paid course platforms, and
  some member-only video flows.

## V2.1: Browser ASR Realtime Path

Goal: quasi-realtime subtitles when no source subtitles exist.

Design:

- Capture tab audio when allowed.
- Stream PCM chunks over local WebSocket.
- Use WhisperLive/OpenVINO or SimulStreaming-style local agreement.
- Stable output target: 5-10 seconds delay. A 3 second lower bound causes too
  much subtitle flicker for Chinese output.

## V3.0: TTS M1, VoxCPM2 Primary Dubbing

Goal: first product-grade offline Chinese dubbing.

Primary backend: VoxCPM2 via OpenAI-compatible HTTP endpoint.

Integration status:

- Client backend is implemented as `AIVT_TTS_BACKEND=voxcpm2`.
- Each cue sends translated text, source reference audio, source transcript,
  target duration, and style hint to `/audio/speech`.
- Remaining work is packaging or documenting the external VoxCPM2 service.

Why:

- Tokenizer-free architecture is a good fit for English/Chinese mixed technical
  course text.
- It supports 30 languages without language tags.
- It supports controllable voice cloning and ultimate cloning with reference
  audio plus transcript.
- It outputs 48kHz audio.
- It can be served through vLLM-Omni at `/v1/audio/speech`, keeping PyTorch and
  CUDA out of the Windows GUI process.
- Apache-2.0 license is compatible with commercial product work.

Deployment:

- Local development: WSL2 + CUDA service on `127.0.0.1:8000`.
- Windows app: call the local or remote HTTP endpoint.
- CI: do not require VoxCPM2; keep mock/SAPI smoke tests.

Sources:

- https://github.com/OpenBMB/VoxCPM
- https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/online_serving/voxcpm2/

## V3.1: TTS M2, IndexTTS-2 Duration Fallback

Goal: handle cues where duration tolerance is tight.

Backend: IndexTTS-2.

Why:

- Designed around precise duration control for video dubbing.
- Supports explicit duration-controlled generation and natural-duration mode.
- Separates speaker identity from emotion/style prompt.

Routing:

- Default cues use VoxCPM2.
- Tight cues use IndexTTS-2 when `duration_tolerance < 0.05`.
- If VoxCPM2 output is outside the tolerance window after retry, route the cue
  to IndexTTS-2.

Source:

- https://github.com/index-tts/index-tts
- https://arxiv.org/abs/2506.21619

## V3.2: Speaker Diarization

Goal: avoid one cloned voice reading multiple people.

Required:

- pyannote diarization or equivalent local speaker segmentation.
- Cue schema already includes `speaker_id` from V1.1.
- Extract reference clips per speaker.
- Generate one reference voice per speaker.

## V3.3: VoxCPM2 LoRA Course Voice

Goal: course-specific voice quality.

Flow:

- Use first 5-10 minutes of clean lecturer audio.
- Build VoxCPM2 LoRA manifest with exact transcript.
- Train or import course LoRA.
- Store LoRA metadata and hash in job artifacts.

Important resource note:

- Inference can fit smaller machines, but LoRA training for VoxCPM2 may need
  substantially more VRAM than casual users have. Treat LoRA as pro/offline
  mode, not baseline.

## V3.4: Voice Conversion And Emotion Control

Goal: optional post-processing for speaker similarity or expressiveness.

Candidate:

- OpenVoice as a tone-color/style converter when the main TTS voice is not close
  enough.

## Reference TTS Interface

```python
class ReferenceTtsBackend(Protocol):
    name: str
    supports_hard_duration: bool

    def synthesize(
        self,
        target_text: str,
        target_duration: float,
        reference_audio: Path,
        reference_text: str | None,
        style_hint: str | None,
        lora_path: Path | None,
    ) -> SynthesizeResult:
        ...


@dataclass
class SynthesizeResult:
    audio_path: Path
    measured_duration: float
    duration_strategy: Literal["native", "soft_prompt", "post_atempo"]
```

## Duration Fitting Policy

Push duration budget into translation first. TTS retries are not the primary
solution.

Tolerance:

- Accept when error <= `min(0.08 * duration, 0.4)` seconds.
- For strict lip-sync cues, use `duration_tolerance < 0.05`.

Handling:

- <= 5% or <= 200 ms: accept.
- 5-8%: bounded, inaudible `atempo`.
- 8-15%: bounded `atempo` and mark cue warning.
- 15-25%: try another LLM candidate; if still poor, route to IndexTTS-2.
- > 25%: split, merge adjacent cues, or mark warning in job log.

`atempo` safety:

- Preferred range: 0.85-1.15.
- Hard range: 0.75-1.25.
- Anything outside hard range is a quality failure, not a normal fallback.

## Background Audio Policy

Do not rely on simple ducking for course content. If original audio is mostly
lecturer speech, ducking produces two voices fighting each other.

Correct path:

1. Use Demucs `htdemucs_ft` or MDX-Net to separate vocals/accompaniment.
2. Remove or heavily reduce original vocals.
3. Mix Chinese dubbing over the accompaniment/background lane.
4. Keep both output variants:
   - Chinese dub + background,
   - Chinese dub only.

## Logging Policy

Every job must log:

- `job_id`
- stage
- source path or output path when relevant
- stage duration in milliseconds
- full exception text on failure
- config snapshot
- model names and versions
- model weight hash when available
- ffmpeg version
- random seed

Logs are JSON lines so bug reports can attach one job log without needing the
entire app state.

## Test Policy

Every pipeline-facing change should run:

```powershell
.\.venv\Scripts\python -m pytest -q
```

Additional required suites:

- Golden SRT comparison: cue count and text strict, timestamps within +/-50 ms.
- Dubbing duration regression: fixed 30 second fixture, each cue within policy.
- Optional LLM endpoint smoke: set `AIVT_SMOKE_LLM=1` and point
  `AIVT_TRANSLATOR_API_BASE` at the local OpenAI-compatible server.
- Optional Silero VAD smoke: set `AIVT_SMOKE_VAD_AUDIO` to a real speech WAV.
- Optional WhisperX alignment smoke: set `AIVT_SMOKE_ALIGN_AUDIO` and
  `AIVT_SMOKE_ALIGN_TEXT` for a real English speech WAV.
- ASR WER smoke test: fixed known English fixture, WER target under 5% for the
  selected real model.
- API security test: unknown output names and path traversal attempts return
  404.

## Performance Targets

Initial targets on RTX 4060 / 16 GB RAM:

- `tiny/base` ASR subtitle mode: <= 0.3x realtime.
- `small` ASR subtitle mode: <= 0.6x realtime.
- Offline high-quality dubbed video: allowed to be slower than realtime until
  quality reaches product threshold.
