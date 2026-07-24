# Werk1112 media inference

Werk1112 is an inference router, not a workflow engine. A request names one
model and one concrete task. Werk normalizes its inputs, resolves defaults,
validates parameters, estimates the workload, scores runtime candidates,
executes the best accepted runtime (with policy-controlled runtime retry), and
persists outputs and metadata.

## Public commands

```text
werk chat MODEL
werk image generate|edit|upscale MODEL
werk video generate|animate|transform|upscale MODEL
werk audio generate|speak|transcribe|separate MODEL
werk serve
```

The old `werk run` parser remains hidden for compatibility. New applications
should use `chat` for text and the typed media commands for generated files.

Generative prompt priority is:

1. `--prompt`, `--text`, or `--lyrics`;
2. its corresponding `--*-file`;
3. piped standard input;
4. interactive terminal input.

## Canonical tasks

The manifest and API use typed tasks rather than a matrix of booleans:

- text generation and embedding, image understanding;
- image generation, editing, variation, inpainting, outpainting, and upscaling;
- video generation, image-to-video, video-to-video, inpainting, extension,
  upscaling, and frame interpolation;
- audio and music generation, song continuation/variation, TTS, ASR, voice
  conversion, stem generation/separation, and audio enhancement.

One model can declare several tasks. Input modalities (`text`, `image`,
`video`, `audio`) and output modalities (`text`, `image`, `video`, `audio`,
`embedding`) are recorded separately.

## Manifest schema v2

Schema-v2 fields are flattened into the existing manifest JSON:

- `schema_version`, `family`, `architecture`, and `repository_layout`;
- `tasks`, `input_modalities`, and `output_modalities`;
- first-class components and their files, format, precision, and quantization;
- detected generation defaults and parameter constraints;
- compatible runtime hints and optimized artifacts.

Old manifests deserialize with schema version 1 and are enriched from the
installed repository in memory. Existing identity, source, file inventory, and
selected model path remain intact.

Supported layouts are `single_file`, `gguf`, `transformers`, `diffusers`,
`mlx`, `onnx_bundle`, `tensorrt_engine`, and `custom`. Diffusers detection uses
`model_index.json` and well-known component roots:

```text
transformer  unet  vae  scheduler  text_encoder  text_encoder_2
tokenizer    tokenizer_2  encoder  decoder  vocoder
feature_extractor  controlnet  adapter
```

Components remain part of one installed model.

## Effective parameters and provenance

Transport requests contain overrides. The resolver produces an effective value
for every parameter descriptor:

```text
system → task → family → model → runtime → hardware/quality
       → saved profile → request → backend adjustment
```

Every effective value records the winning source. Booleans have inherited,
explicitly enabled, and explicitly disabled states. Internal list overrides
distinguish inherit, replace, add, and clear.

`werk parameters MODEL --backend auto --json` and
`GET /v1/parameters?task=TASK&model=MODEL&backend=auto`
return descriptors with path, CLI flag, type, label, category, default, range,
allowed values, repeatability, advanced status, and memory/quality/runtime
impact. With a model, manifest defaults/constraints and per-runtime parameter
support are included.

Backends report parameters as `native`, `translated`, `emulated`, `ignored`,
`unsupported`, or `model_dependent`. Explicit ignored/unsupported values fail
under `strict` (the API default), warn under `warn`, and continue under
`permissive`.

## Estimate and planning

Media estimates distinguish:

- download and weight payload;
- accelerator and host peak;
- output size;
- fit (`fits`, `tight`, `likely_oom`, or `unknown`);
- confidence (`exact`, `backend_measured`, `architecture_model`, `heuristic`,
  or `unknown`);
- assumptions, warnings, and recommendations.

Image estimates scale with pixels, batches/count, VAE behavior, and offload.
Video estimates additionally scale with frames and temporal windowing. Audio
estimates scale with duration, sample rate, channels, variations, and stems.

The scored planner checks model task, runtime task, repository layout,
family/architecture probe, runtime availability, accelerator, explicitly set
parameters, and workload fit. It distinguishes:

- backend fallback: the same model through another runtime;
- execution degradation: offload, tiling/windowing, or a slower attention path;
- model/quality downgrade: a recommendation that is never silently executed.

## Media companion

Rust remains the control and routing plane. The included Python companion uses
a versioned single-request JSON process protocol with:

```text
health  capabilities  probe-model  estimate  execute
```

It performs lazy, local-only Diffusers/Transformers execution. The companion
sets Hugging Face offline variables and passes `local_files_only=True`; it
never installs a package or downloads model weights. `WERK_MEDIA_COMPANION`
can point to a compatible executable, while `WERK_MEDIA_PYTHON` chooses the
Python interpreter for the included adapter. `WERK_MEDIA_ACCELERATOR` can
explicitly select `cuda`, `rocm`, `mps`/`metal`, or `cpu`. MLX media models
remain catalogable but have no executable adapter in this release.

`werk doctor` reports the protocol and optional dependencies. Missing
Diffusers, Transformers, Pillow, audio/video codecs, or accelerator packages
only disables affected tasks.

### Execution support

| Task group | Catalog / inspect / estimate | Companion execution |
| --- | --- | --- |
| Image generation/edit/inpaint/upscale | Yes | Diffusers pipeline/model dependent |
| Video generation/animate/transform/upscale | Yes | Diffusers plus image/video codec dependencies |
| Audio/music generation | Yes | Diffusers or Transformers pipeline/model dependent |
| Song continuation/variation | Yes | Prepared; no generic adapter yet |
| Text-to-speech | Yes | Transformers TTS pipeline/model dependent |
| Speech-to-text/translation | Yes | Transformers ASR pipeline/model dependent |
| Voice conversion | Yes | Prepared; no generic adapter yet |
| Stem generation/separation | Yes | Prepared; no generic adapter yet |
| Audio enhancement | Yes | Prepared; no generic adapter yet |

Parameters not accepted by a concrete pipeline are reported rather than
silently discarded.

Direct companion output formats are `png`/`jpeg`/`webp` for images,
`mp4`/`gif` for video, `wav`/`flac`/`ogg` for generated audio and TTS, and
`json`/`text`/`srt`/`vtt`/`tsv` for ASR. Codec libraries required by the
selected format must already be installed.

## HTTP API and jobs

Direct endpoints:

```text
POST /v1/chat/completions
POST /v1/images/generations
POST /v1/images/edits
POST /v1/videos/generations
POST /v1/audio/generations
POST /v1/audio/speech
POST /v1/audio/transcriptions
GET  /v1/capabilities
GET  /v1/parameters
GET  /v1/outputs/{id}
```

Long-running jobs:

```text
POST   /v1/jobs
GET    /v1/jobs/{id}
DELETE /v1/jobs/{id}
```

Persisted states are `queued`, `loading`, `running`, `encoding`, `completed`,
`failed`, and `cancelled`. `/v1/audio/speech` returns audio bytes directly;
other synchronous media endpoints return Werk metadata and authenticated
`/v1/outputs/{id}` URLs.

The shared conversation content model can represent text, image, video, audio,
tool calls, and tool results. Available media tools are derived dynamically
from installed models and successful runtime probes. Automatic LLM tool-call
orchestration is intentionally left to Station or another client; Serve never
starts a CLI subprocess.

## Storage and limits

```text
WERK_HOME/
├── models/
├── artifacts/
├── outputs/
└── jobs/
```

Each output metadata record includes ID, task, model, runtime, path, MIME type,
size, dimensions/duration where applicable, seed, effective parameters, and
creation time. Retention only targets `outputs/`, never models. Defaults are 30
days and 20 GiB; use `WERK_OUTPUT_RETENTION_SECONDS` and
`WERK_OUTPUT_MAX_BYTES` to override them.

## Current limitations

- `probe-model` checks local repository metadata, task hints, and dependencies.
  The concrete Diffusers/Transformers pipeline is loaded only during
  execution, so model-specific incompatibility can still surface then.
- JSON local-path and inline-base64 inputs are supported. The offline companion
  does not fetch remote HTTP(S) URLs. OpenAI multipart upload compatibility is
  not implemented yet.
- Persisted job cancellation is cooperative. A native third-party call that
  lacks cancellation may release resources only when it returns.
- The companion currently returns one terminal response rather than granular
  progress events. Persisted phases are available, but `encoding` can be too
  brief to observe for fast jobs.
- Werk serves authenticated whole-file outputs. HTTP byte ranges and
  object-storage export are future work.
- Generic adapters for voice conversion, stems, and enhancement are described
  by the contract but not executable yet.
- The generic Transformers TTS path uses the model's native voice and sample
  rate; explicit voice, speed, pitch, and output resampling remain
  model-specific and are reported as unsupported by this adapter.
