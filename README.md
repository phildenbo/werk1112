# Werk1112

<p align="center">
  <img src="docs/assets/banner_werk.png" alt="Werk1112 startup banner: WERK1112 - Any Model. Anywhere." />
</p>

Werk1112 is a headless local model server in the spirit of Ollama, built around a Rust-first toolchain and an OpenAI-compatible HTTP API. It is intended for external clients such as Open WebUI, LM Studio, and agent tooling.

The app does not provide its own GUI chat. Use the CLI to import/list/inspect models and start the server, then connect a client to the HTTP API.

## Status

This is an early V1 skeleton:

- Development builds are CPU-only by default.
- CUDA, CUDNN, Metal, and MKL are opt-in Cargo features for source builds.
- Release artifacts should bundle companion backends for their target OS; compiled GPU backends are enabled only by the target aliases that require that toolchain.
- `/v1/models` returns installed model manifests in an OpenAI-style model list.
- `/v1/chat/completions` accepts OpenAI-style chat requests.
- Streaming uses `text/event-stream` with `chat.completion.chunk` payloads and a final `data: [DONE]`.
- API streaming deltas are buffered into small text chunks instead of emitting every generated token as its own event.
- CLI chat streams decoded token-pieces by default so the answer appears progressively in the terminal.
- Local GGUF and safetensors model imports are copied into a managed model store.
- Hugging Face pulls use `git clone` for now, so install `git` and `git-lfs` for real model repos.

Current generation support includes Candle-native GGUF/safetensors paths plus external backend hooks. GGUF metadata is parsed and routed to Candle quantized loaders for supported architectures: `llama`, `qwen2`, `phi`, `phi2`, `phi3`, and `gemma3`. Safetensors HF-style directories are loaded through Candle for supported architectures: `llama`, `gemma`, `gemma2`, `qwen2`, `mistral`, and `phi3`. GGUF can also run through an external llama.cpp Vulkan backend. MLX model directories can run through an external `mlx-lm` backend.

## Format Support

| Format | Typical Use | Import/List/Inspect | Backend Status |
| --- | --- | --- | --- |
| Safetensors | Hugging Face training/fine-tuning standard | Yes | Implemented through Candle for selected architectures including Llama |
| GGUF | llama.cpp, Ollama, LM Studio, CPU inference | Yes | Implemented through Candle quantized loaders; also supported through external llama.cpp/Vulkan backend |
| PyTorch (`.pt`, `.pth`, `pytorch_model.bin`) | Training, research, checkpoints | Yes | Backend pending |
| ONNX (`.onnx`) | Framework-independent inference | Yes | ONNX Runtime backend pending |
| MLX (`.npz`, MLX-style dirs) | Apple Silicon / MLX-LM | Yes | Implemented through external `mlx-lm` backend when configured |
| TensorRT Engine (`.engine`, `.plan`) | NVIDIA-optimized inference | Yes | TensorRT backend pending |
| OpenVINO IR (`.xml` + `.bin`) | Intel CPUs, GPUs, NPUs | Yes | OpenVINO backend pending |
| TensorFlow (`.ckpt`, `.pb`) | TensorFlow ecosystem | Yes | TensorFlow backend pending |
| CoreML (`.mlmodel`, `.mlpackage`) | iOS/macOS deployment | Yes | CoreML backend pending |

## Build

The default development build is CPU-only. Release builds use target-specific Cargo aliases. Linux has a portable default build and a separate CUDA build because Candle CUDA kernels require a recent CUDA toolkit at compile time.

```bash
cargo check --locked --no-default-features
cargo build-cpu
```

Target release builds:

```bash
cargo build-windows
cargo build-linux
cargo build-linux-cuda
cargo build-macos-apple-silicon
cargo build-macos-intel
```

Run target release aliases on the matching build OS when GPU acceleration is involved. In practice:

- Run `cargo build-windows` from native Windows PowerShell with the MSVC Rust toolchain and Windows CUDA installed.
- Run `cargo build-linux` from Linux or WSL for the portable Linux artifact.
- Run `cargo build-linux-cuda` from Linux or WSL only when the active Linux CUDA toolkit is new enough for Candle.
- Run `cargo build-macos-apple-silicon` on Apple Silicon macOS.
- Run `cargo build-macos-intel` on Intel macOS.

Do not use WSL to produce the Windows CUDA artifact. WSL can build the Linux artifact.

These aliases expand to normal Cargo target builds:

```text
cargo build-windows              -> x86_64-pc-windows-msvc + release-windows
cargo build-linux                -> x86_64-unknown-linux-gnu + release-linux
cargo build-linux-cuda           -> x86_64-unknown-linux-gnu + release-linux-cuda
cargo build-macos-apple-silicon  -> aarch64-apple-darwin + release-macos-apple-silicon
cargo build-macos-intel          -> x86_64-apple-darwin + release-macos-intel
```

Cargo aliases are subcommands, so the command is `cargo build-windows`, not `cargo build windows`.

If a target build fails with `E0463` / `can't find crate for core` or many dependencies fail immediately, the Rust standard library for that target is not installed in the active toolchain. For CPU-only cross-checks you can install it with `rustup target add <target-triple>`, but CUDA/Metal release artifacts should still be built natively on their target OS.

Release backend bundles:

| Bundle | Compiled backend support | Companion backend support |
| --- | --- | --- |
| `release-windows` | CPU, CUDA | Vulkan via bundled/discoverable `llama-cli`; VLM through a capable external backend |
| `release-linux` | CPU | Vulkan via bundled/discoverable `llama-cli`; VLM through a capable external backend |
| `release-linux-cuda` | CPU, CUDA | Vulkan via bundled/discoverable `llama-cli`; VLM through a capable external backend |
| `release-macos-apple-silicon` | CPU, Metal | MLX via `mlx-lm`; VLM through a capable external backend |
| `release-macos-intel` | CPU, Metal where available | VLM through a capable external backend |

Raw Cargo equivalents:

```bash
cargo build --release --locked --target x86_64-pc-windows-msvc --features release-windows
cargo build --release --locked --target x86_64-unknown-linux-gnu --features release-linux
cargo build --release --locked --target x86_64-unknown-linux-gnu --features release-linux-cuda
cargo build --release --locked --target aarch64-apple-darwin --features release-macos-apple-silicon
cargo build --release --locked --target x86_64-apple-darwin --features release-macos-intel
```

CUDA release builds require a working NVIDIA driver and CUDA toolkit on the build machine. The portable Linux build does not compile Candle CUDA and avoids this requirement. If `cargo build-linux-cuda` fails with `fatal error: cuda_fp8.h: No such file or directory`, the active CUDA toolkit is too old or the wrong `nvcc` is first in `PATH`. Candle CUDA kernels may require headers that are not present in CUDA 11.x. Point the build at a newer installed toolkit:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_ROOT=/usr/local/cuda-13.0
export CUDA_PATH=/usr/local/cuda-13.0
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

nvcc --version
cargo build-linux-cuda
```

If the CUDA build then fails because NVML cannot query the GPU, set the compute capability manually. For example, an RTX 30xx/Ampere `sm_86` GPU uses:

```bash
export CUDA_COMPUTE_CAP=86
cargo build-linux-cuda
```

For a local install, use the same rule. `--locked` keeps the checked-in dependency graph, and `--force` replaces an existing `werk` install. For the portable CPU/Vulkan-capable binary:

```bash
cargo install --path . --locked --force
```

For a CUDA-enabled local install, make sure the newer CUDA toolkit is first:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_ROOT=/usr/local/cuda-13.0
export CUDA_PATH=/usr/local/cuda-13.0
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

cargo install --path . --locked --force --features cuda
```

Native Windows CUDA / PowerShell:

1. Install Rust for Windows with `rustup`.
2. Install Visual Studio Build Tools with the C++ build tools.
3. Install Git, Git LFS, and a Windows CUDA Toolkit new enough to provide `cuda_fp8.h`.
4. Open native Windows PowerShell, not a WSL shell.
5. Build from a Windows filesystem path such as `C:\dev\werk1112`, not from `\\wsl$\...`.

If `rustup default stable-x86_64-pc-windows-msvc` says the toolchain may not be able to run on this system, the command is being run from WSL/Linux. Close that shell and run the Windows release build from PowerShell on Windows.

If PowerShell says `rustup` was not recognized, Rust is not installed for Windows or `%USERPROFILE%\.cargo\bin` is not on `PATH`. Install Rust on Windows, reopen PowerShell, and verify `rustup --version`.

If the PowerShell prompt starts in `\\wsl.localhost\...`, move or clone the project into a Windows path before building:

```powershell
cd C:\dev
git clone <repo-url> werk1112
cd C:\dev\werk1112
```

Build the native Windows CUDA binary:

```powershell
cd C:\dev\werk1112

rustup default stable-x86_64-pc-windows-msvc
git lfs install
nvidia-smi
nvcc --version

$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$env:CUDA_ROOT = $env:CUDA_HOME
$env:CUDA_PATH = $env:CUDA_HOME
$env:CUDA_TOOLKIT_ROOT_DIR = $env:CUDA_HOME
$env:CUDA_COMPUTE_CAP = "86"
$env:Path = "$env:CUDA_HOME\bin;$env:Path"

cargo build-windows
```

The release binary is written to Cargo's target directory:

```text
target/x86_64-pc-windows-msvc/release/werk.exe
target/x86_64-unknown-linux-gnu/release/werk
target/aarch64-apple-darwin/release/werk
```

External backends must be shipped in the artifact or discoverable when selected. For example, the Vulkan backend uses a `llama-cli` binary beside `werk` or the path in `WERK_LLAMA_CLI`. The MLX backend uses `python3 -m mlx_lm.generate` or `WERK_MLX_PYTHON`. VLM request/image support is compiled into every build; actual multimodal generation depends on the chosen model and backend.

Additional low-level acceleration features are available for custom builds:

```bash
cargo build --release --locked --features cuda,cudnn
cargo build --release --locked --features mkl
```

Build features decide what Candle acceleration support is compiled into the binary. Backend selection is a separate CLI option and is the preferred way to choose how a process runs:

```bash
werk --backend auto chat gemma-2b-it
werk --backend cpu chat gemma-2b-it
werk --backend cuda chat gemma-2b-it
werk --backend metal chat gemma-2b-it
werk --backend vulkan chat TinyLLama-1B-GGUF
werk --backend mlx chat mlx-model
werk --backend cuda serve --model gemma-2b-it
```

`--backend auto` uses the target default order: Windows tries CUDA first, macOS Apple Silicon tries MLX first, and Linux tries Vulkan first. If that backend is unavailable, it falls back through the other target backends and then CPU.

`--backend cuda` and `--backend metal` use Candle devices and require binaries built with the matching Cargo features. `--backend mlx` and `--backend vulkan` are backend choices, not Candle devices. `--device` remains as a Candle-only compatibility override, but `--backend` is what end users should use.

## End-User Releases

Release artifacts should be produced with the target Cargo alias on the target platform. A complete artifact is the Cargo-built `werk` binary plus the companion backend files for that platform, such as `llama-cli` for Vulkan or an MLX environment for Apple Silicon. End users should not need Rust, Cargo, Visual Studio, or `nvcc`; those are build-machine requirements only.

Do not ship one artifact per backend. Ship one artifact per target platform that can run the portable backends for that target, then let the user choose with `--backend`. Linux CUDA is the exception: publish the CUDA artifact only when it is built on a validated CUDA toolchain.

Each target artifact should include the supported backends for that build, and users can select one explicitly with `--backend`:

| Platform | Cargo command | Included backend support | Auto default |
| --- | --- | --- | --- |
| Windows 10/11 x64 | `cargo build-windows` | CPU, CUDA, Vulkan via llama.cpp | CUDA |
| Linux x64 | `cargo build-linux` | CPU, Vulkan via llama.cpp | Vulkan |
| Linux x64 CUDA | `cargo build-linux-cuda` | CPU, CUDA, Vulkan via llama.cpp | Vulkan |
| macOS Apple Silicon | `cargo build-macos-apple-silicon` | CPU, Metal, MLX-LM, VLM request support | MLX |
| macOS Intel | `cargo build-macos-intel` | CPU, Metal where available | Metal, then CPU |

Backend selection is per process. There is no persisted setup step.

```bash
werk --backend auto chat model-id
werk --backend cuda chat model-id
werk --backend vulkan chat model-id
werk --backend mlx chat model-id
werk --backend metal chat model-id
werk --backend cpu chat model-id
```

`auto` prefers the most common backend for the target: Windows uses CUDA first, macOS Apple Silicon uses MLX first, and Linux uses Vulkan first. If the preferred backend is unavailable, `auto` falls back through the target's other supported backends and then CPU.

MLX and Metal are not the same backend. Metal is implemented through Candle. MLX is implemented as an external `mlx-lm` backend. Vulkan is implemented as an external llama.cpp backend for GGUF models. Release artifacts should bundle the external backend pieces they need. During development, `WERK_LLAMA_CLI` can point to a specific llama.cpp binary, and `WERK_MLX_PYTHON` can point to a Python environment with `mlx-lm` installed.

VLM support means multimodal model/request support, not a separate backend. VLM-capable models should be routed through a backend that supports image inputs, such as llama.cpp multimodal GGUF builds or MLX VLM backends as those integrations are wired.

CLI image inputs use repeatable `--image` flags:

```bash
werk --backend vulkan run vlm-model "Describe this image." --image ./image.png
werk --backend mlx chat vlm-model --image ./image.png
```

OpenAI-style API image inputs are accepted from `image_url` and `input_image` content parts. Text-only backends return a clear error when image inputs are provided.

## Model Store

The model store is resolved in this order:

1. `WERK_HOME`
2. `$XDG_DATA_HOME/werk1112`
3. Native Windows: `%LOCALAPPDATA%\werk1112`
4. Native Windows fallback: `%USERPROFILE%\AppData\Local\werk1112`
5. Unix fallback: `~/.local/share/werk1112`

Each imported or pulled model is stored under `models/<model-id>/` inside that store. On a default Linux setup, a pulled model named `TinyLLama-1B-GGUF` is saved here:

```text
~/.local/share/werk1112/models/TinyLLama-1B-GGUF/
├── manifest.json
└── files/
    └── ...
```

On native Windows, the same model is saved here by default:

```text
%LOCALAPPDATA%\werk1112\models\TinyLLama-1B-GGUF\
```

`manifest.json` contains source, format, architecture, tokenizer/config paths, model file path, checksums, and backend hints. `files/` contains the copied local model files or downloaded Hugging Face files.

For GGUF repositories that contain many quantizations, new imports prefer a balanced `Q4_K_M` file when it is present instead of taking the first filename alphabetically. The selected file is stored in `manifest.json` as `model_path`, and both `chat` and `serve` use that selected file.

## CLI

Install the CLI from this checkout:

```bash
cargo install --path . --locked
```

From another directory, pass the project path with `--path`:

```bash
cargo install --path ../client --locked
```

After install, use the command directly:

```bash
werk --help
werk serve --help
```

During development, you can also run the local binary without installing:

```bash
cargo run --no-default-features -- <command>
```

Start the server:

```bash
werk serve
```

`serve` starts the OpenAI-compatible API. It exposes all installed models through `/v1/models`; each API request normally chooses the model with its JSON `model` field.

Set a default model for requests that omit `model`:

```bash
werk serve --model gemma-2b-it
```

The default address is:

```text
http://127.0.0.1:11434
```

Override address:

```bash
werk serve --host 0.0.0.0 --port 11434
```

Import a local model file or directory. Files are copied into the managed model store:

```bash
werk import /path/to/model-dir --name llama-local
```

Pull from Hugging Face:

```bash
werk pull org/model-repo --name model-local
```

Pull one file from a Hugging Face repository, useful for GGUF repos that contain many quantizations:

```bash
werk pull TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --file tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --name TinyLlama-1B-GGUF
```

Pull shows live status for each phase. The first phase clones Git metadata with `GIT_LFS_SKIP_SMUDGE=1`; the second phase runs `git lfs pull` and shows `downloading` with either Git LFS percent/speed or a running local bytes/s estimate while Git LFS is quiet. After the download completes, the CLI shows an import step while files are copied into the managed model store.

List installed models:

```bash
werk list
```

Inspect a model manifest:

```bash
werk inspect llama-local
```

Switch an already-installed model to another tracked model file:

```bash
werk select-file TinyLLama-1B-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Use `werk inspect TinyLLama-1B-GGUF` to see the exact filenames under `files`. The `select-file` command accepts either `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` or `files/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`.

Use a custom store for any command:

```bash
werk --model-home /tmp/werk-store list
```

Run one prompt from the terminal:

```bash
werk run gemma-2b-it "Write one sentence about Rust." --max-tokens 64
```

Start an interactive terminal chat:

```bash
werk chat gemma-2b-it --max-tokens 128
```

`--max-tokens` is a hard cap on generated completion tokens. If you set `--max-tokens 32`, the model may stop mid-sentence because the decoder reached the limit, not because the answer is complete. Use a larger value such as `--max-tokens 64` or `--max-tokens 128` for normal chat.

Terminal chat prints decoded token-pieces as soon as the backend produces them, so text appears progressively after `assistant>`. To reduce terminal flushes, switch back to chunked output:

```bash
werk chat gemma-2b-it --stream-granularity chunk
```

Timing and throughput stats are quiet by default. Add `--verbose` to `run` or `chat` for Ollama-style stats:

```bash
werk chat TinyLlama-1B-GGUF --max-tokens 128 --verbose
```

Example verbose output:

```text
total duration:       461.318ms
load duration:        139.4804ms
prompt eval count:    41 token(s)
prompt eval duration: 43.805ms
prompt eval rate:     935.97 tokens/s
eval count:           21 token(s)
eval duration:        241.897ms
eval rate:            86.81 tokens/s
```

`prompt eval` is prompt/prefill time. `eval` is assistant-token decode time. `total` also includes model load and tokenizer overhead for that turn. For TinyLlama GGUF on a CUDA build, use `Q4_K_M` as the default balance of speed and quality; `Q2_K` is smaller but noticeably worse, and larger quants can be slower.

The CLI chat is only a terminal workflow. The project still does not ship a GUI; external tools should use the HTTP API.

## OpenAI-Compatible API

Configure compatible clients with this base URL:

```text
http://127.0.0.1:11434/v1
```

List models:

```bash
curl http://127.0.0.1:11434/v1/models
```

Non-streaming chat completion:

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama-local",
    "messages": [
      {"role": "user", "content": "Write one sentence about Rust."}
    ],
    "temperature": 0.7,
    "max_completion_tokens": 32
  }'
```

Non-streaming calls do not print anything until the full completion is finished. For large models on CPU, prefer streaming while testing.

Streaming chat completion:

```bash
curl -N http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama-local",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Write one sentence about Rust."}
    ],
    "max_completion_tokens": 32
  }'
```

The stream sends chunks like:

```text
data: {"object":"chat.completion.chunk",...,"delta":{"role":"assistant"}}
data: {"object":"chat.completion.chunk",...,"delta":{"content":"Rust is a systems"}}
data: {"object":"chat.completion.chunk",...,"delta":{"content":" programming language..."}}
data: {"object":"chat.completion.chunk",...,"finish_reason":"stop"}
data: [DONE]
```

Text deltas are intentionally chunked. They are not one event per token.

## Next Work

- Extend safetensors execution coverage beyond `gemma`, `gemma2`, `qwen2`, `mistral`, and `phi3`.
- Add richer chat template support from tokenizer/model metadata.
- Add more GGUF architectures as Candle support allows.
- Add backends for ONNX Runtime, MLX, TensorRT, OpenVINO, TensorFlow, CoreML, and direct PyTorch checkpoint execution/conversion.
- Add optional llama.cpp backend behind the existing backend trait if broad GGUF compatibility becomes more important than Rust-only execution.
- Add embeddings and tool-call response support after the chat/models baseline is stable.
