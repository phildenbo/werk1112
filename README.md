# Werk1112

<p align="center">
  <img src="docs/assets/banner_werk.png" alt="Werk1112 startup banner: WERK1112 - Any Model. Anywhere." />
</p>

Werk1112 is a headless local model server in the spirit of Ollama, built around a Rust-first toolchain and an OpenAI-compatible HTTP API. It is intended for external clients such as Open WebUI, LM Studio, and agent tooling.

The app does not provide its own GUI chat. Use the CLI to import/list/inspect models and start the server, then connect a client to the HTTP API.

## Status

This is an early V1 skeleton:

- Default local installs build the recommended runtime stack for the current Windows/Linux CUDA-first workflow.
- CUDA, CUDNN, Metal, and MKL are opt-in Cargo features for source builds.
- Release artifacts are one binary per target OS; the default Windows/Linux artifacts are CUDA-first because the current Rust llama.cpp binding cannot compile CUDA and Vulkan into the same binary.
- `/v1/models` returns installed model manifests in an OpenAI-style model list.
- `/v1/chat/completions` accepts OpenAI-style chat requests.
- Streaming uses `text/event-stream` with `chat.completion.chunk` payloads and a final `data: [DONE]`.
- API streaming deltas are buffered into small text chunks instead of emitting every generated token as its own event.
- CLI chat streams decoded token-pieces by default so the answer appears progressively in the terminal.
- Local GGUF and safetensors model imports are copied into a managed model store.
- Hugging Face pulls use `git clone` for now, so install `git` and `git-lfs` for real model repos.

Current generation support is selected through the Werk1112 Runtime Planner. Werk is an inference router, not a Candle wrapper: GGUF uses a persistent llama.cpp `llama-server` process as the hot path, so decode, sampling, KV cache, logits, and GPU execution stay inside llama.cpp. HF safetensors directories prefer Burn when a usable Burn runner is available, with Candle kept as a compatibility fallback for text-only models. MLX model directories can run through an external `mlx-lm` backend.

## Format Support

| Format | Typical Use | Import/List/Inspect | Backend Status |
| --- | --- | --- | --- |
| Safetensors | Hugging Face training/fine-tuning standard | Yes | Burn CUDA/CPU when a runner is available; Candle CUDA/CPU/Metal compatibility fallback; MLX through `mlx-lm` when selected |
| GGUF | llama.cpp, Ollama, LM Studio, CPU inference | Yes | CUDA/Vulkan/CPU through persistent llama.cpp server; Candle is legacy/fallback only |
| PyTorch (`.pt`, `.pth`, `pytorch_model.bin`) | Training, research, checkpoints | Yes | Catalog/import only; no generation backend yet |
| ONNX (`.onnx`) | Framework-independent inference | Yes | ONNX Runtime CUDA/CPU when a runner is available |
| MLX (`.npz`, MLX-style dirs) | Apple Silicon / MLX-LM | Yes | Implemented through external `mlx-lm` backend when configured |
| TensorRT Engine (`.engine`, `.plan`) | NVIDIA-optimized inference | Yes | Catalog/import only; no generation backend yet |
| OpenVINO IR (`.xml` + `.bin`) | Intel CPUs, GPUs, NPUs | Yes | Catalog/import only; no generation backend yet |
| TensorFlow (`.ckpt`, `.pb`) | TensorFlow ecosystem | Yes | Catalog/import only; no generation backend yet |
| CoreML (`.mlmodel`, `.mlpackage`) | iOS/macOS deployment | Yes | Catalog/import only; no generation backend yet |

## Werk1112 Runtime Planner

Werk routes by requested backend, model format, model architecture, input capabilities, compiled features, and discovered companion runtimes. `--backend auto` may fall back to CPU. Explicit GPU requests such as `--backend cuda` or `--backend vulkan` do not silently fall back to CPU.

| Runtime | Formats | Accelerators | VLM | Status |
| --- | --- | --- | --- | --- |
| llama.cpp server | GGUF | CUDA, Vulkan, CPU | Pending mmproj/image wiring | Primary GGUF hot path |
| Burn | Safetensors | CUDA, CPU | No | Primary safetensors path when a runner is available |
| ONNX Runtime | Managed ONNX artifacts/direct ONNX when selected | CUDA, CPU | No | Explicit opt-in route |
| Candle | GGUF legacy, safetensors | CUDA, Metal, CPU | No | Implemented for selected architectures |
| vLLM | Selected HF safetensors | CUDA | Backend-dependent | Explicit opt-in route |
| MLX | MLX dirs, selected HF safetensors | Apple Silicon / MLX | Backend-dependent | Implemented through `mlx-lm` when configured |

Planner policy:

- GGUF routes to llama.cpp server CUDA, Vulkan, or CPU first. Candle GGUF is legacy fallback/debug only.
- Safetensors routing is format-based: Burn first, then Candle fallback. Burn and Candle decide internally whether they can load a model.
- `--backend vllm` is a strict vLLM-only safetensors route and never falls back to Candle.
- `--backend cpu` is CPU-only, `--backend cuda` is CUDA-only, `--backend vulkan` is Vulkan-only, and `--backend candle` is an explicit Candle route.
- Image requests filter to VLM-capable runtimes before loading; Candle text routes reject image input.
- `--debug` prints the candidate runtime decisions and rejection reasons. `--verbose` stays focused on backend/timing stats.

## Build

Default local installs build the recommended runtime stack so users do not need to know which internal runtime powers each model format. For the current Windows/Linux workflow that means CUDA support for llama.cpp and Candle, plus packaged companion runners such as Burn when the release artifact includes them. Release builds use one target-specific Cargo alias per deployed end-user artifact. Each artifact includes the supported backends for that platform, and users choose the active backend at runtime with `--backend`.

```bash
cargo install --path . --locked --force
```

For a portable CPU-only development binary, opt out of default features:

```bash
cargo check --locked --no-default-features
cargo build-cpu
```

Target release builds:

```bash
cargo build-windows
cargo build-linux
cargo build-macos-apple-silicon
```

Run target release aliases on the matching build OS when GPU acceleration is involved. In practice:

- Run `cargo build-windows` from native Windows PowerShell with the MSVC Rust toolchain and Windows CUDA installed.
- Run `cargo build-linux` from Linux or WSL with Linux CUDA installed.
- Run `cargo build-macos-apple-silicon` on Apple Silicon macOS.

Do not use WSL to produce the Windows artifact. WSL can build the Linux artifact.

These aliases expand to normal Cargo target builds:

```text
cargo build-windows              -> no default features + x86_64-pc-windows-msvc + release-windows
cargo build-linux                -> no default features + x86_64-unknown-linux-gnu + release-linux
cargo build-macos-apple-silicon  -> no default features + aarch64-apple-darwin + release-macos-apple-silicon
```

Cargo aliases are subcommands, so the command is `cargo build-windows`, not `cargo build windows`.

If a target build fails with `E0463` / `can't find crate for core` or many dependencies fail immediately, the Rust standard library for that target is not installed in the active toolchain. For CPU-only cross-checks you can install it with `rustup target add <target-triple>`, but CUDA/Metal release artifacts should still be built natively on their target OS.

Release backend bundles:

| Bundle | Compiled backend support | Companion backend support |
| --- | --- | --- |
| `release-windows` | CPU, CUDA | GGUF via persistent llama.cpp server; VLM through a capable backend integration |
| `release-linux` | CPU, CUDA | GGUF via persistent llama.cpp server; VLM through a capable backend integration |
| `release-macos-apple-silicon` | CPU | MLX via `mlx-lm`; VLM through a capable external backend |

Raw Cargo equivalents:

```bash
cargo build --release --locked --no-default-features --target x86_64-pc-windows-msvc --features release-windows
cargo build --release --locked --no-default-features --target x86_64-unknown-linux-gnu --features release-linux
cargo build --release --locked --no-default-features --target aarch64-apple-darwin --features release-macos-apple-silicon
```

Windows and Linux release builds compile llama.cpp CUDA, so they require a working NVIDIA driver, CUDA toolkit, C/C++ toolchain, and libclang on the build machine. If `cargo build-windows` fails with ``nvcc --version` failed` / `program not found`, the CUDA toolkit is not on `PATH` in that PowerShell session. On Windows with CUDA 13.x, CCCL can fail with `MSVC/cl.exe with traditional preprocessor is used`; the native Windows setup below passes `/Zc:preprocessor` to MSVC through the `CL` environment variable. Point the build at the intended installed toolkit:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_ROOT=/usr/local/cuda-13.0
export CUDA_PATH=/usr/local/cuda-13.0
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

nvcc --version
cargo build-linux
```

If the CUDA build then fails because NVML cannot query the GPU, set the compute capability manually. For example, an RTX 30xx/Ampere `sm_86` GPU uses:

```bash
export CUDA_COMPUTE_CAP=86
cargo build-linux
```

For a local install, use the same rule. `--locked` keeps the checked-in dependency graph, and `--force` replaces an existing `werk` install. The default install builds the recommended stack:

```bash
cargo install --path . --locked --force
```

For a CUDA-enabled local install with an explicit feature selection, make sure the newer CUDA toolkit is first:

```bash
sudo apt-get update
sudo apt-get install -y clang libclang-dev

export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_ROOT=/usr/local/cuda-13.0
export CUDA_PATH=/usr/local/cuda-13.0
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

cargo install --path . --locked --force --features cuda
```

If you are building inside WSL with Ubuntu's CUDA 11.5 package, `nvcc` may fail inside `llama_cpp_sys` with errors from `/usr/include/c++/11/bits/std_function.h`. That means CUDA 11.5 is using GCC 11 as the host compiler. The repository Cargo config defaults Linux CUDA builds to GCC/G++ 10, so install those once:

```bash
sudo apt-get install -y gcc-10 g++-10

cargo install --path . --locked --force --features cuda
```

To override that default, set `CC_x86_64_unknown_linux_gnu` or `CXX_x86_64_unknown_linux_gnu` in your shell before running Cargo.

If the build fails in `candle-kernels` with `fatal error: cuda_fp8.h: No such file or directory`, the active CUDA toolkit is too old for Candle CUDA. Install a newer CUDA toolkit and make sure its `bin` and `include` directories come before Ubuntu's `/usr` CUDA package. For a CPU-only development build, use `cargo install --path . --locked --force --no-default-features`.

Native Windows CUDA / Developer PowerShell:

1. Install Rust for Windows with `rustup`.
2. Install Visual Studio Build Tools with the `Desktop development with C++` workload.
3. Install Git, Git LFS, LLVM/libclang, and a Windows CUDA Toolkit.
4. Open native Windows Developer PowerShell, not a WSL shell.
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

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsInstall) {
  throw "Visual Studio C++ build tools not found. Install Visual Studio Build Tools with Desktop development with C++."
}

Import-Module (Join-Path $vsInstall "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath $vsInstall -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"

$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
$env:CUDA_HOME = Get-ChildItem $cudaRoot -Directory |
  Sort-Object Name -Descending |
  Select-Object -First 1 -ExpandProperty FullName

if (-not $env:CUDA_HOME) {
  throw "CUDA Toolkit not found under $cudaRoot. Install the CUDA Toolkit, not only the NVIDIA driver."
}

$env:CUDA_ROOT = $env:CUDA_HOME
$env:CUDA_PATH = $env:CUDA_HOME
$env:CUDA_TOOLKIT_ROOT_DIR = $env:CUDA_HOME
$env:CUDA_COMPUTE_CAP = "86"
$env:Path = "$env:CUDA_HOME\bin;$env:Path"
$env:CL = "/Zc:preprocessor"

if (-not (Test-Path "$env:CUDA_HOME\bin\nvcc.exe")) {
  throw "nvcc.exe not found in $env:CUDA_HOME\bin. Check the CUDA Toolkit installation."
}

where.exe cl
$clPath = (Get-Command cl.exe).Source
if ($clPath -notmatch "\\Hostx64\\x64\\cl\.exe$") {
  throw "MSVC is not in x64 mode. Re-run Enter-VsDevShell with -arch=x64 -host_arch=x64, or open the x64 Native Tools shell."
}
$env:NVCC_CCBIN = Split-Path $clPath
cl
where.exe nvcc
nvcc --version
cargo build-windows
```

The release binary is written to Cargo's target directory:

```text
target/x86_64-pc-windows-msvc/release/werk.exe
target/x86_64-unknown-linux-gnu/release/werk
target/aarch64-apple-darwin/release/werk
```

Runtime backend setup should be a black box for end users. GGUF execution uses a persistent llama.cpp server backend so the decode loop, sampling, KV cache, logits, and GPU execution stay inside llama.cpp. HF safetensors execution prefers Burn when a usable Burn runner is packaged, installed, or configured, so Candle is a compatibility fallback instead of the primary target. In `--backend auto`, Werk skips unavailable companion runtimes quietly and uses the best working fallback; `--verbose` prints only the selected backend and one concise fallback note, while `--debug` prints full discovery paths and rejection details. Source builds may need managed or PATH-provided companion runtimes; use `werk backend list`, `werk backend doctor --debug`, `werk backend install llama-cuda`, and `werk artifacts build <model>` for local development. `WERK_LLAMA_CTX`, `WERK_LLAMA_BATCH`, `WERK_LLAMA_UBATCH`, and `WERK_LLAMA_MAIN_GPU` are advanced GGUF tuning overrides. `WERK_BURN_RUNNER_CUDA`, `WERK_BURN_RUNNER_CPU`, and `WERK_BURN_RUNNER` are advanced Burn runner overrides. `WERK_ONNX_RUNTIME_CUDA`, `WERK_ONNX_RUNTIME_CPU`, `WERK_ONNX_RUNTIME`, `WERK_ONNX_RUNTIME_BUNDLE_CUDA`, `WERK_ONNX_RUNTIME_BUNDLE_CPU`, and `WERK_ONNX_EXPORTER` are advanced ONNX artifact/runtime overrides. `WERK_VLLM_PYTHON`, `WERK_VLLM_HOST`, `WERK_VLLM_PORT`, and `WERK_VLLM_ARGS` remain available for explicit vLLM routes. The MLX backend uses `python3 -m mlx_lm.generate` or `WERK_MLX_PYTHON`. VLM request/image support is compiled into every build; actual multimodal generation depends on the chosen model and backend.

Additional low-level acceleration features are available for custom builds:

```bash
cargo build --release --locked --features mkl
cargo build --release --locked --features candle-cuda
cargo build --release --locked --features cuda,cudnn
```

The top-level `cuda` feature means CUDA support across Werk: llama.cpp CUDA plus Candle CUDA. It is included by the default `recommended` feature and can still be requested explicitly:

```bash
cargo install --path . --locked --force --features cuda
```

The lower-level `candle-cuda` feature is still available for custom builds that only need Candle CUDA without llama.cpp CUDA. `cudnn` builds on Candle CUDA. These features may require CUDA headers such as `cuda_fp8.h`, which are not present in Ubuntu's CUDA 11.5 package.

Build features decide what Candle and llama.cpp acceleration support is compiled into the binary. Backend selection is a separate CLI option and is the preferred way to choose how a process runs:

```bash
werk --backend auto chat gemma-2b-it
werk --backend cpu chat gemma-2b-it
werk --backend cuda chat gemma-2b-it
werk --backend metal chat gemma-2b-it
werk --backend vulkan chat TinyLLama-1B-GGUF
werk --backend mlx chat mlx-model
werk --backend candle chat debug-model
werk --backend burn chat phi-3-mini-4k-instruct
werk --backend onnx chat phi-3-mini-4k-instruct
werk --backend vllm chat phi-3-mini-4k-instruct
werk --backend cuda serve --model gemma-2b-it
```

`--backend auto` is format-aware. For GGUF on Windows/Linux it tries llama.cpp server CUDA, Vulkan, CPU, then Candle legacy GPU/CPU fallback. For safetensors it tries Burn first when a Burn runner is usable, then Candle CUDA/Metal/CPU as a compatibility fallback. With `--verbose`, Burn-to-Candle fallback is reported as one concise note; use `--debug` for paths, attempted runtimes, and rejection reasons. On Apple Silicon it prefers MLX for MLX models and may use Burn CPU, MLX, or Candle for safetensors depending on availability.

For GGUF models, `--backend cuda`, `--backend vulkan`, and `--backend cpu` use the persistent llama.cpp server backend when that server is available. For safetensors models, those same backend names mean "choose the best runtime for that accelerator," not "force Candle": CUDA tries Burn CUDA then Candle CUDA, and CPU tries Burn CPU then Candle CPU. `--backend burn` is strict Burn only and never falls back to Candle. `--backend onnx` is strict ONNX Runtime only: it never falls back to Candle and fails with actionable diagnostics if no runner is installed. `--backend vllm` remains an explicit vLLM-only route and fails clearly if vLLM is missing, broken, or cannot load the model. Explicit GPU backend requests do not silently fall back to CPU; they fail with an actionable error if the requested runtime is unavailable. `--backend candle` is available for debugging or fallback verification. `--device` remains as a Candle-only compatibility override, but `--backend` is what end users should use.

## End-User Releases

Release artifacts should be produced with the target Cargo alias on the target platform. A complete GGUF release includes the Cargo-built `werk` binary plus the platform llama.cpp server companion that Werk discovers or bundles for that artifact. End users should not need Rust, Cargo, Visual Studio, CMake, Git, libclang, or `nvcc`; those are build-machine/source-install requirements only.

Do not ship one artifact per backend unless a backend cannot be co-compiled. The current Windows/Linux release artifact is CUDA-first; Vulkan requires a separate custom non-CUDA build because `llama_cpp_sys` rejects CUDA and Vulkan in the same binary.

Each target artifact should include the supported backends for that build, and users can select one explicitly with `--backend`:

| Platform | Cargo command | Included backend support | Auto default |
| --- | --- | --- | --- |
| Windows 10/11 x64 | `cargo build-windows` | CPU, CUDA via llama.cpp server | llama.cpp server CUDA |
| Linux x64 | `cargo build-linux` | CPU, CUDA via llama.cpp server | llama.cpp server CUDA |
| macOS Apple Silicon | `cargo build-macos-apple-silicon` | CPU, MLX-LM, VLM request support | MLX |

Backend selection is per process. Managed runtime provisioning and generated artifacts are cached under `WERK_HOME` when needed.

```bash
werk --backend auto chat model-id
werk --backend cuda chat model-id
werk --backend vulkan chat model-id
werk --backend mlx chat model-id
werk --backend metal chat model-id
werk --backend cpu chat model-id
werk --backend candle chat model-id
werk --backend burn chat model-id
werk --backend onnx chat model-id
werk --backend vllm chat model-id
```

`auto` is format-aware: GGUF uses llama.cpp server CUDA, Vulkan, CPU, then Candle legacy GPU/CPU fallback; safetensors uses Burn before Candle fallback when a Burn runner is packaged or installed; MLX-format models use MLX. Explicit `burn`, `onnx`, and `vllm` requests do not fallback, and explicit GPU requests do not silently fall back to CPU.

MLX and Metal are not the same backend. Metal is implemented through Candle. MLX is implemented as an external `mlx-lm` backend. CUDA, Vulkan, and CPU GGUF hot paths are implemented through persistent llama.cpp server backends. `WERK_MLX_PYTHON` can point to a Python environment with `mlx-lm` installed.

VLM support means multimodal model/request support, not a separate backend. Today image inputs should be routed to an image-capable MLX/VLM backend. Candle is text-only, and llama.cpp VLM support is pending until mmproj/image argument handling is wired.

CLI image inputs use repeatable `--image` flags:

```bash
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
├── files/
│   └── ...
└── artifacts/
    └── onnx/
```

On native Windows, the same model is saved here by default:

```text
%LOCALAPPDATA%\werk1112\models\TinyLLama-1B-GGUF\
```

`manifest.json` contains source, format, architecture, tokenizer/config paths, model file path, checksums, backend hints, and optimized artifact metadata. `files/` contains the copied local model files or downloaded Hugging Face files. `artifacts/` contains generated runtime artifacts such as ONNX exports.

For GGUF repositories that contain many quantizations, new imports prefer a balanced `Q4_K_M` file when it is present instead of taking the first filename alphabetically. The selected file is stored in `manifest.json` as `model_path`, and both `chat` and `serve` use that selected file.

## CLI

Install the CLI from this checkout with the recommended runtime stack:

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

During development, you can also run the local CPU-only binary without installing:

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

Build, list, or rebuild optimized runtime artifacts for a safetensors model:

```bash
werk artifacts build phi-3-mini-4k-instruct
werk artifacts list phi-3-mini-4k-instruct
werk artifacts rebuild phi-3-mini-4k-instruct
```

When an explicit ONNX Runtime route is selected and no ready ONNX artifact exists, Werk attempts the same managed artifact build automatically. Auto SafeTensors routing uses Burn before Candle and does not require ONNX artifacts. Set `WERK_ONNX_EXPORTER` to a custom exporter binary, or install `optimum-cli` / `optimum.exporters.onnx` for the default exporter path.

Remove an installed model from Werk's managed store. This deletes the managed copy under `WERK_HOME`; it does not delete the original local import source or any upstream Hugging Face repository:

```bash
werk remove model-local
werk rm model-local
```

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

- Extend safetensors Burn coverage and ship managed Burn runner artifacts for release builds.
- Add richer chat template support from tokenizer/model metadata.
- Keep improving GGUF performance through persistent llama.cpp server defaults and runtime tuning.
- Add real generation backends for TensorRT, OpenVINO, TensorFlow, CoreML, and direct PyTorch checkpoint execution/conversion when they can execute end-to-end.
- Add embeddings and tool-call response support after the chat/models baseline is stable.
