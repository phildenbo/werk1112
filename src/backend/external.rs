use anyhow::{Context, Result, anyhow, bail};
#[cfg(feature = "llama-cpp")]
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use std::{
    env, fs,
    io::{self, Read},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::Instant,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

#[cfg(feature = "llama-cpp")]
use super::ChatGenerationSession;
use super::{
    GenerateRequest, GenerateResponse, GenerateStream, GenerateStreamEvent, GenerationBackend,
    GenerationTimings,
};
use crate::model_store::{ModelFormat, ModelManifest, ModelStore};
use serde_json::{Value, json};

#[cfg(feature = "llama-cpp")]
use llama_cpp::{
    LlamaModel, LlamaParams, LlamaSession, SessionParams, Token,
    standard_sampler::{SamplerStage, StandardSampler},
};

#[cfg(feature = "llama-cpp")]
const DEFAULT_N_CTX: u32 = 0;
#[cfg(feature = "llama-cpp")]
const DEFAULT_N_BATCH: u32 = 2048;
#[cfg(feature = "llama-cpp")]
const DEFAULT_N_UBATCH: u32 = 512;
const GEMMA4_UNIFIED_MODEL_TYPE: &str = "gemma4_unified";
const GEMMA4_UNIFIED_MLX_COMPAT_DIR: &str = "mlx-gemma4-unified-text";
const GEMMA4_UNIFIED_MLX_COMPAT_MODEL_FILE: &str = "werk_gemma4_unified_compat.py";
const GEMMA4_UNIFIED_MLX_COMPAT_MODEL_PY: &str = r#"from mlx_lm.models import gemma4 as _gemma4

ModelArgs = _gemma4.ModelArgs

_NON_TEXT_PREFIXES = (
    "vision_embedder.",
    "vision_tower.",
    "multi_modal_projector.",
    "audio_tower.",
    "embed_audio.",
    "embed_vision.",
)


def _is_non_text_weight(name):
    name = name.removeprefix("model.")
    return name.startswith(_NON_TEXT_PREFIXES)


class Model(_gemma4.Model):
    def sanitize(self, weights):
        weights = {
            key: value
            for key, value in weights.items()
            if not _is_non_text_weight(key)
        }
        return super().sanitize(weights)
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LlamaCppMode {
    Cuda,
    Rocm,
    Vulkan,
    Metal,
    Cpu,
}

#[cfg(feature = "llama-cpp")]
#[derive(Clone)]
pub struct LlamaCppBackend {
    store: ModelStore,
    mode: LlamaCppMode,
    models: Arc<Mutex<HashMap<String, Arc<LlamaModel>>>>,
}

#[cfg(feature = "llama-cpp")]
struct LlamaCppChatSession {
    state: Arc<Mutex<LlamaCppChatState>>,
}

#[cfg(feature = "llama-cpp")]
struct LlamaCppChatState {
    model: Arc<LlamaModel>,
    session: LlamaSession,
    params: SessionParams,
}

#[cfg(not(feature = "llama-cpp"))]
#[derive(Clone)]
pub struct LlamaCppBackend {
    _store: ModelStore,
    mode: LlamaCppMode,
}

#[derive(Debug, Clone)]
pub struct MlxBackend {
    store: ModelStore,
    python: PathBuf,
    module: String,
}

#[derive(Debug, Clone)]
pub struct MlxVlmBackend {
    store: ModelStore,
    python: PathBuf,
    module: String,
}

impl LlamaCppMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Cuda => "llama-cpp-cuda",
            Self::Rocm => "llama-cpp-rocm",
            Self::Vulkan => "llama-cpp-vulkan",
            Self::Metal => "llama-cpp-metal",
            Self::Cpu => "llama-cpp-cpu",
        }
    }

    #[cfg(feature = "llama-cpp")]
    fn display_name(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Rocm => "ROCm/HIP",
            Self::Vulkan => "Vulkan",
            Self::Metal => "Metal",
            Self::Cpu => "CPU",
        }
    }

    #[cfg(feature = "llama-cpp")]
    fn gpu_layers(self) -> u32 {
        match self {
            Self::Cpu => 0,
            Self::Cuda | Self::Rocm | Self::Vulkan | Self::Metal => 999,
        }
    }

    #[cfg(feature = "llama-cpp")]
    fn uses_gpu(self) -> bool {
        !matches!(self, Self::Cpu)
    }

    #[cfg(feature = "llama-cpp")]
    fn compiled(self) -> bool {
        match self {
            Self::Cpu => cfg!(feature = "llama-cpp"),
            Self::Cuda => cfg!(feature = "llama-legacy-cuda"),
            Self::Rocm => false,
            Self::Vulkan => cfg!(feature = "llama-legacy-vulkan"),
            Self::Metal => false,
        }
    }

    fn unavailable_message(self) -> String {
        match self {
            Self::Cuda => {
                "llama.cpp CUDA backend is not compiled into this binary; build/install with --features llama-legacy-cuda".to_string()
            }
            Self::Vulkan => {
                "llama.cpp Vulkan backend is not compiled into this binary; build/install with --features llama-legacy-vulkan".to_string()
            }
            Self::Rocm => {
                "legacy llama.cpp ROCm/HIP backend is not implemented; use the persistent llama.cpp server ROCm route via --backend rocm".to_string()
            }
            Self::Metal => {
                "legacy llama.cpp Metal backend is not implemented; use the persistent llama.cpp server Metal route via --backend metal".to_string()
            }
            Self::Cpu => {
                "llama.cpp CPU backend is not compiled into this binary; build/install with --features llama-cpp".to_string()
            }
        }
    }
}

#[cfg(feature = "llama-cpp")]
impl LlamaCppBackend {
    pub fn new(store: ModelStore, mode: LlamaCppMode) -> Self {
        Self {
            store,
            mode,
            models: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn probe(mode: LlamaCppMode) -> Result<String> {
        if !mode.compiled() {
            bail!("{}", mode.unavailable_message());
        }
        Ok(format!("in-process llama.cpp {}", mode.display_name()))
    }

    fn cached_model(&self, manifest: &ModelManifest) -> Result<(Arc<LlamaModel>, f64)> {
        if manifest.format != ModelFormat::Gguf {
            bail!(
                "llama.cpp {} backend supports GGUF models only",
                self.mode.display_name()
            );
        }

        let model_path = manifest
            .model_path
            .as_deref()
            .context("GGUF manifest has no model_path")?;
        let cache_key = format!("{}:{model_path}:{}", manifest.id, self.mode.label());

        if let Some(model) = self
            .models
            .lock()
            .map_err(|_| anyhow!("llama.cpp model cache mutex poisoned"))?
            .get(&cache_key)
            .cloned()
        {
            return Ok((model, 0.0));
        }

        if !self.mode.compiled() {
            bail!("{}", self.mode.unavailable_message());
        }

        let absolute_model_path = self.store.absolute_model_file(manifest, model_path);
        eprintln!(
            "Loading model '{}' with in-process llama.cpp {}",
            manifest.id,
            self.mode.display_name()
        );
        let started = Instant::now();
        let model = LlamaModel::load_from_file(&absolute_model_path, self.model_params())
            .map_err(|err| anyhow!("failed to load GGUF with llama.cpp: {err}"))?;
        let load_seconds = started.elapsed().as_secs_f64();
        eprintln!(
            "Loaded model '{}' with llama.cpp {} in {:.2}s",
            manifest.id,
            self.mode.display_name(),
            load_seconds
        );

        let model = Arc::new(model);
        self.models
            .lock()
            .map_err(|_| anyhow!("llama.cpp model cache mutex poisoned"))?
            .insert(cache_key, model.clone());
        Ok((model, load_seconds))
    }

    fn model_params(&self) -> LlamaParams {
        let mut params = LlamaParams::default();
        params.n_gpu_layers = self.mode.gpu_layers();
        params.use_mmap = true;
        params.main_gpu = env_u32("WERK_LLAMA_MAIN_GPU").unwrap_or(params.main_gpu);
        params
    }

    fn session_params(&self, request: &GenerateRequest) -> SessionParams {
        self.session_params_for_seed(request.seed)
    }

    fn session_params_for_seed(&self, seed: Option<u64>) -> SessionParams {
        let mut params = SessionParams::default();
        params.n_ctx = env_u32("WERK_LLAMA_CTX").unwrap_or(DEFAULT_N_CTX);
        params.n_batch = env_u32("WERK_LLAMA_BATCH").unwrap_or(DEFAULT_N_BATCH);
        params.n_ubatch = env_u32("WERK_LLAMA_UBATCH").unwrap_or(DEFAULT_N_UBATCH);
        params.n_threads = env_u32("WERK_LLAMA_THREADS").unwrap_or(params.n_threads);
        params.n_threads_batch =
            env_u32("WERK_LLAMA_THREADS_BATCH").unwrap_or(params.n_threads_batch);
        params.offload_kqv = self.mode.uses_gpu();
        params.seed = seed.map(seed_u32).unwrap_or(u32::MAX);
        params
    }

    fn create_session(
        &self,
        model: &LlamaModel,
        seed: Option<u64>,
    ) -> Result<(LlamaSession, SessionParams)> {
        let params = self.session_params_for_seed(seed);
        let mut session = model
            .create_session(params.clone())
            .map_err(|err| anyhow!("failed to create llama.cpp session: {err}"))?;
        warmup_session(&mut session);
        Ok((session, params))
    }

    fn generate_inner(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
        tx: Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
    ) -> Result<GenerateResponse> {
        if !request.image_urls.is_empty() {
            bail!(
                "llama.cpp Rust backend is text-only for now; use a VLM-capable backend/model for image inputs"
            );
        }

        let total_started = Instant::now();
        let (model, load_seconds) = self.cached_model(manifest)?;
        let prompt_tokens = model
            .tokenize_bytes(request.prompt.as_bytes(), false, true)
            .map_err(|err| anyhow!("failed to tokenize prompt with llama.cpp: {err}"))?;
        let prompt_token_count = prompt_tokens.len();

        let mut session = model
            .create_session(self.session_params(&request))
            .map_err(|err| anyhow!("failed to create llama.cpp session: {err}"))?;
        let prompt_started = Instant::now();
        session
            .advance_context_with_tokens(&prompt_tokens)
            .map_err(|err| anyhow!("failed to evaluate prompt with llama.cpp: {err}"))?;
        let prompt_seconds = prompt_started.elapsed().as_secs_f64();

        let decode_started = Instant::now();
        let completion = session
            .start_completing_with(sampler_for(&request), request.max_tokens)
            .map_err(|err| anyhow!("failed to start llama.cpp completion: {err}"))?;
        let mut finish_reason = "length".to_string();
        let mut text = String::new();
        let mut completion_tokens = 0usize;

        for chunk in completion.into_strings() {
            completion_tokens += 1;
            if chunk.is_empty() {
                continue;
            }
            let previous_len = text.len();
            text.push_str(&chunk);

            if let Some(stop_index) = first_stop_index(&text, &request.stop) {
                if stop_index > previous_len {
                    send_text_chunk(&tx, text[previous_len..stop_index].to_string())?;
                }
                text.truncate(stop_index);
                finish_reason = "stop".to_string();
                break;
            }

            send_text_chunk(&tx, chunk)?;
        }

        let decode_seconds = decode_started.elapsed().as_secs_f64();
        Ok(GenerateResponse {
            text,
            prompt_tokens: prompt_token_count,
            completion_tokens,
            finish_reason,
            timings: GenerationTimings {
                load_seconds,
                warmup_seconds: 0.0,
                first_token_seconds: 0.0,
                prompt_seconds,
                decode_seconds,
                total_seconds: total_started.elapsed().as_secs_f64(),
            },
            backend_diagnostics: Vec::new(),
        })
    }
}

#[cfg(feature = "llama-cpp")]
impl LlamaCppChatState {
    fn generate_inner(
        &mut self,
        request: GenerateRequest,
        tx: Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
    ) -> Result<GenerateResponse> {
        if !request.image_urls.is_empty() {
            bail!(
                "llama.cpp Rust backend is text-only for now; use a VLM-capable backend/model for image inputs"
            );
        }

        let total_started = Instant::now();
        let prompt_tokens = self
            .model
            .tokenize_bytes(request.prompt.as_bytes(), false, true)
            .map_err(|err| anyhow!("failed to tokenize prompt with llama.cpp: {err}"))?;
        if prompt_tokens.is_empty() {
            bail!("prompt is empty after tokenization");
        }
        if self.params.n_ctx > 0 && prompt_tokens.len() >= self.params.n_ctx as usize {
            bail!(
                "chat context is full for this model ({} prompt tokens, {} token context); increase WERK_LLAMA_CTX or start a new chat",
                prompt_tokens.len(),
                self.params.n_ctx
            );
        }

        let previous_tokens = self.session.context();
        let shared_prefix = shared_prefix_len(&previous_tokens, &prompt_tokens);
        let prompt_token_count = prompt_tokens.len().saturating_sub(shared_prefix);

        let prompt_started = Instant::now();
        self.session
            .set_context_to_tokens(&prompt_tokens)
            .map_err(|err| anyhow!("failed to update llama.cpp chat context: {err}"))?;
        let prompt_seconds = prompt_started.elapsed().as_secs_f64();

        let decode_started = Instant::now();
        let max_predictions = if self.params.n_ctx > 0 {
            request.max_tokens.min(
                (self.params.n_ctx as usize)
                    .saturating_sub(self.session.context_size())
                    .saturating_sub(1)
                    .max(1),
            )
        } else {
            request.max_tokens
        };
        let completion = self
            .session
            .start_completing_with(sampler_for(&request), max_predictions)
            .map_err(|err| anyhow!("failed to start llama.cpp completion: {err}"))?;
        let mut finish_reason = "length".to_string();
        let mut text = String::new();
        let mut completion_tokens = 0usize;

        for chunk in completion.into_strings() {
            completion_tokens += 1;
            if chunk.is_empty() {
                continue;
            }
            let previous_len = text.len();
            text.push_str(&chunk);

            if let Some(stop_index) = first_stop_index(&text, &request.stop) {
                if stop_index > previous_len {
                    send_text_chunk(&tx, text[previous_len..stop_index].to_string())?;
                }
                text.truncate(stop_index);
                finish_reason = "stop".to_string();
                break;
            }

            send_text_chunk(&tx, chunk)?;
        }

        let decode_seconds = decode_started.elapsed().as_secs_f64();
        Ok(GenerateResponse {
            text,
            prompt_tokens: prompt_token_count,
            completion_tokens,
            finish_reason,
            timings: GenerationTimings {
                load_seconds: 0.0,
                warmup_seconds: 0.0,
                first_token_seconds: 0.0,
                prompt_seconds,
                decode_seconds,
                total_seconds: total_started.elapsed().as_secs_f64(),
            },
            backend_diagnostics: Vec::new(),
        })
    }
}

#[cfg(feature = "llama-cpp")]
impl ChatGenerationSession for LlamaCppChatSession {
    fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        self.state
            .lock()
            .map_err(|_| anyhow!("llama.cpp chat session mutex poisoned"))
            .and_then(|mut state| state.generate_inner(request, None))
    }

    fn generate_stream(&self, request: GenerateRequest) -> GenerateStream {
        let state = self.state.clone();
        let (tx, rx) = mpsc::channel(16);

        tokio::task::spawn_blocking(move || {
            let result = state
                .lock()
                .map_err(|_| anyhow!("llama.cpp chat session mutex poisoned"))
                .and_then(|mut state| state.generate_inner(request, Some(tx.clone())));
            match result {
                Ok(response) => {
                    let _ = tx.blocking_send(Ok(GenerateStreamEvent::Done {
                        finish_reason: response.finish_reason,
                        prompt_tokens: response.prompt_tokens,
                        completion_tokens: response.completion_tokens,
                        timings: response.timings,
                        backend_diagnostics: response.backend_diagnostics,
                    }));
                }
                Err(err) => {
                    let _ = tx.blocking_send(Err(format_error_chain(&err)));
                }
            }
        });

        Box::pin(ReceiverStream::new(rx))
    }
}

#[cfg(feature = "llama-cpp")]
impl GenerationBackend for LlamaCppBackend {
    fn prepare(&self, manifest: &ModelManifest) -> Result<()> {
        self.cached_model(manifest).map(|_| ())
    }

    fn start_chat_session(
        &self,
        manifest: &ModelManifest,
        seed: Option<u64>,
    ) -> Result<Option<Box<dyn ChatGenerationSession>>> {
        if manifest.format != ModelFormat::Gguf {
            return Ok(None);
        }
        let (model, _) = self.cached_model(manifest)?;
        let (session, params) = self.create_session(&model, seed)?;
        Ok(Some(Box::new(LlamaCppChatSession {
            state: Arc::new(Mutex::new(LlamaCppChatState {
                model,
                session,
                params,
            })),
        })))
    }

    fn generate(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> Result<GenerateResponse> {
        self.generate_inner(manifest, request, None)
    }

    fn generate_stream(&self, manifest: ModelManifest, request: GenerateRequest) -> GenerateStream {
        let backend = self.clone();
        let (tx, rx) = mpsc::channel(16);
        tokio::task::spawn_blocking(move || {
            let result = backend.generate_inner(&manifest, request, Some(tx.clone()));
            match result {
                Ok(response) => {
                    let _ = tx.blocking_send(Ok(GenerateStreamEvent::Done {
                        finish_reason: response.finish_reason,
                        prompt_tokens: response.prompt_tokens,
                        completion_tokens: response.completion_tokens,
                        timings: response.timings,
                        backend_diagnostics: response.backend_diagnostics,
                    }));
                }
                Err(err) => {
                    let _ = tx.blocking_send(Err(format_error_chain(&err)));
                }
            }
        });
        Box::pin(ReceiverStream::new(rx))
    }
}

#[cfg(not(feature = "llama-cpp"))]
impl LlamaCppBackend {
    pub fn new(store: ModelStore, mode: LlamaCppMode) -> Self {
        Self {
            _store: store,
            mode,
        }
    }

    pub fn probe(mode: LlamaCppMode) -> Result<String> {
        bail!("{}", mode.unavailable_message())
    }
}

#[cfg(not(feature = "llama-cpp"))]
impl GenerationBackend for LlamaCppBackend {
    fn generate(
        &self,
        _manifest: &ModelManifest,
        _request: GenerateRequest,
    ) -> Result<GenerateResponse> {
        bail!("{}", self.mode.unavailable_message())
    }

    fn generate_stream(
        &self,
        _manifest: ModelManifest,
        _request: GenerateRequest,
    ) -> GenerateStream {
        Box::pin(tokio_stream::iter(vec![Err(self
            .mode
            .unavailable_message())]))
    }
}

impl MlxBackend {
    pub fn new(store: ModelStore) -> Self {
        Self {
            store,
            python: backend_program("WERK_MLX_PYTHON", default_python()),
            module: env::var("WERK_MLX_MODULE").unwrap_or_else(|_| "mlx_lm.generate".to_string()),
        }
    }

    pub fn probe() -> Result<String> {
        let python = backend_program("WERK_MLX_PYTHON", default_python());
        let output = Command::new(&python)
            .args(["-c", "import mlx_lm"])
            .output()
            .with_context(|| {
                format!(
                    "failed to execute {}; set WERK_MLX_PYTHON to a Python with mlx-lm installed",
                    python.display()
                )
            })?;
        if !output.status.success() {
            bail!(
                "mlx-lm is not importable with {}: {}",
                python.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        Ok(format!("mlx-lm via {}", python.display()))
    }

    fn command_for(&self, manifest: &ModelManifest, request: &GenerateRequest) -> Result<Command> {
        if !matches!(manifest.format, ModelFormat::Mlx | ModelFormat::SafeTensors) {
            bail!("mlx backend supports MLX or Hugging Face-style safetensors model directories");
        }
        let model_dir = resolve_mlx_model_dir(&self.store, manifest)?;
        let use_gemma4_unified_compat = is_gemma4_unified_compat_dir(&model_dir);

        let mut command = self.mlx_generate_command();
        command
            .arg("--model")
            .arg(&model_dir)
            .arg("--prompt")
            .arg(&request.prompt)
            .arg("--max-tokens")
            .arg(request.max_tokens.to_string());
        if use_gemma4_unified_compat {
            command
                .arg("--chat-template-config")
                .arg(r#"{"enable_thinking":false}"#);
        }
        if let Some(temperature) = request.temperature {
            command.arg("--temp").arg(format_float(temperature));
        }
        for image in &request.image_urls {
            command.arg("--image").arg(image);
        }
        Ok(command)
    }

    fn mlx_generate_command(&self) -> Command {
        if env::var("WERK_MLX_MODULE")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .is_some()
        {
            return python_module_command(&self.python, &self.module);
        }

        if let Ok(path) = env::var("WERK_MLX_GENERATE")
            && !path.trim().is_empty()
        {
            return Command::new(PathBuf::from(path));
        }

        if let Some(generator) = sibling_program(&self.python, mlx_generate_program()) {
            return Command::new(generator);
        }
        if let Some(generator) = find_program_in_path(mlx_generate_program()) {
            return Command::new(generator);
        }

        python_module_command(&self.python, &self.module)
    }
}

impl MlxVlmBackend {
    pub fn new(store: ModelStore) -> Self {
        Self {
            store,
            python: backend_program("WERK_MLX_VLM_PYTHON", default_python()),
            module: env::var("WERK_MLX_VLM_MODULE").unwrap_or_else(|_| "mlx_vlm".to_string()),
        }
    }

    pub fn probe() -> Result<String> {
        let python = backend_program("WERK_MLX_VLM_PYTHON", default_python());
        let output = Command::new(&python)
            .args(["-c", "import mlx_vlm"])
            .output()
            .with_context(|| {
                format!(
                    "failed to execute {}; set WERK_MLX_VLM_PYTHON to a Python with mlx-vlm installed",
                    python.display()
                )
            })?;
        if !output.status.success() {
            bail!(
                "mlx-vlm is not importable with {}: {}",
                python.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        Ok(format!("mlx-vlm via {}", python.display()))
    }

    fn command_for(
        &self,
        manifest: &ModelManifest,
        request: &GenerateRequest,
        verbose_backend_output: bool,
    ) -> Result<Command> {
        if !matches!(manifest.format, ModelFormat::Mlx | ModelFormat::SafeTensors) {
            bail!(
                "mlx-vlm backend supports MLX or Hugging Face-style safetensors model directories"
            );
        }
        let model_dir = original_mlx_model_dir(&self.store, manifest)?;

        let mut command = self.mlx_vlm_generate_command();
        command
            .arg("--model")
            .arg(&model_dir)
            .arg("--prompt")
            .arg(&request.prompt)
            .arg("--max-tokens")
            .arg(request.max_tokens.to_string());
        if !verbose_backend_output {
            command.arg("--no-verbose");
        }
        if let Some(temperature) = request.temperature {
            command.arg("--temperature").arg(format_float(temperature));
        }
        if let Some(top_p) = request.top_p {
            command.arg("--top-p").arg(format_float(top_p));
        }
        if let Some(seed) = request.seed {
            command.arg("--seed").arg(seed_u32(seed).to_string());
        }
        if !request.image_urls.is_empty() {
            command.arg("--image");
            for image in &request.image_urls {
                command.arg(image);
            }
        }
        Ok(command)
    }

    fn mlx_vlm_generate_command(&self) -> Command {
        if env::var("WERK_MLX_VLM_MODULE")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .is_some()
        {
            return python_module_subcommand(&self.python, &self.module, "generate");
        }

        if let Ok(path) = env::var("WERK_MLX_VLM_GENERATE")
            && !path.trim().is_empty()
        {
            return Command::new(PathBuf::from(path));
        }

        if let Some(generator) = sibling_program(&self.python, mlx_vlm_generate_program()) {
            return Command::new(generator);
        }
        if let Some(generator) = find_program_in_path(mlx_vlm_generate_program()) {
            return Command::new(generator);
        }

        python_module_subcommand(&self.python, &self.module, "generate")
    }
}

fn original_mlx_model_dir(store: &ModelStore, manifest: &ModelManifest) -> Result<PathBuf> {
    let model_dir = store.model_dir(&manifest.id).join("files");
    if !model_dir.is_dir() {
        bail!(
            "model files directory does not exist: {}",
            model_dir.display()
        );
    }
    Ok(model_dir)
}

fn resolve_mlx_model_dir(store: &ModelStore, manifest: &ModelManifest) -> Result<PathBuf> {
    let model_dir = original_mlx_model_dir(store, manifest)?;

    if gemma4_unified_compat_config(&model_dir)?.is_some() {
        ensure_gemma4_unified_compat_dir(store, manifest, &model_dir)
    } else {
        Ok(model_dir)
    }
}

fn is_gemma4_unified_compat_dir(path: &Path) -> bool {
    path.file_name().and_then(|name| name.to_str()) == Some(GEMMA4_UNIFIED_MLX_COMPAT_DIR)
}

fn ensure_gemma4_unified_compat_dir(
    store: &ModelStore,
    manifest: &ModelManifest,
    model_dir: &Path,
) -> Result<PathBuf> {
    let compat_dir = store
        .artifacts_dir(&manifest.id)
        .join(GEMMA4_UNIFIED_MLX_COMPAT_DIR);
    fs::create_dir_all(&compat_dir).with_context(|| {
        format!(
            "failed to create MLX compatibility directory {}",
            compat_dir.display()
        )
    })?;
    mirror_mlx_compat_files(model_dir, model_dir, &compat_dir)?;
    let compat_config = gemma4_unified_compat_config(model_dir)?
        .context("Gemma4 unified compatibility requested for non-Gemma4 config")?;
    fs::write(
        compat_dir.join("config.json"),
        serde_json::to_vec_pretty(&compat_config)?,
    )
    .with_context(|| {
        format!(
            "failed to write MLX compatibility config {}",
            compat_dir.join("config.json").display()
        )
    })?;
    fs::write(
        compat_dir.join(GEMMA4_UNIFIED_MLX_COMPAT_MODEL_FILE),
        GEMMA4_UNIFIED_MLX_COMPAT_MODEL_PY,
    )
    .with_context(|| {
        format!(
            "failed to write MLX compatibility model shim {}",
            compat_dir
                .join(GEMMA4_UNIFIED_MLX_COMPAT_MODEL_FILE)
                .display()
        )
    })?;
    Ok(compat_dir)
}

fn gemma4_unified_compat_config(model_dir: &Path) -> Result<Option<Value>> {
    let config_path = model_dir.join("config.json");
    if !config_path.is_file() {
        return Ok(None);
    }

    let mut config: Value =
        serde_json::from_slice(&fs::read(&config_path).with_context(|| {
            format!("failed to read MLX model config {}", config_path.display())
        })?)?;
    if config.get("model_type").and_then(Value::as_str) != Some(GEMMA4_UNIFIED_MODEL_TYPE) {
        return Ok(None);
    }

    config["model_type"] = json!("gemma4");
    config["architectures"] = json!(["Gemma4ForCausalLM"]);
    config["model_file"] = json!(GEMMA4_UNIFIED_MLX_COMPAT_MODEL_FILE);
    if let Some(text_config) = config.get_mut("text_config")
        && text_config.is_object()
    {
        text_config["model_type"] = json!("gemma4_text");
    }
    Ok(Some(config))
}

fn mirror_mlx_compat_files(
    source_root: &Path,
    source_dir: &Path,
    target_root: &Path,
) -> Result<()> {
    for entry in fs::read_dir(source_dir)
        .with_context(|| format!("failed to read model directory {}", source_dir.display()))?
    {
        let entry = entry?;
        let source_path = entry.path();
        let relative_path = source_path
            .strip_prefix(source_root)
            .with_context(|| format!("failed to relativize {}", source_path.display()))?;
        if relative_path == Path::new("config.json") {
            continue;
        }

        let target_path = target_root.join(relative_path);
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            fs::create_dir_all(&target_path).with_context(|| {
                format!(
                    "failed to create MLX compatibility subdirectory {}",
                    target_path.display()
                )
            })?;
            mirror_mlx_compat_files(source_root, &source_path, target_root)?;
        } else if file_type.is_file() || source_path.is_file() {
            if let Some(parent) = target_path.parent() {
                fs::create_dir_all(parent)?;
            }
            link_or_copy_file(&source_path, &target_path)?;
        }
    }
    Ok(())
}

fn link_or_copy_file(source: &Path, target: &Path) -> Result<()> {
    if target.exists() {
        let source_len = fs::metadata(source)?.len();
        let target_len = fs::metadata(target)?.len();
        if source_len == target_len {
            return Ok(());
        }
        fs::remove_file(target)
            .with_context(|| format!("failed to replace {}", target.display()))?;
    }

    match fs::hard_link(source, target) {
        Ok(()) => Ok(()),
        Err(_) => {
            fs::copy(source, target).with_context(|| {
                format!(
                    "failed to copy {} to {}",
                    source.display(),
                    target.display()
                )
            })?;
            Ok(())
        }
    }
}

impl GenerationBackend for MlxBackend {
    fn prepare(&self, manifest: &ModelManifest) -> Result<()> {
        if !matches!(manifest.format, ModelFormat::Mlx | ModelFormat::SafeTensors) {
            bail!("mlx backend supports MLX or Hugging Face-style safetensors model directories");
        }
        Self::probe()?;
        resolve_mlx_model_dir(&self.store, manifest)?;
        Ok(())
    }

    fn generate(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> Result<GenerateResponse> {
        let mut command = self.command_for(manifest, &request)?;
        let started = Instant::now();
        let output = command
            .output()
            .with_context(|| format!("failed to execute {}", self.python.display()))?;
        if !output.status.success() {
            bail!(
                "mlx generation failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        let raw_text = String::from_utf8_lossy(&output.stdout).to_string();
        let cleaned = clean_mlx_generate_output(&raw_text);
        if cleaned.saw_think_block && cleaned.text.is_empty() {
            bail!(
                "mlx generation ended after hidden reasoning without producing assistant text; retry with a larger --max-tokens value"
            );
        }
        let mut text = cleaned.text;
        let finish_reason =
            mlx_finish_reason(&raw_text, request.max_tokens, &mut text, &request.stop);
        let elapsed = started.elapsed().as_secs_f64();
        Ok(external_response(
            request.prompt.as_str(),
            text.trim().to_string(),
            finish_reason,
            elapsed,
            mlx_generate_stats(&raw_text),
        ))
    }

    fn generate_stream(&self, manifest: ModelManifest, request: GenerateRequest) -> GenerateStream {
        let backend = self.clone();
        let (tx, rx) = mpsc::channel(16);
        tokio::task::spawn_blocking(move || {
            let result = backend.generate_streaming_subprocess(&manifest, request, tx.clone());
            if let Err(err) = result {
                let _ = tx.blocking_send(Err(format_error_chain(&err)));
            }
        });
        Box::pin(ReceiverStream::new(rx))
    }
}

impl GenerationBackend for MlxVlmBackend {
    fn prepare(&self, manifest: &ModelManifest) -> Result<()> {
        if !matches!(manifest.format, ModelFormat::Mlx | ModelFormat::SafeTensors) {
            bail!(
                "mlx-vlm backend supports MLX or Hugging Face-style safetensors model directories"
            );
        }
        Self::probe()?;
        original_mlx_model_dir(&self.store, manifest)?;
        Ok(())
    }

    fn generate(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> Result<GenerateResponse> {
        let mut command = self.command_for(manifest, &request, request.verbose)?;
        let program = command.get_program().to_string_lossy().to_string();
        let started = Instant::now();
        let output = command
            .output()
            .with_context(|| format!("failed to execute {program}"))?;
        if !output.status.success() {
            bail!(
                "mlx-vlm generation failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        let raw_text = String::from_utf8_lossy(&output.stdout).to_string();
        let cleaned = clean_mlx_generate_output(&raw_text);
        if cleaned.saw_think_block && cleaned.text.is_empty() {
            bail!(
                "mlx-vlm generation ended after hidden reasoning without producing assistant text; retry with a larger --max-tokens value"
            );
        }
        let mut text = cleaned.text;
        let finish_reason =
            mlx_finish_reason(&raw_text, request.max_tokens, &mut text, &request.stop);
        let elapsed = started.elapsed().as_secs_f64();
        Ok(external_response(
            request.prompt.as_str(),
            text.trim().to_string(),
            finish_reason,
            elapsed,
            mlx_generate_stats(&raw_text),
        ))
    }

    fn generate_stream(&self, manifest: ModelManifest, request: GenerateRequest) -> GenerateStream {
        let backend = self.clone();
        let (tx, rx) = mpsc::channel(16);
        tokio::task::spawn_blocking(move || {
            let result = backend.generate_streaming_subprocess(&manifest, request, tx.clone());
            if let Err(err) = result {
                let _ = tx.blocking_send(Err(format_error_chain(&err)));
            }
        });
        Box::pin(ReceiverStream::new(rx))
    }
}

impl MlxBackend {
    fn generate_streaming_subprocess(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
        tx: mpsc::Sender<Result<GenerateStreamEvent, String>>,
    ) -> Result<()> {
        let mut command = self.command_for(manifest, &request)?;
        command
            .arg("--verbose")
            .arg("True")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let program = command.get_program().to_string_lossy().to_string();
        let started = Instant::now();
        let mut child = command
            .spawn()
            .with_context(|| format!("failed to execute {program}"))?;
        let mut stdout = child
            .stdout
            .take()
            .context("failed to capture mlx generation stdout")?;
        let mut stderr = child
            .stderr
            .take()
            .context("failed to capture mlx generation stderr")?;
        let stderr_handle = thread::spawn(move || {
            let mut text = String::new();
            let _ = stderr.read_to_string(&mut text);
            text
        });

        let mut raw = Vec::new();
        let mut emitted = String::new();
        let mut buffer = [0u8; 4096];

        loop {
            match stdout.read(&mut buffer) {
                Ok(0) => break,
                Ok(n) => {
                    raw.extend_from_slice(&buffer[..n]);
                    send_mlx_stream_delta(&raw, &mut emitted, &request.stop, &tx)?;
                }
                Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                Err(err) => return Err(err.into()),
            }
        }

        let status = child.wait().context("failed waiting for mlx generation")?;
        let stderr = stderr_handle
            .join()
            .unwrap_or_else(|_| "failed to read mlx stderr".to_string());
        if !status.success() {
            bail!("mlx generation failed: {}", stderr.trim());
        }

        let raw_text = String::from_utf8_lossy(&raw);
        let cleaned = clean_mlx_generate_output(&raw_text);
        if cleaned.saw_think_block && cleaned.text.is_empty() {
            bail!(
                "mlx generation ended after hidden reasoning without producing assistant text; retry with a larger --max-tokens value"
            );
        }

        let mut final_text = cleaned.text;
        let finish_reason = mlx_finish_reason(
            &raw_text,
            request.max_tokens,
            &mut final_text,
            &request.stop,
        );
        send_remaining_stream_delta(&final_text, &mut emitted, &tx)?;

        let elapsed = started.elapsed().as_secs_f64();
        let response = external_response(
            request.prompt.as_str(),
            final_text.trim().to_string(),
            finish_reason,
            elapsed,
            mlx_generate_stats(&raw_text),
        );
        tx.blocking_send(Ok(GenerateStreamEvent::Done {
            finish_reason: response.finish_reason,
            prompt_tokens: response.prompt_tokens,
            completion_tokens: response.completion_tokens,
            timings: response.timings,
            backend_diagnostics: response.backend_diagnostics,
        }))
        .map_err(|err| anyhow!("stream receiver closed: {err}"))?;

        Ok(())
    }
}

impl MlxVlmBackend {
    fn generate_streaming_subprocess(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
        tx: mpsc::Sender<Result<GenerateStreamEvent, String>>,
    ) -> Result<()> {
        let mut command = self.command_for(manifest, &request, true)?;
        command.stdout(Stdio::piped()).stderr(Stdio::piped());

        let program = command.get_program().to_string_lossy().to_string();
        let started = Instant::now();
        let mut child = command
            .spawn()
            .with_context(|| format!("failed to execute {program}"))?;
        let mut stdout = child
            .stdout
            .take()
            .context("failed to capture mlx-vlm generation stdout")?;
        let mut stderr = child
            .stderr
            .take()
            .context("failed to capture mlx-vlm generation stderr")?;
        let stderr_handle = thread::spawn(move || {
            let mut text = String::new();
            let _ = stderr.read_to_string(&mut text);
            text
        });

        let mut raw = Vec::new();
        let mut emitted = String::new();
        let mut buffer = [0u8; 4096];

        loop {
            match stdout.read(&mut buffer) {
                Ok(0) => break,
                Ok(n) => {
                    raw.extend_from_slice(&buffer[..n]);
                    send_mlx_stream_delta(&raw, &mut emitted, &request.stop, &tx)?;
                }
                Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                Err(err) => return Err(err.into()),
            }
        }

        let status = child
            .wait()
            .context("failed waiting for mlx-vlm generation")?;
        let stderr = stderr_handle
            .join()
            .unwrap_or_else(|_| "failed to read mlx-vlm stderr".to_string());
        if !status.success() {
            bail!("mlx-vlm generation failed: {}", stderr.trim());
        }

        let raw_text = String::from_utf8_lossy(&raw);
        let cleaned = clean_mlx_generate_output(&raw_text);
        if cleaned.saw_think_block && cleaned.text.is_empty() {
            bail!(
                "mlx-vlm generation ended after hidden reasoning without producing assistant text; retry with a larger --max-tokens value"
            );
        }

        let mut final_text = cleaned.text;
        let finish_reason = mlx_finish_reason(
            &raw_text,
            request.max_tokens,
            &mut final_text,
            &request.stop,
        );
        send_remaining_stream_delta(&final_text, &mut emitted, &tx)?;

        let elapsed = started.elapsed().as_secs_f64();
        let response = external_response(
            request.prompt.as_str(),
            final_text.trim().to_string(),
            finish_reason,
            elapsed,
            mlx_generate_stats(&raw_text),
        );
        tx.blocking_send(Ok(GenerateStreamEvent::Done {
            finish_reason: response.finish_reason,
            prompt_tokens: response.prompt_tokens,
            completion_tokens: response.completion_tokens,
            timings: response.timings,
            backend_diagnostics: response.backend_diagnostics,
        }))
        .map_err(|err| anyhow!("stream receiver closed: {err}"))?;

        Ok(())
    }
}

#[cfg(feature = "llama-cpp")]
fn sampler_for(request: &GenerateRequest) -> StandardSampler {
    let temperature = request.temperature.unwrap_or(0.8);
    if temperature <= 0.0 {
        return StandardSampler::new_greedy();
    }

    let mut stages = Vec::new();
    stages.push(SamplerStage::Temperature(temperature as f32));
    if let Some(top_p) = request.top_p
        && top_p > 0.0
        && top_p < 1.0
    {
        stages.push(SamplerStage::TopP(top_p as f32));
    }
    StandardSampler::new_softmax(stages, 1)
}

#[cfg(feature = "llama-cpp")]
fn send_text_chunk(
    tx: &Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
    chunk: String,
) -> Result<()> {
    if chunk.is_empty() {
        return Ok(());
    }
    if let Some(tx) = tx {
        tx.blocking_send(Ok(GenerateStreamEvent::TextChunk(chunk)))
            .map_err(|err| anyhow!("stream receiver closed: {err}"))?;
    }
    Ok(())
}

#[cfg(feature = "llama-cpp")]
fn shared_prefix_len(previous: &[Token], next: &[Token]) -> usize {
    previous
        .iter()
        .zip(next)
        .position(|(left, right)| left != right)
        .unwrap_or(previous.len().min(next.len()))
}

#[cfg(feature = "llama-cpp")]
fn warmup_session(session: &mut LlamaSession) {
    if env_false("WERK_LLAMA_WARMUP") {
        return;
    }

    if session.advance_context(b" ").is_ok() {
        let _ = session.truncate_context(0);
    }
}

fn external_response(
    prompt: &str,
    text: String,
    finish_reason: String,
    elapsed: f64,
    stats: MlxGenerateStats,
) -> GenerateResponse {
    let prompt_tokens = stats
        .prompt_tokens
        .unwrap_or_else(|| estimate_tokens(prompt));
    let completion_tokens = stats
        .generation_tokens
        .unwrap_or_else(|| estimate_tokens(&text));
    let prompt_seconds =
        duration_from_rate(prompt_tokens, stats.prompt_tokens_per_second).unwrap_or(0.0);
    let decode_seconds = duration_from_rate(completion_tokens, stats.generation_tokens_per_second)
        .unwrap_or(elapsed);
    let load_seconds = if stats.has_rates() {
        (elapsed - prompt_seconds - decode_seconds).max(0.0)
    } else {
        0.0
    };

    GenerateResponse {
        prompt_tokens,
        completion_tokens,
        text,
        finish_reason,
        timings: GenerationTimings {
            load_seconds,
            warmup_seconds: 0.0,
            first_token_seconds: 0.0,
            prompt_seconds,
            decode_seconds,
            total_seconds: elapsed,
        },
        backend_diagnostics: stats.native_lines,
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
struct MlxGenerateStats {
    prompt_tokens: Option<usize>,
    prompt_tokens_per_second: Option<f64>,
    generation_tokens: Option<usize>,
    generation_tokens_per_second: Option<f64>,
    native_lines: Vec<String>,
}

impl MlxGenerateStats {
    fn has_rates(&self) -> bool {
        self.prompt_tokens_per_second.is_some() || self.generation_tokens_per_second.is_some()
    }
}

fn duration_from_rate(tokens: usize, tokens_per_second: Option<f64>) -> Option<f64> {
    let tokens_per_second = tokens_per_second?;
    if tokens == 0 || tokens_per_second <= 0.0 {
        return None;
    }
    Some(tokens as f64 / tokens_per_second)
}

fn truncate_at_stop(text: &mut String, stops: &[String]) -> String {
    if let Some(index) = first_stop_index(text, stops) {
        text.truncate(index);
        "stop".to_string()
    } else {
        "length".to_string()
    }
}

fn first_stop_index(text: &str, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop))
        .min()
}

fn send_mlx_stream_delta(
    raw: &[u8],
    emitted: &mut String,
    stops: &[String],
    tx: &mpsc::Sender<Result<GenerateStreamEvent, String>>,
) -> Result<String> {
    let raw_text = String::from_utf8_lossy(raw);
    let cleaned = clean_mlx_generate_output(&raw_text);
    let mut visible = cleaned.text;
    let finish_reason = truncate_at_stop(&mut visible, stops);
    send_remaining_stream_delta(&visible, emitted, tx)?;
    Ok(finish_reason)
}

fn send_remaining_stream_delta(
    visible: &str,
    emitted: &mut String,
    tx: &mpsc::Sender<Result<GenerateStreamEvent, String>>,
) -> Result<()> {
    if visible.len() <= emitted.len() || !visible.starts_with(emitted.as_str()) {
        return Ok(());
    }

    let start = emitted.len();
    let delta = &visible[start..];
    if !delta.is_empty() {
        tx.blocking_send(Ok(GenerateStreamEvent::TextChunk(delta.to_string())))
            .map_err(|err| anyhow!("stream receiver closed: {err}"))?;
    }
    emitted.push_str(delta);
    Ok(())
}

fn mlx_finish_reason(
    raw_output: &str,
    max_tokens: usize,
    text: &mut String,
    stops: &[String],
) -> String {
    let stop_reason = truncate_at_stop(text, stops);
    if stop_reason == "stop" {
        return stop_reason;
    }

    match mlx_generation_tokens(raw_output) {
        Some(tokens) if tokens < max_tokens => "stop".to_string(),
        _ => "length".to_string(),
    }
}

fn mlx_generation_tokens(output: &str) -> Option<usize> {
    mlx_generate_stats(output).generation_tokens
}

fn mlx_generate_stats(output: &str) -> MlxGenerateStats {
    let mut stats = MlxGenerateStats::default();
    for line in output.lines() {
        let line = line.trim();
        if let Some((tokens, rate)) = parse_mlx_stat_line(line, "Prompt:") {
            stats.prompt_tokens = Some(tokens);
            stats.prompt_tokens_per_second = rate;
            stats.native_lines.push(line.to_string());
        } else if let Some((tokens, rate)) = parse_mlx_stat_line(line, "Generation:") {
            stats.generation_tokens = Some(tokens);
            stats.generation_tokens_per_second = rate;
            stats.native_lines.push(line.to_string());
        } else if line.starts_with("Peak memory:") {
            stats.native_lines.push(line.to_string());
        }
    }
    stats
}

fn parse_mlx_stat_line(line: &str, prefix: &str) -> Option<(usize, Option<f64>)> {
    let rest = line.strip_prefix(prefix)?.trim_start();
    let (count_text, rate_text) = rest.split_once(',').unwrap_or((rest, ""));
    let tokens = count_text.split_whitespace().next()?.parse().ok()?;
    let rate = rate_text
        .trim()
        .split_whitespace()
        .next()
        .and_then(|value| value.parse().ok());
    Some((tokens, rate))
}

fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

fn seed_u32(seed: u64) -> u32 {
    u32::try_from(seed).unwrap_or_else(|_| {
        let high = (seed >> 32) as u32;
        let low = seed as u32;
        high ^ low
    })
}

#[cfg(feature = "llama-cpp")]
fn env_u32(name: &str) -> Option<u32> {
    env::var(name).ok()?.parse().ok()
}

#[cfg(feature = "llama-cpp")]
fn env_false(name: &str) -> bool {
    matches!(
        env::var(name).ok().as_deref(),
        Some("0" | "false" | "False" | "FALSE" | "no" | "No" | "NO" | "off" | "Off" | "OFF")
    )
}

fn format_float(value: f64) -> String {
    let mut text = format!("{value:.6}");
    while text.contains('.') && text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.pop();
    }
    text
}

fn backend_program(env_name: &str, default_name: &str) -> PathBuf {
    if let Ok(path) = env::var(env_name)
        && !path.trim().is_empty()
    {
        return PathBuf::from(path);
    }
    if let Ok(current_exe) = env::current_exe()
        && let Some(dir) = current_exe.parent()
    {
        let sibling = dir.join(default_name);
        if sibling.is_file() {
            return sibling;
        }
    }
    PathBuf::from(default_name)
}

fn sibling_program(program: &PathBuf, sibling_name: &str) -> Option<PathBuf> {
    program
        .parent()
        .map(|dir| dir.join(sibling_name))
        .filter(|path| path.is_file())
}

fn find_program_in_path(name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(name);
    if path.components().count() > 1 && path.is_file() {
        return Some(path);
    }
    let path_var = env::var_os("PATH")?;
    for dir in env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn python_module_command(python: &PathBuf, module: &str) -> Command {
    let mut command = Command::new(python);
    command.arg("-m").arg(module);
    command
}

fn python_module_subcommand(python: &PathBuf, module: &str, subcommand: &str) -> Command {
    let mut command = Command::new(python);
    command.arg("-m").arg(module).arg(subcommand);
    command
}

fn mlx_generate_program() -> &'static str {
    if cfg!(windows) {
        "mlx_lm.generate.exe"
    } else {
        "mlx_lm.generate"
    }
}

fn mlx_vlm_generate_program() -> &'static str {
    if cfg!(windows) {
        "mlx_vlm.generate.exe"
    } else {
        "mlx_vlm.generate"
    }
}

fn format_error_chain(err: &anyhow::Error) -> String {
    let messages = err.chain().map(ToString::to_string).collect::<Vec<_>>();
    messages.join(": ")
}

fn default_python() -> &'static str {
    if cfg!(windows) {
        "python.exe"
    } else {
        "python3"
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CleanedMlxOutput {
    text: String,
    saw_think_block: bool,
    unclosed_think_block: bool,
}

fn clean_mlx_generate_output(output: &str) -> CleanedMlxOutput {
    let mut cleaned = Vec::new();
    let mut saw_think_block = false;
    let mut in_think_block = false;
    let mut in_hidden_channel = false;
    let lines = output.lines().collect::<Vec<_>>();
    let output_ends_with_newline = output.ends_with('\n') || output.ends_with('\r');

    for (index, line) in lines.iter().enumerate() {
        let mut rest = *line;
        let mut visible = String::new();
        let incomplete_last_line = index + 1 == lines.len() && !output_ends_with_newline;

        loop {
            if in_hidden_channel {
                if let Some(end) = rest.find("<channel|>") {
                    rest = &rest[end + "<channel|>".len()..];
                    in_hidden_channel = false;
                    continue;
                }
                break;
            }

            if in_think_block {
                if let Some(end) = rest.find("</think>") {
                    rest = &rest[end + "</think>".len()..];
                    in_think_block = false;
                    continue;
                }
                break;
            }

            if let Some(start) = rest.find("<think>") {
                saw_think_block = true;
                visible.push_str(&rest[..start]);
                rest = &rest[start + "<think>".len()..];
                in_think_block = true;
                continue;
            }

            if let Some((hidden, remainder)) = parse_mlx_channel_marker(rest.trim_start()) {
                if hidden {
                    saw_think_block = true;
                    in_hidden_channel = true;
                    rest = remainder;
                    continue;
                }
                in_hidden_channel = false;
                rest = remainder;
                continue;
            }

            visible.push_str(rest);
            break;
        }

        if incomplete_last_line {
            trim_hidden_tag_prefix_suffix(&mut visible);
        }

        let visible = visible.trim();
        if visible.is_empty()
            || is_mlx_noise_line(visible)
            || (incomplete_last_line && is_possible_mlx_noise_prefix(visible))
        {
            continue;
        }
        cleaned.push(visible.to_string());
    }

    CleanedMlxOutput {
        text: cleaned.join("\n").trim().to_string(),
        saw_think_block,
        unclosed_think_block: in_think_block || in_hidden_channel,
    }
}

fn parse_mlx_channel_marker(line: &str) -> Option<(bool, &str)> {
    for prefix in ["<|channel>", "<|channel|>"] {
        let Some(rest) = line.strip_prefix(prefix) else {
            continue;
        };
        let name_end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        let channel = &rest[..name_end];
        let remainder = &rest[name_end..];
        return match channel {
            "thought" | "analysis" => Some((true, remainder)),
            "final" | "answer" => Some((false, remainder)),
            _ => None,
        };
    }

    None
}

fn is_mlx_noise_line(line: &str) -> bool {
    line == "=========="
        || line.starts_with("Prompt:")
        || line.starts_with("Files:")
        || line.starts_with("Generation:")
        || line.starts_with("Peak memory:")
        || line.starts_with("<|image|>")
        || line.starts_with("<|audio|>")
        || line.starts_with("<|video|>")
        || line.starts_with("<|turn>")
        || line.starts_with("<turn|>")
        || line.starts_with("assistant:<turn|>")
        || line == "Assistant:"
        || line.starts_with("Calling `python -m mlx_lm.generate")
        || line.starts_with("Calling `python -m mlx_vlm.generate")
        || line.contains("directly is deprecated")
}

fn is_possible_mlx_noise_prefix(line: &str) -> bool {
    [
        "==========",
        "Prompt:",
        "Files:",
        "Generation:",
        "Peak memory:",
        "<|image|>",
        "<|audio|>",
        "<|video|>",
        "<|turn>",
        "<turn|>",
        "assistant:<turn|>",
        "Assistant:",
        "Calling `python -m mlx_lm.generate",
        "Calling `python -m mlx_vlm.generate",
    ]
    .iter()
    .any(|noise| noise.starts_with(line))
}

fn trim_hidden_tag_prefix_suffix(visible: &mut String) {
    for tag in [
        "<think>",
        "</think>",
        "<|channel>thought",
        "<|channel>analysis",
        "<|channel>final",
        "<|channel|>thought",
        "<|channel|>analysis",
        "<|channel|>final",
        "<channel|>",
        "<|image|>",
        "<|audio|>",
        "<|video|>",
        "<|turn>",
        "<turn|>",
    ] {
        let max_prefix_len = visible.len().min(tag.len().saturating_sub(1));
        for prefix_len in (1..=max_prefix_len).rev() {
            if visible.ends_with(&tag[..prefix_len]) {
                let new_len = visible.len() - prefix_len;
                visible.truncate(new_len);
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::StreamGranularity;
    use crate::model_store::ModelSource;

    #[test]
    fn stop_detection_uses_first_matching_stop() {
        let stops = vec!["</s>".to_string(), "END".to_string()];
        assert_eq!(first_stop_index("hello END </s>", &stops), Some(6));
    }

    #[test]
    fn stop_truncation_marks_finish_reason() {
        let mut text = "hello</s>ignored".to_string();
        let reason = truncate_at_stop(&mut text, &["</s>".to_string()]);
        assert_eq!(text, "hello");
        assert_eq!(reason, "stop");
    }

    #[test]
    fn large_seed_is_folded_into_u32() {
        assert_eq!(seed_u32(42), 42);
        assert_eq!(seed_u32(0x0000_0001_0000_0002), 3);
    }

    #[test]
    fn mlx_output_cleaner_removes_deprecation_warning() {
        let output = "Calling `python -m mlx_lm.generate...` directly is deprecated.\nHello.";

        assert_eq!(clean_mlx_generate_output(output).text, "Hello.");
    }

    #[test]
    fn gemma4_unified_config_is_detected_and_patched_for_mlx() {
        let root = test_root("gemma4-unified-detect");
        let model_dir = root.join("files");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(
            model_dir.join("config.json"),
            r#"{"model_type":"gemma4_unified","architectures":["Gemma4UnifiedForConditionalGeneration"],"text_config":{"model_type":"gemma4_unified_text"}}"#,
        )
        .unwrap();

        let patched = gemma4_unified_compat_config(&model_dir).unwrap().unwrap();

        assert_eq!(
            patched.get("model_type").and_then(Value::as_str),
            Some("gemma4")
        );
        assert_eq!(
            patched
                .get("architectures")
                .and_then(Value::as_array)
                .and_then(|items| items.first())
                .and_then(Value::as_str),
            Some("Gemma4ForCausalLM")
        );
        assert_eq!(
            patched
                .get("text_config")
                .and_then(|value| value.get("model_type"))
                .and_then(Value::as_str),
            Some("gemma4_text")
        );
        assert_eq!(
            patched.get("model_file").and_then(Value::as_str),
            Some(GEMMA4_UNIFIED_MLX_COMPAT_MODEL_FILE)
        );
    }

    #[test]
    fn gemma4_unified_compat_view_preserves_original_and_links_files() {
        let store = test_store("gemma4-unified-view");
        let manifest = test_manifest("Gemma4-12B-JANG", "gemma4_unified");
        let model_dir = store.model_dir(&manifest.id).join("files");
        fs::create_dir_all(model_dir.join("nested")).unwrap();
        let original_config = r#"{"model_type":"gemma4_unified","architectures":["Gemma4UnifiedForConditionalGeneration"],"text_config":{"model_type":"gemma4_unified_text"}}"#;
        fs::write(model_dir.join("config.json"), original_config).unwrap();
        fs::write(model_dir.join("tokenizer.json"), "{}").unwrap();
        fs::write(
            model_dir.join("model-00001-of-00001.safetensors"),
            "weights",
        )
        .unwrap();
        fs::write(
            model_dir.join("nested").join("chat_template.jinja"),
            "{{ prompt }}",
        )
        .unwrap();

        let compat_dir = ensure_gemma4_unified_compat_dir(&store, &manifest, &model_dir).unwrap();

        assert_eq!(
            fs::read_to_string(model_dir.join("config.json")).unwrap(),
            original_config
        );
        let compat_config: Value =
            serde_json::from_slice(&fs::read(compat_dir.join("config.json")).unwrap()).unwrap();
        assert_eq!(
            compat_config.get("model_type").and_then(Value::as_str),
            Some("gemma4")
        );
        assert_eq!(
            compat_config.get("model_file").and_then(Value::as_str),
            Some(GEMMA4_UNIFIED_MLX_COMPAT_MODEL_FILE)
        );
        assert!(compat_dir.join("tokenizer.json").is_file());
        assert!(
            compat_dir
                .join("model-00001-of-00001.safetensors")
                .is_file()
        );
        assert!(
            compat_dir
                .join(GEMMA4_UNIFIED_MLX_COMPAT_MODEL_FILE)
                .is_file()
        );
        assert!(
            fs::read_to_string(compat_dir.join(GEMMA4_UNIFIED_MLX_COMPAT_MODEL_FILE))
                .unwrap()
                .contains("vision_embedder.")
        );
        assert!(
            compat_dir
                .join("nested")
                .join("chat_template.jinja")
                .is_file()
        );
    }

    #[test]
    fn normal_mlx_model_uses_original_model_directory() {
        let store = test_store("normal-mlx-dir");
        let manifest = test_manifest("Qwen3-4B-mlx", "qwen3");
        let model_dir = store.model_dir(&manifest.id).join("files");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("config.json"), r#"{"model_type":"qwen3"}"#).unwrap();

        let resolved = resolve_mlx_model_dir(&store, &manifest).unwrap();

        assert_eq!(resolved, model_dir);
        assert!(!store.artifacts_dir(&manifest.id).exists());
    }

    #[test]
    fn gemma4_unified_command_disables_template_thinking() {
        let store = test_store("gemma4-unified-command");
        let manifest = test_manifest("Gemma4-12B-JANG", "gemma4_unified");
        let model_dir = store.model_dir(&manifest.id).join("files");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(
            model_dir.join("config.json"),
            r#"{"model_type":"gemma4_unified","architectures":["Gemma4UnifiedForConditionalGeneration"],"text_config":{"model_type":"gemma4_unified_text"}}"#,
        )
        .unwrap();

        let backend = MlxBackend {
            store,
            python: PathBuf::from("python3"),
            module: "mlx_lm.generate".to_string(),
        };
        let command = backend
            .command_for(&manifest, &test_request("Hello"))
            .unwrap();
        let args = command
            .get_args()
            .map(|arg| arg.to_string_lossy().to_string())
            .collect::<Vec<_>>();
        let config_index = args
            .iter()
            .position(|arg| arg == "--chat-template-config")
            .expect("Gemma4 unified compat command should set chat template config");

        assert_eq!(
            args.get(config_index + 1).map(String::as_str),
            Some(r#"{"enable_thinking":false}"#)
        );
    }

    #[test]
    fn mlx_vlm_command_uses_original_model_dir_and_single_image_list() {
        let store = test_store("mlx-vlm-command");
        let manifest = test_manifest("Gemma4-12B-JANG", "gemma4_unified");
        let model_dir = store.model_dir(&manifest.id).join("files");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(
            model_dir.join("config.json"),
            r#"{"model_type":"gemma4_unified"}"#,
        )
        .unwrap();

        let backend = MlxVlmBackend {
            store,
            python: PathBuf::from("python3"),
            module: "mlx_vlm".to_string(),
        };
        let mut request = test_request("Describe this image.");
        request.image_urls = vec!["one.png".to_string(), "two.png".to_string()];
        request.verbose = false;

        let command = backend.command_for(&manifest, &request, false).unwrap();
        let args = command
            .get_args()
            .map(|arg| arg.to_string_lossy().to_string())
            .collect::<Vec<_>>();

        let model_index = args.iter().position(|arg| arg == "--model").unwrap();
        assert_eq!(
            args.get(model_index + 1).map(PathBuf::from),
            Some(model_dir)
        );
        assert!(args.iter().any(|arg| arg == "--no-verbose"));
        let image_index = args.iter().position(|arg| arg == "--image").unwrap();
        assert_eq!(
            args.get(image_index + 1).map(String::as_str),
            Some("one.png")
        );
        assert_eq!(
            args.get(image_index + 2).map(String::as_str),
            Some("two.png")
        );
        assert_eq!(args.iter().filter(|arg| *arg == "--image").count(), 1);
    }

    #[test]
    fn mlx_output_cleaner_removes_separator_lines() {
        let output = "==========\nHello.\n==========";

        assert_eq!(clean_mlx_generate_output(output).text, "Hello.");
    }

    #[test]
    fn mlx_output_cleaner_removes_multiline_think_blocks() {
        let output = "<think>\nreasoning here\nstill reasoning\n</think>\nAnswer.";

        assert_eq!(clean_mlx_generate_output(output).text, "Answer.");
    }

    #[test]
    fn mlx_output_cleaner_removes_stats_lines() {
        let output = "Answer.\nPrompt: 30 tokens, 332.363 tokens-per-sec\nGeneration: 256 tokens, 98.430 tokens-per-sec\nPeak memory: 2.310 GB";

        assert_eq!(clean_mlx_generate_output(output).text, "Answer.");
    }

    #[test]
    fn mlx_output_cleaner_preserves_answer_after_thinking_block() {
        let output = "==========\n<think>\nreasoning here\n</think>\n\nRust is a systems programming language.\n==========\nPrompt: 30 tokens, 332.363 tokens-per-sec\nGeneration: 256 tokens, 98.430 tokens-per-sec\nPeak memory: 2.310 GB";

        assert_eq!(
            clean_mlx_generate_output(output).text,
            "Rust is a systems programming language."
        );
    }

    #[test]
    fn mlx_output_cleaner_preserves_answer_after_gemma_thought_channel() {
        let output = "<|channel>thought\nreasoning here\n<channel|>\n<|channel>final\nRust is fast and memory-safe.";

        let cleaned = clean_mlx_generate_output(output);

        assert_eq!(cleaned.text, "Rust is fast and memory-safe.");
        assert!(cleaned.saw_think_block);
        assert!(!cleaned.unclosed_think_block);
    }

    #[test]
    fn mlx_output_cleaner_removes_mlx_vlm_prompt_template_lines() {
        let output = "==========\nPrompt: <|image|>user: What color is this image?\n<|image|>user: What color is this image?\nassistant:<turn|>\n<|turn>model\nThe image is a solid dark red color.\n==========\nPrompt: 278 tokens, 142.909 tokens-per-sec\nGeneration: 14 tokens, 11.134 tokens-per-sec\nPeak memory: 15.366 GB";

        assert_eq!(
            clean_mlx_generate_output(output).text,
            "The image is a solid dark red color."
        );
    }

    #[test]
    fn mlx_output_cleaner_reports_gemma_thought_channel_without_answer() {
        let output = "<|channel>thought\nstill reasoning";

        let cleaned = clean_mlx_generate_output(output);

        assert!(cleaned.text.is_empty());
        assert!(cleaned.saw_think_block);
        assert!(cleaned.unclosed_think_block);
    }

    #[test]
    fn mlx_output_cleaner_leaves_ordinary_text_unchanged() {
        let output = "First line.\nSecond line.";

        assert_eq!(clean_mlx_generate_output(output).text, output);
    }

    #[test]
    fn mlx_output_cleaner_reports_unclosed_think_block() {
        let output =
            "==========\n<think>\nstill reasoning\nGeneration: 256 tokens, 97.366 tokens-per-sec";
        let cleaned = clean_mlx_generate_output(output);

        assert!(cleaned.text.is_empty());
        assert!(cleaned.saw_think_block);
        assert!(cleaned.unclosed_think_block);
    }

    #[test]
    fn mlx_output_cleaner_reports_closed_think_without_answer() {
        let output = "<think>\nreasoning here\n</think>\n==========\nGeneration: 256 tokens, 97.366 tokens-per-sec";
        let cleaned = clean_mlx_generate_output(output);

        assert!(cleaned.text.is_empty());
        assert!(cleaned.saw_think_block);
        assert!(!cleaned.unclosed_think_block);
    }

    #[test]
    fn mlx_output_cleaner_holds_partial_noise_prefixes() {
        let output = "Answer so far\n====";

        assert_eq!(clean_mlx_generate_output(output).text, "Answer so far");

        let output = "Answer so far\nGener";

        assert_eq!(clean_mlx_generate_output(output).text, "Answer so far");
    }

    #[test]
    fn mlx_output_cleaner_holds_partial_think_tag() {
        let output = "Answer before tag <thi";

        assert_eq!(clean_mlx_generate_output(output).text, "Answer before tag");
    }

    #[test]
    fn mlx_generation_token_count_is_parsed_from_stats() {
        let output = "Answer\n==========\nPrompt: 5 tokens, 1.000 tokens-per-sec\nGeneration: 511 tokens, 10.000 tokens-per-sec\nPeak memory: 2.000 GB";

        assert_eq!(mlx_generation_tokens(output), Some(511));
    }

    #[test]
    fn mlx_generate_stats_parse_prompt_and_generation_rates() {
        let output = "Answer\n==========\nPrompt: 30 tokens, 332.363 tokens-per-sec\nGeneration: 256 tokens, 98.430 tokens-per-sec\nPeak memory: 2.310 GB";
        let stats = mlx_generate_stats(output);

        assert_eq!(stats.prompt_tokens, Some(30));
        assert_eq!(stats.generation_tokens, Some(256));
        assert_eq!(stats.prompt_tokens_per_second, Some(332.363));
        assert_eq!(stats.generation_tokens_per_second, Some(98.430));
    }

    #[test]
    fn mlx_external_response_uses_native_stats_for_verbose_timings() {
        let stats = mlx_generate_stats(
            "Prompt: 30 tokens, 300.000 tokens-per-sec\nGeneration: 256 tokens, 128.000 tokens-per-sec",
        );
        let response = external_response(
            "hello world",
            "Visible answer.".to_string(),
            "stop".to_string(),
            5.0,
            stats,
        );

        assert_eq!(response.prompt_tokens, 30);
        assert_eq!(response.completion_tokens, 256);
        assert!((response.timings.prompt_seconds - 0.1).abs() < 1e-9);
        assert!((response.timings.decode_seconds - 2.0).abs() < 1e-9);
        assert!((response.timings.load_seconds - 2.9).abs() < 1e-9);
        assert_eq!(response.timings.total_seconds, 5.0);
    }

    #[test]
    fn mlx_finish_reason_uses_generation_token_count() {
        let mut stopped = "Answer".to_string();
        let stop_output = "Answer\nGeneration: 12 tokens, 10.000 tokens-per-sec";
        assert_eq!(
            mlx_finish_reason(stop_output, 2048, &mut stopped, &[]),
            "stop"
        );

        let mut limited = "Answer".to_string();
        let limited_output = "Answer\nGeneration: 2048 tokens, 10.000 tokens-per-sec";
        assert_eq!(
            mlx_finish_reason(limited_output, 2048, &mut limited, &[]),
            "length"
        );
    }

    fn test_store(name: &str) -> ModelStore {
        ModelStore::resolve(Some(test_root(name).join("store"))).unwrap()
    }

    fn test_root(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        env::temp_dir().join(format!(
            "werk-external-{name}-{}-{nanos}",
            std::process::id()
        ))
    }

    fn test_manifest(id: &str, architecture: &str) -> ModelManifest {
        ModelManifest {
            id: id.to_string(),
            source: ModelSource::LocalPath {
                path: "test".to_string(),
            },
            format: ModelFormat::Mlx,
            architecture: Some(architecture.to_string()),
            tokenizer_path: Some("files/tokenizer.json".to_string()),
            config_path: Some("files/config.json".to_string()),
            model_path: Some("files/model-00001-of-00001.safetensors".to_string()),
            backend: "mlx".to_string(),
            created_unix: 1,
            files: Vec::new(),
            artifacts: Vec::new(),
        }
    }

    fn test_request(prompt: &str) -> GenerateRequest {
        GenerateRequest {
            prompt: prompt.to_string(),
            messages: Vec::new(),
            image_urls: Vec::new(),
            max_tokens: 128,
            temperature: None,
            top_p: None,
            stop: Vec::new(),
            seed: None,
            stream_granularity: StreamGranularity::Token,
            verbose: false,
            debug: false,
        }
    }
}
