use anyhow::{Context, Result, anyhow, bail};
#[cfg(feature = "llama-cpp")]
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use std::{env, path::PathBuf, process::Command, time::Instant};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

#[cfg(feature = "llama-cpp")]
use super::ChatGenerationSession;
use super::{
    GenerateRequest, GenerateResponse, GenerateStream, GenerateStreamEvent, GenerationBackend,
    GenerationTimings,
};
use crate::model_store::{ModelFormat, ModelManifest, ModelStore};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LlamaCppMode {
    Cuda,
    Vulkan,
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

impl LlamaCppMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Cuda => "llama-cpp-cuda",
            Self::Vulkan => "llama-cpp-vulkan",
            Self::Cpu => "llama-cpp-cpu",
        }
    }

    #[cfg(feature = "llama-cpp")]
    fn display_name(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Vulkan => "Vulkan",
            Self::Cpu => "CPU",
        }
    }

    #[cfg(feature = "llama-cpp")]
    fn gpu_layers(self) -> u32 {
        match self {
            Self::Cpu => 0,
            Self::Cuda | Self::Vulkan => 999,
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
            Self::Cuda => cfg!(all(feature = "llama-cpp", feature = "cuda")),
            Self::Vulkan => cfg!(all(feature = "llama-cpp", feature = "vulkan")),
        }
    }

    fn unavailable_message(self) -> String {
        match self {
            Self::Cuda => {
                "llama.cpp CUDA backend is not compiled into this binary; build/install with --features cuda".to_string()
            }
            Self::Vulkan => {
                "llama.cpp Vulkan backend is not compiled into this binary; build/install with --features vulkan".to_string()
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
        let model_dir = self.store.model_dir(&manifest.id).join("files");
        if !model_dir.is_dir() {
            bail!(
                "model files directory does not exist: {}",
                model_dir.display()
            );
        }

        let mut command = Command::new(&self.python);
        command
            .arg("-m")
            .arg(&self.module)
            .arg("--model")
            .arg(model_dir)
            .arg("--prompt")
            .arg(&request.prompt)
            .arg("--max-tokens")
            .arg(request.max_tokens.to_string());
        if let Some(temperature) = request.temperature {
            command.arg("--temp").arg(format_float(temperature));
        }
        for image in &request.image_urls {
            command.arg("--image").arg(image);
        }
        Ok(command)
    }
}

impl GenerationBackend for MlxBackend {
    fn prepare(&self, manifest: &ModelManifest) -> Result<()> {
        if !matches!(manifest.format, ModelFormat::Mlx | ModelFormat::SafeTensors) {
            bail!("mlx backend supports MLX or Hugging Face-style safetensors model directories");
        }
        Self::probe()?;
        let model_dir = self.store.model_dir(&manifest.id).join("files");
        if !model_dir.is_dir() {
            bail!(
                "model files directory does not exist: {}",
                model_dir.display()
            );
        }
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
        let mut text = String::from_utf8_lossy(&output.stdout).to_string();
        let finish_reason = truncate_at_stop(&mut text, &request.stop);
        let elapsed = started.elapsed().as_secs_f64();
        Ok(external_response(
            request.prompt.as_str(),
            text.trim().to_string(),
            finish_reason,
            elapsed,
        ))
    }

    fn generate_stream(&self, manifest: ModelManifest, request: GenerateRequest) -> GenerateStream {
        let backend = self.clone();
        let (tx, rx) = mpsc::channel(16);
        tokio::task::spawn_blocking(move || {
            let result = backend.generate(&manifest, request).and_then(|response| {
                if !response.text.is_empty() {
                    let _ =
                        tx.blocking_send(Ok(GenerateStreamEvent::TextChunk(response.text.clone())));
                }
                tx.blocking_send(Ok(GenerateStreamEvent::Done {
                    finish_reason: response.finish_reason,
                    prompt_tokens: response.prompt_tokens,
                    completion_tokens: response.completion_tokens,
                    timings: response.timings,
                }))
                .map_err(|err| anyhow!("stream receiver closed: {err}"))
            });
            if let Err(err) = result {
                let _ = tx.blocking_send(Err(format_error_chain(&err)));
            }
        });
        Box::pin(ReceiverStream::new(rx))
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
) -> GenerateResponse {
    GenerateResponse {
        prompt_tokens: estimate_tokens(prompt),
        completion_tokens: estimate_tokens(&text),
        text,
        finish_reason,
        timings: GenerationTimings {
            load_seconds: 0.0,
            warmup_seconds: 0.0,
            first_token_seconds: 0.0,
            prompt_seconds: 0.0,
            decode_seconds: elapsed,
            total_seconds: elapsed,
        },
    }
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

fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

#[cfg(any(feature = "llama-cpp", test))]
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
