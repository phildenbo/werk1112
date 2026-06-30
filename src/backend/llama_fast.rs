#[cfg(feature = "llama-fast")]
mod imp {
    use anyhow::{Context, Result, anyhow, bail};
    use std::{
        collections::HashMap,
        ffi::CString,
        os::raw::{c_char, c_void},
        path::PathBuf,
        sync::{Arc, Mutex, Once},
        time::Instant,
    };
    #[cfg(unix)]
    use std::{io::Write, os::fd::RawFd};
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    use crate::model_store::{ModelFormat, ModelManifest, ModelStore};

    use crate::backend::{
        ChatGenerationSession, GenerateRequest, GenerateResponse, GenerateStream,
        GenerateStreamEvent, GenerationBackend, GenerationTimings, LlamaCppMode, LlamaKvCacheType,
        LlamaRuntimeOptions,
    };

    use llama_cpp_sys::{
        ggml_numa_strategy, ggml_type, llama_batch, llama_batch_free, llama_batch_init,
        llama_context, llama_context_default_params, llama_decode, llama_free, llama_free_model,
        llama_get_logits_ith, llama_get_timings, llama_kv_cache_seq_rm, llama_load_model_from_file,
        llama_log_set, llama_model, llama_model_default_params, llama_n_batch, llama_n_ctx,
        llama_n_ubatch, llama_n_vocab, llama_new_context_with_model, llama_numa_init, llama_pos,
        llama_reset_timings, llama_sample_temp, llama_sample_token, llama_sample_token_greedy,
        llama_sample_top_p, llama_token, llama_token_data, llama_token_data_array, llama_token_eos,
        llama_token_eot, llama_token_to_piece, llama_tokenize,
    };

    const DEFAULT_N_CTX: u32 = 0;
    const DEFAULT_N_BATCH: u32 = 2048;
    const DEFAULT_N_UBATCH: u32 = 512;

    #[derive(Clone)]
    pub struct LlamaFastBackend {
        store: ModelStore,
        mode: LlamaCppMode,
        runtime_options: LlamaRuntimeOptions,
        models: Arc<Mutex<HashMap<String, Arc<LlamaFastModel>>>>,
    }

    struct LlamaFastModel {
        ptr: *mut llama_model,
        vocab: usize,
        eos: llama_token,
        eot: llama_token,
    }

    struct LlamaFastContext {
        model: Arc<LlamaFastModel>,
        ctx: *mut llama_context,
        batch: FastBatch,
        params: FastSessionParams,
        tokens: Vec<llama_token>,
        candidates: Vec<llama_token_data>,
        decoder: Utf8PieceDecoder,
        last_batch_size: usize,
        pending_warmup_seconds: f64,
    }

    struct LlamaFastChatSession {
        state: Arc<Mutex<LlamaFastContext>>,
    }

    #[derive(Debug, Clone, Copy)]
    struct FastSessionParams {
        n_ctx: usize,
        n_batch: usize,
        n_ubatch: u32,
        n_threads: u32,
        n_threads_batch: u32,
        gpu_layers: i32,
        main_gpu: i32,
        seed: u32,
        offload_kqv: bool,
        kv_cache_type: LlamaKvCacheType,
        flash_attn: Option<bool>,
        warmup_tokens: usize,
    }

    #[derive(Debug, Clone, serde::Serialize)]
    pub struct LlamaFastRuntimeReport {
        pub backend: &'static str,
        pub compiled: bool,
        pub runtime: &'static str,
        pub native_commit: &'static str,
        pub modern_sampler: bool,
        pub flash_attn_requested: Option<bool>,
        pub flash_attn_supported: bool,
        pub cuda_compute_cap: Option<String>,
        pub ctx_size: usize,
        pub batch_size: usize,
        pub ubatch_size: u32,
        pub threads: u32,
        pub threads_batch: u32,
        pub gpu_layers: i32,
        pub main_gpu: i32,
        pub kv_cache_type: &'static str,
        pub kv_offload: bool,
        pub warmup_tokens: usize,
        pub warnings: Vec<String>,
    }

    struct FastBatch {
        raw: llama_batch,
        capacity: usize,
    }

    #[derive(Default)]
    struct Utf8PieceDecoder {
        pending: Vec<u8>,
    }

    struct NativeStderrGuard {
        #[cfg(unix)]
        saved_fd: RawFd,
    }

    unsafe impl Send for LlamaFastModel {}
    unsafe impl Sync for LlamaFastModel {}
    unsafe impl Send for LlamaFastContext {}

    impl LlamaFastBackend {
        pub fn new(store: ModelStore, mode: LlamaCppMode) -> Self {
            Self::new_with_options(store, mode, LlamaRuntimeOptions::default())
        }

        pub fn new_with_options(
            store: ModelStore,
            mode: LlamaCppMode,
            runtime_options: LlamaRuntimeOptions,
        ) -> Self {
            Self {
                store,
                mode,
                runtime_options,
                models: Arc::new(Mutex::new(HashMap::new())),
            }
        }

        pub fn probe(mode: LlamaCppMode) -> Result<String> {
            if !compiled(mode) {
                bail!("{}", unavailable_message(mode));
            }
            Ok(format!("llama.cpp legacy FFI {}", display_name(mode)))
        }

        pub fn runtime_report(
            mode: LlamaCppMode,
            runtime_options: &LlamaRuntimeOptions,
        ) -> LlamaFastRuntimeReport {
            let params = session_params(mode, None, runtime_options);
            let mut warnings = Vec::new();
            if runtime_options.flash_attn.is_some() {
                warnings.push(
                    "flash attention was requested, but llama_cpp_sys 0.3.2 does not expose a flash_attn context parameter".to_string(),
                );
            }
            if cfg!(feature = "cuda-mmq") {
                warnings.push(
                    "Werk's cuda-mmq feature is a compatibility alias to llama-legacy-cuda; forced MMQ is disabled because llama_cpp_sys 0.3.2 builds CUDA with -arch=all and fails device-link before sm_86".to_string(),
                );
            }
            warnings.push(
                "using legacy llama_cpp_sys 0.3.2; Ollama-class throughput likely requires a newer pinned llama.cpp runtime".to_string(),
            );
            LlamaFastRuntimeReport {
                backend: label(mode),
                compiled: compiled(mode),
                runtime: "llama_cpp_sys",
                native_commit: "unknown-legacy-0.3.2",
                modern_sampler: false,
                flash_attn_requested: params.flash_attn,
                flash_attn_supported: false,
                cuda_compute_cap: std::env::var("CUDA_COMPUTE_CAP").ok(),
                ctx_size: params.n_ctx,
                batch_size: params.n_batch,
                ubatch_size: params.n_ubatch,
                threads: params.n_threads,
                threads_batch: params.n_threads_batch,
                gpu_layers: params.gpu_layers,
                main_gpu: params.main_gpu,
                kv_cache_type: params.kv_cache_type.label(),
                kv_offload: params.offload_kqv,
                warmup_tokens: params.warmup_tokens,
                warnings,
            }
        }

        fn cached_model(&self, manifest: &ModelManifest) -> Result<(Arc<LlamaFastModel>, f64)> {
            if manifest.format != ModelFormat::Gguf {
                bail!(
                    "llama.cpp legacy FFI {} backend supports GGUF models only",
                    display_name(self.mode)
                );
            }

            let model_path = manifest
                .model_path
                .as_deref()
                .context("GGUF manifest has no model_path")?;
            let params = session_params(self.mode, None, &self.runtime_options);
            let cache_key = format!(
                "{}:{model_path}:{}:gpu_layers={}:main_gpu={}",
                manifest.id,
                label(self.mode),
                params.gpu_layers,
                params.main_gpu
            );

            if let Some(model) = self
                .models
                .lock()
                .map_err(|_| anyhow!("llama.cpp legacy FFI model cache mutex poisoned"))?
                .get(&cache_key)
                .cloned()
            {
                return Ok((model, 0.0));
            }

            if !compiled(self.mode) {
                bail!("{}", unavailable_message(self.mode));
            }

            let absolute_model_path = self.store.absolute_model_file(manifest, model_path);
            eprintln!(
                "Loading model '{}' with llama.cpp legacy FFI {}",
                manifest.id,
                display_name(self.mode)
            );
            let started = Instant::now();
            let stderr_guard = NativeStderrGuard::silence_if_needed();
            ensure_llama_backend();
            let model = LlamaFastModel::load(&absolute_model_path, &params)?;
            drop(stderr_guard);
            let load_seconds = started.elapsed().as_secs_f64();
            eprintln!(
                "Loaded model '{}' with llama.cpp legacy FFI {} in {:.2}s",
                manifest.id,
                display_name(self.mode),
                load_seconds
            );

            let model = Arc::new(model);
            self.models
                .lock()
                .map_err(|_| anyhow!("llama.cpp legacy FFI model cache mutex poisoned"))?
                .insert(cache_key, model.clone());
            Ok((model, load_seconds))
        }

        fn create_context(
            &self,
            model: Arc<LlamaFastModel>,
            seed: Option<u64>,
        ) -> Result<LlamaFastContext> {
            LlamaFastContext::new(
                model,
                session_params(self.mode, seed, &self.runtime_options),
            )
        }

        fn generate_inner(
            &self,
            manifest: &ModelManifest,
            request: GenerateRequest,
            tx: Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
        ) -> Result<GenerateResponse> {
            if !request.image_urls.is_empty() {
                bail!(
                    "llama.cpp legacy FFI backend is text-only for now; use a VLM-capable backend/model for image inputs"
                );
            }

            let total_started = Instant::now();
            let (model, load_seconds) = self.cached_model(manifest)?;
            let mut context = self.create_context(model, request.seed)?;
            let mut response = context.generate(request, tx)?;
            response.timings.load_seconds = load_seconds;
            response.timings.total_seconds = total_started.elapsed().as_secs_f64();
            Ok(response)
        }
    }

    impl GenerationBackend for LlamaFastBackend {
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
            let context = self.create_context(model, seed)?;
            Ok(Some(Box::new(LlamaFastChatSession {
                state: Arc::new(Mutex::new(context)),
            })))
        }

        fn generate(
            &self,
            manifest: &ModelManifest,
            request: GenerateRequest,
        ) -> Result<GenerateResponse> {
            self.generate_inner(manifest, request, None)
        }

        fn generate_stream(
            &self,
            manifest: ModelManifest,
            request: GenerateRequest,
        ) -> GenerateStream {
            let backend = self.clone();
            let (tx, rx) = mpsc::channel(16);
            tokio::task::spawn_blocking(move || {
                let result = backend.generate_inner(&manifest, request, Some(tx.clone()));
                send_stream_result(tx, result);
            });
            Box::pin(ReceiverStream::new(rx))
        }
    }

    impl ChatGenerationSession for LlamaFastChatSession {
        fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
            self.state
                .lock()
                .map_err(|_| anyhow!("llama.cpp legacy FFI chat session mutex poisoned"))
                .and_then(|mut state| state.generate(request, None))
        }

        fn generate_stream(&self, request: GenerateRequest) -> GenerateStream {
            let state = self.state.clone();
            let (tx, rx) = mpsc::channel(16);
            tokio::task::spawn_blocking(move || {
                let result = state
                    .lock()
                    .map_err(|_| anyhow!("llama.cpp legacy FFI chat session mutex poisoned"))
                    .and_then(|mut state| state.generate(request, Some(tx.clone())));
                send_stream_result(tx, result);
            });
            Box::pin(ReceiverStream::new(rx))
        }
    }

    impl LlamaFastModel {
        fn load(path: &PathBuf, session_params: &FastSessionParams) -> Result<Self> {
            let c_path = CString::new(path.to_string_lossy().as_bytes()).map_err(|_| {
                anyhow!(
                    "model path contains an interior NUL byte: {}",
                    path.display()
                )
            })?;
            let mut params = unsafe { llama_model_default_params() };
            params.n_gpu_layers = session_params.gpu_layers;
            params.use_mmap = true;
            params.main_gpu = session_params.main_gpu;

            let ptr = unsafe { llama_load_model_from_file(c_path.as_ptr(), params) };
            if ptr.is_null() {
                bail!(
                    "failed to load GGUF model with llama.cpp legacy FFI: {}",
                    path.display()
                );
            }

            let vocab = unsafe { llama_n_vocab(ptr) };
            if vocab <= 0 {
                unsafe {
                    llama_free_model(ptr);
                }
                bail!("llama.cpp reported an invalid vocabulary size: {vocab}");
            }

            Ok(Self {
                ptr,
                vocab: vocab as usize,
                eos: unsafe { llama_token_eos(ptr) },
                eot: unsafe { llama_token_eot(ptr) },
            })
        }

        fn tokenize(&self, text: &str) -> Result<Vec<llama_token>> {
            let bytes = text.as_bytes();
            let mut tokens = vec![0; bytes.len().saturating_add(8).max(8)];
            let mut written = unsafe {
                llama_tokenize(
                    self.ptr,
                    bytes.as_ptr() as *const c_char,
                    i32::try_from(bytes.len()).context("prompt is too large to tokenize")?,
                    tokens.as_mut_ptr(),
                    i32::try_from(tokens.len()).context("token buffer is too large")?,
                    false,
                    true,
                )
            };

            if written < 0 {
                tokens.resize(written.unsigned_abs() as usize, 0);
                written = unsafe {
                    llama_tokenize(
                        self.ptr,
                        bytes.as_ptr() as *const c_char,
                        i32::try_from(bytes.len()).context("prompt is too large to tokenize")?,
                        tokens.as_mut_ptr(),
                        i32::try_from(tokens.len()).context("token buffer is too large")?,
                        false,
                        true,
                    )
                };
            }

            if written < 0 {
                bail!("failed to tokenize prompt with llama.cpp legacy FFI");
            }
            tokens.truncate(written as usize);
            Ok(tokens)
        }

        fn token_piece(&self, token: llama_token) -> Result<Vec<u8>> {
            let mut buffer = vec![0u8; 32];
            let mut written = unsafe {
                llama_token_to_piece(
                    self.ptr,
                    token,
                    buffer.as_mut_ptr() as *mut c_char,
                    i32::try_from(buffer.len()).context("piece buffer is too large")?,
                )
            };

            if written < 0 {
                buffer.resize(written.unsigned_abs() as usize, 0);
                written = unsafe {
                    llama_token_to_piece(
                        self.ptr,
                        token,
                        buffer.as_mut_ptr() as *mut c_char,
                        i32::try_from(buffer.len()).context("piece buffer is too large")?,
                    )
                };
            }

            if written < 0 {
                bail!("failed to decode token piece with llama.cpp legacy FFI");
            }
            buffer.truncate(written as usize);
            Ok(buffer)
        }
    }

    impl Drop for LlamaFastModel {
        fn drop(&mut self) {
            unsafe {
                llama_free_model(self.ptr);
            }
        }
    }

    impl LlamaFastContext {
        fn new(model: Arc<LlamaFastModel>, mut params: FastSessionParams) -> Result<Self> {
            let mut context_params = unsafe { llama_context_default_params() };
            context_params.seed = params.seed;
            context_params.n_ctx = params.n_ctx as u32;
            context_params.n_batch = params.n_batch as u32;
            context_params.n_ubatch = params.n_ubatch;
            context_params.n_threads = params.n_threads;
            context_params.n_threads_batch = params.n_threads_batch;
            let kv_type = kv_cache_ggml_type(params.kv_cache_type);
            context_params.type_k = kv_type;
            context_params.type_v = kv_type;
            context_params.offload_kqv = params.offload_kqv;

            let ctx = unsafe { llama_new_context_with_model(model.ptr, context_params) };
            if ctx.is_null() {
                bail!("failed to create llama.cpp legacy FFI context");
            }
            params.n_ctx = unsafe { llama_n_ctx(ctx) } as usize;
            params.n_batch = unsafe { llama_n_batch(ctx) } as usize;
            params.n_ubatch = unsafe { llama_n_ubatch(ctx) };

            let batch = FastBatch::new(params.n_batch)?;
            let candidates = (0..model.vocab)
                .map(|id| llama_token_data {
                    id: id as llama_token,
                    logit: 0.0,
                    p: 0.0,
                })
                .collect();

            let mut context = Self {
                model,
                ctx,
                batch,
                params,
                tokens: Vec::with_capacity(params.n_ctx.min(8192)),
                candidates,
                decoder: Utf8PieceDecoder::default(),
                last_batch_size: 0,
                pending_warmup_seconds: 0.0,
            };
            context.pending_warmup_seconds = context.warmup();
            Ok(context)
        }

        fn generate(
            &mut self,
            request: GenerateRequest,
            tx: Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
        ) -> Result<GenerateResponse> {
            if !request.image_urls.is_empty() {
                bail!(
                    "llama.cpp legacy FFI backend is text-only for now; use a VLM-capable backend/model for image inputs"
                );
            }

            let total_started = Instant::now();
            unsafe {
                llama_reset_timings(self.ctx);
            }
            self.decoder.reset();

            let prompt_tokens = self.model.tokenize(&request.prompt)?;
            if prompt_tokens.is_empty() {
                bail!("prompt is empty after tokenization");
            }
            if prompt_tokens.len() >= self.params.n_ctx {
                bail!(
                    "chat context is full for this model ({} prompt tokens, {} token context); increase WERK_LLAMA_CTX or start a new chat",
                    prompt_tokens.len(),
                    self.params.n_ctx
                );
            }

            let prompt_started = Instant::now();
            let prompt_tokens_evaluated = self.set_context_to_tokens(&prompt_tokens)?;
            let fallback_prompt_seconds = prompt_started.elapsed().as_secs_f64();

            let max_predictions = request.max_tokens.min(
                self.params
                    .n_ctx
                    .saturating_sub(self.tokens.len())
                    .saturating_sub(1)
                    .max(1),
            );

            let mut text = String::new();
            let mut completion_tokens = 0usize;
            let mut finish_reason = "length".to_string();

            let decode_started = Instant::now();
            let mut first_token_seconds = 0.0;
            for _ in 0..max_predictions {
                let first_token_started = if completion_tokens == 0 {
                    Some(Instant::now())
                } else {
                    None
                };
                let token = self.sample_token(&request)?;
                if token == self.model.eos || token == self.model.eot {
                    finish_reason = "stop".to_string();
                    break;
                }

                self.decode_generated_token(token)?;
                completion_tokens += 1;
                if let Some(started) = first_token_started {
                    first_token_seconds = started.elapsed().as_secs_f64();
                }

                let piece = self.model.token_piece(token)?;
                let chunk = self.decoder.push(&piece);
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

            let tail = self.decoder.finish();
            if !tail.is_empty() && finish_reason != "stop" {
                let previous_len = text.len();
                text.push_str(&tail);
                if let Some(stop_index) = first_stop_index(&text, &request.stop) {
                    if stop_index > previous_len {
                        send_text_chunk(&tx, text[previous_len..stop_index].to_string())?;
                    }
                    text.truncate(stop_index);
                    finish_reason = "stop".to_string();
                } else {
                    send_text_chunk(&tx, tail)?;
                }
            }

            let fallback_decode_seconds = decode_started.elapsed().as_secs_f64();
            let warmup_seconds = std::mem::take(&mut self.pending_warmup_seconds);
            let timings = self.timings(
                total_started.elapsed().as_secs_f64(),
                warmup_seconds,
                first_token_seconds,
                fallback_prompt_seconds,
                fallback_decode_seconds,
            );
            Ok(GenerateResponse {
                text,
                prompt_tokens: prompt_tokens_evaluated,
                completion_tokens,
                finish_reason,
                timings,
            })
        }

        fn set_context_to_tokens(&mut self, prompt_tokens: &[llama_token]) -> Result<usize> {
            let shared_prefix = shared_prefix_len(&self.tokens, prompt_tokens);

            if shared_prefix < self.tokens.len() {
                let removed =
                    unsafe { llama_kv_cache_seq_rm(self.ctx, -1, shared_prefix as llama_pos, -1) };
                if !removed {
                    bail!("failed to trim llama.cpp KV cache for prompt reuse");
                }
                self.tokens.truncate(shared_prefix);
                self.last_batch_size = 0;
            }

            let new_tokens = &prompt_tokens[shared_prefix..];
            if new_tokens.is_empty() && self.last_batch_size == 0 {
                self.refresh_logits_from_last_token()?;
            } else {
                self.advance_prompt_tokens(new_tokens)?;
            }
            Ok(new_tokens.len())
        }

        fn refresh_logits_from_last_token(&mut self) -> Result<()> {
            let Some(token) = self.tokens.last().copied() else {
                return Ok(());
            };
            let pos = self.tokens.len() - 1;
            let removed = unsafe { llama_kv_cache_seq_rm(self.ctx, -1, pos as llama_pos, -1) };
            if !removed {
                bail!("failed to refresh llama.cpp logits after KV cache trim");
            }
            self.tokens.truncate(pos);
            self.advance_prompt_tokens(&[token])
        }

        fn advance_prompt_tokens(&mut self, tokens: &[llama_token]) -> Result<()> {
            if tokens.is_empty() {
                return Ok(());
            }

            let n_batch = self.params.n_batch.max(1);
            let mut advanced = 0usize;
            for chunk in tokens.chunks(n_batch) {
                let is_last = advanced + chunk.len() == tokens.len();
                let pos = self.tokens.len() + advanced;
                self.batch.fill(chunk, pos as llama_pos, is_last)?;
                let rc = unsafe { llama_decode(self.ctx, self.batch.raw) };
                if rc != 0 {
                    bail!("llama.cpp prompt decode failed with status {rc}");
                }
                self.last_batch_size = chunk.len();
                advanced += chunk.len();
            }

            self.tokens.extend_from_slice(tokens);
            Ok(())
        }

        fn decode_generated_token(&mut self, token: llama_token) -> Result<()> {
            self.batch
                .fill(&[token], self.tokens.len() as llama_pos, true)?;
            let rc = unsafe { llama_decode(self.ctx, self.batch.raw) };
            if rc != 0 {
                bail!("llama.cpp token decode failed with status {rc}");
            }
            self.tokens.push(token);
            self.last_batch_size = 1;
            Ok(())
        }

        fn sample_token(&mut self, request: &GenerateRequest) -> Result<llama_token> {
            let logits_index = self.last_batch_size.saturating_sub(1);
            let logits = unsafe { llama_get_logits_ith(self.ctx, logits_index as i32) };
            if logits.is_null() {
                bail!("llama.cpp did not return logits for the current context");
            }

            let temperature = request.temperature.unwrap_or(0.8) as f32;
            if temperature <= 0.0 {
                return Ok(argmax_token(logits, self.model.vocab));
            }

            for (id, candidate) in self.candidates.iter_mut().enumerate() {
                candidate.id = id as llama_token;
                candidate.logit = unsafe { *logits.add(id) };
                candidate.p = 0.0;
            }

            let mut candidates = llama_token_data_array {
                data: self.candidates.as_mut_ptr(),
                size: self.candidates.len(),
                sorted: false,
            };

            let token = unsafe {
                if temperature <= 0.0 {
                    llama_sample_token_greedy(self.ctx, &mut candidates)
                } else {
                    llama_sample_temp(self.ctx, &mut candidates, temperature);
                    if let Some(top_p) = request.top_p
                        && top_p > 0.0
                        && top_p < 1.0
                    {
                        llama_sample_top_p(self.ctx, &mut candidates, top_p as f32, 1);
                    }
                    llama_sample_token(self.ctx, &mut candidates)
                }
            };
            Ok(token)
        }

        fn warmup(&mut self) -> f64 {
            if self.params.warmup_tokens == 0 {
                return 0.0;
            }
            let started = Instant::now();
            let warmup_tokens = vec![self.model.eos; self.params.warmup_tokens];
            if self.advance_prompt_tokens(&warmup_tokens).is_ok() {
                unsafe {
                    let _ = llama_kv_cache_seq_rm(self.ctx, -1, 0, -1);
                }
                self.tokens.clear();
                self.last_batch_size = 0;
                started.elapsed().as_secs_f64()
            } else {
                0.0
            }
        }

        fn timings(
            &self,
            total_seconds: f64,
            warmup_seconds: f64,
            first_token_seconds: f64,
            fallback_prompt_seconds: f64,
            fallback_decode_seconds: f64,
        ) -> GenerationTimings {
            let timings = unsafe { llama_get_timings(self.ctx) };
            let prompt_seconds =
                if timings.t_p_eval_ms > 0.0 && timings.t_p_eval_ms / 1000.0 <= total_seconds {
                    timings.t_p_eval_ms / 1000.0
                } else {
                    fallback_prompt_seconds
                };
            let decode_seconds =
                if timings.t_eval_ms > 0.0 && timings.t_eval_ms / 1000.0 <= total_seconds {
                    timings.t_eval_ms / 1000.0
                } else {
                    fallback_decode_seconds
                };
            GenerationTimings {
                load_seconds: 0.0,
                warmup_seconds,
                first_token_seconds,
                prompt_seconds,
                decode_seconds,
                total_seconds,
            }
        }
    }

    impl Drop for LlamaFastContext {
        fn drop(&mut self) {
            unsafe {
                llama_free(self.ctx);
            }
        }
    }

    impl FastBatch {
        fn new(capacity: usize) -> Result<Self> {
            if capacity == 0 || capacity > i32::MAX as usize {
                bail!("invalid llama.cpp batch capacity: {capacity}");
            }
            let raw = unsafe { llama_batch_init(capacity as i32, 0, 1) };
            if raw.token.is_null()
                || raw.pos.is_null()
                || raw.n_seq_id.is_null()
                || raw.seq_id.is_null()
                || raw.logits.is_null()
            {
                unsafe {
                    llama_batch_free(raw);
                }
                bail!("failed to allocate llama.cpp batch");
            }
            Ok(Self { raw, capacity })
        }

        fn fill(
            &mut self,
            tokens: &[llama_token],
            start_pos: llama_pos,
            logits_last: bool,
        ) -> Result<()> {
            if tokens.len() > self.capacity {
                bail!(
                    "llama.cpp batch capacity exceeded: {} tokens for capacity {}",
                    tokens.len(),
                    self.capacity
                );
            }
            self.raw.n_tokens = tokens.len() as i32;
            for (index, token) in tokens.iter().copied().enumerate() {
                unsafe {
                    *self.raw.token.add(index) = token;
                    *self.raw.pos.add(index) = start_pos + index as llama_pos;
                    *self.raw.n_seq_id.add(index) = 1;
                    *(*self.raw.seq_id.add(index)).add(0) = 0;
                    *self.raw.logits.add(index) =
                        i8::from(logits_last && index + 1 == tokens.len());
                }
            }
            Ok(())
        }
    }

    impl Drop for FastBatch {
        fn drop(&mut self) {
            unsafe {
                llama_batch_free(self.raw);
            }
        }
    }

    impl NativeStderrGuard {
        fn silence_if_needed() -> Option<Self> {
            if env_true("WERK_LLAMA_LOG") {
                return None;
            }
            Self::silence()
        }

        #[cfg(unix)]
        fn silence() -> Option<Self> {
            let _ = std::io::stderr().flush();
            unsafe {
                let saved_fd = libc::dup(libc::STDERR_FILENO);
                if saved_fd < 0 {
                    return None;
                }

                let null_path = c"/dev/null";
                let null_fd = libc::open(null_path.as_ptr(), libc::O_WRONLY);
                if null_fd < 0 {
                    libc::close(saved_fd);
                    return None;
                }

                if libc::dup2(null_fd, libc::STDERR_FILENO) < 0 {
                    libc::close(null_fd);
                    libc::close(saved_fd);
                    return None;
                }

                libc::close(null_fd);
                Some(Self { saved_fd })
            }
        }

        #[cfg(not(unix))]
        fn silence() -> Option<Self> {
            None
        }
    }

    impl Drop for NativeStderrGuard {
        fn drop(&mut self) {
            #[cfg(unix)]
            unsafe {
                let _ = std::io::stderr().flush();
                libc::dup2(self.saved_fd, libc::STDERR_FILENO);
                libc::close(self.saved_fd);
            }
        }
    }

    impl Utf8PieceDecoder {
        fn reset(&mut self) {
            self.pending.clear();
        }

        fn push(&mut self, bytes: &[u8]) -> String {
            self.pending.extend_from_slice(bytes);
            match std::str::from_utf8(&self.pending) {
                Ok(text) => {
                    let text = text.to_string();
                    self.pending.clear();
                    text
                }
                Err(err) => {
                    let valid_up_to = err.valid_up_to();
                    if valid_up_to == 0 {
                        String::new()
                    } else {
                        let text =
                            String::from_utf8_lossy(&self.pending[..valid_up_to]).to_string();
                        self.pending.drain(..valid_up_to);
                        text
                    }
                }
            }
        }

        fn finish(&mut self) -> String {
            if self.pending.is_empty() {
                String::new()
            } else {
                let text = String::from_utf8_lossy(&self.pending).to_string();
                self.pending.clear();
                text
            }
        }
    }

    fn send_stream_result(
        tx: mpsc::Sender<Result<GenerateStreamEvent, String>>,
        result: Result<GenerateResponse>,
    ) {
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
    }

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

    fn session_params(
        mode: LlamaCppMode,
        seed: Option<u64>,
        options: &LlamaRuntimeOptions,
    ) -> FastSessionParams {
        let offload_kqv = options
            .kv_offload
            .or_else(|| env_bool("WERK_LLAMA_KV_OFFLOAD"))
            .unwrap_or(!matches!(mode, LlamaCppMode::Cpu));
        let warmup_tokens = if env_false("WERK_LLAMA_WARMUP") {
            0
        } else {
            options
                .warmup_tokens
                .or_else(|| env_usize("WERK_LLAMA_WARMUP_TOKENS"))
                .unwrap_or(1)
        };
        FastSessionParams {
            n_ctx: options
                .ctx_size
                .or_else(|| env_usize("WERK_LLAMA_CTX"))
                .unwrap_or(DEFAULT_N_CTX as usize),
            n_batch: options
                .batch_size
                .or_else(|| env_usize("WERK_LLAMA_BATCH"))
                .unwrap_or(DEFAULT_N_BATCH as usize),
            n_ubatch: options
                .ubatch_size
                .or_else(|| env_u32("WERK_LLAMA_UBATCH"))
                .unwrap_or(DEFAULT_N_UBATCH),
            n_threads: options
                .threads
                .or_else(|| env_u32("WERK_LLAMA_THREADS"))
                .unwrap_or_else(default_threads),
            n_threads_batch: options
                .threads_batch
                .or_else(|| env_u32("WERK_LLAMA_THREADS_BATCH"))
                .unwrap_or_else(default_threads),
            gpu_layers: options
                .gpu_layers
                .or_else(|| env_i32("WERK_LLAMA_GPU_LAYERS"))
                .unwrap_or_else(|| gpu_layers(mode)),
            main_gpu: options
                .main_gpu
                .or_else(|| env_i32("WERK_LLAMA_MAIN_GPU"))
                .unwrap_or(0),
            seed: seed.map(seed_u32).unwrap_or(u32::MAX),
            offload_kqv,
            kv_cache_type: options
                .kv_cache_type
                .or_else(|| env_kv_cache_type("WERK_LLAMA_KV_CACHE_TYPE"))
                .unwrap_or(LlamaKvCacheType::F16),
            flash_attn: options
                .flash_attn
                .or_else(|| env_bool("WERK_LLAMA_FLASH_ATTN")),
            warmup_tokens,
        }
    }

    fn compiled(mode: LlamaCppMode) -> bool {
        match mode {
            LlamaCppMode::Cpu => cfg!(feature = "llama-fast"),
            LlamaCppMode::Cuda => cfg!(feature = "llama-legacy-cuda"),
            LlamaCppMode::Rocm => false,
            LlamaCppMode::Vulkan => cfg!(feature = "llama-legacy-vulkan"),
            LlamaCppMode::Metal => false,
        }
    }

    fn display_name(mode: LlamaCppMode) -> &'static str {
        match mode {
            LlamaCppMode::Cuda => "CUDA",
            LlamaCppMode::Rocm => "ROCm/HIP",
            LlamaCppMode::Vulkan => "Vulkan",
            LlamaCppMode::Metal => "Metal",
            LlamaCppMode::Cpu => "CPU",
        }
    }

    fn label(mode: LlamaCppMode) -> &'static str {
        match mode {
            LlamaCppMode::Cuda => "llama-legacy-cuda",
            LlamaCppMode::Rocm => "llama-legacy-rocm",
            LlamaCppMode::Vulkan => "llama-legacy-vulkan",
            LlamaCppMode::Metal => "llama-legacy-metal",
            LlamaCppMode::Cpu => "llama-legacy-cpu",
        }
    }

    fn gpu_layers(mode: LlamaCppMode) -> i32 {
        match mode {
            LlamaCppMode::Cpu => 0,
            LlamaCppMode::Cuda
            | LlamaCppMode::Rocm
            | LlamaCppMode::Vulkan
            | LlamaCppMode::Metal => 999,
        }
    }

    fn unavailable_message(mode: LlamaCppMode) -> String {
        match mode {
            LlamaCppMode::Cuda => {
                "llama.cpp legacy FFI CUDA backend is not compiled into this binary; build/install with --features llama-legacy-cuda".to_string()
            }
            LlamaCppMode::Rocm => {
                "llama.cpp legacy FFI ROCm backend is not implemented; use the persistent llama.cpp server ROCm route with --backend rocm".to_string()
            }
            LlamaCppMode::Vulkan => {
                "llama.cpp legacy FFI Vulkan backend is not compiled into this binary; build/install with --features llama-legacy-vulkan".to_string()
            }
            LlamaCppMode::Metal => {
                "llama.cpp legacy FFI Metal backend is not implemented; use the persistent llama.cpp server Metal route with --backend metal".to_string()
            }
            LlamaCppMode::Cpu => {
                "llama.cpp legacy FFI CPU backend is not compiled into this binary; build/install with --features llama-fast".to_string()
            }
        }
    }

    fn ensure_llama_backend() {
        static INIT: Once = Once::new();
        INIT.call_once(|| unsafe {
            if !env_true("WERK_LLAMA_LOG") {
                llama_log_set(Some(silent_llama_log), std::ptr::null_mut());
            }
            llama_cpp_sys::llama_backend_init();
            llama_numa_init(ggml_numa_strategy::GGML_NUMA_STRATEGY_DISTRIBUTE);
        });
    }

    unsafe extern "C" fn silent_llama_log(
        _level: llama_cpp_sys::ggml_log_level,
        _text: *const c_char,
        _user_data: *mut c_void,
    ) {
    }

    fn first_stop_index(text: &str, stops: &[String]) -> Option<usize> {
        stops
            .iter()
            .filter(|stop| !stop.is_empty())
            .filter_map(|stop| text.find(stop))
            .min()
    }

    fn shared_prefix_len(previous: &[llama_token], next: &[llama_token]) -> usize {
        previous
            .iter()
            .zip(next)
            .position(|(left, right)| left != right)
            .unwrap_or(previous.len().min(next.len()))
    }

    fn seed_u32(seed: u64) -> u32 {
        u32::try_from(seed).unwrap_or_else(|_| {
            let high = (seed >> 32) as u32;
            let low = seed as u32;
            high ^ low
        })
    }

    fn argmax_token(logits: *const f32, vocab: usize) -> llama_token {
        let mut best_id = 0usize;
        let mut best_logit = f32::NEG_INFINITY;
        for id in 0..vocab {
            let logit = unsafe { *logits.add(id) };
            if logit > best_logit {
                best_logit = logit;
                best_id = id;
            }
        }
        best_id as llama_token
    }

    fn kv_cache_ggml_type(cache_type: LlamaKvCacheType) -> ggml_type {
        match cache_type {
            LlamaKvCacheType::F16 => ggml_type::GGML_TYPE_F16,
            LlamaKvCacheType::F32 => ggml_type::GGML_TYPE_F32,
            LlamaKvCacheType::Q8_0 => ggml_type::GGML_TYPE_Q8_0,
        }
    }

    fn default_threads() -> u32 {
        std::thread::available_parallelism()
            .map(|threads| threads.get().saturating_sub(1).max(1) as u32)
            .unwrap_or(1)
    }

    fn env_u32(name: &str) -> Option<u32> {
        std::env::var(name).ok()?.parse().ok()
    }

    fn env_i32(name: &str) -> Option<i32> {
        std::env::var(name).ok()?.parse().ok()
    }

    fn env_usize(name: &str) -> Option<usize> {
        std::env::var(name).ok()?.parse().ok()
    }

    fn env_bool(name: &str) -> Option<bool> {
        let value = std::env::var(name).ok()?;
        parse_bool(&value)
    }

    fn parse_bool(value: &str) -> Option<bool> {
        match value {
            "1" | "true" | "True" | "TRUE" | "yes" | "Yes" | "YES" | "on" | "On" | "ON" => {
                Some(true)
            }
            "0" | "false" | "False" | "FALSE" | "no" | "No" | "NO" | "off" | "Off" | "OFF" => {
                Some(false)
            }
            _ => None,
        }
    }

    fn env_kv_cache_type(name: &str) -> Option<LlamaKvCacheType> {
        let value = std::env::var(name).ok()?;
        parse_kv_cache_type(&value)
    }

    fn parse_kv_cache_type(value: &str) -> Option<LlamaKvCacheType> {
        match value {
            "f16" | "F16" => Some(LlamaKvCacheType::F16),
            "f32" | "F32" => Some(LlamaKvCacheType::F32),
            "q8_0" | "q8-0" | "Q8_0" | "Q8-0" => Some(LlamaKvCacheType::Q8_0),
            _ => None,
        }
    }

    fn env_false(name: &str) -> bool {
        matches!(
            std::env::var(name).ok().as_deref(),
            Some("0" | "false" | "False" | "FALSE" | "no" | "No" | "NO" | "off" | "Off" | "OFF")
        )
    }

    fn env_true(name: &str) -> bool {
        matches!(
            std::env::var(name).ok().as_deref(),
            Some("1" | "true" | "True" | "TRUE" | "yes" | "Yes" | "YES" | "on" | "On" | "ON")
        )
    }

    fn format_error_chain(err: &anyhow::Error) -> String {
        let messages = err.chain().map(ToString::to_string).collect::<Vec<_>>();
        messages.join(": ")
    }
}

#[cfg(not(feature = "llama-fast"))]
mod imp {
    use anyhow::{Result, bail};

    use crate::backend::{
        GenerateRequest, GenerateResponse, GenerateStream, GenerationBackend, LlamaCppMode,
        LlamaRuntimeOptions,
    };
    use crate::model_store::{ModelManifest, ModelStore};

    #[derive(Clone)]
    pub struct LlamaFastBackend {
        _store: ModelStore,
        mode: LlamaCppMode,
    }

    impl LlamaFastBackend {
        pub fn new(store: ModelStore, mode: LlamaCppMode) -> Self {
            Self::new_with_options(store, mode, LlamaRuntimeOptions::default())
        }

        pub fn new_with_options(
            store: ModelStore,
            mode: LlamaCppMode,
            _runtime_options: LlamaRuntimeOptions,
        ) -> Self {
            Self {
                _store: store,
                mode,
            }
        }

        pub fn probe(mode: LlamaCppMode) -> Result<String> {
            bail!("{}", unavailable_message(mode))
        }

        pub fn runtime_report(
            mode: LlamaCppMode,
            _runtime_options: &LlamaRuntimeOptions,
        ) -> LlamaFastRuntimeReport {
            LlamaFastRuntimeReport {
                backend: match mode {
                    LlamaCppMode::Cuda => "llama-legacy-cuda",
                    LlamaCppMode::Rocm => "llama-legacy-rocm",
                    LlamaCppMode::Vulkan => "llama-legacy-vulkan",
                    LlamaCppMode::Metal => "llama-legacy-metal",
                    LlamaCppMode::Cpu => "llama-legacy-cpu",
                },
                compiled: false,
                runtime: "llama_cpp_sys",
                native_commit: "unavailable",
                modern_sampler: false,
                flash_attn_requested: None,
                flash_attn_supported: false,
                cuda_compute_cap: std::env::var("CUDA_COMPUTE_CAP").ok(),
                ctx_size: 0,
                batch_size: 0,
                ubatch_size: 0,
                threads: 0,
                threads_batch: 0,
                gpu_layers: 0,
                main_gpu: 0,
                kv_cache_type: "unknown",
                kv_offload: false,
                warmup_tokens: 0,
                warnings: vec![unavailable_message(mode)],
            }
        }
    }

    #[derive(Debug, Clone, serde::Serialize)]
    pub struct LlamaFastRuntimeReport {
        pub backend: &'static str,
        pub compiled: bool,
        pub runtime: &'static str,
        pub native_commit: &'static str,
        pub modern_sampler: bool,
        pub flash_attn_requested: Option<bool>,
        pub flash_attn_supported: bool,
        pub cuda_compute_cap: Option<String>,
        pub ctx_size: usize,
        pub batch_size: usize,
        pub ubatch_size: u32,
        pub threads: u32,
        pub threads_batch: u32,
        pub gpu_layers: i32,
        pub main_gpu: i32,
        pub kv_cache_type: &'static str,
        pub kv_offload: bool,
        pub warmup_tokens: usize,
        pub warnings: Vec<String>,
    }

    impl GenerationBackend for LlamaFastBackend {
        fn generate(
            &self,
            _manifest: &ModelManifest,
            _request: GenerateRequest,
        ) -> Result<GenerateResponse> {
            bail!("{}", unavailable_message(self.mode))
        }

        fn generate_stream(
            &self,
            _manifest: ModelManifest,
            _request: GenerateRequest,
        ) -> GenerateStream {
            Box::pin(tokio_stream::iter(vec![Err(unavailable_message(
                self.mode,
            ))]))
        }
    }

    fn unavailable_message(mode: LlamaCppMode) -> String {
        match mode {
            LlamaCppMode::Cuda => {
                "llama.cpp legacy FFI CUDA backend is not compiled into this binary; build/install with --features llama-legacy-cuda".to_string()
            }
            LlamaCppMode::Rocm => {
                "llama.cpp legacy FFI ROCm backend is not implemented; use the persistent llama.cpp server ROCm route with --backend rocm".to_string()
            }
            LlamaCppMode::Vulkan => {
                "llama.cpp legacy FFI Vulkan backend is not compiled into this binary; build/install with --features llama-legacy-vulkan".to_string()
            }
            LlamaCppMode::Metal => {
                "llama.cpp legacy FFI Metal backend is not implemented; use the persistent llama.cpp server Metal route with --backend metal".to_string()
            }
            LlamaCppMode::Cpu => {
                "llama.cpp legacy FFI CPU backend is not compiled into this binary; build/install with --features llama-fast".to_string()
            }
        }
    }
}

pub use imp::{LlamaFastBackend, LlamaFastRuntimeReport};
