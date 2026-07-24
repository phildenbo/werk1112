use anyhow::{Context, Result, anyhow, bail};
use candle_core::{
    D, DType, Device, IndexOp, Tensor,
    quantized::gguf_file::{self, Value as GgufValue},
};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::{
        gemma, gemma2, llama, mistral, phi3, quantized_gemma3, quantized_llama, quantized_phi,
        quantized_phi3, quantized_qwen2, quantized_qwen3, qwen2, qwen3,
    },
};
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use std::{fs, path::PathBuf};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokenizers::{
    AddedToken, DecoderWrapper, Tokenizer,
    decoders::{byte_fallback::ByteFallback, sequence::Sequence},
    models::unigram::Unigram,
    pre_tokenizers::metaspace::{Metaspace, PrependScheme},
    processors::template::TemplateProcessing,
};
use tokio::sync::mpsc::{self, Sender};
use tokio_stream::wrappers::ReceiverStream;

use super::{
    GenerateRequest, GenerateResponse, GenerateStream, GenerateStreamEvent, GenerationBackend,
    GenerationTimings, StreamGranularity,
};
use crate::model_store::{ModelFormat, ModelManifest, ModelStore};

#[derive(Clone)]
pub struct CandleBackend {
    store: ModelStore,
    device: Device,
    cache: Arc<Mutex<HashMap<String, Arc<Mutex<CachedModel>>>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandleDeviceMode {
    Auto,
    Cpu,
    Cuda,
    Metal,
}

impl CandleBackend {
    pub fn new(store: ModelStore) -> Result<Self> {
        Self::new_with_device(store, CandleDeviceMode::Auto)
    }

    pub fn new_with_device(store: ModelStore, device_mode: CandleDeviceMode) -> Result<Self> {
        let device = select_device(device_mode)?;
        eprintln!("Using {} backend", candle_backend_name(&device));
        Ok(Self {
            store,
            device,
            cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    fn generate_inner(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
        on_token: Option<TokenCallback>,
    ) -> Result<GenerateResponse> {
        generate_with_candle(self, manifest, request, on_token)
    }

    fn cached_model(&self, manifest: &ModelManifest) -> Result<(Arc<Mutex<CachedModel>>, f64)> {
        let cache_key = cache_key(manifest);
        if let Some(cached) = self
            .cache
            .lock()
            .map_err(|_| anyhow!("model cache mutex poisoned"))?
            .get(&cache_key)
            .cloned()
        {
            return Ok((cached, 0.0));
        }

        eprintln!(
            "Loading model '{}' ({:?}, architecture: {})",
            manifest.id,
            manifest.format,
            manifest.architecture.as_deref().unwrap_or("unknown")
        );
        let started = Instant::now();
        let tokenizer = load_tokenizer(&self.store, manifest)?;
        let model = load_candle_model(&self.store, manifest, &self.device)?;
        let load_seconds = started.elapsed().as_secs_f64();
        eprintln!("Loaded model '{}' in {:.2}s", manifest.id, load_seconds);

        let cached = Arc::new(Mutex::new(CachedModel { tokenizer, model }));
        self.cache
            .lock()
            .map_err(|_| anyhow!("model cache mutex poisoned"))?
            .insert(cache_key, cached.clone());
        Ok((cached, load_seconds))
    }
}

pub fn probe_device(mode: CandleDeviceMode) -> Result<String> {
    Ok(format!("{:?}", select_device(mode)?))
}

impl GenerationBackend for CandleBackend {
    fn prepare(&self, manifest: &ModelManifest) -> Result<()> {
        self.cached_model(manifest).map(|_| ())
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
        let stream_granularity = request.stream_granularity;

        tokio::task::spawn_blocking(move || {
            let emitter = StreamEmitter::new(tx.clone(), stream_granularity);
            let callback_emitter = emitter.clone();
            let result = generate_with_candle(
                &backend,
                &manifest,
                request,
                Some(Box::new(move |text| {
                    callback_emitter.push(text);
                })),
            );

            match result {
                Ok(response) => {
                    emitter.flush();
                    let _ = tx.blocking_send(Ok(GenerateStreamEvent::Done {
                        finish_reason: response.finish_reason,
                        prompt_tokens: response.prompt_tokens,
                        completion_tokens: response.completion_tokens,
                        timings: response.timings,
                        backend_diagnostics: response.backend_diagnostics,
                    }));
                }
                Err(err) => {
                    let _ = tx.blocking_send(Err(err.to_string()));
                }
            }
        });

        Box::pin(ReceiverStream::new(rx))
    }
}

fn select_device(mode: CandleDeviceMode) -> Result<Device> {
    match mode {
        CandleDeviceMode::Auto => Ok(Device::new_cuda(0)
            .or_else(|_| Device::new_metal(0))
            .unwrap_or(Device::Cpu)),
        CandleDeviceMode::Cpu => Ok(Device::Cpu),
        CandleDeviceMode::Cuda => select_cuda_device(),
        CandleDeviceMode::Metal => Device::new_metal(0).context(
            "Metal was requested but is unavailable; build with --features metal on macOS/Apple Silicon",
        ),
    }
}

#[cfg(feature = "candle-cuda")]
fn select_cuda_device() -> Result<Device> {
    Device::new_cuda(0).context(
        "Candle CUDA was requested but is unavailable; check the NVIDIA driver, CUDA runtime, and visible GPU",
    )
}

#[cfg(not(feature = "candle-cuda"))]
fn select_cuda_device() -> Result<Device> {
    bail!(
        "This Werk binary was built without Candle CUDA support. Rebuild with: cargo install --path . --locked --force --features cuda"
    )
}

fn candle_backend_name(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "Candle CPU",
        Device::Cuda(_) => "Candle CUDA",
        Device::Metal(_) => "Candle Metal",
    }
}

type TokenCallback = Box<dyn FnMut(String) + Send>;

fn generate_with_candle(
    backend: &CandleBackend,
    manifest: &ModelManifest,
    request: GenerateRequest,
    mut on_token: Option<TokenCallback>,
) -> Result<GenerateResponse> {
    if !request.image_urls.is_empty() {
        bail!(
            "this Candle backend is text-only for now; use a VLM-capable backend/model for image inputs"
        );
    }

    let started = Instant::now();
    let (cached, load_seconds) = backend.cached_model(manifest)?;
    let mut cached = cached
        .lock()
        .map_err(|_| anyhow!("cached model mutex poisoned"))?;
    let tokenizer = cached.tokenizer.clone();
    cached.model.clear_kv_cache()?;
    let mut response = generate_with_loaded_model(
        &mut cached.model,
        &tokenizer,
        &backend.device,
        request,
        on_token.take(),
    )?;
    response.timings.load_seconds = load_seconds;
    response.timings.total_seconds = started.elapsed().as_secs_f64();
    Ok(response)
}

fn cache_key(manifest: &ModelManifest) -> String {
    format!(
        "{}:{:?}:{}:{}:{}",
        manifest.id,
        manifest.format,
        manifest.model_path.as_deref().unwrap_or_default(),
        manifest.tokenizer_path.as_deref().unwrap_or_default(),
        manifest.config_path.as_deref().unwrap_or_default()
    )
}

fn load_tokenizer(store: &ModelStore, manifest: &ModelManifest) -> Result<Tokenizer> {
    if let Some(tokenizer_path) = manifest.tokenizer_path.as_deref() {
        let tokenizer_file = store.absolute_model_file(manifest, tokenizer_path);
        return Tokenizer::from_file(&tokenizer_file).map_err(|err| {
            anyhow::anyhow!(
                "failed to load tokenizer {}: {err}",
                tokenizer_file.display()
            )
        });
    }

    if manifest.format == ModelFormat::Gguf {
        let model_path = manifest
            .model_path
            .as_deref()
            .context("GGUF manifest has no model_path")?;
        return load_embedded_gguf_tokenizer(&store.absolute_model_file(manifest, model_path));
    }

    bail!("manifest has no tokenizer_path; add tokenizer.json beside the model")
}

pub fn candle_gguf_tokenizer_rejection(
    store: &ModelStore,
    manifest: &ModelManifest,
) -> Option<String> {
    if manifest.format != ModelFormat::Gguf || manifest.tokenizer_path.is_some() {
        return None;
    }

    if manifest
        .architecture
        .as_deref()
        .is_some_and(is_gguf_bpe_tokenizer_architecture)
    {
        return Some(
            "Candle GGUF fallback requires tokenizer.json for this architecture; use llama.cpp server for GGUF acceleration or add tokenizer.json beside the model"
                .to_string(),
        );
    }

    let model_path = manifest.model_path.as_deref()?;
    let path = store.absolute_model_file(manifest, model_path);
    let tokenizer_model = read_gguf_tokenizer_model(&path).ok()?;
    if tokenizer_model == "llama" {
        None
    } else {
        Some(format!(
            "Candle GGUF fallback requires tokenizer.json for tokenizer.ggml.model='{tokenizer_model}'; use llama.cpp server for this GGUF or add tokenizer.json beside the model"
        ))
    }
}

fn is_gguf_bpe_tokenizer_architecture(architecture: &str) -> bool {
    matches!(
        architecture.to_ascii_lowercase().as_str(),
        "qwen" | "qwen2" | "qwen3"
    )
}

fn read_gguf_tokenizer_model(path: &PathBuf) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let content = gguf_file::Content::read(&mut file)?;
    gguf_string(&content, "tokenizer.ggml.model")
}

fn generate_with_loaded_model(
    model: &mut CandleModel,
    tokenizer: &Tokenizer,
    device: &Device,
    request: GenerateRequest,
    mut on_token: Option<TokenCallback>,
) -> Result<GenerateResponse> {
    let tokenize_started = Instant::now();
    let mut tokens = tokenizer
        .encode(request.prompt.clone(), true)
        .map_err(|err| anyhow::anyhow!("tokenization failed: {err}"))?
        .get_ids()
        .to_vec();
    if tokens.is_empty() {
        bail!("tokenizer produced no prompt tokens");
    }
    let prompt_tokens = tokens.len();

    let eos_token = eos_token_id(tokenizer);
    let mut token_selector = TokenSelector::new(
        request.seed.unwrap_or(299792458),
        request.temperature,
        request.top_p,
    );
    let mut generated = Vec::new();
    let mut generated_text = String::new();
    let mut decode_stream = tokenizer.decode_stream(true);
    let mut streamed_len = 0usize;
    let mut finish_reason = "length".to_string();
    let mut matched_stop_string = false;
    let max_stop_len = max_stop_len(&request.stop);
    let mut profile = request.debug.then(DecodeLoopProfile::default);

    let prompt_started = Instant::now();
    let mut input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
    let mut index_pos = 0usize;
    let mut logits = model.forward(&input, index_pos)?;
    index_pos += tokens.len();
    let prompt_seconds = prompt_started.elapsed().as_secs_f64();

    let decode_started = Instant::now();
    for _ in 0..request.max_tokens {
        let next_logits = last_logits(&logits)?;
        let next_token =
            select_next_token(&mut token_selector, &next_logits, device, &mut profile)?;
        if Some(next_token) == eos_token {
            finish_reason = "stop".to_string();
            break;
        }

        tokens.push(next_token);
        generated.push(next_token);

        let tokenizer_started = profile.as_ref().map(|_| Instant::now());
        let piece = decode_stream
            .step(next_token)
            .map_err(|err| anyhow!("incremental token decode failed: {err}"))?
            .unwrap_or_default();
        if let (Some(profile), Some(started)) = (profile.as_mut(), tokenizer_started) {
            profile.tokenizer += started.elapsed();
        }
        let stop_check_from = stop_check_start(generated_text.len(), max_stop_len);
        generated_text.push_str(&piece);

        let stop_started = profile.as_ref().map(|_| Instant::now());
        let reached_stop = stop_reached_from(&mut generated_text, &request.stop, stop_check_from);
        let safe_len = if reached_stop {
            generated_text.len()
        } else {
            stop_guard_emit_len(&generated_text, &request.stop, max_stop_len)
        };
        if let (Some(profile), Some(started)) = (profile.as_mut(), stop_started) {
            profile.stop_check += started.elapsed();
        }

        let callback_started = profile.as_ref().map(|_| Instant::now());
        if let Some(callback) = on_token.as_mut()
            && safe_len > streamed_len
        {
            let safe_piece = generated_text[streamed_len..safe_len].to_string();
            callback(safe_piece);
            streamed_len = safe_len;
        }
        if let (Some(profile), Some(started)) = (profile.as_mut(), callback_started) {
            profile.callback += started.elapsed();
        }

        if reached_stop {
            finish_reason = "stop".to_string();
            matched_stop_string = true;
            break;
        }

        let forward_started = profile.as_ref().map(|_| Instant::now());
        input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        logits = model.forward(&input, index_pos)?;
        if profile.is_some() {
            device.synchronize()?;
        }
        index_pos += 1;
        if let (Some(profile), Some(started)) = (profile.as_mut(), forward_started) {
            profile.forward += started.elapsed();
        }
    }
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    if !matched_stop_string
        && let Some(callback) = on_token.as_mut()
        && generated_text.len() > streamed_len
    {
        let callback_started = profile.as_ref().map(|_| Instant::now());
        callback(generated_text[streamed_len..].to_string());
        if let (Some(profile), Some(started)) = (profile.as_mut(), callback_started) {
            profile.callback += started.elapsed();
        }
    }
    if let Some(mut profile) = profile {
        profile.total = Duration::from_secs_f64(decode_seconds);
        profile.print(generated.len());
    }

    Ok(GenerateResponse {
        text: generated_text,
        prompt_tokens,
        completion_tokens: generated.len(),
        finish_reason,
        timings: GenerationTimings {
            load_seconds: 0.0,
            warmup_seconds: 0.0,
            first_token_seconds: 0.0,
            prompt_seconds,
            decode_seconds,
            total_seconds: tokenize_started.elapsed().as_secs_f64(),
        },
        backend_diagnostics: Vec::new(),
    })
}

struct CachedModel {
    tokenizer: Tokenizer,
    model: CandleModel,
}

#[derive(Default)]
struct DecodeLoopProfile {
    forward: Duration,
    greedy_argmax_launch: Duration,
    greedy_argmax_sync: Duration,
    greedy_scalar_transfer: Duration,
    greedy_total: Duration,
    stochastic_sample: Duration,
    tokenizer: Duration,
    stop_check: Duration,
    callback: Duration,
    total: Duration,
}

impl DecodeLoopProfile {
    fn print(&self, tokens: usize) {
        eprintln!(
            "Candle decode profile\n\
             \n\
             tokens: {tokens}\n\
             \n\
             forward:\n\
               total: {}\n\
               avg/token: {}\n\
             \n\
             deterministic selection:\n\
               total: {}\n\
               avg/token: {}\n\
             \n\
             argmax launch:\n\
               total: {}\n\
               avg/token: {}\n\
             \n\
             argmax sync:\n\
               total: {}\n\
               avg/token: {}\n\
             \n\
             scalar transfer:\n\
               total: {}\n\
               avg/token: {}\n\
             \n\
             stochastic sampling:\n\
               total: {}\n\
               avg/token: {}\n\
             \n\
             tokenizer:\n\
               total: {}\n\
               avg/token: {}\n\
             \n\
             stop check:\n\
               total: {}\n\
               avg/token: {}\n\
             \n\
             callback:\n\
               total: {}\n\
               avg/token: {}\n\
             \n\
             total:\n\
               total: {}\n\
               avg/token: {}",
            format_duration(self.forward),
            format_average_duration(self.forward, tokens),
            format_duration(self.greedy_total),
            format_average_duration(self.greedy_total, tokens),
            format_duration(self.greedy_argmax_launch),
            format_average_duration(self.greedy_argmax_launch, tokens),
            format_duration(self.greedy_argmax_sync),
            format_average_duration(self.greedy_argmax_sync, tokens),
            format_duration(self.greedy_scalar_transfer),
            format_average_duration(self.greedy_scalar_transfer, tokens),
            format_duration(self.stochastic_sample),
            format_average_duration(self.stochastic_sample, tokens),
            format_duration(self.tokenizer),
            format_average_duration(self.tokenizer, tokens),
            format_duration(self.stop_check),
            format_average_duration(self.stop_check, tokens),
            format_duration(self.callback),
            format_average_duration(self.callback, tokens),
            format_duration(self.total),
            format_average_duration(self.total, tokens),
        );
    }
}

#[allow(clippy::large_enum_variant)]
enum TokenSelector {
    Greedy,
    Stochastic(LogitsProcessor),
}

impl TokenSelector {
    fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        if is_deterministic_temperature(temperature) {
            Self::Greedy
        } else {
            Self::Stochastic(LogitsProcessor::new(seed, temperature, top_p))
        }
    }
}

fn is_deterministic_temperature(temperature: Option<f64>) -> bool {
    temperature
        .map(|temperature| temperature < 1e-7)
        .unwrap_or(true)
}

fn format_duration(duration: Duration) -> String {
    format!("{:.6}s", duration.as_secs_f64())
}

fn format_average_duration(duration: Duration, tokens: usize) -> String {
    if tokens == 0 {
        "n/a".to_string()
    } else {
        format!("{:.3}ms", duration.as_secs_f64() * 1000.0 / tokens as f64)
    }
}

fn select_next_token(
    selector: &mut TokenSelector,
    logits: &Tensor,
    device: &Device,
    profile: &mut Option<DecodeLoopProfile>,
) -> Result<u32> {
    match selector {
        TokenSelector::Greedy => select_greedy_token(logits, device, profile.as_mut()),
        TokenSelector::Stochastic(logits_processor) => {
            let started = profile.as_ref().map(|_| Instant::now());
            let next_token = logits_processor.sample(logits)?;
            if let (Some(profile), Some(started)) = (profile.as_mut(), started) {
                profile.stochastic_sample += started.elapsed();
            }
            Ok(next_token)
        }
    }
}

fn select_greedy_token(
    logits: &Tensor,
    device: &Device,
    profile: Option<&mut DecodeLoopProfile>,
) -> Result<u32> {
    if let Some(profile) = profile {
        let total_started = Instant::now();

        let argmax_started = Instant::now();
        let next_token = logits.argmax(D::Minus1)?;
        profile.greedy_argmax_launch += argmax_started.elapsed();

        let sync_started = Instant::now();
        device.synchronize()?;
        profile.greedy_argmax_sync += sync_started.elapsed();

        let scalar_started = Instant::now();
        let next_token = next_token.to_scalar::<u32>()?;
        profile.greedy_scalar_transfer += scalar_started.elapsed();
        profile.greedy_total += total_started.elapsed();

        return Ok(next_token);
    }

    Ok(logits.argmax(D::Minus1)?.to_scalar::<u32>()?)
}

fn stop_reached_from(text: &mut String, stops: &[String], start: usize) -> bool {
    let start = floor_char_boundary(text, start);
    for stop in stops {
        if stop.is_empty() {
            continue;
        }
        if let Some(index) = text[start..].find(stop) {
            text.truncate(start + index);
            return true;
        }
    }
    false
}

fn max_stop_len(stops: &[String]) -> usize {
    stops.iter().map(String::len).max().unwrap_or(0)
}

fn stop_check_start(previous_len: usize, max_stop_len: usize) -> usize {
    previous_len.saturating_sub(max_stop_len.saturating_sub(1))
}

fn stop_guard_emit_len(text: &str, stops: &[String], max_stop_len: usize) -> usize {
    let holdback = stops
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| stop_prefix_suffix_len(text, stop, max_stop_len))
        .max()
        .unwrap_or(0);
    text.len().saturating_sub(holdback)
}

fn stop_prefix_suffix_len(text: &str, stop: &str, max_stop_len: usize) -> Option<usize> {
    let suffix_start = text
        .len()
        .saturating_sub(max_stop_len.saturating_sub(1).min(text.len()));
    let suffix_start = floor_char_boundary(text, suffix_start);
    text[suffix_start..]
        .char_indices()
        .map(|(index, _)| &text[suffix_start + index..])
        .find(|suffix| suffix.len() < stop.len() && stop.starts_with(*suffix))
        .map(str::len)
}

fn floor_char_boundary(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn eos_token_id(tokenizer: &Tokenizer) -> Option<u32> {
    ["<eos>", "</s>", "<|end_of_text|>", "<|im_end|>"]
        .into_iter()
        .find_map(|token| tokenizer.token_to_id(token))
}

fn last_logits(logits: &Tensor) -> Result<Tensor> {
    match logits.rank() {
        1 => Ok(logits.clone()),
        2 => Ok(logits.i(0)?),
        3 => {
            let dims = logits.dims();
            Ok(logits.i((0, dims[1] - 1))?)
        }
        rank => bail!("unexpected logits rank {rank}"),
    }
}

fn load_embedded_gguf_tokenizer(path: &PathBuf) -> Result<Tokenizer> {
    let mut file = fs::File::open(path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let tokenizer_model = gguf_string(&content, "tokenizer.ggml.model")?;

    match tokenizer_model.as_str() {
        "llama" => build_llama_gguf_tokenizer(&content),
        other => bail!(
            "GGUF model embeds tokenizer.ggml.model='{other}', but only embedded llama/SentencePiece tokenizers are supported today; add tokenizer.json beside the model"
        ),
    }
}

fn build_llama_gguf_tokenizer(content: &gguf_file::Content) -> Result<Tokenizer> {
    let tokens = gguf_string_array(content, "tokenizer.ggml.tokens")?;
    let scores = gguf_f64_array(content, "tokenizer.ggml.scores")?;
    if tokens.len() != scores.len() {
        bail!(
            "GGUF tokenizer metadata mismatch: {} tokens but {} scores",
            tokens.len(),
            scores.len()
        );
    }

    let vocab = tokens
        .iter()
        .cloned()
        .zip(scores)
        .collect::<Vec<(String, f64)>>();
    let unk_id = gguf_usize_opt(content, "tokenizer.ggml.unknown_token_id")
        .or_else(|| tokens.iter().position(|token| token == "<unk>"));
    let byte_fallback = tokens
        .iter()
        .any(|token| token.starts_with("<0x") && token.ends_with('>'));
    let model = Unigram::from(vocab, unk_id, byte_fallback)
        .map_err(|err| anyhow!("failed to build GGUF llama tokenizer: {err}"))?;

    let mut tokenizer = Tokenizer::new(model);
    let metaspace = Metaspace::new('▁', PrependScheme::Always, true);
    tokenizer.with_pre_tokenizer(Some(metaspace.clone()));
    if byte_fallback {
        tokenizer.with_decoder(Some(Sequence::new(vec![
            DecoderWrapper::ByteFallback(ByteFallback::new()),
            DecoderWrapper::Metaspace(metaspace),
        ])));
    } else {
        tokenizer.with_decoder(Some(metaspace));
    }

    let special_ids = [
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
        "tokenizer.ggml.separator_token_id",
    ];
    let special_tokens = special_ids
        .into_iter()
        .filter_map(|key| gguf_usize_opt(content, key))
        .filter_map(|id| tokens.get(id))
        .map(|token| AddedToken::from(token.clone(), true))
        .collect::<Vec<_>>();
    if !special_tokens.is_empty() {
        tokenizer.add_special_tokens(&special_tokens);
    }

    add_llama_post_processor(content, &tokens, &mut tokenizer)?;
    Ok(tokenizer)
}

fn add_llama_post_processor(
    content: &gguf_file::Content,
    tokens: &[String],
    tokenizer: &mut Tokenizer,
) -> Result<()> {
    let add_bos = gguf_bool_opt(content, "tokenizer.ggml.add_bos_token").unwrap_or(false);
    let add_eos = gguf_bool_opt(content, "tokenizer.ggml.add_eos_token").unwrap_or(false);
    if !add_bos && !add_eos {
        return Ok(());
    }

    let bos = gguf_usize_opt(content, "tokenizer.ggml.bos_token_id")
        .and_then(|id| tokens.get(id).map(|token| (token.clone(), id as u32)));
    let eos = gguf_usize_opt(content, "tokenizer.ggml.eos_token_id")
        .and_then(|id| tokens.get(id).map(|token| (token.clone(), id as u32)));

    let mut single = Vec::new();
    let mut pair = Vec::new();
    let mut special_tokens = Vec::new();

    if add_bos {
        let (token, id) = bos.context("GGUF tokenizer requests BOS but has no bos_token_id")?;
        single.push(token.clone());
        pair.push(token.clone());
        special_tokens.push((token, id));
    }

    single.push("$A".to_string());
    pair.push("$A:0".to_string());
    pair.push("$B:1".to_string());

    if add_eos {
        let (token, id) = eos.context("GGUF tokenizer requests EOS but has no eos_token_id")?;
        single.push(token.clone());
        pair.push(token.clone());
        special_tokens.push((token, id));
    }

    let processor = TemplateProcessing::builder()
        .try_single(single.join(" "))
        .map_err(|err| anyhow!("failed to configure GGUF tokenizer single template: {err}"))?
        .try_pair(pair.join(" "))
        .map_err(|err| anyhow!("failed to configure GGUF tokenizer pair template: {err}"))?
        .special_tokens(special_tokens)
        .build()
        .map_err(|err| anyhow!("failed to configure GGUF tokenizer templates: {err}"))?;
    tokenizer.with_post_processor(Some(processor));
    Ok(())
}

fn gguf_value<'a>(content: &'a gguf_file::Content, key: &str) -> Result<&'a GgufValue> {
    content
        .metadata
        .get(key)
        .with_context(|| format!("GGUF metadata has no {key}"))
}

fn gguf_string(content: &gguf_file::Content, key: &str) -> Result<String> {
    match gguf_value(content, key)? {
        GgufValue::String(value) => Ok(value.clone()),
        other => bail!("GGUF metadata {key} is not a string: {other:?}"),
    }
}

fn gguf_bool_opt(content: &gguf_file::Content, key: &str) -> Option<bool> {
    match content.metadata.get(key) {
        Some(GgufValue::Bool(value)) => Some(*value),
        _ => None,
    }
}

fn gguf_usize_opt(content: &gguf_file::Content, key: &str) -> Option<usize> {
    match content.metadata.get(key) {
        Some(GgufValue::U8(value)) => Some(*value as usize),
        Some(GgufValue::U16(value)) => Some(*value as usize),
        Some(GgufValue::U32(value)) => Some(*value as usize),
        Some(GgufValue::U64(value)) => usize::try_from(*value).ok(),
        Some(GgufValue::I8(value)) if *value >= 0 => Some(*value as usize),
        Some(GgufValue::I16(value)) if *value >= 0 => Some(*value as usize),
        Some(GgufValue::I32(value)) if *value >= 0 => Some(*value as usize),
        Some(GgufValue::I64(value)) if *value >= 0 => usize::try_from(*value).ok(),
        _ => None,
    }
}

fn gguf_string_array(content: &gguf_file::Content, key: &str) -> Result<Vec<String>> {
    let values = match gguf_value(content, key)? {
        GgufValue::Array(values) => values,
        other => bail!("GGUF metadata {key} is not an array: {other:?}"),
    };
    values
        .iter()
        .enumerate()
        .map(|(index, value)| match value {
            GgufValue::String(value) => Ok(value.clone()),
            other => bail!("GGUF metadata {key}[{index}] is not a string: {other:?}"),
        })
        .collect()
}

fn gguf_f64_array(content: &gguf_file::Content, key: &str) -> Result<Vec<f64>> {
    let values = match gguf_value(content, key)? {
        GgufValue::Array(values) => values,
        other => bail!("GGUF metadata {key} is not an array: {other:?}"),
    };
    values
        .iter()
        .enumerate()
        .map(|(index, value)| match value {
            GgufValue::F32(value) => Ok(*value as f64),
            GgufValue::F64(value) => Ok(*value),
            other => bail!("GGUF metadata {key}[{index}] is not a float: {other:?}"),
        })
        .collect()
}

enum CandleModel {
    Llama(quantized_llama::ModelWeights),
    Qwen2(quantized_qwen2::ModelWeights),
    Qwen3(quantized_qwen3::ModelWeights),
    Phi(quantized_phi::ModelWeights),
    Phi3(quantized_phi3::ModelWeights),
    Gemma3(quantized_gemma3::ModelWeights),
    SafeGemma(gemma::Model),
    SafeGemma2(gemma2::Model),
    SafeQwen2(qwen2::ModelForCausalLM),
    SafeQwen3(qwen3::ModelForCausalLM),
    SafeMistral(mistral::Model),
    SafePhi3(phi3::Model),
    SafeLlama(SafeLlamaModel),
}

struct SafeLlamaModel {
    model: llama::Llama,
    cache: llama::Cache,
    config: llama::Config,
    device: Device,
    dtype: DType,
}

impl SafeLlamaModel {
    fn new(config: llama::Config, vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let model = llama::Llama::load(vb, &config)?;
        let cache = llama::Cache::new(true, dtype, &config, device)?;
        Ok(Self {
            model,
            cache,
            config,
            device: device.clone(),
            dtype,
        })
    }

    fn clear_kv_cache(&mut self) -> Result<()> {
        self.cache = llama::Cache::new(true, self.dtype, &self.config, &self.device)?;
        Ok(())
    }

    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        Ok(self.model.forward(input, index_pos, &mut self.cache)?)
    }
}

impl CandleModel {
    fn clear_kv_cache(&mut self) -> Result<()> {
        match self {
            Self::Llama(_) | Self::Qwen2(_) | Self::Phi(_) | Self::Phi3(_) | Self::Gemma3(_) => {}
            Self::Qwen3(model) => model.clear_kv_cache(),
            Self::SafeGemma(model) => model.clear_kv_cache(),
            Self::SafeGemma2(model) => model.clear_kv_cache(),
            Self::SafeQwen2(model) => model.clear_kv_cache(),
            Self::SafeQwen3(model) => model.clear_kv_cache(),
            Self::SafeMistral(model) => model.clear_kv_cache(),
            Self::SafePhi3(model) => model.clear_kv_cache(),
            Self::SafeLlama(model) => model.clear_kv_cache()?,
        }
        Ok(())
    }

    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        match self {
            Self::Llama(model) => Ok(model.forward(input, index_pos)?),
            Self::Qwen2(model) => Ok(model.forward(input, index_pos)?),
            Self::Qwen3(model) => Ok(model.forward(input, index_pos)?),
            Self::Phi(model) => Ok(model.forward(input, index_pos)?),
            Self::Phi3(model) => Ok(model.forward(input, index_pos)?),
            Self::Gemma3(model) => Ok(model.forward(input, index_pos)?),
            Self::SafeGemma(model) => Ok(model.forward(input, index_pos)?),
            Self::SafeGemma2(model) => Ok(model.forward(input, index_pos)?),
            Self::SafeQwen2(model) => Ok(model.forward(input, index_pos)?),
            Self::SafeQwen3(model) => Ok(model.forward(input, index_pos)?),
            Self::SafeMistral(model) => Ok(model.forward(input, index_pos)?),
            Self::SafePhi3(model) => Ok(model.forward(input, index_pos)?),
            Self::SafeLlama(model) => model.forward(input, index_pos),
        }
    }
}

fn load_candle_model(
    store: &ModelStore,
    manifest: &ModelManifest,
    device: &Device,
) -> Result<CandleModel> {
    match manifest.format {
        ModelFormat::Gguf => {
            let model_path = manifest
                .model_path
                .as_deref()
                .context("manifest has no model_path")?;
            load_gguf_model(
                &store.absolute_model_file(manifest, model_path),
                manifest.architecture.as_deref(),
                device,
            )
        }
        ModelFormat::SafeTensors => load_safetensors_model(store, manifest, device),
        ModelFormat::PyTorch
        | ModelFormat::Onnx
        | ModelFormat::Mlx
        | ModelFormat::TensorRt
        | ModelFormat::OpenVino
        | ModelFormat::TensorFlow
        | ModelFormat::CoreMl => bail!(
            "model '{}' is {:?}: {}; this server can catalog/import it, but execution needs the {} backend to be implemented",
            manifest.id,
            manifest.format,
            manifest.format.backend_status(),
            manifest.format.backend_hint()
        ),
        ModelFormat::Unknown => bail!(
            "model '{}' has unknown format; supported execution formats today are GGUF and safetensors",
            manifest.id
        ),
    }
}

fn load_gguf_model(
    path: &PathBuf,
    architecture: Option<&str>,
    device: &Device,
) -> Result<CandleModel> {
    let mut file = fs::File::open(path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let architecture = architecture
        .map(str::to_string)
        .or_else(|| {
            content
                .metadata
                .get("general.architecture")
                .and_then(|value| value.to_string().ok())
                .cloned()
        })
        .unwrap_or_else(|| "unknown".to_string());

    match architecture.as_str() {
        "llama" => Ok(CandleModel::Llama(
            quantized_llama::ModelWeights::from_gguf(content, &mut file, device)?,
        )),
        "qwen2" => Ok(CandleModel::Qwen2(
            quantized_qwen2::ModelWeights::from_gguf(content, &mut file, device)?,
        )),
        "qwen3" => Ok(CandleModel::Qwen3(
            quantized_qwen3::ModelWeights::from_gguf(content, &mut file, device)?,
        )),
        "phi" | "phi2" => Ok(CandleModel::Phi(quantized_phi::ModelWeights::from_gguf(
            content, &mut file, device,
        )?)),
        "phi3" => Ok(CandleModel::Phi3(quantized_phi3::ModelWeights::from_gguf(
            false, content, &mut file, device,
        )?)),
        "gemma3" => Ok(CandleModel::Gemma3(
            quantized_gemma3::ModelWeights::from_gguf(content, &mut file, device)?,
        )),
        other => bail!(
            "unsupported GGUF architecture '{other}' for Candle backend; supported: llama, qwen2, qwen3, phi/phi2, phi3, gemma3"
        ),
    }
}

fn load_safetensors_model(
    store: &ModelStore,
    manifest: &ModelManifest,
    device: &Device,
) -> Result<CandleModel> {
    let architecture = manifest
        .architecture
        .as_deref()
        .context("safetensors manifest has no architecture; add config.json with model_type")?;
    let config_path = manifest
        .config_path
        .as_deref()
        .context("safetensors model requires config.json")?;
    let config_path = store.absolute_model_file(manifest, config_path);
    let weight_paths = safetensor_paths(store, manifest)?;
    if weight_paths.is_empty() {
        bail!(
            "safetensors model '{}' has no .safetensors files",
            manifest.id
        );
    }

    let dtype = safetensors_load_dtype(&config_path, device)?;
    eprintln!("Loading safetensors weights as {dtype:?}");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_paths, dtype, device)? };

    match architecture {
        "llama" => {
            let cfg: llama::LlamaConfig = read_config(&config_path)?;
            let cfg = cfg.into_config(false);
            Ok(CandleModel::SafeLlama(SafeLlamaModel::new(
                cfg, vb, device, dtype,
            )?))
        }
        "gemma" => {
            let cfg: gemma::Config = read_config(&config_path)?;
            Ok(CandleModel::SafeGemma(gemma::Model::new(false, &cfg, vb)?))
        }
        "gemma2" => {
            let cfg: gemma2::Config = read_config(&config_path)?;
            Ok(CandleModel::SafeGemma2(gemma2::Model::new(
                false, &cfg, vb,
            )?))
        }
        "qwen2" => {
            let cfg: qwen2::Config = read_config(&config_path)?;
            Ok(CandleModel::SafeQwen2(qwen2::ModelForCausalLM::new(
                &cfg, vb,
            )?))
        }
        "qwen3" => {
            let cfg: qwen3::Config = read_config(&config_path)?;
            Ok(CandleModel::SafeQwen3(qwen3::ModelForCausalLM::new(
                &cfg, vb,
            )?))
        }
        "mistral" => {
            let cfg: mistral::Config = read_config(&config_path)?;
            Ok(CandleModel::SafeMistral(mistral::Model::new(&cfg, vb)?))
        }
        "phi3" => {
            let cfg: phi3::Config = read_config(&config_path)?;
            Ok(CandleModel::SafePhi3(phi3::Model::new(&cfg, vb)?))
        }
        other => bail!(
            "Safetensors architecture '{other}' is not supported by Candle backend yet; supported: llama, gemma, gemma2, qwen2, qwen3, mistral, phi3"
        ),
    }
}

fn safetensor_paths(store: &ModelStore, manifest: &ModelManifest) -> Result<Vec<PathBuf>> {
    let mut paths = manifest
        .files
        .iter()
        .filter(|file| file.path.ends_with(".safetensors"))
        .map(|file| store.absolute_model_file(manifest, &file.path))
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

#[derive(serde::Deserialize)]
struct DTypeConfig {
    torch_dtype: Option<String>,
    dtype: Option<String>,
}

fn safetensors_load_dtype(config_path: &PathBuf, device: &Device) -> Result<DType> {
    let config: DTypeConfig = read_config(config_path)?;
    let dtype = config
        .torch_dtype
        .as_deref()
        .or(config.dtype.as_deref())
        .map(|dtype| dtype.to_ascii_lowercase());

    Ok(match dtype.as_deref() {
        Some("bfloat16" | "bf16") if device.supports_bf16() => DType::BF16,
        Some("bfloat16" | "bf16") => DType::F32,
        Some("float16" | "fp16" | "f16" | "half") if device.is_cuda() || device.is_metal() => {
            DType::F16
        }
        Some("float16" | "fp16" | "f16" | "half") => DType::F32,
        Some("float32" | "fp32" | "f32") => DType::F32,
        _ => DType::F32,
    })
}

fn read_config<T: DeserializeOwned>(path: &PathBuf) -> Result<T> {
    let data =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&data).with_context(|| format!("failed to parse {}", path.display()))
}

#[derive(Clone)]
struct StreamEmitter {
    tx: Sender<Result<GenerateStreamEvent, String>>,
    buffer: Arc<Mutex<String>>,
    granularity: StreamGranularity,
}

impl StreamEmitter {
    fn new(
        tx: Sender<Result<GenerateStreamEvent, String>>,
        granularity: StreamGranularity,
    ) -> Self {
        Self {
            tx,
            buffer: Arc::new(Mutex::new(String::new())),
            granularity,
        }
    }

    fn push(&self, text: String) {
        if text.is_empty() {
            return;
        }

        if self.granularity == StreamGranularity::Token {
            let _ = self
                .tx
                .blocking_send(Ok(GenerateStreamEvent::TextChunk(text)));
            return;
        }

        let mut buffer = self.buffer.lock().expect("chunk buffer poisoned");
        buffer.push_str(&text);
        if should_flush_chunk(&buffer, &text) {
            let chunk = std::mem::take(&mut *buffer);
            drop(buffer);
            let _ = self
                .tx
                .blocking_send(Ok(GenerateStreamEvent::TextChunk(chunk)));
        }
    }

    fn flush(&self) {
        if self.granularity == StreamGranularity::Token {
            return;
        }

        let mut buffer = self.buffer.lock().expect("chunk buffer poisoned");
        if buffer.is_empty() {
            return;
        }
        let chunk = std::mem::take(&mut *buffer);
        drop(buffer);
        let _ = self
            .tx
            .blocking_send(Ok(GenerateStreamEvent::TextChunk(chunk)));
    }
}

fn should_flush_chunk(buffer: &str, latest_piece: &str) -> bool {
    buffer.len() >= 48
        || latest_piece.contains('\n')
        || buffer
            .chars()
            .last()
            .map(|ch| matches!(ch, '.' | '!' | '?' | ':' | ';'))
            .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_guard_holds_partial_stop_prefix() {
        let stops = vec!["\nHuman:".to_string()];
        let text = "Rust is fast.\nHum";

        assert_eq!(
            &text[..stop_guard_emit_len(text, &stops, max_stop_len(&stops))],
            "Rust is fast."
        );
    }

    #[test]
    fn stop_guard_releases_newline_when_not_stop_prefix() {
        let stops = vec!["\nHuman:".to_string()];
        let text = "Rust is fast.\nNext sentence.";

        assert_eq!(
            stop_guard_emit_len(text, &stops, max_stop_len(&stops)),
            text.len()
        );
    }

    #[test]
    fn stop_reached_from_finds_stop_spanning_previous_suffix() {
        let stops = vec!["\nHuman:".to_string()];
        let mut text = "Rust is fast.\nHuman: next".to_string();
        let start = stop_check_start("Rust is fast.\nHum".len(), max_stop_len(&stops));

        assert!(stop_reached_from(&mut text, &stops, start));
        assert_eq!(text, "Rust is fast.");
    }

    #[test]
    fn token_selector_uses_greedy_for_zero_temperature() {
        assert!(matches!(
            TokenSelector::new(42, Some(0.0), None),
            TokenSelector::Greedy
        ));
    }

    #[test]
    fn token_selector_uses_stochastic_for_nonzero_temperature() {
        assert!(matches!(
            TokenSelector::new(42, Some(0.8), None),
            TokenSelector::Stochastic(_)
        ));
    }
}
