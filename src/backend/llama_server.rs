use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, VecDeque},
    env, fs,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::{
    ChatGenerationSession, GenerateRequest, GenerateResponse, GenerateStream, GenerateStreamEvent,
    GenerationBackend, GenerationTimings, LlamaCppMode, LlamaRuntimeOptions,
};
use crate::model_store::{ModelFormat, ModelManifest, ModelStore};

const DEFAULT_CTX_SIZE: usize = 4096;
const DEFAULT_BATCH_SIZE: usize = 2048;
const DEFAULT_UBATCH_SIZE: u32 = 512;
const HEALTH_TIMEOUT: Duration = Duration::from_secs(180);
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Clone)]
pub struct LlamaServerBackend {
    store: ModelStore,
    mode: LlamaCppMode,
    runtime_options: LlamaRuntimeOptions,
    servers: Arc<Mutex<HashMap<String, Arc<LlamaServerProcess>>>>,
}

struct LlamaServerProcess {
    child: Mutex<Child>,
    executable: PathBuf,
    discovery_source: String,
    args: Vec<String>,
    model_path: PathBuf,
    url: String,
    pid: u32,
    mode: LlamaCppMode,
    log_tail: Arc<Mutex<VecDeque<String>>>,
}

struct LlamaServerChatSession {
    server: Arc<LlamaServerProcess>,
}

#[derive(Debug, Clone)]
pub struct LlamaServerDiscovery {
    pub mode: LlamaCppMode,
    pub path: Option<PathBuf>,
    pub source: String,
    pub attempts: Vec<LlamaServerDiscoveryAttempt>,
}

#[derive(Debug, Clone)]
pub struct LlamaServerDiscoveryAttempt {
    pub label: String,
    pub path: Option<PathBuf>,
    pub exists: bool,
}

#[derive(Debug, Clone)]
pub struct BackendDoctorCheck {
    pub name: String,
    pub ok: bool,
    pub detail: String,
}

#[derive(Default)]
struct ServerCompletion {
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    prompt_seconds: f64,
    decode_seconds: f64,
    first_token_seconds: f64,
    finish_reason: String,
}

impl LlamaServerBackend {
    pub fn new(
        store: ModelStore,
        mode: LlamaCppMode,
        runtime_options: LlamaRuntimeOptions,
    ) -> Self {
        Self {
            store,
            mode,
            runtime_options,
            servers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn probe(store: &ModelStore, mode: LlamaCppMode) -> Result<String> {
        let discovery = require_llama_server(store, mode)?;
        let executable = discovery
            .path
            .as_ref()
            .context("llama-server discovery had no executable path")?;
        Ok(format!(
            "llama.cpp server {} at {}",
            display_name(mode),
            executable.display()
        ))
    }

    pub fn discover(store: &ModelStore, mode: LlamaCppMode) -> LlamaServerDiscovery {
        discover_llama_server(store, mode)
    }

    pub fn missing_message(store: &ModelStore, mode: LlamaCppMode) -> String {
        missing_llama_server_message(&discover_llama_server(store, mode))
    }

    fn cached_server(
        &self,
        manifest: &ModelManifest,
    ) -> Result<(Arc<LlamaServerProcess>, bool, f64)> {
        if manifest.format != ModelFormat::Gguf {
            bail!(
                "llama.cpp server {} backend supports GGUF models only",
                display_name(self.mode)
            );
        }

        let model_path = manifest
            .model_path
            .as_deref()
            .context("GGUF manifest has no model_path")?;
        let absolute_model_path = self.store.absolute_model_file(manifest, model_path);
        let key = format!(
            "{}:{}:{}:{}:{}:{}:{}",
            manifest.id,
            absolute_model_path.display(),
            label(self.mode),
            self.runtime_options.ctx_size.unwrap_or(DEFAULT_CTX_SIZE),
            self.runtime_options
                .batch_size
                .unwrap_or(DEFAULT_BATCH_SIZE),
            self.runtime_options
                .ubatch_size
                .unwrap_or(DEFAULT_UBATCH_SIZE),
            self.runtime_options
                .gpu_layers
                .unwrap_or_else(|| gpu_layers(self.mode))
        );

        if let Some(server) = self
            .servers
            .lock()
            .map_err(|_| anyhow!("llama-server cache mutex poisoned"))?
            .get(&key)
            .cloned()
            && server.is_running()
        {
            return Ok((server, true, 0.0));
        }

        let started = Instant::now();
        let server = Arc::new(LlamaServerProcess::start(
            &self.store,
            self.mode,
            &absolute_model_path,
            &self.runtime_options,
        )?);
        let load_seconds = started.elapsed().as_secs_f64();
        self.servers
            .lock()
            .map_err(|_| anyhow!("llama-server cache mutex poisoned"))?
            .insert(key, server.clone());
        Ok((server, false, load_seconds))
    }

    fn generate_inner(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
        tx: Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
    ) -> Result<GenerateResponse> {
        if !request.image_urls.is_empty() {
            bail!(
                "llama.cpp server backend is text-only for now; use a VLM-capable backend/model for image inputs"
            );
        }

        let total_started = Instant::now();
        let (server, reused, load_seconds) = self.cached_server(manifest)?;
        server.print_debug(&request, reused);
        let completion = server.complete(&request, tx)?;
        Ok(GenerateResponse {
            text: completion.text,
            prompt_tokens: completion.prompt_tokens,
            completion_tokens: completion.completion_tokens,
            finish_reason: completion.finish_reason,
            timings: GenerationTimings {
                load_seconds,
                warmup_seconds: 0.0,
                first_token_seconds: completion.first_token_seconds,
                prompt_seconds: completion.prompt_seconds,
                decode_seconds: completion.decode_seconds,
                total_seconds: total_started.elapsed().as_secs_f64(),
            },
        })
    }
}

impl GenerationBackend for LlamaServerBackend {
    fn prepare(&self, manifest: &ModelManifest) -> Result<()> {
        self.cached_server(manifest).map(|_| ())
    }

    fn start_chat_session(
        &self,
        manifest: &ModelManifest,
        _seed: Option<u64>,
    ) -> Result<Option<Box<dyn ChatGenerationSession>>> {
        if manifest.format != ModelFormat::Gguf {
            return Ok(None);
        }
        let (server, _, _) = self.cached_server(manifest)?;
        Ok(Some(Box::new(LlamaServerChatSession { server })))
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
            send_stream_result(tx, result);
        });
        Box::pin(ReceiverStream::new(rx))
    }
}

impl ChatGenerationSession for LlamaServerChatSession {
    fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let total_started = Instant::now();
        self.server.print_debug(&request, true);
        let completion = self.server.complete(&request, None)?;
        Ok(GenerateResponse {
            text: completion.text,
            prompt_tokens: completion.prompt_tokens,
            completion_tokens: completion.completion_tokens,
            finish_reason: completion.finish_reason,
            timings: GenerationTimings {
                load_seconds: 0.0,
                warmup_seconds: 0.0,
                first_token_seconds: completion.first_token_seconds,
                prompt_seconds: completion.prompt_seconds,
                decode_seconds: completion.decode_seconds,
                total_seconds: total_started.elapsed().as_secs_f64(),
            },
        })
    }

    fn generate_stream(&self, request: GenerateRequest) -> GenerateStream {
        let server = self.server.clone();
        let (tx, rx) = mpsc::channel(16);
        tokio::task::spawn_blocking(move || {
            let total_started = Instant::now();
            server.print_debug(&request, true);
            let result = server
                .complete(&request, Some(tx.clone()))
                .map(|completion| GenerateResponse {
                    text: completion.text,
                    prompt_tokens: completion.prompt_tokens,
                    completion_tokens: completion.completion_tokens,
                    finish_reason: completion.finish_reason,
                    timings: GenerationTimings {
                        load_seconds: 0.0,
                        warmup_seconds: 0.0,
                        first_token_seconds: completion.first_token_seconds,
                        prompt_seconds: completion.prompt_seconds,
                        decode_seconds: completion.decode_seconds,
                        total_seconds: total_started.elapsed().as_secs_f64(),
                    },
                });
            send_stream_result(tx, result);
        });
        Box::pin(ReceiverStream::new(rx))
    }
}

impl LlamaServerProcess {
    fn start(
        store: &ModelStore,
        mode: LlamaCppMode,
        model_path: &PathBuf,
        runtime_options: &LlamaRuntimeOptions,
    ) -> Result<Self> {
        let discovery = require_llama_server(store, mode)?;
        let executable = discovery
            .path
            .clone()
            .context("llama-server discovery had no executable path")?;
        let port = free_local_port()?;
        let url = format!("http://127.0.0.1:{port}");
        let supported = supported_args(&executable);
        let args = llama_server_args(mode, model_path, port, runtime_options, &supported);

        eprintln!("Using llama.cpp server {} backend", display_name(mode));
        let mut command = Command::new(&executable);
        command.args(&args);
        let log_tail = Arc::new(Mutex::new(VecDeque::new()));
        if env_true("WERK_LLAMA_LOG") {
            command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
        } else {
            command.stdout(Stdio::piped()).stderr(Stdio::piped());
        }
        let mut child = command
            .spawn()
            .with_context(|| format!("failed to start llama-server at {}", executable.display()))?;
        if !env_true("WERK_LLAMA_LOG") {
            if let Some(stdout) = child.stdout.take() {
                spawn_log_tail_reader("stdout", stdout, log_tail.clone());
            }
            if let Some(stderr) = child.stderr.take() {
                spawn_log_tail_reader("stderr", stderr, log_tail.clone());
            }
        }
        let pid = child.id();
        let process = Self {
            child: Mutex::new(child),
            executable,
            discovery_source: discovery.source,
            args,
            model_path: model_path.clone(),
            url,
            pid,
            mode,
            log_tail,
        };
        process.wait_until_ready()?;
        Ok(process)
    }

    fn complete(
        &self,
        request: &GenerateRequest,
        tx: Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
    ) -> Result<ServerCompletion> {
        let started = Instant::now();
        let body = completion_body(request);
        let mut stream = post_json(&self.url, "/completion", &body)?;
        let mut completion = ServerCompletion {
            finish_reason: "length".to_string(),
            ..Default::default()
        };
        let mut sse = SseAccumulator::default();

        stream_body(&mut stream, |bytes| {
            sse.push(bytes, |event| {
                if event == "[DONE]" {
                    completion.finish_reason = "stop".to_string();
                    return Ok(());
                }
                let value: Value = serde_json::from_str(event)
                    .with_context(|| format!("invalid llama-server SSE event: {event}"))?;
                update_completion_from_event(&mut completion, &value);
                if let Some(chunk) = value.get("content").and_then(Value::as_str)
                    && !chunk.is_empty()
                {
                    if completion.first_token_seconds <= 0.0 {
                        completion.first_token_seconds = started.elapsed().as_secs_f64();
                    }
                    completion.text.push_str(chunk);
                    send_text_chunk(&tx, chunk.to_string())?;
                }
                Ok(())
            })
        })?;

        finalize_completion_stats(&mut completion, request, started.elapsed().as_secs_f64());
        Ok(completion)
    }

    fn wait_until_ready(&self) -> Result<()> {
        let started = Instant::now();
        loop {
            if let Some(status) = self.try_wait_status()? {
                bail!(
                    "llama-server exited before becoming healthy ({status}){}",
                    self.formatted_log_tail()
                );
            }
            if let Ok(response) = get(&self.url, "/health")
                && response.status == 200
            {
                return Ok(());
            }
            if started.elapsed() > HEALTH_TIMEOUT {
                bail!(
                    "timed out waiting for llama-server {} at {}{}",
                    self.pid,
                    self.url,
                    self.formatted_log_tail()
                );
            }
            std::thread::sleep(HEALTH_POLL_INTERVAL);
        }
    }

    fn is_running(&self) -> bool {
        matches!(self.try_wait_status(), Ok(None))
    }

    fn try_wait_status(&self) -> Result<Option<ExitStatus>> {
        let mut child = self
            .child
            .lock()
            .map_err(|_| anyhow!("llama-server child mutex poisoned"))?;
        Ok(child.try_wait()?)
    }

    fn formatted_log_tail(&self) -> String {
        let Ok(tail) = self.log_tail.lock() else {
            return String::new();
        };
        if tail.is_empty() {
            return String::new();
        }
        format!(
            "\n\nllama-server output tail:\n{}",
            tail.iter().cloned().collect::<Vec<_>>().join("\n")
        )
    }

    fn print_debug(&self, request: &GenerateRequest, reused: bool) {
        if !request.debug {
            return;
        }
        eprintln!("selected backend: {}", label(self.mode));
        eprintln!(
            "actual engine: llama.cpp server {} backend",
            display_name(self.mode)
        );
        eprintln!(
            "llama-server executable path: {}",
            self.executable.display()
        );
        eprintln!("discovery source: {}", self.discovery_source);
        eprintln!("full llama-server args: {}", shell_join(&self.args));
        eprintln!("model path: {}", self.model_path.display());
        eprintln!("server PID: {}", self.pid);
        eprintln!("server URL: {}", self.url);
        eprintln!("reused existing server: {reused}");
    }
}

impl Drop for LlamaServerProcess {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

struct HttpResponse {
    status: u16,
    headers: Vec<(String, String)>,
    reader: BufReader<TcpStream>,
}

#[derive(Default)]
struct SseAccumulator {
    pending: Vec<u8>,
}

impl SseAccumulator {
    fn push<F>(&mut self, bytes: &[u8], mut on_event: F) -> Result<()>
    where
        F: FnMut(&str) -> Result<()>,
    {
        self.pending.extend_from_slice(bytes);
        while let Some(index) = find_sse_boundary(&self.pending) {
            let event = self.pending.drain(..index).collect::<Vec<_>>();
            while matches!(self.pending.first(), Some(b'\r' | b'\n')) {
                self.pending.remove(0);
            }
            let event = String::from_utf8_lossy(&event);
            for line in event.lines() {
                if let Some(data) = line.strip_prefix("data:") {
                    on_event(data.trim())?;
                }
            }
        }
        Ok(())
    }
}

fn llama_server_args(
    mode: LlamaCppMode,
    model_path: &PathBuf,
    port: u16,
    runtime_options: &LlamaRuntimeOptions,
    supported: &SupportedArgs,
) -> Vec<String> {
    let mut args = vec![
        "--model".to_string(),
        model_path.display().to_string(),
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        port.to_string(),
        "-ngl".to_string(),
        runtime_options
            .gpu_layers
            .unwrap_or_else(|| gpu_layers(mode))
            .to_string(),
        "-c".to_string(),
        runtime_options
            .ctx_size
            .unwrap_or(DEFAULT_CTX_SIZE)
            .to_string(),
        "-b".to_string(),
        runtime_options
            .batch_size
            .unwrap_or(DEFAULT_BATCH_SIZE)
            .to_string(),
        "-ub".to_string(),
        runtime_options
            .ubatch_size
            .unwrap_or(DEFAULT_UBATCH_SIZE)
            .to_string(),
        "-np".to_string(),
        "1".to_string(),
    ];

    if supported.flash_attn {
        args.push("-fa".to_string());
        args.push(
            match runtime_options.flash_attn {
                Some(true) => "on",
                Some(false) => "off",
                None => "auto",
            }
            .to_string(),
        );
    }
    if supported.kv_offload && runtime_options.kv_offload != Some(false) {
        args.push("-kvo".to_string());
    }
    if let Some(main_gpu) = runtime_options.main_gpu {
        args.push("-mg".to_string());
        args.push(main_gpu.to_string());
    }
    if let Some(cache_type) = runtime_options.kv_cache_type {
        args.push("-ctk".to_string());
        args.push(cache_type.label().to_string());
        args.push("-ctv".to_string());
        args.push(cache_type.label().to_string());
    }
    if let Some(threads) = runtime_options.threads {
        args.push("-t".to_string());
        args.push(threads.to_string());
    }
    if let Some(threads_batch) = runtime_options.threads_batch {
        args.push("-tb".to_string());
        args.push(threads_batch.to_string());
    }
    if supported.perf {
        args.push("--perf".to_string());
    }
    if supported.log_disable && !env_true("WERK_LLAMA_LOG") {
        args.push("--log-disable".to_string());
    }
    if let Ok(extra) = env::var("WERK_LLAMA_ARGS") {
        args.extend(split_args(&extra));
    }
    args
}

fn completion_body(request: &GenerateRequest) -> Value {
    let mut body = json!({
        "prompt": request.prompt,
        "n_predict": request.max_tokens,
        "stream": true,
        "cache_prompt": true,
    });
    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(top_p) = request.top_p {
        body["top_p"] = json!(top_p);
    }
    if !request.stop.is_empty() {
        body["stop"] = json!(request.stop);
    }
    if let Some(seed) = request.seed {
        body["seed"] = json!(seed);
    }
    body
}

fn update_completion_from_event(completion: &mut ServerCompletion, value: &Value) {
    if value.get("stop").and_then(Value::as_bool).unwrap_or(false) {
        completion.finish_reason = "stop".to_string();
    }
    if let Some(tokens) = value.get("tokens").and_then(Value::as_array) {
        completion.completion_tokens += tokens.len();
    } else if value
        .get("content")
        .and_then(Value::as_str)
        .map(|content| !content.is_empty())
        .unwrap_or(false)
    {
        completion.completion_tokens += 1;
    }

    if let Some(value) = usize_field_any(
        value,
        &[
            &["tokens_predicted"],
            &["predicted_n"],
            &["n_predicted"],
            &["timings", "predicted_n"],
            &["timings", "n_predicted"],
            &["timings", "tokens_predicted"],
        ],
    ) && should_update_count(completion.completion_tokens, value)
    {
        completion.completion_tokens = value;
    }
    if let Some(value) = usize_field_any(
        value,
        &[
            &["tokens_evaluated"],
            &["prompt_n"],
            &["n_prompt"],
            &["timings", "prompt_n"],
            &["timings", "n_prompt"],
            &["timings", "tokens_evaluated"],
        ],
    ) && should_update_count(completion.prompt_tokens, value)
    {
        completion.prompt_tokens = value;
    }
    if let Some(ms) = number_field_any(value, &[&["timings", "prompt_ms"], &["prompt_ms"]])
        && should_update_seconds(completion.prompt_seconds, ms / 1000.0)
    {
        completion.prompt_seconds = ms / 1000.0;
    }
    if let Some(ms) = number_field_any(value, &[&["timings", "predicted_ms"], &["predicted_ms"]])
        && should_update_seconds(completion.decode_seconds, ms / 1000.0)
    {
        completion.decode_seconds = ms / 1000.0;
    }
}

fn finalize_completion_stats(
    completion: &mut ServerCompletion,
    request: &GenerateRequest,
    elapsed_seconds: f64,
) {
    if completion.prompt_tokens == 0 && !request.prompt.trim().is_empty() {
        completion.prompt_tokens = estimate_tokens(&request.prompt);
    }
    if completion.prompt_seconds <= 0.0 && completion.first_token_seconds > 0.0 {
        completion.prompt_seconds = completion.first_token_seconds;
    }
    if completion.decode_seconds <= 0.0 {
        completion.decode_seconds = if completion.first_token_seconds > 0.0
            && elapsed_seconds > completion.first_token_seconds
        {
            elapsed_seconds - completion.first_token_seconds
        } else {
            elapsed_seconds
        };
    }
    if completion.completion_tokens == 0 {
        completion.completion_tokens = estimate_tokens(&completion.text);
    }
}

fn should_update_count(current: usize, next: usize) -> bool {
    next > 0 || current == 0
}

fn should_update_seconds(current: f64, next: f64) -> bool {
    next > 0.0 || current <= 0.0
}

fn usize_field_any(value: &Value, paths: &[&[&str]]) -> Option<usize> {
    number_field_any(value, paths).map(|value| value as usize)
}

fn number_field_any(value: &Value, paths: &[&[&str]]) -> Option<f64> {
    paths.iter().find_map(|path| number_field(value, path))
}

fn number_field(value: &Value, path: &[&str]) -> Option<f64> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current
        .as_f64()
        .or_else(|| current.as_u64().map(|v| v as f64))
}

fn get(base_url: &str, path: &str) -> Result<HttpResponse> {
    request(base_url, path, "GET", None)
}

fn post_json(base_url: &str, path: &str, body: &Value) -> Result<HttpResponse> {
    request(base_url, path, "POST", Some(body))
}

fn request(base_url: &str, path: &str, method: &str, body: Option<&Value>) -> Result<HttpResponse> {
    let (_, host, port) = parse_local_url(base_url)?;
    let mut stream = TcpStream::connect((host.as_str(), port))
        .with_context(|| format!("failed to connect to llama-server at {base_url}"))?;
    stream.set_nodelay(true).ok();
    let body_text = body.map(serde_json::to_string).transpose()?;
    let mut request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\nAccept: text/event-stream\r\n"
    );
    if let Some(body_text) = &body_text {
        request.push_str("Content-Type: application/json\r\n");
        request.push_str(&format!("Content-Length: {}\r\n", body_text.len()));
    }
    request.push_str("\r\n");
    stream.write_all(request.as_bytes())?;
    if let Some(body_text) = body_text {
        stream.write_all(body_text.as_bytes())?;
    }
    stream.flush()?;

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line)?;
    let status = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(|| anyhow!("invalid HTTP response from llama-server: {status_line:?}"))?;
    let mut headers = Vec::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
        }
    }
    if status >= 400 {
        let mut text = String::new();
        let _ = reader.read_to_string(&mut text);
        bail!("llama-server HTTP {status}: {}", text.trim());
    }
    Ok(HttpResponse {
        status,
        headers,
        reader,
    })
}

fn stream_body<F>(response: &mut HttpResponse, mut on_bytes: F) -> Result<()>
where
    F: FnMut(&[u8]) -> Result<()>,
{
    if header_contains(&response.headers, "transfer-encoding", "chunked") {
        loop {
            let mut size_line = String::new();
            response.reader.read_line(&mut size_line)?;
            let size_text = size_line
                .trim()
                .split_once(';')
                .map(|(size, _)| size)
                .unwrap_or_else(|| size_line.trim());
            let size = usize::from_str_radix(size_text, 16)
                .with_context(|| format!("invalid chunk size from llama-server: {size_text}"))?;
            if size == 0 {
                break;
            }
            let mut chunk = vec![0u8; size];
            response.reader.read_exact(&mut chunk)?;
            on_bytes(&chunk)?;
            let mut crlf = [0u8; 2];
            response.reader.read_exact(&mut crlf)?;
        }
    } else {
        let mut buffer = [0u8; 8192];
        loop {
            let n = response.reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            on_bytes(&buffer[..n])?;
        }
    }
    Ok(())
}

fn parse_local_url(url: &str) -> Result<(String, String, u16)> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| anyhow!("only local http llama-server URLs are supported: {url}"))?;
    let (host_port, _) = rest.split_once('/').unwrap_or((rest, ""));
    let (host, port) = host_port
        .rsplit_once(':')
        .ok_or_else(|| anyhow!("llama-server URL has no port: {url}"))?;
    Ok(("http".to_string(), host.to_string(), port.parse()?))
}

fn header_contains(headers: &[(String, String)], name: &str, needle: &str) -> bool {
    headers.iter().any(|(header, value)| {
        header.eq_ignore_ascii_case(name) && value.to_ascii_lowercase().contains(needle)
    })
}

#[derive(Default)]
struct SupportedArgs {
    flash_attn: bool,
    kv_offload: bool,
    perf: bool,
    log_disable: bool,
}

fn supported_args(executable: &PathBuf) -> SupportedArgs {
    let Ok(output) = Command::new(executable).arg("--help").output() else {
        return SupportedArgs::default();
    };
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    SupportedArgs {
        flash_attn: text.contains("--flash-attn") || text.contains("-fa,"),
        kv_offload: text.contains("--kv-offload") || text.contains("-kvo,"),
        perf: text.contains("--perf"),
        log_disable: text.contains("--log-disable"),
    }
}

pub fn install_managed_llama_server(store: &ModelStore, mode: LlamaCppMode) -> Result<PathBuf> {
    let root = managed_backend_dir(store, mode);
    let source_dir = root.join("llama.cpp");
    let build_dir = root.join("build");
    fs::create_dir_all(&root)
        .with_context(|| format!("failed to create backend cache {}", root.display()))?;

    if !source_dir.join(".git").is_dir() {
        if source_dir.exists() {
            bail!(
                "managed llama.cpp source directory exists but is not a git checkout: {}",
                source_dir.display()
            );
        }
        eprintln!("Cloning llama.cpp into {}", source_dir.display());
        run_command(
            Command::new("git")
                .arg("clone")
                .arg("--depth")
                .arg("1")
                .arg("https://github.com/ggml-org/llama.cpp")
                .arg(&source_dir),
            "failed to clone llama.cpp; install git and check network access",
        )?;
    } else {
        eprintln!(
            "Using existing llama.cpp checkout at {}",
            source_dir.display()
        );
    }

    if build_dir.join("CMakeCache.txt").is_file() {
        eprintln!(
            "Resetting previous llama.cpp build cache at {}",
            build_dir.display()
        );
        fs::remove_dir_all(&build_dir).with_context(|| {
            format!(
                "failed to reset previous llama.cpp build cache {}",
                build_dir.display()
            )
        })?;
    }

    eprintln!("Configuring llama.cpp {} build", display_name(mode));
    let mut configure = Command::new("cmake");
    configure
        .arg("-B")
        .arg(&build_dir)
        .arg("-S")
        .arg(&source_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DLLAMA_BUILD_SERVER=ON")
        .arg("-DLLAMA_BUILD_TOOLS=ON")
        .arg("-DLLAMA_BUILD_UI=OFF")
        .arg("-DLLAMA_BUILD_TESTS=OFF")
        .arg("-DLLAMA_BUILD_EXAMPLES=OFF")
        .arg("-DLLAMA_BUILD_APP=OFF");
    match mode {
        LlamaCppMode::Cuda => {
            configure.arg("-DGGML_CUDA=ON");
            if let Some(arch) = cuda_architecture() {
                eprintln!("Using CUDA architecture {arch}");
                configure.arg(format!("-DCMAKE_CUDA_ARCHITECTURES={arch}"));
            }
            if let Some(host_compiler) = cuda_host_compiler() {
                eprintln!("Using CUDA host compiler {}", host_compiler.display());
                configure.arg(format!(
                    "-DCMAKE_CUDA_HOST_COMPILER={}",
                    host_compiler.display()
                ));
                if let Some(c_compiler) = matching_c_compiler(&host_compiler) {
                    configure.arg(format!("-DCMAKE_C_COMPILER={}", c_compiler.display()));
                }
                configure.arg(format!("-DCMAKE_CXX_COMPILER={}", host_compiler.display()));
            }
        }
        LlamaCppMode::Vulkan => {
            configure.arg("-DGGML_VULKAN=ON");
        }
        LlamaCppMode::Cpu => {}
    }
    run_command(
        &mut configure,
        "failed to configure llama.cpp with CMake; install cmake and the required native toolchain",
    )?;

    eprintln!("Building llama-server");
    run_command(
        Command::new("cmake")
            .arg("--build")
            .arg(&build_dir)
            .arg("--config")
            .arg("Release")
            .arg("--target")
            .arg("llama-server")
            .arg("-j"),
        "failed to build llama-server with CMake",
    )?;

    let executable = find_managed_server(store, mode).ok_or_else(|| {
        anyhow!(
            "llama-server build finished, but no executable was found under {}",
            root.display()
        )
    })?;
    validate_llama_server(&executable, mode)?;
    fs::write(
        managed_path_file(store, mode),
        executable.display().to_string(),
    )?;
    Ok(executable)
}

pub fn backend_doctor_checks(store: &ModelStore) -> Vec<BackendDoctorCheck> {
    let mut checks = vec![
        command_check("git", &["--version"], "required to clone llama.cpp"),
        command_check(
            "cmake",
            &["--version"],
            "required to configure/build llama.cpp",
        ),
        command_check(
            "nvidia-smi",
            &[],
            "required to verify NVIDIA driver visibility for CUDA backends",
        ),
        command_check(
            "nvcc",
            &["--version"],
            "required to build llama.cpp CUDA backends from source",
        ),
    ];
    checks.push(cuda_host_compiler_check());
    checks.push(cuda_architecture_check());
    checks.push(cache_write_check(store));
    checks
}

pub fn managed_backend_dir(store: &ModelStore, mode: LlamaCppMode) -> PathBuf {
    store
        .home()
        .join("backends")
        .join(install_target_name(mode))
}

pub fn llama_server_help_ok(path: &Path) -> bool {
    Command::new(path)
        .arg("--help")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn validate_llama_server(path: &Path, mode: LlamaCppMode) -> Result<()> {
    let output = Command::new(path)
        .arg("--help")
        .output()
        .with_context(|| format!("failed to run {} --help", path.display()))?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !output.status.success() {
        bail!(
            "llama-server --help failed for {} with {}:\n{}",
            path.display(),
            output.status,
            tail_lines(&text, 30)
        );
    }
    if mode == LlamaCppMode::Cuda && cuda_init_failed(&text) {
        bail!(
            "CUDA llama-server was built, but CUDA runtime validation failed:\n{}\n\nFix the NVIDIA driver/CUDA runtime mismatch, then rerun `werk backend install llama-cuda`.",
            tail_lines(&text, 30)
        );
    }
    Ok(())
}

fn require_llama_server(store: &ModelStore, mode: LlamaCppMode) -> Result<LlamaServerDiscovery> {
    let discovery = discover_llama_server(store, mode);
    if discovery.path.is_some() {
        Ok(discovery)
    } else {
        bail!("{}", missing_llama_server_message(&discovery))
    }
}

fn discover_llama_server(store: &ModelStore, mode: LlamaCppMode) -> LlamaServerDiscovery {
    let mut attempts = Vec::new();
    let specific_env = mode_env_name(mode);
    if let Some(path) = env::var_os(specific_env).map(PathBuf::from) {
        let exists = path.is_file();
        attempts.push(LlamaServerDiscoveryAttempt {
            label: specific_env.to_string(),
            path: Some(path.clone()),
            exists,
        });
        if exists {
            return LlamaServerDiscovery {
                mode,
                path: Some(path),
                source: format!("env {specific_env}"),
                attempts,
            };
        }
    } else {
        attempts.push(LlamaServerDiscoveryAttempt {
            label: specific_env.to_string(),
            path: None,
            exists: false,
        });
    }

    if let Some(path) = env::var_os("WERK_LLAMA_SERVER").map(PathBuf::from) {
        let exists = path.is_file();
        attempts.push(LlamaServerDiscoveryAttempt {
            label: "WERK_LLAMA_SERVER".to_string(),
            path: Some(path.clone()),
            exists,
        });
        if exists {
            return LlamaServerDiscovery {
                mode,
                path: Some(path),
                source: "env WERK_LLAMA_SERVER".to_string(),
                attempts,
            };
        }
    } else {
        attempts.push(LlamaServerDiscoveryAttempt {
            label: "WERK_LLAMA_SERVER".to_string(),
            path: None,
            exists: false,
        });
    }

    if let Some(path) = find_in_path(default_executable_name()) {
        attempts.push(LlamaServerDiscoveryAttempt {
            label: format!("PATH: {}", default_executable_name()),
            path: Some(path.clone()),
            exists: true,
        });
        return LlamaServerDiscovery {
            mode,
            path: Some(path),
            source: "PATH".to_string(),
            attempts,
        };
    }
    attempts.push(LlamaServerDiscoveryAttempt {
        label: format!("PATH: {}", default_executable_name()),
        path: None,
        exists: false,
    });

    let managed_root = managed_backend_dir(store, mode);
    if let Some(path) = find_managed_server(store, mode) {
        attempts.push(LlamaServerDiscoveryAttempt {
            label: "managed cache".to_string(),
            path: Some(path.clone()),
            exists: true,
        });
        return LlamaServerDiscovery {
            mode,
            path: Some(path),
            source: "managed cache".to_string(),
            attempts,
        };
    }
    attempts.push(LlamaServerDiscoveryAttempt {
        label: "managed cache".to_string(),
        path: Some(managed_root),
        exists: false,
    });

    LlamaServerDiscovery {
        mode,
        path: None,
        source: "missing".to_string(),
        attempts,
    }
}

fn missing_llama_server_message(discovery: &LlamaServerDiscovery) -> String {
    let mut message = format!(
        "No {} llama-server found.\n\nTried:",
        display_name(discovery.mode)
    );
    for attempt in &discovery.attempts {
        let status = if attempt.exists { "exists" } else { "missing" };
        let path = attempt
            .path
            .as_ref()
            .map(|path| format!(": {}", path.display()))
            .unwrap_or_default();
        message.push_str(&format!("\n- {}{} ({status})", attempt.label, path));
    }
    message.push_str("\n\nFix:");
    message.push_str(&format!(
        "\n- set {}=/path/to/llama-server",
        mode_env_name(discovery.mode)
    ));
    message.push_str(&format!(
        "\n- or run: werk backend install {}",
        install_target_name(discovery.mode)
    ));
    message.push_str("\n- or use: werk --backend cpu ...");
    message
}

fn find_managed_server(store: &ModelStore, mode: LlamaCppMode) -> Option<PathBuf> {
    let path_file = managed_path_file(store, mode);
    if let Ok(path) = fs::read_to_string(&path_file) {
        let path = PathBuf::from(path.trim());
        if path.is_file() {
            return Some(path);
        }
    }
    managed_server_candidates(store, mode)
        .into_iter()
        .find(|path| path.is_file())
}

fn managed_server_candidates(store: &ModelStore, mode: LlamaCppMode) -> Vec<PathBuf> {
    let root = managed_backend_dir(store, mode);
    let source = root.join("llama.cpp");
    let build = root.join("build");
    let exe = default_executable_name();
    vec![
        root.join(exe),
        root.join("bin").join(exe),
        build.join("bin").join(exe),
        build.join("bin").join("Release").join(exe),
        build.join("tools").join("server").join(exe),
        source.join("build").join("bin").join(exe),
        source.join("build").join("bin").join("Release").join(exe),
        source.join("build").join("tools").join("server").join(exe),
    ]
}

fn managed_path_file(store: &ModelStore, mode: LlamaCppMode) -> PathBuf {
    managed_backend_dir(store, mode).join("llama-server.path")
}

fn run_command(command: &mut Command, context: &str) -> Result<()> {
    let status = command.status().with_context(|| context.to_string())?;
    if !status.success() {
        bail!("{context}; command exited with {status}");
    }
    Ok(())
}

fn cuda_host_compiler() -> Option<PathBuf> {
    if let Some(path) = env::var_os("WERK_LLAMA_CUDA_HOST_COMPILER").map(PathBuf::from) {
        if path.is_file() {
            return Some(path);
        }
    }
    if cuda_major_version().is_some_and(|major| major <= 11) {
        let candidate = PathBuf::from("/usr/bin/g++-10");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn matching_c_compiler(cxx_compiler: &Path) -> Option<PathBuf> {
    let file_name = cxx_compiler.file_name()?.to_string_lossy();
    let c_name = if file_name.starts_with("g++") {
        file_name.replacen("g++", "gcc", 1)
    } else if file_name.starts_with("clang++") {
        file_name.replacen("clang++", "clang", 1)
    } else {
        return None;
    };
    let c_compiler = cxx_compiler.with_file_name(c_name);
    c_compiler.is_file().then_some(c_compiler)
}

fn cuda_major_version() -> Option<u32> {
    let output = Command::new("nvcc").arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let marker = "release ";
    let start = text.find(marker)? + marker.len();
    text[start..]
        .split(|ch: char| !ch.is_ascii_digit())
        .next()?
        .parse()
        .ok()
}

fn cuda_architecture() -> Option<String> {
    if let Ok(value) = env::var("WERK_LLAMA_CUDA_ARCH")
        && !value.trim().is_empty()
    {
        return Some(value.trim().to_string());
    }
    if let Ok(value) = env::var("CUDAARCHS")
        && !value.trim().is_empty()
    {
        return Some(value.trim().to_string());
    }
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line| {
            let arch = line.trim().replace('.', "");
            (!arch.is_empty() && arch.chars().all(|ch| ch.is_ascii_digit())).then_some(arch)
        })
}

fn cuda_host_compiler_check() -> BackendDoctorCheck {
    if let Some(path) = cuda_host_compiler() {
        return BackendDoctorCheck {
            name: "CUDA host compiler".to_string(),
            ok: true,
            detail: format!("{}", path.display()),
        };
    }
    let detail = if cuda_major_version().is_some_and(|major| major <= 11) {
        "CUDA 11.x detected; install g++-10 or set WERK_LLAMA_CUDA_HOST_COMPILER"
    } else {
        "using default compiler selected by CMake/NVCC"
    };
    BackendDoctorCheck {
        name: "CUDA host compiler".to_string(),
        ok: true,
        detail: detail.to_string(),
    }
}

fn cuda_architecture_check() -> BackendDoctorCheck {
    match cuda_architecture() {
        Some(arch) => BackendDoctorCheck {
            name: "CUDA architecture".to_string(),
            ok: true,
            detail: arch,
        },
        None => BackendDoctorCheck {
            name: "CUDA architecture".to_string(),
            ok: false,
            detail: "could not detect; set WERK_LLAMA_CUDA_ARCH, for example 86 for RTX 3090"
                .to_string(),
        },
    }
}

fn command_check(command: &str, args: &[&str], detail: &str) -> BackendDoctorCheck {
    match Command::new(command).args(args).output() {
        Ok(output) if output.status.success() => BackendDoctorCheck {
            name: command.to_string(),
            ok: true,
            detail: String::from_utf8_lossy(&output.stdout)
                .lines()
                .next()
                .unwrap_or(detail)
                .to_string(),
        },
        Ok(output) => BackendDoctorCheck {
            name: command.to_string(),
            ok: false,
            detail: format!("{detail}; command exited with {}", output.status),
        },
        Err(err) => BackendDoctorCheck {
            name: command.to_string(),
            ok: false,
            detail: format!("{detail}; {err}"),
        },
    }
}

fn spawn_log_tail_reader<R>(label: &'static str, reader: R, tail: Arc<Mutex<VecDeque<String>>>)
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let reader = BufReader::new(reader);
        for line in reader.lines().map_while(Result::ok) {
            if let Ok(mut tail) = tail.lock() {
                if tail.len() >= 80 {
                    tail.pop_front();
                }
                tail.push_back(format!("{label}: {line}"));
            }
        }
    });
}

fn cuda_init_failed(text: &str) -> bool {
    let text = text.to_ascii_lowercase();
    text.contains("failed to initialize cuda")
        || text.contains("cuda driver version is insufficient")
        || text.contains("cuda error")
}

fn tail_lines(text: &str, max_lines: usize) -> String {
    let mut lines = text.lines().collect::<Vec<_>>();
    if lines.len() > max_lines {
        lines = lines.split_off(lines.len() - max_lines);
    }
    lines.join("\n")
}

fn cache_write_check(store: &ModelStore) -> BackendDoctorCheck {
    let dir = store.home().join("backends");
    let result = fs::create_dir_all(&dir).and_then(|_| {
        let path = dir.join(".werk-write-test");
        fs::write(&path, b"ok")?;
        fs::remove_file(path)
    });
    match result {
        Ok(()) => BackendDoctorCheck {
            name: "managed backend cache".to_string(),
            ok: true,
            detail: format!("writable: {}", dir.display()),
        },
        Err(err) => BackendDoctorCheck {
            name: "managed backend cache".to_string(),
            ok: false,
            detail: format!("not writable: {} ({err})", dir.display()),
        },
    }
}

fn mode_env_name(mode: LlamaCppMode) -> &'static str {
    match mode {
        LlamaCppMode::Cuda => "WERK_LLAMA_SERVER_CUDA",
        LlamaCppMode::Vulkan => "WERK_LLAMA_SERVER_VULKAN",
        LlamaCppMode::Cpu => "WERK_LLAMA_SERVER_CPU",
    }
}

fn install_target_name(mode: LlamaCppMode) -> &'static str {
    match mode {
        LlamaCppMode::Cuda => "llama-cuda",
        LlamaCppMode::Vulkan => "llama-vulkan",
        LlamaCppMode::Cpu => "llama-cpu",
    }
}

fn find_in_path(name: &str) -> Option<PathBuf> {
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
        #[cfg(windows)]
        {
            let candidate = dir.join(format!("{name}.exe"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn default_executable_name() -> &'static str {
    if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    }
}

fn free_local_port() -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    Ok(listener.local_addr()?.port())
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
    if let Some(tx) = tx {
        tx.blocking_send(Ok(GenerateStreamEvent::TextChunk(chunk)))
            .map_err(|err| anyhow!("stream receiver closed: {err}"))?;
    }
    Ok(())
}

fn split_args(input: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    let mut escape = false;
    for ch in input.chars() {
        if escape {
            current.push(ch);
            escape = false;
            continue;
        }
        if ch == '\\' {
            escape = true;
            continue;
        }
        if let Some(active) = quote {
            if ch == active {
                quote = None;
            } else {
                current.push(ch);
            }
            continue;
        }
        if ch == '\'' || ch == '"' {
            quote = Some(ch);
        } else if ch.is_whitespace() {
            if !current.is_empty() {
                args.push(std::mem::take(&mut current));
            }
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        args.push(current);
    }
    args
}

fn shell_join(args: &[String]) -> String {
    args.iter()
        .map(|arg| {
            if arg
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_./:=,".contains(&b))
            {
                arg.clone()
            } else {
                format!("'{}'", arg.replace('\'', "'\\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn find_sse_boundary(bytes: &[u8]) -> Option<usize> {
    bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .or_else(|| bytes.windows(2).position(|window| window == b"\n\n"))
}

fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count().max(1)
}

fn gpu_layers(mode: LlamaCppMode) -> i32 {
    match mode {
        LlamaCppMode::Cpu => 0,
        LlamaCppMode::Cuda | LlamaCppMode::Vulkan => 999,
    }
}

fn display_name(mode: LlamaCppMode) -> &'static str {
    match mode {
        LlamaCppMode::Cuda => "CUDA",
        LlamaCppMode::Vulkan => "Vulkan",
        LlamaCppMode::Cpu => "CPU",
    }
}

fn label(mode: LlamaCppMode) -> &'static str {
    match mode {
        LlamaCppMode::Cuda => "llama-server-cuda",
        LlamaCppMode::Vulkan => "llama-server-vulkan",
        LlamaCppMode::Cpu => "llama-server-cpu",
    }
}

fn env_true(name: &str) -> bool {
    matches!(
        env::var(name).ok().as_deref(),
        Some("1" | "true" | "True" | "TRUE" | "yes" | "Yes" | "YES" | "on" | "On" | "ON")
    )
}

fn format_error_chain(err: &anyhow::Error) -> String {
    err.chain()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(": ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        env,
        ffi::OsString,
        sync::{Mutex as StdMutex, OnceLock},
        time::{SystemTime, UNIX_EPOCH},
    };

    static ENV_LOCK: OnceLock<StdMutex<()>> = OnceLock::new();

    struct EnvGuard {
        values: Vec<(&'static str, Option<OsString>)>,
    }

    impl EnvGuard {
        fn new(names: &[&'static str]) -> Self {
            Self {
                values: names
                    .iter()
                    .map(|name| (*name, env::var_os(name)))
                    .collect(),
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (name, value) in &self.values {
                unsafe {
                    if let Some(value) = value {
                        env::set_var(name, value);
                    } else {
                        env::remove_var(name);
                    }
                }
            }
        }
    }

    #[test]
    fn managed_cache_path_uses_requested_backend_name() {
        let root = temp_root("managed-path");
        let store = ModelStore::resolve(Some(root.clone())).unwrap();
        assert_eq!(
            managed_backend_dir(&store, LlamaCppMode::Cuda),
            root.join("backends").join("llama-cuda")
        );
        assert_eq!(
            managed_backend_dir(&store, LlamaCppMode::Vulkan),
            root.join("backends").join("llama-vulkan")
        );
        assert_eq!(
            managed_backend_dir(&store, LlamaCppMode::Cpu),
            root.join("backends").join("llama-cpu")
        );
    }

    #[test]
    fn discovery_precedence_specific_env_wins_over_generic_path_and_cache() {
        let _lock = ENV_LOCK.get_or_init(|| StdMutex::new(())).lock().unwrap();
        let _guard = EnvGuard::new(&["WERK_LLAMA_SERVER_CUDA", "WERK_LLAMA_SERVER", "PATH"]);
        let root = temp_root("specific-env");
        let specific = touch(root.join("specific").join(default_executable_name()));
        let generic = touch(root.join("generic").join(default_executable_name()));
        let path_dir = root.join("path");
        let _path_server = touch(path_dir.join(default_executable_name()));
        let store = ModelStore::resolve(Some(root.join("werk-home"))).unwrap();
        let cache_server = touch(
            managed_backend_dir(&store, LlamaCppMode::Cuda)
                .join("build")
                .join("bin")
                .join(default_executable_name()),
        );
        fs::write(
            managed_path_file(&store, LlamaCppMode::Cuda),
            cache_server.display().to_string(),
        )
        .unwrap();

        unsafe {
            env::set_var("WERK_LLAMA_SERVER_CUDA", &specific);
            env::set_var("WERK_LLAMA_SERVER", &generic);
            env::set_var("PATH", &path_dir);
        }

        let discovery = discover_llama_server(&store, LlamaCppMode::Cuda);
        assert_eq!(discovery.path.as_deref(), Some(specific.as_path()));
        assert_eq!(discovery.source, "env WERK_LLAMA_SERVER_CUDA");
    }

    #[test]
    fn discovery_precedence_generic_env_wins_over_path_and_cache() {
        let _lock = ENV_LOCK.get_or_init(|| StdMutex::new(())).lock().unwrap();
        let _guard = EnvGuard::new(&["WERK_LLAMA_SERVER_CUDA", "WERK_LLAMA_SERVER", "PATH"]);
        let root = temp_root("generic-env");
        let generic = touch(root.join("generic").join(default_executable_name()));
        let path_dir = root.join("path");
        let _path_server = touch(path_dir.join(default_executable_name()));
        let store = ModelStore::resolve(Some(root.join("werk-home"))).unwrap();
        let cache_server = touch(
            managed_backend_dir(&store, LlamaCppMode::Cuda)
                .join("build")
                .join("bin")
                .join(default_executable_name()),
        );
        fs::write(
            managed_path_file(&store, LlamaCppMode::Cuda),
            cache_server.display().to_string(),
        )
        .unwrap();

        unsafe {
            env::remove_var("WERK_LLAMA_SERVER_CUDA");
            env::set_var("WERK_LLAMA_SERVER", &generic);
            env::set_var("PATH", &path_dir);
        }

        let discovery = discover_llama_server(&store, LlamaCppMode::Cuda);
        assert_eq!(discovery.path.as_deref(), Some(generic.as_path()));
        assert_eq!(discovery.source, "env WERK_LLAMA_SERVER");
    }

    #[test]
    fn discovery_uses_managed_cache_after_env_and_path() {
        let _lock = ENV_LOCK.get_or_init(|| StdMutex::new(())).lock().unwrap();
        let _guard = EnvGuard::new(&["WERK_LLAMA_SERVER_CUDA", "WERK_LLAMA_SERVER", "PATH"]);
        let root = temp_root("managed-cache");
        let store = ModelStore::resolve(Some(root.join("werk-home"))).unwrap();
        let cache_server = touch(
            managed_backend_dir(&store, LlamaCppMode::Cuda)
                .join("build")
                .join("bin")
                .join(default_executable_name()),
        );
        unsafe {
            env::remove_var("WERK_LLAMA_SERVER_CUDA");
            env::remove_var("WERK_LLAMA_SERVER");
            env::set_var("PATH", root.join("empty-path"));
        }

        let discovery = discover_llama_server(&store, LlamaCppMode::Cuda);
        assert_eq!(discovery.path.as_deref(), Some(cache_server.as_path()));
        assert_eq!(discovery.source, "managed cache");
    }

    #[test]
    fn missing_backend_error_contains_all_attempted_locations() {
        let _lock = ENV_LOCK.get_or_init(|| StdMutex::new(())).lock().unwrap();
        let _guard = EnvGuard::new(&["WERK_LLAMA_SERVER_CUDA", "WERK_LLAMA_SERVER", "PATH"]);
        let root = temp_root("missing");
        let store = ModelStore::resolve(Some(root.join("werk-home"))).unwrap();
        unsafe {
            env::remove_var("WERK_LLAMA_SERVER_CUDA");
            env::remove_var("WERK_LLAMA_SERVER");
            env::set_var("PATH", root.join("empty-path"));
        }

        let message = LlamaServerBackend::missing_message(&store, LlamaCppMode::Cuda);
        assert!(message.contains("No CUDA llama-server found."));
        assert!(message.contains("WERK_LLAMA_SERVER_CUDA"));
        assert!(message.contains("WERK_LLAMA_SERVER"));
        assert!(message.contains("PATH: llama-server"));
        assert!(message.contains("managed cache"));
        assert!(message.contains("werk backend install llama-cuda"));
    }

    #[test]
    fn completion_event_parses_nested_timings_object() {
        let mut completion = ServerCompletion::default();
        update_completion_from_event(
            &mut completion,
            &json!({
                "timings": {
                    "prompt_n": 46,
                    "prompt_ms": 214.738,
                    "predicted_n": 745,
                    "predicted_ms": 16418.107
                }
            }),
        );

        assert_eq!(completion.prompt_tokens, 46);
        assert_eq!(completion.completion_tokens, 745);
        assert!((completion.prompt_seconds - 0.214738).abs() < 0.000001);
        assert!((completion.decode_seconds - 16.418107).abs() < 0.000001);
    }

    #[test]
    fn completion_event_parses_top_level_legacy_fields() {
        let mut completion = ServerCompletion::default();
        update_completion_from_event(
            &mut completion,
            &json!({
                "prompt_n": 12,
                "prompt_ms": 50.0,
                "predicted_n": 34,
                "predicted_ms": 700.0
            }),
        );

        assert_eq!(completion.prompt_tokens, 12);
        assert_eq!(completion.completion_tokens, 34);
        assert!((completion.prompt_seconds - 0.05).abs() < 0.000001);
        assert!((completion.decode_seconds - 0.7).abs() < 0.000001);
    }

    #[test]
    fn completion_event_does_not_overwrite_good_values_with_zero_or_missing_fields() {
        let mut completion = ServerCompletion {
            prompt_tokens: 46,
            completion_tokens: 745,
            prompt_seconds: 0.214738,
            decode_seconds: 16.418107,
            ..Default::default()
        };
        update_completion_from_event(
            &mut completion,
            &json!({
                "timings": {
                    "prompt_n": 0,
                    "prompt_ms": 0.0,
                    "predicted_n": 0,
                    "predicted_ms": 0.0
                }
            }),
        );
        update_completion_from_event(&mut completion, &json!({ "stop": true }));

        assert_eq!(completion.prompt_tokens, 46);
        assert_eq!(completion.completion_tokens, 745);
        assert!((completion.prompt_seconds - 0.214738).abs() < 0.000001);
        assert!((completion.decode_seconds - 16.418107).abs() < 0.000001);
        assert_eq!(completion.finish_reason, "stop");
    }

    #[test]
    fn completion_stats_fallback_estimates_non_empty_prompt_tokens() {
        let mut completion = ServerCompletion {
            text: "hello from llama".to_string(),
            first_token_seconds: 0.123,
            ..Default::default()
        };
        let request = GenerateRequest {
            prompt: "Write a sentence about Rust.".to_string(),
            messages: Vec::new(),
            image_urls: Vec::new(),
            max_tokens: 32,
            temperature: None,
            top_p: None,
            stop: Vec::new(),
            seed: None,
            stream_granularity: crate::backend::StreamGranularity::Chunk,
            verbose: false,
            debug: false,
        };

        finalize_completion_stats(&mut completion, &request, 0.5);

        assert!(completion.prompt_tokens > 0);
        assert_eq!(completion.prompt_seconds, 0.123);
        assert!((completion.decode_seconds - 0.377).abs() < 0.000001);
        assert!(completion.completion_tokens > 0);
    }

    #[test]
    fn server_args_pass_flash_attention_value_before_kv_offload() {
        let args = llama_server_args(
            LlamaCppMode::Cuda,
            &PathBuf::from("/tmp/model.gguf"),
            12345,
            &LlamaRuntimeOptions::default(),
            &SupportedArgs {
                flash_attn: true,
                kv_offload: true,
                perf: true,
                log_disable: true,
            },
        );
        let flash_attn = args.iter().position(|arg| arg == "-fa").unwrap();
        assert_eq!(args.get(flash_attn + 1).map(String::as_str), Some("auto"));
        assert_ne!(args.get(flash_attn + 1).map(String::as_str), Some("-kvo"));
        assert!(args.iter().any(|arg| arg == "-kvo"));

        let runtime_options = LlamaRuntimeOptions {
            flash_attn: Some(false),
            ..LlamaRuntimeOptions::default()
        };
        let args = llama_server_args(
            LlamaCppMode::Cuda,
            &PathBuf::from("/tmp/model.gguf"),
            12345,
            &runtime_options,
            &SupportedArgs {
                flash_attn: true,
                kv_offload: true,
                perf: false,
                log_disable: false,
            },
        );
        let flash_attn = args.iter().position(|arg| arg == "-fa").unwrap();
        assert_eq!(args.get(flash_attn + 1).map(String::as_str), Some("off"));
    }

    fn temp_root(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = env::temp_dir().join(format!("werk1112-{name}-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn touch(path: PathBuf) -> PathBuf {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"#!/bin/sh\nexit 0\n").unwrap();
        path
    }
}
