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
    BackendDoctorCheck, ChatGenerationSession, GenerateRequest, GenerateResponse, GenerateStream,
    GenerateStreamEvent, GenerationBackend, GenerationTimings,
};
use crate::model_store::{ModelFormat, ModelManifest, ModelStore};

const HEALTH_TIMEOUT: Duration = Duration::from_secs(300);
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(250);
const WSL_VLLM_MESSAGE: &str = "vLLM is a Linux-native runtime. Your environment appears to be WSL, where vLLM can fail because required GPU memory features such as UVA/CUDA IPC are unavailable. Werk will fall back to Candle CUDA. For best vLLM support use native Linux or a remote vLLM server.";

#[derive(Clone)]
pub struct VllmBackend {
    store: ModelStore,
    servers: Arc<Mutex<HashMap<String, Arc<VllmProcess>>>>,
}

struct VllmProcess {
    child: Option<Mutex<Child>>,
    command_label: String,
    discovery_source: String,
    args: Vec<String>,
    model_dir: PathBuf,
    model_name: String,
    url: String,
    pid: Option<u32>,
    log_tail: Arc<Mutex<VecDeque<String>>>,
}

struct VllmChatSession {
    server: Arc<VllmProcess>,
}

#[derive(Debug, Clone)]
pub struct VllmDiscovery {
    pub command: Option<VllmCommand>,
    pub source: String,
    pub attempts: Vec<VllmDiscoveryAttempt>,
}

#[derive(Debug, Clone)]
pub struct VllmDiscoveryAttempt {
    pub label: String,
    pub path: Option<PathBuf>,
    pub exists: bool,
    pub usable: bool,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct VllmHealthStatus {
    pub installed_label: &'static str,
    pub health_label: &'static str,
    pub healthy: bool,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub enum VllmCommand {
    Python(PathBuf),
    Executable(PathBuf),
    Remote { host: String, port: u16 },
}

#[derive(Default)]
struct VllmCompletion {
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    prompt_seconds: f64,
    decode_seconds: f64,
    first_token_seconds: f64,
    finish_reason: String,
}

impl VllmBackend {
    pub fn new(store: ModelStore) -> Self {
        Self {
            store,
            servers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn probe(store: &ModelStore) -> Result<String> {
        let discovery = discover_vllm(store);
        if let Some(reason) = local_vllm_platform_rejection_for_discovery(&discovery) {
            bail!("{reason}");
        }
        let Some(command) = discovery.command.as_ref() else {
            bail!("{}", missing_vllm_message(&discovery));
        };
        ensure_vllm_platform_eligible(command)?;
        match command {
            VllmCommand::Remote { host, port } => {
                Ok(format!("vLLM OpenAI server at http://{host}:{port}"))
            }
            command => Ok(format!(
                "vLLM {} ({})",
                command.display(),
                vllm_version(command).unwrap_or_else(|| "version unknown".to_string())
            )),
        }
    }

    pub fn probe_rocm(store: &ModelStore) -> Result<String> {
        let discovery = require_vllm(store)?;
        let command = discovery
            .command
            .as_ref()
            .context("vLLM discovery had no command")?;
        ensure_vllm_platform_eligible(command)?;
        let detail = vllm_rocm_capability(command)?;
        Ok(format!("vLLM ROCm ({detail})"))
    }

    pub fn discover(store: &ModelStore) -> VllmDiscovery {
        discover_vllm(store)
    }

    pub fn health(store: &ModelStore) -> VllmHealthStatus {
        let discovery = discover_vllm(store);
        vllm_health(&discovery)
    }

    pub fn missing_message(store: &ModelStore) -> String {
        let discovery = discover_vllm(store);
        if let Some(reason) = local_vllm_platform_rejection_for_discovery(&discovery) {
            return reason.to_string();
        }
        missing_vllm_message(&discovery)
    }

    pub fn unavailable_reason(store: &ModelStore) -> String {
        let discovery = discover_vllm(store);
        if let Some(reason) = local_vllm_platform_rejection_for_discovery(&discovery) {
            return reason.to_string();
        }
        if let Some(command) = discovery.command.as_ref()
            && let Err(err) = ensure_vllm_platform_eligible(command)
        {
            return compact_error(&err.to_string());
        }
        concise_vllm_unavailable_reason(&discovery)
    }

    pub fn rocm_unavailable_reason(store: &ModelStore) -> String {
        let discovery = discover_vllm(store);
        if let Some(reason) = local_vllm_platform_rejection_for_discovery(&discovery) {
            return reason.to_string();
        }
        let Some(command) = discovery.command.as_ref() else {
            return concise_vllm_unavailable_reason(&discovery);
        };
        if let Err(err) = ensure_vllm_platform_eligible(command) {
            return compact_error(&err.to_string());
        }
        vllm_rocm_capability(command)
            .err()
            .map(|err| compact_error(&err.to_string()))
            .unwrap_or_else(|| "vLLM ROCm runtime is unavailable".to_string())
    }

    fn cached_server(&self, manifest: &ModelManifest) -> Result<(Arc<VllmProcess>, bool, f64)> {
        if manifest.format != ModelFormat::SafeTensors {
            bail!("vLLM backend supports HF safetensors model directories only");
        }

        let model_dir = resolve_vllm_model_dir(&self.store, manifest)?;

        let key = format!(
            "{}:{}:{}",
            manifest.id,
            model_dir.display(),
            env::var("WERK_VLLM_ARGS").unwrap_or_default()
        );
        if let Some(server) = self
            .servers
            .lock()
            .map_err(|_| anyhow!("vLLM server cache mutex poisoned"))?
            .get(&key)
            .cloned()
            && server.is_running()
        {
            return Ok((server, true, 0.0));
        }

        let started = Instant::now();
        let server = Arc::new(VllmProcess::start(&self.store, manifest, &model_dir)?);
        let load_seconds = started.elapsed().as_secs_f64();
        self.servers
            .lock()
            .map_err(|_| anyhow!("vLLM server cache mutex poisoned"))?
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
            bail!("vLLM text backend received image inputs; use a VLM-capable model/runtime");
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
            backend_diagnostics: Vec::new(),
        })
    }
}

impl GenerationBackend for VllmBackend {
    fn prepare(&self, manifest: &ModelManifest) -> Result<()> {
        self.cached_server(manifest).map(|_| ())
    }

    fn start_chat_session(
        &self,
        manifest: &ModelManifest,
        _seed: Option<u64>,
    ) -> Result<Option<Box<dyn ChatGenerationSession>>> {
        if manifest.format != ModelFormat::SafeTensors {
            return Ok(None);
        }
        let (server, _, _) = self.cached_server(manifest)?;
        Ok(Some(Box::new(VllmChatSession { server })))
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

impl ChatGenerationSession for VllmChatSession {
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
            backend_diagnostics: Vec::new(),
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
                    backend_diagnostics: Vec::new(),
                });
            send_stream_result(tx, result);
        });
        Box::pin(ReceiverStream::new(rx))
    }
}

impl VllmProcess {
    fn start(store: &ModelStore, manifest: &ModelManifest, model_dir: &Path) -> Result<Self> {
        let discovery = require_vllm(store)?;
        let command = discovery
            .command
            .clone()
            .context("vLLM discovery had no command")?;
        let log_tail = Arc::new(Mutex::new(VecDeque::new()));
        eprintln!("Using vLLM CUDA backend");

        if let VllmCommand::Remote { host, port } = command {
            let process = Self {
                child: None,
                command_label: "remote vLLM OpenAI server".to_string(),
                discovery_source: discovery.source,
                args: Vec::new(),
                model_dir: model_dir.to_path_buf(),
                model_name: manifest.id.clone(),
                url: format!("http://{host}:{port}"),
                pid: None,
                log_tail,
            };
            process.wait_until_ready()?;
            return Ok(process);
        }

        let port = free_local_port()?;
        let url = format!("http://127.0.0.1:{port}");
        let args = vllm_server_args(&command, model_dir, &manifest.id, port);
        let mut child_command = Command::new(command.executable());
        child_command.args(&args);
        if env_true("WERK_VLLM_LOG") {
            child_command
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit());
        } else {
            child_command.stdout(Stdio::piped()).stderr(Stdio::piped());
        }
        let mut child = child_command.spawn().with_context(|| {
            format!(
                "failed to start vLLM server using {}",
                command.executable().display()
            )
        })?;
        if !env_true("WERK_VLLM_LOG") {
            if let Some(stdout) = child.stdout.take() {
                spawn_log_tail_reader("stdout", stdout, log_tail.clone());
            }
            if let Some(stderr) = child.stderr.take() {
                spawn_log_tail_reader("stderr", stderr, log_tail.clone());
            }
        }
        let pid = child.id();
        let process = Self {
            child: Some(Mutex::new(child)),
            command_label: command.display(),
            discovery_source: discovery.source,
            args,
            model_dir: model_dir.to_path_buf(),
            model_name: manifest.id.clone(),
            url,
            pid: Some(pid),
            log_tail,
        };
        process.wait_until_ready()?;
        Ok(process)
    }

    fn complete(
        &self,
        request: &GenerateRequest,
        tx: Option<mpsc::Sender<Result<GenerateStreamEvent, String>>>,
    ) -> Result<VllmCompletion> {
        let started = Instant::now();
        let body = chat_completion_body(&self.model_name, request);
        let mut stream = post_json(&self.url, "/v1/chat/completions", &body)?;
        let mut completion = VllmCompletion {
            finish_reason: "length".to_string(),
            ..Default::default()
        };
        let mut sse = SseAccumulator::default();

        stream_body(&mut stream, |bytes| {
            sse.push(bytes, |event| {
                if event == "[DONE]" {
                    return Ok(());
                }
                let value: Value = serde_json::from_str(event)
                    .with_context(|| format!("invalid vLLM SSE event: {event}"))?;
                update_completion_from_event(&mut completion, &value);
                if let Some(chunk) = delta_content(&value)
                    && !chunk.is_empty()
                {
                    if completion.first_token_seconds <= 0.0 {
                        completion.first_token_seconds = started.elapsed().as_secs_f64();
                    }
                    completion.text.push_str(&chunk);
                    send_text_chunk(&tx, chunk)?;
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
                let reason = format!(
                    "vLLM server exited before becoming healthy ({status}){}",
                    self.formatted_log_tail()
                );
                if let Some(message) = wsl_vllm_health_failure_message(&reason) {
                    bail!("{message}");
                }
                bail!("{reason}");
            }
            if get(&self.url, "/health")
                .map(|response| response.status == 200)
                .unwrap_or(false)
                || get(&self.url, "/v1/models")
                    .map(|response| response.status == 200)
                    .unwrap_or(false)
            {
                return Ok(());
            }
            if started.elapsed() > HEALTH_TIMEOUT {
                let reason = format!(
                    "timed out waiting for vLLM server at {}{}",
                    self.url,
                    self.formatted_log_tail()
                );
                if let Some(message) = wsl_vllm_health_failure_message(&reason) {
                    bail!("{message}");
                }
                bail!("{reason}");
            }
            thread::sleep(HEALTH_POLL_INTERVAL);
        }
    }

    fn is_running(&self) -> bool {
        if self.child.is_none() {
            return get(&self.url, "/v1/models")
                .map(|response| response.status == 200)
                .unwrap_or(false);
        }
        matches!(self.try_wait_status(), Ok(None))
    }

    fn try_wait_status(&self) -> Result<Option<ExitStatus>> {
        let Some(child) = &self.child else {
            return Ok(None);
        };
        let mut child = child
            .lock()
            .map_err(|_| anyhow!("vLLM child mutex poisoned"))?;
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
            "\n\nvLLM output tail:\n{}",
            tail.iter().cloned().collect::<Vec<_>>().join("\n")
        )
    }

    fn print_debug(&self, request: &GenerateRequest, reused: bool) {
        if !request.debug {
            return;
        }
        eprintln!("selected backend: vllm-cuda");
        eprintln!("actual engine: vLLM OpenAI-compatible server");
        eprintln!("vLLM executable: {}", self.command_label);
        eprintln!("discovery source: {}", self.discovery_source);
        if self.args.is_empty() {
            eprintln!("full vLLM args: <remote server>");
        } else {
            eprintln!("full vLLM args: {}", shell_join(&self.args));
        }
        eprintln!("model path: {}", self.model_dir.display());
        eprintln!(
            "server PID: {}",
            self.pid
                .map(|pid| pid.to_string())
                .unwrap_or_else(|| "external".to_string())
        );
        eprintln!("server URL: {}", self.url);
        eprintln!("reused existing server: {reused}");
    }
}

impl Drop for VllmProcess {
    fn drop(&mut self) {
        if let Some(child) = &self.child
            && let Ok(mut child) = child.lock()
        {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl VllmCommand {
    fn executable(&self) -> PathBuf {
        match self {
            Self::Python(path) | Self::Executable(path) => path.clone(),
            Self::Remote { .. } => PathBuf::new(),
        }
    }

    fn display(&self) -> String {
        match self {
            Self::Python(path) => {
                format!("{} -m vllm.entrypoints.openai.api_server", path.display())
            }
            Self::Executable(path) => path.display().to_string(),
            Self::Remote { host, port } => format!("http://{host}:{port}"),
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

fn vllm_server_args(
    command: &VllmCommand,
    model_dir: &Path,
    model_name: &str,
    port: u16,
) -> Vec<String> {
    let mut args = match command {
        VllmCommand::Python(_) => vec![
            "-m".to_string(),
            "vllm.entrypoints.openai.api_server".to_string(),
            "--model".to_string(),
            model_dir.display().to_string(),
        ],
        VllmCommand::Executable(_) => vec!["serve".to_string(), model_dir.display().to_string()],
        VllmCommand::Remote { .. } => Vec::new(),
    };
    args.extend([
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        port.to_string(),
        "--served-model-name".to_string(),
        model_name.to_string(),
    ]);
    if let Ok(extra) = env::var("WERK_VLLM_ARGS") {
        args.extend(split_args(&extra));
    }
    args
}

fn resolve_vllm_model_dir(store: &ModelStore, manifest: &ModelManifest) -> Result<PathBuf> {
    let root = store.model_dir(&manifest.id);
    if !root.is_dir() {
        bail!(
            "model directory for '{}' does not exist: {}",
            manifest.id,
            root.display()
        );
    }

    if let Some(config_path) = manifest.config_path.as_deref() {
        let config = store.absolute_model_file(manifest, config_path);
        let dir = config.parent().with_context(|| {
            format!(
                "manifest config_path '{}' has no parent directory",
                config_path
            )
        })?;
        if dir.join("config.json").is_file() {
            return Ok(dir.to_path_buf());
        }
    }

    let files_dir = root.join("files");
    if files_dir.join("config.json").is_file() {
        return Ok(files_dir);
    }

    if root.join("config.json").is_file() {
        return Ok(root);
    }

    bail!(
        "vLLM requires a Hugging Face model directory containing config.json for '{}'; tried manifest config_path {}, files directory {}, and model root {}",
        manifest.id,
        manifest
            .config_path
            .as_deref()
            .map(|path| store
                .absolute_model_file(manifest, path)
                .display()
                .to_string())
            .unwrap_or_else(|| "<none>".to_string()),
        files_dir.display(),
        root.display()
    )
}

fn chat_completion_body(model_name: &str, request: &GenerateRequest) -> Value {
    let messages = if request.messages.is_empty() {
        json!([{
            "role": "user",
            "content": request.prompt,
        }])
    } else {
        json!(request.messages)
    };
    let mut body = json!({
        "model": model_name,
        "messages": messages,
        "max_tokens": request.max_tokens,
        "stream": true,
        "stream_options": {"include_usage": true},
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

fn update_completion_from_event(completion: &mut VllmCompletion, value: &Value) {
    if let Some(choice) = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        && let Some(reason) = choice.get("finish_reason").and_then(Value::as_str)
        && !reason.is_empty()
    {
        completion.finish_reason = reason.to_string();
    }
    if let Some(usage) = value.get("usage") {
        if let Some(tokens) = usage.get("prompt_tokens").and_then(Value::as_u64) {
            completion.prompt_tokens = tokens as usize;
        }
        if let Some(tokens) = usage.get("completion_tokens").and_then(Value::as_u64) {
            completion.completion_tokens = tokens as usize;
        }
    }
}

fn delta_content(value: &Value) -> Option<String> {
    value
        .get("choices")?
        .as_array()?
        .first()?
        .get("delta")?
        .get("content")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn finalize_completion_stats(
    completion: &mut VllmCompletion,
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

pub fn install_managed_vllm(store: &ModelStore) -> Result<PathBuf> {
    let platform = current_vllm_platform();
    if let Some(reason) = managed_vllm_install_rejection(platform) {
        bail!("{reason}");
    }
    if platform == VllmPlatform::Wsl {
        eprintln!("{WSL_VLLM_MESSAGE}");
    }

    let root = managed_vllm_dir(store);
    let venv = root.join("venv");
    fs::create_dir_all(&root)
        .with_context(|| format!("failed to create vLLM backend cache {}", root.display()))?;

    let python = find_bootstrap_python().ok_or_else(|| {
        anyhow!("no Python interpreter found; install python3 or set WERK_VLLM_PYTHON")
    })?;
    if !managed_vllm_python(store).is_file() {
        eprintln!("Creating vLLM virtualenv at {}", venv.display());
        run_command(
            Command::new(&python).arg("-m").arg("venv").arg(&venv),
            "failed to create vLLM virtualenv",
        )?;
    }
    let venv_python = managed_vllm_python(store);
    eprintln!("Installing vLLM into {}", venv.display());
    run_command(
        Command::new(&venv_python)
            .arg("-m")
            .arg("pip")
            .arg("install")
            .arg("--upgrade")
            .arg("pip"),
        "failed to upgrade pip in vLLM virtualenv",
    )?;
    run_command(
        Command::new(&venv_python)
            .arg("-m")
            .arg("pip")
            .arg("install")
            .arg("vllm"),
        "failed to install vLLM; check network access, Python version, CUDA/PyTorch wheel availability",
    )?;
    validate_vllm_python(&venv_python)?;
    Ok(venv_python)
}

pub fn vllm_doctor_checks(store: &ModelStore) -> Vec<BackendDoctorCheck> {
    let discovery = discover_vllm(store);
    let health = vllm_health(&discovery);
    let mut checks = Vec::new();
    checks.push(BackendDoctorCheck {
        name: "vLLM discovery".to_string(),
        ok: discovery.command.is_some(),
        detail: discovery.source.clone(),
    });
    checks.push(BackendDoctorCheck {
        name: "vLLM Python path".to_string(),
        ok: matches!(discovery.command, Some(VllmCommand::Python(_))),
        detail: match &discovery.command {
            Some(VllmCommand::Python(path)) => path.display().to_string(),
            Some(VllmCommand::Executable(path)) => format!("using executable {}", path.display()),
            Some(VllmCommand::Remote { host, port }) => {
                format!("using remote http://{host}:{port}")
            }
            None => format!("managed path {}", managed_vllm_python(store).display()),
        },
    });
    checks.push(BackendDoctorCheck {
        name: "vLLM installed".to_string(),
        ok: discovery.command.is_some(),
        detail: discovery
            .command
            .as_ref()
            .and_then(vllm_version)
            .unwrap_or_else(|| "not installed".to_string()),
    });
    checks.push(BackendDoctorCheck {
        name: "vLLM health".to_string(),
        ok: health.healthy,
        detail: format!("{}: {}", health.health_label, health.detail),
    });
    checks.push(command_check(
        "nvidia-smi",
        &[],
        "required to verify CUDA visibility for vLLM",
    ));
    checks.push(BackendDoctorCheck {
        name: "vLLM version".to_string(),
        ok: discovery.command.is_some(),
        detail: discovery
            .command
            .as_ref()
            .and_then(vllm_version)
            .unwrap_or_else(|| "unknown".to_string()),
    });
    checks
}

fn vllm_health(discovery: &VllmDiscovery) -> VllmHealthStatus {
    vllm_health_for_platform(discovery, current_vllm_platform())
}

fn vllm_health_for_platform(discovery: &VllmDiscovery, platform: VllmPlatform) -> VllmHealthStatus {
    match discovery.command.as_ref() {
        Some(VllmCommand::Remote { host, port }) => VllmHealthStatus {
            installed_label: "remote",
            health_label: "healthy",
            healthy: true,
            detail: format!(
                "remote OpenAI-compatible vLLM endpoint reachable at http://{host}:{port}"
            ),
        },
        Some(command) => match local_vllm_platform_rejection(platform) {
            Some(reason) => VllmHealthStatus {
                installed_label: "yes",
                health_label: if platform == VllmPlatform::Wsl {
                    "best-effort on WSL"
                } else {
                    "unsupported"
                },
                healthy: false,
                detail: reason.to_string(),
            },
            None => VllmHealthStatus {
                installed_label: "yes",
                health_label: "eligible",
                healthy: true,
                detail: vllm_version(command).unwrap_or_else(|| "version unknown".to_string()),
            },
        },
        None => {
            match local_vllm_platform_rejection_for_discovery_with_platform(discovery, platform) {
                Some(reason) => VllmHealthStatus {
                    installed_label: "no",
                    health_label: if platform == VllmPlatform::Wsl {
                        "best-effort on WSL"
                    } else {
                        "unsupported"
                    },
                    healthy: false,
                    detail: reason.to_string(),
                },
                None => VllmHealthStatus {
                    installed_label: "no",
                    health_label: "missing",
                    healthy: false,
                    detail: concise_vllm_unavailable_reason(discovery),
                },
            }
        }
    }
}

fn ensure_vllm_platform_eligible(command: &VllmCommand) -> Result<()> {
    if matches!(command, VllmCommand::Remote { .. }) {
        return Ok(());
    }
    if let Some(reason) = local_vllm_platform_rejection(current_vllm_platform()) {
        bail!("{reason}");
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VllmPlatform {
    NativeLinux,
    Wsl,
    NativeWindows,
    Macos,
    Unsupported,
}

fn current_vllm_platform() -> VllmPlatform {
    if cfg!(target_os = "linux") {
        if is_wsl_environment() {
            VllmPlatform::Wsl
        } else {
            VllmPlatform::NativeLinux
        }
    } else if cfg!(target_os = "windows") {
        VllmPlatform::NativeWindows
    } else if cfg!(target_os = "macos") {
        VllmPlatform::Macos
    } else {
        VllmPlatform::Unsupported
    }
}

fn local_vllm_platform_rejection(platform: VllmPlatform) -> Option<&'static str> {
    match platform {
        VllmPlatform::NativeLinux => None,
        VllmPlatform::Wsl => Some(WSL_VLLM_MESSAGE),
        VllmPlatform::NativeWindows => Some(
            "vLLM is a Linux-native runtime. Native Windows local vLLM is not eligible. Use native Linux or a remote vLLM server.",
        ),
        VllmPlatform::Macos => Some(
            "vLLM is a Linux-native runtime. Local managed vLLM is not eligible on macOS. Use native Linux or a remote vLLM server.",
        ),
        VllmPlatform::Unsupported => Some(
            "vLLM is a Linux-native runtime. Local vLLM is not eligible on this platform. Use native Linux or a remote vLLM server.",
        ),
    }
}

fn managed_vllm_install_rejection(platform: VllmPlatform) -> Option<&'static str> {
    match platform {
        VllmPlatform::NativeLinux | VllmPlatform::Wsl => None,
        VllmPlatform::NativeWindows | VllmPlatform::Macos | VllmPlatform::Unsupported => {
            local_vllm_platform_rejection(platform)
        }
    }
}

fn local_vllm_platform_rejection_for_discovery(discovery: &VllmDiscovery) -> Option<&'static str> {
    local_vllm_platform_rejection_for_discovery_with_platform(discovery, current_vllm_platform())
}

fn local_vllm_platform_rejection_for_discovery_with_platform(
    discovery: &VllmDiscovery,
    platform: VllmPlatform,
) -> Option<&'static str> {
    if matches!(discovery.command, Some(VllmCommand::Remote { .. }))
        || discovery_has_remote_attempt(discovery)
    {
        return None;
    }
    local_vllm_platform_rejection(platform)
}

fn discovery_has_remote_attempt(discovery: &VllmDiscovery) -> bool {
    discovery
        .attempts
        .iter()
        .any(|attempt| attempt.label == "WERK_VLLM_HOST/WERK_VLLM_PORT")
}

fn is_wsl_environment() -> bool {
    env::var_os("WSL_DISTRO_NAME").is_some()
        || env::var_os("WSL_INTEROP").is_some()
        || fs::read_to_string("/proc/sys/kernel/osrelease")
            .map(|text| linux_release_looks_like_wsl(&text))
            .unwrap_or(false)
        || fs::read_to_string("/proc/version")
            .map(|text| linux_release_looks_like_wsl(&text))
            .unwrap_or(false)
}

fn linux_release_looks_like_wsl(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("microsoft") || lower.contains("wsl")
}

fn wsl_vllm_health_failure_message(reason: &str) -> Option<String> {
    if current_vllm_platform() != VllmPlatform::Wsl || !is_wsl_sensitive_vllm_failure(reason) {
        return None;
    }
    Some(format!(
        "{WSL_VLLM_MESSAGE}\n\nvLLM health probe failed: {}",
        compact_error(reason)
    ))
}

fn is_wsl_sensitive_vllm_failure(reason: &str) -> bool {
    let lower = reason.to_ascii_lowercase();
    lower.contains("uva")
        || lower.contains("pin_memory")
        || lower.contains("pin memory")
        || lower.contains("engine-core")
        || lower.contains("engine_core")
        || lower.contains("engine core")
        || lower.contains("cuda ipc")
        || lower.contains("cuda_ipc")
        || lower.contains("cudaipc")
        || (lower.contains("multiprocessing")
            && (lower.contains("start") || lower.contains("spawn") || lower.contains("failed")))
}

pub fn managed_vllm_dir(store: &ModelStore) -> PathBuf {
    store.home().join("backends").join("vllm")
}

fn managed_vllm_python(store: &ModelStore) -> PathBuf {
    if cfg!(windows) {
        managed_vllm_dir(store)
            .join("venv")
            .join("Scripts")
            .join("python.exe")
    } else {
        managed_vllm_dir(store)
            .join("venv")
            .join("bin")
            .join("python")
    }
}

fn require_vllm(store: &ModelStore) -> Result<VllmDiscovery> {
    let discovery = discover_vllm(store);
    if discovery.command.is_some() {
        Ok(discovery)
    } else {
        bail!("{}", missing_vllm_message(&discovery))
    }
}

fn discover_vllm(store: &ModelStore) -> VllmDiscovery {
    let mut attempts = Vec::new();

    if let (Ok(host), Ok(port)) = (env::var("WERK_VLLM_HOST"), env::var("WERK_VLLM_PORT")) {
        match port.parse::<u16>() {
            Ok(port) => {
                let usable = remote_supports_vllm(&host, port);
                attempts.push(VllmDiscoveryAttempt {
                    label: "WERK_VLLM_HOST/WERK_VLLM_PORT".to_string(),
                    path: None,
                    exists: usable,
                    usable,
                    detail: if usable {
                        format!("remote vLLM server reachable at http://{host}:{port}")
                    } else {
                        format!("remote vLLM server is not reachable at http://{host}:{port}")
                    },
                });
                if usable {
                    let command = VllmCommand::Remote { host, port };
                    return VllmDiscovery {
                        command: Some(command),
                        source: "env WERK_VLLM_HOST/WERK_VLLM_PORT".to_string(),
                        attempts,
                    };
                }
            }
            Err(err) => attempts.push(VllmDiscoveryAttempt {
                label: "WERK_VLLM_PORT".to_string(),
                path: None,
                exists: true,
                usable: false,
                detail: format!("invalid port: {err}"),
            }),
        }
    }

    if let Some(path) = env::var_os("WERK_VLLM_PYTHON").map(PathBuf::from) {
        let (usable, detail) = python_vllm_status(&path);
        attempts.push(VllmDiscoveryAttempt {
            label: "WERK_VLLM_PYTHON".to_string(),
            path: Some(path.clone()),
            exists: path.is_file(),
            usable,
            detail,
        });
        if usable {
            return VllmDiscovery {
                command: Some(VllmCommand::Python(path)),
                source: "env WERK_VLLM_PYTHON".to_string(),
                attempts,
            };
        }
    }

    let managed_python = managed_vllm_python(store);
    let (managed_usable, managed_detail) = python_vllm_status(&managed_python);
    attempts.push(VllmDiscoveryAttempt {
        label: "managed venv".to_string(),
        path: Some(managed_python.clone()),
        exists: managed_python.is_file(),
        usable: managed_usable,
        detail: managed_detail,
    });
    if managed_usable {
        return VllmDiscovery {
            command: Some(VllmCommand::Python(managed_python)),
            source: "managed venv".to_string(),
            attempts,
        };
    }

    if let Some(path) = find_in_path(vllm_executable_name()) {
        let usable = executable_supports_vllm(&path);
        attempts.push(VllmDiscoveryAttempt {
            label: format!("PATH: {}", vllm_executable_name()),
            path: Some(path.clone()),
            exists: true,
            usable,
            detail: if usable {
                "vLLM executable ok".to_string()
            } else {
                "vLLM executable did not run".to_string()
            },
        });
        if usable {
            return VllmDiscovery {
                command: Some(VllmCommand::Executable(path)),
                source: "PATH".to_string(),
                attempts,
            };
        }
    } else {
        attempts.push(VllmDiscoveryAttempt {
            label: format!("PATH: {}", vllm_executable_name()),
            path: None,
            exists: false,
            usable: false,
            detail: "not found".to_string(),
        });
    }

    for name in ["python3", "python"] {
        if let Some(path) = find_in_path(name) {
            let (usable, detail) = python_vllm_status(&path);
            attempts.push(VllmDiscoveryAttempt {
                label: format!("PATH: {name}"),
                path: Some(path.clone()),
                exists: true,
                usable,
                detail,
            });
            if usable {
                return VllmDiscovery {
                    command: Some(VllmCommand::Python(path)),
                    source: format!("PATH {name}"),
                    attempts,
                };
            }
        }
    }

    VllmDiscovery {
        command: None,
        source: "missing".to_string(),
        attempts,
    }
}

fn missing_vllm_message(discovery: &VllmDiscovery) -> String {
    let mut message = "No vLLM runtime found.\n\nTried:".to_string();
    for attempt in &discovery.attempts {
        let path = attempt
            .path
            .as_ref()
            .map(|path| format!(": {}", path.display()))
            .unwrap_or_default();
        let exists = if attempt.exists { "exists" } else { "missing" };
        let usable = if attempt.usable {
            "usable"
        } else {
            "not usable"
        };
        message.push_str(&format!(
            "\n- {}{} ({exists}, {usable}): {}",
            attempt.label, path, attempt.detail
        ));
    }
    message.push_str("\n\nFix:");
    message.push_str("\n- set WERK_VLLM_PYTHON=/path/to/python-with-vllm");
    message.push_str("\n- or set WERK_VLLM_HOST=127.0.0.1 and WERK_VLLM_PORT=<port>");
    message.push_str("\n- or run: werk backend install vllm");
    message.push_str("\n- or use: werk --backend candle ...");
    message
}

fn concise_vllm_unavailable_reason(discovery: &VllmDiscovery) -> String {
    if discovery.attempts.is_empty() {
        return "No vLLM runtime found; run: werk backend install vllm".to_string();
    }
    if let Some(attempt) = discovery
        .attempts
        .iter()
        .find(|attempt| attempt.exists && !attempt.usable)
        .or_else(|| discovery.attempts.iter().find(|attempt| !attempt.usable))
    {
        let path = attempt
            .path
            .as_ref()
            .map(|path| format!(" ({})", path.display()))
            .unwrap_or_default();
        format!("{}{}: {}", attempt.label, path, attempt.detail)
    } else {
        "No vLLM runtime found; run: werk backend install vllm".to_string()
    }
}

fn python_vllm_status(path: &Path) -> (bool, String) {
    if !path.is_file() {
        return (false, "Python path does not exist".to_string());
    }
    match Command::new(path)
        .arg("-c")
        .arg("import vllm; import vllm.entrypoints.openai.api_server")
        .output()
    {
        Ok(output) if output.status.success() => (true, "vLLM OpenAI server import ok".to_string()),
        Ok(output) => (
            false,
            command_failure_detail("Python cannot import vLLM OpenAI server", &output),
        ),
        Err(err) => (false, format!("failed to run Python: {err}")),
    }
}

fn python_rocm_status(path: &Path) -> (bool, String) {
    if !path.is_file() {
        return (false, "Python path does not exist".to_string());
    }
    match Command::new(path)
        .arg("-c")
        .arg(
            "import torch; hip = getattr(torch.version, 'hip', None); \
             assert hip, 'torch.version.hip is not set'; print(hip)",
        )
        .output()
    {
        Ok(output) if output.status.success() => (
            true,
            format!(
                "PyTorch ROCm/HIP runtime detected ({})",
                String::from_utf8_lossy(&output.stdout).trim()
            ),
        ),
        Ok(output) => (
            false,
            command_failure_detail("Python does not expose a ROCm/HIP PyTorch stack", &output),
        ),
        Err(err) => (false, format!("failed to run Python: {err}")),
    }
}

fn vllm_rocm_capability(command: &VllmCommand) -> Result<String> {
    match command {
        VllmCommand::Python(path) => {
            let (usable, detail) = python_rocm_status(path);
            if usable {
                Ok(detail)
            } else {
                bail!(
                    "vLLM is installed, but the Python environment is not ROCm-capable: {detail}. Install vLLM with a ROCm/HIP PyTorch build or use --backend cuda/auto."
                )
            }
        }
        VllmCommand::Remote { host, port } => {
            if remote_rocm_explicitly_confirmed() {
                Ok(format!(
                    "remote vLLM endpoint at http://{host}:{port} marked ROCm-capable by environment"
                ))
            } else {
                bail!(
                    "remote vLLM endpoint is reachable at http://{host}:{port}, but ROCm capability cannot be inferred. Set WERK_VLLM_ACCELERATOR=rocm or WERK_VLLM_ROCM=1 for an explicitly ROCm-backed remote server."
                )
            }
        }
        VllmCommand::Executable(path) => {
            bail!(
                "vLLM executable {} is installed, but ROCm capability cannot be verified from the executable. Set WERK_VLLM_PYTHON to a Python environment where torch.version.hip is set, or use a remote endpoint with WERK_VLLM_ACCELERATOR=rocm.",
                path.display()
            )
        }
    }
}

fn remote_rocm_explicitly_confirmed() -> bool {
    env::var("WERK_VLLM_ACCELERATOR")
        .map(|value| value.eq_ignore_ascii_case("rocm") || value.eq_ignore_ascii_case("hip"))
        .unwrap_or(false)
        || env::var("WERK_VLLM_ROCM")
            .map(|value| {
                matches!(
                    value.to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on" | "rocm" | "hip"
                )
            })
            .unwrap_or(false)
}

fn executable_supports_vllm(path: &Path) -> bool {
    Command::new(path)
        .arg("serve")
        .arg("--help")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn validate_vllm_python(path: &Path) -> Result<()> {
    let (usable, detail) = python_vllm_status(path);
    if !usable {
        bail!(
            "vLLM installation validation failed for {}: {}",
            path.display(),
            detail
        );
    }
    let output = Command::new(path)
        .arg("-m")
        .arg("vllm.entrypoints.openai.api_server")
        .arg("--help")
        .output()
        .with_context(|| {
            format!(
                "failed to validate vLLM OpenAI server entrypoint with {}",
                path.display()
            )
        })?;
    if !output.status.success() {
        bail!(
            "vLLM OpenAI server entrypoint validation failed for {}: {}",
            path.display(),
            command_failure_detail("server module did not start", &output)
        );
    }
    Ok(())
}

fn remote_supports_vllm(host: &str, port: u16) -> bool {
    let url = format!("http://{host}:{port}");
    get(&url, "/v1/models")
        .map(|response| response.status == 200)
        .unwrap_or(false)
        || get(&url, "/health")
            .map(|response| response.status == 200)
            .unwrap_or(false)
}

fn command_failure_detail(prefix: &str, output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let detail = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        output.status.to_string()
    };
    format!("{prefix}: {detail}")
}

fn compact_error(reason: &str) -> String {
    reason.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn vllm_version(command: &VllmCommand) -> Option<String> {
    match command {
        VllmCommand::Python(path) => {
            let output = Command::new(path)
                .arg("-c")
                .arg("import vllm; print(getattr(vllm, '__version__', 'unknown'))")
                .output()
                .ok()?;
            output
                .status
                .success()
                .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
        }
        VllmCommand::Executable(path) => {
            let output = Command::new(path).arg("--version").output().ok()?;
            output.status.success().then(|| {
                String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .next()
                    .unwrap_or("unknown")
                    .trim()
                    .to_string()
            })
        }
        VllmCommand::Remote { host, port } => get(&format!("http://{host}:{port}"), "/v1/models")
            .ok()
            .filter(|response| response.status == 200)
            .map(|_| "remote server reachable".to_string()),
    }
}

fn get(base_url: &str, path: &str) -> Result<HttpResponse> {
    request(base_url, path, "GET", None)
}

fn post_json(base_url: &str, path: &str, body: &Value) -> Result<HttpResponse> {
    request(base_url, path, "POST", Some(body))
}

fn request(base_url: &str, path: &str, method: &str, body: Option<&Value>) -> Result<HttpResponse> {
    let (_, host, port) = parse_http_url(base_url)?;
    let mut stream = TcpStream::connect((host.as_str(), port))
        .with_context(|| format!("failed to connect to vLLM server at {base_url}"))?;
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
        .ok_or_else(|| anyhow!("invalid HTTP response from vLLM server: {status_line:?}"))?;
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
        bail!("vLLM HTTP {status}: {}", text.trim());
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
                .with_context(|| format!("invalid chunk size from vLLM: {size_text}"))?;
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

fn parse_http_url(url: &str) -> Result<(String, String, u16)> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| anyhow!("only http vLLM URLs are supported: {url}"))?;
    let (host_port, _) = rest.split_once('/').unwrap_or((rest, ""));
    let (host, port) = host_port
        .rsplit_once(':')
        .ok_or_else(|| anyhow!("vLLM URL has no port: {url}"))?;
    Ok(("http".to_string(), host.to_string(), port.parse()?))
}

fn header_contains(headers: &[(String, String)], name: &str, needle: &str) -> bool {
    headers.iter().any(|(header, value)| {
        header.eq_ignore_ascii_case(name) && value.to_ascii_lowercase().contains(needle)
    })
}

fn find_sse_boundary(bytes: &[u8]) -> Option<usize> {
    bytes
        .windows(2)
        .position(|window| window == b"\n\n")
        .or_else(|| bytes.windows(4).position(|window| window == b"\r\n\r\n"))
}

fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count().max(1)
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
                backend_diagnostics: response.backend_diagnostics,
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

fn run_command(command: &mut Command, context: &str) -> Result<()> {
    let status = command.status().with_context(|| context.to_string())?;
    if !status.success() {
        bail!("{context}; command exited with {status}");
    }
    Ok(())
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

fn find_bootstrap_python() -> Option<PathBuf> {
    env::var_os("WERK_VLLM_PYTHON")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| find_in_path("python3"))
        .or_else(|| find_in_path("python"))
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

fn vllm_executable_name() -> &'static str {
    if cfg!(windows) { "vllm.exe" } else { "vllm" }
}

fn free_local_port() -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    Ok(listener.local_addr()?.port())
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

fn env_true(name: &str) -> bool {
    env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

fn shell_join(args: &[String]) -> String {
    args.iter()
        .map(|arg| {
            if arg
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || "-_./:=+".contains(ch))
            {
                arg.clone()
            } else {
                format!("'{}'", arg.replace('\'', "'\\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_error_chain(err: &anyhow::Error) -> String {
    let mut parts = err.chain().map(ToString::to_string).collect::<Vec<_>>();
    parts.dedup();
    parts.join(": ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_store::ModelSource;
    use crate::openai::{ChatMessage, MessageContent};
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn chat_completion_body_uses_openai_messages() {
        let request = GenerateRequest {
            prompt: "ignored rendered prompt".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some(MessageContent::Text("hello".to_string())),
                name: None,
            }],
            image_urls: Vec::new(),
            max_tokens: 32,
            temperature: Some(0.2),
            top_p: None,
            stop: vec!["stop".to_string()],
            seed: Some(7),
            stream_granularity: super::super::StreamGranularity::Chunk,
            verbose: false,
            debug: false,
        };
        let body = chat_completion_body("model", &request);
        assert_eq!(body["model"], "model");
        assert_eq!(body["messages"][0]["content"], "hello");
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn parses_vllm_streaming_delta_and_usage() {
        let mut completion = VllmCompletion::default();
        let value = json!({
            "choices": [{"delta": {"content": "hello"}, "finish_reason": null}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2}
        });
        assert_eq!(delta_content(&value).as_deref(), Some("hello"));
        update_completion_from_event(&mut completion, &value);
        assert_eq!(completion.prompt_tokens, 4);
        assert_eq!(completion.completion_tokens, 2);
    }

    #[test]
    fn vllm_model_dir_prefers_manifest_config_parent() {
        let store = test_store("vllm-config-parent");
        let manifest = test_manifest(
            "Qwen/Qwen3-4B",
            Some("files/snapshots/main/config.json"),
            Some("files/model.safetensors"),
        );
        let root = store.model_dir(&manifest.id);
        fs::create_dir_all(root.join("files/snapshots/main")).unwrap();
        fs::create_dir_all(root.join("files")).unwrap();
        fs::write(root.join("files/snapshots/main/config.json"), b"{}").unwrap();
        fs::write(root.join("files/config.json"), b"{}").unwrap();

        let resolved = resolve_vllm_model_dir(&store, &manifest).unwrap();
        assert_eq!(resolved, root.join("files/snapshots/main"));
    }

    #[test]
    fn vllm_model_dir_falls_back_to_files_dir_with_config() {
        let store = test_store("vllm-files-dir");
        let manifest = test_manifest("Qwen/Qwen3-4B", None, Some("files/model.safetensors"));
        let root = store.model_dir(&manifest.id);
        fs::create_dir_all(root.join("files")).unwrap();
        fs::write(root.join("files/config.json"), b"{}").unwrap();

        let resolved = resolve_vllm_model_dir(&store, &manifest).unwrap();
        assert_eq!(resolved, root.join("files"));
    }

    #[test]
    fn vllm_model_dir_uses_root_only_when_root_contains_config() {
        let store = test_store("vllm-root-dir");
        let manifest = test_manifest("Qwen/Qwen3-4B", None, None);
        let root = store.model_dir(&manifest.id);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("config.json"), b"{}").unwrap();

        let resolved = resolve_vllm_model_dir(&store, &manifest).unwrap();
        assert_eq!(resolved, root);
    }

    #[test]
    fn vllm_model_dir_rejects_store_root_without_config() {
        let store = test_store("vllm-no-config");
        let manifest = test_manifest("Qwen/Qwen3-4B", None, Some("files/model.safetensors"));
        let root = store.model_dir(&manifest.id);
        fs::create_dir_all(root.join("files")).unwrap();

        let err = resolve_vllm_model_dir(&store, &manifest).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("config.json"));
        assert!(message.contains("Qwen/Qwen3-4B"));
    }

    #[test]
    fn vllm_args_keep_logical_served_model_name() {
        let model_dir = PathBuf::from("/tmp/werk-model/files");
        let args = vllm_server_args(
            &VllmCommand::Python(PathBuf::from("/usr/bin/python3")),
            &model_dir,
            "Qwen/Qwen3-4B",
            12345,
        );
        let model_arg = args
            .windows(2)
            .find(|pair| pair[0] == "--model")
            .map(|pair| pair[1].as_str());
        let served_name = args
            .windows(2)
            .find(|pair| pair[0] == "--served-model-name")
            .map(|pair| pair[1].as_str());
        assert_eq!(model_arg, Some("/tmp/werk-model/files"));
        assert_eq!(served_name, Some("Qwen/Qwen3-4B"));
    }

    #[test]
    fn linux_release_detection_identifies_wsl() {
        assert!(linux_release_looks_like_wsl(
            "5.15.167.4-microsoft-standard-WSL2"
        ));
        assert!(linux_release_looks_like_wsl("Linux version 6.6.0 WSL2"));
        assert!(!linux_release_looks_like_wsl("6.8.0-63-generic"));
    }

    #[test]
    fn local_vllm_policy_rejects_wsl_with_fallback_message() {
        let reason = local_vllm_platform_rejection(VllmPlatform::Wsl).unwrap();
        assert!(reason.contains("vLLM is a Linux-native runtime"));
        assert!(reason.contains("Werk will fall back to Candle CUDA"));
        assert!(reason.contains("remote vLLM server"));
        assert!(local_vllm_platform_rejection(VllmPlatform::NativeLinux).is_none());
    }

    #[test]
    fn managed_vllm_install_policy_allows_linux_and_wsl_only() {
        assert!(managed_vllm_install_rejection(VllmPlatform::NativeLinux).is_none());
        assert!(managed_vllm_install_rejection(VllmPlatform::Wsl).is_none());

        let windows = managed_vllm_install_rejection(VllmPlatform::NativeWindows).unwrap();
        assert!(windows.contains("Native Windows local vLLM is not eligible"));

        let macos = managed_vllm_install_rejection(VllmPlatform::Macos).unwrap();
        assert!(macos.contains("not eligible on macOS"));
    }

    #[test]
    fn vllm_health_marks_wsl_local_as_best_effort_but_remote_healthy() {
        let local = VllmDiscovery {
            command: Some(VllmCommand::Python(PathBuf::from("/tmp/python"))),
            source: "test".to_string(),
            attempts: Vec::new(),
        };
        let health = vllm_health_for_platform(&local, VllmPlatform::Wsl);
        assert_eq!(health.installed_label, "yes");
        assert_eq!(health.health_label, "best-effort on WSL");
        assert!(!health.healthy);

        let remote = VllmDiscovery {
            command: Some(VllmCommand::Remote {
                host: "127.0.0.1".to_string(),
                port: 8000,
            }),
            source: "test".to_string(),
            attempts: Vec::new(),
        };
        let health = vllm_health_for_platform(&remote, VllmPlatform::Wsl);
        assert_eq!(health.installed_label, "remote");
        assert_eq!(health.health_label, "healthy");
        assert!(health.healthy);
    }

    #[test]
    fn vllm_health_marks_wsl_missing_local_as_best_effort() {
        let discovery = VllmDiscovery {
            command: None,
            source: "missing".to_string(),
            attempts: Vec::new(),
        };
        let health = vllm_health_for_platform(&discovery, VllmPlatform::Wsl);
        assert_eq!(health.installed_label, "no");
        assert_eq!(health.health_label, "best-effort on WSL");
        assert!(!health.healthy);
        assert!(health.detail.contains("Werk will fall back to Candle CUDA"));
    }

    #[test]
    fn wsl_sensitive_vllm_failure_markers_are_detected() {
        assert!(is_wsl_sensitive_vllm_failure("UVA is not available"));
        assert!(is_wsl_sensitive_vllm_failure("pin_memory failed"));
        assert!(is_wsl_sensitive_vllm_failure("CUDA IPC handle failed"));
        assert!(is_wsl_sensitive_vllm_failure("engine-core failed to start"));
        assert!(is_wsl_sensitive_vllm_failure(
            "multiprocessing spawn failed during startup"
        ));
        assert!(!is_wsl_sensitive_vllm_failure("model file not found"));
    }

    #[cfg(unix)]
    #[test]
    fn vllm_rocm_capability_accepts_python_with_hip_stack() {
        let python = fake_python("rocm-ok", "printf '6.3.0\\n'\nexit 0\n");
        let detail = vllm_rocm_capability(&VllmCommand::Python(python)).unwrap();
        assert!(detail.contains("PyTorch ROCm/HIP runtime detected"));
        assert!(detail.contains("6.3.0"));
    }

    #[cfg(unix)]
    #[test]
    fn vllm_rocm_capability_rejects_python_without_hip_stack() {
        let python = fake_python(
            "rocm-missing",
            "printf 'torch.version.hip is not set\\n' >&2\nexit 1\n",
        );
        let err = vllm_rocm_capability(&VllmCommand::Python(python)).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("not ROCm-capable"));
        assert!(message.contains("ROCm/HIP PyTorch"));
    }

    #[test]
    fn vllm_rocm_capability_rejects_plain_executable_discovery() {
        let err = vllm_rocm_capability(&VllmCommand::Executable(PathBuf::from("/usr/bin/vllm")))
            .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("ROCm capability cannot be verified"));
        assert!(message.contains("WERK_VLLM_PYTHON"));
    }

    #[cfg(unix)]
    fn fake_python(name: &str, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "werk1112-vllm-{name}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("python");
        fs::write(&path, format!("#!/bin/sh\n{body}")).unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).unwrap();
        path
    }

    fn test_store(name: &str) -> ModelStore {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "werk1112-vllm-store-{name}-{}-{nanos}",
            std::process::id()
        ));
        ModelStore::resolve(Some(root)).unwrap()
    }

    fn test_manifest(
        id: &str,
        config_path: Option<&str>,
        model_path: Option<&str>,
    ) -> ModelManifest {
        ModelManifest {
            id: id.to_string(),
            source: ModelSource::LocalPath {
                path: "test".to_string(),
            },
            format: ModelFormat::SafeTensors,
            architecture: Some("qwen3".to_string()),
            tokenizer_path: Some("files/tokenizer.json".to_string()),
            config_path: config_path.map(str::to_string),
            model_path: model_path.map(str::to_string),
            backend: "test".to_string(),
            created_unix: 1,
            files: Vec::new(),
            artifacts: Vec::new(),
            metadata: Default::default(),
        }
    }
}
