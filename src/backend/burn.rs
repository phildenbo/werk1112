use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::{
    GenerateRequest, GenerateResponse, GenerateStream, GenerateStreamEvent, GenerationBackend,
    GenerationTimings,
};
use crate::model_store::{ModelFormat, ModelManifest, ModelStore};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BurnMode {
    Cuda,
    Cpu,
}

impl BurnMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Cuda => "burn-cuda",
            Self::Cpu => "burn-cpu",
        }
    }

    pub fn display(self) -> &'static str {
        match self {
            Self::Cuda => "Burn CUDA",
            Self::Cpu => "Burn CPU",
        }
    }

    fn backend_arg(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }
}

#[derive(Clone)]
pub struct BurnBackend {
    store: ModelStore,
    mode: BurnMode,
}

#[derive(Debug, Clone)]
pub struct BurnDiscovery {
    pub path: Option<PathBuf>,
    pub source: String,
    pub attempts: Vec<BurnDiscoveryAttempt>,
}

#[derive(Debug, Clone)]
pub struct BurnDiscoveryAttempt {
    pub label: String,
    pub path: Option<PathBuf>,
    pub exists: bool,
    pub usable: bool,
    pub detail: String,
}

impl BurnBackend {
    pub fn new(store: ModelStore, mode: BurnMode) -> Self {
        Self { store, mode }
    }

    pub fn probe(store: &ModelStore, mode: BurnMode) -> Result<String> {
        let discovery = discover_burn(store, mode);
        let path = discovery
            .path
            .as_ref()
            .ok_or_else(|| anyhow!("{}", missing_message_from_discovery(mode, &discovery)))?;
        Ok(format!("{} runner {}", mode.display(), path.display()))
    }

    pub fn discover(store: &ModelStore, mode: BurnMode) -> BurnDiscovery {
        discover_burn(store, mode)
    }

    pub fn missing_message(store: &ModelStore, mode: BurnMode) -> String {
        missing_message_from_discovery(mode, &discover_burn(store, mode))
    }

    pub fn unavailable_reason(store: &ModelStore, mode: BurnMode) -> String {
        concise_unavailable_reason(&discover_burn(store, mode))
    }

    fn runner(&self) -> Result<PathBuf> {
        discover_burn(&self.store, self.mode)
            .path
            .as_ref()
            .cloned()
            .ok_or_else(|| anyhow!("{}", Self::missing_message(&self.store, self.mode)))
    }

    fn generate_inner(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> Result<GenerateResponse> {
        if manifest.format != ModelFormat::SafeTensors {
            bail!("Burn backend supports HF safetensors model directories only");
        }
        if !request.image_urls.is_empty() {
            bail!("Burn text backend received image inputs; use a VLM-capable model/runtime");
        }

        let total_started = Instant::now();
        let runner = self.runner()?;
        let model_dir = self.store.model_dir(&manifest.id);
        if !model_dir.is_dir() {
            bail!(
                "model directory for '{}' does not exist: {}",
                manifest.id,
                model_dir.display()
            );
        }
        if request.debug {
            eprintln!("selected backend: {}", self.mode.label());
            eprintln!("Burn runner: {}", runner.display());
            eprintln!("model path: {}", model_dir.display());
        }

        let started = Instant::now();
        let mut command = Command::new(&runner);
        command
            .arg("--model")
            .arg(&model_dir)
            .arg("--prompt")
            .arg(&request.prompt)
            .arg("--max-tokens")
            .arg(request.max_tokens.to_string())
            .arg("--backend")
            .arg(self.mode.backend_arg())
            .arg("--json");
        if let Some(temperature) = request.temperature {
            command.arg("--temperature").arg(temperature.to_string());
        }
        if let Some(top_p) = request.top_p {
            command.arg("--top-p").arg(top_p.to_string());
        }
        if let Some(seed) = request.seed {
            command.arg("--seed").arg(seed.to_string());
        }

        let output = command
            .output()
            .with_context(|| format!("failed to run Burn runner {}", runner.display()))?;
        if !output.status.success() {
            bail!("Burn runner failed: {}", command_output_detail(&output));
        }
        let value: Value = serde_json::from_slice(&output.stdout).with_context(|| {
            format!(
                "Burn runner returned invalid JSON: {}",
                String::from_utf8_lossy(&output.stdout)
            )
        })?;
        let text = value
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("Burn runner JSON missing string field 'text'"))?
            .to_string();
        let prompt_tokens = value
            .get("prompt_tokens")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or_else(|| request.prompt.split_whitespace().count().max(1));
        let completion_tokens = value
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or_else(|| text.split_whitespace().count().max(1));
        let elapsed = started.elapsed().as_secs_f64();
        Ok(GenerateResponse {
            text,
            prompt_tokens,
            completion_tokens,
            finish_reason: value
                .get("finish_reason")
                .and_then(Value::as_str)
                .unwrap_or("stop")
                .to_string(),
            timings: GenerationTimings {
                load_seconds: 0.0,
                warmup_seconds: 0.0,
                first_token_seconds: 0.0,
                prompt_seconds: 0.0,
                decode_seconds: elapsed,
                total_seconds: total_started.elapsed().as_secs_f64(),
            },
        })
    }
}

impl GenerationBackend for BurnBackend {
    fn prepare(&self, _manifest: &ModelManifest) -> Result<()> {
        eprintln!("Using {} backend", self.mode.display());
        self.runner().map(|_| ())
    }

    fn generate(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> Result<GenerateResponse> {
        self.generate_inner(manifest, request)
    }

    fn generate_stream(&self, manifest: ModelManifest, request: GenerateRequest) -> GenerateStream {
        let backend = self.clone();
        let (tx, rx) = mpsc::channel(4);
        tokio::task::spawn_blocking(move || {
            let result = backend.generate_inner(&manifest, request);
            match result {
                Ok(response) => {
                    if !response.text.is_empty() {
                        let _ = tx.blocking_send(Ok(GenerateStreamEvent::TextChunk(
                            response.text.clone(),
                        )));
                    }
                    let _ = tx.blocking_send(Ok(GenerateStreamEvent::Done {
                        finish_reason: response.finish_reason,
                        prompt_tokens: response.prompt_tokens,
                        completion_tokens: response.completion_tokens,
                        timings: response.timings,
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

fn discover_burn(store: &ModelStore, mode: BurnMode) -> BurnDiscovery {
    let mut attempts = Vec::new();
    let env_name = match mode {
        BurnMode::Cuda => "WERK_BURN_RUNNER_CUDA",
        BurnMode::Cpu => "WERK_BURN_RUNNER_CPU",
    };
    for (label, path) in [
        (
            env_name.to_string(),
            env::var_os(env_name).map(PathBuf::from),
        ),
        (
            "WERK_BURN_RUNNER".to_string(),
            env::var_os("WERK_BURN_RUNNER").map(PathBuf::from),
        ),
        (
            "managed cache".to_string(),
            Some(managed_runner_path(store, mode)),
        ),
        (
            "PATH: werk-burn-runner".to_string(),
            find_in_path(runner_name()),
        ),
    ] {
        let Some(path) = path else {
            attempts.push(BurnDiscoveryAttempt {
                label,
                path: None,
                exists: false,
                usable: false,
                detail: "not set".to_string(),
            });
            continue;
        };
        let usable = runner_help_ok(&path);
        attempts.push(BurnDiscoveryAttempt {
            label: label.clone(),
            path: Some(path.clone()),
            exists: path.is_file(),
            usable,
            detail: if usable {
                "runner --help ok".to_string()
            } else {
                "runner missing or --help failed".to_string()
            },
        });
        if usable {
            return BurnDiscovery {
                path: Some(path),
                source: label,
                attempts,
            };
        }
    }
    BurnDiscovery {
        path: None,
        source: "missing".to_string(),
        attempts,
    }
}

pub fn managed_runner_path(store: &ModelStore, mode: BurnMode) -> PathBuf {
    store
        .home()
        .join("backends")
        .join(mode.label())
        .join(runner_name())
}

fn missing_message_from_discovery(mode: BurnMode, discovery: &BurnDiscovery) -> String {
    let mut message = format!(
        "{} requested but no runner is installed.\n\nTried:",
        mode.display()
    );
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
    message.push_str("\n- install a Werk release artifact that includes the Burn runner");
    message.push_str("\n- or set WERK_BURN_RUNNER=/path/to/werk-burn-runner");
    message.push_str(&format!(
        "\n- or set WERK_BURN_RUNNER_{}=/path/to/werk-burn-runner",
        match mode {
            BurnMode::Cuda => "CUDA",
            BurnMode::Cpu => "CPU",
        }
    ));
    message
}

fn concise_unavailable_reason(discovery: &BurnDiscovery) -> String {
    if discovery
        .attempts
        .iter()
        .any(|attempt| attempt.exists && !attempt.usable)
    {
        "Burn runner validation failed".to_string()
    } else {
        "Burn runner not installed".to_string()
    }
}

fn runner_help_ok(path: &Path) -> bool {
    path.is_file()
        && Command::new(path)
            .arg("--help")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
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

fn runner_name() -> &'static str {
    if cfg!(windows) {
        "werk-burn-runner.exe"
    } else {
        "werk-burn-runner"
    }
}

fn command_output_detail(output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        output.status.to_string()
    }
}
