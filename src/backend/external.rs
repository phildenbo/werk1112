use anyhow::{Context, Result, anyhow, bail};
use std::{
    env, fs,
    io::Read,
    path::PathBuf,
    process::{Command, Stdio},
    thread,
    time::Instant,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::{
    GenerateRequest, GenerateResponse, GenerateStream, GenerateStreamEvent, GenerationBackend,
    GenerationTimings,
};
use crate::model_store::{ModelFormat, ModelManifest, ModelStore};

#[derive(Debug, Clone)]
pub struct LlamaCppBackend {
    store: ModelStore,
    executable: PathBuf,
}

#[derive(Debug, Clone)]
pub struct MlxBackend {
    store: ModelStore,
    python: PathBuf,
    module: String,
}

impl LlamaCppBackend {
    pub fn new_vulkan(store: ModelStore) -> Self {
        Self {
            store,
            executable: backend_program("WERK_LLAMA_CLI", default_llama_cli()),
        }
    }

    pub fn probe_vulkan() -> Result<String> {
        let executable = backend_program("WERK_LLAMA_CLI", default_llama_cli());
        let output = Command::new(&executable)
            .arg("--help")
            .output()
            .with_context(|| {
                format!(
                    "failed to execute {}; set WERK_LLAMA_CLI to a llama.cpp Vulkan llama-cli binary",
                    executable.display()
                )
            })?;
        if !output.status.success() {
            bail!(
                "{} --help failed: {}",
                executable.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        Ok(format!("llama.cpp CLI at {}", executable.display()))
    }

    fn command_for(&self, manifest: &ModelManifest, request: &GenerateRequest) -> Result<Command> {
        if manifest.format != ModelFormat::Gguf {
            bail!("vulkan backend uses llama.cpp and currently supports GGUF models only");
        }
        let model_path = manifest
            .model_path
            .as_deref()
            .context("GGUF manifest has no model_path")?;
        let mut command = Command::new(&self.executable);
        command
            .arg("-m")
            .arg(self.store.absolute_model_file(manifest, model_path))
            .arg("-p")
            .arg(&request.prompt)
            .arg("-n")
            .arg(request.max_tokens.to_string())
            .arg("-ngl")
            .arg("999")
            .arg("--no-display-prompt");
        if let Some(temperature) = request.temperature {
            command.arg("--temp").arg(format_float(temperature));
        }
        if let Some(top_p) = request.top_p {
            command.arg("--top-p").arg(format_float(top_p));
        }
        if let Some(seed) = request.seed {
            command.arg("--seed").arg(seed.to_string());
        }
        for image in &request.image_urls {
            command.arg("--image").arg(image);
        }
        Ok(command)
    }
}

impl GenerationBackend for LlamaCppBackend {
    fn generate(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> Result<GenerateResponse> {
        let mut command = self.command_for(manifest, &request)?;
        let started = Instant::now();
        let output = command
            .output()
            .with_context(|| format!("failed to execute {}", self.executable.display()))?;
        if !output.status.success() {
            bail!(
                "llama.cpp generation failed: {}",
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
            let result = backend
                .command_for(&manifest, &request)
                .and_then(|command| {
                    run_streaming_command(
                        command,
                        &request,
                        "llama.cpp generation failed",
                        tx.clone(),
                    )
                });
            if let Err(err) = result {
                let _ = tx.blocking_send(Err(err.to_string()));
            }
        });
        Box::pin(ReceiverStream::new(rx))
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
                let _ = tx.blocking_send(Err(err.to_string()));
            }
        });
        Box::pin(ReceiverStream::new(rx))
    }
}

fn run_streaming_command(
    mut command: Command,
    request: &GenerateRequest,
    error_context: &str,
    tx: mpsc::Sender<Result<GenerateStreamEvent, String>>,
) -> Result<()> {
    let started = Instant::now();
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn external backend process")?;
    let mut stdout = child
        .stdout
        .take()
        .context("failed to capture external backend stdout")?;
    let stderr = child.stderr.take();
    let stderr_reader = stderr.map(|mut stderr| {
        thread::spawn(move || {
            let mut text = String::new();
            let _ = stderr.read_to_string(&mut text);
            text
        })
    });

    let mut text = String::new();
    let mut buffer = [0_u8; 256];
    loop {
        let n = stdout
            .read(&mut buffer)
            .context("failed to read external backend stdout")?;
        if n == 0 {
            break;
        }
        let chunk = String::from_utf8_lossy(&buffer[..n]).to_string();
        text.push_str(&chunk);
        let _ = tx.blocking_send(Ok(GenerateStreamEvent::TextChunk(chunk)));
    }

    let status = child
        .wait()
        .context("failed to wait for external backend")?;
    let stderr = stderr_reader
        .map(|reader| reader.join().unwrap_or_default())
        .unwrap_or_default();
    if !status.success() {
        bail!("{error_context}: {}", stderr.trim());
    }

    let mut final_text = text;
    let finish_reason = truncate_at_stop(&mut final_text, &request.stop);
    let elapsed = started.elapsed().as_secs_f64();
    let prompt_tokens = estimate_tokens(&request.prompt);
    let completion_tokens = estimate_tokens(&final_text);
    tx.blocking_send(Ok(GenerateStreamEvent::Done {
        finish_reason,
        prompt_tokens,
        completion_tokens,
        timings: external_timings(elapsed),
    }))
    .map_err(|err| anyhow!("stream receiver closed: {err}"))?;
    Ok(())
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
        timings: external_timings(elapsed),
    }
}

fn external_timings(elapsed: f64) -> GenerationTimings {
    GenerationTimings {
        load_seconds: 0.0,
        prompt_seconds: 0.0,
        decode_seconds: elapsed,
        total_seconds: elapsed,
    }
}

fn truncate_at_stop(text: &mut String, stops: &[String]) -> String {
    for stop in stops {
        if stop.is_empty() {
            continue;
        }
        if let Some(index) = text.find(stop) {
            text.truncate(index);
            return "stop".to_string();
        }
    }
    "length".to_string()
}

fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count()
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

fn default_llama_cli() -> &'static str {
    if cfg!(windows) {
        "llama-cli.exe"
    } else {
        "llama-cli"
    }
}

fn default_python() -> &'static str {
    if cfg!(windows) {
        "python.exe"
    } else {
        "python3"
    }
}

#[allow(dead_code)]
fn _assert_program_path_is_readable(path: &PathBuf) -> bool {
    fs::metadata(path)
        .map(|metadata| metadata.is_file())
        .unwrap_or(false)
}
