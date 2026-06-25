use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
use std::{
    env, fs,
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
use crate::model_store::{ArtifactKind, ArtifactStatus, ModelFormat, ModelManifest, ModelStore};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxRuntimeMode {
    Cuda,
    Rocm,
    Cpu,
}

impl OnnxRuntimeMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Cuda => "onnxruntime-cuda",
            Self::Rocm => "onnxruntime-rocm",
            Self::Cpu => "onnxruntime-cpu",
        }
    }

    pub fn display(self) -> &'static str {
        match self {
            Self::Cuda => "ONNX Runtime CUDA",
            Self::Rocm => "ONNX Runtime ROCm",
            Self::Cpu => "ONNX Runtime CPU",
        }
    }

    fn bundle_env(self) -> &'static str {
        match self {
            Self::Cuda => "WERK_ONNX_RUNTIME_BUNDLE_CUDA",
            Self::Rocm => "WERK_ONNX_RUNTIME_BUNDLE_ROCM",
            Self::Cpu => "WERK_ONNX_RUNTIME_BUNDLE_CPU",
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OnnxProvisionOptions {
    pub install_missing_runtime: bool,
    pub verbose: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxRuntimeAvailability {
    Ready,
    Installable,
    Unavailable,
}

#[derive(Clone)]
pub struct OnnxRuntimeBackend {
    store: ModelStore,
    mode: OnnxRuntimeMode,
}

#[derive(Debug, Clone)]
pub struct OnnxRuntimeDiscovery {
    pub path: Option<PathBuf>,
    pub source: String,
    pub attempts: Vec<OnnxRuntimeAttempt>,
}

#[derive(Debug, Clone)]
pub struct OnnxRuntimeAttempt {
    pub label: String,
    pub path: Option<PathBuf>,
    pub exists: bool,
    pub usable: bool,
    pub detail: String,
}

impl OnnxRuntimeBackend {
    pub fn new(store: ModelStore, mode: OnnxRuntimeMode) -> Self {
        Self { store, mode }
    }

    pub fn probe(store: &ModelStore, mode: OnnxRuntimeMode) -> Result<String> {
        let discovery = discover_onnx_runtime(store, mode);
        let path = discovery
            .path
            .as_ref()
            .ok_or_else(|| anyhow!("{}", missing_message_from_discovery(mode, &discovery)))?;
        Ok(format!("{} runner {}", mode.display(), path.display()))
    }

    pub fn discover(store: &ModelStore, mode: OnnxRuntimeMode) -> OnnxRuntimeDiscovery {
        discover_onnx_runtime(store, mode)
    }

    pub fn missing_message(store: &ModelStore, mode: OnnxRuntimeMode) -> String {
        missing_message_from_discovery(mode, &discover_onnx_runtime(store, mode))
    }

    pub fn unavailable_reason(store: &ModelStore, mode: OnnxRuntimeMode) -> String {
        concise_unavailable_reason(&discover_onnx_runtime(store, mode))
    }

    pub fn availability(store: &ModelStore, mode: OnnxRuntimeMode) -> OnnxRuntimeAvailability {
        let discovery = discover_onnx_runtime(store, mode);
        if discovery.path.is_some() {
            OnnxRuntimeAvailability::Ready
        } else if find_bundled_runner(mode).is_some() {
            OnnxRuntimeAvailability::Installable
        } else {
            OnnxRuntimeAvailability::Unavailable
        }
    }

    pub fn ensure_available_for_model(
        store: &ModelStore,
        manifest: &ModelManifest,
        mode: OnnxRuntimeMode,
    ) -> Result<()> {
        Self::ensure_available_for_model_with_options(
            store,
            manifest,
            mode,
            OnnxProvisionOptions::default(),
        )
    }

    pub fn ensure_available_for_model_with_options(
        store: &ModelStore,
        manifest: &ModelManifest,
        mode: OnnxRuntimeMode,
        options: OnnxProvisionOptions,
    ) -> Result<()> {
        if !matches!(
            manifest.format,
            ModelFormat::SafeTensors | ModelFormat::Onnx
        ) {
            bail!("ONNX Runtime route requires a safetensors source model or direct ONNX model");
        }
        let mut discovery = discover_onnx_runtime(store, mode);
        if discovery.path.is_none() {
            if options.install_missing_runtime {
                install_managed_onnx_runtime(store, mode)?;
                discovery = discover_onnx_runtime(store, mode);
            }
        }
        if discovery.path.is_none() {
            bail!("{}", missing_message_from_discovery(mode, &discovery));
        }
        if options.verbose {
            eprintln!("Selected runtime: {}", mode.display());
            eprintln!("Runtime status: ready");
        }
        if manifest.format == ModelFormat::Onnx {
            if options.verbose {
                eprintln!("Artifact: direct ONNX model");
                eprintln!("Result: runtime ready");
            }
            return Ok(());
        }
        if store.ready_onnx_artifact(manifest).is_some() {
            if options.verbose {
                eprintln!("Artifact: ready");
                eprintln!("Result: runtime ready");
            }
            return Ok(());
        }
        if options.verbose {
            eprintln!("Artifact: building ONNX export");
        }
        if let Err(err) = store
            .build_onnx_artifact(&manifest.id, false)
            .with_context(|| "ONNX artifact generation failed")
        {
            if options.verbose {
                eprintln!("Result: artifact build failed: {err}");
            }
            return Err(err);
        }
        if options.verbose {
            eprintln!("Result: runtime ready");
        }
        Ok(())
    }

    fn runner(&self) -> Result<PathBuf> {
        discover_onnx_runtime(&self.store, self.mode)
            .path
            .as_ref()
            .cloned()
            .ok_or_else(|| anyhow!("{}", Self::missing_message(&self.store, self.mode)))
    }

    fn model_path(&self, manifest: &ModelManifest) -> Result<PathBuf> {
        if manifest.format == ModelFormat::Onnx {
            let path = manifest
                .model_path
                .as_deref()
                .context("ONNX manifest has no model_path")?;
            return Ok(self.store.absolute_model_file(manifest, path));
        }
        if manifest.format != ModelFormat::SafeTensors {
            bail!(
                "ONNX Runtime backend supports safetensors source models and direct ONNX models only"
            );
        }
        if let Some(artifact) = self.store.ready_onnx_artifact(manifest) {
            if artifact.status != ArtifactStatus::Ready || artifact.kind != ArtifactKind::Onnx {
                bail!("ONNX artifact for '{}' is not ready", manifest.id);
            }
            return Ok(self.store.model_dir(&manifest.id).join(&artifact.path));
        }
        let artifact = self.store.build_onnx_artifact(&manifest.id, false)?;
        if artifact.status != ArtifactStatus::Ready || artifact.kind != ArtifactKind::Onnx {
            bail!("ONNX artifact for '{}' is not ready", manifest.id);
        }
        Ok(self.store.model_dir(&manifest.id).join(&artifact.path))
    }

    fn generate_inner(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> Result<GenerateResponse> {
        if !request.image_urls.is_empty() {
            bail!(
                "ONNX Runtime text backend received image inputs; use a VLM-capable model/runtime"
            );
        }
        let total_started = Instant::now();
        let runner = self.runner()?;
        let model_path = self.model_path(manifest)?;
        if request.verbose {
            eprintln!("Starting generation...");
        }
        if request.debug {
            eprintln!("selected backend: {}", self.mode.label());
            eprintln!("ONNX Runtime runner: {}", runner.display());
            eprintln!("ONNX model: {}", model_path.display());
        }
        let started = Instant::now();
        let output = Command::new(&runner)
            .arg("--model")
            .arg(&model_path)
            .arg("--prompt")
            .arg(&request.prompt)
            .arg("--max-tokens")
            .arg(request.max_tokens.to_string())
            .arg("--backend")
            .arg(match self.mode {
                OnnxRuntimeMode::Cuda => "cuda",
                OnnxRuntimeMode::Rocm => "rocm",
                OnnxRuntimeMode::Cpu => "cpu",
            })
            .arg("--json")
            .output()
            .with_context(|| format!("failed to run ONNX Runtime runner {}", runner.display()))?;
        if !output.status.success() {
            bail!(
                "ONNX Runtime runner failed: {}",
                command_output_detail(&output)
            );
        }
        let value: Value = serde_json::from_slice(&output.stdout).with_context(|| {
            format!(
                "ONNX Runtime runner returned invalid JSON: {}",
                String::from_utf8_lossy(&output.stdout)
            )
        })?;
        let text = value
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("ONNX Runtime runner JSON missing string field 'text'"))?
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

impl GenerationBackend for OnnxRuntimeBackend {
    fn prepare(&self, manifest: &ModelManifest) -> Result<()> {
        eprintln!("Using {} backend", self.mode.display());
        Self::ensure_available_for_model(&self.store, manifest, self.mode)
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

fn discover_onnx_runtime(store: &ModelStore, mode: OnnxRuntimeMode) -> OnnxRuntimeDiscovery {
    let mut attempts = Vec::new();
    let env_name = match mode {
        OnnxRuntimeMode::Cuda => "WERK_ONNX_RUNTIME_CUDA",
        OnnxRuntimeMode::Rocm => "WERK_ONNX_RUNTIME_ROCM",
        OnnxRuntimeMode::Cpu => "WERK_ONNX_RUNTIME_CPU",
    };
    for (label, path) in [
        (
            env_name.to_string(),
            env::var_os(env_name).map(PathBuf::from),
        ),
        (
            "WERK_ONNX_RUNTIME".to_string(),
            env::var_os("WERK_ONNX_RUNTIME").map(PathBuf::from),
        ),
        (
            "managed cache".to_string(),
            Some(managed_runner_path(store, mode)),
        ),
        (
            "PATH: werk-onnx-runner".to_string(),
            find_in_path(runner_name()),
        ),
    ] {
        let Some(path) = path else {
            attempts.push(OnnxRuntimeAttempt {
                label,
                path: None,
                exists: false,
                usable: false,
                detail: "not set".to_string(),
            });
            continue;
        };
        let usable = runner_help_ok(&path);
        attempts.push(OnnxRuntimeAttempt {
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
            return OnnxRuntimeDiscovery {
                path: Some(path),
                source: label,
                attempts,
            };
        }
    }
    OnnxRuntimeDiscovery {
        path: None,
        source: "missing".to_string(),
        attempts,
    }
}

pub fn managed_runner_path(store: &ModelStore, mode: OnnxRuntimeMode) -> PathBuf {
    let name = match mode {
        OnnxRuntimeMode::Cuda => "onnxruntime-cuda",
        OnnxRuntimeMode::Rocm => "onnxruntime-rocm",
        OnnxRuntimeMode::Cpu => "onnxruntime-cpu",
    };
    store.home().join("backends").join(name).join(runner_name())
}

pub fn install_managed_onnx_runtime(store: &ModelStore, mode: OnnxRuntimeMode) -> Result<PathBuf> {
    let source =
        find_bundled_runner(mode).ok_or_else(|| anyhow!("{}", missing_bundle_message(mode)))?;
    let dest = managed_runner_path(store, mode);
    fs::create_dir_all(
        dest.parent()
            .ok_or_else(|| anyhow!("invalid managed ONNX Runtime path {}", dest.display()))?,
    )?;
    if source != dest {
        fs::copy(&source, &dest).with_context(|| {
            format!(
                "failed to copy ONNX Runtime runner from {} to {}",
                source.display(),
                dest.display()
            )
        })?;
    }
    make_executable(&dest)?;
    if !runner_help_ok(&dest) {
        bail!(
            "installed ONNX Runtime runner did not pass --help validation: {}",
            dest.display()
        );
    }
    Ok(dest)
}

fn find_bundled_runner(mode: OnnxRuntimeMode) -> Option<PathBuf> {
    bundled_runner_candidates(mode)
        .into_iter()
        .find(|path| runner_help_ok(path))
}

fn bundled_runner_candidates(mode: OnnxRuntimeMode) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = env::var_os(mode.bundle_env()).map(PathBuf::from) {
        candidates.push(path);
    }
    if let Some(path) = env::var_os("WERK_ONNX_RUNTIME_BUNDLE").map(PathBuf::from) {
        candidates.push(path);
    }

    let backend_dir = match mode {
        OnnxRuntimeMode::Cuda => "onnxruntime-cuda",
        OnnxRuntimeMode::Rocm => "onnxruntime-rocm",
        OnnxRuntimeMode::Cpu => "onnxruntime-cpu",
    };

    if let Ok(exe) = env::current_exe()
        && let Some(dir) = exe.parent()
    {
        candidates.push(dir.join("backends").join(backend_dir).join(runner_name()));
        candidates.push(dir.join(backend_dir).join(runner_name()));
    }

    if let Some(manifest_dir) = option_env!("CARGO_MANIFEST_DIR") {
        let root = PathBuf::from(manifest_dir);
        candidates.push(root.join("backends").join(backend_dir).join(runner_name()));
        candidates.push(
            root.join("dist")
                .join("backends")
                .join(backend_dir)
                .join(runner_name()),
        );
    }

    candidates
}

fn missing_bundle_message(mode: OnnxRuntimeMode) -> String {
    let mut message = format!(
        "No bundled {} runner found for provisioning.",
        mode.display()
    );
    message.push_str("\n\nTried:");
    for candidate in bundled_runner_candidates(mode) {
        message.push_str(&format!("\n- {}", candidate.display()));
    }
    message.push_str("\n\nFix:");
    message.push_str(&format!(
        "\n- set {}=/path/to/{}",
        mode.bundle_env(),
        runner_name()
    ));
    message.push_str("\n- or ship the runner under backends/<runtime>/ next to the werk binary");
    message.push_str("\n- or set WERK_ONNX_RUNTIME=/path/to/werk-onnx-runner");
    message
}

fn make_executable(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(path)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)?;
    }
    Ok(())
}

fn missing_message_from_discovery(
    mode: OnnxRuntimeMode,
    discovery: &OnnxRuntimeDiscovery,
) -> String {
    let mut message = format!("No {} runner found.\n\nTried:", mode.display());
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
    message.push_str("\n- set WERK_ONNX_RUNTIME=/path/to/werk-onnx-runner");
    message.push_str("\n- or install a managed ONNX Runtime runner artifact for Werk");
    message
}

fn concise_unavailable_reason(discovery: &OnnxRuntimeDiscovery) -> String {
    if discovery
        .attempts
        .iter()
        .any(|attempt| attempt.exists && !attempt.usable)
    {
        "runner validation failed".to_string()
    } else {
        "runner not installed or bundled".to_string()
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
        "werk-onnx-runner.exe"
    } else {
        "werk-onnx-runner"
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
