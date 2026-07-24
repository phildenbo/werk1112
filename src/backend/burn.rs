use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
use std::{
    env, fs,
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

#[cfg(feature = "burn-runtime")]
#[path = "burn_phi3.rs"]
mod burn_phi3;

use super::{
    BackendDoctorCheck, GenerateRequest, GenerateResponse, GenerateStream, GenerateStreamEvent,
    GenerationBackend,
};
use crate::{
    model_store::{ModelFormat, ModelManifest, ModelStore},
    openai::{ChatMessage, MessageContent, messages_to_prompt_for_model},
};

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
}

#[derive(Clone)]
pub struct BurnBackend {
    store: ModelStore,
    mode: BurnMode,
    cache: Arc<Mutex<Option<BurnPreparedModel>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnRuntimeStatus {
    pub mode: BurnMode,
    pub available: bool,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnProbeReport {
    pub mode: BurnMode,
    pub available: bool,
    pub architecture: String,
    pub reason: String,
    pub checks: Vec<BurnProbeCheck>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnProbeCheck {
    pub name: &'static str,
    pub ok: bool,
    pub detail: String,
}

struct BurnPreparedModel {
    architecture: String,
    tokenizer: Tokenizer,
    prompt_smoke: String,
    #[cfg(feature = "burn-runtime")]
    runtime: burn_phi3::BurnPhi3Runtime,
}

impl BurnBackend {
    pub fn new(store: ModelStore, mode: BurnMode) -> Self {
        Self {
            store,
            mode,
            cache: Arc::new(Mutex::new(None)),
        }
    }

    pub fn probe(store: &ModelStore, manifest: &ModelManifest, mode: BurnMode) -> Result<String> {
        let report = Self::probe_report(store, manifest, mode);
        if report.available {
            Ok(format!(
                "{} validated for {}",
                mode.display(),
                report.architecture
            ))
        } else {
            bail!("{}", report.reason)
        }
    }

    pub fn probe_report(
        store: &ModelStore,
        manifest: &ModelManifest,
        mode: BurnMode,
    ) -> BurnProbeReport {
        let mut checks = Vec::new();
        let mut architecture = manifest
            .architecture
            .clone()
            .unwrap_or_else(|| "unknown".to_string());

        push_check(
            &mut checks,
            "format",
            manifest.format == ModelFormat::SafeTensors,
            "SafeTensors model directory required",
        );
        if manifest.format != ModelFormat::SafeTensors {
            return rejected(
                mode,
                architecture,
                checks,
                "Burn supports SafeTensors models only",
            );
        }

        match runtime_status(mode) {
            Ok(status) => push_check(&mut checks, "device", true, status.detail),
            Err(err) => {
                let reason = err.to_string();
                push_check(&mut checks, "device", false, reason.clone());
                return rejected(mode, architecture, checks, reason);
            }
        }

        let config = match read_config(store, manifest) {
            Ok(config) => {
                push_check(&mut checks, "config", true, "config.json parsed");
                config
            }
            Err(err) => {
                let reason = err.to_string();
                push_check(&mut checks, "config", false, reason.clone());
                return rejected(mode, architecture, checks, reason);
            }
        };
        if architecture == "unknown"
            && let Some(model_type) = config.get("model_type").and_then(Value::as_str)
        {
            architecture = model_type.to_string();
        }

        let tokenizer = match load_tokenizer(store, manifest) {
            Ok(tokenizer) => {
                push_check(&mut checks, "tokenizer", true, "tokenizer.json loaded");
                tokenizer
            }
            Err(err) => {
                let reason = err.to_string();
                push_check(&mut checks, "tokenizer", false, reason.clone());
                return rejected(mode, architecture, checks, reason);
            }
        };

        match prompt_smoke(manifest, &tokenizer) {
            Ok(_) => push_check(
                &mut checks,
                "chat-prompt",
                true,
                "Werk chat prompt tokenizes",
            ),
            Err(err) => {
                let reason = err.to_string();
                push_check(&mut checks, "chat-prompt", false, reason.clone());
                return rejected(mode, architecture, checks, reason);
            }
        }

        match safetensor_files(store, manifest) {
            Ok(paths) if !paths.is_empty() => push_check(
                &mut checks,
                "weights",
                true,
                format!("{} safetensors file(s) discovered", paths.len()),
            ),
            Ok(_) => {
                let reason = "SafeTensors model has no .safetensors weight files".to_string();
                push_check(&mut checks, "weights", false, reason.clone());
                return rejected(mode, architecture, checks, reason);
            }
            Err(err) => {
                let reason = err.to_string();
                push_check(&mut checks, "weights", false, reason.clone());
                return rejected(mode, architecture, checks, reason);
            }
        }

        let model_type = config
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or(architecture.as_str());
        if !model_type.eq_ignore_ascii_case(architecture.as_str()) && architecture == "unknown" {
            push_check(
                &mut checks,
                "architecture",
                false,
                "config.json has no usable model_type",
            );
            return rejected(
                mode,
                architecture,
                checks,
                "safetensors manifest has no architecture and config.json has no model_type",
            );
        }

        let architecture_supported = burn_architecture_enabled(architecture.as_str());
        push_check(
            &mut checks,
            "architecture",
            architecture_supported,
            if architecture_supported {
                format!("{} has a validated Burn implementation", architecture)
            } else {
                format!(
                    "{} has no validated Burn generation implementation yet",
                    architecture
                )
            },
        );
        if !architecture_supported {
            return rejected(
                mode,
                architecture.clone(),
                checks,
                format!(
                    "Burn {} probe failed quality/performance gate for architecture '{}'",
                    mode.accelerator_label(),
                    architecture
                ),
            );
        }

        match burn_architecture_probe(store, manifest, architecture.as_str()) {
            Ok(detail) => {
                push_check(&mut checks, "prepare", true, detail);
                BurnProbeReport {
                    mode,
                    available: true,
                    architecture,
                    reason: format!(
                        "{} passed capability probe and enablement gate",
                        mode.display()
                    ),
                    checks,
                }
            }
            Err(err) => {
                let reason = err.to_string();
                push_check(&mut checks, "prepare", false, reason.clone());
                rejected(mode, architecture, checks, reason)
            }
        }
    }

    pub fn runtime_status(mode: BurnMode) -> BurnRuntimeStatus {
        match runtime_status(mode) {
            Ok(status) => BurnRuntimeStatus {
                mode,
                available: true,
                detail: status.detail,
            },
            Err(err) => BurnRuntimeStatus {
                mode,
                available: false,
                detail: err.to_string(),
            },
        }
    }

    pub fn missing_message(store: &ModelStore, manifest: &ModelManifest, mode: BurnMode) -> String {
        let report = Self::probe_report(store, manifest, mode);
        let fix = match mode {
            BurnMode::Cuda => {
                "\n\nFix: install native CUDA runtime/toolkit and NCCL through your system package manager, or set NCCL_HOME/CUDA_HOME to native installations. On WSL, ensure /usr/lib/wsl/lib is visible and remove CUDA stubs from LD_LIBRARY_PATH. Run `werk backend doctor --debug` for details."
            }
            BurnMode::Cpu => "\n\nFix: rebuild with Burn CPU support or use --backend candle.",
        };
        format!(
            "{} requested but the Burn capability probe rejected model '{}'.\n\nReason: {}\n\nBurn is only selected when architecture, weights, tokenizer, generation smoke test, and quality/speed gates pass. Use --backend candle to force the compatibility runtime.{}",
            mode.display(),
            manifest.id,
            report.reason,
            fix
        )
    }

    pub fn unavailable_reason(
        store: &ModelStore,
        manifest: &ModelManifest,
        mode: BurnMode,
    ) -> String {
        Self::probe_report(store, manifest, mode).reason
    }

    fn prepare_model(&self, manifest: &ModelManifest) -> Result<BurnPreparedModel> {
        Self::probe(&self.store, manifest, self.mode).map_err(|err| anyhow!("{err}"))?;

        let tokenizer = load_tokenizer(&self.store, manifest)?;
        let prompt_smoke = prompt_smoke(manifest, &tokenizer)?;
        #[cfg(feature = "burn-runtime")]
        let runtime =
            burn_phi3::BurnPhi3Runtime::load(&self.store, manifest, tokenizer.clone(), self.mode)?;
        Ok(BurnPreparedModel {
            architecture: manifest
                .architecture
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            tokenizer,
            prompt_smoke,
            #[cfg(feature = "burn-runtime")]
            runtime,
        })
    }

    fn ensure_model(&self, manifest: &ModelManifest) -> Result<()> {
        if self
            .cache
            .lock()
            .map_err(|_| anyhow!("Burn model cache mutex poisoned"))?
            .is_some()
        {
            return Ok(());
        }

        let model = self.prepare_model(manifest)?;
        *self
            .cache
            .lock()
            .map_err(|_| anyhow!("Burn model cache mutex poisoned"))? = Some(model);
        Ok(())
    }

    fn with_model<T>(
        &self,
        manifest: &ModelManifest,
        f: impl FnOnce(&mut BurnPreparedModel) -> Result<T>,
    ) -> Result<T> {
        self.ensure_model(manifest)?;
        let mut guard = self
            .cache
            .lock()
            .map_err(|_| anyhow!("Burn model cache mutex poisoned"))?;
        let model = guard
            .as_mut()
            .ok_or_else(|| anyhow!("Burn model cache was unexpectedly empty"))?;
        f(model)
    }

    fn generate_inner(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
        on_token: Option<Box<dyn FnMut(String) + Send>>,
    ) -> Result<GenerateResponse> {
        self.with_model(manifest, |model| {
            let _ = (&model.architecture, &model.tokenizer, &model.prompt_smoke);
            #[cfg(feature = "burn-runtime")]
            {
                model.runtime.generate(request, on_token)
            }
            #[cfg(not(feature = "burn-runtime"))]
            {
                let _ = (request, on_token);
                bail!(
                    "{} has no enabled text generation implementation for '{}' yet; Candle remains the compatibility fallback",
                    self.mode.display(),
                    manifest.id
                )
            }
        })
    }
}

pub fn burn_doctor_checks() -> Vec<BackendDoctorCheck> {
    vec![
        burn_cuda_feature_check(),
        burn_command_check(
            "nvidia-smi",
            &[],
            "required to verify NVIDIA driver visibility for Burn CUDA",
        ),
        cuda_visible_devices_check(),
        libcuda_runtime_path_check(),
        burn_command_check(
            "nvcc",
            &["--version"],
            "required to compile native CUDA Burn dependencies from source",
        ),
        libcudart_library_check(),
        nccl_library_check(),
        burn_runtime_check(BurnMode::Cuda),
        burn_runtime_check(BurnMode::Cpu),
    ]
}

fn burn_cuda_feature_check() -> BackendDoctorCheck {
    if cfg!(feature = "burn-cuda") {
        BackendDoctorCheck {
            name: "Burn CUDA feature".to_string(),
            ok: true,
            detail: "compiled with burn-cuda".to_string(),
        }
    } else {
        BackendDoctorCheck {
            name: "Burn CUDA feature".to_string(),
            ok: false,
            detail: "not compiled. Rebuild with: cargo install --path . --locked --force --features burn-cuda".to_string(),
        }
    }
}

fn cuda_visible_devices_check() -> BackendDoctorCheck {
    match env::var("CUDA_VISIBLE_DEVICES") {
        Ok(value) if value.trim().is_empty() || value.trim() == "-1" => BackendDoctorCheck {
            name: "CUDA_VISIBLE_DEVICES".to_string(),
            ok: false,
            detail: format!("set to {value:?}, which hides CUDA devices from Burn"),
        },
        Ok(value) => BackendDoctorCheck {
            name: "CUDA_VISIBLE_DEVICES".to_string(),
            ok: true,
            detail: format!("set to {value:?}"),
        },
        Err(env::VarError::NotPresent) => BackendDoctorCheck {
            name: "CUDA_VISIBLE_DEVICES".to_string(),
            ok: true,
            detail: "not set".to_string(),
        },
        Err(err) => BackendDoctorCheck {
            name: "CUDA_VISIBLE_DEVICES".to_string(),
            ok: false,
            detail: err.to_string(),
        },
    }
}

fn libcuda_runtime_path_check() -> BackendDoctorCheck {
    if let Some(stub) = cuda_stub_in_runtime_path() {
        return BackendDoctorCheck {
            name: "libcuda runtime".to_string(),
            ok: false,
            detail: format!(
                "CUDA stub path appears in LD_LIBRARY_PATH: {}. Remove stubs from runtime path; on WSL put /usr/lib/wsl/lib first.",
                stub.display()
            ),
        };
    }
    if let Some(path) = libcuda_from_ldconfig() {
        return BackendDoctorCheck {
            name: "libcuda runtime".to_string(),
            ok: true,
            detail: path,
        };
    }
    for path in [
        "/usr/lib/wsl/lib/libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    ] {
        let path = PathBuf::from(path);
        if path.exists() {
            return BackendDoctorCheck {
                name: "libcuda runtime".to_string(),
                ok: true,
                detail: path.display().to_string(),
            };
        }
    }
    BackendDoctorCheck {
        name: "libcuda runtime".to_string(),
        ok: false,
        detail: "libcuda.so.1 was not found by ldconfig or common WSL/Linux locations".to_string(),
    }
}

fn cuda_stub_in_runtime_path() -> Option<PathBuf> {
    env::var_os("LD_LIBRARY_PATH").and_then(|paths| {
        env::split_paths(&paths).find(|path| {
            path.components()
                .any(|component| component.as_os_str() == "stubs")
        })
    })
}

fn libcuda_from_ldconfig() -> Option<String> {
    let output = Command::new("ldconfig").arg("-p").output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find(|line| line.contains("libcuda.so.1"))
        .map(|line| format!("ldconfig: {}", line.trim()))
}

fn libcudart_library_check() -> BackendDoctorCheck {
    if let Some(path) = libcudart_from_env() {
        return BackendDoctorCheck {
            name: "libcudart".to_string(),
            ok: true,
            detail: path,
        };
    }
    if let Some(path) = library_from_ldconfig("libcudart.so") {
        return BackendDoctorCheck {
            name: "libcudart".to_string(),
            ok: true,
            detail: path,
        };
    }
    if let Some(path) = libcudart_from_common_paths() {
        return BackendDoctorCheck {
            name: "libcudart".to_string(),
            ok: true,
            detail: path,
        };
    }
    BackendDoctorCheck {
        name: "libcudart".to_string(),
        ok: false,
        detail: "not found. Fix: install the native CUDA toolkit/runtime and expose libcudart.so through the system linker path or CUDA_HOME.".to_string(),
    }
}

fn libcudart_from_env() -> Option<String> {
    for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
        let Some(root) = env::var_os(var).map(PathBuf::from) else {
            continue;
        };
        for subdir in ["lib64", "lib"] {
            if let Some(path) =
                existing_library_path(root.join(subdir), &["libcudart.so", "libcudart.so.12"])
            {
                return Some(format!("{var}: {}", path.display()));
            }
        }
    }
    if let Some(paths) = env::var_os("LD_LIBRARY_PATH") {
        for dir in env::split_paths(&paths) {
            if let Some(path) = existing_library_path(dir, &["libcudart.so", "libcudart.so.12"]) {
                return Some(format!("LD_LIBRARY_PATH: {}", path.display()));
            }
        }
    }
    None
}

fn libcudart_from_common_paths() -> Option<String> {
    [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-13.0/lib64",
        "/usr/local/cuda-12.8/lib64",
        "/usr/local/cuda-12.6/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ]
    .into_iter()
    .map(PathBuf::from)
    .find_map(|dir| existing_library_path(dir, &["libcudart.so", "libcudart.so.12"]))
    .map(|path| path.display().to_string())
}

fn burn_runtime_check(mode: BurnMode) -> BackendDoctorCheck {
    let status = BurnBackend::runtime_status(mode);
    BackendDoctorCheck {
        name: format!("{} tensor smoke", mode.display()),
        ok: status.available,
        detail: status.detail,
    }
}

fn burn_command_check(command: &str, args: &[&str], detail: &str) -> BackendDoctorCheck {
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
            detail: String::from_utf8_lossy(&output.stderr)
                .lines()
                .next()
                .unwrap_or(detail)
                .to_string(),
        },
        Err(err) => BackendDoctorCheck {
            name: command.to_string(),
            ok: false,
            detail: format!("{detail}: {err}"),
        },
    }
}

fn nccl_library_check() -> BackendDoctorCheck {
    if let Some(path) = nccl_from_env() {
        return BackendDoctorCheck {
            name: "NCCL library".to_string(),
            ok: true,
            detail: path,
        };
    }
    if let Some(path) = nccl_from_ldconfig() {
        return BackendDoctorCheck {
            name: "NCCL library".to_string(),
            ok: true,
            detail: path,
        };
    }
    if let Some(path) = nccl_from_common_paths() {
        return BackendDoctorCheck {
            name: "NCCL library".to_string(),
            ok: true,
            detail: path,
        };
    }
    BackendDoctorCheck {
        name: "NCCL library".to_string(),
        ok: false,
        detail: "not found. Fix: install NCCL through your system package manager or set NCCL_HOME to a native NCCL installation containing lib/libnccl.so.".to_string(),
    }
}

fn nccl_from_env() -> Option<String> {
    for var in ["NCCL_HOME", "NCCL_ROOT"] {
        let Some(root) = env::var_os(var).map(PathBuf::from) else {
            continue;
        };
        for subdir in ["lib", "lib64"] {
            if let Some(path) = existing_nccl_path(root.join(subdir)) {
                return Some(format!("{var}: {}", path.display()));
            }
        }
    }
    if let Some(paths) = env::var_os("LD_LIBRARY_PATH") {
        for dir in env::split_paths(&paths) {
            if let Some(path) = existing_nccl_path(dir) {
                return Some(format!("LD_LIBRARY_PATH: {}", path.display()));
            }
        }
    }
    None
}

fn nccl_from_ldconfig() -> Option<String> {
    library_from_ldconfig("libnccl.so")
}

fn library_from_ldconfig(library: &str) -> Option<String> {
    let output = Command::new("ldconfig").arg("-p").output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find(|line| line.contains(library))
        .map(|line| format!("ldconfig: {}", line.trim()))
}

fn nccl_from_common_paths() -> Option<String> {
    [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-13.0/lib64",
        "/usr/local/cuda-12.8/lib64",
        "/usr/local/cuda-12.6/lib64",
    ]
    .into_iter()
    .map(PathBuf::from)
    .find_map(existing_nccl_path)
    .map(|path| path.display().to_string())
}

fn existing_nccl_path(dir: PathBuf) -> Option<PathBuf> {
    existing_library_path(dir, &["libnccl.so", "libnccl.so.2"])
}

fn existing_library_path(dir: PathBuf, names: &[&str]) -> Option<PathBuf> {
    for name in names {
        let path = dir.join(name);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

impl BurnMode {
    fn accelerator_label(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Cpu => "CPU",
        }
    }
}

impl GenerationBackend for BurnBackend {
    fn prepare(&self, manifest: &ModelManifest) -> Result<()> {
        self.ensure_model(manifest)?;
        eprintln!("Using {} backend", self.mode.display());
        Ok(())
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
        let (tx, rx) = mpsc::channel(4);
        tokio::task::spawn_blocking(move || {
            let callback_tx = tx.clone();
            let result = backend.generate_inner(
                &manifest,
                request,
                Some(Box::new(move |text| {
                    let _ = callback_tx.blocking_send(Ok(GenerateStreamEvent::TextChunk(text)));
                })),
            );
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
                    let _ = tx.blocking_send(Err(err.to_string()));
                }
            }
        });
        Box::pin(ReceiverStream::new(rx))
    }
}

fn rejected(
    mode: BurnMode,
    architecture: String,
    checks: Vec<BurnProbeCheck>,
    reason: impl Into<String>,
) -> BurnProbeReport {
    BurnProbeReport {
        mode,
        available: false,
        architecture,
        reason: reason.into(),
        checks,
    }
}

fn push_check(
    checks: &mut Vec<BurnProbeCheck>,
    name: &'static str,
    ok: bool,
    detail: impl Into<String>,
) {
    checks.push(BurnProbeCheck {
        name,
        ok,
        detail: detail.into(),
    });
}

struct RuntimeOk {
    detail: String,
}

fn runtime_status(mode: BurnMode) -> Result<RuntimeOk> {
    match mode {
        BurnMode::Cuda => burn_cuda_status(),
        BurnMode::Cpu => burn_cpu_status(),
    }
}

#[cfg(feature = "burn-cuda")]
fn burn_cuda_status() -> Result<RuntimeOk> {
    use burn::{backend::Cuda, tensor::Tensor};

    burn_cuda_prerequisites()?;
    match catch_unwind_without_panic_output(|| -> Result<RuntimeOk> {
        let device = Default::default();
        let data = Tensor::<Cuda, 1>::from_floats([1.0_f32], &device)
            .try_into_data()
            .context("Burn CUDA tensor smoke test failed")?;
        let value = data
            .to_vec::<f32>()
            .context("Burn CUDA tensor data conversion failed")?;
        if value.first().copied() != Some(1.0) {
            bail!("Burn CUDA tensor smoke test returned unexpected data");
        }
        Ok(RuntimeOk {
            detail: "Burn CUDA tensor smoke test passed".to_string(),
        })
    }) {
        Ok(result) => result,
        Err(_) => bail!("Burn CUDA tensor smoke test panicked during CUDA initialization"),
    }
}

#[cfg(feature = "burn-cuda")]
fn burn_cuda_prerequisites() -> Result<()> {
    let mut errors = Vec::new();
    if let Err(err) = nvidia_smi_device_visible() {
        errors.push(err.to_string());
    }
    if let Err(err) = libcuda_runtime_visible() {
        errors.push(err.to_string());
    }
    if let Err(err) = libcudart_library_visible() {
        errors.push(err.to_string());
    }
    if let Err(err) = cuda_driver_device_visible() {
        errors.push(err.to_string());
    }
    if let Err(err) = nccl_library_visible() {
        errors.push(err.to_string());
    }
    if errors.is_empty() {
        Ok(())
    } else {
        bail!("{}", errors.join("; "))
    }
}

#[cfg(feature = "burn-cuda")]
fn libcuda_runtime_visible() -> Result<()> {
    if let Some(stub) = cuda_stub_in_runtime_path() {
        bail!(
            "CUDA stub path is present in LD_LIBRARY_PATH for Burn CUDA: {}. Remove CUDA stubs from runtime path and put /usr/lib/wsl/lib first on WSL",
            stub.display()
        );
    }
    if libcuda_from_ldconfig().is_some()
        || PathBuf::from("/usr/lib/wsl/lib/libcuda.so.1").exists()
        || PathBuf::from("/usr/lib/x86_64-linux-gnu/libcuda.so.1").exists()
    {
        Ok(())
    } else {
        bail!("libcuda.so.1 is not visible to the dynamic linker for Burn CUDA")
    }
}

#[cfg(feature = "burn-cuda")]
fn libcudart_library_visible() -> Result<()> {
    if libcudart_from_env().is_some()
        || library_from_ldconfig("libcudart.so").is_some()
        || libcudart_from_common_paths().is_some()
    {
        Ok(())
    } else {
        bail!(
            "libcudart.so is not visible to Burn CUDA; install the native CUDA toolkit/runtime or set CUDA_HOME"
        )
    }
}

#[cfg(feature = "burn-cuda")]
fn nvidia_smi_device_visible() -> Result<()> {
    match Command::new("nvidia-smi").arg("-L").output() {
        Ok(output) if output.status.success() => Ok(()),
        Ok(output) => {
            let detail = String::from_utf8_lossy(&output.stderr)
                .lines()
                .next()
                .unwrap_or("no CUDA-capable device is visible to nvidia-smi")
                .to_string();
            bail!("CUDA device is not visible for Burn CUDA: {detail}")
        }
        Err(err) => bail!("CUDA device is not visible for Burn CUDA: nvidia-smi failed: {err}"),
    }
}

#[cfg(feature = "burn-cuda")]
fn cuda_driver_device_visible() -> Result<()> {
    let count = cudarc::driver::CudaContext::device_count()
        .map_err(|err| anyhow!("CUDA driver API cannot enumerate devices for Burn CUDA: {err}"))?;
    if count > 0 {
        Ok(())
    } else {
        bail!("CUDA driver API reports zero devices for Burn CUDA")
    }
}

#[cfg(feature = "burn-cuda")]
fn nccl_library_visible() -> Result<()> {
    if nccl_from_env().is_some()
        || nccl_from_ldconfig().is_some()
        || nccl_from_common_paths().is_some()
    {
        Ok(())
    } else {
        bail!(
            "NCCL library is not visible for Burn CUDA; install NCCL through your system package manager or set NCCL_HOME to a native NCCL installation"
        )
    }
}

#[cfg(feature = "burn-cuda")]
fn catch_unwind_without_panic_output<T>(
    f: impl FnOnce() -> T + std::panic::UnwindSafe,
) -> std::thread::Result<T> {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(f);
    std::panic::set_hook(previous);
    result
}

#[cfg(not(feature = "burn-cuda"))]
fn burn_cuda_status() -> Result<RuntimeOk> {
    bail!(
        "This Werk binary was built without Burn CUDA support. Rebuild with: cargo install --path . --locked --force --features burn-cuda. Burn CUDA requires native CUDA and NCCL system libraries."
    )
}

#[cfg(feature = "burn-cpu")]
fn burn_cpu_status() -> Result<RuntimeOk> {
    use burn::{backend::Flex, tensor::Tensor};

    let device = Default::default();
    let data = Tensor::<Flex, 1>::from_floats([1.0_f32], &device)
        .try_into_data()
        .context("Burn CPU tensor smoke test failed")?;
    let value = data
        .to_vec::<f32>()
        .context("Burn CPU tensor data conversion failed")?;
    if value.first().copied() != Some(1.0) {
        bail!("Burn CPU tensor smoke test returned unexpected data");
    }
    Ok(RuntimeOk {
        detail: "Burn CPU tensor smoke test passed".to_string(),
    })
}

#[cfg(not(feature = "burn-cpu"))]
fn burn_cpu_status() -> Result<RuntimeOk> {
    bail!(
        "This Werk binary was built without Burn CPU support. Rebuild with: cargo install --path . --locked --force --features burn-cpu"
    )
}

fn read_config(store: &ModelStore, manifest: &ModelManifest) -> Result<Value> {
    let config_path = manifest
        .config_path
        .as_deref()
        .context("safetensors model requires config.json")?;
    let path = store.absolute_model_file(manifest, config_path);
    let data =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&data).with_context(|| format!("failed to parse {}", path.display()))
}

fn load_tokenizer(store: &ModelStore, manifest: &ModelManifest) -> Result<Tokenizer> {
    let tokenizer_path = manifest
        .tokenizer_path
        .as_deref()
        .context("safetensors model requires tokenizer.json")?;
    let path = store.absolute_model_file(manifest, tokenizer_path);
    Tokenizer::from_file(&path)
        .map_err(|err| anyhow!("failed to load tokenizer {}: {err}", path.display()))
}

fn prompt_smoke(manifest: &ModelManifest, tokenizer: &Tokenizer) -> Result<String> {
    let prompt = messages_to_prompt_for_model(
        manifest,
        &[ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text("Say ok.".to_string())),
            name: None,
        }],
    )
    .prompt;
    let token_count = tokenizer
        .encode(prompt.clone(), true)
        .map_err(|err| anyhow!("Burn tokenizer smoke test failed: {err}"))?
        .len();
    if token_count == 0 {
        bail!("Burn tokenizer smoke test produced zero tokens");
    }
    Ok(prompt)
}

fn safetensor_files(store: &ModelStore, manifest: &ModelManifest) -> Result<Vec<PathBuf>> {
    let mut paths = manifest
        .files
        .iter()
        .filter(|file| file.path.ends_with(".safetensors"))
        .map(|file| store.absolute_model_file(manifest, &file.path))
        .collect::<Vec<_>>();
    paths.sort();
    for path in &paths {
        if !path.is_file() {
            bail!("missing safetensors weight file {}", path.display());
        }
    }
    Ok(paths)
}

fn burn_architecture_enabled(architecture: &str) -> bool {
    architecture.eq_ignore_ascii_case("phi3")
}

fn burn_architecture_probe(
    store: &ModelStore,
    manifest: &ModelManifest,
    architecture: &str,
) -> Result<String> {
    match architecture {
        architecture if architecture.eq_ignore_ascii_case("phi3") => {
            burn_phi3_probe(store, manifest)
        }
        other => bail!("{other} has no validated Burn generation implementation yet"),
    }
}

#[cfg(feature = "burn-runtime")]
fn burn_phi3_probe(store: &ModelStore, manifest: &ModelManifest) -> Result<String> {
    burn_phi3::probe_phi3_files(store, manifest)
}

#[cfg(not(feature = "burn-runtime"))]
fn burn_phi3_probe(_store: &ModelStore, _manifest: &ModelManifest) -> Result<String> {
    bail!("This Werk binary was built without Burn runtime support")
}

#[cfg(test)]
mod tests {
    use super::burn_architecture_enabled;

    #[test]
    fn only_phi3_is_enabled_for_burn_generation() {
        assert!(burn_architecture_enabled("phi3"));
        for architecture in ["qwen2", "gemma", "gemma2", "mistral", "llama"] {
            assert!(!burn_architecture_enabled(architecture));
        }
    }
}
