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

const ONNX_GENAI_PYTHON_SCRIPT: &str = r#"
import argparse
import json
import os
import time

import onnxruntime_genai as og

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--prompt", required=True)
parser.add_argument("--max-tokens", required=True, type=int)
parser.add_argument("--temperature", type=float)
parser.add_argument("--top-p", type=float)
parser.add_argument("--seed", type=int)
parser.add_argument("--stop-json", default="[]")
args = parser.parse_args()

started = time.monotonic()
model = og.Model(args.model)
loaded_at = time.monotonic()
tokenizer = og.Tokenizer(model)
input_ids = tokenizer.encode(args.prompt)
prompt_tokens = len(input_ids)

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}

def int_value(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except Exception:
            return None
    return None

def first_config_int(configs, paths):
    for config in configs:
        for path in paths:
            value = config
            for key in path:
                if not isinstance(value, dict) or key not in value:
                    value = None
                    break
                value = value[key]
            parsed = int_value(value)
            if parsed and parsed > 0:
                return parsed
    return None

genai_config = load_json(os.path.join(args.model, "genai_config.json"))
hf_config = load_json(os.path.join(args.model, "config.json"))
configs = [genai_config, hf_config]
context_length = first_config_int(configs, [
    ("model", "context_length"),
    ("search", "max_length"),
    ("max_position_embeddings",),
    ("n_positions",),
    ("seq_length",),
    ("context_length",),
])

requested_max_new_tokens = max(1, args.max_tokens)
if context_length:
    available_new_tokens = max(1, context_length - prompt_tokens)
    max_new_tokens = min(requested_max_new_tokens, available_new_tokens)
else:
    max_new_tokens = requested_max_new_tokens
max_length = prompt_tokens + max_new_tokens

def values_as_ints(value):
    if isinstance(value, list):
        return [parsed for parsed in (int_value(item) for item in value) if parsed is not None]
    parsed = int_value(value)
    return [] if parsed is None else [parsed]

eos_token_ids = set()
for config in configs:
    if isinstance(config, dict):
        eos_token_ids.update(values_as_ints(config.get("eos_token_id")))
        model_config = config.get("model")
        if isinstance(model_config, dict):
            eos_token_ids.update(values_as_ints(model_config.get("eos_token_id")))
try:
    eos_token_ids.update(values_as_ints(tokenizer.eos_token_ids))
except Exception:
    pass

try:
    stops = json.loads(args.stop_json)
except Exception:
    stops = []
default_stops = [
    "<|end|>",
    "<|endoftext|>",
    "<|im_end|>",
    "<|eot_id|>",
    "<|eom_id|>",
    "</s>",
]
stop_strings = []
for stop in list(stops) + default_stops:
    if isinstance(stop, str) and stop and stop not in stop_strings:
        stop_strings.append(stop)

params = og.GeneratorParams(model)
search_options = {
    "max_length": max_length,
}
if args.temperature is not None:
    search_options["temperature"] = args.temperature
if args.top_p is not None:
    search_options["top_p"] = args.top_p
if args.seed is not None:
    search_options["random_seed"] = args.seed
params.set_search_options(**search_options)

generator = og.Generator(model, params)
generator.append_tokens(input_ids)
stream = tokenizer.create_stream()
generation_started = time.monotonic()

text = ""
completion_tokens = 0
finish_reason = "generator_done"
first_token_at = None

while completion_tokens < max_new_tokens:
    if generator.is_done():
        finish_reason = "generator_done"
        break

    generator.generate_next_token()
    next_tokens = generator.get_next_tokens()
    if len(next_tokens) == 0:
        finish_reason = "generator_done" if generator.is_done() else "eos"
        break

    token_id = int(next_tokens[0])
    if first_token_at is None:
        first_token_at = time.monotonic()
    completion_tokens += 1

    piece = stream.decode(token_id)
    candidate = text + piece

    if token_id in eos_token_ids:
        text = candidate
        finish_reason = "eos"
        break

    stop_match = None
    for stop in stop_strings:
        index = candidate.find(stop)
        if index >= 0:
            if stop_match is None or index < stop_match:
                stop_match = index
    if stop_match is not None:
        text = candidate[:stop_match]
        finish_reason = "stop_sequence"
        break

    text = candidate
else:
    if completion_tokens >= max_new_tokens:
        finish_reason = "max_new_tokens"

for stop in stop_strings:
    index = text.find(stop)
    if index >= 0:
        text = text[:index]
        if finish_reason not in ("eos", "stop_sequence"):
            finish_reason = "stop_sequence"
            break

print(json.dumps({
    "text": text,
    "prompt_tokens": prompt_tokens,
    "completion_tokens": completion_tokens,
    "finish_reason": finish_reason,
    "stop_reason": finish_reason,
    "requested_max_new_tokens": requested_max_new_tokens,
    "max_new_tokens": max_new_tokens,
    "max_length": max_length,
    "context_length": context_length,
    "load_seconds": loaded_at - started,
    "prompt_seconds": None,
    "first_token_seconds": 0.0 if first_token_at is None else first_token_at - generation_started,
    "decode_seconds": time.monotonic() - generation_started,
}))
"#;

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
        if let Some(path) = discovery.path.as_ref() {
            return Ok(format!("{} runner {}", mode.display(), path.display()));
        }
        if mode == OnnxRuntimeMode::Cpu
            && let Some(runtime) = discover_onnx_genai_python()
        {
            return Ok(format!(
                "{} via Python onnxruntime-genai {}",
                mode.display(),
                runtime.python.display()
            ));
        }
        Err(anyhow!(
            "{}",
            missing_message_from_discovery(mode, &discovery)
        ))
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
            if mode == OnnxRuntimeMode::Cpu
                && onnx_genai_model_dir(store, manifest).is_some()
                && discover_onnx_genai_python().is_some()
            {
                if options.verbose {
                    eprintln!(
                        "Selected runtime: {} via Python onnxruntime-genai",
                        mode.display()
                    );
                    eprintln!("Runtime status: ready");
                }
                return Ok(());
            }
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
        let model_path = self.model_path(manifest)?;
        if let Ok(runner) = self.runner() {
            return self.generate_with_runner(
                manifest,
                request,
                total_started,
                &runner,
                &model_path,
            );
        }

        if self.mode == OnnxRuntimeMode::Cpu
            && let Some(model_dir) = onnx_genai_model_dir(&self.store, manifest)
            && let Some(runtime) = discover_onnx_genai_python()
        {
            return self.generate_with_python_genai(
                request,
                total_started,
                &runtime.python,
                &model_dir,
            );
        }

        bail!("{}", Self::missing_message(&self.store, self.mode));
    }

    fn generate_with_runner(
        &self,
        _manifest: &ModelManifest,
        request: GenerateRequest,
        total_started: Instant,
        runner: &Path,
        model_path: &Path,
    ) -> Result<GenerateResponse> {
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
        let prompt_seconds = value
            .get("prompt_seconds")
            .and_then(Value::as_f64)
            .unwrap_or(f64::NAN);
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
                prompt_seconds,
                decode_seconds: elapsed,
                total_seconds: total_started.elapsed().as_secs_f64(),
            },
            backend_diagnostics: Vec::new(),
        })
    }

    fn generate_with_python_genai(
        &self,
        request: GenerateRequest,
        total_started: Instant,
        python: &Path,
        model_dir: &Path,
    ) -> Result<GenerateResponse> {
        if request.verbose {
            eprintln!("Starting generation...");
        }
        if request.debug {
            eprintln!(
                "selected backend: {} via Python onnxruntime-genai",
                self.mode.label()
            );
            eprintln!("Python: {}", python.display());
            eprintln!("ONNX GenAI model: {}", model_dir.display());
        }
        let stop_json = serde_json::to_string(&request.stop)?;
        let started = Instant::now();
        let mut command = Command::new(python);
        command
            .arg("-c")
            .arg(ONNX_GENAI_PYTHON_SCRIPT)
            .arg("--model")
            .arg(model_dir)
            .arg("--prompt")
            .arg(&request.prompt)
            .arg("--max-tokens")
            .arg(request.max_tokens.to_string())
            .arg("--stop-json")
            .arg(stop_json);
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
            .with_context(|| format!("failed to run Python ONNX GenAI {}", python.display()))?;
        if !output.status.success() {
            bail!(
                "Python ONNX GenAI runner failed: {}",
                command_output_detail(&output)
            );
        }

        let value: Value = serde_json::from_slice(&output.stdout).with_context(|| {
            format!(
                "Python ONNX GenAI runner returned invalid JSON: {}",
                String::from_utf8_lossy(&output.stdout)
            )
        })?;
        let text = value
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("Python ONNX GenAI JSON missing string field 'text'"))?
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
        let decode_seconds = value
            .get("decode_seconds")
            .and_then(Value::as_f64)
            .unwrap_or_else(|| started.elapsed().as_secs_f64());
        let first_token_seconds = value
            .get("first_token_seconds")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let load_seconds = value
            .get("load_seconds")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let prompt_seconds = value
            .get("prompt_seconds")
            .and_then(Value::as_f64)
            .unwrap_or(f64::NAN);
        let mut backend_diagnostics = Vec::new();
        for (label, key) in [
            ("stop reason", "stop_reason"),
            ("requested max new tokens", "requested_max_new_tokens"),
            ("effective max new tokens", "max_new_tokens"),
            ("effective max length", "max_length"),
            ("context length", "context_length"),
        ] {
            if let Some(value) = value.get(key) {
                match value {
                    Value::String(text) => backend_diagnostics.push(format!("{label}: {text}")),
                    Value::Number(number) => backend_diagnostics.push(format!("{label}: {number}")),
                    _ => {}
                }
            }
        }
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
                load_seconds,
                warmup_seconds: 0.0,
                first_token_seconds,
                prompt_seconds,
                decode_seconds,
                total_seconds: total_started.elapsed().as_secs_f64(),
            },
            backend_diagnostics,
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
    message.push_str(
        "\n- or install Python ONNX GenAI support with `python3 -m pip install onnxruntime-genai`",
    );
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

#[derive(Debug, Clone)]
struct OnnxGenaiPythonRuntime {
    python: PathBuf,
}

fn discover_onnx_genai_python() -> Option<OnnxGenaiPythonRuntime> {
    let mut candidates = Vec::<PathBuf>::new();
    for env_name in ["WERK_ONNX_GENAI_PYTHON", "WERK_ONNX_RUNTIME_PYTHON"] {
        if let Some(path) = env::var_os(env_name).map(PathBuf::from) {
            candidates.push(path);
        }
    }
    if let Some(path) = find_in_path("python3") {
        candidates.push(path);
    }
    if let Some(path) = find_in_path("python") {
        candidates.push(path);
    }

    let mut seen = Vec::<PathBuf>::new();
    candidates.into_iter().find_map(|python| {
        if seen.iter().any(|path| path == &python) {
            return None;
        }
        seen.push(python.clone());
        python_supports_onnx_genai(&python).then_some(OnnxGenaiPythonRuntime { python })
    })
}

fn python_supports_onnx_genai(python: &Path) -> bool {
    python.is_file()
        && Command::new(python)
            .args(["-c", "import onnxruntime_genai"])
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
}

fn onnx_genai_model_dir(store: &ModelStore, manifest: &ModelManifest) -> Option<PathBuf> {
    if manifest.format != ModelFormat::Onnx {
        return None;
    }

    if let Some(path) = manifest
        .model_path
        .as_deref()
        .map(|path| store.absolute_model_file(manifest, path))
    {
        if is_onnx_genai_dir(&path) {
            return Some(path);
        }
        if let Some(parent) = path.parent()
            && is_onnx_genai_dir(parent)
        {
            return Some(parent.to_path_buf());
        }
    }

    let mut candidates = manifest
        .files
        .iter()
        .filter(|file| file.path.ends_with("/genai_config.json"))
        .filter_map(|file| {
            store
                .model_dir(&manifest.id)
                .join(&file.path)
                .parent()
                .map(Path::to_path_buf)
        })
        .collect::<Vec<_>>();
    candidates
        .sort_by(|left, right| onnx_genai_dir_priority(left).cmp(&onnx_genai_dir_priority(right)));
    candidates.into_iter().find(|path| is_onnx_genai_dir(path))
}

fn is_onnx_genai_dir(path: &Path) -> bool {
    path.is_dir() && path.join("genai_config.json").is_file()
}

fn onnx_genai_dir_priority(path: &Path) -> (usize, String) {
    let text = path.to_string_lossy().to_ascii_lowercase();
    let priority = if text.contains("cpu_and_mobile") || text.contains("/cpu") {
        0
    } else if text.contains("cuda") {
        1
    } else if text.contains("directml") {
        2
    } else {
        3
    };
    (priority, text)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_store::{ModelFile, ModelSource};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_dir(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        env::temp_dir().join(format!("werk-onnxruntime-{name}-{unique}"))
    }

    fn manifest_with_model_path(id: &str, model_path: Option<&str>) -> ModelManifest {
        ModelManifest {
            id: id.to_string(),
            source: ModelSource::LocalPath {
                path: "test".to_string(),
            },
            format: ModelFormat::Onnx,
            architecture: Some("phi3".to_string()),
            tokenizer_path: None,
            config_path: None,
            model_path: model_path.map(str::to_string),
            backend: "onnxruntime".to_string(),
            created_unix: 0,
            files: Vec::new(),
            artifacts: Vec::new(),
        }
    }

    #[test]
    fn onnx_genai_model_dir_uses_manifest_model_parent() {
        let tmp = test_dir("manifest-parent");
        let store = ModelStore::resolve(Some(tmp.clone())).unwrap();
        let model_dir = store
            .model_dir("phi")
            .join("files/cpu_and_mobile/cpu-int4-rtn-block-32");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("genai_config.json"), b"{}").unwrap();
        fs::write(model_dir.join("model.onnx"), b"onnx").unwrap();

        let manifest = manifest_with_model_path(
            "phi",
            Some("files/cpu_and_mobile/cpu-int4-rtn-block-32/model.onnx"),
        );

        assert_eq!(onnx_genai_model_dir(&store, &manifest), Some(model_dir));

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn onnx_genai_model_dir_prefers_cpu_candidate_from_files() {
        let tmp = test_dir("cpu-priority");
        let store = ModelStore::resolve(Some(tmp.clone())).unwrap();
        let root = store.model_dir("phi");
        let cuda_dir = root.join("files/cuda/cuda-fp16");
        let cpu_dir = root.join("files/cpu_and_mobile/cpu-int4-rtn-block-32");
        fs::create_dir_all(&cuda_dir).unwrap();
        fs::create_dir_all(&cpu_dir).unwrap();
        fs::write(cuda_dir.join("genai_config.json"), b"{}").unwrap();
        fs::write(cpu_dir.join("genai_config.json"), b"{}").unwrap();

        let mut manifest = manifest_with_model_path("phi", None);
        manifest.files = vec![
            ModelFile {
                path: "files/cuda/cuda-fp16/genai_config.json".to_string(),
                size: 2,
                checksum: "crc32:0".to_string(),
            },
            ModelFile {
                path: "files/cpu_and_mobile/cpu-int4-rtn-block-32/genai_config.json".to_string(),
                size: 2,
                checksum: "crc32:0".to_string(),
            },
        ];

        assert_eq!(onnx_genai_model_dir(&store, &manifest), Some(cpu_dir));

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn onnx_genai_model_dir_ignores_non_onnx_manifest() {
        let tmp = test_dir("non-onnx");
        let store = ModelStore::resolve(Some(tmp.clone())).unwrap();
        let mut manifest = manifest_with_model_path("phi", Some("files/model.onnx"));
        manifest.format = ModelFormat::SafeTensors;

        assert!(onnx_genai_model_dir(&store, &manifest).is_none());

        let _ = fs::remove_dir_all(tmp);
    }
}
