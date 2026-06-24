mod burn;
mod candle;
mod external;
mod llama_fast;
mod llama_server;
mod onnxruntime;
mod vllm;

use anyhow::Result;
use std::pin::Pin;
use tokio_stream::Stream;

pub use burn::{
    BurnBackend, BurnDiscovery, BurnMode, managed_runner_path as managed_burn_runner_path,
};
pub use candle::{CandleBackend, CandleDeviceMode, probe_device};
pub use external::{LlamaCppBackend, LlamaCppMode, MlxBackend};
pub use llama_fast::{LlamaFastBackend, LlamaFastRuntimeReport};
pub use llama_server::{
    BackendDoctorCheck, LlamaServerBackend, LlamaServerDiscovery, backend_doctor_checks,
    install_managed_llama_server, llama_server_help_ok, managed_backend_dir,
};
pub use onnxruntime::{
    OnnxProvisionOptions, OnnxRuntimeAvailability, OnnxRuntimeBackend, OnnxRuntimeMode,
    install_managed_onnx_runtime, managed_runner_path,
};
pub use vllm::{
    VllmBackend, VllmDiscovery, install_managed_vllm, managed_vllm_dir, vllm_doctor_checks,
};

use crate::{
    model_store::{ModelFormat, ModelManifest},
    openai::ChatMessage,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendRuntime {
    Burn,
    Candle,
    LlamaServer,
    LlamaLegacy,
    LlamaHighlevel,
    Vllm,
    OnnxRuntime,
    Mlx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendAccelerator {
    Auto,
    Cpu,
    Cuda,
    Vulkan,
    Wgpu,
    Metal,
    Mlx,
    DirectMl,
    TensorRt,
    OpenVino,
    CoreMl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuntimeId {
    Burn,
    LlamaServerCuda,
    LlamaServerVulkan,
    LlamaServerCpu,
    CandleCuda,
    CandleMetal,
    CandleCpu,
    Mlx,
    VllmCuda,
    OnnxRuntimeCuda,
    OnnxRuntimeCpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeCapabilities {
    pub text_generation: bool,
    pub vision_language: bool,
    pub embeddings: bool,
    pub streaming: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct RuntimeDescriptor {
    pub id: RuntimeId,
    pub runtime: BackendRuntime,
    pub display_name: &'static str,
    pub supported_formats: &'static [ModelFormat],
    pub supported_architectures: &'static [&'static str],
    pub accelerators: &'static [BackendAccelerator],
    pub capabilities: RuntimeCapabilities,
    pub priority: i32,
    pub implemented: bool,
    pub install_target: Option<&'static str>,
}

const TEXT_STREAMING: RuntimeCapabilities = RuntimeCapabilities {
    text_generation: true,
    vision_language: false,
    embeddings: false,
    streaming: true,
};

const MLX_CAPABILITIES: RuntimeCapabilities = RuntimeCapabilities {
    text_generation: true,
    vision_language: true,
    embeddings: false,
    streaming: false,
};

const GGUF_FORMATS: &[ModelFormat] = &[ModelFormat::Gguf];
const SAFETENSORS_FORMATS: &[ModelFormat] = &[ModelFormat::SafeTensors];
const ONNX_RUNTIME_FORMATS: &[ModelFormat] = &[ModelFormat::SafeTensors, ModelFormat::Onnx];
const CANDLE_FORMATS: &[ModelFormat] = &[ModelFormat::Gguf, ModelFormat::SafeTensors];
const MLX_FORMATS: &[ModelFormat] = &[ModelFormat::Mlx, ModelFormat::SafeTensors];

const ANY_ARCH: &[&str] = &[];
const VLLM_ARCHES: &[&str] = &[
    "llama", "qwen2", "qwen3", "mistral", "mixtral", "phi3", "gemma", "gemma2", "gemma3",
];
pub const RUNTIME_REGISTRY: &[RuntimeDescriptor] = &[
    RuntimeDescriptor {
        id: RuntimeId::Burn,
        runtime: BackendRuntime::Burn,
        display_name: "Burn",
        supported_formats: SAFETENSORS_FORMATS,
        supported_architectures: ANY_ARCH,
        accelerators: &[BackendAccelerator::Cuda, BackendAccelerator::Cpu],
        capabilities: TEXT_STREAMING,
        priority: 980,
        implemented: true,
        install_target: None,
    },
    RuntimeDescriptor {
        id: RuntimeId::LlamaServerCuda,
        runtime: BackendRuntime::LlamaServer,
        display_name: "llama.cpp server CUDA",
        supported_formats: GGUF_FORMATS,
        supported_architectures: ANY_ARCH,
        accelerators: &[BackendAccelerator::Cuda],
        capabilities: TEXT_STREAMING,
        priority: 1000,
        implemented: true,
        install_target: Some("llama-cuda"),
    },
    RuntimeDescriptor {
        id: RuntimeId::LlamaServerVulkan,
        runtime: BackendRuntime::LlamaServer,
        display_name: "llama.cpp server Vulkan",
        supported_formats: GGUF_FORMATS,
        supported_architectures: ANY_ARCH,
        accelerators: &[BackendAccelerator::Vulkan],
        capabilities: TEXT_STREAMING,
        priority: 900,
        implemented: true,
        install_target: Some("llama-vulkan"),
    },
    RuntimeDescriptor {
        id: RuntimeId::LlamaServerCpu,
        runtime: BackendRuntime::LlamaServer,
        display_name: "llama.cpp server CPU",
        supported_formats: GGUF_FORMATS,
        supported_architectures: ANY_ARCH,
        accelerators: &[BackendAccelerator::Cpu],
        capabilities: TEXT_STREAMING,
        priority: 800,
        implemented: true,
        install_target: Some("llama-cpu"),
    },
    RuntimeDescriptor {
        id: RuntimeId::CandleCuda,
        runtime: BackendRuntime::Candle,
        display_name: "Candle CUDA",
        supported_formats: CANDLE_FORMATS,
        supported_architectures: ANY_ARCH,
        accelerators: &[BackendAccelerator::Cuda],
        capabilities: TEXT_STREAMING,
        priority: 700,
        implemented: true,
        install_target: None,
    },
    RuntimeDescriptor {
        id: RuntimeId::OnnxRuntimeCuda,
        runtime: BackendRuntime::OnnxRuntime,
        display_name: "ONNX Runtime CUDA",
        supported_formats: ONNX_RUNTIME_FORMATS,
        supported_architectures: ANY_ARCH,
        accelerators: &[BackendAccelerator::Cuda],
        capabilities: TEXT_STREAMING,
        priority: 960,
        implemented: true,
        install_target: None,
    },
    RuntimeDescriptor {
        id: RuntimeId::CandleMetal,
        runtime: BackendRuntime::Candle,
        display_name: "Candle Metal",
        supported_formats: CANDLE_FORMATS,
        supported_architectures: ANY_ARCH,
        accelerators: &[BackendAccelerator::Metal],
        capabilities: TEXT_STREAMING,
        priority: 650,
        implemented: true,
        install_target: None,
    },
    RuntimeDescriptor {
        id: RuntimeId::CandleCpu,
        runtime: BackendRuntime::Candle,
        display_name: "Candle CPU",
        supported_formats: CANDLE_FORMATS,
        supported_architectures: ANY_ARCH,
        accelerators: &[BackendAccelerator::Cpu],
        capabilities: TEXT_STREAMING,
        priority: 100,
        implemented: true,
        install_target: None,
    },
    RuntimeDescriptor {
        id: RuntimeId::OnnxRuntimeCpu,
        runtime: BackendRuntime::OnnxRuntime,
        display_name: "ONNX Runtime CPU",
        supported_formats: ONNX_RUNTIME_FORMATS,
        supported_architectures: ANY_ARCH,
        accelerators: &[BackendAccelerator::Cpu],
        capabilities: TEXT_STREAMING,
        priority: 760,
        implemented: true,
        install_target: None,
    },
    RuntimeDescriptor {
        id: RuntimeId::VllmCuda,
        runtime: BackendRuntime::Vllm,
        display_name: "vLLM CUDA",
        supported_formats: SAFETENSORS_FORMATS,
        supported_architectures: VLLM_ARCHES,
        accelerators: &[BackendAccelerator::Cuda],
        capabilities: TEXT_STREAMING,
        priority: 950,
        implemented: true,
        install_target: Some("vllm"),
    },
    RuntimeDescriptor {
        id: RuntimeId::Mlx,
        runtime: BackendRuntime::Mlx,
        display_name: "MLX",
        supported_formats: MLX_FORMATS,
        supported_architectures: ANY_ARCH,
        accelerators: &[BackendAccelerator::Mlx],
        capabilities: MLX_CAPABILITIES,
        priority: 850,
        implemented: true,
        install_target: None,
    },
];

pub fn runtime_registry() -> &'static [RuntimeDescriptor] {
    RUNTIME_REGISTRY
}

pub fn runtime_descriptor(id: RuntimeId) -> &'static RuntimeDescriptor {
    RUNTIME_REGISTRY
        .iter()
        .find(|runtime| runtime.id == id)
        .expect("runtime descriptor exists")
}

pub fn runtime_supports_model(
    descriptor: &RuntimeDescriptor,
    format: &ModelFormat,
    architecture: Option<&str>,
) -> bool {
    if !descriptor
        .supported_formats
        .iter()
        .any(|item| item == format)
    {
        return false;
    }
    if descriptor.supported_architectures.is_empty() {
        return true;
    }
    architecture
        .map(|architecture| {
            descriptor
                .supported_architectures
                .iter()
                .any(|supported| supported.eq_ignore_ascii_case(architecture))
        })
        .unwrap_or(false)
}

pub fn backend_supports_format(runtime: BackendRuntime, format: &ModelFormat) -> bool {
    match runtime {
        BackendRuntime::Candle => matches!(format, ModelFormat::Gguf | ModelFormat::SafeTensors),
        BackendRuntime::Burn => matches!(format, ModelFormat::SafeTensors),
        BackendRuntime::LlamaServer
        | BackendRuntime::LlamaLegacy
        | BackendRuntime::LlamaHighlevel => matches!(format, ModelFormat::Gguf),
        BackendRuntime::Vllm => matches!(format, ModelFormat::SafeTensors),
        BackendRuntime::OnnxRuntime => {
            matches!(format, ModelFormat::SafeTensors | ModelFormat::Onnx)
        }
        BackendRuntime::Mlx => matches!(format, ModelFormat::Mlx | ModelFormat::SafeTensors),
    }
}

pub fn backend_supports_images(runtime: BackendRuntime) -> bool {
    matches!(runtime, BackendRuntime::Mlx)
}

pub fn backend_supports_accelerator(
    runtime: BackendRuntime,
    accelerator: BackendAccelerator,
) -> bool {
    match runtime {
        BackendRuntime::Candle => matches!(
            accelerator,
            BackendAccelerator::Auto
                | BackendAccelerator::Cpu
                | BackendAccelerator::Cuda
                | BackendAccelerator::Metal
        ),
        BackendRuntime::Burn => matches!(
            accelerator,
            BackendAccelerator::Auto | BackendAccelerator::Cpu | BackendAccelerator::Cuda
        ),
        BackendRuntime::LlamaServer
        | BackendRuntime::LlamaLegacy
        | BackendRuntime::LlamaHighlevel => matches!(
            accelerator,
            BackendAccelerator::Cpu | BackendAccelerator::Cuda | BackendAccelerator::Vulkan
        ),
        BackendRuntime::Vllm => matches!(accelerator, BackendAccelerator::Cuda),
        BackendRuntime::OnnxRuntime => {
            matches!(
                accelerator,
                BackendAccelerator::Cuda | BackendAccelerator::Cpu
            )
        }
        BackendRuntime::Mlx => matches!(accelerator, BackendAccelerator::Mlx),
    }
}

pub fn explain_backend_rejection(
    runtime: BackendRuntime,
    format: &ModelFormat,
    has_images: bool,
) -> Option<&'static str> {
    if !backend_supports_format(runtime, format) {
        return Some(match runtime {
            BackendRuntime::Candle => "Candle supports GGUF and safetensors only",
            BackendRuntime::Burn => "Burn supports safetensors only",
            BackendRuntime::LlamaServer => "llama.cpp server supports GGUF only",
            BackendRuntime::LlamaLegacy | BackendRuntime::LlamaHighlevel => {
                "llama.cpp legacy backends support GGUF only"
            }
            BackendRuntime::Vllm => "vLLM supports selected HF safetensors models only",
            BackendRuntime::OnnxRuntime => {
                "ONNX Runtime supports ONNX models and selected HF safetensors models with managed ONNX artifacts"
            }
            BackendRuntime::Mlx => "MLX supports MLX and HF-style safetensors only",
        });
    }
    if has_images && !backend_supports_images(runtime) {
        return Some("backend is text-only");
    }
    None
}

pub fn select_backend_for_request<T, F>(
    candidates: &[T],
    format: &ModelFormat,
    has_images: bool,
    mut runtime_for: F,
) -> Option<T>
where
    T: Copy,
    F: FnMut(T) -> BackendRuntime,
{
    candidates.iter().copied().find(|candidate| {
        explain_backend_rejection(runtime_for(*candidate), format, has_images).is_none()
    })
}

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub messages: Vec<ChatMessage>,
    pub image_urls: Vec<String>,
    pub max_tokens: usize,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub stop: Vec<String>,
    pub seed: Option<u64>,
    pub stream_granularity: StreamGranularity,
    pub verbose: bool,
    pub debug: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamGranularity {
    Token,
    Chunk,
}

#[derive(Debug, Clone)]
pub struct GenerateResponse {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
    pub timings: GenerationTimings,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GenerationTimings {
    pub load_seconds: f64,
    pub warmup_seconds: f64,
    pub first_token_seconds: f64,
    pub prompt_seconds: f64,
    pub decode_seconds: f64,
    pub total_seconds: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum LlamaKvCacheType {
    F16,
    F32,
    Q8_0,
}

impl LlamaKvCacheType {
    pub fn label(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::Q8_0 => "q8_0",
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LlamaRuntimeOptions {
    pub ctx_size: Option<usize>,
    pub batch_size: Option<usize>,
    pub ubatch_size: Option<u32>,
    pub gpu_layers: Option<i32>,
    pub main_gpu: Option<i32>,
    pub kv_cache_type: Option<LlamaKvCacheType>,
    pub flash_attn: Option<bool>,
    pub kv_offload: Option<bool>,
    pub warmup_tokens: Option<usize>,
    pub threads: Option<u32>,
    pub threads_batch: Option<u32>,
}

#[derive(Debug, Clone)]
pub enum GenerateStreamEvent {
    TextChunk(String),
    Done {
        finish_reason: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        timings: GenerationTimings,
    },
}

pub type GenerateStream =
    Pin<Box<dyn Stream<Item = Result<GenerateStreamEvent, String>> + Send + 'static>>;

pub trait ChatGenerationSession: Send + Sync {
    fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse>;
    fn generate_stream(&self, request: GenerateRequest) -> GenerateStream;
}

pub trait GenerationBackend: Send + Sync {
    fn prepare(&self, _manifest: &ModelManifest) -> Result<()> {
        Ok(())
    }

    fn start_chat_session(
        &self,
        _manifest: &ModelManifest,
        _seed: Option<u64>,
    ) -> Result<Option<Box<dyn ChatGenerationSession>>> {
        Ok(None)
    }

    fn generate(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> Result<GenerateResponse>;
    fn generate_stream(&self, manifest: ModelManifest, request: GenerateRequest) -> GenerateStream;
}
