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

pub use burn::{BurnBackend, BurnMode, BurnProbeReport, BurnRuntimeStatus, burn_doctor_checks};
pub use candle::{CandleBackend, CandleDeviceMode, candle_gguf_tokenizer_rejection, probe_device};
pub use external::{
    LlamaCppBackend, LlamaCppMode, MlxBackend, MlxVlmBackend, TransformersCompatBackend,
    is_transformers_compat_model,
};
pub use llama_fast::{LlamaFastBackend, LlamaFastRuntimeReport};
pub use llama_server::{
    BackendDoctorCheck, LlamaServerBackend, LlamaServerDiscovery, LlamaServerInstallOptions,
    backend_doctor_checks, install_managed_llama_server, install_managed_llama_server_with_options,
    llama_server_help_ok, managed_backend_dir,
};
pub use onnxruntime::{
    OnnxProvisionOptions, OnnxRuntimeAvailability, OnnxRuntimeBackend, OnnxRuntimeMode,
    install_managed_onnx_runtime, managed_runner_path,
};
pub use vllm::{
    VllmBackend, VllmDiscovery, install_managed_vllm, managed_vllm_dir, vllm_doctor_checks,
};

use crate::{
    capabilities::{InferenceTask, RepositoryLayout},
    inference::ParameterSupportStatus,
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
    TransformersCompat,
    Vllm,
    OnnxRuntime,
    Mlx,
    MlxVlm,
    MediaCompanion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendAccelerator {
    Auto,
    Cpu,
    Cuda,
    Rocm,
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
    BurnCuda,
    BurnCpu,
    LlamaServerCuda,
    LlamaServerRocm,
    LlamaServerVulkan,
    LlamaServerMetal,
    LlamaServerCpu,
    CandleCuda,
    CandleMetal,
    CandleCpu,
    TransformersCompat,
    Mlx,
    MlxVlm,
    VllmCuda,
    VllmRocm,
    OnnxRuntimeCuda,
    OnnxRuntimeRocm,
    OnnxRuntimeCpu,
    MediaCompanionCuda,
    MediaCompanionRocm,
    MediaCompanionMetal,
    MediaCompanionCpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeCapabilities {
    pub text_generation: bool,
    pub vision_language: bool,
    pub embeddings: bool,
    pub streaming: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterSupportPath {
    Exact(&'static str),
    Prefix(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParameterSupportRule {
    pub path: ParameterSupportPath,
    pub status: ParameterSupportStatus,
}

impl ParameterSupportRule {
    pub const fn exact(path: &'static str, status: ParameterSupportStatus) -> Self {
        Self {
            path: ParameterSupportPath::Exact(path),
            status,
        }
    }

    pub const fn prefix(prefix: &'static str, status: ParameterSupportStatus) -> Self {
        Self {
            path: ParameterSupportPath::Prefix(prefix),
            status,
        }
    }

    fn matches(self, path: &str) -> bool {
        match self.path {
            ParameterSupportPath::Exact(expected) => path == expected,
            ParameterSupportPath::Prefix(prefix) => path.starts_with(prefix),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RuntimeDescriptor {
    pub id: RuntimeId,
    pub runtime: BackendRuntime,
    pub display_name: &'static str,
    pub supported_formats: &'static [ModelFormat],
    pub supported_architectures: &'static [&'static str],
    pub supported_tasks: &'static [InferenceTask],
    pub supported_layouts: &'static [RepositoryLayout],
    pub accelerators: &'static [BackendAccelerator],
    pub parameter_support: &'static [ParameterSupportRule],
    pub capabilities: RuntimeCapabilities,
    pub supports_offloading: bool,
    pub supports_quantization: bool,
    pub supports_compile: bool,
    pub supports_batching: bool,
    pub priority: i32,
    pub implemented: bool,
    pub install_target: Option<&'static str>,
}

impl RuntimeDescriptor {
    pub fn supports_task(&self, task: InferenceTask) -> bool {
        self.supported_tasks.contains(&task)
    }

    pub fn supports_layout(&self, layout: RepositoryLayout) -> bool {
        self.supported_layouts.is_empty() || self.supported_layouts.contains(&layout)
    }

    pub fn parameter_support_status(&self, path: &str) -> ParameterSupportStatus {
        self.parameter_support
            .iter()
            .copied()
            .find(|rule| matches!(rule.path, ParameterSupportPath::Exact(_)) && rule.matches(path))
            .or_else(|| {
                self.parameter_support
                    .iter()
                    .copied()
                    .filter(|rule| {
                        matches!(rule.path, ParameterSupportPath::Prefix(_)) && rule.matches(path)
                    })
                    .max_by_key(|rule| match rule.path {
                        ParameterSupportPath::Prefix(prefix) => prefix.len(),
                        ParameterSupportPath::Exact(_) => usize::MAX,
                    })
            })
            .map(|rule| rule.status)
            .unwrap_or(ParameterSupportStatus::ModelDependent)
    }
}

const TEXT_STREAMING: RuntimeCapabilities = RuntimeCapabilities {
    text_generation: true,
    vision_language: false,
    embeddings: false,
    streaming: true,
};

const TEXT_EMBEDDING_STREAMING: RuntimeCapabilities = RuntimeCapabilities {
    text_generation: true,
    vision_language: false,
    embeddings: true,
    streaming: true,
};

const MEDIA_CAPABILITIES: RuntimeCapabilities = RuntimeCapabilities {
    text_generation: false,
    vision_language: false,
    embeddings: false,
    streaming: false,
};

const MLX_CAPABILITIES: RuntimeCapabilities = RuntimeCapabilities {
    text_generation: true,
    vision_language: true,
    embeddings: false,
    streaming: true,
};

const MLX_TEXT_CAPABILITIES: RuntimeCapabilities = RuntimeCapabilities {
    text_generation: true,
    vision_language: false,
    embeddings: false,
    streaming: true,
};

const GGUF_FORMATS: &[ModelFormat] = &[ModelFormat::Gguf];
const SAFETENSORS_FORMATS: &[ModelFormat] = &[ModelFormat::SafeTensors];
const ONNX_RUNTIME_FORMATS: &[ModelFormat] = &[ModelFormat::SafeTensors, ModelFormat::Onnx];
const CANDLE_FORMATS: &[ModelFormat] = &[ModelFormat::Gguf, ModelFormat::SafeTensors];
const MLX_FORMATS: &[ModelFormat] = &[ModelFormat::Mlx, ModelFormat::SafeTensors];
const MEDIA_FORMATS: &[ModelFormat] = &[ModelFormat::SafeTensors, ModelFormat::PyTorch];

const ANY_ARCH: &[&str] = &[];
const VLLM_ARCHES: &[&str] = &[
    "llama", "qwen2", "qwen3", "mistral", "mixtral", "phi3", "gemma", "gemma2", "gemma3",
];
const TRANSFORMERS_COMPAT_ARCHES: &[&str] = &["chatglm"];
const MLX_VLM_ARCHES: &[&str] = &["gemma4_unified"];

const TEXT_GENERATION_TASKS: &[InferenceTask] = &[InferenceTask::TextGeneration];
const TEXT_AND_EMBEDDING_TASKS: &[InferenceTask] =
    &[InferenceTask::TextGeneration, InferenceTask::TextEmbedding];
const VLM_TASKS: &[InferenceTask] = &[
    InferenceTask::TextGeneration,
    InferenceTask::ImageUnderstanding,
];
const MEDIA_TASKS: &[InferenceTask] = &[
    InferenceTask::ImageGeneration,
    InferenceTask::ImageEditing,
    InferenceTask::ImageVariation,
    InferenceTask::ImageInpainting,
    InferenceTask::ImageOutpainting,
    InferenceTask::ImageUpscaling,
    InferenceTask::VideoGeneration,
    InferenceTask::ImageToVideo,
    InferenceTask::VideoToVideo,
    InferenceTask::VideoInpainting,
    InferenceTask::VideoExtension,
    InferenceTask::VideoUpscaling,
    InferenceTask::FrameInterpolation,
    InferenceTask::AudioGeneration,
    InferenceTask::MusicGeneration,
    InferenceTask::TextToSpeech,
    InferenceTask::SpeechToText,
];

const GGUF_LAYOUTS: &[RepositoryLayout] = &[
    RepositoryLayout::Gguf,
    RepositoryLayout::SingleFile,
    RepositoryLayout::Custom,
];
const TRANSFORMERS_LAYOUTS: &[RepositoryLayout] = &[
    RepositoryLayout::Transformers,
    RepositoryLayout::SingleFile,
    RepositoryLayout::Custom,
];
const CANDLE_LAYOUTS: &[RepositoryLayout] = &[
    RepositoryLayout::Gguf,
    RepositoryLayout::Transformers,
    RepositoryLayout::SingleFile,
    RepositoryLayout::Custom,
];
const ONNX_LAYOUTS: &[RepositoryLayout] = &[
    RepositoryLayout::OnnxBundle,
    RepositoryLayout::Transformers,
    RepositoryLayout::SingleFile,
    RepositoryLayout::Custom,
];
const MLX_LAYOUTS: &[RepositoryLayout] = &[
    RepositoryLayout::Mlx,
    RepositoryLayout::Transformers,
    RepositoryLayout::SingleFile,
    RepositoryLayout::Custom,
];
const MEDIA_LAYOUTS: &[RepositoryLayout] = &[
    RepositoryLayout::Diffusers,
    RepositoryLayout::Transformers,
    RepositoryLayout::SingleFile,
    RepositoryLayout::Custom,
];

const TEXT_PARAMETER_SUPPORT: &[ParameterSupportRule] = &[
    ParameterSupportRule::prefix("text.", ParameterSupportStatus::ModelDependent),
    ParameterSupportRule::prefix("routing.", ParameterSupportStatus::ModelDependent),
];
const MEDIA_PARAMETER_SUPPORT: &[ParameterSupportRule] = &[
    ParameterSupportRule::exact(
        "routing.allow_disk_offload",
        ParameterSupportStatus::Unsupported,
    ),
    ParameterSupportRule::exact("routing.compile", ParameterSupportStatus::Unsupported),
    ParameterSupportRule::exact("routing.quantization", ParameterSupportStatus::Unsupported),
    ParameterSupportRule::exact(
        "routing.attention_backend",
        ParameterSupportStatus::Unsupported,
    ),
    ParameterSupportRule::exact("image.output_path", ParameterSupportStatus::Unsupported),
    ParameterSupportRule::exact("video.output_path", ParameterSupportStatus::Unsupported),
    ParameterSupportRule::exact("audio.output_path", ParameterSupportStatus::Unsupported),
    ParameterSupportRule::exact("tts.output_path", ParameterSupportStatus::Unsupported),
    ParameterSupportRule::exact("stt.output_path", ParameterSupportStatus::Unsupported),
    ParameterSupportRule::exact("image.width", ParameterSupportStatus::Translated),
    ParameterSupportRule::exact("image.height", ParameterSupportStatus::Translated),
    ParameterSupportRule::exact("image.steps", ParameterSupportStatus::Translated),
    ParameterSupportRule::exact("image.guidance", ParameterSupportStatus::Translated),
    ParameterSupportRule::exact("image.seed", ParameterSupportStatus::Translated),
    ParameterSupportRule::exact("image.output_format", ParameterSupportStatus::Native),
    ParameterSupportRule::exact("video.width", ParameterSupportStatus::Translated),
    ParameterSupportRule::exact("video.height", ParameterSupportStatus::Translated),
    ParameterSupportRule::exact("video.frames", ParameterSupportStatus::Translated),
    ParameterSupportRule::exact("video.fps", ParameterSupportStatus::Translated),
    ParameterSupportRule::exact("video.steps", ParameterSupportStatus::Translated),
    ParameterSupportRule::exact("video.guidance", ParameterSupportStatus::Translated),
    ParameterSupportRule::exact("video.seed", ParameterSupportStatus::Translated),
    ParameterSupportRule::exact("video.output_format", ParameterSupportStatus::Native),
    ParameterSupportRule::exact("audio.duration", ParameterSupportStatus::Translated),
    ParameterSupportRule::exact("audio.steps", ParameterSupportStatus::Translated),
    ParameterSupportRule::exact("audio.guidance", ParameterSupportStatus::Translated),
    ParameterSupportRule::exact("audio.seed", ParameterSupportStatus::Translated),
    ParameterSupportRule::exact("audio.sample_rate", ParameterSupportStatus::Unsupported),
    ParameterSupportRule::exact("audio.output_format", ParameterSupportStatus::Native),
    ParameterSupportRule::exact("tts.voice", ParameterSupportStatus::Unsupported),
    ParameterSupportRule::exact("tts.speed", ParameterSupportStatus::Unsupported),
    ParameterSupportRule::exact("tts.pitch", ParameterSupportStatus::Unsupported),
    ParameterSupportRule::exact("tts.sample_rate", ParameterSupportStatus::Unsupported),
    ParameterSupportRule::exact("tts.streaming", ParameterSupportStatus::Unsupported),
    ParameterSupportRule::exact("tts.output_format", ParameterSupportStatus::Native),
    ParameterSupportRule::exact("stt.language", ParameterSupportStatus::Translated),
    ParameterSupportRule::exact("stt.temperature", ParameterSupportStatus::Translated),
    ParameterSupportRule::exact("stt.output_format", ParameterSupportStatus::Native),
    ParameterSupportRule::prefix("image.", ParameterSupportStatus::ModelDependent),
    ParameterSupportRule::prefix("video.", ParameterSupportStatus::ModelDependent),
    ParameterSupportRule::prefix("audio.", ParameterSupportStatus::ModelDependent),
    ParameterSupportRule::prefix("tts.", ParameterSupportStatus::ModelDependent),
    ParameterSupportRule::prefix("stt.", ParameterSupportStatus::ModelDependent),
    ParameterSupportRule::prefix("routing.", ParameterSupportStatus::Translated),
];
pub const RUNTIME_REGISTRY: &[RuntimeDescriptor] = &[
    RuntimeDescriptor {
        id: RuntimeId::BurnCuda,
        runtime: BackendRuntime::Burn,
        display_name: "Burn CUDA",
        supported_formats: SAFETENSORS_FORMATS,
        supported_architectures: ANY_ARCH,
        supported_tasks: TEXT_GENERATION_TASKS,
        supported_layouts: TRANSFORMERS_LAYOUTS,
        accelerators: &[BackendAccelerator::Cuda],
        parameter_support: TEXT_PARAMETER_SUPPORT,
        capabilities: TEXT_STREAMING,
        supports_offloading: false,
        supports_quantization: false,
        supports_compile: false,
        supports_batching: true,
        priority: 980,
        implemented: cfg!(feature = "burn-cuda"),
        install_target: None,
    },
    RuntimeDescriptor {
        id: RuntimeId::BurnCpu,
        runtime: BackendRuntime::Burn,
        display_name: "Burn CPU",
        supported_formats: SAFETENSORS_FORMATS,
        supported_architectures: ANY_ARCH,
        supported_tasks: TEXT_GENERATION_TASKS,
        supported_layouts: TRANSFORMERS_LAYOUTS,
        accelerators: &[BackendAccelerator::Cpu],
        parameter_support: TEXT_PARAMETER_SUPPORT,
        capabilities: TEXT_STREAMING,
        supports_offloading: false,
        supports_quantization: false,
        supports_compile: false,
        supports_batching: true,
        priority: 780,
        implemented: cfg!(feature = "burn-cpu"),
        install_target: None,
    },
    RuntimeDescriptor {
        id: RuntimeId::LlamaServerCuda,
        runtime: BackendRuntime::LlamaServer,
        display_name: "llama.cpp server CUDA",
        supported_formats: GGUF_FORMATS,
        supported_architectures: ANY_ARCH,
        supported_tasks: TEXT_AND_EMBEDDING_TASKS,
        supported_layouts: GGUF_LAYOUTS,
        accelerators: &[BackendAccelerator::Cuda],
        parameter_support: TEXT_PARAMETER_SUPPORT,
        capabilities: TEXT_EMBEDDING_STREAMING,
        supports_offloading: true,
        supports_quantization: true,
        supports_compile: false,
        supports_batching: true,
        priority: 1000,
        implemented: true,
        install_target: Some("llama-cuda"),
    },
    RuntimeDescriptor {
        id: RuntimeId::LlamaServerRocm,
        runtime: BackendRuntime::LlamaServer,
        display_name: "llama.cpp server ROCm/HIP",
        supported_formats: GGUF_FORMATS,
        supported_architectures: ANY_ARCH,
        supported_tasks: TEXT_AND_EMBEDDING_TASKS,
        supported_layouts: GGUF_LAYOUTS,
        accelerators: &[BackendAccelerator::Rocm],
        parameter_support: TEXT_PARAMETER_SUPPORT,
        capabilities: TEXT_EMBEDDING_STREAMING,
        supports_offloading: true,
        supports_quantization: true,
        supports_compile: false,
        supports_batching: true,
        priority: 950,
        implemented: true,
        install_target: Some("llama-rocm"),
    },
    RuntimeDescriptor {
        id: RuntimeId::LlamaServerVulkan,
        runtime: BackendRuntime::LlamaServer,
        display_name: "llama.cpp server Vulkan",
        supported_formats: GGUF_FORMATS,
        supported_architectures: ANY_ARCH,
        supported_tasks: TEXT_AND_EMBEDDING_TASKS,
        supported_layouts: GGUF_LAYOUTS,
        accelerators: &[BackendAccelerator::Vulkan],
        parameter_support: TEXT_PARAMETER_SUPPORT,
        capabilities: TEXT_EMBEDDING_STREAMING,
        supports_offloading: true,
        supports_quantization: true,
        supports_compile: false,
        supports_batching: true,
        priority: 900,
        implemented: true,
        install_target: Some("llama-vulkan"),
    },
    RuntimeDescriptor {
        id: RuntimeId::LlamaServerMetal,
        runtime: BackendRuntime::LlamaServer,
        display_name: "llama.cpp server Metal",
        supported_formats: GGUF_FORMATS,
        supported_architectures: ANY_ARCH,
        supported_tasks: TEXT_AND_EMBEDDING_TASKS,
        supported_layouts: GGUF_LAYOUTS,
        accelerators: &[BackendAccelerator::Metal],
        parameter_support: TEXT_PARAMETER_SUPPORT,
        capabilities: TEXT_EMBEDDING_STREAMING,
        supports_offloading: true,
        supports_quantization: true,
        supports_compile: false,
        supports_batching: true,
        priority: 925,
        implemented: true,
        install_target: Some("llama-metal"),
    },
    RuntimeDescriptor {
        id: RuntimeId::LlamaServerCpu,
        runtime: BackendRuntime::LlamaServer,
        display_name: "llama.cpp server CPU",
        supported_formats: GGUF_FORMATS,
        supported_architectures: ANY_ARCH,
        supported_tasks: TEXT_AND_EMBEDDING_TASKS,
        supported_layouts: GGUF_LAYOUTS,
        accelerators: &[BackendAccelerator::Cpu],
        parameter_support: TEXT_PARAMETER_SUPPORT,
        capabilities: TEXT_EMBEDDING_STREAMING,
        supports_offloading: true,
        supports_quantization: true,
        supports_compile: false,
        supports_batching: true,
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
        supported_tasks: TEXT_GENERATION_TASKS,
        supported_layouts: CANDLE_LAYOUTS,
        accelerators: &[BackendAccelerator::Cuda],
        parameter_support: TEXT_PARAMETER_SUPPORT,
        capabilities: TEXT_STREAMING,
        supports_offloading: true,
        supports_quantization: true,
        supports_compile: false,
        supports_batching: true,
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
        supported_tasks: TEXT_AND_EMBEDDING_TASKS,
        supported_layouts: ONNX_LAYOUTS,
        accelerators: &[BackendAccelerator::Cuda],
        parameter_support: TEXT_PARAMETER_SUPPORT,
        capabilities: TEXT_EMBEDDING_STREAMING,
        supports_offloading: false,
        supports_quantization: false,
        supports_compile: false,
        supports_batching: true,
        priority: 960,
        implemented: true,
        install_target: None,
    },
    RuntimeDescriptor {
        id: RuntimeId::OnnxRuntimeRocm,
        runtime: BackendRuntime::OnnxRuntime,
        display_name: "ONNX Runtime ROCm",
        supported_formats: ONNX_RUNTIME_FORMATS,
        supported_architectures: ANY_ARCH,
        supported_tasks: TEXT_AND_EMBEDDING_TASKS,
        supported_layouts: ONNX_LAYOUTS,
        accelerators: &[BackendAccelerator::Rocm],
        parameter_support: TEXT_PARAMETER_SUPPORT,
        capabilities: TEXT_EMBEDDING_STREAMING,
        supports_offloading: false,
        supports_quantization: false,
        supports_compile: false,
        supports_batching: true,
        priority: 955,
        implemented: true,
        install_target: None,
    },
    RuntimeDescriptor {
        id: RuntimeId::CandleMetal,
        runtime: BackendRuntime::Candle,
        display_name: "Candle Metal",
        supported_formats: CANDLE_FORMATS,
        supported_architectures: ANY_ARCH,
        supported_tasks: TEXT_GENERATION_TASKS,
        supported_layouts: CANDLE_LAYOUTS,
        accelerators: &[BackendAccelerator::Metal],
        parameter_support: TEXT_PARAMETER_SUPPORT,
        capabilities: TEXT_STREAMING,
        supports_offloading: true,
        supports_quantization: true,
        supports_compile: false,
        supports_batching: true,
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
        supported_tasks: TEXT_GENERATION_TASKS,
        supported_layouts: CANDLE_LAYOUTS,
        accelerators: &[BackendAccelerator::Cpu],
        parameter_support: TEXT_PARAMETER_SUPPORT,
        capabilities: TEXT_STREAMING,
        supports_offloading: false,
        supports_quantization: true,
        supports_compile: false,
        supports_batching: true,
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
        supported_tasks: TEXT_AND_EMBEDDING_TASKS,
        supported_layouts: ONNX_LAYOUTS,
        accelerators: &[BackendAccelerator::Cpu],
        parameter_support: TEXT_PARAMETER_SUPPORT,
        capabilities: TEXT_EMBEDDING_STREAMING,
        supports_offloading: false,
        supports_quantization: true,
        supports_compile: true,
        supports_batching: true,
        priority: 760,
        implemented: true,
        install_target: None,
    },
    RuntimeDescriptor {
        id: RuntimeId::TransformersCompat,
        runtime: BackendRuntime::TransformersCompat,
        display_name: "Transformers compatibility",
        supported_formats: SAFETENSORS_FORMATS,
        supported_architectures: TRANSFORMERS_COMPAT_ARCHES,
        supported_tasks: TEXT_GENERATION_TASKS,
        supported_layouts: TRANSFORMERS_LAYOUTS,
        accelerators: &[BackendAccelerator::Auto],
        parameter_support: TEXT_PARAMETER_SUPPORT,
        capabilities: TEXT_STREAMING,
        supports_offloading: true,
        supports_quantization: true,
        supports_compile: true,
        supports_batching: true,
        priority: 840,
        implemented: true,
        install_target: None,
    },
    RuntimeDescriptor {
        id: RuntimeId::VllmCuda,
        runtime: BackendRuntime::Vllm,
        display_name: "vLLM CUDA",
        supported_formats: SAFETENSORS_FORMATS,
        supported_architectures: VLLM_ARCHES,
        supported_tasks: TEXT_GENERATION_TASKS,
        supported_layouts: TRANSFORMERS_LAYOUTS,
        accelerators: &[BackendAccelerator::Cuda],
        parameter_support: TEXT_PARAMETER_SUPPORT,
        capabilities: TEXT_STREAMING,
        supports_offloading: true,
        supports_quantization: true,
        supports_compile: true,
        supports_batching: true,
        priority: 950,
        implemented: true,
        install_target: Some("vllm"),
    },
    RuntimeDescriptor {
        id: RuntimeId::VllmRocm,
        runtime: BackendRuntime::Vllm,
        display_name: "vLLM ROCm",
        supported_formats: SAFETENSORS_FORMATS,
        supported_architectures: VLLM_ARCHES,
        supported_tasks: TEXT_GENERATION_TASKS,
        supported_layouts: TRANSFORMERS_LAYOUTS,
        accelerators: &[BackendAccelerator::Rocm],
        parameter_support: TEXT_PARAMETER_SUPPORT,
        capabilities: TEXT_STREAMING,
        supports_offloading: true,
        supports_quantization: true,
        supports_compile: true,
        supports_batching: true,
        priority: 945,
        implemented: true,
        install_target: Some("vllm"),
    },
    RuntimeDescriptor {
        id: RuntimeId::MlxVlm,
        runtime: BackendRuntime::MlxVlm,
        display_name: "MLX-VLM",
        supported_formats: MLX_FORMATS,
        supported_architectures: MLX_VLM_ARCHES,
        supported_tasks: VLM_TASKS,
        supported_layouts: MLX_LAYOUTS,
        accelerators: &[BackendAccelerator::Mlx],
        parameter_support: TEXT_PARAMETER_SUPPORT,
        capabilities: MLX_CAPABILITIES,
        supports_offloading: false,
        supports_quantization: true,
        supports_compile: false,
        supports_batching: true,
        priority: 875,
        implemented: true,
        install_target: None,
    },
    RuntimeDescriptor {
        id: RuntimeId::Mlx,
        runtime: BackendRuntime::Mlx,
        display_name: "MLX",
        supported_formats: MLX_FORMATS,
        supported_architectures: ANY_ARCH,
        supported_tasks: TEXT_GENERATION_TASKS,
        supported_layouts: MLX_LAYOUTS,
        accelerators: &[BackendAccelerator::Mlx],
        parameter_support: TEXT_PARAMETER_SUPPORT,
        capabilities: MLX_TEXT_CAPABILITIES,
        supports_offloading: false,
        supports_quantization: true,
        supports_compile: false,
        supports_batching: true,
        priority: 850,
        implemented: true,
        install_target: None,
    },
    RuntimeDescriptor {
        id: RuntimeId::MediaCompanionCuda,
        runtime: BackendRuntime::MediaCompanion,
        display_name: "Media companion CUDA",
        supported_formats: MEDIA_FORMATS,
        supported_architectures: ANY_ARCH,
        supported_tasks: MEDIA_TASKS,
        supported_layouts: MEDIA_LAYOUTS,
        accelerators: &[BackendAccelerator::Cuda],
        parameter_support: MEDIA_PARAMETER_SUPPORT,
        capabilities: MEDIA_CAPABILITIES,
        supports_offloading: true,
        supports_quantization: false,
        supports_compile: false,
        supports_batching: true,
        priority: 1000,
        implemented: true,
        install_target: None,
    },
    RuntimeDescriptor {
        id: RuntimeId::MediaCompanionRocm,
        runtime: BackendRuntime::MediaCompanion,
        display_name: "Media companion ROCm",
        supported_formats: MEDIA_FORMATS,
        supported_architectures: ANY_ARCH,
        supported_tasks: MEDIA_TASKS,
        supported_layouts: MEDIA_LAYOUTS,
        accelerators: &[BackendAccelerator::Rocm],
        parameter_support: MEDIA_PARAMETER_SUPPORT,
        capabilities: MEDIA_CAPABILITIES,
        supports_offloading: true,
        supports_quantization: false,
        supports_compile: false,
        supports_batching: true,
        priority: 990,
        implemented: true,
        install_target: None,
    },
    RuntimeDescriptor {
        id: RuntimeId::MediaCompanionMetal,
        runtime: BackendRuntime::MediaCompanion,
        display_name: "Media companion Metal",
        supported_formats: MEDIA_FORMATS,
        supported_architectures: ANY_ARCH,
        supported_tasks: MEDIA_TASKS,
        supported_layouts: MEDIA_LAYOUTS,
        accelerators: &[BackendAccelerator::Metal],
        parameter_support: MEDIA_PARAMETER_SUPPORT,
        capabilities: MEDIA_CAPABILITIES,
        supports_offloading: true,
        supports_quantization: false,
        supports_compile: false,
        supports_batching: true,
        priority: 980,
        implemented: true,
        install_target: None,
    },
    RuntimeDescriptor {
        id: RuntimeId::MediaCompanionCpu,
        runtime: BackendRuntime::MediaCompanion,
        display_name: "Media companion CPU",
        supported_formats: MEDIA_FORMATS,
        supported_architectures: ANY_ARCH,
        supported_tasks: MEDIA_TASKS,
        supported_layouts: MEDIA_LAYOUTS,
        accelerators: &[BackendAccelerator::Cpu],
        parameter_support: MEDIA_PARAMETER_SUPPORT,
        capabilities: MEDIA_CAPABILITIES,
        supports_offloading: true,
        supports_quantization: false,
        supports_compile: false,
        supports_batching: true,
        priority: 500,
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

pub fn runtime_supports_task(descriptor: &RuntimeDescriptor, task: InferenceTask) -> bool {
    descriptor.supports_task(task)
}

pub fn runtime_supports_layout(descriptor: &RuntimeDescriptor, layout: RepositoryLayout) -> bool {
    descriptor.supports_layout(layout)
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
        BackendRuntime::TransformersCompat => matches!(format, ModelFormat::SafeTensors),
        BackendRuntime::Mlx | BackendRuntime::MlxVlm => {
            matches!(format, ModelFormat::Mlx | ModelFormat::SafeTensors)
        }
        BackendRuntime::MediaCompanion => MEDIA_FORMATS.contains(format),
    }
}

pub fn backend_supports_images(runtime: BackendRuntime) -> bool {
    matches!(runtime, BackendRuntime::MlxVlm)
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
            BackendAccelerator::Cpu
                | BackendAccelerator::Cuda
                | BackendAccelerator::Rocm
                | BackendAccelerator::Vulkan
        ),
        BackendRuntime::Vllm => matches!(
            accelerator,
            BackendAccelerator::Cuda | BackendAccelerator::Rocm
        ),
        BackendRuntime::OnnxRuntime => {
            matches!(
                accelerator,
                BackendAccelerator::Cuda | BackendAccelerator::Rocm | BackendAccelerator::Cpu
            )
        }
        BackendRuntime::TransformersCompat => matches!(
            accelerator,
            BackendAccelerator::Auto
                | BackendAccelerator::Cpu
                | BackendAccelerator::Cuda
                | BackendAccelerator::Metal
        ),
        BackendRuntime::Mlx | BackendRuntime::MlxVlm => {
            matches!(accelerator, BackendAccelerator::Mlx)
        }
        BackendRuntime::MediaCompanion => matches!(
            accelerator,
            BackendAccelerator::Cpu
                | BackendAccelerator::Cuda
                | BackendAccelerator::Rocm
                | BackendAccelerator::Metal
        ),
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
            BackendRuntime::TransformersCompat => {
                "Transformers compatibility supports selected raw HF safetensors models"
            }
            BackendRuntime::Mlx => "MLX supports MLX and HF-style safetensors only",
            BackendRuntime::MlxVlm => "MLX-VLM supports MLX and HF-style safetensors VLMs only",
            BackendRuntime::MediaCompanion => {
                "media companion supports safetensors, PyTorch, ONNX, MLX, TensorFlow, and custom media repositories"
            }
        });
    }
    if has_images && !backend_supports_images(runtime) {
        if runtime == BackendRuntime::MediaCompanion {
            return Some("media companion does not implement image-understanding chat requests");
        }
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
    pub backend_diagnostics: Vec<String>,
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
        backend_diagnostics: Vec<String>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_companion_registry_covers_media_tasks_and_layouts() {
        let companions = runtime_registry()
            .iter()
            .filter(|descriptor| descriptor.runtime == BackendRuntime::MediaCompanion)
            .collect::<Vec<_>>();

        assert_eq!(companions.len(), 4);
        for descriptor in companions {
            assert!(descriptor.implemented);
            assert!(descriptor.supports_offloading);
            assert!(!descriptor.supports_quantization);
            assert!(!descriptor.supports_compile);
            assert!(descriptor.supports_batching);
            for task in MEDIA_TASKS {
                assert!(
                    descriptor.supports_task(*task),
                    "{} is missing task {task}",
                    descriptor.display_name
                );
            }
            for layout in MEDIA_LAYOUTS {
                assert!(
                    descriptor.supports_layout(*layout),
                    "{} is missing layout {layout}",
                    descriptor.display_name
                );
            }
        }
    }

    #[test]
    fn registry_exposes_typed_text_tasks_and_layouts() {
        let llama = runtime_descriptor(RuntimeId::LlamaServerCpu);
        assert!(llama.supports_task(InferenceTask::TextGeneration));
        assert!(llama.supports_task(InferenceTask::TextEmbedding));
        assert!(llama.supports_layout(RepositoryLayout::Gguf));

        let mlx_vlm = runtime_descriptor(RuntimeId::MlxVlm);
        assert!(mlx_vlm.supports_task(InferenceTask::ImageUnderstanding));
        assert!(mlx_vlm.supports_layout(RepositoryLayout::Mlx));
        assert!(!mlx_vlm.supports_task(InferenceTask::ImageGeneration));
    }

    #[test]
    fn parameter_support_prefers_exact_paths_then_longest_prefix() {
        let descriptor = runtime_descriptor(RuntimeId::MediaCompanionCuda);
        assert_eq!(
            descriptor.parameter_support_status("image.steps"),
            ParameterSupportStatus::Translated
        );
        assert_eq!(
            descriptor.parameter_support_status("image.output_format"),
            ParameterSupportStatus::Native
        );
        assert_eq!(
            descriptor.parameter_support_status("routing.compile"),
            ParameterSupportStatus::Unsupported
        );
        assert_eq!(
            descriptor.parameter_support_status("routing.allow_disk_offload"),
            ParameterSupportStatus::Unsupported
        );
        assert_eq!(
            descriptor.parameter_support_status("vendor.extension"),
            ParameterSupportStatus::ModelDependent
        );
    }

    #[test]
    fn legacy_chat_selection_rejects_media_companion() {
        assert_eq!(
            explain_backend_rejection(
                BackendRuntime::MediaCompanion,
                &ModelFormat::SafeTensors,
                true,
            ),
            Some("media companion does not implement image-understanding chat requests")
        );
        assert_eq!(
            select_backend_for_request(
                &[BackendRuntime::MediaCompanion],
                &ModelFormat::SafeTensors,
                true,
                |runtime| runtime,
            ),
            None
        );
    }
}
