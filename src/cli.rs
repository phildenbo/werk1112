use anyhow::{Context, Result, anyhow, bail};
use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::{
    collections::{BTreeMap, HashMap},
    env, fs,
    io::{self, IsTerminal, Read, Write},
    net::{IpAddr, SocketAddr},
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio_stream::StreamExt;

#[cfg(feature = "burn-experimental")]
use crate::backend::burn_doctor_checks;
use crate::{
    api::{ApiState, serve},
    api_keys,
    backend::{
        BackendAccelerator, BackendRuntime, BurnBackend, BurnMode, CandleBackend, CandleDeviceMode,
        ChatGenerationSession, GenerateRequest, GenerateStreamEvent, GenerationBackend,
        GenerationTimings, LlamaCppBackend, LlamaCppMode, LlamaFastBackend, LlamaFastRuntimeReport,
        LlamaKvCacheType, LlamaRuntimeOptions, LlamaServerBackend, LlamaServerDiscovery,
        LlamaServerInstallOptions, MlxBackend, MlxVlmBackend, OnnxProvisionOptions,
        OnnxRuntimeAvailability, OnnxRuntimeBackend, OnnxRuntimeMode, RuntimeId, StreamGranularity,
        TransformersCompatBackend, VllmBackend, backend_doctor_checks,
        backend_supports_accelerator, backend_supports_format,
        backend_supports_images as runtime_supports_images, candle_gguf_tokenizer_rejection,
        install_managed_llama_server, install_managed_llama_server_with_options,
        install_managed_onnx_runtime, install_managed_vllm, llama_server_help_ok,
        managed_backend_dir, managed_runner_path as managed_onnx_runner_path, managed_vllm_dir,
        probe_device, runtime_descriptor, runtime_registry, runtime_supports_model,
        vllm_doctor_checks,
    },
    banner::print_banner,
    capabilities::{InferenceTask, InputModality, OutputModality, RepositoryLayout},
    inference::{
        InferenceInput, InferenceInputSource, InferenceRequest, OverrideBool, ParameterPolicy,
        ParameterValue, RoutingOverrides, WorkloadEstimate, parameter_schema,
        parameter_schema_for_manifest,
    },
    inference_service::{InferenceResult, InferenceService},
    media_cli::{
        AudioCommands, ImageCommands, RoutingArgs, VideoCommands, collect_raw_overrides,
        parse_set_overrides,
    },
    media_companion::CompanionClient,
    model_store::{
        ArtifactStatus, ModelArtifact, ModelFormat, ModelManifest, ModelSource, ModelStore,
        PullProgress,
    },
    openai::{
        ChatMessage, ChatTemplateOptions, ChatTemplateSource, MessageContent, PromptSpec,
        messages_to_prompt_for_model, messages_to_prompt_for_model_with_template,
    },
    runtime_planner::{
        RequestCapabilities, RequestedBackend, RuntimeAvailability, RuntimeDecisionStatus,
        plan_runtime, runtime_candidate_ids, select_runtime,
    },
};

const DEFAULT_MAX_NEW_TOKENS: usize = 256;
const MIB: u64 = 1024 * 1024;
#[cfg(test)]
const GIB: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Clone, Parser)]
#[command(
    name = "werk",
    version,
    about = "Headless local model server with an OpenAI-compatible API",
    long_about = "Werk1112 imports local or Hugging Face models into a managed store, serves an OpenAI-compatible HTTP API, and stays UI-free for external clients such as Open WebUI, LM Studio, and agent tools."
)]
pub struct Cli {
    #[arg(
        long,
        global = true,
        env = "WERK_HOME",
        help = "Model store directory; defaults to WERK_HOME, XDG_DATA_HOME/werk1112, or ~/.local/share/werk1112"
    )]
    pub model_home: Option<PathBuf>,

    #[arg(
        long,
        global = true,
        value_enum,
        help = "Candle-only device override for this command: auto, cpu, cuda, or metal"
    )]
    pub device: Option<DeviceArg>,

    #[arg(
        long,
        global = true,
        value_enum,
        default_value_t = BackendArg::Auto,
        help = "Backend for this process: auto, cpu, cuda, rocm, vulkan, metal, mlx, onnx, transformers, vllm, candle, llama-highlevel, or llama-legacy"
    )]
    pub backend: BackendArg,

    #[arg(
        long,
        global = true,
        action = ArgAction::SetTrue,
        conflicts_with = "no_auto_install_backends",
        help = "Allow Werk to provision missing managed runtime backends during selection"
    )]
    pub auto_install_backends: bool,

    #[arg(
        long,
        global = true,
        action = ArgAction::SetTrue,
        help = "Disable automatic managed runtime provisioning"
    )]
    pub no_auto_install_backends: bool,

    #[command(flatten)]
    pub llama: LlamaRuntimeArgs,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum DeviceArg {
    Auto,
    Cpu,
    Cuda,
    Metal,
}

fn parse_inference_task(value: &str) -> std::result::Result<InferenceTask, String> {
    value.parse()
}

fn parse_input_modality(value: &str) -> std::result::Result<InputModality, String> {
    value.parse()
}

fn parse_output_modality(value: &str) -> std::result::Result<OutputModality, String> {
    value.parse()
}

fn parse_repository_layout(value: &str) -> std::result::Result<RepositoryLayout, String> {
    value.parse()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum StreamGranularityArg {
    Token,
    Chunk,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ChatTemplateArg {
    Model,
    Generic,
    Phi3,
    Llama3,
    Gemma,
    Chatml,
    #[value(name = "qwen-chatml")]
    QwenChatml,
    None,
}

impl ChatTemplateArg {
    fn template_name(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Generic => "generic",
            Self::Phi3 => "phi3",
            Self::Llama3 => "llama3",
            Self::Gemma => "gemma",
            Self::Chatml => "chatml",
            Self::QwenChatml => "qwen-chatml",
            Self::None => "none",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BackendArg {
    Auto,
    #[cfg_attr(not(feature = "burn-experimental"), value(skip))]
    Burn,
    Candle,
    Cpu,
    Cuda,
    LlamaHighlevel,
    LlamaLegacy,
    Metal,
    Mlx,
    Onnx,
    Rocm,
    Transformers,
    Vllm,
    Vulkan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum KvCacheTypeArg {
    F16,
    F32,
    Q8_0,
}

impl From<KvCacheTypeArg> for LlamaKvCacheType {
    fn from(value: KvCacheTypeArg) -> Self {
        match value {
            KvCacheTypeArg::F16 => Self::F16,
            KvCacheTypeArg::F32 => Self::F32,
            KvCacheTypeArg::Q8_0 => Self::Q8_0,
        }
    }
}

#[derive(Debug, Clone, Args, Default)]
pub struct LlamaRuntimeArgs {
    #[arg(
        long,
        global = true,
        env = "WERK_LLAMA_CTX",
        help = "llama.cpp context size; 0 uses the model default"
    )]
    pub ctx_size: Option<usize>,

    #[arg(
        long,
        global = true,
        env = "WERK_LLAMA_BATCH",
        help = "llama.cpp logical prompt batch size"
    )]
    pub batch_size: Option<u32>,

    #[arg(
        long,
        global = true,
        env = "WERK_LLAMA_UBATCH",
        help = "llama.cpp physical compute micro-batch size"
    )]
    pub ubatch_size: Option<u32>,

    #[arg(
        long,
        global = true,
        env = "WERK_LLAMA_GPU_LAYERS",
        help = "llama.cpp GPU layers; high values mean all layers"
    )]
    pub gpu_layers: Option<i32>,

    #[arg(
        long,
        global = true,
        env = "WERK_LLAMA_MAIN_GPU",
        help = "llama.cpp main GPU index"
    )]
    pub main_gpu: Option<i32>,

    #[arg(
        long,
        global = true,
        value_enum,
        env = "WERK_LLAMA_KV_CACHE_TYPE",
        help = "llama.cpp KV cache type: f16, f32, or q8-0"
    )]
    pub kv_cache_type: Option<KvCacheTypeArg>,

    #[arg(
        long,
        global = true,
        env = "WERK_LLAMA_FLASH_ATTN",
        num_args = 0..=1,
        default_missing_value = "true",
        value_parser = clap::value_parser!(bool),
        help = "Request llama.cpp flash attention when the native runtime exposes it"
    )]
    pub flash_attn: Option<bool>,

    #[arg(
        long,
        global = true,
        env = "WERK_LLAMA_KV_OFFLOAD",
        num_args = 0..=1,
        default_missing_value = "true",
        value_parser = clap::value_parser!(bool),
        help = "Control llama.cpp K/Q/V and KV-cache GPU offload"
    )]
    pub kv_offload: Option<bool>,

    #[arg(
        long,
        global = true,
        env = "WERK_LLAMA_WARMUP_TOKENS",
        help = "Synthetic tokens decoded when creating a llama.cpp context; 0 disables prewarm"
    )]
    pub warmup_tokens: Option<usize>,

    #[arg(
        long,
        global = true,
        env = "WERK_LLAMA_THREADS",
        help = "llama.cpp generation CPU helper threads"
    )]
    pub threads: Option<u32>,

    #[arg(
        long,
        global = true,
        env = "WERK_LLAMA_THREADS_BATCH",
        help = "llama.cpp prompt-eval CPU helper threads"
    )]
    pub threads_batch: Option<u32>,
}

impl LlamaRuntimeArgs {
    fn to_options(&self) -> LlamaRuntimeOptions {
        LlamaRuntimeOptions {
            ctx_size: self.ctx_size,
            batch_size: self.batch_size.map(|value| value as usize),
            ubatch_size: self.ubatch_size,
            gpu_layers: self.gpu_layers,
            main_gpu: self.main_gpu,
            kv_cache_type: self.kv_cache_type.map(Into::into),
            flash_attn: self.flash_attn,
            kv_offload: self.kv_offload,
            warmup_tokens: self.warmup_tokens,
            threads: self.threads,
            threads_batch: self.threads_batch,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
pub enum BenchCompareArg {
    None,
    Legacy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BackendInstallArg {
    LlamaCuda,
    LlamaRocm,
    LlamaVulkan,
    LlamaMetal,
    LlamaCpu,
    OnnxCuda,
    OnnxRocm,
    OnnxCpu,
    #[value(name = "vllm")]
    Vllm,
}

impl BackendInstallArg {
    fn mode(self) -> Option<LlamaCppMode> {
        match self {
            Self::LlamaCuda => Some(LlamaCppMode::Cuda),
            Self::LlamaRocm => Some(LlamaCppMode::Rocm),
            Self::LlamaVulkan => Some(LlamaCppMode::Vulkan),
            Self::LlamaMetal => Some(LlamaCppMode::Metal),
            Self::LlamaCpu => Some(LlamaCppMode::Cpu),
            Self::OnnxCuda | Self::OnnxRocm | Self::OnnxCpu | Self::Vllm => None,
        }
    }

    fn onnx_mode(self) -> Option<OnnxRuntimeMode> {
        match self {
            Self::OnnxCuda => Some(OnnxRuntimeMode::Cuda),
            Self::OnnxRocm => Some(OnnxRuntimeMode::Rocm),
            Self::OnnxCpu => Some(OnnxRuntimeMode::Cpu),
            _ => None,
        }
    }
}

impl From<StreamGranularityArg> for StreamGranularity {
    fn from(value: StreamGranularityArg) -> Self {
        match value {
            StreamGranularityArg::Token => Self::Token,
            StreamGranularityArg::Chunk => Self::Chunk,
        }
    }
}

impl From<DeviceArg> for CandleDeviceMode {
    fn from(value: DeviceArg) -> Self {
        match value {
            DeviceArg::Auto => Self::Auto,
            DeviceArg::Cpu => Self::Cpu,
            DeviceArg::Cuda => Self::Cuda,
            DeviceArg::Metal => Self::Metal,
        }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Subcommand)]
pub enum Commands {
    #[command(about = "Start the OpenAI-compatible HTTP server")]
    Serve {
        #[arg(long, default_value = "127.0.0.1", help = "Address to bind")]
        host: String,

        #[arg(long, default_value_t = 11434, help = "Port to bind")]
        port: u16,

        #[arg(long, help = "Default model for API requests that omit model")]
        model: Option<String>,

        #[arg(
            long,
            env = "WERK_API_KEY",
            hide_env_values = true,
            conflicts_with = "api_keys",
            help = "Require OpenAI-style Authorization: Bearer <key> for /v1 requests"
        )]
        api_key: Option<String>,

        #[arg(
            long = "api-keys",
            env = "WERK_API_KEYS",
            value_name = "PATH",
            help = "Load OpenAI-style bearer keys from an API keys TOML file"
        )]
        api_keys: Option<PathBuf>,

        #[arg(
            long,
            action = ArgAction::SetTrue,
            conflicts_with_all = ["api_key", "api_keys"],
            help = "Disable API key auth for local development"
        )]
        allow_unauthenticated: bool,

        #[arg(long, help = "Print HTTP request and generation logs")]
        verbose: bool,
    },

    #[command(
        about = "Run one prompt against an installed model and print the response",
        hide = true
    )]
    Run {
        #[arg(help = "Installed model id")]
        model: String,

        #[arg(required = true, num_args = 1.., help = "Prompt text")]
        prompt: Vec<String>,

        #[arg(long, default_value_t = DEFAULT_MAX_NEW_TOKENS, help = "Maximum generated tokens")]
        max_tokens: usize,

        #[arg(long, help = "Sampling temperature")]
        temperature: Option<f64>,

        #[arg(long, help = "Nucleus sampling top-p")]
        top_p: Option<f64>,

        #[arg(long, help = "RNG seed")]
        seed: Option<u64>,

        #[arg(
            long,
            value_enum,
            help = "Override chat templating: model, generic, phi3, llama3, gemma, chatml, qwen-chatml, or none"
        )]
        chat_template: Option<ChatTemplateArg>,

        #[arg(
            long = "image",
            value_name = "PATH_OR_URL",
            help = "Attach an image for VLM-capable backends; may be repeated"
        )]
        images: Vec<String>,

        #[arg(long, help = "Print Ollama-style timing and throughput stats")]
        verbose: bool,

        #[arg(long, help = "Print backend internals and resolved runtime details")]
        debug: bool,
    },

    #[command(about = "Start an interactive terminal chat with an installed model")]
    Chat {
        #[arg(help = "Installed model id")]
        model: String,

        #[arg(
            long,
            default_value_t = DEFAULT_MAX_NEW_TOKENS,
            help = "Maximum generated tokens per turn"
        )]
        max_tokens: usize,

        #[arg(long, help = "Sampling temperature")]
        temperature: Option<f64>,

        #[arg(long, help = "Nucleus sampling top-p")]
        top_p: Option<f64>,

        #[arg(long, help = "RNG seed")]
        seed: Option<u64>,

        #[arg(
            long,
            value_enum,
            help = "Override chat templating: model, generic, phi3, llama3, gemma, chatml, qwen-chatml, or none"
        )]
        chat_template: Option<ChatTemplateArg>,

        #[arg(
            long = "no-history",
            alias = "single-turn",
            help = "Do not include previous chat turns in the next prompt"
        )]
        no_history: bool,

        #[arg(
            long = "image",
            value_name = "PATH_OR_URL",
            help = "Attach images to each turn for VLM-capable backends; may be repeated"
        )]
        images: Vec<String>,

        #[arg(
            long,
            value_enum,
            default_value_t = StreamGranularityArg::Token,
            help = "How terminal chat streams text: token or chunk"
        )]
        stream_granularity: StreamGranularityArg,

        #[arg(long, help = "Print Ollama-style timing and throughput stats")]
        verbose: bool,

        #[arg(long, help = "Print backend internals and resolved runtime details")]
        debug: bool,
    },

    #[command(about = "Generate, edit, or upscale images")]
    Image {
        #[command(subcommand)]
        command: ImageCommands,
    },

    #[command(about = "Generate, animate, transform, or upscale video")]
    Video {
        #[command(subcommand)]
        command: VideoCommands,
    },

    #[command(about = "Generate audio, synthesize speech, transcribe, or separate stems")]
    Audio {
        #[command(subcommand)]
        command: AudioCommands,
    },

    #[command(about = "Estimate whether a local or Hugging Face model is likely to fit in memory")]
    Estimate {
        #[arg(help = "Installed model id or Hugging Face repo id")]
        model: String,

        #[arg(
            long,
            value_parser = parse_inference_task,
            help = "Estimate an inference task using the canonical media request pipeline"
        )]
        task: Option<InferenceTask>,

        #[arg(long, help = "Requested output width")]
        width: Option<u32>,

        #[arg(long, help = "Requested output height")]
        height: Option<u32>,

        #[arg(long, help = "Requested video frame count")]
        frames: Option<u32>,

        #[arg(long, value_name = "SECONDS", help = "Requested output duration")]
        duration: Option<f64>,

        #[arg(long, help = "Requested batch size")]
        batch_size: Option<u32>,

        #[arg(long, help = "Requested audio sample rate")]
        sample_rate: Option<u32>,

        #[arg(long, help = "Requested audio channel count")]
        channels: Option<u16>,

        #[arg(long, help = "Requested diffusion or sampling step count")]
        steps: Option<u32>,

        #[arg(
            long,
            help = "For remote Hugging Face estimates, estimate one repository file, for example model.Q4_K_M.gguf"
        )]
        file: Option<String>,

        #[arg(long, help = "Print machine-readable estimate JSON")]
        json: bool,

        #[arg(long, help = "Print weight-file accounting details")]
        verbose: bool,
    },

    #[command(about = "Benchmark an installed model backend")]
    Bench {
        #[arg(help = "Installed model id")]
        model: String,

        #[arg(long, help = "Prompt text for the benchmark")]
        prompt: String,

        #[arg(
            long,
            alias = "tokens",
            default_value_t = 256,
            help = "Maximum generated tokens per run"
        )]
        max_tokens: usize,

        #[arg(long, default_value_t = 5, help = "Measured runs")]
        runs: usize,

        #[arg(long, default_value_t = 1, help = "Warmup runs before measurement")]
        warmups: usize,

        #[arg(
            long,
            default_value_t = 0.0,
            help = "Benchmark sampling temperature; 0 uses greedy decoding"
        )]
        temperature: f64,

        #[arg(long, help = "Benchmark nucleus sampling top-p")]
        top_p: Option<f64>,

        #[arg(long, default_value_t = 42, help = "Benchmark RNG seed")]
        seed: u64,

        #[arg(long, value_enum, default_value_t = BenchCompareArg::None, help = "Also benchmark another backend family")]
        compare: BenchCompareArg,

        #[arg(long, help = "Print resolved llama.cpp runtime settings")]
        print_native_info: bool,

        #[arg(long, help = "Print machine-readable benchmark JSON")]
        json: bool,

        #[arg(long, help = "Print backend internals during benchmark runs")]
        debug: bool,
    },

    #[command(about = "Inspect Werk runtime diagnostics")]
    Doctor {
        #[command(subcommand)]
        command: Option<DoctorCommands>,

        #[arg(
            long,
            value_parser = parse_inference_task,
            help = "Limit media diagnostics to an inference task"
        )]
        task: Option<InferenceTask>,

        #[arg(long, help = "Limit diagnostics to a runtime name")]
        runtime: Option<String>,

        #[arg(long, help = "Inspect routing for an installed model")]
        model: Option<String>,
    },

    #[command(about = "Manage local runtime backends")]
    Backend {
        #[command(subcommand)]
        command: BackendCommands,
    },

    #[command(about = "Manage optimized runtime artifacts for installed models")]
    Artifacts {
        #[command(subcommand)]
        command: ArtifactCommands,
    },

    #[command(about = "Manage external service authentication")]
    Auth {
        #[command(subcommand)]
        command: AuthCommands,
    },

    #[command(about = "Copy a local model file or directory into the managed model store")]
    Import {
        #[arg(help = "Model file or directory to copy")]
        path: PathBuf,

        #[arg(long, help = "Installed model id")]
        name: String,
    },

    #[command(about = "Pull a Hugging Face repository into the managed model store")]
    Pull {
        #[arg(help = "Hugging Face repo id, for example org/model")]
        repo: String,

        #[arg(long, help = "Installed model id; defaults to the repo id")]
        name: Option<String>,

        #[arg(
            long,
            help = "Download one repository file, for example model.Q4_K_M.gguf"
        )]
        file: Option<String>,
    },

    #[command(
        about = "Remove an installed model from the managed model store",
        alias = "rm",
        alias = "delete"
    )]
    Remove {
        #[arg(help = "Installed model id")]
        id: String,
    },

    #[command(about = "List installed models")]
    List {
        #[arg(long, value_parser = parse_inference_task, help = "Filter by supported task")]
        task: Option<InferenceTask>,

        #[arg(
            long = "input-modality",
            value_parser = parse_input_modality,
            help = "Filter by input modality"
        )]
        input: Option<InputModality>,

        #[arg(
            long = "output-modality",
            value_parser = parse_output_modality,
            help = "Filter by output modality"
        )]
        output: Option<OutputModality>,

        #[arg(long, help = "Filter by model family")]
        family: Option<String>,

        #[arg(long, value_parser = parse_repository_layout, help = "Filter by repository layout")]
        layout: Option<RepositoryLayout>,

        #[arg(long, help = "Print manifests as JSON")]
        json: bool,
    },

    #[command(about = "Describe task parameters, defaults, and resolution sources")]
    Parameters {
        #[arg(value_name = "MODEL", help = "Installed model id")]
        model: Option<String>,

        #[arg(
            long,
            value_parser = parse_inference_task,
            help = "Inference task whose parameter contract should be shown"
        )]
        task: Option<InferenceTask>,

        #[arg(long, help = "Print machine-readable parameter information")]
        json: bool,

        #[arg(long, help = "Print a canonical request example")]
        example: bool,

        #[arg(long, help = "Resolve and include parameter provenance")]
        sources: bool,
    },

    #[command(about = "Print a model manifest as JSON")]
    Inspect {
        #[arg(help = "Installed model id")]
        id: String,
    },

    #[command(about = "Select which tracked model file an installed manifest uses")]
    SelectFile {
        #[arg(help = "Installed model id")]
        id: String,

        #[arg(help = "Model file path, for example model.Q4_K_M.gguf or files/model.Q4_K_M.gguf")]
        file: String,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum DoctorCommands {
    #[command(about = "Print llama.cpp performance/runtime diagnostics for an installed model")]
    Perf {
        #[arg(help = "Installed model id")]
        model: String,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum BackendCommands {
    #[command(about = "Install a managed llama.cpp server backend")]
    Install {
        #[arg(
            value_enum,
            value_name = "BACKEND",
            help = "Backend to install, for example llama-cuda, onnx-cuda, or onnx-cpu"
        )]
        target: BackendInstallArg,
    },

    #[command(about = "List discovered llama-server backends")]
    List,

    #[command(about = "Check tools required for managed backend builds")]
    Doctor {
        #[arg(
            long,
            help = "Print detailed backend discovery paths and rejection reasons"
        )]
        debug: bool,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum ArtifactCommands {
    #[command(about = "Build missing optimized artifacts for an installed model")]
    Build {
        #[arg(help = "Installed model id")]
        model: String,
    },

    #[command(about = "List optimized artifacts for an installed model")]
    List {
        #[arg(help = "Installed model id")]
        model: String,
    },

    #[command(about = "Rebuild optimized artifacts for an installed model")]
    Rebuild {
        #[arg(help = "Installed model id")]
        model: String,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum AuthCommands {
    #[command(
        name = "huggingface",
        about = "Manage Hugging Face authentication",
        alias = "hf"
    )]
    HuggingFace {
        #[command(subcommand)]
        command: HuggingFaceAuthCommands,
    },

    #[command(
        name = "api-key",
        about = "Generate API keys for the OpenAI-compatible server",
        alias = "api-keys"
    )]
    ApiKey {
        #[command(subcommand)]
        command: ApiKeyAuthCommands,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum ApiKeyAuthCommands {
    #[command(about = "Create an API keys TOML file for `werk serve`")]
    Generate {
        #[arg(
            long,
            value_name = "PATH",
            help = "API keys file to create; defaults to ~/.config/werk1112/api-keys.toml"
        )]
        path: Option<PathBuf>,

        #[arg(
            long,
            default_value = "default",
            help = "Human-readable name stored next to the generated key"
        )]
        name: String,

        #[arg(long, help = "Overwrite an existing API keys file")]
        force: bool,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum HuggingFaceAuthCommands {
    #[command(about = "Store a Hugging Face access token for gated model pulls")]
    Login {
        #[arg(
            long,
            help = "Hugging Face access token; omit to enter it interactively"
        )]
        token: Option<String>,
    },

    #[command(about = "Show whether Werk can find a Hugging Face token")]
    Status,

    #[command(about = "Remove the Werk-stored Hugging Face token")]
    Logout,
}

pub async fn run_from_env() -> Result<()> {
    run(Cli::parse()).await
}

pub async fn run(cli: Cli) -> Result<()> {
    let model_home = cli.model_home;
    let device_override = cli.device;
    let backend_override = cli.backend;
    let selection_options = SelectionOptions::from_cli(
        backend_override,
        cli.auto_install_backends,
        cli.no_auto_install_backends,
    );
    let llama_options = cli.llama.to_options();
    let command = cli.command.unwrap_or(Commands::Serve {
        host: "127.0.0.1".to_string(),
        port: 11434,
        model: None,
        api_key: None,
        api_keys: None,
        allow_unauthenticated: false,
        verbose: false,
    });
    let selection_options =
        selection_options.with_backend_install_output(command_backend_install_verbose(&command));

    if should_print_startup_banner(&command) {
        print_banner();
    }

    match command {
        Commands::Serve {
            host,
            port,
            model,
            api_key,
            api_keys,
            allow_unauthenticated,
            verbose,
        } => {
            let store = ModelStore::resolve(model_home)?;
            store.ensure()?;
            let api_keys = resolve_api_keys(api_key, api_keys, allow_unauthenticated)?;
            let backend_choice = resolve_backend(backend_override, device_override)?;
            let ip: IpAddr = host.parse()?;
            let addr = SocketAddr::new(ip, port);
            let backend = build_generation_backend(
                store.clone(),
                backend_choice,
                llama_options.clone(),
                selection_options,
            )?;
            let prompt_options_resolver = {
                Arc::new(
                    move |store: &ModelStore, manifest: &ModelManifest, has_images: bool| {
                        let selected_backend = selected_backend_for_request(
                            store,
                            backend_choice,
                            manifest,
                            has_images,
                            selection_options,
                        )?;
                        if verbose {
                            eprintln!(
                                "[werk serve] route model={} backend={}",
                                manifest.id,
                                verbose_backend_label(selected_backend)
                            );
                        }
                        Ok(chat_template_options_for_backend(
                            manifest,
                            selected_backend,
                            None,
                        ))
                    },
                )
            };
            if let Some(model) = model.as_deref() {
                let manifest = store.get(model)?;
                with_terminal_spinner(
                    terminal_spinner_enabled(false),
                    format!("Loading default model '{model}'..."),
                    || backend.prepare(&manifest),
                )?;
                println!("Default model available: {model}");
            }
            let api_state = ApiState::new_with_default_model_prompt_options_and_verbose(
                store,
                backend,
                model,
                Some(prompt_options_resolver),
                verbose,
            )
            .with_api_keys(api_keys);
            serve(addr, api_state).await
        }
        Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            top_p,
            seed,
            chat_template,
            images,
            verbose,
            debug,
        } => {
            let prompt = prompt.join(" ");
            let store = ModelStore::resolve(model_home)?;
            let backend_choice = resolve_backend(backend_override, device_override)?;
            let manifest = store.get(&model)?;
            let selected_backend = selected_backend_for_request(
                &store,
                backend_choice,
                &manifest,
                !images.is_empty(),
                selection_options,
            )?;
            print_routing_debug(
                &store,
                backend_override,
                backend_choice,
                &manifest,
                !images.is_empty(),
                selected_backend,
                debug,
            );
            print_verbose_fallback_note(
                &store,
                backend_choice,
                &manifest,
                !images.is_empty(),
                selected_backend,
                verbose,
            );
            let backend_to_build =
                backend_to_build_for_request(backend_choice, selected_backend, &manifest);
            let backend = build_generation_backend(
                store,
                backend_to_build,
                llama_options.clone(),
                selection_options,
            )?;
            let messages = vec![ChatMessage {
                role: "user".to_string(),
                content: Some(MessageContent::Text(prompt)),
                name: None,
            }];
            let prompt = prompt_for_backend(&manifest, &messages, selected_backend, chat_template);
            let prompt_diagnostics = prompt_diagnostics(&prompt, messages.len(), None);
            let request_messages = generation_request_messages(&prompt, &messages);
            let request = GenerateRequest {
                prompt: prompt.prompt,
                messages: request_messages,
                image_urls: images,
                max_tokens,
                temperature,
                top_p,
                stop: prompt.stop,
                seed,
                stream_granularity: StreamGranularity::Chunk,
                verbose,
                debug,
            };
            let response = with_terminal_spinner(
                terminal_spinner_enabled(debug),
                format!("Running model '{}'...", manifest.id),
                || backend.generate(&manifest, request),
            )?;
            println!("{}", response.text.trim());
            io::stdout().flush()?;
            if verbose {
                let mut stderr = io::stderr().lock();
                writeln!(stderr)?;
                write_verbose_stats(
                    &mut stderr,
                    Some(verbose_backend_label(selected_backend)),
                    response.prompt_tokens,
                    response.completion_tokens,
                    &response.finish_reason,
                    response.timings,
                    &merged_diagnostics(&prompt_diagnostics, &response.backend_diagnostics),
                )?;
            }
            Ok(())
        }
        Commands::Chat {
            model,
            max_tokens,
            temperature,
            top_p,
            seed,
            chat_template,
            no_history,
            images,
            stream_granularity,
            verbose,
            debug,
        } => {
            let store = ModelStore::resolve(model_home)?;
            let backend_choice = resolve_backend(backend_override, device_override)?;
            let manifest = store.get(&model)?;
            let selected_backend = selected_backend_for_request(
                &store,
                backend_choice,
                &manifest,
                !images.is_empty(),
                selection_options,
            )?;
            print_routing_debug(
                &store,
                backend_override,
                backend_choice,
                &manifest,
                !images.is_empty(),
                selected_backend,
                debug,
            );
            print_verbose_fallback_note(
                &store,
                backend_choice,
                &manifest,
                !images.is_empty(),
                selected_backend,
                verbose,
            );
            let backend_to_build =
                backend_to_build_for_request(backend_choice, selected_backend, &manifest);
            let backend = build_generation_backend(
                store,
                backend_to_build,
                llama_options.clone(),
                selection_options,
            )?;
            chat_loop(
                backend,
                manifest,
                selected_backend,
                max_tokens,
                temperature,
                top_p,
                seed,
                !no_history,
                chat_template,
                images,
                stream_granularity.into(),
                verbose,
                debug,
                terminal_spinner_enabled(debug),
            )
            .await
        }
        Commands::Image { command } => {
            let store = ModelStore::resolve(model_home)?;
            run_image_command(&store, backend_override, device_override, command)
        }
        Commands::Video { command } => {
            let store = ModelStore::resolve(model_home)?;
            run_video_command(&store, backend_override, device_override, command)
        }
        Commands::Audio { command } => {
            let store = ModelStore::resolve(model_home)?;
            run_audio_command(&store, backend_override, device_override, command)
        }
        Commands::Estimate {
            model,
            task,
            width,
            height,
            frames,
            duration,
            batch_size,
            sample_rate,
            channels,
            steps,
            file,
            json,
            verbose,
        } => {
            let store = ModelStore::resolve(model_home)?;
            if let Some(task) = task {
                if file.is_some() {
                    bail!("--file cannot be combined with --task workload estimates");
                }
                let service = InferenceService::new(store);
                let request = workload_estimate_request(
                    model,
                    task,
                    backend_override,
                    device_override,
                    WorkloadEstimateArgs {
                        width,
                        height,
                        frames,
                        duration,
                        batch_size,
                        sample_rate,
                        channels,
                        steps,
                    },
                );
                let _effective = service.resolve(request.clone())?;
                let report = service.estimate(request)?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&report)?);
                } else {
                    print_workload_estimate(&report, verbose);
                }
            } else {
                let report = estimate_model_or_huggingface(
                    &store,
                    &model,
                    file.as_deref(),
                    detect_system_memory(),
                )?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&report)?);
                } else {
                    print_estimate_report(&report, verbose);
                }
            }
            Ok(())
        }
        Commands::Bench {
            model,
            prompt,
            max_tokens,
            runs,
            warmups,
            temperature,
            top_p,
            seed,
            compare,
            print_native_info,
            json,
            debug,
        } => {
            let store = ModelStore::resolve(model_home)?;
            let backend_choice = resolve_backend(backend_override, device_override)?;
            let manifest = store.get(&model)?;
            let report = bench_model(
                store,
                manifest,
                backend_choice,
                llama_options.clone(),
                prompt,
                max_tokens,
                runs,
                warmups,
                temperature,
                top_p,
                seed,
                compare,
                debug,
                selection_options,
            )?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_bench_report(&report, print_native_info);
            }
            Ok(())
        }
        Commands::Doctor {
            command,
            task,
            runtime,
            model,
        } => {
            let store = ModelStore::resolve(model_home)?;
            match command {
                Some(DoctorCommands::Perf { model }) => {
                    let backend_choice = resolve_backend(backend_override, device_override)?;
                    let manifest = store.get(&model)?;
                    print_perf_doctor(
                        &store,
                        &manifest,
                        backend_choice,
                        &llama_options,
                        selection_options,
                    )
                }
                None => print_inference_doctor(
                    &store,
                    backend_override,
                    device_override,
                    task,
                    runtime.as_deref(),
                    model.as_deref(),
                ),
            }
        }
        Commands::Backend { command } => {
            let store = ModelStore::resolve(model_home)?;
            match command {
                BackendCommands::Install { target } => {
                    if let Some(mode) = target.mode() {
                        let executable = install_managed_llama_server(&store, mode)?;
                        println!(
                            "Installed {} llama-server: {}",
                            display_llama_mode(mode),
                            executable.display()
                        );
                    } else if let Some(mode) = target.onnx_mode() {
                        let executable = install_managed_onnx_runtime(&store, mode)?;
                        println!(
                            "Installed {} runner: {}",
                            mode.display(),
                            executable.display()
                        );
                    } else if target == BackendInstallArg::Vllm {
                        let python = install_managed_vllm(&store)?;
                        println!("Installed vLLM backend: {}", python.display());
                    }
                    Ok(())
                }
                BackendCommands::List => {
                    print_backend_list(&store);
                    Ok(())
                }
                BackendCommands::Doctor { debug } => {
                    print_backend_doctor(&store, debug);
                    Ok(())
                }
            }
        }
        Commands::Artifacts { command } => {
            let store = ModelStore::resolve(model_home)?;
            match command {
                ArtifactCommands::Build { model } => {
                    let artifact = store.build_onnx_artifact(&model, false)?;
                    print_artifact_result("Built", &model, &artifact);
                    Ok(())
                }
                ArtifactCommands::List { model } => {
                    let artifacts = store.list_artifacts(&model)?;
                    print_artifact_list(&model, &artifacts);
                    Ok(())
                }
                ArtifactCommands::Rebuild { model } => {
                    let artifact = store.build_onnx_artifact(&model, true)?;
                    print_artifact_result("Rebuilt", &model, &artifact);
                    Ok(())
                }
            }
        }
        Commands::Auth { command } => match command {
            AuthCommands::HuggingFace { command } => {
                let store = ModelStore::resolve(model_home)?;
                match command {
                    HuggingFaceAuthCommands::Login { token } => {
                        let token = match token {
                            Some(token) => token,
                            None => prompt_huggingface_token()?,
                        };
                        let path = store.save_huggingface_token(&token)?;
                        println!("Saved Hugging Face token for Werk: {}", path.display());
                        println!(
                            "For gated models, also accept the model conditions on Hugging Face before pulling."
                        );
                        Ok(())
                    }
                    HuggingFaceAuthCommands::Status => {
                        let status = store.huggingface_auth_status()?;
                        if let Some(source) = status.source {
                            println!("Hugging Face token: configured ({source})");
                        } else {
                            println!(
                                "Hugging Face token: not configured. Run `werk auth huggingface login` or set HF_TOKEN."
                            );
                        }
                        Ok(())
                    }
                    HuggingFaceAuthCommands::Logout => {
                        if store.delete_huggingface_token()? {
                            println!("Removed Werk-stored Hugging Face token.");
                        } else {
                            println!("No Werk-stored Hugging Face token was found.");
                        }
                        Ok(())
                    }
                }
            }
            AuthCommands::ApiKey { command } => match command {
                ApiKeyAuthCommands::Generate { path, name, force } => {
                    let path = path
                        .map(Ok)
                        .unwrap_or_else(api_keys::default_api_keys_path)?;
                    let entry = api_keys::write_api_keys_file(&path, &name, force)?;
                    println!("Created Werk API keys file: {}", path.display());
                    println!("Name: {}", entry.name);
                    println!("API key: {}", entry.key);
                    println!(
                        "Use this value as the OpenAI API key, sent as Authorization: Bearer <key>."
                    );
                    Ok(())
                }
            },
        },
        Commands::Import { path, name } => {
            let store = ModelStore::resolve(model_home)?;
            let manifest = store.import_path(&path, &name)?;
            print_manifest_summary("Imported", &manifest);
            Ok(())
        }
        Commands::Pull { repo, name, file } => {
            let store = ModelStore::resolve(model_home)?;
            let progress = pull_progress_bar();
            let manifest = store.pull_from_huggingface_with_progress(
                &repo,
                name.as_deref(),
                file.as_deref(),
                |event| {
                    update_pull_progress(&progress, event);
                },
            )?;
            progress.finish_and_clear();
            print_manifest_summary("Pulled", &manifest);
            Ok(())
        }
        Commands::Remove { id } => {
            let store = ModelStore::resolve(model_home)?;
            let manifest = store.remove(&id)?;
            println!(
                "Removed {} ({:?}) from {}",
                manifest.id,
                manifest.format,
                store.model_dir(&manifest.id).display()
            );
            Ok(())
        }
        Commands::List {
            task,
            input,
            output,
            family,
            layout,
            json,
        } => {
            let store = ModelStore::resolve(model_home)?;
            let backend_filter = (backend_override != BackendArg::Auto)
                .then(|| requested_backend_label(backend_override));
            let manifests = store
                .list()?
                .into_iter()
                .filter(|manifest| {
                    manifest_matches_list_filters(
                        manifest,
                        task,
                        input,
                        output,
                        family.as_deref(),
                        layout,
                        backend_filter,
                    )
                })
                .collect::<Vec<_>>();
            if json {
                println!("{}", serde_json::to_string_pretty(&manifests)?);
                return Ok(());
            }
            if manifests.is_empty() {
                println!("No matching models installed in {}", store.home().display());
            } else {
                println!(
                    "{:<26} {:<14} {:<14} {:<18} TASKS",
                    "MODEL", "LAYOUT", "FAMILY", "ARCHITECTURE"
                );
                for manifest in manifests {
                    println!(
                        "{:<26} {:<14} {:<14} {:<18} {}",
                        manifest.id,
                        manifest.metadata.repository_layout,
                        manifest.metadata.family.as_deref().unwrap_or("-"),
                        manifest.architecture.unwrap_or_else(|| "-".to_string()),
                        join_display(&manifest.metadata.tasks)
                    );
                }
            }
            Ok(())
        }
        Commands::Parameters {
            model,
            task,
            json,
            example,
            sources,
        } => {
            let store = ModelStore::resolve(model_home)?;
            print_parameters(
                &store,
                backend_override,
                device_override,
                model.as_deref(),
                task,
                json,
                example,
                sources,
            )
        }
        Commands::Inspect { id } => {
            let store = ModelStore::resolve(model_home)?;
            let manifest = store.get(&id)?;
            let mut value = serde_json::to_value(&manifest)?;
            if let Some(fields) = value.as_object_mut() {
                fields.insert(
                    "host_resources".to_string(),
                    serde_json::to_value(crate::inference_service::detect_host_resources())?,
                );
            }
            println!("{}", serde_json::to_string_pretty(&value)?);
            Ok(())
        }
        Commands::SelectFile { id, file } => {
            let store = ModelStore::resolve(model_home)?;
            let manifest = store.set_model_file(&id, &file)?;
            println!(
                "Selected {} for {}",
                manifest.model_path.as_deref().unwrap_or("unknown"),
                manifest.id
            );
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct WorkloadEstimateArgs {
    width: Option<u32>,
    height: Option<u32>,
    frames: Option<u32>,
    duration: Option<f64>,
    batch_size: Option<u32>,
    sample_rate: Option<u32>,
    channels: Option<u16>,
    steps: Option<u32>,
}

fn run_image_command(
    store: &ModelStore,
    backend: BackendArg,
    device: Option<DeviceArg>,
    command: ImageCommands,
) -> Result<()> {
    match command {
        ImageCommands::Generate(args) => {
            let prompt = resolve_primary_text(
                args.prompt.prompt.as_deref(),
                args.prompt.prompt_file.as_deref(),
                true,
                "prompt",
            )?;
            let negative_prompt = resolve_optional_text(
                args.prompt.negative_prompt.as_deref(),
                args.prompt.negative_prompt_file.as_deref(),
            )?;
            if args.conditioning.mask.is_some() && args.conditioning.init_image.is_none() {
                bail!("--mask requires --init-image for image generation");
            }
            let task = if args.conditioning.mask.is_some() {
                InferenceTask::ImageInpainting
            } else if args.conditioning.init_image.is_some() {
                InferenceTask::ImageEditing
            } else {
                InferenceTask::ImageGeneration
            };
            let mut inputs = Vec::new();
            if let Some(initial_image) = args.conditioning.init_image.as_deref() {
                inputs.push(string_input(InputModality::Image, "image", initial_image));
            }
            if let Some(mask) = args.conditioning.mask.as_deref() {
                inputs.push(path_input(InputModality::Image, "mask_image", mask));
            }
            execute_media_args(
                store,
                backend,
                device,
                &args.model,
                task,
                prompt,
                negative_prompt,
                inputs,
                &args.routing,
                &args,
            )
        }
        ImageCommands::Edit(args) => {
            let prompt = resolve_primary_text(
                args.prompt.prompt.as_deref(),
                args.prompt.prompt_file.as_deref(),
                true,
                "prompt",
            )?;
            let negative_prompt = resolve_optional_text(
                args.prompt.negative_prompt.as_deref(),
                args.prompt.negative_prompt_file.as_deref(),
            )?;
            let task = if args.conditioning.mask.is_some() {
                InferenceTask::ImageInpainting
            } else {
                InferenceTask::ImageEditing
            };
            let mut inputs = vec![path_input(InputModality::Image, "image", &args.image)];
            if let Some(mask) = args.conditioning.mask.as_deref() {
                inputs.push(path_input(InputModality::Image, "mask_image", mask));
            }
            execute_media_args(
                store,
                backend,
                device,
                &args.model,
                task,
                prompt,
                negative_prompt,
                inputs,
                &args.routing,
                &args,
            )
        }
        ImageCommands::Upscale(args) => {
            let prompt = resolve_primary_text(
                args.prompt.prompt.as_deref(),
                args.prompt.prompt_file.as_deref(),
                false,
                "prompt",
            )?;
            let negative_prompt = resolve_optional_text(
                args.prompt.negative_prompt.as_deref(),
                args.prompt.negative_prompt_file.as_deref(),
            )?;
            execute_media_args(
                store,
                backend,
                device,
                &args.model,
                InferenceTask::ImageUpscaling,
                prompt,
                negative_prompt,
                vec![path_input(InputModality::Image, "image", &args.image)],
                &args.routing,
                &args,
            )
        }
    }
}

fn run_video_command(
    store: &ModelStore,
    backend: BackendArg,
    device: Option<DeviceArg>,
    command: VideoCommands,
) -> Result<()> {
    match command {
        VideoCommands::Generate(args) => {
            let task = if args.initial_image.is_some() {
                InferenceTask::ImageToVideo
            } else {
                InferenceTask::VideoGeneration
            };
            let prompt = resolve_primary_text(
                args.prompt.prompt.as_deref(),
                args.prompt.prompt_file.as_deref(),
                true,
                "prompt",
            )?;
            let negative_prompt = resolve_optional_text(
                args.prompt.negative_prompt.as_deref(),
                args.prompt.negative_prompt_file.as_deref(),
            )?;
            let inputs = args
                .initial_image
                .as_deref()
                .map(|path| vec![path_input(InputModality::Image, "initial_image", path)])
                .unwrap_or_default();
            execute_media_args(
                store,
                backend,
                device,
                &args.model,
                task,
                prompt,
                negative_prompt,
                inputs,
                &args.routing,
                &args,
            )
        }
        VideoCommands::Animate(args) => {
            let prompt = resolve_primary_text(
                args.prompt.prompt.as_deref(),
                args.prompt.prompt_file.as_deref(),
                true,
                "prompt",
            )?;
            let negative_prompt = resolve_optional_text(
                args.prompt.negative_prompt.as_deref(),
                args.prompt.negative_prompt_file.as_deref(),
            )?;
            execute_media_args(
                store,
                backend,
                device,
                &args.model,
                InferenceTask::ImageToVideo,
                prompt,
                negative_prompt,
                vec![path_input(
                    InputModality::Image,
                    "initial_image",
                    &args.image,
                )],
                &args.routing,
                &args,
            )
        }
        VideoCommands::Transform(args) => {
            let prompt = resolve_primary_text(
                args.prompt.prompt.as_deref(),
                args.prompt.prompt_file.as_deref(),
                true,
                "prompt",
            )?;
            let negative_prompt = resolve_optional_text(
                args.prompt.negative_prompt.as_deref(),
                args.prompt.negative_prompt_file.as_deref(),
            )?;
            execute_media_args(
                store,
                backend,
                device,
                &args.model,
                InferenceTask::VideoToVideo,
                prompt,
                negative_prompt,
                vec![path_input(
                    InputModality::Video,
                    "source_video",
                    &args.video,
                )],
                &args.routing,
                &args,
            )
        }
        VideoCommands::Upscale(args) => execute_media_args(
            store,
            backend,
            device,
            &args.model,
            InferenceTask::VideoUpscaling,
            None,
            None,
            vec![path_input(
                InputModality::Video,
                "source_video",
                &args.video,
            )],
            &args.routing,
            &args,
        ),
    }
}

fn run_audio_command(
    store: &ModelStore,
    backend: BackendArg,
    device: Option<DeviceArg>,
    command: AudioCommands,
) -> Result<()> {
    match command {
        AudioCommands::Generate(mut args) => {
            let task = store
                .get(&args.model)
                .map(|manifest| {
                    if args.conditioning.source_audio.is_some()
                        && (args.conditioning.continuation_start.is_some()
                            || args.conditioning.continuation_duration.is_some())
                        && manifest.supports_task(InferenceTask::SongContinuation)
                    {
                        InferenceTask::SongContinuation
                    } else if args.conditioning.source_audio.is_some()
                        && manifest.supports_task(InferenceTask::SongVariation)
                    {
                        InferenceTask::SongVariation
                    } else if args.conditioning.source_audio.is_some()
                        && manifest.supports_task(InferenceTask::SongContinuation)
                    {
                        InferenceTask::SongContinuation
                    } else if manifest.supports_task(InferenceTask::MusicGeneration) {
                        InferenceTask::MusicGeneration
                    } else {
                        InferenceTask::AudioGeneration
                    }
                })
                .unwrap_or(InferenceTask::AudioGeneration);
            let lyrics = resolve_optional_text(
                args.lyrics.lyrics.as_deref(),
                args.lyrics.lyrics_file.as_deref(),
            )?;
            let prompt = if lyrics.is_some() {
                resolve_optional_text(
                    args.prompt.prompt.as_deref(),
                    args.prompt.prompt_file.as_deref(),
                )?
            } else {
                resolve_primary_text(
                    args.prompt.prompt.as_deref(),
                    args.prompt.prompt_file.as_deref(),
                    false,
                    "prompt",
                )?
            };
            args.lyrics.lyrics = lyrics.clone();
            args.lyrics.lyrics_file = None;
            let prompt = match (prompt, lyrics.as_deref()) {
                (Some(prompt), _) => Some(prompt),
                // The canonical lyrics value remains available as
                // `audio.lyrics`. Reuse its text as the required generative
                // prompt without decorating it; the companion can then merge
                // prompt and lyrics idempotently.
                (None, Some(lyrics)) => Some(lyrics.to_string()),
                (None, None) if task.requires_prompt() => {
                    resolve_primary_text(None, None, true, "prompt")?
                }
                (None, None) => None,
            };
            let negative_prompt = resolve_optional_text(
                args.prompt.negative_prompt.as_deref(),
                args.prompt.negative_prompt_file.as_deref(),
            )?;
            let mut inputs = Vec::new();
            for (role, path) in [
                ("input_audio", args.conditioning.source_audio.as_deref()),
                (
                    "reference_audio",
                    args.conditioning.reference_audio.as_deref(),
                ),
                (
                    "instrumental_audio",
                    args.conditioning.instrumental_audio.as_deref(),
                ),
                ("vocal_audio", args.conditioning.vocal_audio.as_deref()),
                ("melody_audio", args.conditioning.melody_audio.as_deref()),
                ("rhythm_audio", args.conditioning.rhythm_audio.as_deref()),
                ("chord_audio", args.conditioning.chord_audio.as_deref()),
            ] {
                if let Some(path) = path {
                    inputs.push(path_input(InputModality::Audio, role, path));
                }
            }
            execute_media_args(
                store,
                backend,
                device,
                &args.model,
                task,
                prompt,
                negative_prompt,
                inputs,
                &args.routing,
                &args,
            )
        }
        AudioCommands::Speak(args) => {
            let text = resolve_primary_text(
                args.text.text.as_deref(),
                args.text.text_file.as_deref(),
                true,
                "text",
            )?;
            execute_media_args(
                store,
                backend,
                device,
                &args.model,
                InferenceTask::TextToSpeech,
                text,
                None,
                Vec::new(),
                &args.routing,
                &args,
            )
        }
        AudioCommands::Transcribe(args) => execute_media_args(
            store,
            backend,
            device,
            &args.model,
            InferenceTask::SpeechToText,
            None,
            None,
            vec![path_input(
                InputModality::Audio,
                "input_audio",
                &args.input_audio,
            )],
            &args.routing,
            &args,
        ),
        AudioCommands::Separate(args) => execute_media_args(
            store,
            backend,
            device,
            &args.model,
            InferenceTask::StemSeparation,
            None,
            None,
            vec![path_input(
                InputModality::Audio,
                "input_audio",
                &args.input_audio,
            )],
            &args.routing,
            &args,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_media_args<T: Serialize>(
    store: &ModelStore,
    backend: BackendArg,
    device: Option<DeviceArg>,
    model: &str,
    task: InferenceTask,
    prompt: Option<String>,
    negative_prompt: Option<String>,
    inputs: Vec<InferenceInput>,
    routing_args: &RoutingArgs,
    args: &T,
) -> Result<()> {
    let requested_output = requested_media_output(args)?;
    let mut request = InferenceRequest::new(model, task);
    request.prompt = prompt;
    request.negative_prompt = negative_prompt;
    request.inputs = inputs;
    request.parameters = media_parameters(args, routing_args, task)?;
    request.routing = media_routing(routing_args, backend, device)?;

    let result = InferenceService::new(store.clone()).execute(request)?;
    print_inference_result(&result);
    if let Some(destination) = requested_output {
        for published in publish_cli_outputs(&result, &destination)? {
            println!("saved> {}", published.display());
        }
    }
    Ok(())
}

fn requested_media_output<T: Serialize>(args: &T) -> Result<Option<PathBuf>> {
    let raw = collect_raw_overrides(args)?;
    Ok(raw.get("output").and_then(Value::as_str).map(PathBuf::from))
}

fn publish_cli_outputs(result: &InferenceResult, destination: &Path) -> Result<Vec<PathBuf>> {
    let outputs = result.outputs.iter().collect::<Vec<_>>();
    if outputs.is_empty() {
        bail!("inference completed without outputs to publish");
    }
    let destination_is_directory = destination.is_dir()
        || outputs.len() > 1
        || (!destination.exists() && destination.extension().is_none());
    if outputs.len() > 1 && destination.exists() && !destination.is_dir() {
        bail!(
            "multiple outputs require a destination directory, but {} is a file",
            destination.display()
        );
    }
    if destination_is_directory {
        fs::create_dir_all(destination).with_context(|| {
            format!(
                "failed to create output directory {}",
                destination.display()
            )
        })?;
    } else if let Some(parent) = destination.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory {}", parent.display()))?;
    }

    let mut published = Vec::with_capacity(outputs.len());
    for output in outputs {
        let source = Path::new(&output.path);
        let target = if destination_is_directory {
            let filename = source
                .file_name()
                .ok_or_else(|| anyhow!("output has no filename: {}", source.display()))?;
            destination.join(filename)
        } else {
            destination.to_path_buf()
        };
        if target.exists() {
            bail!(
                "refusing to overwrite existing output {}; choose another --output path",
                target.display()
            );
        }
        fs::copy(source, &target).with_context(|| {
            format!(
                "failed to copy managed output {} to {}",
                source.display(),
                target.display()
            )
        })?;
        published.push(target);
    }
    Ok(published)
}

fn resolve_primary_text(
    explicit: Option<&str>,
    file: Option<&Path>,
    required: bool,
    label: &str,
) -> Result<Option<String>> {
    if let Some(value) = explicit {
        let value = value.trim();
        if !value.is_empty() {
            return Ok(Some(value.to_string()));
        }
    }
    if let Some(path) = file {
        let value = fs::read_to_string(path)
            .with_context(|| format!("failed to read {} file {}", label, path.display()))?;
        let value = value.trim();
        if !value.is_empty() {
            return Ok(Some(value.to_string()));
        }
    }
    if !io::stdin().is_terminal() {
        let mut value = String::new();
        io::stdin().lock().read_to_string(&mut value)?;
        let value = value.trim();
        if !value.is_empty() {
            return Ok(Some(value.to_string()));
        }
    } else if required {
        print!("{label}> ");
        io::stdout().flush()?;
        let mut value = String::new();
        io::stdin().read_line(&mut value)?;
        let value = value.trim();
        if !value.is_empty() {
            return Ok(Some(value.to_string()));
        }
    }
    if required {
        bail!(
            "{} is required; pass --{}, --{}-file, pipe stdin, or enter it interactively",
            label,
            label,
            label
        );
    }
    Ok(None)
}

fn resolve_optional_text(explicit: Option<&str>, file: Option<&Path>) -> Result<Option<String>> {
    if let Some(value) = explicit {
        return Ok((!value.trim().is_empty()).then(|| value.trim().to_string()));
    }
    let Some(path) = file else {
        return Ok(None);
    };
    let value = fs::read_to_string(path)
        .with_context(|| format!("failed to read text file {}", path.display()))?;
    Ok((!value.trim().is_empty()).then(|| value.trim().to_string()))
}

fn path_input(modality: InputModality, role: &str, path: &Path) -> InferenceInput {
    string_input(modality, role, &path.to_string_lossy())
}

fn string_input(modality: InputModality, role: &str, value: &str) -> InferenceInput {
    let value = value.to_string();
    let (source, mime_type) = if value.starts_with("http://") || value.starts_with("https://") {
        (InferenceInputSource::Url { url: value }, None)
    } else if let Some((header, data)) = value.split_once(";base64,")
        && value.starts_with("data:")
    {
        (
            InferenceInputSource::Base64 {
                data: data.to_string(),
            },
            header.strip_prefix("data:").map(str::to_string),
        )
    } else {
        (InferenceInputSource::Path { path: value }, None)
    };
    InferenceInput {
        modality,
        role: role.to_string(),
        source,
        mime_type,
    }
}

fn media_routing(
    args: &RoutingArgs,
    backend: BackendArg,
    device: Option<DeviceArg>,
) -> Result<RoutingOverrides> {
    let backend_accelerator = match backend {
        BackendArg::Cpu => Some("cpu"),
        BackendArg::Cuda => Some("cuda"),
        BackendArg::Rocm => Some("rocm"),
        BackendArg::Metal => Some("metal"),
        _ => None,
    };
    let device_accelerator = device
        .filter(|device| *device != DeviceArg::Auto)
        .map(device_arg_label);
    if let (Some(accelerator), Some(device)) = (args.accelerator.as_deref(), device_accelerator)
        && !accelerator.eq_ignore_ascii_case(device)
    {
        bail!(
            "conflicting media accelerator selections: --accelerator {accelerator} and --device {device}"
        );
    }
    let accelerator = args
        .accelerator
        .clone()
        .or_else(|| device_accelerator.map(str::to_string))
        .or_else(|| backend_accelerator.map(str::to_string));
    let backend = if backend == BackendArg::Auto || backend_accelerator.is_some() {
        None
    } else {
        Some(requested_backend_label(backend).to_string())
    };
    Ok(RoutingOverrides {
        backend,
        accelerator,
        device: device.map(device_arg_label).map(str::to_string),
        precision: args.precision.clone(),
        quantization: args.quantization.clone(),
        profile: args.profile.clone(),
        quality: args.quality.clone(),
        performance_preference: args.performance_preference.clone(),
        fallback_policy: args.fallback_policy.clone(),
        parameter_policy: args
            .parameter_policy
            .as_deref()
            .unwrap_or("strict")
            .parse::<ParameterPolicy>()
            .map_err(anyhow::Error::msg)?,
        allow_cpu_offload: canonical_bool(args.allow_cpu_offload, args.no_allow_cpu_offload),
        allow_sequential_offload: canonical_bool(
            args.allow_sequential_offload,
            args.no_allow_sequential_offload,
        ),
        allow_component_offload: canonical_bool(
            args.allow_component_offload,
            args.no_allow_component_offload,
        ),
        allow_disk_offload: canonical_bool(args.allow_disk_offload, args.no_allow_disk_offload),
        attention_backend: args.attention_backend.clone(),
        compile: canonical_bool(args.compile, args.no_compile),
        timeout_seconds: args.timeout,
    })
}

fn canonical_bool(enabled: bool, disabled: bool) -> OverrideBool {
    match (enabled, disabled) {
        (true, false) => OverrideBool::Enabled,
        (false, true) => OverrideBool::Disabled,
        _ => OverrideBool::Inherit,
    }
}

fn device_arg_label(device: DeviceArg) -> &'static str {
    match device {
        DeviceArg::Auto => "auto",
        DeviceArg::Cpu => "cpu",
        DeviceArg::Cuda => "cuda",
        DeviceArg::Metal => "metal",
    }
}

fn media_parameters<T: Serialize>(
    args: &T,
    routing: &RoutingArgs,
    task: InferenceTask,
) -> Result<BTreeMap<String, ParameterValue>> {
    let mut raw = collect_raw_overrides(args)?;
    for path in MEDIA_TRANSPORT_FIELDS
        .iter()
        .chain(MEDIA_ROUTING_FIELDS.iter())
    {
        raw.remove(*path);
    }
    expand_structured_media_values(&mut raw)?;
    for (path, value) in parse_set_overrides(&routing.set).map_err(anyhow::Error::msg)? {
        raw.insert(path, value);
    }
    normalize_negative_bool_pairs(&mut raw);

    raw.into_iter()
        .map(|(path, value)| {
            let path = normalize_media_parameter_path(task, &path);
            Ok((path, ParameterValue::from_json(value)?))
        })
        .collect()
}

const MEDIA_TRANSPORT_FIELDS: &[&str] = &[
    "model",
    "prompt",
    "prompt_file",
    "negative_prompt",
    "negative_prompt_file",
    "text",
    "text_file",
    "lyrics_file",
    "image",
    "init_image",
    "mask",
    "video",
    "input_audio",
    "initial_image",
    "output",
    "output_path",
];

const MEDIA_ROUTING_FIELDS: &[&str] = &[
    "accelerator",
    "precision",
    "quantization",
    "profile",
    "quality",
    "performance_preference",
    "fallback_policy",
    "parameter_policy",
    "allow_cpu_offload",
    "no_allow_cpu_offload",
    "allow_sequential_offload",
    "no_allow_sequential_offload",
    "allow_component_offload",
    "no_allow_component_offload",
    "allow_disk_offload",
    "no_allow_disk_offload",
    "attention_backend",
    "compile",
    "no_compile",
    "timeout",
    "set",
];

fn expand_structured_media_values(raw: &mut BTreeMap<String, Value>) -> Result<()> {
    for path in [
        "controls",
        "loras",
        "adapters",
        "camera_keyframes",
        "prompt_keyframes",
        "guidance_schedule",
        "denoise_schedule",
        "adapter_schedule",
        "instruments",
        "mix_controls",
        "mastering_controls",
    ] {
        let Some(Value::Array(values)) = raw.get_mut(path) else {
            continue;
        };
        for value in values.iter_mut() {
            let Value::String(spec) = value else {
                continue;
            };
            if !spec.contains('=') {
                continue;
            }
            *value = parse_structured_spec(spec)
                .with_context(|| format!("invalid structured value for {path}"))?;
        }
    }

    let structured = raw
        .keys()
        .filter(|path| path.ends_with("_json") || path.ends_with("_file"))
        .cloned()
        .collect::<Vec<_>>();
    for path in structured {
        let Some(source) = raw.remove(&path) else {
            continue;
        };
        let (target, value) = if let Some(target) = path.strip_suffix("_json") {
            let encoded = source
                .as_str()
                .ok_or_else(|| anyhow!("{path} must contain JSON text"))?;
            let value = serde_json::from_str(encoded)
                .with_context(|| format!("invalid JSON supplied for {path}"))?;
            (target.to_string(), value)
        } else {
            let target = path.trim_end_matches("_file").to_string();
            let file = source
                .as_str()
                .ok_or_else(|| anyhow!("{path} must contain a file path"))?;
            let encoded = fs::read_to_string(file)
                .with_context(|| format!("failed to read structured override file {file}"))?;
            let value = serde_json::from_str(&encoded)
                .with_context(|| format!("invalid JSON in structured override file {file}"))?;
            (target, value)
        };
        merge_media_value(raw, target, value);
    }
    Ok(())
}

fn parse_structured_spec(spec: &str) -> Result<Value> {
    let mut object = serde_json::Map::new();
    for field in spec.split(',') {
        let (name, raw_value) = field
            .split_once('=')
            .ok_or_else(|| anyhow!("expected comma-separated key=value fields"))?;
        let name = name.trim().replace('-', "_");
        if name.is_empty() {
            bail!("structured field name must not be empty");
        }
        if object.contains_key(&name) {
            bail!("structured field '{name}' is repeated");
        }
        let raw_value = raw_value.trim();
        if raw_value.is_empty() {
            bail!("structured field '{name}' has an empty value");
        }
        let value = serde_json::from_str(raw_value)
            .unwrap_or_else(|_| Value::String(raw_value.to_string()));
        object.insert(name, value);
    }
    if object.is_empty() {
        bail!("structured value must contain at least one key=value field");
    }
    Ok(Value::Object(object))
}

fn merge_media_value(raw: &mut BTreeMap<String, Value>, path: String, value: Value) {
    match (raw.remove(&path), value) {
        (Some(Value::Array(mut current)), Value::Array(additional)) => {
            current.extend(additional);
            raw.insert(path, Value::Array(current));
        }
        (_, value) => {
            raw.insert(path, value);
        }
    }
}

fn normalize_negative_bool_pairs(raw: &mut BTreeMap<String, Value>) {
    if raw.remove("exclude_audio") == Some(Value::Bool(true)) {
        raw.insert("include_audio".to_string(), Value::Bool(false));
    }
    let negative_paths = raw
        .iter()
        .filter(|(path, value)| {
            *value == &Value::Bool(true)
                && path
                    .rsplit_once('.')
                    .map(|(_, name)| name.starts_with("no_"))
                    .unwrap_or_else(|| path.starts_with("no_"))
        })
        .map(|(path, _)| path.clone())
        .collect::<Vec<_>>();
    for negative_path in negative_paths {
        raw.remove(&negative_path);
        let positive_path = match negative_path.rsplit_once('.') {
            Some((prefix, name)) => format!("{prefix}.{}", name.trim_start_matches("no_")),
            None => negative_path.trim_start_matches("no_").to_string(),
        };
        raw.insert(positive_path, Value::Bool(false));
    }
}

fn normalize_media_parameter_path(task: InferenceTask, path: &str) -> String {
    let path = path.replace('-', "_");
    if path.contains('.') {
        return path;
    }
    let normalized = match (task, path.as_str()) {
        (
            InferenceTask::ImageGeneration
            | InferenceTask::ImageEditing
            | InferenceTask::ImageVariation
            | InferenceTask::ImageInpainting
            | InferenceTask::ImageOutpainting
            | InferenceTask::ImageUpscaling,
            "image_vae_tiling",
        ) => "vae_tiling",
        (
            InferenceTask::ImageGeneration
            | InferenceTask::ImageEditing
            | InferenceTask::ImageVariation
            | InferenceTask::ImageInpainting
            | InferenceTask::ImageOutpainting
            | InferenceTask::ImageUpscaling,
            "image_vae_slicing",
        ) => "vae_slicing",
        (
            InferenceTask::ImageGeneration
            | InferenceTask::ImageEditing
            | InferenceTask::ImageVariation
            | InferenceTask::ImageInpainting
            | InferenceTask::ImageOutpainting
            | InferenceTask::ImageUpscaling,
            "post_upscale",
        ) => "post_upscaling",
        (
            InferenceTask::VideoGeneration
            | InferenceTask::ImageToVideo
            | InferenceTask::VideoToVideo
            | InferenceTask::VideoInpainting
            | InferenceTask::VideoExtension
            | InferenceTask::VideoUpscaling
            | InferenceTask::FrameInterpolation,
            "loop",
        ) => "looping",
        (InferenceTask::SpeechToText, "task") => "operation",
        (
            InferenceTask::AudioGeneration
            | InferenceTask::MusicGeneration
            | InferenceTask::SongContinuation
            | InferenceTask::SongVariation,
            "num_variations",
        ) => "variations",
        (
            InferenceTask::AudioGeneration
            | InferenceTask::MusicGeneration
            | InferenceTask::SongContinuation
            | InferenceTask::SongVariation,
            "bpm_min",
        ) => "tempo_min",
        (
            InferenceTask::AudioGeneration
            | InferenceTask::MusicGeneration
            | InferenceTask::SongContinuation
            | InferenceTask::SongVariation,
            "bpm_max",
        ) => "tempo_max",
        (
            InferenceTask::AudioGeneration
            | InferenceTask::MusicGeneration
            | InferenceTask::SongContinuation
            | InferenceTask::SongVariation,
            "vocal_register",
        ) => "register",
        (
            InferenceTask::AudioGeneration
            | InferenceTask::MusicGeneration
            | InferenceTask::SongContinuation
            | InferenceTask::SongVariation,
            "vocal_range",
        ) => "range",
        (
            InferenceTask::AudioGeneration
            | InferenceTask::MusicGeneration
            | InferenceTask::SongContinuation
            | InferenceTask::SongVariation,
            "vocal_language",
        ) => "language",
        (
            InferenceTask::AudioGeneration
            | InferenceTask::MusicGeneration
            | InferenceTask::SongContinuation
            | InferenceTask::SongVariation,
            "vocal_accent",
        ) => "accent",
        (
            InferenceTask::AudioGeneration
            | InferenceTask::MusicGeneration
            | InferenceTask::SongContinuation
            | InferenceTask::SongVariation,
            "vocal_delivery",
        ) => "delivery",
        (
            InferenceTask::AudioGeneration
            | InferenceTask::MusicGeneration
            | InferenceTask::SongContinuation
            | InferenceTask::SongVariation,
            "vocal_emotion",
        ) => "emotion",
        (
            InferenceTask::AudioGeneration
            | InferenceTask::MusicGeneration
            | InferenceTask::SongContinuation
            | InferenceTask::SongVariation,
            "vocal_power",
        ) => "power",
        (InferenceTask::TextToSpeech, "loudness_lufs") => "loudness",
        (
            InferenceTask::AudioGeneration
            | InferenceTask::MusicGeneration
            | InferenceTask::SongContinuation
            | InferenceTask::SongVariation
            | InferenceTask::TextToSpeech
            | InferenceTask::StemSeparation,
            "format",
        ) => "output_format",
        _ => path.as_str(),
    };
    format!("{}.{}", task.parameter_namespace(), normalized)
}

fn print_inference_result(result: &InferenceResult) {
    println!(
        "{} {} via {} ({})",
        result.task, result.model, result.runtime, result.id
    );
    for output in &result.outputs {
        println!("output> {}", output.path);
        println!(
            "  id={} mime={} size={}{}{}{}",
            output.id,
            output.mime_type,
            format_bytes(output.size_bytes),
            output
                .width
                .map(|width| format!(" width={width}"))
                .unwrap_or_default(),
            output
                .height
                .map(|height| format!(" height={height}"))
                .unwrap_or_default(),
            output
                .duration
                .map(|duration| format!(" duration={duration:.2}s"))
                .unwrap_or_default(),
        );
    }
    for warning in &result.warnings {
        eprintln!("warning: {warning}");
    }
}

fn workload_estimate_request(
    model: String,
    task: InferenceTask,
    backend: BackendArg,
    device: Option<DeviceArg>,
    args: WorkloadEstimateArgs,
) -> InferenceRequest {
    let mut request = InferenceRequest::new(model, task);
    if task.requires_prompt() {
        request.prompt = Some("<estimate>".to_string());
    }
    request.inputs = estimate_inputs_for_task(task);
    let namespace = task.parameter_namespace();
    let mut insert = |name: &str, value: ParameterValue| {
        request
            .parameters
            .insert(format!("{namespace}.{name}"), value);
    };
    if let Some(value) = args.width {
        insert("width", value.into());
    }
    if let Some(value) = args.height {
        insert("height", value.into());
    }
    if let Some(value) = args.frames {
        insert("frames", value.into());
    }
    if let Some(value) = args.duration {
        insert("duration", value.into());
    }
    if let Some(value) = args.batch_size {
        insert("batch_size", value.into());
    }
    if let Some(value) = args.sample_rate {
        insert("sample_rate", value.into());
    }
    if let Some(value) = args.channels {
        insert("channels", ParameterValue::Integer(i64::from(value)));
    }
    if let Some(value) = args.steps {
        insert("steps", value.into());
    }
    request.routing.backend = Some(requested_backend_label(backend).to_string());
    request.routing.device = device.map(device_arg_label).map(str::to_string);
    request
}

fn estimate_inputs_for_task(task: InferenceTask) -> Vec<InferenceInput> {
    let placeholder = |modality, role: &str| InferenceInput {
        modality,
        role: role.to_string(),
        source: InferenceInputSource::Path {
            path: "<estimate-input>".to_string(),
        },
        mime_type: None,
    };
    use InferenceTask::*;
    match task {
        ImageUnderstanding | ImageEditing | ImageVariation | ImageUpscaling => {
            vec![placeholder(InputModality::Image, "image")]
        }
        ImageInpainting | ImageOutpainting => vec![
            placeholder(InputModality::Image, "image"),
            placeholder(InputModality::Image, "mask_image"),
        ],
        ImageToVideo => vec![placeholder(InputModality::Image, "initial_image")],
        VideoToVideo | VideoExtension | VideoUpscaling | FrameInterpolation => {
            vec![placeholder(InputModality::Video, "source_video")]
        }
        VideoInpainting => vec![
            placeholder(InputModality::Video, "source_video"),
            placeholder(InputModality::Video, "mask_video"),
        ],
        SongContinuation | SongVariation | SpeechToText | StemGeneration | StemSeparation
        | AudioEnhancement => vec![placeholder(InputModality::Audio, "input_audio")],
        VoiceConversion => vec![
            placeholder(InputModality::Audio, "input_audio"),
            placeholder(InputModality::Audio, "reference_audio"),
        ],
        TextGeneration | TextEmbedding | ImageGeneration | VideoGeneration | AudioGeneration
        | MusicGeneration | TextToSpeech => Vec::new(),
    }
}

fn print_workload_estimate(report: &WorkloadEstimate, verbose: bool) {
    println!("Task: {}", report.task);
    println!(
        "Fit: {} (confidence: {})",
        format!("{:?}", report.fit).to_ascii_lowercase(),
        format!("{:?}", report.confidence).to_ascii_lowercase()
    );
    for (label, value) in [
        ("Download", report.download_size_bytes),
        ("Weights", report.weight_payload_bytes),
        ("Accelerator peak", report.accelerator_peak_bytes),
        ("Host peak", report.host_peak_bytes),
        ("Output", report.output_size_bytes),
    ] {
        println!(
            "{label}: {}",
            value
                .map(format_bytes)
                .unwrap_or_else(|| "unknown".to_string())
        );
    }
    for warning in &report.warnings {
        println!("Warning: {warning}");
    }
    for recommendation in &report.recommendations {
        println!("Recommendation: {recommendation}");
    }
    if verbose {
        for assumption in &report.assumptions {
            println!("Assumption: {assumption}");
        }
    }
}

fn manifest_matches_list_filters(
    manifest: &ModelManifest,
    task: Option<InferenceTask>,
    input: Option<InputModality>,
    output: Option<OutputModality>,
    family: Option<&str>,
    layout: Option<RepositoryLayout>,
    backend: Option<&str>,
) -> bool {
    task.is_none_or(|task| manifest.metadata.tasks.contains(&task))
        && input.is_none_or(|input| manifest.metadata.input_modalities.contains(&input))
        && output.is_none_or(|output| manifest.metadata.output_modalities.contains(&output))
        && family.is_none_or(|family| {
            manifest
                .metadata
                .family
                .as_deref()
                .is_some_and(|value| value.eq_ignore_ascii_case(family))
        })
        && layout.is_none_or(|layout| manifest.metadata.repository_layout == layout)
        && backend.is_none_or(|backend| {
            manifest.backend.eq_ignore_ascii_case(backend)
                || manifest.metadata.compatible_runtimes.iter().any(|runtime| {
                    runtime
                        .to_ascii_lowercase()
                        .contains(&backend.to_ascii_lowercase())
                })
        })
}

fn join_display<T: std::fmt::Display>(values: &[T]) -> String {
    if values.is_empty() {
        "-".to_string()
    } else {
        values
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }
}

#[allow(clippy::too_many_arguments)]
fn print_parameters(
    store: &ModelStore,
    backend: BackendArg,
    device: Option<DeviceArg>,
    model: Option<&str>,
    task: Option<InferenceTask>,
    json_output: bool,
    example: bool,
    sources: bool,
) -> Result<()> {
    let manifest = model.map(|model| store.get(model)).transpose()?;
    let task = task
        .or_else(|| {
            manifest
                .as_ref()
                .and_then(|manifest| manifest.metadata.tasks.first().copied())
        })
        .ok_or_else(|| anyhow!("provide --task or an installed MODEL with declared tasks"))?;
    if let Some(manifest) = &manifest
        && !manifest.supports_task(task)
    {
        bail!(
            "model '{}' does not declare support for task {task}",
            manifest.id
        );
    }

    let descriptors = match manifest.as_ref() {
        Some(manifest) => parameter_schema_for_manifest(task, manifest)?,
        None => parameter_schema(task),
    };
    let mut runtime_candidates = manifest
        .as_ref()
        .map(|manifest| {
            InferenceService::new(store.clone())
                .parameter_probe(manifest, task)
                .map(|probe| probe.candidates)
        })
        .transpose()?
        .unwrap_or_default();
    if backend != BackendArg::Auto {
        let filter = requested_backend_label(backend);
        runtime_candidates.retain(|candidate| {
            candidate.id.eq_ignore_ascii_case(filter)
                || candidate.backend.eq_ignore_ascii_case(filter)
                || format!("{:?}", candidate.accelerator).eq_ignore_ascii_case(filter)
                || (filter == "metal"
                    && format!("{:?}", candidate.accelerator).eq_ignore_ascii_case("mps"))
        });
        if manifest.is_some() && runtime_candidates.is_empty() {
            bail!("no runtime matching backend '{filter}' supports model/task");
        }
    }
    let parameter_support = runtime_candidates
        .iter()
        .filter(|candidate| candidate.available)
        .max_by_key(|candidate| candidate.priority)
        .or_else(|| {
            runtime_candidates
                .iter()
                .max_by_key(|candidate| candidate.priority)
        })
        .map(|candidate| candidate.parameter_support.clone());
    let example_request = workload_estimate_request(
        manifest
            .as_ref()
            .map(|manifest| manifest.id.clone())
            .unwrap_or_else(|| "MODEL".to_string()),
        task,
        backend,
        device,
        WorkloadEstimateArgs::default(),
    );
    let resolved = if sources {
        let manifest = manifest
            .as_ref()
            .ok_or_else(|| anyhow!("--sources requires an installed MODEL"))?;
        let mut request = example_request.clone();
        request.model = manifest.id.clone();
        Some(InferenceService::new(store.clone()).resolve(request)?)
    } else {
        None
    };

    if json_output || example || sources {
        let payload = json!({
            "model": manifest.as_ref().map(|manifest| manifest.id.as_str()),
            "task": task,
            "backend": requested_backend_label(backend),
            "parameters": descriptors,
            "parameter_support": parameter_support,
            "runtime_candidates": runtime_candidates,
            "model_constraints": manifest
                .as_ref()
                .map(|manifest| &manifest.metadata.parameter_constraints),
            "example": example.then_some(&example_request),
            "sources": resolved.as_ref().map(|request| &request.parameters),
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    println!("Parameters for {task}");
    println!(
        "{:<34} {:<12} {:<16} {:<18} {:<15} DEFAULT",
        "PATH", "TYPE", "CATEGORY", "CLI FLAG", "SUPPORT"
    );
    for descriptor in descriptors {
        println!(
            "{:<34} {:<12} {:<16} {:<18} {:<15} {}",
            descriptor.path,
            format!("{:?}", descriptor.value_type).to_ascii_lowercase(),
            descriptor.category,
            descriptor.cli_flag,
            parameter_support
                .as_ref()
                .and_then(|support| support.get(&descriptor.path))
                .map(|status| format!("{status:?}").to_ascii_lowercase())
                .unwrap_or_else(|| "-".to_string()),
            descriptor
                .default
                .as_ref()
                .map(|value| serde_json::to_string(value).unwrap_or_else(|_| "-".to_string()))
                .unwrap_or_else(|| "-".to_string())
        );
    }
    Ok(())
}

fn print_inference_doctor(
    store: &ModelStore,
    backend: BackendArg,
    device: Option<DeviceArg>,
    task: Option<InferenceTask>,
    runtime: Option<&str>,
    model: Option<&str>,
) -> Result<()> {
    println!("Werk runtime diagnostics");
    print_backend_doctor(store, false);

    let report = CompanionClient::discover_doctor_report();
    println!(
        "Media companion: {} ({})",
        if report.available {
            "available"
        } else {
            "unavailable"
        },
        report.summary
    );
    if let Some(launcher) = report.launcher.as_deref() {
        println!("Companion launcher: {launcher}");
    }
    let runtime_lower = runtime.map(str::to_ascii_lowercase);
    for check in report.checks.iter().filter(|check| {
        runtime_lower.as_ref().is_none_or(|runtime| {
            check.name.to_ascii_lowercase().contains(runtime)
                || check.detail.to_ascii_lowercase().contains(runtime)
        })
    }) {
        let status = if check.available {
            "ok"
        } else if check.required {
            "missing"
        } else {
            "optional"
        };
        println!("{:<12} {:<24} {}", status, check.name, check.detail);
    }

    if let Some(task) = task {
        let compatible = store
            .list()?
            .into_iter()
            .filter(|manifest| manifest.supports_task(task))
            .map(|manifest| manifest.id)
            .collect::<Vec<_>>();
        println!(
            "Task {task}: {} installed model(s){}",
            compatible.len(),
            if compatible.is_empty() {
                String::new()
            } else {
                format!(" ({})", compatible.join(", "))
            }
        );
    }

    if let Some(model) = model {
        let manifest = store.get(model)?;
        let selected_task = task
            .or_else(|| manifest.metadata.tasks.first().copied())
            .ok_or_else(|| anyhow!("model '{model}' does not declare an inference task"))?;
        let mut request = workload_estimate_request(
            manifest.id.clone(),
            selected_task,
            backend,
            device,
            WorkloadEstimateArgs::default(),
        );
        if let Some(runtime) = runtime {
            request.routing.backend = Some(runtime.to_string());
            request.routing.fallback_policy = Some("none".to_string());
        }
        match InferenceService::new(store.clone()).plan(request) {
            Ok((_, estimate, plan)) => {
                println!(
                    "Model {}: task={}, layout={}, fit={:?}, runtime={}",
                    manifest.id,
                    selected_task,
                    manifest.metadata.repository_layout,
                    estimate.fit,
                    plan.selected_runtime.as_deref().unwrap_or("none")
                );
                for candidate in plan.candidates {
                    println!(
                        "  {:<24} {:?}: {}",
                        candidate.runtime_id,
                        candidate.status,
                        if candidate.reasons.is_empty() {
                            "eligible".to_string()
                        } else {
                            candidate.reasons.join("; ")
                        }
                    );
                }
            }
            Err(error) => println!("Model {}: diagnostic warning: {error:#}", manifest.id),
        }
    }
    Ok(())
}

fn should_print_startup_banner(command: &Commands) -> bool {
    should_print_startup_banner_for(
        command,
        io::stdout().is_terminal(),
        io::stdin().is_terminal(),
    )
}

fn resolve_api_keys(
    api_key: Option<String>,
    api_keys_path: Option<PathBuf>,
    allow_unauthenticated: bool,
) -> Result<Vec<String>> {
    if allow_unauthenticated {
        return Ok(Vec::new());
    }

    let mut keys = Vec::new();
    if let Some(key) = api_key {
        let key = key.trim().to_string();
        if key.is_empty() {
            bail!("--api-key / WERK_API_KEY must not be empty");
        }
        keys.push(key);
    }

    if let Some(path) = api_keys_path {
        keys.extend(
            api_keys::load_api_keys_file(&path)?
                .into_iter()
                .map(|entry| entry.key),
        );
    } else if let Ok(path) = api_keys::default_api_keys_path()
        && path.is_file()
    {
        keys.extend(
            api_keys::load_api_keys_file(&path)?
                .into_iter()
                .map(|entry| entry.key),
        );
    }

    if keys.is_empty() {
        bail!(
            "werk serve requires API key auth by default. Run `werk auth api-key generate`, pass `--api-key <key>` or WERK_API_KEY, or use `--allow-unauthenticated` for local development."
        );
    }

    Ok(keys)
}

fn should_print_startup_banner_for(
    command: &Commands,
    stdout_is_terminal: bool,
    stdin_is_terminal: bool,
) -> bool {
    if !stdout_is_terminal {
        return false;
    }

    match command {
        Commands::Serve { .. } | Commands::Run { .. } => true,
        Commands::Chat { .. } => stdin_is_terminal,
        Commands::Import { .. }
        | Commands::Pull { .. }
        | Commands::Remove { .. }
        | Commands::Image { .. }
        | Commands::Video { .. }
        | Commands::Audio { .. }
        | Commands::Estimate { .. }
        | Commands::Bench { .. }
        | Commands::Doctor { .. }
        | Commands::Backend { .. }
        | Commands::Artifacts { .. }
        | Commands::Auth { .. }
        | Commands::List { .. }
        | Commands::Parameters { .. }
        | Commands::Inspect { .. }
        | Commands::SelectFile { .. } => false,
    }
}

fn command_backend_install_verbose(command: &Commands) -> bool {
    match command {
        Commands::Serve { verbose, .. } => *verbose,
        Commands::Run { verbose, debug, .. } | Commands::Chat { verbose, debug, .. } => {
            *verbose || *debug
        }
        Commands::Bench { debug, .. } => *debug,
        Commands::Import { .. }
        | Commands::Pull { .. }
        | Commands::Remove { .. }
        | Commands::Image { .. }
        | Commands::Video { .. }
        | Commands::Audio { .. }
        | Commands::Estimate { .. }
        | Commands::Doctor { .. }
        | Commands::Backend { .. }
        | Commands::Artifacts { .. }
        | Commands::Auth { .. }
        | Commands::List { .. }
        | Commands::Parameters { .. }
        | Commands::Inspect { .. }
        | Commands::SelectFile { .. } => false,
    }
}

#[derive(Debug, Clone, Serialize)]
struct EstimateReport {
    model: String,
    source_url: Option<String>,
    format: String,
    architecture: String,
    backend_hint: String,
    model_files_bytes: u64,
    weight_files_bytes: u64,
    runtime_overhead_bytes: u64,
    kv_cache_bytes: u64,
    estimated_total_bytes: u64,
    system_total_bytes: u64,
    system_available_bytes: Option<u64>,
    weight_files: Vec<EstimateFileEntry>,
    ignored_files: Vec<EstimateFileEntry>,
    selected_model_files: Vec<String>,
    config_used: bool,
    confidence: EstimateConfidence,
    measured_peak_memory_bytes: Option<u64>,
    notes: Vec<String>,
    result: EstimateResult,
    recommendation: String,
}

#[derive(Debug, Clone, Serialize)]
struct EstimateFileEntry {
    path: String,
    size: u64,
    reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum EstimateResult {
    Ok,
    Warning,
    LikelyOom,
}

impl EstimateResult {
    fn display(self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::Warning => "warning",
            Self::LikelyOom => "likely OOM",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
enum EstimateConfidence {
    Low,
    Medium,
    High,
}

impl EstimateConfidence {
    fn display(self) -> &'static str {
        match self {
            Self::High => "HIGH",
            Self::Medium => "MEDIUM",
            Self::Low => "LOW",
        }
    }

    fn min(self, other: Self) -> Self {
        if self <= other { self } else { other }
    }
}

#[derive(Debug, Clone, Copy)]
struct SystemMemory {
    total_bytes: Option<u64>,
    available_bytes: Option<u64>,
}

#[derive(Debug, Clone)]
struct WeightAccounting {
    counted: Vec<EstimateFileEntry>,
    ignored: Vec<EstimateFileEntry>,
    selected: Vec<String>,
    confidence: EstimateConfidence,
}

impl WeightAccounting {
    fn total_bytes(&self) -> u64 {
        self.counted.iter().map(|file| file.size).sum()
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
struct EstimateConfig {
    hidden_size: Option<u64>,
    num_hidden_layers: Option<u64>,
    num_attention_heads: Option<u64>,
    num_key_value_heads: Option<u64>,
    head_dim: Option<u64>,
    max_position_embeddings: Option<u64>,
    sliding_window: Option<u64>,
    dtype: Option<String>,
    vocab_size: Option<u64>,
    architectures: Vec<String>,
    model_type: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct KvCacheEstimate {
    bytes: u64,
    confidence: EstimateConfidence,
    config_used: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
struct EstimateObservation {
    model: String,
    backend: Option<String>,
    architecture: Option<String>,
    format: Option<String>,
    measured_peak_memory_bytes: Option<u64>,
    prompt_tps: Option<f64>,
    generation_tps: Option<f64>,
    timestamp: Option<u64>,
}

#[derive(Debug, Clone)]
struct RemoteHfModel {
    repo: String,
    config: Option<Value>,
    files: Vec<RemoteHfFile>,
    gated: bool,
}

#[derive(Debug, Clone)]
struct RemoteHfFile {
    path: String,
    size: u64,
}

fn estimate_model_or_huggingface(
    store: &ModelStore,
    model: &str,
    include_file: Option<&str>,
    system: SystemMemory,
) -> Result<EstimateReport> {
    match store.get(model) {
        Ok(manifest) => {
            if include_file.is_some() {
                bail!(
                    "`--file` is only supported for remote Hugging Face estimates before a model is pulled"
                );
            }
            Ok(estimate_model_memory(store, &manifest, system))
        }
        Err(err) if err.to_string() == format!("model '{model}' is not installed") => {
            if looks_like_huggingface_repo_id(model) {
                return estimate_huggingface_model(store, model, include_file, system);
            }
            bail!("model '{model}' is not installed; run `werk pull {model}` first")
        }
        Err(err) => Err(err),
    }
}

fn looks_like_huggingface_repo_id(model: &str) -> bool {
    let model = model.trim();
    if model.is_empty()
        || model.starts_with('-')
        || model.starts_with('/')
        || model.starts_with('.')
        || model.ends_with('/')
        || model.contains("..")
        || model.contains("://")
        || model.contains('\\')
    {
        return false;
    }

    let mut parts = model.split('/');
    matches!(
        (parts.next(), parts.next(), parts.next()),
        (Some(namespace), Some(repo), None) if !namespace.is_empty() && !repo.is_empty()
    )
}

fn estimate_model_memory(
    store: &ModelStore,
    manifest: &ModelManifest,
    system: SystemMemory,
) -> EstimateReport {
    let accounting = estimate_weight_accounting(store, manifest);
    let model_files_bytes = accounting.total_bytes();
    let config = read_estimate_config(store, manifest);
    let runtime_overhead_bytes = runtime_overhead_bytes(&manifest.format, model_files_bytes);
    let kv_cache = kv_cache_estimate(model_files_bytes, manifest.architecture.as_deref(), &config);
    let estimated_total_bytes = model_files_bytes
        .saturating_add(runtime_overhead_bytes)
        .saturating_add(kv_cache.bytes);
    let result = estimate_result(
        estimated_total_bytes,
        system.total_bytes,
        system.available_bytes,
    );
    let confidence = accounting
        .confidence
        .min(kv_cache.confidence)
        .min(if model_files_bytes > 0 {
            EstimateConfidence::High
        } else {
            EstimateConfidence::Low
        });

    EstimateReport {
        model: manifest.id.clone(),
        source_url: estimate_source_url(manifest),
        format: format_label(&manifest.format).to_string(),
        architecture: manifest
            .architecture
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        backend_hint: estimate_backend_hint(manifest).to_string(),
        model_files_bytes,
        weight_files_bytes: model_files_bytes,
        runtime_overhead_bytes,
        kv_cache_bytes: kv_cache.bytes,
        estimated_total_bytes,
        system_total_bytes: system.total_bytes.unwrap_or(0),
        system_available_bytes: system.available_bytes,
        weight_files: accounting.counted,
        ignored_files: accounting.ignored,
        selected_model_files: accounting.selected,
        config_used: kv_cache.config_used,
        confidence,
        measured_peak_memory_bytes: latest_estimate_observation(store, manifest)
            .and_then(|observation| observation.measured_peak_memory_bytes),
        notes: Vec::new(),
        result,
        recommendation: estimate_recommendation(result).to_string(),
    }
}

fn estimate_huggingface_model(
    store: &ModelStore,
    repo: &str,
    include_file: Option<&str>,
    system: SystemMemory,
) -> Result<EstimateReport> {
    validate_huggingface_repo_for_estimate(repo)?;
    let token = store.huggingface_http_token()?;
    let remote = fetch_remote_huggingface_model(repo, token.as_deref())?;
    if remote.gated && token.is_none() {
        bail!(
            "Hugging Face gated model requires browser agreement: {repo} (https://huggingface.co/{repo}). Open the model page, accept the conditions, then run `werk auth huggingface login` or set HF_TOKEN and retry."
        );
    }

    let mut manifest = remote_hf_manifest(&remote, include_file)?;
    let mut accounting = if manifest.format == ModelFormat::SafeTensors
        || manifest.format == ModelFormat::Mlx
        || manifest.format == ModelFormat::PyTorch
    {
        remote
            .files
            .iter()
            .find(|file| file.path.ends_with(".safetensors.index.json"))
            .and_then(|file| {
                fetch_huggingface_json_file(repo, &file.path, token.as_deref())
                    .ok()
                    .and_then(|value| {
                        safetensors_index_weight_accounting_from_value(
                            &manifest,
                            &format!("files/{}", file.path),
                            &value,
                        )
                    })
            })
            .unwrap_or_else(|| estimate_weight_accounting_without_store(&manifest))
    } else {
        estimate_weight_accounting_without_store(&manifest)
    };

    if let Some(include_file) = include_file {
        let selected_path = format!("files/{}", normalize_remote_hf_file_path(include_file)?);
        if let Some(file) = manifest
            .files
            .iter()
            .find(|file| file.path == selected_path)
        {
            accounting = single_selected_weight_accounting(
                &manifest,
                file,
                "explicit --file selected for remote estimate",
            );
            manifest.model_path = Some(selected_path);
        } else {
            bail!("file '{include_file}' was not found in Hugging Face repo '{repo}'");
        }
    }

    let model_files_bytes = accounting.total_bytes();
    let config = remote.config.as_ref().map(parse_estimate_config);
    let runtime_overhead_bytes = runtime_overhead_bytes(&manifest.format, model_files_bytes);
    let kv_cache = kv_cache_estimate(model_files_bytes, manifest.architecture.as_deref(), &config);
    let estimated_total_bytes = model_files_bytes
        .saturating_add(runtime_overhead_bytes)
        .saturating_add(kv_cache.bytes);
    let result = estimate_result(
        estimated_total_bytes,
        system.total_bytes,
        system.available_bytes,
    );
    let confidence = accounting
        .confidence
        .min(kv_cache.confidence)
        .min(if model_files_bytes > 0 {
            EstimateConfidence::Medium
        } else {
            EstimateConfidence::Low
        });
    let mut notes = vec![
        "Remote estimate uses Hugging Face metadata and small config/index files only; it does not download model weights.".to_string(),
    ];
    if model_files_bytes == 0 {
        notes.push(
            "Hugging Face metadata did not include file sizes, so the memory estimate is incomplete."
                .to_string(),
        );
    }
    if manifest.architecture.as_deref() == Some("chatglm")
        && cfg!(target_os = "macos")
        && manifest.format == ModelFormat::SafeTensors
    {
        notes.push(
            "Raw ChatGLM/GLM Hugging Face repositories may need MLX conversion before mlx-lm can load them."
                .to_string(),
        );
    }

    Ok(EstimateReport {
        model: repo.to_string(),
        source_url: Some(format!("https://huggingface.co/{repo}")),
        format: format_label(&manifest.format).to_string(),
        architecture: manifest
            .architecture
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        backend_hint: estimate_backend_hint(&manifest).to_string(),
        model_files_bytes,
        weight_files_bytes: model_files_bytes,
        runtime_overhead_bytes,
        kv_cache_bytes: kv_cache.bytes,
        estimated_total_bytes,
        system_total_bytes: system.total_bytes.unwrap_or(0),
        system_available_bytes: system.available_bytes,
        weight_files: accounting.counted,
        ignored_files: accounting.ignored,
        selected_model_files: accounting.selected,
        config_used: kv_cache.config_used,
        confidence,
        measured_peak_memory_bytes: None,
        notes,
        result,
        recommendation: estimate_recommendation(result).to_string(),
    })
}

fn validate_huggingface_repo_for_estimate(repo: &str) -> Result<()> {
    if repo.trim().is_empty() || repo.starts_with('-') || repo.contains("..") {
        bail!("invalid Hugging Face repo id: {repo}");
    }
    Ok(())
}

fn fetch_remote_huggingface_model(repo: &str, token: Option<&str>) -> Result<RemoteHfModel> {
    let api_url = format!(
        "https://huggingface.co/api/models/{}?blobs=true",
        percent_encode_hf_path(repo)
    );
    let metadata = fetch_huggingface_json_url(&api_url, token).map_err(|err| {
        anyhow!(
            "failed to read Hugging Face metadata for {repo} (https://huggingface.co/{repo}): {err}"
        )
    })?;
    let config = fetch_huggingface_json_file(repo, "config.json", token).ok();
    Ok(parse_remote_huggingface_model(repo, &metadata, config))
}

fn fetch_huggingface_json_file(repo: &str, path: &str, token: Option<&str>) -> Result<Value> {
    let path = normalize_remote_hf_file_path(path)?;
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        percent_encode_hf_path(repo),
        percent_encode_hf_path(&path)
    );
    fetch_huggingface_json_url(&url, token)
}

fn fetch_huggingface_json_url(url: &str, token: Option<&str>) -> Result<Value> {
    let text = fetch_huggingface_text_url(url, token)?;
    serde_json::from_str(&text).map_err(|err| anyhow!("Hugging Face response was not JSON: {err}"))
}

fn fetch_huggingface_text_url(url: &str, token: Option<&str>) -> Result<String> {
    let mut command = Command::new("curl");
    command.args([
        "-sSL",
        "--max-time",
        "20",
        "-A",
        "werk1112",
        "-w",
        "\n%{http_code}",
    ]);
    if let Some(token) = token.map(str::trim).filter(|token| !token.is_empty()) {
        command
            .arg("-H")
            .arg(format!("Authorization: Bearer {token}"));
    }
    command.arg(url);

    let output = command
        .output()
        .map_err(|err| anyhow!("failed to execute curl for Hugging Face metadata: {err}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("curl failed: {}", stderr.trim());
    }

    let stdout = String::from_utf8(output.stdout)
        .map_err(|err| anyhow!("Hugging Face response was not valid UTF-8: {err}"))?;
    let Some((body, status)) = stdout.rsplit_once('\n') else {
        bail!("Hugging Face response did not include an HTTP status");
    };
    let status = status.trim().parse::<u16>().unwrap_or(0);
    if !(200..300).contains(&status) {
        let detail = body.trim();
        if detail.is_empty() {
            bail!("Hugging Face returned HTTP {status}");
        }
        bail!("Hugging Face returned HTTP {status}: {detail}");
    }
    Ok(body.to_string())
}

fn parse_remote_huggingface_model(
    repo: &str,
    metadata: &Value,
    config: Option<Value>,
) -> RemoteHfModel {
    RemoteHfModel {
        repo: repo.to_string(),
        config,
        files: parse_remote_hf_files(metadata),
        gated: value_is_remote_hf_gated(metadata.get("gated")),
    }
}

fn parse_remote_hf_files(metadata: &Value) -> Vec<RemoteHfFile> {
    let mut files = metadata
        .get("siblings")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|file| {
            let path = file
                .get("rfilename")
                .or_else(|| file.get("path"))
                .and_then(Value::as_str)?;
            Some(RemoteHfFile {
                path: path.replace('\\', "/"),
                size: parse_remote_hf_file_size(file).unwrap_or(0),
            })
        })
        .collect::<Vec<_>>();
    files.sort_by(|left, right| left.path.cmp(&right.path));
    files.dedup_by(|left, right| left.path == right.path);
    files
}

fn parse_remote_hf_file_size(file: &Value) -> Option<u64> {
    json_u64(file.get("size"))
        .or_else(|| json_u64(file.get("blob_size")))
        .or_else(|| file.get("lfs").and_then(|lfs| json_u64(lfs.get("size"))))
        .or_else(|| {
            file.get("lfs")
                .and_then(|lfs| json_u64(lfs.get("blob_size")))
        })
}

fn json_u64(value: Option<&Value>) -> Option<u64> {
    match value? {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}

fn value_is_remote_hf_gated(value: Option<&Value>) -> bool {
    match value {
        Some(Value::Bool(gated)) => *gated,
        Some(Value::String(gated)) => !matches!(gated.as_str(), "" | "false" | "False" | "none"),
        _ => false,
    }
}

fn remote_hf_manifest(remote: &RemoteHfModel, include_file: Option<&str>) -> Result<ModelManifest> {
    let normalized_include = include_file
        .map(normalize_remote_hf_file_path)
        .transpose()?;
    let format = normalized_include
        .as_deref()
        .map(remote_detect_format_for_file_path)
        .unwrap_or_else(|| remote_detect_format(remote));
    let files = remote
        .files
        .iter()
        .map(|file| crate::model_store::ModelFile {
            path: format!("files/{}", file.path),
            size: file.size,
            checksum: "remote-metadata".to_string(),
        })
        .collect::<Vec<_>>();
    let model_path = if let Some(include_file) = normalized_include.as_deref() {
        let selected_path = format!("files/{include_file}");
        if !files.iter().any(|file| file.path == selected_path) {
            bail!(
                "file '{include_file}' was not found in Hugging Face repo '{}'",
                remote.repo
            );
        }
        Some(selected_path)
    } else {
        remote_selected_model_path(&remote.files, &format)
    };
    let tokenizer_path = remote
        .files
        .iter()
        .find(|file| file.path.ends_with("tokenizer.json"))
        .map(|file| format!("files/{}", file.path));
    let config_path = remote
        .files
        .iter()
        .find(|file| file.path == "config.json" || file.path.ends_with("/config.json"))
        .map(|file| format!("files/{}", file.path));
    let architecture = remote_architecture_from_config(remote.config.as_ref());

    Ok(ModelManifest {
        id: remote.repo.clone(),
        source: ModelSource::HuggingFace {
            repo: remote.repo.clone(),
        },
        format: format.clone(),
        architecture,
        tokenizer_path,
        config_path,
        model_path,
        backend: format.backend_hint().to_string(),
        created_unix: 0,
        files,
        artifacts: Vec::new(),
        metadata: Default::default(),
    })
}

fn remote_detect_format(remote: &RemoteHfModel) -> ModelFormat {
    let repo_lower = remote.repo.to_ascii_lowercase();
    if remote
        .files
        .iter()
        .any(|file| extension_eq_str(&file.path, "gguf"))
    {
        ModelFormat::Gguf
    } else if remote
        .files
        .iter()
        .any(|file| extension_eq_str(&file.path, "npz"))
        || repo_lower.contains("mlx")
        || remote
            .files
            .iter()
            .any(|file| file.path.to_ascii_lowercase().contains("mlx"))
    {
        ModelFormat::Mlx
    } else if remote
        .files
        .iter()
        .any(|file| extension_eq_str(&file.path, "safetensors"))
    {
        ModelFormat::SafeTensors
    } else if remote
        .files
        .iter()
        .any(|file| extension_eq_str(&file.path, "onnx"))
    {
        ModelFormat::Onnx
    } else if remote
        .files
        .iter()
        .any(|file| extension_eq_str(&file.path, "pt") || extension_eq_str(&file.path, "pth"))
        || remote
            .files
            .iter()
            .any(|file| file.path.ends_with("pytorch_model.bin"))
    {
        ModelFormat::PyTorch
    } else {
        ModelFormat::Unknown
    }
}

fn remote_detect_format_for_file_path(path: &str) -> ModelFormat {
    if extension_eq_str(path, "gguf") {
        ModelFormat::Gguf
    } else if extension_eq_str(path, "safetensors") {
        ModelFormat::SafeTensors
    } else if extension_eq_str(path, "npz") {
        ModelFormat::Mlx
    } else if extension_eq_str(path, "onnx") {
        ModelFormat::Onnx
    } else if extension_eq_str(path, "pt")
        || extension_eq_str(path, "pth")
        || path.ends_with("pytorch_model.bin")
    {
        ModelFormat::PyTorch
    } else {
        ModelFormat::Unknown
    }
}

fn remote_selected_model_path(files: &[RemoteHfFile], format: &ModelFormat) -> Option<String> {
    let path = match format {
        ModelFormat::Gguf => files
            .iter()
            .filter(|file| extension_eq_str(&file.path, "gguf"))
            .min_by(|left, right| {
                remote_gguf_priority(&left.path)
                    .cmp(&remote_gguf_priority(&right.path))
                    .then_with(|| left.path.cmp(&right.path))
            })
            .map(|file| file.path.clone()),
        ModelFormat::SafeTensors => files
            .iter()
            .find(|file| extension_eq_str(&file.path, "safetensors"))
            .map(|file| file.path.clone()),
        ModelFormat::Mlx => files
            .iter()
            .find(|file| extension_eq_str(&file.path, "npz"))
            .or_else(|| {
                files
                    .iter()
                    .find(|file| extension_eq_str(&file.path, "safetensors"))
            })
            .map(|file| file.path.clone()),
        ModelFormat::PyTorch => files
            .iter()
            .find(|file| extension_eq_str(&file.path, "pt"))
            .or_else(|| {
                files
                    .iter()
                    .find(|file| extension_eq_str(&file.path, "pth"))
            })
            .or_else(|| {
                files
                    .iter()
                    .find(|file| file.path.ends_with("pytorch_model.bin"))
            })
            .map(|file| file.path.clone()),
        ModelFormat::Onnx => files
            .iter()
            .find(|file| extension_eq_str(&file.path, "onnx"))
            .map(|file| file.path.clone()),
        ModelFormat::TensorRt
        | ModelFormat::OpenVino
        | ModelFormat::TensorFlow
        | ModelFormat::CoreMl
        | ModelFormat::Unknown => None,
    }?;
    Some(format!("files/{path}"))
}

fn remote_gguf_priority(path: &str) -> usize {
    let lower = path.to_ascii_lowercase();
    [
        "q4_k_m", "q5_k_m", "q4_k_s", "q5_k_s", "q6_k", "q8_0", "q3_k_m", "q3_k_l", "q3_k_s",
        "q4_0", "q5_0", "q2_k",
    ]
    .iter()
    .position(|quant| lower.contains(quant))
    .unwrap_or(usize::MAX)
}

fn remote_architecture_from_config(config: Option<&Value>) -> Option<String> {
    let value = config?;
    let text_config = value.get("text_config").unwrap_or(value);
    text_config
        .get("model_type")
        .and_then(Value::as_str)
        .or_else(|| value.get("model_type").and_then(Value::as_str))
        .or_else(|| {
            value
                .get("architectures")
                .and_then(Value::as_array)
                .and_then(|items| items.first())
                .and_then(Value::as_str)
        })
        .map(ToString::to_string)
}

fn normalize_remote_hf_file_path(file: &str) -> Result<String> {
    let mut path = file.trim().replace('\\', "/");
    while let Some(rest) = path.strip_prefix("./") {
        path = rest.to_string();
    }
    if let Some(rest) = path.strip_prefix("files/") {
        path = rest.to_string();
    }
    if path.is_empty()
        || path.starts_with('/')
        || path
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        bail!("Hugging Face file must be a relative path inside the repository");
    }
    Ok(path)
}

fn percent_encode_hf_path(path: &str) -> String {
    path.as_bytes()
        .iter()
        .flat_map(|byte| match *byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                vec![*byte as char]
            }
            byte => format!("%{byte:02X}").chars().collect(),
        })
        .collect()
}

fn estimate_source_url(manifest: &ModelManifest) -> Option<String> {
    match &manifest.source {
        ModelSource::HuggingFace { repo } => Some(format!("https://huggingface.co/{repo}")),
        ModelSource::LocalPath { .. } => None,
    }
}

fn print_estimate_report(report: &EstimateReport, verbose: bool) {
    print!("{}", format_estimate_report(report, verbose));
}

fn format_estimate_report(report: &EstimateReport, verbose: bool) -> String {
    let mut output = String::new();
    output.push_str(&format!("Model:        {}\n", report.model));
    if let Some(source_url) = &report.source_url {
        output.push_str(&format!("Source:       {source_url}\n"));
    }
    output.push_str(&format!("Format:       {}\n", report.format));
    output.push_str(&format!("Architecture: {}\n", report.architecture));
    output.push_str(&format!("Backend:      {}\n", report.backend_hint));
    output.push('\n');
    output.push_str(&format!(
        "Weights:      {}\n",
        format_bytes(report.model_files_bytes)
    ));
    output.push_str(&format!(
        "Runtime:      {}\n",
        format_bytes(report.runtime_overhead_bytes)
    ));
    output.push_str(&format!(
        "KV cache:     {}\n",
        format_bytes(report.kv_cache_bytes)
    ));
    output.push_str(&format!(
        "Total:        {}\n",
        format_bytes(report.estimated_total_bytes)
    ));
    output.push('\n');
    output.push_str(&format!(
        "System memory:     {}\n",
        format_optional_bytes(Some(report.system_total_bytes))
    ));
    output.push_str(&format!(
        "Available memory:  {}\n",
        format_optional_bytes(report.system_available_bytes)
    ));
    output.push_str(&format!(
        "Confidence:        {}\n",
        report.confidence.display()
    ));
    if let Some(measured_peak) = report.measured_peak_memory_bytes {
        output.push('\n');
        output.push_str(&format!(
            "Measured peak:     {}\n",
            format_bytes(measured_peak)
        ));
    }
    if !report.notes.is_empty() {
        output.push('\n');
        output.push_str("Notes:\n");
        for note in &report.notes {
            output.push_str(&format!("  - {note}\n"));
        }
    }
    if verbose {
        output.push('\n');
        output.push_str("Selected model file(s):\n");
        if report.selected_model_files.is_empty() {
            output.push_str("  - none\n");
        } else {
            for path in &report.selected_model_files {
                output.push_str(&format!("  - {path}\n"));
            }
        }
        output.push_str("Weight files counted:\n");
        if report.weight_files.is_empty() {
            output.push_str("  - none\n");
        } else {
            for file in &report.weight_files {
                output.push_str(&format!(
                    "  - {} ({}, {})\n",
                    file.path,
                    format_bytes(file.size),
                    file.reason
                ));
            }
        }
        output.push_str("Files ignored:\n");
        if report.ignored_files.is_empty() {
            output.push_str("  - none\n");
        } else {
            for file in &report.ignored_files {
                output.push_str(&format!(
                    "  - {} ({}, {})\n",
                    file.path,
                    format_bytes(file.size),
                    file.reason
                ));
            }
        }
        output.push_str(&format!(
            "Total counted weight bytes: {}\n",
            report.model_files_bytes
        ));
    }
    output.push('\n');
    output.push_str(&format!("Result:       {}\n", report.result.display()));
    output.push('\n');
    output.push_str("Recommendation:\n");
    output.push_str(&format!("  {}\n", report.recommendation));
    output
}

fn format_optional_bytes(bytes: Option<u64>) -> String {
    bytes
        .filter(|bytes| *bytes > 0)
        .map(format_bytes)
        .unwrap_or_else(|| "unknown".to_string())
}

fn estimate_recommendation(result: EstimateResult) -> &'static str {
    match result {
        EstimateResult::Ok => "This model is likely to fit under the current heuristic.",
        EstimateResult::Warning => {
            "This model may fit, but it is close to the available-memory limit; close memory-heavy applications or reduce max tokens."
        }
        EstimateResult::LikelyOom => {
            "Use a smaller or quantized model, reduce max tokens, or close memory-heavy applications."
        }
    }
}

fn estimate_result(
    estimated_total_bytes: u64,
    system_total_bytes: Option<u64>,
    system_available_bytes: Option<u64>,
) -> EstimateResult {
    if let Some(available) = system_available_bytes.filter(|bytes| *bytes > 0) {
        return classify_against_limit(estimated_total_bytes, available, 0.70, 0.85);
    }
    if let Some(total) = system_total_bytes.filter(|bytes| *bytes > 0) {
        return classify_against_limit(estimated_total_bytes, total, 0.50, 0.65);
    }
    EstimateResult::Warning
}

fn classify_against_limit(
    estimated_total_bytes: u64,
    limit_bytes: u64,
    ok_ratio: f64,
    warning_ratio: f64,
) -> EstimateResult {
    let estimated = estimated_total_bytes as f64;
    let limit = limit_bytes as f64;
    if estimated <= limit * ok_ratio {
        EstimateResult::Ok
    } else if estimated <= limit * warning_ratio {
        EstimateResult::Warning
    } else {
        EstimateResult::LikelyOom
    }
}

#[cfg(test)]
fn estimate_model_files_bytes(manifest: &ModelManifest) -> u64 {
    estimate_weight_accounting_without_store(manifest).total_bytes()
}

fn estimate_weight_accounting(store: &ModelStore, manifest: &ModelManifest) -> WeightAccounting {
    if matches!(
        manifest.format,
        ModelFormat::SafeTensors | ModelFormat::Mlx | ModelFormat::PyTorch
    ) && let Some(accounting) = safetensors_index_weight_accounting(store, manifest)
    {
        return accounting;
    }
    estimate_weight_accounting_without_store(manifest)
}

fn estimate_weight_accounting_without_store(manifest: &ModelManifest) -> WeightAccounting {
    let selected_model_path = manifest.model_path.clone();
    let selected = selected_model_path.iter().cloned().collect::<Vec<_>>();
    let selected_file = selected_model_path
        .as_deref()
        .and_then(|path| manifest.files.iter().find(|file| file.path == path));

    if matches!(manifest.format, ModelFormat::Gguf | ModelFormat::Onnx)
        && let Some(file) = selected_file
    {
        return single_selected_weight_accounting(manifest, file, "selected runtime model file");
    }

    if matches!(
        manifest.format,
        ModelFormat::SafeTensors | ModelFormat::Mlx | ModelFormat::PyTorch
    ) {
        let safetensors = manifest
            .files
            .iter()
            .filter(|file| extension_eq_str(&file.path, "safetensors"))
            .collect::<Vec<_>>();
        if safetensors.len() == 1 {
            return single_selected_weight_accounting(
                manifest,
                safetensors[0],
                "single safetensors weight file",
            );
        }
    }

    let mut counted = Vec::new();
    let mut ignored = Vec::new();
    for file in &manifest.files {
        if should_ignore_estimate_file(&file.path, false) {
            ignored.push(estimate_file_entry(file, "non-weight metadata/cache file"));
        } else if is_estimate_weight_file(&file.path) {
            counted.push(estimate_file_entry(file, "recognized weight file"));
        } else {
            ignored.push(estimate_file_entry(file, "not a recognized weight file"));
        }
    }

    let confidence = if counted.is_empty() {
        EstimateConfidence::Low
    } else if selected_model_path.is_some()
        || manifest
            .files
            .iter()
            .filter(|file| is_estimate_weight_file(&file.path))
            .count()
            == counted.len()
    {
        EstimateConfidence::Medium
    } else {
        EstimateConfidence::Low
    };

    WeightAccounting {
        counted,
        ignored,
        selected,
        confidence,
    }
}

fn single_selected_weight_accounting(
    manifest: &ModelManifest,
    selected_file: &crate::model_store::ModelFile,
    reason: &str,
) -> WeightAccounting {
    let mut ignored = Vec::new();
    for file in &manifest.files {
        if file.path != selected_file.path {
            ignored.push(estimate_file_entry(file, "not selected for this model"));
        }
    }
    WeightAccounting {
        counted: vec![estimate_file_entry(selected_file, reason)],
        ignored,
        selected: vec![selected_file.path.clone()],
        confidence: EstimateConfidence::High,
    }
}

fn safetensors_index_weight_accounting(
    store: &ModelStore,
    manifest: &ModelManifest,
) -> Option<WeightAccounting> {
    let index_path = find_safetensors_index_path(manifest)?;
    let index_abs = store.model_dir(&manifest.id).join(&index_path);
    let data = fs::read_to_string(index_abs).ok()?;
    let value: Value = serde_json::from_str(&data).ok()?;
    safetensors_index_weight_accounting_from_value(manifest, &index_path, &value)
}

fn safetensors_index_weight_accounting_from_value(
    manifest: &ModelManifest,
    index_path: &str,
    value: &Value,
) -> Option<WeightAccounting> {
    let weight_map = value.get("weight_map")?.as_object()?;
    let index_dir = Path::new(&index_path)
        .parent()
        .map(|path| path.to_string_lossy().replace('\\', "/"))
        .unwrap_or_default();
    let mut shards = weight_map
        .values()
        .filter_map(Value::as_str)
        .map(|path| join_manifest_relative(&index_dir, path))
        .collect::<Vec<_>>();
    shards.sort();
    shards.dedup();
    if shards.is_empty() {
        return None;
    }

    let mut counted = Vec::new();
    let mut ignored = Vec::new();
    let mut missing = false;
    for file in &manifest.files {
        if shards.iter().any(|path| path == &file.path) {
            counted.push(estimate_file_entry(file, "referenced by safetensors index"));
        } else {
            ignored.push(estimate_file_entry(
                file,
                "not referenced by safetensors index",
            ));
        }
    }
    for shard in &shards {
        if !manifest.files.iter().any(|file| file.path == *shard) {
            missing = true;
        }
    }

    Some(WeightAccounting {
        counted,
        ignored,
        selected: shards,
        confidence: if missing {
            EstimateConfidence::Low
        } else {
            EstimateConfidence::High
        },
    })
}

fn find_safetensors_index_path(manifest: &ModelManifest) -> Option<String> {
    if let Some(model_path) = manifest.model_path.as_deref() {
        let path = Path::new(model_path);
        let name = path.file_name().and_then(|name| name.to_str())?;
        if name.ends_with(".safetensors") {
            let index_name = format!("{name}.index.json");
            let candidate = path
                .parent()
                .map(|parent| parent.join(&index_name))
                .unwrap_or_else(|| PathBuf::from(index_name))
                .to_string_lossy()
                .replace('\\', "/");
            if manifest.files.iter().any(|file| file.path == candidate) {
                return Some(candidate);
            }
        }
    }
    manifest
        .files
        .iter()
        .find(|file| file.path.ends_with(".safetensors.index.json"))
        .map(|file| file.path.clone())
}

fn join_manifest_relative(base: &str, path: &str) -> String {
    let normalized = path.replace('\\', "/");
    if normalized.starts_with("files/") || base.is_empty() {
        normalized
    } else {
        format!("{base}/{normalized}")
    }
}

fn estimate_file_entry(file: &crate::model_store::ModelFile, reason: &str) -> EstimateFileEntry {
    EstimateFileEntry {
        path: file.path.clone(),
        size: file.size,
        reason: reason.to_string(),
    }
}

fn should_ignore_estimate_file(path: &str, selected_runtime_artifact: bool) -> bool {
    let lower = path.to_ascii_lowercase();
    if lower.split('/').any(|part| {
        matches!(
            part,
            ".git" | ".cache" | "cache" | "tmp" | "temp" | "__pycache__"
        )
    }) {
        return true;
    }
    if !selected_runtime_artifact && lower.split('/').any(|part| part == "artifacts") {
        return true;
    }
    let name = Path::new(&lower)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    matches!(
        name,
        "readme.md"
            | "license"
            | "license.md"
            | "tokenizer.json"
            | "tokenizer_config.json"
            | "special_tokens_map.json"
            | "generation_config.json"
            | "config.json"
            | "merges.txt"
            | "vocab.json"
            | "added_tokens.json"
    ) || name.starts_with("chat_template")
}

fn is_estimate_weight_file(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    let name = Path::new(&lower)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if lower.ends_with(".onnx_data") {
        return true;
    }
    if extension_eq_str(&lower, "bin") {
        return name.starts_with("pytorch_model") || name.starts_with("model");
    }
    matches!(
        Path::new(&lower).extension().and_then(|ext| ext.to_str()),
        Some("safetensors" | "gguf" | "onnx" | "pt" | "pth" | "npz")
    )
}

fn extension_eq_str(path: &str, expected: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case(expected))
}

fn runtime_overhead_bytes(format: &ModelFormat, weights: u64) -> u64 {
    match format {
        ModelFormat::Gguf => (256 * MIB).max(scale_bytes(weights, 0.05)),
        ModelFormat::Mlx => (512 * MIB).max(scale_bytes(weights, 0.08)),
        _ => (512 * MIB).max(scale_bytes(weights, 0.10)),
    }
}

fn kv_cache_estimate(
    weights: u64,
    architecture: Option<&str>,
    config: &Option<EstimateConfig>,
) -> KvCacheEstimate {
    if is_memory_heavy_architecture(architecture) {
        return KvCacheEstimate {
            bytes: kv_cache_fallback_bytes(weights, architecture),
            confidence: EstimateConfidence::Low,
            config_used: false,
        };
    }

    if let Some(config) = config
        && let Some(config_estimate) = kv_cache_from_config(config)
    {
        return KvCacheEstimate {
            bytes: config_estimate.bytes,
            confidence: config_estimate.confidence,
            config_used: true,
        };
    }

    KvCacheEstimate {
        bytes: kv_cache_fallback_bytes(weights, architecture),
        confidence: EstimateConfidence::Low,
        config_used: false,
    }
}

fn kv_cache_from_config(config: &EstimateConfig) -> Option<KvCacheEstimate> {
    let hidden_size = config.hidden_size?;
    let layers = config.num_hidden_layers?;
    let attention_heads = config.num_attention_heads?;
    if attention_heads == 0 {
        return None;
    }
    let head_dim = config.head_dim.unwrap_or(hidden_size / attention_heads);
    if head_dim == 0 {
        return None;
    }
    let mut confidence = EstimateConfidence::High;
    let kv_heads = match config.num_key_value_heads {
        Some(kv_heads) => kv_heads,
        None => {
            confidence = EstimateConfidence::Medium;
            attention_heads
        }
    };
    let dtype_bytes = dtype_bytes(config.dtype.as_deref());
    if config.dtype.is_none() {
        confidence = confidence.min(EstimateConfidence::Medium);
    }
    let model_context = match config
        .sliding_window
        .or(config.max_position_embeddings)
        .filter(|ctx| *ctx > 0)
    {
        Some(context) => context,
        None => {
            confidence = confidence.min(EstimateConfidence::Medium);
            4096
        }
    };
    let effective_context = model_context.min(4096);
    Some(KvCacheEstimate {
        bytes: layers
            .saturating_mul(kv_heads)
            .saturating_mul(head_dim)
            .saturating_mul(2)
            .saturating_mul(effective_context)
            .saturating_mul(dtype_bytes),
        confidence,
        config_used: true,
    })
}

fn dtype_bytes(dtype: Option<&str>) -> u64 {
    let dtype = dtype.unwrap_or_default().to_ascii_lowercase();
    if ["fp32", "float32", "f32"]
        .iter()
        .any(|needle| dtype.contains(needle))
    {
        4
    } else if ["int8", "uint8", "i8", "u8"]
        .iter()
        .any(|needle| dtype.contains(needle))
    {
        1
    } else {
        2
    }
}

fn kv_cache_fallback_bytes(weights: u64, architecture: Option<&str>) -> u64 {
    let multiplier = if is_memory_heavy_architecture(architecture) {
        0.60
    } else {
        0.35
    };
    scale_bytes(weights, multiplier)
}

fn is_memory_heavy_architecture(architecture: Option<&str>) -> bool {
    let Some(architecture) = architecture else {
        return false;
    };
    let architecture = architecture.to_ascii_lowercase();
    ["jamba", "mamba", "mixtral", "moe"]
        .iter()
        .any(|needle| architecture.contains(needle))
}

fn scale_bytes(bytes: u64, factor: f64) -> u64 {
    ((bytes as f64) * factor).ceil() as u64
}

fn read_estimate_config(store: &ModelStore, manifest: &ModelManifest) -> Option<EstimateConfig> {
    let config_path = manifest.config_path.as_deref()?;
    let path = store.model_dir(&manifest.id).join(config_path);
    let data = fs::read_to_string(path).ok()?;
    let value: Value = serde_json::from_str(&data).ok()?;
    Some(parse_estimate_config(&value))
}

fn parse_estimate_config(value: &Value) -> EstimateConfig {
    let text_config = value.get("text_config").unwrap_or(value);
    EstimateConfig {
        hidden_size: first_config_u64(
            value,
            &[
                &["hidden_size"],
                &["n_embd"],
                &["text_config", "hidden_size"],
            ],
        ),
        num_hidden_layers: first_config_u64(
            value,
            &[
                &["num_hidden_layers"],
                &["n_layer"],
                &["num_layers"],
                &["text_config", "num_hidden_layers"],
            ],
        ),
        num_attention_heads: first_config_u64(
            value,
            &[
                &["num_attention_heads"],
                &["n_head"],
                &["text_config", "num_attention_heads"],
            ],
        ),
        num_key_value_heads: first_config_u64(
            value,
            &[
                &["num_key_value_heads"],
                &["n_head_kv"],
                &["text_config", "num_key_value_heads"],
            ],
        ),
        head_dim: first_config_u64(value, &[&["head_dim"], &["text_config", "head_dim"]]),
        max_position_embeddings: first_config_u64(
            value,
            &[
                &["max_position_embeddings"],
                &["seq_length"],
                &["context_length"],
                &["model_max_length"],
                &["n_positions"],
                &["n_ctx"],
                &["text_config", "max_position_embeddings"],
                &["text_config", "context_length"],
            ],
        ),
        sliding_window: first_config_u64(
            value,
            &[&["sliding_window"], &["text_config", "sliding_window"]],
        ),
        dtype: first_config_string(
            value,
            &[
                &["torch_dtype"],
                &["dtype"],
                &["text_config", "torch_dtype"],
            ],
        ),
        vocab_size: first_config_u64(value, &[&["vocab_size"], &["text_config", "vocab_size"]]),
        architectures: value
            .get("architectures")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(ToString::to_string)
            .collect(),
        model_type: text_config
            .get("model_type")
            .and_then(Value::as_str)
            .or_else(|| value.get("model_type").and_then(Value::as_str))
            .map(ToString::to_string),
    }
}

fn first_config_u64(value: &Value, paths: &[&[&str]]) -> Option<u64> {
    paths.iter().find_map(|path| {
        let mut current = value;
        for segment in *path {
            current = current.get(*segment)?;
        }
        current.as_u64()
    })
}

fn first_config_string(value: &Value, paths: &[&[&str]]) -> Option<String> {
    paths.iter().find_map(|path| {
        let mut current = value;
        for segment in *path {
            current = current.get(*segment)?;
        }
        current.as_str().map(ToString::to_string)
    })
}

fn latest_estimate_observation(
    store: &ModelStore,
    manifest: &ModelManifest,
) -> Option<EstimateObservation> {
    let path = store
        .home()
        .join("benchmarks")
        .join("estimate-observations.json");
    let data = fs::read_to_string(path).ok()?;
    let observations = serde_json::from_str::<Vec<EstimateObservation>>(&data).ok()?;
    observations
        .into_iter()
        .filter(|observation| observation.model == manifest.id)
        .filter(|observation| observation.measured_peak_memory_bytes.is_some())
        .max_by_key(|observation| observation.timestamp.unwrap_or(0))
}

fn estimate_backend_hint(manifest: &ModelManifest) -> &'static str {
    match manifest.format {
        ModelFormat::Mlx => "MLX",
        ModelFormat::Gguf => "llama.cpp",
        ModelFormat::Onnx => "ONNX Runtime",
        ModelFormat::SafeTensors if cfg!(target_os = "macos") => "MLX",
        ModelFormat::SafeTensors => "Candle",
        ModelFormat::PyTorch => "PyTorch",
        ModelFormat::TensorRt => "TensorRT",
        ModelFormat::OpenVino => "OpenVINO",
        ModelFormat::TensorFlow => "TensorFlow",
        ModelFormat::CoreMl => "Core ML",
        ModelFormat::Unknown => "unknown",
    }
}

fn format_label(format: &ModelFormat) -> &'static str {
    match format {
        ModelFormat::Gguf => "gguf",
        ModelFormat::SafeTensors => "safetensors",
        ModelFormat::PyTorch => "pytorch",
        ModelFormat::Onnx => "onnx",
        ModelFormat::Mlx => "mlx",
        ModelFormat::TensorRt => "tensorrt",
        ModelFormat::OpenVino => "openvino",
        ModelFormat::TensorFlow => "tensorflow",
        ModelFormat::CoreMl => "coreml",
        ModelFormat::Unknown => "unknown",
    }
}

fn detect_system_memory() -> SystemMemory {
    #[cfg(target_os = "linux")]
    {
        return linux_system_memory();
    }
    #[cfg(target_os = "macos")]
    {
        return macos_system_memory();
    }
    #[allow(unreachable_code)]
    SystemMemory {
        total_bytes: None,
        available_bytes: None,
    }
}

#[cfg(target_os = "linux")]
fn linux_system_memory() -> SystemMemory {
    let Ok(data) = fs::read_to_string("/proc/meminfo") else {
        return SystemMemory {
            total_bytes: None,
            available_bytes: None,
        };
    };
    let total_bytes = linux_meminfo_kib(&data, "MemTotal").map(|kib| kib.saturating_mul(1024));
    let available_bytes =
        linux_meminfo_kib(&data, "MemAvailable").map(|kib| kib.saturating_mul(1024));
    SystemMemory {
        total_bytes,
        available_bytes,
    }
}

#[cfg(target_os = "linux")]
fn linux_meminfo_kib(data: &str, key: &str) -> Option<u64> {
    data.lines().find_map(|line| {
        let (name, rest) = line.split_once(':')?;
        if name != key {
            return None;
        }
        rest.split_whitespace().next()?.parse().ok()
    })
}

#[cfg(target_os = "macos")]
fn macos_system_memory() -> SystemMemory {
    let vm_stat = command_stdout("vm_stat", &[]);
    SystemMemory {
        total_bytes: command_stdout("sysctl", &["-n", "hw.memsize"])
            .and_then(|text| text.trim().parse::<u64>().ok())
            .or_else(|| vm_stat.as_deref().and_then(macos_total_memory_from_vm_stat)),
        available_bytes: vm_stat
            .as_deref()
            .and_then(macos_available_memory_from_vm_stat),
    }
}

#[cfg(target_os = "macos")]
fn macos_available_memory_from_vm_stat(vm_stat: &str) -> Option<u64> {
    let page_size = parse_macos_vm_page_size(&vm_stat).or_else(|| {
        command_stdout("sysctl", &["-n", "hw.pagesize"])?
            .trim()
            .parse()
            .ok()
    })?;
    let pages = ["Pages free", "Pages inactive", "Pages speculative"]
        .iter()
        .filter_map(|name| parse_macos_vm_stat_pages(&vm_stat, name))
        .sum::<u64>();
    Some(pages.saturating_mul(page_size))
}

#[cfg(target_os = "macos")]
fn macos_total_memory_from_vm_stat(vm_stat: &str) -> Option<u64> {
    let page_size = parse_macos_vm_page_size(vm_stat).or_else(|| {
        command_stdout("sysctl", &["-n", "hw.pagesize"])?
            .trim()
            .parse()
            .ok()
    })?;
    let pages = [
        "Pages free",
        "Pages active",
        "Pages inactive",
        "Pages speculative",
        "Pages wired down",
        "Pages occupied by compressor",
    ]
    .iter()
    .filter_map(|name| parse_macos_vm_stat_pages(vm_stat, name))
    .sum::<u64>();
    if pages == 0 {
        return None;
    }
    Some(pages.saturating_mul(page_size))
}

#[cfg(target_os = "macos")]
fn parse_macos_vm_page_size(vm_stat: &str) -> Option<u64> {
    let first_line = vm_stat.lines().next()?;
    let marker = "page size of ";
    let start = first_line.find(marker)? + marker.len();
    let rest = &first_line[start..];
    rest.split_whitespace().next()?.parse().ok()
}

#[cfg(target_os = "macos")]
fn parse_macos_vm_stat_pages(vm_stat: &str, key: &str) -> Option<u64> {
    vm_stat.lines().find_map(|line| {
        let (name, rest) = line.split_once(':')?;
        if name.trim() != key {
            return None;
        }
        rest.trim()
            .trim_end_matches('.')
            .replace('.', "")
            .parse::<u64>()
            .ok()
    })
}

#[cfg(target_os = "macos")]
fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
}

fn terminal_spinner_enabled(debug: bool) -> bool {
    io::stderr().is_terminal() && !debug
}

fn with_terminal_spinner<T>(
    enabled: bool,
    message: impl Into<String>,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let spinner = enabled.then(|| {
        let progress = ProgressBar::new_spinner();
        progress.enable_steady_tick(Duration::from_millis(120));
        progress.set_style(ProgressStyle::with_template("{spinner:.cyan} {msg}").unwrap());
        progress.set_message(message.into());
        progress
    });

    let result = operation();
    if let Some(progress) = spinner {
        progress.finish_and_clear();
    }
    result
}

enum ChatInputReader {
    #[cfg(unix)]
    Terminal(TerminalLineReader),
    Stdin(io::Stdin),
}

impl ChatInputReader {
    fn new() -> Self {
        #[cfg(unix)]
        {
            if io::stdin().is_terminal() && io::stdout().is_terminal() {
                return Self::Terminal(TerminalLineReader::default());
            }
        }

        Self::Stdin(io::stdin())
    }

    fn read_line(&mut self, prompt: &str) -> Result<Option<String>> {
        match self {
            #[cfg(unix)]
            Self::Terminal(reader) => reader.read_line(prompt),
            Self::Stdin(stdin) => {
                print!("{prompt}");
                io::stdout().flush()?;

                let mut input = String::new();
                let n = stdin.read_line(&mut input)?;
                if n == 0 {
                    println!();
                    return Ok(None);
                }

                Ok(Some(input))
            }
        }
    }
}

#[cfg(unix)]
#[derive(Debug, Default)]
struct TerminalLineReader {
    history: Vec<String>,
}

#[cfg(unix)]
impl TerminalLineReader {
    fn read_line(&mut self, prompt: &str) -> Result<Option<String>> {
        let _raw_mode = TerminalRawMode::enable()?;
        {
            let mut stdout = io::stdout().lock();
            write!(stdout, "{prompt}")?;
            stdout.flush()?;
        }

        let stdin = io::stdin();
        let mut stdin = stdin.lock();
        let mut line = EditableLine::default();
        let mut draft = String::new();
        let mut history_index = None;

        loop {
            let byte = read_raw_byte(&mut stdin)?;
            let mut redraw = false;

            match byte {
                b'\r' | b'\n' => {
                    println!();
                    let input = line.as_string();
                    self.push_history(&input);
                    return Ok(Some(input));
                }
                0x04 if line.is_empty() => {
                    println!();
                    return Ok(None);
                }
                0x03 => {
                    println!("^C");
                    return Ok(None);
                }
                0x01 => redraw = line.move_home(),
                0x05 => redraw = line.move_end(),
                0x0b => redraw = line.delete_to_end(),
                0x15 => redraw = line.clear(),
                0x17 => redraw = line.delete_word_before_cursor(),
                0x7f | 0x08 => redraw = line.backspace(),
                b'\x1b' => {
                    let command = read_escape_command(&mut stdin)?;
                    redraw = self.apply_command(command, &mut line, &mut draft, &mut history_index);
                }
                byte if byte >= 0x20 => {
                    if let Some(ch) = read_utf8_char(byte, &mut stdin)? {
                        line.insert(ch);
                        redraw = true;
                    }
                }
                _ => {}
            }

            if redraw {
                redraw_editable_line(prompt, &line)?;
            }
        }
    }

    fn push_history(&mut self, input: &str) {
        if input.trim().is_empty() || self.history.last().is_some_and(|last| last == input) {
            return;
        }

        self.history.push(input.to_string());
        const HISTORY_LIMIT: usize = 200;
        if self.history.len() > HISTORY_LIMIT {
            self.history.remove(0);
        }
    }

    fn apply_command(
        &self,
        command: LineEditCommand,
        line: &mut EditableLine,
        draft: &mut String,
        history_index: &mut Option<usize>,
    ) -> bool {
        match command {
            LineEditCommand::None => false,
            LineEditCommand::MoveLeft => line.move_left(),
            LineEditCommand::MoveRight => line.move_right(),
            LineEditCommand::MoveWordLeft => line.move_word_left(),
            LineEditCommand::MoveWordRight => line.move_word_right(),
            LineEditCommand::MoveHome => line.move_home(),
            LineEditCommand::MoveEnd => line.move_end(),
            LineEditCommand::Delete => line.delete(),
            LineEditCommand::HistoryPrevious => self.history_previous(line, draft, history_index),
            LineEditCommand::HistoryNext => self.history_next(line, draft, history_index),
        }
    }

    fn history_previous(
        &self,
        line: &mut EditableLine,
        draft: &mut String,
        history_index: &mut Option<usize>,
    ) -> bool {
        if self.history.is_empty() {
            return false;
        }

        let next_index = match *history_index {
            Some(0) => return false,
            Some(index) => index - 1,
            None => {
                *draft = line.as_string();
                self.history.len() - 1
            }
        };

        *history_index = Some(next_index);
        line.replace(&self.history[next_index]);
        true
    }

    fn history_next(
        &self,
        line: &mut EditableLine,
        draft: &str,
        history_index: &mut Option<usize>,
    ) -> bool {
        let Some(index) = *history_index else {
            return false;
        };

        if index + 1 < self.history.len() {
            let next_index = index + 1;
            *history_index = Some(next_index);
            line.replace(&self.history[next_index]);
        } else {
            *history_index = None;
            line.replace(draft);
        }

        true
    }
}

#[cfg(unix)]
#[derive(Debug, Default)]
struct EditableLine {
    buffer: Vec<char>,
    cursor: usize,
}

#[cfg(unix)]
impl EditableLine {
    fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    fn as_string(&self) -> String {
        self.buffer.iter().collect()
    }

    fn replace(&mut self, value: &str) {
        self.buffer = value.chars().collect();
        self.cursor = self.buffer.len();
    }

    fn insert(&mut self, ch: char) {
        self.buffer.insert(self.cursor, ch);
        self.cursor += 1;
    }

    fn backspace(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }

        self.cursor -= 1;
        self.buffer.remove(self.cursor);
        true
    }

    fn delete(&mut self) -> bool {
        if self.cursor >= self.buffer.len() {
            return false;
        }

        self.buffer.remove(self.cursor);
        true
    }

    fn delete_to_end(&mut self) -> bool {
        if self.cursor >= self.buffer.len() {
            return false;
        }

        self.buffer.truncate(self.cursor);
        true
    }

    fn clear(&mut self) -> bool {
        if self.buffer.is_empty() {
            return false;
        }

        self.buffer.clear();
        self.cursor = 0;
        true
    }

    fn delete_word_before_cursor(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }

        let original_cursor = self.cursor;
        while self.cursor > 0 && self.buffer[self.cursor - 1].is_whitespace() {
            self.cursor -= 1;
        }
        while self.cursor > 0 && !self.buffer[self.cursor - 1].is_whitespace() {
            self.cursor -= 1;
        }
        self.buffer.drain(self.cursor..original_cursor);
        true
    }

    fn move_left(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }

        self.cursor -= 1;
        true
    }

    fn move_right(&mut self) -> bool {
        if self.cursor >= self.buffer.len() {
            return false;
        }

        self.cursor += 1;
        true
    }

    fn move_word_left(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }

        while self.cursor > 0 && self.buffer[self.cursor - 1].is_whitespace() {
            self.cursor -= 1;
        }
        while self.cursor > 0 && !self.buffer[self.cursor - 1].is_whitespace() {
            self.cursor -= 1;
        }
        true
    }

    fn move_word_right(&mut self) -> bool {
        if self.cursor >= self.buffer.len() {
            return false;
        }

        while self.cursor < self.buffer.len() && !self.buffer[self.cursor].is_whitespace() {
            self.cursor += 1;
        }
        while self.cursor < self.buffer.len() && self.buffer[self.cursor].is_whitespace() {
            self.cursor += 1;
        }
        true
    }

    fn move_home(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }

        self.cursor = 0;
        true
    }

    fn move_end(&mut self) -> bool {
        if self.cursor == self.buffer.len() {
            return false;
        }

        self.cursor = self.buffer.len();
        true
    }
}

#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineEditCommand {
    None,
    MoveLeft,
    MoveRight,
    MoveWordLeft,
    MoveWordRight,
    MoveHome,
    MoveEnd,
    Delete,
    HistoryPrevious,
    HistoryNext,
}

#[cfg(unix)]
struct TerminalRawMode {
    fd: libc::c_int,
    original: libc::termios,
}

#[cfg(unix)]
impl TerminalRawMode {
    fn enable() -> Result<Self> {
        let fd = io::stdin().as_raw_fd();
        let mut original = unsafe { std::mem::zeroed::<libc::termios>() };
        if unsafe { libc::tcgetattr(fd, &mut original) } != 0 {
            return Err(io::Error::last_os_error().into());
        }

        let mut raw = original;
        raw.c_lflag &= !(libc::ICANON | libc::ECHO | libc::ISIG);
        raw.c_cc[libc::VMIN] = 0;
        raw.c_cc[libc::VTIME] = 1;

        if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &raw) } != 0 {
            return Err(io::Error::last_os_error().into());
        }

        Ok(Self { fd, original })
    }
}

#[cfg(unix)]
impl Drop for TerminalRawMode {
    fn drop(&mut self) {
        unsafe {
            libc::tcsetattr(self.fd, libc::TCSANOW, &self.original);
        }
    }
}

#[cfg(unix)]
fn read_raw_byte(reader: &mut impl Read) -> Result<u8> {
    loop {
        if let Some(byte) = read_raw_byte_optional(reader)? {
            return Ok(byte);
        }
    }
}

#[cfg(unix)]
fn read_raw_byte_optional(reader: &mut impl Read) -> Result<Option<u8>> {
    let mut byte = [0u8; 1];
    loop {
        match reader.read(&mut byte) {
            Ok(0) => return Ok(None),
            Ok(_) => return Ok(Some(byte[0])),
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(err.into()),
        }
    }
}

#[cfg(unix)]
fn read_utf8_char(first_byte: u8, reader: &mut impl Read) -> Result<Option<char>> {
    if first_byte < 0x80 {
        return Ok(Some(first_byte as char));
    }

    let width = match first_byte {
        0xc2..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf4 => 4,
        _ => return Ok(None),
    };

    let mut bytes = vec![first_byte];
    for _ in 1..width {
        let Some(byte) = read_raw_byte_optional(reader)? else {
            return Ok(None);
        };
        bytes.push(byte);
    }

    Ok(std::str::from_utf8(&bytes)
        .ok()
        .and_then(|value| value.chars().next()))
}

#[cfg(unix)]
fn read_escape_command(reader: &mut impl Read) -> Result<LineEditCommand> {
    let Some(first) = read_raw_byte_optional(reader)? else {
        return Ok(LineEditCommand::None);
    };

    match first {
        b'b' => Ok(LineEditCommand::MoveWordLeft),
        b'f' => Ok(LineEditCommand::MoveWordRight),
        b'[' => read_csi_command(reader),
        b'O' => match read_raw_byte_optional(reader)? {
            Some(b'H') => Ok(LineEditCommand::MoveHome),
            Some(b'F') => Ok(LineEditCommand::MoveEnd),
            _ => Ok(LineEditCommand::None),
        },
        _ => Ok(LineEditCommand::None),
    }
}

#[cfg(unix)]
fn read_csi_command(reader: &mut impl Read) -> Result<LineEditCommand> {
    let Some(first) = read_raw_byte_optional(reader)? else {
        return Ok(LineEditCommand::None);
    };

    match first {
        b'A' => Ok(LineEditCommand::HistoryPrevious),
        b'B' => Ok(LineEditCommand::HistoryNext),
        b'C' => Ok(LineEditCommand::MoveRight),
        b'D' => Ok(LineEditCommand::MoveLeft),
        b'H' => Ok(LineEditCommand::MoveHome),
        b'F' => Ok(LineEditCommand::MoveEnd),
        b'1'..=b'9' => read_numbered_csi_command(first, reader),
        _ => Ok(LineEditCommand::None),
    }
}

#[cfg(unix)]
fn read_numbered_csi_command(first: u8, reader: &mut impl Read) -> Result<LineEditCommand> {
    let mut bytes = vec![first];

    loop {
        let Some(byte) = read_raw_byte_optional(reader)? else {
            return Ok(LineEditCommand::None);
        };

        match byte {
            b'~' => {
                return Ok(match bytes.as_slice() {
                    b"1" | b"7" => LineEditCommand::MoveHome,
                    b"3" => LineEditCommand::Delete,
                    b"4" | b"8" => LineEditCommand::MoveEnd,
                    _ => LineEditCommand::None,
                });
            }
            b'C' => {
                return Ok(if bytes.contains(&b';') {
                    LineEditCommand::MoveWordRight
                } else {
                    LineEditCommand::MoveRight
                });
            }
            b'D' => {
                return Ok(if bytes.contains(&b';') {
                    LineEditCommand::MoveWordLeft
                } else {
                    LineEditCommand::MoveLeft
                });
            }
            b'H' => return Ok(LineEditCommand::MoveHome),
            b'F' => return Ok(LineEditCommand::MoveEnd),
            byte if byte.is_ascii_digit() || byte == b';' => bytes.push(byte),
            _ => return Ok(LineEditCommand::None),
        }
    }
}

#[cfg(unix)]
fn redraw_editable_line(prompt: &str, line: &EditableLine) -> Result<()> {
    let text = line.as_string();
    let chars_after_cursor = line.buffer.len().saturating_sub(line.cursor);
    let mut stdout = io::stdout().lock();
    write!(stdout, "\r\x1b[2K{prompt}{text}")?;
    if chars_after_cursor > 0 {
        write!(stdout, "\x1b[{chars_after_cursor}D")?;
    }
    stdout.flush()?;
    Ok(())
}

#[derive(Debug)]
struct AssistantPendingSpinner {
    enabled: bool,
    visible: bool,
    frame_index: usize,
}

impl AssistantPendingSpinner {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            visible: false,
            frame_index: 0,
        }
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn tick(&mut self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        const FRAMES: [&str; 4] = ["-", "\\", "|", "/"];
        let frame = FRAMES[self.frame_index % FRAMES.len()];
        self.frame_index += 1;
        self.visible = true;

        let mut stdout = io::stdout().lock();
        write!(stdout, "\r\x1b[2Kassistant> {frame}")?;
        stdout.flush()?;
        Ok(())
    }

    fn clear(&mut self) -> Result<()> {
        if !self.visible {
            return Ok(());
        }

        self.visible = false;
        let mut stdout = io::stdout().lock();
        write!(stdout, "\r\x1b[2Kassistant> ")?;
        stdout.flush()?;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
async fn chat_loop(
    backend: Arc<dyn GenerationBackend>,
    manifest: ModelManifest,
    selected_backend: BackendChoice,
    max_tokens: usize,
    temperature: Option<f64>,
    top_p: Option<f64>,
    seed: Option<u64>,
    history_enabled: bool,
    chat_template: Option<ChatTemplateArg>,
    images: Vec<String>,
    stream_granularity: StreamGranularity,
    verbose: bool,
    debug: bool,
    show_loading_spinner: bool,
) -> Result<()> {
    let chat_session =
        prepare_backend_for_chat(backend.as_ref(), &manifest, seed, show_loading_spinner)?;

    println!(
        "Chatting with {}. Type /exit or /quit to stop.",
        manifest.id
    );
    let mut messages = Vec::new();
    let mut input_reader = ChatInputReader::new();

    loop {
        let Some(input) = input_reader.read_line("you> ")? else {
            break;
        };

        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if matches!(input, "/exit" | "/quit") {
            break;
        }

        let user_message = ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text(input.to_string())),
            name: None,
        };

        let request_messages =
            request_messages_for_turn(&mut messages, user_message, history_enabled);
        let prompt = prompt_for_backend(
            &manifest,
            &request_messages,
            selected_backend,
            chat_template,
        );
        let prompt_diagnostics =
            prompt_diagnostics(&prompt, request_messages.len(), Some(history_enabled));
        let generation_messages = generation_request_messages(&prompt, &request_messages);
        let request = GenerateRequest {
            prompt: prompt.prompt,
            messages: generation_messages,
            image_urls: images.clone(),
            max_tokens,
            temperature,
            top_p,
            stop: prompt.stop,
            seed,
            stream_granularity,
            verbose,
            debug,
        };

        print!("assistant> ");
        io::stdout().flush()?;

        let mut assistant = String::new();
        let mut prompt_tokens = 0usize;
        let mut completion_tokens = 0usize;
        let mut finish_reason = String::new();
        let mut timings = None;
        let mut backend_diagnostics = Vec::new();
        let mut last_flush = Instant::now();
        let mut pending_spinner =
            AssistantPendingSpinner::new(io::stdout().is_terminal() && !debug);
        let mut stream = if let Some(session) = chat_session.as_ref() {
            session.generate_stream(request)
        } else {
            backend.generate_stream(manifest.clone(), request)
        };
        loop {
            let event = tokio::time::timeout(Duration::from_millis(120), stream.next()).await;
            let Some(event) = (match event {
                Ok(event) => event,
                Err(_) => {
                    if assistant.is_empty() && pending_spinner.is_enabled() {
                        pending_spinner.tick()?;
                    }
                    continue;
                }
            }) else {
                pending_spinner.clear()?;
                break;
            };

            match event {
                Ok(GenerateStreamEvent::TextChunk(chunk)) => {
                    if !chunk.is_empty() {
                        pending_spinner.clear()?;
                    }
                    print!("{chunk}");
                    if chunk.contains('\n') || last_flush.elapsed() >= Duration::from_millis(16) {
                        io::stdout().flush()?;
                        last_flush = Instant::now();
                    }
                    assistant.push_str(&chunk);
                }
                Ok(GenerateStreamEvent::Done {
                    finish_reason: response_finish_reason,
                    prompt_tokens: tokens_in,
                    completion_tokens: tokens,
                    timings: response_timings,
                    backend_diagnostics: response_backend_diagnostics,
                }) => {
                    finish_reason = response_finish_reason;
                    prompt_tokens = tokens_in;
                    completion_tokens = tokens;
                    timings = Some(response_timings);
                    backend_diagnostics = prompt_diagnostics.clone();
                    backend_diagnostics.extend(response_backend_diagnostics);
                    pending_spinner.clear()?;
                    break;
                }
                Err(message) => {
                    pending_spinner.clear()?;
                    println!("\nerror: {message}");
                    break;
                }
            }
        }
        io::stdout().flush()?;
        println!();
        if matches!(finish_reason.as_str(), "length" | "max_new_tokens")
            && !assistant.trim().is_empty()
        {
            println!(
                "note: response reached --max-tokens ({max_tokens}) and may be incomplete; rerun with a larger --max-tokens value for more."
            );
        }
        if verbose && let Some(timings) = timings {
            let mut stdout = io::stdout().lock();
            writeln!(stdout)?;
            write_verbose_stats(
                &mut stdout,
                Some(verbose_backend_label(selected_backend)),
                prompt_tokens,
                completion_tokens,
                &finish_reason,
                timings,
                &backend_diagnostics,
            )?;
        }

        if history_enabled && !assistant.trim().is_empty() {
            messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: Some(MessageContent::Text(assistant)),
                name: None,
            });
        }
    }

    Ok(())
}

fn request_messages_for_turn(
    history: &mut Vec<ChatMessage>,
    user_message: ChatMessage,
    history_enabled: bool,
) -> Vec<ChatMessage> {
    if history_enabled {
        history.push(user_message);
        history.clone()
    } else {
        vec![user_message]
    }
}

fn prompt_for_backend(
    manifest: &ModelManifest,
    messages: &[ChatMessage],
    backend: BackendChoice,
    override_arg: Option<ChatTemplateArg>,
) -> PromptSpec {
    messages_to_prompt_for_model_with_template(
        manifest,
        messages,
        chat_template_options_for_backend(manifest, backend, override_arg),
    )
}

fn generation_request_messages(prompt: &PromptSpec, messages: &[ChatMessage]) -> Vec<ChatMessage> {
    if prompt.chat_template.source == ChatTemplateSource::Model {
        messages.to_vec()
    } else {
        Vec::new()
    }
}

fn chat_template_options_for_backend(
    manifest: &ModelManifest,
    backend: BackendChoice,
    override_arg: Option<ChatTemplateArg>,
) -> ChatTemplateOptions<'static> {
    ChatTemplateOptions {
        default_source: chat_template_default_source(manifest, backend),
        model_template_preferred: chat_template_model_preferred(manifest, backend),
        override_name: override_arg.map(ChatTemplateArg::template_name),
    }
}

fn chat_template_default_source(
    manifest: &ModelManifest,
    backend: BackendChoice,
) -> ChatTemplateSource {
    match backend {
        BackendChoice::LlamaServer(_)
        | BackendChoice::LlamaFast(_)
        | BackendChoice::LlamaHighlevel(_)
            if manifest.format == ModelFormat::Gguf =>
        {
            ChatTemplateSource::Model
        }
        BackendChoice::Mlx
        | BackendChoice::MlxVlm
        | BackendChoice::TransformersCompat
        | BackendChoice::Vllm
        | BackendChoice::VllmRocm => ChatTemplateSource::Model,
        _ => ChatTemplateSource::Werk,
    }
}

fn chat_template_model_preferred(manifest: &ModelManifest, backend: BackendChoice) -> bool {
    matches!(backend, BackendChoice::TransformersCompat)
        || matches!(
            backend,
            BackendChoice::LlamaServer(_)
                | BackendChoice::LlamaFast(_)
                | BackendChoice::LlamaHighlevel(_)
        ) && manifest.format == ModelFormat::Gguf
        || matches!(backend, BackendChoice::Vllm | BackendChoice::VllmRocm)
}

fn prompt_diagnostics(
    prompt: &PromptSpec,
    message_count: usize,
    history_enabled: Option<bool>,
) -> Vec<String> {
    let mut diagnostics = vec![
        format!("history messages: {message_count}"),
        format!(
            "chat template source: {}",
            prompt.chat_template.source.as_str()
        ),
        format!("chat template: {}", prompt.chat_template.name),
        format!(
            "chat template applied by werk: {}",
            if prompt.chat_template.applied_by_werk {
                "yes"
            } else {
                "no"
            }
        ),
    ];
    if let Some(override_name) = &prompt.chat_template.override_from_cli {
        diagnostics.push(format!("chat template override: {override_name}"));
    }
    if let Some(token) = prompt.assistant_end_token {
        diagnostics.push(format!("assistant end token: {token}"));
    }
    if let Some(history_enabled) = history_enabled {
        diagnostics.push(format!(
            "history enabled: {}",
            if history_enabled { "yes" } else { "no" }
        ));
    }
    diagnostics
}

fn merged_diagnostics(first: &[String], second: &[String]) -> Vec<String> {
    let mut merged = first.to_vec();
    merged.extend_from_slice(second);
    merged
}

fn prompt_huggingface_token() -> Result<String> {
    print!("Hugging Face token: ");
    io::stdout().flush()?;
    let mut token = String::new();
    io::stdin().read_line(&mut token)?;
    let token = token.trim().to_string();
    if token.is_empty() {
        bail!("Hugging Face token cannot be empty");
    }
    Ok(token)
}

fn prepare_backend_for_chat(
    backend: &dyn GenerationBackend,
    manifest: &ModelManifest,
    seed: Option<u64>,
    show_loading_spinner: bool,
) -> Result<Option<Box<dyn ChatGenerationSession>>> {
    with_terminal_spinner(
        show_loading_spinner,
        format!("Loading model '{}'...", manifest.id),
        || {
            backend.prepare(manifest)?;
            backend.start_chat_session(manifest, seed)
        },
    )
}

#[derive(Debug, Serialize)]
struct BenchReport {
    model: String,
    prompt: String,
    max_tokens: usize,
    warmups: usize,
    runs: usize,
    temperature: f64,
    top_p: Option<f64>,
    seed: u64,
    compare: BenchCompareArg,
    results: Vec<BenchBackendReport>,
}

#[derive(Debug, Serialize)]
struct BenchBackendReport {
    backend: &'static str,
    runtime: Option<LlamaFastRuntimeReport>,
    samples: Vec<BenchSample>,
    median_eval_tokens_per_second: Option<f64>,
    median_total_seconds: Option<f64>,
    median_first_token_seconds: Option<f64>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct BenchSample {
    prompt_tokens: usize,
    completion_tokens: usize,
    load_seconds: f64,
    warmup_seconds: f64,
    first_token_seconds: f64,
    total_seconds: f64,
    prompt_seconds: f64,
    decode_seconds: f64,
    eval_tokens_per_second: f64,
}

#[allow(clippy::too_many_arguments)]
fn bench_model(
    store: ModelStore,
    manifest: ModelManifest,
    backend_choice: BackendChoice,
    runtime_options: LlamaRuntimeOptions,
    prompt: String,
    max_tokens: usize,
    runs: usize,
    warmups: usize,
    temperature: f64,
    top_p: Option<f64>,
    seed: u64,
    compare: BenchCompareArg,
    debug: bool,
    selection_options: SelectionOptions,
) -> Result<BenchReport> {
    if runs == 0 {
        bail!("--runs must be greater than 0");
    }

    let prompt_spec = messages_to_prompt_for_model(
        &manifest,
        &[ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text(prompt.clone())),
            name: None,
        }],
    );
    let choices = benchmark_choices(backend_choice, &manifest, compare);
    let mut results = Vec::with_capacity(choices.len());

    for choice in choices {
        let backend_label = backend_label(choice);
        let runtime = runtime_report_for_choice(choice, &runtime_options);
        let result = run_benchmark_choice(
            store.clone(),
            &manifest,
            choice,
            runtime_options.clone(),
            &prompt_spec.prompt,
            &prompt_spec.stop,
            max_tokens,
            runs,
            warmups,
            temperature,
            top_p,
            seed,
            debug,
            selection_options,
        );
        results.push(match result {
            Ok(samples) => BenchBackendReport {
                backend: backend_label,
                runtime,
                median_eval_tokens_per_second: median(
                    samples
                        .iter()
                        .map(|sample| sample.eval_tokens_per_second)
                        .collect(),
                ),
                median_total_seconds: median(
                    samples.iter().map(|sample| sample.total_seconds).collect(),
                ),
                median_first_token_seconds: median(
                    samples
                        .iter()
                        .map(|sample| sample.first_token_seconds)
                        .collect(),
                ),
                samples,
                error: None,
            },
            Err(err) => BenchBackendReport {
                backend: backend_label,
                runtime,
                samples: Vec::new(),
                median_eval_tokens_per_second: None,
                median_total_seconds: None,
                median_first_token_seconds: None,
                error: Some(err.to_string()),
            },
        });
    }

    Ok(BenchReport {
        model: manifest.id,
        prompt,
        max_tokens,
        warmups,
        runs,
        temperature,
        top_p,
        seed,
        compare,
        results,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_benchmark_choice(
    store: ModelStore,
    manifest: &ModelManifest,
    choice: BackendChoice,
    runtime_options: LlamaRuntimeOptions,
    prompt: &str,
    stop: &[String],
    max_tokens: usize,
    runs: usize,
    warmups: usize,
    temperature: f64,
    top_p: Option<f64>,
    seed: u64,
    debug: bool,
    selection_options: SelectionOptions,
) -> Result<Vec<BenchSample>> {
    let backend = build_concrete_backend(store, choice, runtime_options, selection_options)?;
    backend.prepare(manifest)?;
    let session = backend.start_chat_session(manifest, Some(seed))?;

    let mut samples = Vec::with_capacity(runs);
    for index in 0..warmups + runs {
        let request = GenerateRequest {
            prompt: prompt.to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some(MessageContent::Text(prompt.to_string())),
                name: None,
            }],
            image_urls: Vec::new(),
            max_tokens,
            temperature: Some(temperature),
            top_p,
            stop: stop.to_vec(),
            seed: Some(seed),
            stream_granularity: StreamGranularity::Chunk,
            verbose: false,
            debug,
        };
        let response = if let Some(session) = session.as_ref() {
            session.generate(request)?
        } else {
            backend.generate(manifest, request)?
        };

        if index >= warmups {
            samples.push(BenchSample {
                prompt_tokens: response.prompt_tokens,
                completion_tokens: response.completion_tokens,
                load_seconds: response.timings.load_seconds,
                warmup_seconds: response.timings.warmup_seconds,
                first_token_seconds: response.timings.first_token_seconds,
                total_seconds: response.timings.total_seconds,
                prompt_seconds: response.timings.prompt_seconds,
                decode_seconds: response.timings.decode_seconds,
                eval_tokens_per_second: rate(
                    response.completion_tokens,
                    response.timings.decode_seconds,
                ),
            });
        }
    }
    Ok(samples)
}

fn benchmark_choices(
    choice: BackendChoice,
    manifest: &ModelManifest,
    compare: BenchCompareArg,
) -> Vec<BackendChoice> {
    if manifest.format != ModelFormat::Gguf {
        return vec![choice];
    }

    match choice {
        BackendChoice::Auto => {
            let mode = preferred_llama_mode();
            benchmark_llama_choices(mode, compare)
        }
        BackendChoice::GgufPreferred { llama, .. } | BackendChoice::LlamaServer(llama) => {
            benchmark_llama_choices(llama, compare)
        }
        BackendChoice::LlamaFast(_) | BackendChoice::LlamaHighlevel(_) => vec![choice],
        _ => vec![choice],
    }
}

fn benchmark_llama_choices(mode: LlamaCppMode, compare: BenchCompareArg) -> Vec<BackendChoice> {
    let mut choices = vec![BackendChoice::LlamaServer(mode)];
    if compare == BenchCompareArg::Legacy {
        choices.push(BackendChoice::LlamaFast(mode));
    }
    choices
}

fn print_bench_report(report: &BenchReport, print_native_info: bool) {
    println!("Benchmark: {}", report.model);
    println!(
        "runs: {}, warmups: {}, max tokens: {}, temperature: {}, seed: {}",
        report.runs, report.warmups, report.max_tokens, report.temperature, report.seed
    );
    for result in &report.results {
        println!();
        println!("backend: {}", result.backend);
        if print_native_info && let Some(runtime) = &result.runtime {
            print_runtime_report(runtime);
        }
        if let Some(error) = &result.error {
            println!("error: {error}");
            continue;
        }
        if let Some(rate) = result.median_eval_tokens_per_second {
            println!("median eval rate: {rate:.2} tokens/s");
        }
        if let Some(total) = result.median_total_seconds {
            println!("median total: {}", format_duration(total));
        }
        if let Some(first_token) = result.median_first_token_seconds {
            println!("median first token: {}", format_duration(first_token));
        }
        for (index, sample) in result.samples.iter().enumerate() {
            println!(
                "  run {:>2}: {:>7.2} tok/s, {} token(s), first {}, total {}",
                index + 1,
                sample.eval_tokens_per_second,
                sample.completion_tokens,
                format_duration(sample.first_token_seconds),
                format_duration(sample.total_seconds)
            );
        }
    }
}

fn runtime_report_for_choice(
    choice: BackendChoice,
    runtime_options: &LlamaRuntimeOptions,
) -> Option<LlamaFastRuntimeReport> {
    match choice {
        BackendChoice::LlamaFast(mode) => {
            Some(LlamaFastBackend::runtime_report(mode, runtime_options))
        }
        _ => None,
    }
}

fn print_perf_doctor(
    store: &ModelStore,
    manifest: &ModelManifest,
    backend_choice: BackendChoice,
    runtime_options: &LlamaRuntimeOptions,
    selection_options: SelectionOptions,
) -> Result<()> {
    let selected =
        selected_backend_for_request(store, backend_choice, manifest, false, selection_options)?;
    println!("Werk1112 performance diagnostics");
    println!("model: {}", manifest.id);
    println!("format: {:?}", manifest.format);
    println!(
        "architecture: {}",
        manifest.architecture.as_deref().unwrap_or("unknown")
    );
    println!("selected backend: {}", backend_label(selected));

    if let Some(report) = runtime_report_for_choice(selected, runtime_options) {
        print_runtime_report(&report);
    } else {
        println!("runtime: {}", backend_label(selected));
        println!(
            "note: detailed legacy FFI diagnostics are available only for llama-legacy backends"
        );
    }

    Ok(())
}

fn print_backend_list(store: &ModelStore) {
    println!("llama-server discovery");
    println!(
        "{:<8} {:<16} {:<7} {:<7} PATH",
        "BACKEND", "SOURCE", "EXISTS", "HELP"
    );
    for mode in [
        LlamaCppMode::Cuda,
        LlamaCppMode::Rocm,
        LlamaCppMode::Vulkan,
        LlamaCppMode::Metal,
        LlamaCppMode::Cpu,
    ] {
        let discovery = LlamaServerBackend::discover(store, mode);
        print_backend_discovery(&discovery);
    }

    #[cfg(feature = "burn-experimental")]
    {
        println!();
        println!("Burn runtime");
        println!("{:<8} {:<16} {:<7} DETAIL", "BACKEND", "SOURCE", "READY");
        for mode in [BurnMode::Cuda, BurnMode::Cpu] {
            print_burn_discovery(store, mode);
        }
    }

    println!();
    println!("ONNX Runtime discovery");
    println!(
        "{:<8} {:<16} {:<7} {:<7} PATH",
        "BACKEND", "SOURCE", "EXISTS", "HELP"
    );
    for mode in [
        OnnxRuntimeMode::Cuda,
        OnnxRuntimeMode::Rocm,
        OnnxRuntimeMode::Cpu,
    ] {
        print_onnxruntime_discovery(store, mode);
    }

    println!();
    println!("vLLM discovery");
    let discovery = VllmBackend::discover(store);
    let health = VllmBackend::health(store);
    let vllm_path = discovery
        .attempts
        .iter()
        .find(|attempt| attempt.usable)
        .and_then(|attempt| attempt.path.as_ref())
        .or_else(|| {
            discovery
                .attempts
                .iter()
                .find(|attempt| attempt.label == "managed venv")
                .and_then(|attempt| attempt.path.as_ref())
        })
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| managed_vllm_dir(store).join("venv").display().to_string());
    println!(
        "{:<8} {:<16} {:<10} {:<18} PATH",
        "BACKEND", "SOURCE", "INSTALLED", "HEALTH"
    );
    println!(
        "{:<8} {:<16} {:<10} {:<18} {}",
        "vLLM", discovery.source, health.installed_label, health.health_label, vllm_path
    );

    println!();
    println!(
        "{:<24} {:<12} {:<12} {:<8} INSTALL",
        "RUNTIME", "STATE", "ACCEL", "VLM"
    );
    for runtime in runtime_registry().iter().filter(|runtime| {
        cfg!(feature = "burn-experimental") || runtime.runtime != BackendRuntime::Burn
    }) {
        println!(
            "{:<24} {:<12} {:<12} {:<8} {}",
            runtime.display_name,
            if runtime.implemented {
                "implemented"
            } else {
                "pending"
            },
            runtime
                .accelerators
                .iter()
                .map(|accelerator| format!("{accelerator:?}"))
                .collect::<Vec<_>>()
                .join("/"),
            yes_no(runtime.capabilities.vision_language),
            runtime.install_target.unwrap_or("-")
        );
    }
}

#[cfg(feature = "burn-experimental")]
fn print_burn_discovery(store: &ModelStore, mode: BurnMode) {
    let _ = store;
    let status = BurnBackend::runtime_status(mode);
    println!(
        "{:<8} {:<16} {:<7} {}",
        match mode {
            BurnMode::Cuda => "CUDA",
            BurnMode::Cpu => "CPU",
        },
        "in-process",
        yes_no(status.available),
        status.detail
    );
}

fn print_backend_discovery(discovery: &LlamaServerDiscovery) {
    let path = discovery
        .path
        .as_ref()
        .or_else(|| {
            discovery
                .attempts
                .iter()
                .find(|attempt| attempt.label == "managed cache")
                .and_then(|attempt| attempt.path.as_ref())
        })
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "-".to_string());
    let exists = discovery.path.is_some();
    let help = discovery
        .path
        .as_deref()
        .map(llama_server_help_ok)
        .unwrap_or(false);
    println!(
        "{:<8} {:<16} {:<7} {:<7} {}",
        display_llama_mode(discovery.mode),
        discovery.source,
        yes_no(exists),
        yes_no(help),
        path
    );
}

fn print_onnxruntime_discovery(store: &ModelStore, mode: OnnxRuntimeMode) {
    let discovery = OnnxRuntimeBackend::discover(store, mode);
    let path = discovery
        .path
        .as_ref()
        .or_else(|| {
            discovery
                .attempts
                .iter()
                .find(|attempt| attempt.label == "managed cache")
                .and_then(|attempt| attempt.path.as_ref())
        })
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| managed_onnx_runner_path(store, mode).display().to_string());
    let help = discovery
        .attempts
        .iter()
        .find(|attempt| attempt.usable)
        .map(|attempt| attempt.usable)
        .unwrap_or(false);
    println!(
        "{:<8} {:<16} {:<7} {:<7} {}",
        match mode {
            OnnxRuntimeMode::Cuda => "CUDA",
            OnnxRuntimeMode::Rocm => "ROCm",
            OnnxRuntimeMode::Cpu => "CPU",
        },
        discovery.source,
        yes_no(discovery.path.is_some()),
        yes_no(help),
        path
    );
}

fn print_backend_doctor(store: &ModelStore, debug: bool) {
    println!("Werk1112 backend diagnostics");
    println!(
        "executable: {}",
        env::current_exe()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|err| format!("unknown ({err})"))
    );
    println!("compiled runtimes: {}", compiled_runtime_summary());
    println!("managed cache: {}", store.home().join("backends").display());
    println!(
        "CUDA cache: {}",
        managed_backend_dir(store, LlamaCppMode::Cuda).display()
    );
    println!(
        "ROCm cache: {}",
        managed_backend_dir(store, LlamaCppMode::Rocm).display()
    );
    println!(
        "Metal cache: {}",
        managed_backend_dir(store, LlamaCppMode::Metal).display()
    );
    println!();
    for check in backend_doctor_checks(store) {
        println!(
            "{:<24} {:<7} {}",
            check.name,
            doctor_check_status(&check),
            check.detail
        );
    }
    for check in vllm_doctor_checks(store) {
        println!(
            "{:<24} {:<7} {}",
            check.name,
            doctor_check_status(&check),
            check.detail
        );
    }
    #[cfg(feature = "burn-experimental")]
    for check in burn_doctor_checks() {
        println!(
            "{:<24} {:<7} {}",
            check.name,
            if check.ok { "ok" } else { "missing" },
            check.detail
        );
    }
    println!();
    println!("{:<24} {:<12} DETAIL", "RUNTIME", "STATUS");
    #[cfg(feature = "burn-experimental")]
    for mode in [BurnMode::Cuda, BurnMode::Cpu] {
        let status = BurnBackend::runtime_status(mode);
        println!(
            "{:<24} {:<12} {}",
            mode.display(),
            if status.available {
                "ready"
            } else {
                "unavailable"
            },
            status.detail
        );
        if debug {
            println!(
                "  Burn {} is an in-process probe-gated runtime",
                mode.label()
            );
        }
    }
    for mode in [
        OnnxRuntimeMode::Cuda,
        OnnxRuntimeMode::Rocm,
        OnnxRuntimeMode::Cpu,
    ] {
        let discovery = OnnxRuntimeBackend::discover(store, mode);
        let availability = OnnxRuntimeBackend::availability(store, mode);
        let status = match availability {
            OnnxRuntimeAvailability::Ready => "ready",
            OnnxRuntimeAvailability::Installable => "installable",
            OnnxRuntimeAvailability::Unavailable => "unavailable",
        };
        let detail = match availability {
            OnnxRuntimeAvailability::Ready => discovery
                .path
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "runner ready".to_string()),
            OnnxRuntimeAvailability::Installable => "bundled runner can be installed".to_string(),
            OnnxRuntimeAvailability::Unavailable => {
                OnnxRuntimeBackend::unavailable_reason(store, mode)
            }
        };
        println!("{:<24} {:<12} {}", mode.display(), status, detail);
        if debug {
            print_onnxruntime_debug_details(store, mode);
        }
    }
}

fn compiled_runtime_summary() -> String {
    let mut features = Vec::new();
    if cfg!(feature = "burn-cuda") {
        features.push("burn-cuda");
    }
    if cfg!(feature = "burn-cpu") {
        features.push("burn-cpu");
    }
    if cfg!(feature = "candle-cuda") {
        features.push("candle-cuda");
    }
    if cfg!(feature = "llama-legacy-cuda") {
        features.push("llama-legacy-cuda");
    }
    if cfg!(feature = "llama-legacy-vulkan") {
        features.push("llama-legacy-vulkan");
    }
    if cfg!(feature = "metal") {
        features.push("metal");
    }
    if features.is_empty() {
        "cpu/minimal".to_string()
    } else {
        features.join(", ")
    }
}

fn print_runtime_report(report: &LlamaFastRuntimeReport) {
    println!("runtime: {} {}", report.runtime, report.native_commit);
    println!("compiled: {}", report.compiled);
    println!("modern sampler: {}", report.modern_sampler);
    println!("flash attention supported: {}", report.flash_attn_supported);
    if let Some(requested) = report.flash_attn_requested {
        println!("flash attention requested: {requested}");
    }
    if let Some(cap) = &report.cuda_compute_cap {
        println!("CUDA_COMPUTE_CAP: {cap}");
    }
    println!(
        "ctx/batch/ubatch: {}/{}/{}",
        report.ctx_size, report.batch_size, report.ubatch_size
    );
    println!(
        "threads: generation {}, batch {}",
        report.threads, report.threads_batch
    );
    println!(
        "gpu layers/main gpu: {}/{}",
        report.gpu_layers, report.main_gpu
    );
    println!(
        "KV cache: {}, offload: {}",
        report.kv_cache_type, report.kv_offload
    );
    println!("warmup tokens: {}", report.warmup_tokens);
    for warning in &report.warnings {
        println!("warning: {warning}");
    }
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        Some((values[mid - 1] + values[mid]) / 2.0)
    } else {
        Some(values[mid])
    }
}

fn write_verbose_stats<W: Write>(
    writer: &mut W,
    backend: Option<&str>,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish_reason: &str,
    timings: GenerationTimings,
    backend_diagnostics: &[String],
) -> io::Result<()> {
    if let Some(backend) = backend {
        writeln!(writer, "{:<22}{}", "backend:", backend)?;
    }
    writeln!(
        writer,
        "{:<22}{}",
        "total duration:",
        format_duration(timings.total_seconds)
    )?;
    writeln!(
        writer,
        "{:<22}{}",
        "load duration:",
        format_duration(timings.load_seconds)
    )?;
    if timings.warmup_seconds > 0.0 {
        writeln!(
            writer,
            "{:<22}{}",
            "warmup duration:",
            format_duration(timings.warmup_seconds)
        )?;
    }
    writeln!(
        writer,
        "{:<22}{} token(s)",
        "prompt eval count:", prompt_tokens
    )?;
    writeln!(
        writer,
        "{:<22}{}",
        "prompt eval duration:",
        format_optional_duration(timings.prompt_seconds)
    )?;
    if timings.prompt_seconds.is_finite() {
        writeln!(
            writer,
            "{:<22}{:.2} tokens/s",
            "prompt eval rate:",
            rate(prompt_tokens, timings.prompt_seconds)
        )?;
    } else {
        writeln!(writer, "{:<22}N/A", "prompt eval rate:")?;
    }
    writeln!(
        writer,
        "{:<22}{} token(s)",
        "eval count:", completion_tokens
    )?;
    writeln!(
        writer,
        "{:<22}{}",
        "eval duration:",
        format_duration(timings.decode_seconds)
    )?;
    if timings.first_token_seconds > 0.0 {
        writeln!(
            writer,
            "{:<22}{}",
            "first token:",
            format_duration(timings.first_token_seconds)
        )?;
    }
    writeln!(
        writer,
        "{:<22}{:.2} tokens/s",
        "eval rate:",
        rate(completion_tokens, timings.decode_seconds)
    )?;
    if !finish_reason.is_empty() {
        writeln!(writer, "{:<22}{}", "finish reason:", finish_reason)?;
    }
    if !backend_diagnostics.is_empty() {
        writeln!(writer)?;
        writeln!(writer, "backend stats:")?;
        for line in backend_diagnostics {
            writeln!(writer, "  {line}")?;
        }
    }
    Ok(())
}

fn rate(tokens: usize, seconds: f64) -> f64 {
    if !seconds.is_finite() || seconds <= 0.0 {
        0.0
    } else {
        tokens as f64 / seconds
    }
}

fn format_optional_duration(seconds: f64) -> String {
    if seconds.is_finite() {
        format_duration(seconds)
    } else {
        "N/A".to_string()
    }
}

fn format_duration(seconds: f64) -> String {
    let seconds = seconds.max(0.0);
    if seconds >= 1.0 {
        trim_float(format!("{seconds:.6}")) + "s"
    } else if seconds >= 0.001 {
        trim_float(format!("{:.4}", seconds * 1000.0)) + "ms"
    } else {
        trim_float(format!("{:.3}", seconds * 1_000_000.0)) + "us"
    }
}

fn trim_float(mut value: String) -> String {
    while value.contains('.') && value.ends_with('0') {
        value.pop();
    }
    if value.ends_with('.') {
        value.pop();
    }
    value
}

#[derive(Debug, Clone, Copy)]
enum BackendChoice {
    Auto,
    GgufPreferred {
        llama: LlamaCppMode,
        candle: CandleDeviceMode,
    },
    Candle(CandleDeviceMode),
    LlamaServer(LlamaCppMode),
    LlamaFast(LlamaCppMode),
    LlamaHighlevel(LlamaCppMode),
    Burn(BurnMode),
    Mlx,
    MlxVlm,
    OnnxRuntime(OnnxRuntimeMode),
    TransformersCompat,
    Vllm,
    VllmRocm,
}

#[derive(Debug, Clone, Copy, Default)]
struct SelectionOptions {
    provision_missing_backends: bool,
    verbose_backend_installs: bool,
}

impl SelectionOptions {
    fn from_cli(backend: BackendArg, auto_install: bool, no_auto_install: bool) -> Self {
        let default_provision = matches!(backend, BackendArg::Auto);
        Self {
            provision_missing_backends: !no_auto_install && (auto_install || default_provision),
            verbose_backend_installs: false,
        }
    }

    fn with_backend_install_output(self, verbose: bool) -> Self {
        Self {
            verbose_backend_installs: verbose,
            ..self
        }
    }
}

struct AutoBackend {
    store: ModelStore,
    runtime_options: LlamaRuntimeOptions,
    selection_options: SelectionOptions,
    backends: Mutex<HashMap<&'static str, Arc<dyn GenerationBackend>>>,
}

struct GgufPreferredBackend {
    store: ModelStore,
    gguf_backend: BackendChoice,
    fallback_backend: BackendChoice,
    runtime_options: LlamaRuntimeOptions,
    selection_options: SelectionOptions,
    backends: Mutex<HashMap<&'static str, Arc<dyn GenerationBackend>>>,
}

impl AutoBackend {
    fn new(
        store: ModelStore,
        runtime_options: LlamaRuntimeOptions,
        selection_options: SelectionOptions,
    ) -> Self {
        Self {
            store,
            runtime_options,
            selection_options,
            backends: Mutex::new(HashMap::new()),
        }
    }

    fn backend_for(&self, manifest: &ModelManifest) -> Result<Arc<dyn GenerationBackend>> {
        self.backend_for_request(manifest, false)
    }

    fn backend_for_request(
        &self,
        manifest: &ModelManifest,
        has_images: bool,
    ) -> Result<Arc<dyn GenerationBackend>> {
        let selected = selected_backend_for_request(
            &self.store,
            BackendChoice::Auto,
            manifest,
            has_images,
            self.selection_options,
        )?;
        self.cached_backend(selected)
    }

    fn cached_backend(&self, backend: BackendChoice) -> Result<Arc<dyn GenerationBackend>> {
        let key = backend_label(backend);
        if let Some(backend) = self
            .backends
            .lock()
            .map_err(|_| anyhow!("auto backend cache mutex poisoned"))?
            .get(key)
            .cloned()
        {
            return Ok(backend);
        }

        let backend = build_concrete_backend(
            self.store.clone(),
            backend,
            self.runtime_options.clone(),
            self.selection_options,
        )?;
        self.backends
            .lock()
            .map_err(|_| anyhow!("auto backend cache mutex poisoned"))?
            .insert(key, backend.clone());
        Ok(backend)
    }
}

impl GenerationBackend for AutoBackend {
    fn prepare(&self, manifest: &ModelManifest) -> Result<()> {
        self.backend_for(manifest)?.prepare(manifest)
    }

    fn start_chat_session(
        &self,
        manifest: &ModelManifest,
        seed: Option<u64>,
    ) -> Result<Option<Box<dyn ChatGenerationSession>>> {
        self.backend_for(manifest)?
            .start_chat_session(manifest, seed)
    }

    fn generate(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> Result<crate::backend::GenerateResponse> {
        self.backend_for_request(manifest, !request.image_urls.is_empty())?
            .generate(manifest, request)
    }

    fn generate_stream(
        &self,
        manifest: ModelManifest,
        request: GenerateRequest,
    ) -> crate::backend::GenerateStream {
        match self.backend_for_request(&manifest, !request.image_urls.is_empty()) {
            Ok(backend) => backend.generate_stream(manifest, request),
            Err(err) => Box::pin(tokio_stream::iter(vec![Err(err.to_string())])),
        }
    }
}

impl GgufPreferredBackend {
    fn new(
        store: ModelStore,
        llama: LlamaCppMode,
        candle: CandleDeviceMode,
        runtime_options: LlamaRuntimeOptions,
        selection_options: SelectionOptions,
    ) -> Self {
        Self {
            store,
            gguf_backend: BackendChoice::LlamaServer(llama),
            fallback_backend: BackendChoice::Candle(candle),
            runtime_options,
            selection_options,
            backends: Mutex::new(HashMap::new()),
        }
    }

    fn backend_for(&self, manifest: &ModelManifest) -> Result<Arc<dyn GenerationBackend>> {
        self.backend_for_request(manifest, false)
    }

    fn backend_for_request(
        &self,
        manifest: &ModelManifest,
        has_images: bool,
    ) -> Result<Arc<dyn GenerationBackend>> {
        let requested = match (self.gguf_backend, self.fallback_backend) {
            (BackendChoice::LlamaServer(llama), BackendChoice::Candle(candle)) => {
                BackendChoice::GgufPreferred { llama, candle }
            }
            _ => self.gguf_backend,
        };
        let selected = selected_backend_for_request(
            &self.store,
            requested,
            manifest,
            has_images,
            self.selection_options,
        )?;
        self.cached_backend(selected)
    }

    fn cached_backend(&self, backend: BackendChoice) -> Result<Arc<dyn GenerationBackend>> {
        let key = backend_label(backend);
        if let Some(backend) = self
            .backends
            .lock()
            .map_err(|_| anyhow!("backend cache mutex poisoned"))?
            .get(key)
            .cloned()
        {
            return Ok(backend);
        }

        let backend = build_concrete_backend(
            self.store.clone(),
            backend,
            self.runtime_options.clone(),
            self.selection_options,
        )?;
        self.backends
            .lock()
            .map_err(|_| anyhow!("backend cache mutex poisoned"))?
            .insert(key, backend.clone());
        Ok(backend)
    }
}

impl GenerationBackend for GgufPreferredBackend {
    fn prepare(&self, manifest: &ModelManifest) -> Result<()> {
        self.backend_for(manifest)?.prepare(manifest)
    }

    fn start_chat_session(
        &self,
        manifest: &ModelManifest,
        seed: Option<u64>,
    ) -> Result<Option<Box<dyn ChatGenerationSession>>> {
        self.backend_for(manifest)?
            .start_chat_session(manifest, seed)
    }

    fn generate(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> Result<crate::backend::GenerateResponse> {
        self.backend_for_request(manifest, !request.image_urls.is_empty())?
            .generate(manifest, request)
    }

    fn generate_stream(
        &self,
        manifest: ModelManifest,
        request: GenerateRequest,
    ) -> crate::backend::GenerateStream {
        match self.backend_for_request(&manifest, !request.image_urls.is_empty()) {
            Ok(backend) => backend.generate_stream(manifest, request),
            Err(err) => Box::pin(tokio_stream::iter(vec![Err(err.to_string())])),
        }
    }
}

fn backend_arg_to_choice(backend: BackendArg) -> BackendChoice {
    match backend {
        BackendArg::Auto => BackendChoice::Auto,
        BackendArg::Burn => BackendChoice::Burn(preferred_burn_mode()),
        BackendArg::Candle => BackendChoice::Candle(CandleDeviceMode::Auto),
        BackendArg::Cpu => BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Cpu,
            candle: CandleDeviceMode::Cpu,
        },
        BackendArg::Cuda => BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Cuda,
            candle: CandleDeviceMode::Cuda,
        },
        BackendArg::Metal => BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Metal,
            candle: CandleDeviceMode::Metal,
        },
        BackendArg::Mlx => BackendChoice::Mlx,
        BackendArg::Onnx => BackendChoice::OnnxRuntime(preferred_onnx_mode()),
        BackendArg::Rocm => BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Rocm,
            candle: CandleDeviceMode::Auto,
        },
        BackendArg::Transformers => BackendChoice::TransformersCompat,
        BackendArg::Vllm => BackendChoice::Vllm,
        BackendArg::Vulkan => BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Vulkan,
            candle: CandleDeviceMode::Auto,
        },
        BackendArg::LlamaHighlevel => BackendChoice::LlamaHighlevel(preferred_llama_mode()),
        BackendArg::LlamaLegacy => BackendChoice::LlamaFast(preferred_llama_mode()),
    }
}

fn preferred_llama_mode() -> LlamaCppMode {
    if cfg!(target_os = "macos") {
        LlamaCppMode::Metal
    } else if cfg!(feature = "llama-legacy-cuda") {
        LlamaCppMode::Cuda
    } else if cfg!(feature = "llama-legacy-vulkan") {
        LlamaCppMode::Vulkan
    } else {
        LlamaCppMode::Cpu
    }
}

fn preferred_onnx_mode() -> OnnxRuntimeMode {
    if cfg!(any(windows, target_os = "linux")) {
        OnnxRuntimeMode::Cuda
    } else {
        OnnxRuntimeMode::Cpu
    }
}

fn preferred_burn_mode() -> BurnMode {
    if cfg!(feature = "burn-cuda") && cfg!(any(windows, target_os = "linux")) {
        BurnMode::Cuda
    } else {
        BurnMode::Cpu
    }
}

fn resolve_backend(
    backend: BackendArg,
    device_override: Option<DeviceArg>,
) -> Result<BackendChoice> {
    if backend != BackendArg::Auto && device_override.is_some() {
        bail!("use either --backend or --device, not both");
    }
    if let Some(device) = device_override {
        return Ok(BackendChoice::Candle(device.into()));
    }
    Ok(backend_arg_to_choice(backend))
}

fn build_generation_backend(
    store: ModelStore,
    backend: BackendChoice,
    runtime_options: LlamaRuntimeOptions,
    selection_options: SelectionOptions,
) -> Result<Arc<dyn GenerationBackend>> {
    match backend {
        BackendChoice::Auto => Ok(Arc::new(AutoBackend::new(
            store,
            runtime_options,
            selection_options,
        ))),
        backend => build_concrete_backend(store, backend, runtime_options, selection_options),
    }
}

fn build_concrete_backend(
    store: ModelStore,
    backend: BackendChoice,
    runtime_options: LlamaRuntimeOptions,
    selection_options: SelectionOptions,
) -> Result<Arc<dyn GenerationBackend>> {
    match backend {
        BackendChoice::Auto => bail!("auto backend cannot be built as a concrete backend"),
        BackendChoice::GgufPreferred { llama, candle } => Ok(Arc::new(GgufPreferredBackend::new(
            store,
            llama,
            candle,
            runtime_options,
            selection_options,
        ))),
        BackendChoice::Candle(mode) => Ok(Arc::new(CandleBackend::new_with_device(store, mode)?)),
        BackendChoice::LlamaServer(mode) => Ok(Arc::new(LlamaServerBackend::new(
            store,
            mode,
            runtime_options,
        ))),
        BackendChoice::LlamaFast(mode) => Ok(Arc::new(LlamaFastBackend::new_with_options(
            store,
            mode,
            runtime_options,
        ))),
        BackendChoice::LlamaHighlevel(mode) => Ok(Arc::new(LlamaCppBackend::new(store, mode))),
        BackendChoice::Burn(mode) => Ok(Arc::new(BurnBackend::new(store, mode))),
        BackendChoice::Mlx => Ok(Arc::new(MlxBackend::new(store))),
        BackendChoice::MlxVlm => Ok(Arc::new(MlxVlmBackend::new(store))),
        BackendChoice::OnnxRuntime(mode) => Ok(Arc::new(OnnxRuntimeBackend::new(store, mode))),
        BackendChoice::TransformersCompat => Ok(Arc::new(TransformersCompatBackend::new(store))),
        BackendChoice::Vllm | BackendChoice::VllmRocm => Ok(Arc::new(VllmBackend::new(store))),
    }
}

#[cfg(test)]
fn auto_candidates_for_manifest(manifest: &ModelManifest) -> Vec<BackendChoice> {
    auto_runtime_candidates_for_manifest(manifest)
        .iter()
        .copied()
        .filter_map(runtime_id_to_backend)
        .collect()
}

#[cfg(test)]
fn auto_runtime_candidates_for_manifest(manifest: &ModelManifest) -> Vec<RuntimeId> {
    runtime_candidate_ids(manifest, RequestedBackend::Auto)
}

fn runtime_id_to_backend(id: RuntimeId) -> Option<BackendChoice> {
    match id {
        RuntimeId::BurnCuda => Some(BackendChoice::Burn(BurnMode::Cuda)),
        RuntimeId::BurnCpu => Some(BackendChoice::Burn(BurnMode::Cpu)),
        RuntimeId::LlamaServerCuda => Some(BackendChoice::LlamaServer(LlamaCppMode::Cuda)),
        RuntimeId::LlamaServerRocm => Some(BackendChoice::LlamaServer(LlamaCppMode::Rocm)),
        RuntimeId::LlamaServerVulkan => Some(BackendChoice::LlamaServer(LlamaCppMode::Vulkan)),
        RuntimeId::LlamaServerMetal => Some(BackendChoice::LlamaServer(LlamaCppMode::Metal)),
        RuntimeId::LlamaServerCpu => Some(BackendChoice::LlamaServer(LlamaCppMode::Cpu)),
        RuntimeId::CandleCuda => Some(BackendChoice::Candle(CandleDeviceMode::Cuda)),
        RuntimeId::CandleMetal => Some(BackendChoice::Candle(CandleDeviceMode::Metal)),
        RuntimeId::CandleCpu => Some(BackendChoice::Candle(CandleDeviceMode::Cpu)),
        RuntimeId::Mlx => Some(BackendChoice::Mlx),
        RuntimeId::MlxVlm => Some(BackendChoice::MlxVlm),
        RuntimeId::OnnxRuntimeCuda => Some(BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cuda)),
        RuntimeId::OnnxRuntimeRocm => Some(BackendChoice::OnnxRuntime(OnnxRuntimeMode::Rocm)),
        RuntimeId::OnnxRuntimeCpu => Some(BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cpu)),
        RuntimeId::TransformersCompat => Some(BackendChoice::TransformersCompat),
        RuntimeId::VllmCuda => Some(BackendChoice::Vllm),
        RuntimeId::VllmRocm => Some(BackendChoice::VllmRocm),
        RuntimeId::MediaCompanionCuda
        | RuntimeId::MediaCompanionRocm
        | RuntimeId::MediaCompanionMetal
        | RuntimeId::MediaCompanionCpu => None,
    }
}

fn runtime_id_to_backend_for_request(
    id: RuntimeId,
    _requested: RequestedBackend,
) -> Option<BackendChoice> {
    match id {
        RuntimeId::BurnCuda => Some(BackendChoice::Burn(BurnMode::Cuda)),
        RuntimeId::BurnCpu => Some(BackendChoice::Burn(BurnMode::Cpu)),
        _ => runtime_id_to_backend(id),
    }
}

fn backend_to_runtime_id(backend: BackendChoice) -> Option<RuntimeId> {
    match backend {
        BackendChoice::Burn(BurnMode::Cuda) => Some(RuntimeId::BurnCuda),
        BackendChoice::Burn(BurnMode::Cpu) => Some(RuntimeId::BurnCpu),
        BackendChoice::LlamaServer(LlamaCppMode::Cuda) => Some(RuntimeId::LlamaServerCuda),
        BackendChoice::LlamaServer(LlamaCppMode::Rocm) => Some(RuntimeId::LlamaServerRocm),
        BackendChoice::LlamaServer(LlamaCppMode::Vulkan) => Some(RuntimeId::LlamaServerVulkan),
        BackendChoice::LlamaServer(LlamaCppMode::Metal) => Some(RuntimeId::LlamaServerMetal),
        BackendChoice::LlamaServer(LlamaCppMode::Cpu) => Some(RuntimeId::LlamaServerCpu),
        BackendChoice::Candle(CandleDeviceMode::Cuda) => Some(RuntimeId::CandleCuda),
        BackendChoice::Candle(CandleDeviceMode::Metal) => Some(RuntimeId::CandleMetal),
        BackendChoice::Candle(CandleDeviceMode::Cpu)
        | BackendChoice::Candle(CandleDeviceMode::Auto) => Some(RuntimeId::CandleCpu),
        BackendChoice::Mlx => Some(RuntimeId::Mlx),
        BackendChoice::MlxVlm => Some(RuntimeId::MlxVlm),
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cuda) => Some(RuntimeId::OnnxRuntimeCuda),
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Rocm) => Some(RuntimeId::OnnxRuntimeRocm),
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cpu) => Some(RuntimeId::OnnxRuntimeCpu),
        BackendChoice::TransformersCompat => Some(RuntimeId::TransformersCompat),
        BackendChoice::Vllm => Some(RuntimeId::VllmCuda),
        BackendChoice::VllmRocm => Some(RuntimeId::VllmRocm),
        BackendChoice::Auto
        | BackendChoice::GgufPreferred { .. }
        | BackendChoice::LlamaFast(_)
        | BackendChoice::LlamaHighlevel(_) => None,
    }
}

fn candle_runtime_candidates(mode: CandleDeviceMode, manifest: &ModelManifest) -> Vec<RuntimeId> {
    match mode {
        CandleDeviceMode::Auto => runtime_candidate_ids(manifest, RequestedBackend::Candle),
        CandleDeviceMode::Cpu => vec![RuntimeId::CandleCpu],
        CandleDeviceMode::Cuda => vec![RuntimeId::CandleCuda],
        CandleDeviceMode::Metal => vec![RuntimeId::CandleMetal],
    }
}

fn select_backend_from_runtime_candidates(
    store: &ModelStore,
    candidates: &[RuntimeId],
    manifest: &ModelManifest,
    requested: RequestedBackend,
    capabilities: RequestCapabilities,
    selection_options: SelectionOptions,
) -> Result<BackendChoice> {
    let mut rejected = Vec::new();
    for candidate in candidates {
        let descriptor = runtime_descriptor(*candidate);
        if !runtime_supports_model(
            descriptor,
            &manifest.format,
            manifest.architecture.as_deref(),
        ) {
            rejected.push(format!(
                "{}: model format or architecture is not supported",
                descriptor.display_name
            ));
            continue;
        }
        if !descriptor.implemented {
            rejected.push(format!(
                "{}: runtime integration is not implemented yet",
                descriptor.display_name
            ));
            continue;
        }
        if *candidate == RuntimeId::MlxVlm && !capabilities.image_input {
            rejected.push(format!(
                "{}: MLX-VLM is reserved for image requests; text-only MLX uses mlx-lm",
                descriptor.display_name
            ));
            continue;
        }
        if capabilities.image_input && !descriptor.capabilities.vision_language {
            rejected.push(format!(
                "{}: runtime is not VLM-capable",
                descriptor.display_name
            ));
            continue;
        }
        let Some(backend) = runtime_id_to_backend_for_request(*candidate, requested) else {
            rejected.push(format!(
                "{}: runtime has no executable backend yet",
                descriptor.display_name
            ));
            continue;
        };
        if let Some(reason) =
            backend_unavailability_reason(store, backend, manifest, selection_options)
        {
            let reason = if candidates.len() == 1 {
                unavailable_backend_message(store, backend, manifest)
            } else {
                reason
            };
            rejected.push(format!("{}: {}", descriptor.display_name, reason));
            continue;
        }
        return Ok(backend);
    }
    bail!(
        "no compatible runtime available; tried: {}",
        rejected.join("; ")
    )
}

fn backend_supports_manifest(backend: BackendChoice, manifest: &ModelManifest) -> bool {
    match backend {
        BackendChoice::Auto => false,
        BackendChoice::GgufPreferred { .. } => matches!(
            manifest.format,
            ModelFormat::Gguf | ModelFormat::SafeTensors
        ),
        concrete => {
            backend_supports_format(backend_runtime(concrete), &manifest.format)
                && backend_supports_accelerator(
                    backend_runtime(concrete),
                    backend_accelerator(concrete),
                )
        }
    }
}

fn backend_supports_images(backend: BackendChoice) -> bool {
    runtime_supports_images(backend_runtime(backend))
}

fn backend_runtime(backend: BackendChoice) -> BackendRuntime {
    match backend {
        BackendChoice::Auto | BackendChoice::GgufPreferred { .. } => BackendRuntime::Candle,
        BackendChoice::Burn(_) => BackendRuntime::Burn,
        BackendChoice::Candle(_) => BackendRuntime::Candle,
        BackendChoice::LlamaServer(_) => BackendRuntime::LlamaServer,
        BackendChoice::LlamaFast(_) => BackendRuntime::LlamaLegacy,
        BackendChoice::LlamaHighlevel(_) => BackendRuntime::LlamaHighlevel,
        BackendChoice::Mlx => BackendRuntime::Mlx,
        BackendChoice::MlxVlm => BackendRuntime::MlxVlm,
        BackendChoice::OnnxRuntime(_) => BackendRuntime::OnnxRuntime,
        BackendChoice::TransformersCompat => BackendRuntime::TransformersCompat,
        BackendChoice::Vllm | BackendChoice::VllmRocm => BackendRuntime::Vllm,
    }
}

fn backend_accelerator(backend: BackendChoice) -> BackendAccelerator {
    match backend {
        BackendChoice::Auto | BackendChoice::GgufPreferred { .. } => BackendAccelerator::Auto,
        BackendChoice::Candle(CandleDeviceMode::Auto) => BackendAccelerator::Auto,
        BackendChoice::Candle(CandleDeviceMode::Cpu)
        | BackendChoice::Burn(BurnMode::Cpu)
        | BackendChoice::LlamaServer(LlamaCppMode::Cpu)
        | BackendChoice::LlamaFast(LlamaCppMode::Cpu)
        | BackendChoice::LlamaHighlevel(LlamaCppMode::Cpu) => BackendAccelerator::Cpu,
        BackendChoice::Candle(CandleDeviceMode::Cuda)
        | BackendChoice::Burn(BurnMode::Cuda)
        | BackendChoice::LlamaServer(LlamaCppMode::Cuda)
        | BackendChoice::LlamaFast(LlamaCppMode::Cuda)
        | BackendChoice::LlamaHighlevel(LlamaCppMode::Cuda) => BackendAccelerator::Cuda,
        BackendChoice::LlamaServer(LlamaCppMode::Rocm)
        | BackendChoice::LlamaFast(LlamaCppMode::Rocm)
        | BackendChoice::LlamaHighlevel(LlamaCppMode::Rocm)
        | BackendChoice::OnnxRuntime(OnnxRuntimeMode::Rocm)
        | BackendChoice::VllmRocm => BackendAccelerator::Rocm,
        BackendChoice::LlamaServer(LlamaCppMode::Vulkan)
        | BackendChoice::LlamaFast(LlamaCppMode::Vulkan)
        | BackendChoice::LlamaHighlevel(LlamaCppMode::Vulkan) => BackendAccelerator::Vulkan,
        BackendChoice::Candle(CandleDeviceMode::Metal)
        | BackendChoice::LlamaServer(LlamaCppMode::Metal)
        | BackendChoice::LlamaFast(LlamaCppMode::Metal)
        | BackendChoice::LlamaHighlevel(LlamaCppMode::Metal) => BackendAccelerator::Metal,
        BackendChoice::Mlx | BackendChoice::MlxVlm => BackendAccelerator::Mlx,
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cuda) => BackendAccelerator::Cuda,
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cpu) => BackendAccelerator::Cpu,
        BackendChoice::TransformersCompat => BackendAccelerator::Auto,
        BackendChoice::Vllm => BackendAccelerator::Cuda,
    }
}

fn backend_available_for_store(
    store: &ModelStore,
    backend: BackendChoice,
    manifest: &ModelManifest,
    selection_options: SelectionOptions,
) -> bool {
    backend_unavailability_reason(store, backend, manifest, selection_options).is_none()
}

fn backend_unavailability_reason(
    store: &ModelStore,
    backend: BackendChoice,
    manifest: &ModelManifest,
    selection_options: SelectionOptions,
) -> Option<String> {
    match backend {
        BackendChoice::Auto
        | BackendChoice::GgufPreferred { .. }
        | BackendChoice::Candle(CandleDeviceMode::Auto) => None,
        BackendChoice::Candle(CandleDeviceMode::Cpu) => {
            candle_gguf_tokenizer_rejection(store, manifest)
        }
        BackendChoice::Candle(mode) => {
            candle_gguf_tokenizer_rejection(store, manifest).or_else(|| {
                probe_device(mode).err().map(|_| match mode {
                    CandleDeviceMode::Cuda => candle_cuda_rejection_reason(),
                    CandleDeviceMode::Metal => "Candle Metal is unavailable".to_string(),
                    CandleDeviceMode::Auto | CandleDeviceMode::Cpu => {
                        "Candle is unavailable".to_string()
                    }
                })
            })
        }
        BackendChoice::Mlx => MlxBackend::probe()
            .err()
            .map(|_| "mlx-lm is unavailable".to_string()),
        BackendChoice::MlxVlm => MlxVlmBackend::probe().err().map(|_| {
            "mlx-vlm is unavailable; install with `python3 -m pip install mlx-vlm`".to_string()
        }),
        BackendChoice::TransformersCompat => TransformersCompatBackend::probe()
            .err()
            .map(|_| TransformersCompatBackend::unavailable_reason()),
        BackendChoice::Burn(mode) => BurnBackend::probe(store, manifest, mode)
            .err()
            .map(|_| BurnBackend::unavailable_reason(store, manifest, mode)),
        BackendChoice::OnnxRuntime(mode) => {
            let availability = OnnxRuntimeBackend::availability(store, mode);
            let install_missing_runtime = selection_options.provision_missing_backends
                && matches!(availability, OnnxRuntimeAvailability::Installable);
            OnnxRuntimeBackend::ensure_available_for_model_with_options(
                store,
                manifest,
                mode,
                OnnxProvisionOptions {
                    install_missing_runtime,
                    verbose: false,
                },
            )
            .err()
            .map(|err| compact_reason(&err.to_string()))
        }
        BackendChoice::Vllm => VllmBackend::probe(store)
            .err()
            .map(|_| VllmBackend::unavailable_reason(store)),
        BackendChoice::VllmRocm => VllmBackend::probe_rocm(store)
            .err()
            .map(|_| VllmBackend::rocm_unavailable_reason(store)),
        BackendChoice::LlamaServer(mode) => {
            if LlamaServerBackend::probe(store, mode).is_ok() {
                None
            } else if selection_options.provision_missing_backends
                && should_auto_install_llama_server(mode)
            {
                install_managed_llama_server_with_options(
                    store,
                    mode,
                    LlamaServerInstallOptions {
                        verbose: selection_options.verbose_backend_installs,
                    },
                )
                .and_then(|_| LlamaServerBackend::probe(store, mode).map(|_| ()))
                .err()
                .map(|err| compact_reason(&err.to_string()))
            } else {
                Some(LlamaServerBackend::missing_message(store, mode))
            }
        }
        BackendChoice::LlamaFast(mode) => LlamaFastBackend::probe(mode)
            .err()
            .map(|err| compact_reason(&err.to_string())),
        BackendChoice::LlamaHighlevel(mode) => LlamaCppBackend::probe(mode)
            .err()
            .map(|err| compact_reason(&err.to_string())),
    }
}

#[cfg(test)]
fn selected_backend_for_manifest(
    store: &ModelStore,
    backend: BackendChoice,
    manifest: &ModelManifest,
) -> Result<BackendChoice> {
    selected_backend_for_request(store, backend, manifest, false, SelectionOptions::default())
}

fn selected_backend_for_request(
    store: &ModelStore,
    backend: BackendChoice,
    manifest: &ModelManifest,
    has_images: bool,
    selection_options: SelectionOptions,
) -> Result<BackendChoice> {
    let capabilities = request_capabilities(has_images);
    match backend {
        BackendChoice::Auto
        | BackendChoice::GgufPreferred { .. }
        | BackendChoice::Candle(CandleDeviceMode::Auto) => {
            select_backend_with_planner(store, backend, manifest, capabilities, selection_options)
        }
        BackendChoice::Candle(mode) => {
            let candidates = candle_runtime_candidates(mode, manifest);
            let selected = select_backend_from_runtime_candidates(
                store,
                &candidates,
                manifest,
                RequestedBackend::Candle,
                capabilities,
                selection_options,
            )?;
            ensure_backend_supports_images(selected, has_images)?;
            Ok(selected)
        }
        BackendChoice::LlamaServer(_)
        | BackendChoice::Burn(_)
        | BackendChoice::Mlx
        | BackendChoice::MlxVlm
        | BackendChoice::OnnxRuntime(_)
        | BackendChoice::TransformersCompat
        | BackendChoice::Vllm
        | BackendChoice::VllmRocm => {
            let candidates = if matches!(backend, BackendChoice::Mlx) && has_images {
                vec![RuntimeId::MlxVlm, RuntimeId::Mlx]
            } else {
                let Some(runtime_id) = backend_to_runtime_id(backend) else {
                    bail!(
                        "backend {} is not represented in the runtime planner",
                        backend_label(backend)
                    );
                };
                vec![runtime_id]
            };
            let selected = select_backend_from_runtime_candidates(
                store,
                &candidates,
                manifest,
                requested_backend_for_choice(backend),
                capabilities,
                selection_options,
            )?;
            ensure_backend_supports_images(selected, has_images)?;
            Ok(selected)
        }
        BackendChoice::LlamaFast(_) | BackendChoice::LlamaHighlevel(_) => {
            if !backend_supports_manifest(backend, manifest) {
                bail!("{}", unsupported_backend_message(backend, manifest));
            }
            ensure_backend_supports_images(backend, has_images)?;
            if !backend_available_for_store(store, backend, manifest, selection_options) {
                bail!("{}", unavailable_backend_message(store, backend, manifest));
            }
            Ok(backend)
        }
    }
}

fn select_backend_with_planner(
    store: &ModelStore,
    backend: BackendChoice,
    manifest: &ModelManifest,
    capabilities: RequestCapabilities,
    selection_options: SelectionOptions,
) -> Result<BackendChoice> {
    let requested = requested_backend_for_choice(backend);
    let availability =
        runtime_availabilities_for_request(store, manifest, requested, selection_options);
    let selected = select_runtime(manifest, requested, capabilities, &availability)
        .map_err(|err| anyhow!("{}", format_runtime_plan_error(manifest, &err)))?;
    runtime_id_to_backend_for_request(selected.runtime_id, requested).ok_or_else(|| {
        anyhow!(
            "selected runtime {} has no executable backend yet",
            selected.display_name
        )
    })
}

fn verbose_fallback_note(
    store: &ModelStore,
    requested_choice: BackendChoice,
    manifest: &ModelManifest,
    has_images: bool,
    selected: BackendChoice,
) -> Option<String> {
    let _ = (store, requested_choice, manifest, has_images, selected);
    None
}

fn print_verbose_fallback_note(
    store: &ModelStore,
    requested_choice: BackendChoice,
    manifest: &ModelManifest,
    has_images: bool,
    selected: BackendChoice,
    verbose: bool,
) {
    if !verbose {
        return;
    }
    if let Some(note) =
        verbose_fallback_note(store, requested_choice, manifest, has_images, selected)
    {
        eprintln!("{note}");
    }
}

fn compact_reason(reason: &str) -> String {
    reason.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn runtime_availabilities_for_request(
    store: &ModelStore,
    manifest: &ModelManifest,
    requested: RequestedBackend,
    selection_options: SelectionOptions,
) -> Vec<RuntimeAvailability> {
    runtime_candidate_ids_for_selection(store, manifest, requested)
        .into_iter()
        .map(|runtime_id| {
            if let Some(backend) = runtime_id_to_backend_for_request(runtime_id, requested) {
                let reason = runtime_unavailability_reason(
                    store,
                    runtime_id,
                    backend,
                    manifest,
                    selection_options,
                );
                RuntimeAvailability {
                    runtime_id,
                    available: reason.is_none(),
                    reason,
                }
            } else {
                let runtime = runtime_descriptor(runtime_id);
                RuntimeAvailability {
                    runtime_id,
                    available: false,
                    reason: Some(
                        if runtime.implemented {
                            "runtime has no executable backend yet"
                        } else {
                            "runtime integration is not implemented yet"
                        }
                        .to_string(),
                    ),
                }
            }
        })
        .collect()
}

fn runtime_candidate_ids_for_selection(
    store: &ModelStore,
    manifest: &ModelManifest,
    requested: RequestedBackend,
) -> Vec<RuntimeId> {
    let mut candidates = runtime_candidate_ids(manifest, requested);
    if requested == RequestedBackend::Auto
        && manifest.format == ModelFormat::Gguf
        && llama_rocm_auto_probeable(store)
        && !candidates.contains(&RuntimeId::LlamaServerRocm)
    {
        let insert_at = llama_rocm_insert_position(&candidates);
        candidates.insert(insert_at, RuntimeId::LlamaServerRocm);
    }
    candidates
}

fn llama_rocm_insert_position(candidates: &[RuntimeId]) -> usize {
    candidates
        .iter()
        .position(|id| {
            matches!(
                id,
                RuntimeId::LlamaServerVulkan
                    | RuntimeId::LlamaServerMetal
                    | RuntimeId::LlamaServerCpu
            )
        })
        .unwrap_or(candidates.len())
}

fn llama_rocm_auto_probeable(store: &ModelStore) -> bool {
    env::var_os("WERK_LLAMA_SERVER_ROCM").is_some()
        || managed_backend_dir(store, LlamaCppMode::Rocm).exists()
}

fn should_auto_install_llama_server(mode: LlamaCppMode) -> bool {
    cfg!(target_os = "macos") && mode == LlamaCppMode::Metal
}

fn runtime_unavailability_reason(
    store: &ModelStore,
    runtime_id: RuntimeId,
    backend: BackendChoice,
    manifest: &ModelManifest,
    selection_options: SelectionOptions,
) -> Option<String> {
    match runtime_id {
        RuntimeId::VllmRocm => VllmBackend::probe_rocm(store)
            .err()
            .map(|_| VllmBackend::rocm_unavailable_reason(store)),
        _ => backend_unavailability_reason(store, backend, manifest, selection_options),
    }
}

fn requested_backend_for_choice(backend: BackendChoice) -> RequestedBackend {
    match backend {
        BackendChoice::Auto => RequestedBackend::Auto,
        BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Cuda,
            ..
        }
        | BackendChoice::LlamaServer(LlamaCppMode::Cuda) => RequestedBackend::Cuda,
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cuda) => RequestedBackend::Cuda,
        BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Rocm,
            ..
        }
        | BackendChoice::LlamaServer(LlamaCppMode::Rocm) => RequestedBackend::Rocm,
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Rocm) => RequestedBackend::Rocm,
        BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Vulkan,
            ..
        }
        | BackendChoice::LlamaServer(LlamaCppMode::Vulkan) => RequestedBackend::Vulkan,
        BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Metal,
            ..
        }
        | BackendChoice::LlamaServer(LlamaCppMode::Metal) => RequestedBackend::Metal,
        BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Cpu,
            ..
        }
        | BackendChoice::LlamaServer(LlamaCppMode::Cpu) => RequestedBackend::Cpu,
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cpu) => RequestedBackend::Cpu,
        BackendChoice::Burn(_) => RequestedBackend::Burn,
        BackendChoice::Candle(_) => RequestedBackend::Candle,
        BackendChoice::Mlx | BackendChoice::MlxVlm => RequestedBackend::Mlx,
        BackendChoice::TransformersCompat => RequestedBackend::Transformers,
        BackendChoice::Vllm => RequestedBackend::Vllm,
        BackendChoice::VllmRocm => RequestedBackend::Rocm,
        BackendChoice::LlamaFast(_) => RequestedBackend::LlamaLegacy,
        BackendChoice::LlamaHighlevel(_) => RequestedBackend::LlamaHighlevel,
    }
}

fn request_capabilities(has_images: bool) -> RequestCapabilities {
    RequestCapabilities::text_with_images(true, has_images)
}

fn format_runtime_plan_error(
    manifest: &ModelManifest,
    err: &crate::runtime_planner::RuntimePlanError,
) -> String {
    let architecture = manifest.architecture.as_deref().unwrap_or("unknown");
    if err.decisions.is_empty() {
        return format!(
            "no runtime candidates for model '{}' ({:?}, architecture: {architecture})",
            manifest.id, manifest.format
        );
    }
    let tried = err
        .decisions
        .iter()
        .map(|decision| format!("{}: {}", decision.display_name, decision.reason))
        .collect::<Vec<_>>()
        .join("; ");
    format!(
        "no available runtime for model '{}' ({:?}, architecture: {architecture}); tried: {tried}",
        manifest.id, manifest.format
    )
}

fn ensure_backend_supports_images(backend: BackendChoice, has_images: bool) -> Result<()> {
    if !has_images || backend_supports_images(backend) {
        return Ok(());
    }
    bail!(
        "Image input requires a VLM-capable backend. Current backend {} is text-only. Try --backend mlx with a compatible VLM model.",
        verbose_backend_label(backend)
    )
}

fn unavailable_backend_message(
    store: &ModelStore,
    backend: BackendChoice,
    manifest: &ModelManifest,
) -> String {
    match (backend, &manifest.format) {
        (BackendChoice::Candle(CandleDeviceMode::Cuda), ModelFormat::SafeTensors) => {
            candle_cuda_unavailable_message()
        }
        (BackendChoice::Candle(CandleDeviceMode::Metal), ModelFormat::SafeTensors) => {
            "Metal backend requested for safetensors model, but Candle Metal is unavailable. Build with Metal support on macOS or choose --backend cpu.".to_string()
        }
        (BackendChoice::Burn(mode), ModelFormat::SafeTensors) => {
            BurnBackend::missing_message(store, manifest, mode)
        }
        (BackendChoice::LlamaServer(mode), ModelFormat::Gguf) => {
            LlamaServerBackend::missing_message(store, mode)
        }
        (BackendChoice::OnnxRuntime(mode), ModelFormat::SafeTensors | ModelFormat::Onnx) => {
            OnnxRuntimeBackend::missing_message(store, mode)
        }
        (BackendChoice::Vllm, ModelFormat::SafeTensors) => VllmBackend::missing_message(store),
        (BackendChoice::VllmRocm, ModelFormat::SafeTensors) => {
            VllmBackend::rocm_unavailable_reason(store)
        }
        (BackendChoice::Mlx, _) => "mlx-lm is unavailable".to_string(),
        (BackendChoice::MlxVlm, _) => {
            "mlx-vlm is unavailable; install with `python3 -m pip install mlx-vlm`".to_string()
        }
        (BackendChoice::TransformersCompat, ModelFormat::SafeTensors) => {
            TransformersCompatBackend::unavailable_reason()
        }
        _ => format!(
            "backend {} is unavailable for model '{}' with format {:?}",
            backend_label(backend),
            manifest.id,
            manifest.format
        ),
    }
}

fn backend_to_build_for_request(
    requested: BackendChoice,
    selected: BackendChoice,
    manifest: &ModelManifest,
) -> BackendChoice {
    if matches!(
        (requested, manifest.format.clone()),
        (
            BackendChoice::GgufPreferred {
                llama: LlamaCppMode::Cpu,
                ..
            },
            ModelFormat::Gguf
        )
    ) {
        requested
    } else {
        selected
    }
}

fn print_routing_debug(
    store: &ModelStore,
    requested: BackendArg,
    requested_choice: BackendChoice,
    manifest: &ModelManifest,
    has_images: bool,
    selected: BackendChoice,
    debug: bool,
) {
    if !debug {
        return;
    }
    let capabilities = request_capabilities(has_images);
    let requested_backend = requested_backend_for_choice(requested_choice);
    let availability = runtime_availabilities_for_request(
        store,
        manifest,
        requested_backend,
        SelectionOptions::default(),
    );
    let plan = plan_runtime(manifest, requested_backend, capabilities, &availability);

    eprintln!("requested backend: {}", requested_backend_label(requested));
    eprintln!("model format: {:?}", manifest.format);
    eprintln!(
        "architecture: {}",
        manifest.architecture.as_deref().unwrap_or("unknown")
    );
    eprintln!("artifact: {}", artifact_debug_label(store, manifest));
    eprintln!("request capabilities:");
    eprintln!("  text_generation: yes");
    eprintln!(
        "  image_input: {}",
        yes_no(plan.request_capabilities.image_input)
    );
    eprintln!(
        "  embeddings: {}",
        yes_no(plan.request_capabilities.embeddings)
    );
    eprintln!(
        "  streaming: {}",
        yes_no(plan.request_capabilities.streaming)
    );
    eprintln!("candidate runtimes:");
    for decision in &plan.candidates {
        let descriptor = runtime_descriptor(decision.runtime_id);
        let status = match decision.status {
            RuntimeDecisionStatus::Accepted => "accepted",
            RuntimeDecisionStatus::Rejected => "rejected",
        };
        let role = runtime_role(manifest, requested_backend, decision.runtime_id);
        eprintln!(
            "candidate: {} ({role}) -> {status}: {}",
            decision.display_name, decision.reason
        );
        if decision.status == RuntimeDecisionStatus::Rejected
            && let Some(target) = descriptor.install_target
        {
            eprintln!("  install hint: werk backend install {target}");
        }
        #[cfg(feature = "burn-experimental")]
        if matches!(
            decision.runtime_id,
            RuntimeId::BurnCuda | RuntimeId::BurnCpu
        ) && decision.status == RuntimeDecisionStatus::Rejected
        {
            let mode = match decision.runtime_id {
                RuntimeId::BurnCuda => BurnMode::Cuda,
                RuntimeId::BurnCpu => BurnMode::Cpu,
                _ => unreachable!(),
            };
            print_burn_debug_details(store, manifest, mode);
        }
        if matches!(
            decision.runtime_id,
            RuntimeId::OnnxRuntimeCuda | RuntimeId::OnnxRuntimeRocm | RuntimeId::OnnxRuntimeCpu
        ) && decision.status == RuntimeDecisionStatus::Rejected
        {
            let mode = match decision.runtime_id {
                RuntimeId::OnnxRuntimeCuda => OnnxRuntimeMode::Cuda,
                RuntimeId::OnnxRuntimeRocm => OnnxRuntimeMode::Rocm,
                RuntimeId::OnnxRuntimeCpu => OnnxRuntimeMode::Cpu,
                _ => unreachable!(),
            };
            print_onnxruntime_debug_details(store, mode);
        }
    }
    if let Some(planned) = plan.selected {
        eprintln!("selected runtime: {}", planned.display_name);
        eprintln!(
            "selected role: {}",
            runtime_role(manifest, requested_backend, planned.runtime_id)
        );
        eprintln!("reason: {}", planned.reason);
        if candle_safetensors_cuda_fallback_warning(manifest, requested_backend, planned.runtime_id)
        {
            eprintln!(
                "warning: Candle is a compatibility fallback for safetensors CUDA. Install vLLM for better serving performance."
            );
        }
    } else {
        eprintln!("selected runtime: {}", verbose_backend_label(selected));
    }
}

fn runtime_role(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    runtime_id: RuntimeId,
) -> &'static str {
    let descriptor = runtime_descriptor(runtime_id);
    if descriptor.runtime == BackendRuntime::Candle
        && manifest.format == ModelFormat::SafeTensors
        && requested_backend != RequestedBackend::Candle
    {
        "compatibility fallback"
    } else {
        "primary runtime"
    }
}

fn candle_safetensors_cuda_fallback_warning(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    runtime_id: RuntimeId,
) -> bool {
    manifest.format == ModelFormat::SafeTensors
        && matches!(
            requested_backend,
            RequestedBackend::Auto | RequestedBackend::Cuda
        )
        && runtime_id == RuntimeId::CandleCuda
}

#[cfg(feature = "burn-experimental")]
fn print_burn_debug_details(store: &ModelStore, manifest: &ModelManifest, mode: BurnMode) {
    let report = BurnBackend::probe_report(store, manifest, mode);
    eprintln!(
        "  status: {}",
        if report.available {
            "available"
        } else {
            "unavailable"
        }
    );
    eprintln!("  reason: {}", report.reason);
    eprintln!("  architecture: {}", report.architecture);
    eprintln!("  checks:");
    for check in report.checks {
        eprintln!(
            "  - {}: {} ({})",
            check.name,
            if check.ok { "ok" } else { "failed" },
            check.detail
        );
    }
}

fn print_onnxruntime_debug_details(store: &ModelStore, mode: OnnxRuntimeMode) {
    let discovery = OnnxRuntimeBackend::discover(store, mode);
    let availability = OnnxRuntimeBackend::availability(store, mode);
    let status = match availability {
        OnnxRuntimeAvailability::Ready => "ready",
        OnnxRuntimeAvailability::Installable => "installable",
        OnnxRuntimeAvailability::Unavailable => "unavailable",
    };
    let reason = match availability {
        OnnxRuntimeAvailability::Ready => format!("runner discovered from {}", discovery.source),
        OnnxRuntimeAvailability::Installable => "bundled runner can be installed".to_string(),
        OnnxRuntimeAvailability::Unavailable => OnnxRuntimeBackend::unavailable_reason(store, mode),
    };
    eprintln!("  status: {status}");
    eprintln!("  reason: {reason}");
    eprintln!("  tried:");
    for attempt in discovery.attempts {
        match attempt.path {
            Some(path) => eprintln!(
                "  - {}: {} ({})",
                attempt.label,
                path.display(),
                attempt.detail
            ),
            None => eprintln!("  - {}: {}", attempt.label, attempt.detail),
        }
    }
}

#[cfg(test)]
fn routing_candidates_for_debug(
    requested: BackendChoice,
    manifest: &ModelManifest,
) -> Vec<RuntimeId> {
    match requested {
        BackendChoice::Candle(mode) => candle_runtime_candidates(mode, manifest),
        BackendChoice::Auto | BackendChoice::GgufPreferred { .. } => {
            runtime_candidate_ids(manifest, requested_backend_for_choice(requested))
        }
        concrete => backend_to_runtime_id(concrete).into_iter().collect(),
    }
}

fn candle_cuda_unavailable_message() -> String {
    if cfg!(feature = "candle-cuda") {
        "CUDA backend requested for safetensors model, but Candle CUDA is unavailable. Check the NVIDIA driver/CUDA runtime, or use: werk --backend auto / --backend cpu.".to_string()
    } else {
        "CUDA backend requested for safetensors model. This Werk binary was built without Candle CUDA support. Rebuild with: cargo install --path . --locked --force --features cuda. Or use: werk --backend auto / --backend cpu.".to_string()
    }
}

fn candle_cuda_rejection_reason() -> String {
    if cfg!(feature = "candle-cuda") {
        "Candle CUDA is unavailable".to_string()
    } else {
        "This Werk binary was built without Candle CUDA support. Rebuild with: cargo install --path . --locked --force --features cuda".to_string()
    }
}

fn requested_backend_label(backend: BackendArg) -> &'static str {
    match backend {
        BackendArg::Auto => "auto",
        BackendArg::Burn => "burn",
        BackendArg::Candle => "candle",
        BackendArg::Cpu => "cpu",
        BackendArg::Cuda => "cuda",
        BackendArg::LlamaHighlevel => "llama-highlevel",
        BackendArg::LlamaLegacy => "llama-legacy",
        BackendArg::Metal => "metal",
        BackendArg::Mlx => "mlx",
        BackendArg::Onnx => "onnx",
        BackendArg::Rocm => "rocm",
        BackendArg::Transformers => "transformers",
        BackendArg::Vllm => "vllm",
        BackendArg::Vulkan => "vulkan",
    }
}

fn unsupported_backend_message(backend: BackendChoice, manifest: &ModelManifest) -> String {
    match (backend, &manifest.format) {
        (BackendChoice::LlamaServer(LlamaCppMode::Vulkan), ModelFormat::SafeTensors) => {
            "Vulkan backend currently supports GGUF through llama.cpp server only. Safetensors Vulkan execution is not implemented.".to_string()
        }
        (BackendChoice::GgufPreferred { llama: LlamaCppMode::Vulkan, .. }, ModelFormat::SafeTensors) => {
            "Vulkan backend currently supports GGUF through llama.cpp server only. Safetensors Vulkan execution is not implemented.".to_string()
        }
        (BackendChoice::LlamaServer(_), _) => format!(
            "llama.cpp server backend supports GGUF only; model '{}' is {:?}",
            manifest.id, manifest.format
        ),
        (BackendChoice::LlamaFast(_) | BackendChoice::LlamaHighlevel(_), _) => format!(
            "llama.cpp legacy backends support GGUF only; model '{}' is {:?}",
            manifest.id, manifest.format
        ),
        (BackendChoice::Mlx, ModelFormat::Gguf) => {
            "MLX backend does not support GGUF; use --backend cuda, --backend vulkan, or --backend cpu for llama.cpp server".to_string()
        }
        (BackendChoice::Burn(_), ModelFormat::Gguf) => {
            "Burn backend supports safetensors models only; use --backend cuda, --backend vulkan, or --backend cpu for GGUF llama.cpp server".to_string()
        }
        (BackendChoice::OnnxRuntime(_), ModelFormat::Gguf) => {
            "ONNX Runtime backend supports safetensors models with managed ONNX artifacts; use --backend cuda, --backend vulkan, or --backend cpu for GGUF llama.cpp server".to_string()
        }
        (_, ModelFormat::Onnx) => {
            "Direct ONNX model import is catalog-only for now; install a safetensors model and build managed ONNX artifacts with `werk artifacts build <model>`.".to_string()
        }
        (BackendChoice::TransformersCompat, _) => {
            "Transformers compatibility backend supports raw ChatGLM/GLM Hugging Face safetensors models only".to_string()
        }
        (_, ModelFormat::PyTorch) => {
            "PyTorch generation is pending; PyTorch backend is not implemented yet".to_string()
        }
        (_, ModelFormat::TensorRt) => {
            "TensorRT generation is pending; TensorRT backend is not implemented yet".to_string()
        }
        (_, ModelFormat::OpenVino) => {
            "OpenVINO generation is pending; OpenVINO backend is not implemented yet".to_string()
        }
        (_, ModelFormat::CoreMl) => {
            "CoreML generation is pending; CoreML backend is not implemented yet".to_string()
        }
        (_, ModelFormat::TensorFlow) => {
            "TensorFlow generation is pending; TensorFlow backend is not implemented yet".to_string()
        }
        _ => format!(
            "backend {} does not support model '{}' with format {:?}",
            backend_label(backend),
            manifest.id,
            manifest.format
        ),
    }
}

fn backend_label(backend: BackendChoice) -> &'static str {
    match backend {
        BackendChoice::Auto => "auto",
        BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Cuda,
            ..
        } => "cuda",
        BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Rocm,
            ..
        } => "rocm",
        BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Cpu,
            ..
        } => "cpu",
        BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Vulkan,
            ..
        } => "vulkan",
        BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Metal,
            ..
        } => "metal",
        BackendChoice::Candle(CandleDeviceMode::Auto) => "candle-auto",
        BackendChoice::Burn(BurnMode::Cuda) => "burn-cuda",
        BackendChoice::Burn(BurnMode::Cpu) => "burn-cpu",
        BackendChoice::Candle(CandleDeviceMode::Cpu) => "candle-cpu",
        BackendChoice::Candle(CandleDeviceMode::Cuda) => "candle-cuda",
        BackendChoice::Candle(CandleDeviceMode::Metal) => "metal",
        BackendChoice::Mlx => "mlx",
        BackendChoice::MlxVlm => "mlx-vlm",
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cuda) => "onnxruntime-cuda",
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Rocm) => "onnxruntime-rocm",
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cpu) => "onnxruntime-cpu",
        BackendChoice::TransformersCompat => "transformers",
        BackendChoice::Vllm => "vllm-cuda",
        BackendChoice::VllmRocm => "vllm-rocm",
        BackendChoice::LlamaServer(LlamaCppMode::Cuda) => "llama-server-cuda",
        BackendChoice::LlamaServer(LlamaCppMode::Rocm) => "llama-server-rocm",
        BackendChoice::LlamaServer(LlamaCppMode::Vulkan) => "llama-server-vulkan",
        BackendChoice::LlamaServer(LlamaCppMode::Metal) => "llama-server-metal",
        BackendChoice::LlamaServer(LlamaCppMode::Cpu) => "llama-server-cpu",
        BackendChoice::LlamaFast(LlamaCppMode::Cuda) => "llama-legacy-cuda",
        BackendChoice::LlamaFast(LlamaCppMode::Rocm) => "llama-legacy-rocm",
        BackendChoice::LlamaFast(LlamaCppMode::Vulkan) => "llama-legacy-vulkan",
        BackendChoice::LlamaFast(LlamaCppMode::Metal) => "llama-legacy-metal",
        BackendChoice::LlamaFast(LlamaCppMode::Cpu) => "llama-legacy-cpu",
        BackendChoice::LlamaHighlevel(mode) => mode.label(),
    }
}

fn verbose_backend_label(backend: BackendChoice) -> &'static str {
    match backend {
        BackendChoice::Auto => "auto",
        BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Cuda,
            ..
        } => "llama.cpp server CUDA",
        BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Rocm,
            ..
        } => "llama.cpp server ROCm/HIP",
        BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Cpu,
            ..
        } => "llama.cpp server CPU",
        BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Vulkan,
            ..
        } => "llama.cpp server Vulkan",
        BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Metal,
            ..
        } => "llama.cpp server Metal",
        BackendChoice::Candle(CandleDeviceMode::Auto) => "Candle auto",
        BackendChoice::Burn(BurnMode::Cuda) => "Burn CUDA",
        BackendChoice::Burn(BurnMode::Cpu) => "Burn CPU",
        BackendChoice::Candle(CandleDeviceMode::Cpu) => "Candle CPU",
        BackendChoice::Candle(CandleDeviceMode::Cuda) => "Candle CUDA",
        BackendChoice::Candle(CandleDeviceMode::Metal) => "Candle Metal",
        BackendChoice::Mlx => "MLX",
        BackendChoice::MlxVlm => "MLX-VLM",
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cuda) => "ONNX Runtime CUDA",
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Rocm) => "ONNX Runtime ROCm",
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cpu) => "ONNX Runtime CPU",
        BackendChoice::TransformersCompat => "Transformers compatibility",
        BackendChoice::Vllm => "vLLM CUDA",
        BackendChoice::VllmRocm => "vLLM ROCm",
        BackendChoice::LlamaServer(LlamaCppMode::Cuda) => "llama.cpp server CUDA",
        BackendChoice::LlamaServer(LlamaCppMode::Rocm) => "llama.cpp server ROCm/HIP",
        BackendChoice::LlamaServer(LlamaCppMode::Vulkan) => "llama.cpp server Vulkan",
        BackendChoice::LlamaServer(LlamaCppMode::Metal) => "llama.cpp server Metal",
        BackendChoice::LlamaServer(LlamaCppMode::Cpu) => "llama.cpp server CPU",
        BackendChoice::LlamaFast(LlamaCppMode::Cuda) => "llama.cpp legacy FFI CUDA",
        BackendChoice::LlamaFast(LlamaCppMode::Rocm) => "llama.cpp legacy FFI ROCm",
        BackendChoice::LlamaFast(LlamaCppMode::Vulkan) => "llama.cpp legacy FFI Vulkan",
        BackendChoice::LlamaFast(LlamaCppMode::Metal) => "llama.cpp legacy FFI Metal",
        BackendChoice::LlamaFast(LlamaCppMode::Cpu) => "llama.cpp legacy FFI CPU",
        BackendChoice::LlamaHighlevel(LlamaCppMode::Cuda) => "llama.cpp high-level CUDA",
        BackendChoice::LlamaHighlevel(LlamaCppMode::Rocm) => "llama.cpp high-level ROCm",
        BackendChoice::LlamaHighlevel(LlamaCppMode::Vulkan) => "llama.cpp high-level Vulkan",
        BackendChoice::LlamaHighlevel(LlamaCppMode::Metal) => "llama.cpp high-level Metal",
        BackendChoice::LlamaHighlevel(LlamaCppMode::Cpu) => "llama.cpp high-level CPU",
    }
}

fn display_llama_mode(mode: LlamaCppMode) -> &'static str {
    match mode {
        LlamaCppMode::Cuda => "CUDA",
        LlamaCppMode::Rocm => "ROCm/HIP",
        LlamaCppMode::Vulkan => "Vulkan",
        LlamaCppMode::Metal => "Metal",
        LlamaCppMode::Cpu => "CPU",
    }
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn doctor_check_status(check: &crate::backend::BackendDoctorCheck) -> &'static str {
    if check.detail.contains("best-effort on WSL") {
        "warn"
    } else if check.ok {
        "ok"
    } else {
        "missing"
    }
}

fn artifact_debug_label(store: &ModelStore, manifest: &ModelManifest) -> String {
    if manifest.format != ModelFormat::SafeTensors {
        return "none".to_string();
    }
    match store.ready_onnx_artifact(manifest) {
        Some(artifact) => format!("onnx ({})", artifact.path),
        None if manifest.artifacts.iter().any(|artifact| {
            matches!(artifact.kind, crate::model_store::ArtifactKind::Onnx)
                && artifact.status == ArtifactStatus::Failed
        }) =>
        {
            "onnx failed".to_string()
        }
        None => "none".to_string(),
    }
}

fn print_manifest_summary(action: &str, manifest: &ModelManifest) {
    println!(
        "{action} {} ({:?}, architecture: {})",
        manifest.id,
        manifest.format,
        manifest.architecture.as_deref().unwrap_or("unknown")
    );
    println!(
        "  family={} layout={} tasks={}",
        manifest.metadata.family.as_deref().unwrap_or("unknown"),
        manifest.metadata.repository_layout,
        join_display(&manifest.metadata.tasks)
    );
    println!(
        "  components={} size={} precision={} quantization={}",
        manifest.metadata.components.len(),
        format_bytes(
            manifest
                .files
                .iter()
                .map(|file| file.size)
                .fold(0_u64, u64::saturating_add)
        ),
        manifest.metadata.precision.as_deref().unwrap_or("unknown"),
        manifest.metadata.quantization.as_deref().unwrap_or("none")
    );
    println!(
        "  compatible runtimes={}",
        if manifest.metadata.compatible_runtimes.is_empty() {
            "unknown".to_string()
        } else {
            manifest.metadata.compatible_runtimes.join(",")
        }
    );
}

fn print_artifact_result(action: &str, model: &str, artifact: &ModelArtifact) {
    println!(
        "{action} {:?} artifact for {}: {} ({})",
        artifact.kind,
        model,
        artifact.path,
        artifact_status_label(artifact.status.clone())
    );
    if let Some(detail) = artifact.detail.as_deref() {
        println!("{detail}");
    }
}

fn print_artifact_list(model: &str, artifacts: &[ModelArtifact]) {
    if artifacts.is_empty() {
        println!("No artifacts for {model}");
        return;
    }
    println!("{:<12} {:<8} {:<32} DETAIL", "KIND", "STATUS", "PATH");
    for artifact in artifacts {
        println!(
            "{:<12} {:<8} {:<32} {}",
            format!("{:?}", artifact.kind).to_lowercase(),
            artifact_status_label(artifact.status.clone()),
            artifact.path,
            artifact.detail.as_deref().unwrap_or("-")
        );
    }
}

fn artifact_status_label(status: ArtifactStatus) -> &'static str {
    match status {
        ArtifactStatus::Ready => "ready",
        ArtifactStatus::Failed => "failed",
    }
}

fn pull_progress_bar() -> ProgressBar {
    let progress = ProgressBar::new(100);
    progress.enable_steady_tick(Duration::from_millis(120));
    progress.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:32.cyan/blue}] {pos:>3}% {msg}",
        )
        .unwrap()
        .progress_chars("=> "),
    );
    progress
}

fn update_pull_progress(progress: &ProgressBar, event: PullProgress) {
    match event {
        PullProgress::Started { url } => {
            progress.set_position(0);
            progress.set_message(format!("cloning {url}"));
        }
        PullProgress::GitProgress { line } => {
            if let Some(percent) = parse_git_percent(&line) {
                progress.set_position(percent);
            }
            progress.set_message(line);
        }
        PullProgress::CloneFinished => {
            progress.set_position(100);
            progress.set_message("metadata clone complete");
        }
        PullProgress::LfsStarted { file, total_bytes } => {
            progress.set_position(0);
            let target = file
                .as_deref()
                .map(|file| format!(" {file}"))
                .unwrap_or_default();
            let size = total_bytes
                .map(|bytes| format!(" ({})", format_bytes(bytes)))
                .unwrap_or_default();
            progress.set_message(format!("downloading{target}{size}"));
        }
        PullProgress::LfsProgress { line } => {
            if let Some(percent) = parse_git_percent(&line) {
                progress.set_position(percent);
            }
            progress.set_message(line);
        }
        PullProgress::TransferStats {
            bytes,
            total_bytes,
            bytes_per_second,
        } => {
            if let Some(total_bytes) = total_bytes.filter(|total| *total > 0) {
                let percent = bytes.saturating_mul(100) / total_bytes;
                progress.set_position(percent.min(99));
                if bytes >= total_bytes {
                    progress.set_message(format!(
                        "finalizing LFS checkout after {} @ {}/s",
                        format_bytes(total_bytes),
                        format_bytes_per_second(bytes_per_second)
                    ));
                } else {
                    progress.set_message(format!(
                        "downloading {} / {} @ {}/s",
                        format_bytes(bytes),
                        format_bytes(total_bytes),
                        format_bytes_per_second(bytes_per_second)
                    ));
                }
            } else {
                progress.set_message(format!(
                    "downloading {} @ {}/s",
                    format_bytes(bytes),
                    format_bytes_per_second(bytes_per_second)
                ));
            }
        }
        PullProgress::LfsFinished => {
            progress.set_position(100);
            progress.set_message("download complete");
        }
        PullProgress::Importing => {
            progress.set_position(0);
            progress.set_message("importing into model store");
        }
        PullProgress::Finished { files, bytes } => {
            progress.set_position(100);
            progress.set_message(format!("imported {files} files, {}", format_bytes(bytes)));
        }
    }
}

fn parse_git_percent(line: &str) -> Option<u64> {
    line.split_whitespace()
        .find_map(|part| part.strip_suffix('%')?.parse::<u64>().ok())
        .map(|percent| percent.min(100))
}

fn format_bytes(bytes: u64) -> String {
    format_bytes_f64(bytes as f64)
}

fn format_bytes_per_second(bytes_per_second: f64) -> String {
    format_bytes_f64(bytes_per_second.max(0.0))
}

fn format_bytes_f64(bytes: f64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes;
    let mut unit = UNITS[0];
    for candidate in &UNITS[1..] {
        if value < 1024.0 {
            break;
        }
        value /= 1024.0;
        unit = candidate;
    }
    if unit == "B" {
        format!("{value:.0} {unit}")
    } else {
        format!("{value:.2} {unit}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_store::{ModelFile, ModelSource};
    use std::fs;
    use std::sync::{Arc as StdArc, Mutex as StdMutex};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[cfg(unix)]
    #[test]
    fn editable_line_inserts_at_cursor() {
        let mut line = EditableLine::default();
        for ch in "helo".chars() {
            line.insert(ch);
        }

        assert!(line.move_left());
        line.insert('l');

        assert_eq!(line.as_string(), "hello");
        assert_eq!(line.cursor, 4);
    }

    #[cfg(unix)]
    #[test]
    fn editable_line_history_restores_draft() {
        let reader = TerminalLineReader {
            history: vec!["first".to_string(), "second".to_string()],
        };
        let mut line = EditableLine::default();
        let mut draft = String::new();
        let mut history_index = None;
        line.replace("draft");

        assert!(reader.apply_command(
            LineEditCommand::HistoryPrevious,
            &mut line,
            &mut draft,
            &mut history_index,
        ));
        assert_eq!(line.as_string(), "second");

        assert!(reader.apply_command(
            LineEditCommand::HistoryPrevious,
            &mut line,
            &mut draft,
            &mut history_index,
        ));
        assert_eq!(line.as_string(), "first");

        assert!(reader.apply_command(
            LineEditCommand::HistoryNext,
            &mut line,
            &mut draft,
            &mut history_index,
        ));
        assert_eq!(line.as_string(), "second");

        assert!(reader.apply_command(
            LineEditCommand::HistoryNext,
            &mut line,
            &mut draft,
            &mut history_index,
        ));
        assert_eq!(line.as_string(), "draft");
    }

    #[test]
    fn parses_cli_commands() {
        let cli = Cli::try_parse_from([
            "werk", "--device", "cuda", "serve", "--host", "0.0.0.0", "--port", "8080", "--model",
            "m",
        ])
        .unwrap();
        assert_eq!(cli.device, Some(DeviceArg::Cuda));
        match cli.command.unwrap() {
            Commands::Serve {
                host,
                port,
                model,
                api_key,
                api_keys,
                allow_unauthenticated,
                verbose,
            } => {
                assert_eq!(host, "0.0.0.0");
                assert_eq!(port, 8080);
                assert_eq!(model.as_deref(), Some("m"));
                assert!(api_key.is_none());
                assert!(api_keys.is_none());
                assert!(!allow_unauthenticated);
                assert!(!verbose);
            }
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from(["werk", "serve", "--verbose"]).unwrap();
        match cli.command.unwrap() {
            Commands::Serve { verbose, .. } => assert!(verbose),
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from(["werk", "serve", "--api-key", "sk-test"]).unwrap();
        match cli.command.unwrap() {
            Commands::Serve { api_key, .. } => {
                assert_eq!(api_key.as_deref(), Some("sk-test"));
            }
            command => panic!("unexpected command: {command:?}"),
        }

        let cli =
            Cli::try_parse_from(["werk", "serve", "--api-keys", "/tmp/api-keys.toml"]).unwrap();
        match cli.command.unwrap() {
            Commands::Serve { api_keys, .. } => {
                assert_eq!(api_keys.as_deref(), Some(Path::new("/tmp/api-keys.toml")));
            }
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from(["werk", "serve", "--allow-unauthenticated"]).unwrap();
        match cli.command.unwrap() {
            Commands::Serve {
                allow_unauthenticated,
                ..
            } => assert!(allow_unauthenticated),
            command => panic!("unexpected command: {command:?}"),
        }

        assert!(
            Cli::try_parse_from([
                "werk",
                "serve",
                "--api-key",
                "sk-test",
                "--api-keys",
                "/tmp/api-keys.toml",
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "werk",
                "serve",
                "--api-key",
                "sk-test",
                "--allow-unauthenticated",
            ])
            .is_err()
        );

        let cli = Cli::try_parse_from([
            "werk",
            "auth",
            "api-key",
            "generate",
            "--path",
            "/tmp/api-keys.toml",
            "--name",
            "open-webui",
        ])
        .unwrap();
        match cli.command.unwrap() {
            Commands::Auth {
                command:
                    AuthCommands::ApiKey {
                        command:
                            ApiKeyAuthCommands::Generate {
                                path, name, force, ..
                            },
                    },
            } => {
                assert_eq!(path.as_deref(), Some(Path::new("/tmp/api-keys.toml")));
                assert_eq!(name, "open-webui");
                assert!(!force);
            }
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from(["werk", "import", "/tmp/model", "--name", "local"]).unwrap();
        match cli.command.unwrap() {
            Commands::Import { path, name } => {
                assert_eq!(path, PathBuf::from("/tmp/model"));
                assert_eq!(name, "local");
            }
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from([
            "werk",
            "pull",
            "org/repo",
            "--name",
            "repo",
            "--file",
            "model.Q4_K_M.gguf",
        ])
        .unwrap();
        match cli.command.unwrap() {
            Commands::Pull { repo, name, file } => {
                assert_eq!(repo, "org/repo");
                assert_eq!(name.as_deref(), Some("repo"));
                assert_eq!(file.as_deref(), Some("model.Q4_K_M.gguf"));
            }
            command => panic!("unexpected command: {command:?}"),
        }

        let cli =
            Cli::try_parse_from(["werk", "auth", "huggingface", "login", "--token", "hf_test"])
                .unwrap();
        match cli.command.unwrap() {
            Commands::Auth {
                command:
                    AuthCommands::HuggingFace {
                        command: HuggingFaceAuthCommands::Login { token },
                    },
            } => assert_eq!(token.as_deref(), Some("hf_test")),
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from(["werk", "auth", "hf", "status"]).unwrap();
        match cli.command.unwrap() {
            Commands::Auth {
                command:
                    AuthCommands::HuggingFace {
                        command: HuggingFaceAuthCommands::Status,
                    },
            } => {}
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from(["werk", "rm", "repo"]).unwrap();
        match cli.command.unwrap() {
            Commands::Remove { id } => assert_eq!(id, "repo"),
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from([
            "werk",
            "run",
            "gemma-2b-it",
            "hello",
            "--image",
            "image.png",
            "--debug",
        ])
        .unwrap();
        match cli.command.unwrap() {
            Commands::Run {
                model,
                prompt,
                max_tokens,
                images,
                debug,
                ..
            } => {
                assert_eq!(model, "gemma-2b-it");
                assert_eq!(prompt, vec!["hello"]);
                assert_eq!(max_tokens, DEFAULT_MAX_NEW_TOKENS);
                assert_eq!(images, vec!["image.png"]);
                assert!(debug);
            }
            command => panic!("unexpected command: {command:?}"),
        }

        let cli =
            Cli::try_parse_from(["werk", "run", "tiny", "hello", "--max-tokens", "42"]).unwrap();
        match cli.command.unwrap() {
            Commands::Run { max_tokens, .. } => assert_eq!(max_tokens, 42),
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from([
            "werk",
            "chat",
            "gemma-2b-it",
            "--stream-granularity",
            "chunk",
            "--debug",
        ])
        .unwrap();
        match cli.command.unwrap() {
            Commands::Chat {
                model,
                stream_granularity,
                debug,
                ..
            } => {
                assert_eq!(model, "gemma-2b-it");
                assert_eq!(stream_granularity, StreamGranularityArg::Chunk);
                assert!(debug);
            }
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from(["werk", "backend", "install", "llama-cuda"]).unwrap();
        match cli.command.unwrap() {
            Commands::Backend {
                command: BackendCommands::Install { target },
            } => assert_eq!(target, BackendInstallArg::LlamaCuda),
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from(["werk", "backend", "install", "llama-rocm"]).unwrap();
        match cli.command.unwrap() {
            Commands::Backend {
                command: BackendCommands::Install { target },
            } => assert_eq!(target, BackendInstallArg::LlamaRocm),
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from(["werk", "backend", "install", "vllm"]).unwrap();
        match cli.command.unwrap() {
            Commands::Backend {
                command: BackendCommands::Install { target },
            } => assert_eq!(target, BackendInstallArg::Vllm),
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from(["werk", "backend", "install", "onnx-rocm"]).unwrap();
        match cli.command.unwrap() {
            Commands::Backend {
                command: BackendCommands::Install { target },
            } => assert_eq!(target, BackendInstallArg::OnnxRocm),
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from(["werk", "backend", "list"]).unwrap();
        match cli.command.unwrap() {
            Commands::Backend {
                command: BackendCommands::List,
            } => {}
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from(["werk", "artifacts", "build", "phi"]).unwrap();
        match cli.command.unwrap() {
            Commands::Artifacts {
                command: ArtifactCommands::Build { model },
            } => assert_eq!(model, "phi"),
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from(["werk", "--backend", "vulkan", "chat", "tiny"]).unwrap();
        assert_eq!(cli.backend, BackendArg::Vulkan);
        match cli.command.unwrap() {
            Commands::Chat { model, .. } => assert_eq!(model, "tiny"),
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from(["werk", "--backend", "candle", "chat", "tiny"]).unwrap();
        assert_eq!(cli.backend, BackendArg::Candle);
        match cli.command.unwrap() {
            Commands::Chat { model, .. } => assert_eq!(model, "tiny"),
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from(["werk", "--backend", "vllm", "chat", "tiny"]).unwrap();
        assert_eq!(cli.backend, BackendArg::Vllm);
        match cli.command.unwrap() {
            Commands::Chat {
                model, max_tokens, ..
            } => {
                assert_eq!(model, "tiny");
                assert_eq!(max_tokens, DEFAULT_MAX_NEW_TOKENS);
            }
            command => panic!("unexpected command: {command:?}"),
        }

        let cli =
            Cli::try_parse_from(["werk", "--backend", "transformers", "chat", "tiny"]).unwrap();
        assert_eq!(cli.backend, BackendArg::Transformers);
        match cli.command.unwrap() {
            Commands::Chat { model, .. } => assert_eq!(model, "tiny"),
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from(["werk", "chat", "tiny", "--max-tokens", "64"]).unwrap();
        match cli.command.unwrap() {
            Commands::Chat { max_tokens, .. } => assert_eq!(max_tokens, 64),
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from(["werk", "chat", "tiny", "--batch-size", "128"]).unwrap();
        assert_eq!(cli.llama.batch_size, Some(128));
        assert!(matches!(cli.command, Some(Commands::Chat { .. })));

        let cli = Cli::try_parse_from(["werk", "chat", "tiny", "--single-turn"]).unwrap();
        match cli.command.unwrap() {
            Commands::Chat { no_history, .. } => assert!(no_history),
            command => panic!("unexpected command: {command:?}"),
        }

        let cli =
            Cli::try_parse_from(["werk", "chat", "tiny", "--chat-template", "generic"]).unwrap();
        match cli.command.unwrap() {
            Commands::Chat { chat_template, .. } => {
                assert_eq!(chat_template, Some(ChatTemplateArg::Generic));
            }
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from(["werk", "--backend", "rocm", "chat", "tiny"]).unwrap();
        assert_eq!(cli.backend, BackendArg::Rocm);
        match cli.command.unwrap() {
            Commands::Chat { model, .. } => assert_eq!(model, "tiny"),
            command => panic!("unexpected command: {command:?}"),
        }

        let burn = Cli::try_parse_from(["werk", "--backend", "burn", "chat", "tiny"]);
        if cfg!(feature = "burn-experimental") {
            assert!(burn.is_ok());
        } else {
            assert!(burn.is_err());
        }

        let cli = Cli::try_parse_from([
            "werk",
            "--backend",
            "onnx",
            "--no-auto-install-backends",
            "chat",
            "tiny",
        ])
        .unwrap();
        assert_eq!(cli.backend, BackendArg::Onnx);
        assert!(cli.no_auto_install_backends);
        match cli.command.unwrap() {
            Commands::Chat { model, .. } => assert_eq!(model, "tiny"),
            command => panic!("unexpected command: {command:?}"),
        }

        let cli =
            Cli::try_parse_from(["werk", "select-file", "tiny", "tinyllama.Q4_K_M.gguf"]).unwrap();
        match cli.command.unwrap() {
            Commands::SelectFile { id, file } => {
                assert_eq!(id, "tiny");
                assert_eq!(file, "tinyllama.Q4_K_M.gguf");
            }
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from([
            "werk",
            "estimate",
            "org/repo",
            "--file",
            "model.Q4_K_M.gguf",
            "--verbose",
        ])
        .unwrap();
        match cli.command.unwrap() {
            Commands::Estimate {
                model,
                file,
                verbose,
                ..
            } => {
                assert_eq!(model, "org/repo");
                assert_eq!(file.as_deref(), Some("model.Q4_K_M.gguf"));
                assert!(verbose);
            }
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from([
            "werk",
            "--backend",
            "cuda",
            "--ctx-size",
            "2048",
            "--kv-cache-type",
            "q8-0",
            "bench",
            "tiny",
            "--prompt",
            "hello",
            "--runs",
            "3",
            "--warmups",
            "1",
            "--temperature",
            "0.2",
            "--compare",
            "legacy",
            "--json",
        ])
        .unwrap();
        assert_eq!(cli.backend, BackendArg::Cuda);
        assert_eq!(cli.llama.ctx_size, Some(2048));
        assert_eq!(cli.llama.kv_cache_type, Some(KvCacheTypeArg::Q8_0));
        match cli.command.unwrap() {
            Commands::Bench {
                model,
                prompt,
                runs,
                warmups,
                temperature,
                compare,
                json,
                ..
            } => {
                assert_eq!(model, "tiny");
                assert_eq!(prompt, "hello");
                assert_eq!(runs, 3);
                assert_eq!(warmups, 1);
                assert_eq!(temperature, 0.2);
                assert_eq!(compare, BenchCompareArg::Legacy);
                assert!(json);
            }
            command => panic!("unexpected command: {command:?}"),
        }
    }

    #[test]
    fn parses_media_command_families() {
        let cli = Cli::try_parse_from([
            "werk",
            "image",
            "generate",
            "flux",
            "--prompt",
            "orbital station",
            "--width",
            "1024",
            "--batch-size",
            "2",
            "--no-compile",
        ])
        .unwrap();
        match cli.command.unwrap() {
            Commands::Image {
                command: ImageCommands::Generate(args),
            } => {
                assert_eq!(args.model, "flux");
                assert_eq!(args.prompt.prompt.as_deref(), Some("orbital station"));
                assert_eq!(args.dimensions.width, Some(1024));
                assert_eq!(args.dimensions.batch_size, Some(2));
                assert!(args.routing.no_compile);
            }
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from([
            "werk",
            "image",
            "edit",
            "inpaint",
            "--image",
            "source.png",
            "--prompt",
            "remove the sign",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Image {
                command: ImageCommands::Edit(_)
            })
        ));

        let cli = Cli::try_parse_from([
            "werk",
            "video",
            "animate",
            "wan",
            "--image",
            "first.png",
            "--prompt",
            "camera moves forward",
            "--frames",
            "81",
        ])
        .unwrap();
        match cli.command.unwrap() {
            Commands::Video {
                command: VideoCommands::Animate(args),
            } => {
                assert_eq!(args.model, "wan");
                assert_eq!(args.image, PathBuf::from("first.png"));
                assert_eq!(args.core.frames, Some(81));
            }
            command => panic!("unexpected command: {command:?}"),
        }

        for argv in [
            vec!["werk", "video", "generate", "wan", "--prompt", "clouds"],
            vec!["werk", "video", "transform", "wan", "--video", "source.mp4"],
            vec!["werk", "video", "upscale", "wan", "--video", "source.mp4"],
            vec![
                "werk",
                "audio",
                "generate",
                "musicgen",
                "--prompt",
                "ambient score",
            ],
            vec!["werk", "audio", "speak", "kokoro", "--text", "hello"],
            vec![
                "werk",
                "audio",
                "transcribe",
                "whisper",
                "--input",
                "speech.wav",
            ],
            vec!["werk", "audio", "separate", "demucs", "--input", "song.wav"],
        ] {
            Cli::try_parse_from(argv).unwrap();
        }
    }

    #[test]
    fn parses_media_horizontal_commands_and_filters() {
        let cli = Cli::try_parse_from([
            "werk",
            "estimate",
            "flux",
            "--task",
            "image-generation",
            "--width",
            "1024",
            "--height",
            "768",
            "--steps",
            "30",
            "--json",
        ])
        .unwrap();
        match cli.command.unwrap() {
            Commands::Estimate {
                task,
                width,
                height,
                steps,
                json,
                ..
            } => {
                assert_eq!(task, Some(InferenceTask::ImageGeneration));
                assert_eq!(width, Some(1024));
                assert_eq!(height, Some(768));
                assert_eq!(steps, Some(30));
                assert!(json);
            }
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from([
            "werk",
            "list",
            "--task",
            "image-generation",
            "--input-modality",
            "text",
            "--output-modality",
            "image",
            "--family",
            "flux",
            "--layout",
            "diffusers",
            "--json",
        ])
        .unwrap();
        match cli.command.unwrap() {
            Commands::List {
                task,
                input,
                output,
                family,
                layout,
                json,
            } => {
                assert_eq!(task, Some(InferenceTask::ImageGeneration));
                assert_eq!(input, Some(InputModality::Text));
                assert_eq!(output, Some(OutputModality::Image));
                assert_eq!(family.as_deref(), Some("flux"));
                assert_eq!(layout, Some(RepositoryLayout::Diffusers));
                assert!(json);
            }
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from([
            "werk",
            "--backend",
            "cuda",
            "parameters",
            "flux",
            "--task",
            "image-generation",
            "--sources",
        ])
        .unwrap();
        assert_eq!(cli.backend, BackendArg::Cuda);
        assert!(matches!(
            cli.command,
            Some(Commands::Parameters {
                task: Some(InferenceTask::ImageGeneration),
                sources: true,
                ..
            })
        ));

        let cli = Cli::try_parse_from([
            "werk",
            "doctor",
            "--task",
            "music-generation",
            "--runtime",
            "diffusers-cuda",
            "--model",
            "musicgen",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Doctor {
                command: None,
                task: Some(InferenceTask::MusicGeneration),
                ..
            })
        ));
        let cli = Cli::try_parse_from(["werk", "doctor", "perf", "tiny"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Doctor {
                command: Some(DoctorCommands::Perf { .. }),
                ..
            })
        ));
    }

    #[test]
    fn media_arguments_become_canonical_typed_overrides() {
        let cli = Cli::try_parse_from([
            "werk",
            "image",
            "generate",
            "flux",
            "--prompt",
            "test",
            "--width",
            "640",
            "--no-image-vae-tiling",
            "--image-control-json",
            r#"[{"type":"depth","weight":0.8}]"#,
            "--set",
            "image.seed=17",
            "--no-allow-cpu-offload",
        ])
        .unwrap();
        let Commands::Image {
            command: ImageCommands::Generate(args),
        } = cli.command.unwrap()
        else {
            panic!("image command expected");
        };
        let parameters =
            media_parameters(&args, &args.routing, InferenceTask::ImageGeneration).unwrap();
        assert_eq!(
            parameters.get("image.width"),
            Some(&ParameterValue::Integer(640))
        );
        assert_eq!(
            parameters.get("image.vae_tiling"),
            Some(&ParameterValue::Boolean(false))
        );
        assert!(matches!(
            parameters.get("image.controls"),
            Some(ParameterValue::List(values)) if matches!(
                values.first(),
                Some(ParameterValue::Object(_))
            )
        ));
        assert_eq!(
            parameters.get("image.seed"),
            Some(&ParameterValue::Integer(17))
        );
        assert!(!parameters.contains_key("image.prompt"));

        let routing =
            media_routing(&args.routing, BackendArg::Cuda, Some(DeviceArg::Cuda)).unwrap();
        assert_eq!(routing.backend, None);
        assert_eq!(routing.accelerator.as_deref(), Some("cuda"));
        assert_eq!(routing.device.as_deref(), Some("cuda"));
        assert_eq!(routing.allow_cpu_offload, OverrideBool::Disabled);

        let mlx = media_routing(&args.routing, BackendArg::Mlx, None).unwrap();
        assert_eq!(mlx.backend.as_deref(), Some("mlx"));
        assert_eq!(mlx.accelerator, None);
    }

    #[test]
    fn audio_lyrics_remain_a_canonical_parameter_override() {
        let cli = Cli::try_parse_from([
            "werk",
            "audio",
            "generate",
            "music",
            "--prompt",
            "slow synthwave",
            "--lyrics",
            "we cross the night",
        ])
        .unwrap();
        let Commands::Audio {
            command: AudioCommands::Generate(args),
        } = cli.command.unwrap()
        else {
            panic!("audio generate command expected");
        };
        let parameters =
            media_parameters(&args, &args.routing, InferenceTask::MusicGeneration).unwrap();
        assert_eq!(
            parameters.get("audio.lyrics"),
            Some(&ParameterValue::String("we cross the night".to_string()))
        );
        assert!(!parameters.contains_key("audio.prompt"));
    }

    #[test]
    fn no_subcommand_defaults_to_serve_backend_command() {
        let cli = Cli::try_parse_from(["werk"]).unwrap();
        assert!(cli.command.is_none());
        assert_eq!(cli.device, None);
        assert_eq!(cli.backend, BackendArg::Auto);
        assert!(matches!(
            backend_arg_to_choice(BackendArg::Mlx),
            BackendChoice::Mlx
        ));
        assert!(matches!(
            backend_arg_to_choice(BackendArg::Vulkan),
            BackendChoice::GgufPreferred {
                llama: LlamaCppMode::Vulkan,
                candle: CandleDeviceMode::Auto
            }
        ));
        assert!(matches!(
            backend_arg_to_choice(BackendArg::Metal),
            BackendChoice::GgufPreferred {
                llama: LlamaCppMode::Metal,
                candle: CandleDeviceMode::Metal
            }
        ));
        assert!(matches!(
            backend_arg_to_choice(BackendArg::LlamaHighlevel),
            BackendChoice::LlamaHighlevel(_)
        ));
        assert!(matches!(
            backend_arg_to_choice(BackendArg::LlamaLegacy),
            BackendChoice::LlamaFast(_)
        ));
        assert!(matches!(
            backend_arg_to_choice(BackendArg::Cuda),
            BackendChoice::GgufPreferred {
                llama: LlamaCppMode::Cuda,
                candle: CandleDeviceMode::Cuda
            }
        ));
        assert!(matches!(
            backend_arg_to_choice(BackendArg::Rocm),
            BackendChoice::GgufPreferred {
                llama: LlamaCppMode::Rocm,
                candle: CandleDeviceMode::Auto
            }
        ));
        #[cfg(feature = "burn-experimental")]
        assert!(matches!(
            backend_arg_to_choice(BackendArg::Burn),
            BackendChoice::Burn(_)
        ));
        assert!(matches!(
            backend_arg_to_choice(BackendArg::Candle),
            BackendChoice::Candle(CandleDeviceMode::Auto)
        ));
        assert!(matches!(
            backend_arg_to_choice(BackendArg::Vllm),
            BackendChoice::Vllm
        ));
        assert!(matches!(
            backend_arg_to_choice(BackendArg::Onnx),
            BackendChoice::OnnxRuntime(_)
        ));
    }

    #[test]
    fn linux_and_windows_auto_prefer_llama_server_for_gguf() {
        if cfg!(any(windows, target_os = "linux")) {
            let manifest = test_manifest(ModelFormat::Gguf, Some("llama"));
            let order = auto_candidates_for_manifest(&manifest);
            assert!(matches!(
                order[0],
                BackendChoice::LlamaServer(LlamaCppMode::Cuda)
            ));
            assert!(matches!(
                order[1],
                BackendChoice::LlamaServer(LlamaCppMode::Vulkan)
            ));
            assert!(matches!(
                order[2],
                BackendChoice::LlamaServer(LlamaCppMode::Cpu)
            ));
            assert!(matches!(
                order[3],
                BackendChoice::Candle(CandleDeviceMode::Cuda)
            ));
            assert!(matches!(
                order[4],
                BackendChoice::Candle(CandleDeviceMode::Cpu)
            ));
        }
    }

    #[test]
    fn qwen_gguf_without_tokenizer_json_rejects_candle_fallback() {
        if cfg!(any(windows, target_os = "linux")) {
            let store = test_store("qwen-gguf-no-candle-fallback");
            let manifest = test_manifest(ModelFormat::Gguf, Some("qwen3"));
            let err =
                selected_backend_for_manifest(&store, BackendChoice::Auto, &manifest).unwrap_err();
            let message = err.to_string();
            assert!(message.contains("llama.cpp server CUDA"));
            assert!(message.contains("Candle CUDA"));
            assert!(message.contains("Candle GGUF fallback requires tokenizer.json"));
        }
    }

    #[test]
    fn macos_auto_prefers_llama_server_metal_for_gguf() {
        if cfg!(target_os = "macos") {
            let manifest = test_manifest(ModelFormat::Gguf, Some("llama"));
            let order = auto_candidates_for_manifest(&manifest);
            assert!(matches!(
                order[0],
                BackendChoice::LlamaServer(LlamaCppMode::Metal)
            ));
            assert!(matches!(
                order[1],
                BackendChoice::LlamaServer(LlamaCppMode::Cpu)
            ));
            assert!(matches!(
                order[2],
                BackendChoice::Candle(CandleDeviceMode::Metal)
            ));
            assert!(matches!(
                order[3],
                BackendChoice::Candle(CandleDeviceMode::Cpu)
            ));
        }
    }

    #[test]
    fn gguf_auto_includes_rocm_only_when_probeable() {
        if cfg!(any(windows, target_os = "linux")) {
            let manifest = test_manifest(ModelFormat::Gguf, Some("llama"));
            let store = test_store("gguf-auto-rocm-gating");
            let plain =
                runtime_candidate_ids_for_selection(&store, &manifest, RequestedBackend::Auto);
            assert!(!plain.contains(&RuntimeId::LlamaServerRocm));

            install_fake_managed_llama_server(&store, LlamaCppMode::Rocm);
            let gated =
                runtime_candidate_ids_for_selection(&store, &manifest, RequestedBackend::Auto);
            let rocm = gated
                .iter()
                .position(|id| *id == RuntimeId::LlamaServerRocm)
                .unwrap();
            let vulkan = gated
                .iter()
                .position(|id| *id == RuntimeId::LlamaServerVulkan)
                .unwrap();
            assert!(rocm < vulkan);
        }
    }

    #[test]
    fn explicit_rocm_candidates_are_strict_for_compatible_formats() {
        let requested = backend_arg_to_choice(BackendArg::Rocm);

        let gguf = test_manifest(ModelFormat::Gguf, Some("llama"));
        assert_eq!(
            routing_candidates_for_debug(requested, &gguf),
            vec![RuntimeId::LlamaServerRocm]
        );

        let safetensors = test_manifest(ModelFormat::SafeTensors, Some("qwen3"));
        assert_eq!(
            routing_candidates_for_debug(requested, &safetensors),
            vec![RuntimeId::VllmRocm]
        );
        let unknown_safetensors = test_manifest(ModelFormat::SafeTensors, Some("unknown"));
        assert_eq!(
            routing_candidates_for_debug(requested, &unknown_safetensors),
            vec![RuntimeId::VllmRocm]
        );

        let onnx = test_manifest(ModelFormat::Onnx, None);
        assert_eq!(
            routing_candidates_for_debug(requested, &onnx),
            vec![RuntimeId::OnnxRuntimeRocm]
        );
    }

    #[test]
    fn explicit_rocm_selection_does_not_fall_back_to_cpu() {
        let store = test_store("explicit-rocm");
        let requested = backend_arg_to_choice(BackendArg::Rocm);

        let gguf = test_manifest(ModelFormat::Gguf, Some("llama"));
        match selected_backend_for_manifest(&store, requested, &gguf) {
            Ok(selected) => assert!(matches!(
                selected,
                BackendChoice::LlamaServer(LlamaCppMode::Rocm)
            )),
            Err(err) => {
                let message = err.to_string();
                assert!(message.contains("llama.cpp server ROCm/HIP"));
                assert!(!message.contains("llama.cpp server CPU"));
                assert!(!message.contains("Candle CPU"));
            }
        }

        let safetensors = test_manifest(ModelFormat::SafeTensors, Some("qwen3"));
        match selected_backend_for_manifest(&store, requested, &safetensors) {
            Ok(selected) => assert!(matches!(selected, BackendChoice::VllmRocm)),
            Err(err) => {
                let message = err.to_string();
                assert!(message.contains("vLLM ROCm"));
                assert!(message.contains("ROCm") || message.contains("HIP"));
                assert!(!message.contains("Candle CPU"));
            }
        }

        let onnx = test_manifest(ModelFormat::Onnx, None);
        match selected_backend_for_manifest(&store, requested, &onnx) {
            Ok(selected) => assert!(matches!(
                selected,
                BackendChoice::OnnxRuntime(OnnxRuntimeMode::Rocm)
            )),
            Err(err) => {
                let message = err.to_string();
                assert!(message.contains("ONNX Runtime ROCm"));
                assert!(!message.contains("ONNX Runtime CPU"));
            }
        }
    }

    #[test]
    fn onnx_vulkan_and_mlx_gpu_requests_reject_without_cpu_fallback() {
        let store = test_store("unsupported-explicit-accelerators");

        let onnx = test_manifest(ModelFormat::Onnx, None);
        let vulkan = backend_arg_to_choice(BackendArg::Vulkan);
        assert!(routing_candidates_for_debug(vulkan, &onnx).is_empty());
        let err = selected_backend_for_manifest(&store, vulkan, &onnx).unwrap_err();
        assert!(err.to_string().contains("no runtime candidates"));
        assert!(!err.to_string().contains("ONNX Runtime CPU"));

        let mlx = test_manifest(ModelFormat::Mlx, Some("llama"));
        for backend in [BackendArg::Cuda, BackendArg::Rocm, BackendArg::Vulkan] {
            let requested = backend_arg_to_choice(backend);
            assert!(routing_candidates_for_debug(requested, &mlx).is_empty());
            let err = selected_backend_for_manifest(&store, requested, &mlx).unwrap_err();
            assert!(err.to_string().contains("no runtime candidates"));
            assert!(!err.to_string().contains("Candle CPU"));
        }
    }

    #[test]
    fn gguf_explicit_mlx_vllm_and_onnx_reject() {
        let store = test_store("gguf-strict-runtime-rejections");
        let gguf = test_manifest(ModelFormat::Gguf, Some("llama"));

        for requested in [
            BackendChoice::Mlx,
            BackendChoice::Vllm,
            BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cuda),
        ] {
            let err = selected_backend_for_manifest(&store, requested, &gguf).unwrap_err();
            let message = err.to_string();
            assert!(!message.contains("llama.cpp server CPU"));
            assert!(!message.contains("Candle CPU"));
        }
    }

    #[test]
    fn explicit_gguf_rocm_can_select_managed_rocm_server() {
        let store = test_store("gguf-rocm");
        install_fake_managed_llama_server(&store, LlamaCppMode::Rocm);
        let manifest = test_manifest(ModelFormat::Gguf, Some("llama"));
        let selected = selected_backend_for_manifest(
            &store,
            backend_arg_to_choice(BackendArg::Rocm),
            &manifest,
        )
        .unwrap();
        assert!(matches!(
            selected,
            BackendChoice::LlamaServer(LlamaCppMode::Rocm)
        ));
    }

    #[test]
    fn auto_safetensors_prefers_vllm_then_candle_on_linux_and_windows() {
        if cfg!(any(windows, target_os = "linux")) {
            let manifest = test_manifest(ModelFormat::SafeTensors, Some("phi3"));
            let order = auto_candidates_for_manifest(&manifest);
            assert!(matches!(order[0], BackendChoice::Vllm));
            assert!(matches!(
                order[1],
                BackendChoice::Candle(CandleDeviceMode::Cuda)
            ));
            assert!(matches!(
                order[2],
                BackendChoice::Candle(CandleDeviceMode::Cpu)
            ));
            assert!(
                !order
                    .iter()
                    .any(|choice| matches!(choice, BackendChoice::Burn(_)))
            );
        }
    }

    #[test]
    fn runtime_registry_omits_burn_from_normal_policy_unless_experimental() {
        let burn_cuda = runtime_descriptor(RuntimeId::BurnCuda);
        assert_eq!(burn_cuda.display_name, "Burn CUDA");
        assert_eq!(burn_cuda.implemented, cfg!(feature = "burn-cuda"));
        assert_eq!(burn_cuda.install_target, None);
        let burn_cpu = runtime_descriptor(RuntimeId::BurnCpu);
        assert_eq!(burn_cpu.display_name, "Burn CPU");
        assert_eq!(burn_cpu.implemented, cfg!(feature = "burn-cpu"));
        assert_eq!(burn_cpu.install_target, None);

        let normal_runtime_ids = runtime_registry()
            .iter()
            .filter(|runtime| {
                cfg!(feature = "burn-experimental") || runtime.runtime != BackendRuntime::Burn
            })
            .map(|runtime| runtime.id)
            .collect::<Vec<_>>();
        if !cfg!(feature = "burn-experimental") {
            assert!(!normal_runtime_ids.contains(&RuntimeId::BurnCuda));
            assert!(!normal_runtime_ids.contains(&RuntimeId::BurnCpu));
        }
    }

    #[test]
    fn runtime_registry_exposes_real_onnxruntime_runtime() {
        let onnx = runtime_descriptor(RuntimeId::OnnxRuntimeCuda);
        assert_eq!(onnx.display_name, "ONNX Runtime CUDA");
        assert!(onnx.implemented);
        assert_eq!(onnx.install_target, None);
    }

    #[test]
    fn auto_install_policy_defaults_to_auto_only() {
        assert!(
            SelectionOptions::from_cli(BackendArg::Auto, false, false).provision_missing_backends
        );
        assert!(
            !SelectionOptions::from_cli(BackendArg::Auto, false, false).verbose_backend_installs
        );
        assert!(
            SelectionOptions::from_cli(BackendArg::Auto, false, false)
                .with_backend_install_output(true)
                .verbose_backend_installs
        );
        assert!(
            !SelectionOptions::from_cli(BackendArg::Onnx, false, false).provision_missing_backends
        );
        assert!(
            !SelectionOptions::from_cli(BackendArg::Cuda, false, false).provision_missing_backends
        );
        assert!(
            SelectionOptions::from_cli(BackendArg::Cuda, true, false).provision_missing_backends
        );
        assert!(
            !SelectionOptions::from_cli(BackendArg::Auto, false, true).provision_missing_backends
        );
    }

    #[test]
    fn managed_backend_install_output_requires_verbose_or_debug_command() {
        let quiet_chat = Commands::Chat {
            model: "tiny".to_string(),
            max_tokens: 256,
            temperature: None,
            top_p: None,
            seed: None,
            chat_template: None,
            no_history: false,
            images: Vec::new(),
            stream_granularity: StreamGranularityArg::Token,
            verbose: false,
            debug: false,
        };
        assert!(!command_backend_install_verbose(&quiet_chat));

        let verbose_chat = Commands::Chat {
            model: "tiny".to_string(),
            max_tokens: 256,
            temperature: None,
            top_p: None,
            seed: None,
            chat_template: None,
            no_history: false,
            images: Vec::new(),
            stream_granularity: StreamGranularityArg::Token,
            verbose: true,
            debug: false,
        };
        assert!(command_backend_install_verbose(&verbose_chat));

        let debug_run = Commands::Run {
            model: "tiny".to_string(),
            prompt: vec!["hello".to_string()],
            max_tokens: 128,
            temperature: None,
            top_p: None,
            seed: None,
            chat_template: None,
            images: Vec::new(),
            verbose: false,
            debug: true,
        };
        assert!(command_backend_install_verbose(&debug_run));
    }

    #[test]
    fn llama_server_auto_install_policy_preserves_macos_metal_only() {
        assert!(!should_auto_install_llama_server(LlamaCppMode::Cuda));
        assert!(!should_auto_install_llama_server(LlamaCppMode::Rocm));
        assert!(!should_auto_install_llama_server(LlamaCppMode::Vulkan));
        assert_eq!(
            should_auto_install_llama_server(LlamaCppMode::Metal),
            cfg!(target_os = "macos")
        );
        assert!(!should_auto_install_llama_server(LlamaCppMode::Cpu));
    }

    #[test]
    fn safetensors_runtime_candidates_omit_burn_and_keep_cpu_as_auto_fallback() {
        if cfg!(any(windows, target_os = "linux")) {
            let manifest = test_manifest(ModelFormat::SafeTensors, Some("unknown"));
            let candidates = auto_runtime_candidates_for_manifest(&manifest);
            assert_eq!(
                candidates,
                vec![RuntimeId::CandleCuda, RuntimeId::CandleCpu]
            );
            assert!(!candidates.contains(&RuntimeId::BurnCuda));
            assert!(!candidates.contains(&RuntimeId::BurnCpu));

            let concrete = auto_candidates_for_manifest(&manifest);
            assert!(matches!(
                concrete[0],
                BackendChoice::Candle(CandleDeviceMode::Cuda)
            ));
            assert!(matches!(
                concrete[1],
                BackendChoice::Candle(CandleDeviceMode::Cpu)
            ));
        }
    }

    #[test]
    fn safetensors_vulkan_has_no_silent_cpu_or_candle_fallback() {
        let manifest = test_manifest(ModelFormat::SafeTensors, Some("phi3"));
        let requested = backend_arg_to_choice(BackendArg::Vulkan);
        assert!(routing_candidates_for_debug(requested, &manifest).is_empty());

        let store = test_store("safetensors-vulkan-runtime");
        let err = selected_backend_for_manifest(&store, requested, &manifest).unwrap_err();
        assert!(err.to_string().contains("no runtime candidates"));
        assert!(!err.to_string().contains("Candle CPU"));
    }

    #[test]
    fn backend_selection_routes_gguf_cuda_to_llama_server() {
        let store = test_store("gguf-cuda");
        install_fake_managed_llama_server(&store, LlamaCppMode::Cuda);
        let manifest = test_manifest(ModelFormat::Gguf, Some("llama"));
        let selected = selected_backend_for_manifest(
            &store,
            BackendChoice::GgufPreferred {
                llama: LlamaCppMode::Cuda,
                candle: CandleDeviceMode::Cuda,
            },
            &manifest,
        )
        .unwrap();
        assert!(matches!(
            selected,
            BackendChoice::LlamaServer(LlamaCppMode::Cuda)
        ));
    }

    #[test]
    fn backend_selection_routes_gguf_vulkan_to_llama_server() {
        let store = test_store("gguf-vulkan");
        install_fake_managed_llama_server(&store, LlamaCppMode::Vulkan);
        let manifest = test_manifest(ModelFormat::Gguf, Some("llama"));
        let selected = selected_backend_for_manifest(
            &store,
            BackendChoice::LlamaServer(LlamaCppMode::Vulkan),
            &manifest,
        )
        .unwrap();
        assert!(matches!(
            selected,
            BackendChoice::LlamaServer(LlamaCppMode::Vulkan)
        ));
    }

    #[test]
    fn backend_selection_routes_gguf_metal_to_llama_server() {
        if cfg!(target_os = "macos") {
            let store = test_store("gguf-metal");
            install_fake_managed_llama_server(&store, LlamaCppMode::Metal);
            let manifest = test_manifest(ModelFormat::Gguf, Some("llama"));
            let selected = selected_backend_for_manifest(
                &store,
                backend_arg_to_choice(BackendArg::Metal),
                &manifest,
            )
            .unwrap();
            assert!(matches!(
                selected,
                BackendChoice::LlamaServer(LlamaCppMode::Metal)
            ));
        }
    }

    #[test]
    fn backend_selection_routes_safetensors_cuda_without_burn_or_cpu_fallback() {
        let store = test_store("safetensors-cuda");
        let manifest = test_manifest(ModelFormat::SafeTensors, Some("unknown"));
        let result = selected_backend_for_manifest(
            &store,
            BackendChoice::GgufPreferred {
                llama: LlamaCppMode::Cuda,
                candle: CandleDeviceMode::Cuda,
            },
            &manifest,
        );
        match result {
            Ok(selected) => assert!(matches!(
                selected,
                BackendChoice::Candle(CandleDeviceMode::Cuda)
            )),
            Err(err) => {
                let message = err.to_string();
                assert!(message.contains("Candle CUDA"));
                assert!(!message.contains("Burn CUDA"));
                assert!(!message.contains("Burn CPU"));
                assert!(!message.contains("Candle CPU"));
            }
        }
    }

    #[test]
    fn explicit_burn_selection_never_falls_back_to_candle() {
        let store = test_store("explicit-burn-missing");
        let manifest = test_manifest(ModelFormat::SafeTensors, Some("phi3"));
        let err =
            selected_backend_for_manifest(&store, BackendChoice::Burn(BurnMode::Cuda), &manifest)
                .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("Burn"));
        assert!(!message.contains("Candle CUDA"));
    }

    #[test]
    fn explicit_vllm_selection_never_falls_back_to_candle() {
        let store = test_store("explicit-vllm-missing");
        let manifest = test_manifest(ModelFormat::SafeTensors, Some("phi3"));
        let err =
            selected_backend_for_manifest(&store, BackendChoice::Vllm, &manifest).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("vLLM CUDA"));
        assert!(!message.contains("Candle CUDA:"));
    }

    #[test]
    fn explicit_onnx_selection_never_falls_back_to_candle() {
        let store = test_store("explicit-onnx-missing");
        let manifest = test_manifest(ModelFormat::SafeTensors, Some("phi3"));
        let err = selected_backend_for_request(
            &store,
            BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cuda),
            &manifest,
            false,
            SelectionOptions::from_cli(BackendArg::Onnx, false, true),
        )
        .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("ONNX Runtime CUDA"));
        assert!(!message.contains("Candle CUDA"));
    }

    #[test]
    fn auto_safetensors_fallback_note_is_suppressed_outside_debug() {
        let store = test_store("auto-burn-fallback-note");
        let manifest = test_manifest(ModelFormat::SafeTensors, Some("phi3"));
        let note = verbose_fallback_note(
            &store,
            BackendChoice::Auto,
            &manifest,
            false,
            BackendChoice::Candle(CandleDeviceMode::Cuda),
        );
        assert!(note.is_none());
    }

    #[test]
    fn auto_safetensors_can_fallback_to_candle_without_verbose_burn_note() {
        let store = test_store("auto-burn-fallback");
        let manifest = test_manifest(ModelFormat::SafeTensors, Some("unknown"));
        let selected =
            selected_backend_for_manifest(&store, BackendChoice::Auto, &manifest).unwrap();
        assert!(matches!(selected, BackendChoice::Candle(_)));
        let note = verbose_fallback_note(&store, BackendChoice::Auto, &manifest, false, selected);
        assert!(note.is_none());
    }

    #[test]
    fn backend_selection_falls_back_to_candle_cuda_when_vllm_missing() {
        let store = test_store("safetensors-cuda-fallback");
        let manifest = test_manifest(ModelFormat::SafeTensors, Some("unknown"));
        let result = selected_backend_for_manifest(
            &store,
            BackendChoice::GgufPreferred {
                llama: LlamaCppMode::Cuda,
                candle: CandleDeviceMode::Cuda,
            },
            &manifest,
        );
        match result {
            Ok(selected) => assert!(matches!(
                selected,
                BackendChoice::Candle(CandleDeviceMode::Cuda)
            )),
            Err(err) => {
                let message = err.to_string();
                assert!(message.contains("Candle CUDA"));
                assert!(!message.contains("Burn"));
                assert!(!message.contains("Candle CPU"));
            }
        }
    }

    #[test]
    fn explicit_cuda_selection_never_selects_cpu_fallback() {
        let store = test_store("explicit-cuda");
        install_fake_managed_llama_server(&store, LlamaCppMode::Cuda);
        let gguf = test_manifest(ModelFormat::Gguf, Some("llama"));
        let selected = selected_backend_for_manifest(
            &store,
            BackendChoice::GgufPreferred {
                llama: LlamaCppMode::Cuda,
                candle: CandleDeviceMode::Cuda,
            },
            &gguf,
        )
        .unwrap();
        assert!(matches!(
            selected,
            BackendChoice::LlamaServer(LlamaCppMode::Cuda)
        ));

        let safetensors = test_manifest(ModelFormat::SafeTensors, Some("unknown"));
        let result = selected_backend_for_manifest(
            &store,
            BackendChoice::GgufPreferred {
                llama: LlamaCppMode::Cuda,
                candle: CandleDeviceMode::Cuda,
            },
            &safetensors,
        );
        match result {
            Ok(selected) => assert!(matches!(
                selected,
                BackendChoice::Candle(CandleDeviceMode::Cuda)
            )),
            Err(err) => {
                let message = err.to_string();
                assert!(message.contains("Candle CUDA"));
                assert!(!message.contains("Burn"));
                assert!(!message.contains("Candle CPU"));
            }
        }
    }

    #[test]
    fn backend_selection_rejects_safetensors_vulkan() {
        let store = test_store("safetensors-vulkan");
        let manifest = test_manifest(ModelFormat::SafeTensors, Some("phi3"));
        let err = selected_backend_for_manifest(
            &store,
            backend_arg_to_choice(BackendArg::Vulkan),
            &manifest,
        )
        .unwrap_err();
        assert!(err.to_string().contains("no runtime candidates"));
        assert!(!err.to_string().contains("Candle CPU"));
    }

    #[test]
    fn backend_selection_routes_safetensors_cpu_to_candle_cpu() {
        let store = test_store("safetensors-cpu");
        let manifest = test_manifest(ModelFormat::SafeTensors, Some("phi3"));
        let selected = selected_backend_for_manifest(
            &store,
            BackendChoice::GgufPreferred {
                llama: LlamaCppMode::Cpu,
                candle: CandleDeviceMode::Cpu,
            },
            &manifest,
        )
        .unwrap();
        assert!(matches!(
            selected,
            BackendChoice::Candle(CandleDeviceMode::Cpu)
        ));
    }

    #[test]
    fn backend_selection_routes_mlx_format_to_mlx() {
        let store = test_store("mlx");
        let manifest = test_manifest(ModelFormat::Mlx, Some("llama"));
        let result = selected_backend_for_manifest(&store, BackendChoice::Mlx, &manifest);
        match result {
            Ok(selected) => assert!(matches!(selected, BackendChoice::Mlx)),
            Err(err) => assert!(err.to_string().contains("mlx-lm is unavailable")),
        }
    }

    #[test]
    fn image_request_does_not_select_plain_mlx_fallback() {
        let store = test_store("mlx-image-fallback");
        let manifest = test_manifest(ModelFormat::Mlx, Some("gemma4_unified"));
        let err = select_backend_from_runtime_candidates(
            &store,
            &[RuntimeId::Mlx],
            &manifest,
            RequestedBackend::Mlx,
            RequestCapabilities::text_with_images(true, true),
            SelectionOptions::default(),
        )
        .unwrap_err();

        assert!(err.to_string().contains("MLX"));
        assert!(err.to_string().contains("VLM"));
    }

    #[test]
    fn image_inputs_reject_text_only_backends() {
        let err =
            ensure_backend_supports_images(BackendChoice::Candle(CandleDeviceMode::Cuda), true)
                .unwrap_err();
        assert!(err.to_string().contains("text-only"));
    }

    #[test]
    fn prepare_backend_for_chat_prepares_before_session() {
        #[derive(Clone)]
        struct RecordingBackend {
            calls: StdArc<StdMutex<Vec<&'static str>>>,
        }

        impl GenerationBackend for RecordingBackend {
            fn prepare(&self, _manifest: &ModelManifest) -> Result<()> {
                self.calls.lock().unwrap().push("prepare");
                Ok(())
            }

            fn start_chat_session(
                &self,
                _manifest: &ModelManifest,
                _seed: Option<u64>,
            ) -> Result<Option<Box<dyn ChatGenerationSession>>> {
                self.calls.lock().unwrap().push("start_chat_session");
                Ok(None)
            }

            fn generate(
                &self,
                _manifest: &ModelManifest,
                _request: GenerateRequest,
            ) -> Result<crate::backend::GenerateResponse> {
                unreachable!("not used")
            }

            fn generate_stream(
                &self,
                _manifest: ModelManifest,
                _request: GenerateRequest,
            ) -> crate::backend::GenerateStream {
                unreachable!("not used")
            }
        }

        let calls = StdArc::new(StdMutex::new(Vec::new()));
        let backend = RecordingBackend {
            calls: calls.clone(),
        };
        let manifest = test_manifest(ModelFormat::SafeTensors, Some("phi3"));
        let _ = prepare_backend_for_chat(&backend, &manifest, None, false).unwrap();
        assert_eq!(
            calls.lock().unwrap().as_slice(),
            &["prepare", "start_chat_session"]
        );
    }

    #[test]
    fn startup_banner_is_limited_to_interactive_terminal_commands() {
        let serve = Commands::Serve {
            host: "127.0.0.1".to_string(),
            port: 11434,
            model: None,
            api_key: None,
            api_keys: None,
            allow_unauthenticated: false,
            verbose: false,
        };
        assert!(should_print_startup_banner_for(&serve, true, true));
        assert!(!should_print_startup_banner_for(&serve, false, true));

        let run = Commands::Run {
            model: "tiny".to_string(),
            prompt: vec!["hello".to_string()],
            max_tokens: 128,
            temperature: None,
            top_p: None,
            seed: None,
            chat_template: None,
            images: Vec::new(),
            verbose: false,
            debug: false,
        };
        assert!(should_print_startup_banner_for(&run, true, true));
        assert!(!should_print_startup_banner_for(&run, false, true));

        let chat = Commands::Chat {
            model: "tiny".to_string(),
            max_tokens: 256,
            temperature: None,
            top_p: None,
            seed: None,
            chat_template: None,
            no_history: false,
            images: Vec::new(),
            stream_granularity: StreamGranularityArg::Token,
            verbose: false,
            debug: false,
        };
        assert!(should_print_startup_banner_for(&chat, true, true));
        assert!(!should_print_startup_banner_for(&chat, true, false));

        let bench = Commands::Bench {
            model: "tiny".to_string(),
            prompt: "hello".to_string(),
            max_tokens: 128,
            runs: 1,
            warmups: 0,
            temperature: 0.0,
            top_p: None,
            seed: 42,
            compare: BenchCompareArg::None,
            print_native_info: false,
            json: true,
            debug: false,
        };
        assert!(!should_print_startup_banner_for(&bench, true, true));

        assert!(!should_print_startup_banner_for(
            &Commands::List {
                task: None,
                input: None,
                output: None,
                family: None,
                layout: None,
                json: false,
            },
            true,
            true
        ));
        assert!(!should_print_startup_banner_for(
            &Commands::Inspect {
                id: "tiny".to_string()
            },
            true,
            true
        ));
    }

    #[test]
    fn verbose_stats_report_stop_reason_and_unknown_prompt_eval() {
        let mut output = Vec::new();
        write_verbose_stats(
            &mut output,
            Some("ONNX Runtime CPU"),
            12,
            24,
            "stop_sequence",
            GenerationTimings {
                load_seconds: 0.25,
                warmup_seconds: 0.0,
                first_token_seconds: 0.5,
                prompt_seconds: f64::NAN,
                decode_seconds: 1.0,
                total_seconds: 1.25,
            },
            &["effective max new tokens: 256".to_string()],
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();

        assert!(output.contains("prompt eval duration: N/A"));
        assert!(output.contains("prompt eval rate:"));
        assert!(output.contains("N/A"));
        assert!(output.contains("finish reason:"));
        assert!(output.contains("stop_sequence"));
        assert!(output.contains("effective max new tokens: 256"));
    }

    #[test]
    fn gguf_llama_cpp_defaults_to_model_chat_template() {
        let manifest = test_manifest(ModelFormat::Gguf, Some("llama"));
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text("hello".to_string())),
            name: None,
        }];
        let prompt = prompt_for_backend(
            &manifest,
            &messages,
            BackendChoice::LlamaServer(LlamaCppMode::Metal),
            None,
        );
        let diagnostics = prompt_diagnostics(&prompt, messages.len(), Some(true));

        assert_eq!(prompt.chat_template.source, ChatTemplateSource::Model);
        assert_eq!(prompt.chat_template.name, "model");
        assert!(!prompt.chat_template.applied_by_werk);
        assert_eq!(prompt.prompt, "hello");
        assert!(diagnostics.contains(&"chat template source: model".to_string()));
        assert!(diagnostics.contains(&"chat template: model".to_string()));
        assert!(diagnostics.contains(&"chat template applied by werk: no".to_string()));
        assert!(!diagnostics.contains(&"chat template: generic".to_string()));
    }

    #[test]
    fn gguf_model_chat_template_keeps_structured_generation_messages() {
        let manifest = test_manifest(ModelFormat::Gguf, Some("llama"));
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text("hello".to_string())),
            name: None,
        }];
        let prompt = prompt_for_backend(
            &manifest,
            &messages,
            BackendChoice::LlamaServer(LlamaCppMode::Metal),
            None,
        );

        let request_messages = generation_request_messages(&prompt, &messages);

        assert_eq!(request_messages.len(), 1);
        assert_eq!(
            request_messages[0]
                .content
                .as_ref()
                .map(MessageContent::as_text),
            Some("hello".to_string())
        );
    }

    #[test]
    fn explicit_generic_chat_template_overrides_gguf_model_default() {
        let manifest = test_manifest(ModelFormat::Gguf, Some("llama"));
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text("hello".to_string())),
            name: None,
        }];
        let prompt = prompt_for_backend(
            &manifest,
            &messages,
            BackendChoice::LlamaServer(LlamaCppMode::Metal),
            Some(ChatTemplateArg::Generic),
        );

        assert_eq!(prompt.chat_template.source, ChatTemplateSource::Werk);
        assert_eq!(prompt.chat_template.name, "generic");
        assert!(prompt.chat_template.applied_by_werk);
        assert_eq!(
            prompt.chat_template.override_from_cli.as_deref(),
            Some("generic")
        );
        assert!(prompt.prompt.contains("user: hello"));
        assert!(prompt.prompt.ends_with("assistant: "));
    }

    #[test]
    fn werk_applied_chat_template_uses_rendered_prompt_only_for_generation() {
        let manifest = test_manifest(ModelFormat::Gguf, Some("llama"));
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text("hello".to_string())),
            name: None,
        }];
        let prompt = prompt_for_backend(
            &manifest,
            &messages,
            BackendChoice::LlamaServer(LlamaCppMode::Metal),
            Some(ChatTemplateArg::Generic),
        );

        let request_messages = generation_request_messages(&prompt, &messages);

        assert!(request_messages.is_empty());
    }

    #[test]
    fn explicit_none_chat_template_disables_werk_templating() {
        let manifest = test_manifest(ModelFormat::Gguf, Some("llama"));
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text("hello".to_string())),
            name: None,
        }];
        let prompt = prompt_for_backend(
            &manifest,
            &messages,
            BackendChoice::LlamaServer(LlamaCppMode::Metal),
            Some(ChatTemplateArg::None),
        );

        assert_eq!(prompt.chat_template.source, ChatTemplateSource::None);
        assert_eq!(prompt.chat_template.name, "none");
        assert!(!prompt.chat_template.applied_by_werk);
        assert_eq!(prompt.prompt, "hello");
        assert!(prompt.stop.is_empty());
    }

    #[test]
    fn onnx_phi3_still_uses_werk_phi3_chat_template() {
        let manifest = test_manifest(ModelFormat::Onnx, Some("phi3"));
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text("hello".to_string())),
            name: None,
        }];
        let prompt = prompt_for_backend(
            &manifest,
            &messages,
            BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cpu),
            None,
        );

        assert_eq!(prompt.chat_template.source, ChatTemplateSource::Werk);
        assert_eq!(prompt.chat_template.name, "phi3");
        assert!(prompt.chat_template.applied_by_werk);
        assert!(prompt.prompt.starts_with("<|user|>"));
        assert!(prompt.stop.contains(&"<|end|>".to_string()));
    }

    #[test]
    fn transformers_compat_uses_model_chat_template_and_messages() {
        let manifest = test_manifest(ModelFormat::SafeTensors, Some("chatglm"));
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text("hello".to_string())),
            name: None,
        }];
        let prompt = prompt_for_backend(
            &manifest,
            &messages,
            BackendChoice::TransformersCompat,
            None,
        );
        let request_messages = generation_request_messages(&prompt, &messages);

        assert_eq!(prompt.chat_template.source, ChatTemplateSource::Model);
        assert_eq!(prompt.chat_template.name, "model");
        assert!(!prompt.chat_template.applied_by_werk);
        assert_eq!(request_messages.len(), 1);
        assert_eq!(prompt.prompt, "hello");
    }

    #[test]
    fn chat_request_messages_keep_history_by_default() {
        let mut history = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text("first".to_string())),
            name: None,
        }];
        let request_messages = request_messages_for_turn(
            &mut history,
            ChatMessage {
                role: "user".to_string(),
                content: Some(MessageContent::Text("second".to_string())),
                name: None,
            },
            true,
        );

        assert_eq!(request_messages.len(), 2);
        assert_eq!(history.len(), 2);
        assert_eq!(
            request_messages[1]
                .content
                .as_ref()
                .map(MessageContent::as_text),
            Some("second".to_string())
        );
    }

    #[test]
    fn single_turn_request_messages_only_include_current_user_message() {
        let mut history = vec![
            ChatMessage {
                role: "user".to_string(),
                content: Some(MessageContent::Text("first".to_string())),
                name: None,
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: Some(MessageContent::Text("answer".to_string())),
                name: None,
            },
        ];
        let request_messages = request_messages_for_turn(
            &mut history,
            ChatMessage {
                role: "user".to_string(),
                content: Some(MessageContent::Text("second".to_string())),
                name: None,
            },
            false,
        );

        assert_eq!(request_messages.len(), 1);
        assert_eq!(history.len(), 2);
        assert_eq!(
            request_messages[0]
                .content
                .as_ref()
                .map(MessageContent::as_text),
            Some("second".to_string())
        );
    }

    #[test]
    fn estimate_small_model_fits() {
        let store = test_store("estimate-small");
        let manifest = test_manifest_with_weight(
            ModelFormat::SafeTensors,
            Some("llama"),
            "files/model.safetensors",
            2 * GIB,
        );
        let report = estimate_model_memory(
            &store,
            &manifest,
            SystemMemory {
                total_bytes: Some(32 * GIB),
                available_bytes: Some(16 * GIB),
            },
        );

        assert_eq!(report.result, EstimateResult::Ok);
    }

    #[test]
    fn estimate_huggingface_repo_id_detection_is_conservative() {
        assert!(looks_like_huggingface_repo_id("tiiuae/Falcon3-7B-Instruct"));
        assert!(looks_like_huggingface_repo_id("org/repo"));
        assert!(!looks_like_huggingface_repo_id("local-model"));
        assert!(!looks_like_huggingface_repo_id("/tmp/model"));
        assert!(!looks_like_huggingface_repo_id(
            "https://huggingface.co/org/repo"
        ));
    }

    #[test]
    fn estimate_missing_plain_model_keeps_pull_hint() {
        let store = test_store("estimate-plain-missing");
        let err = estimate_model_or_huggingface(
            &store,
            "local-missing",
            None,
            SystemMemory {
                total_bytes: None,
                available_bytes: None,
            },
        )
        .unwrap_err()
        .to_string();

        assert_eq!(
            err,
            "model 'local-missing' is not installed; run `werk pull local-missing` first"
        );
    }

    #[test]
    fn estimate_near_limit_warns() {
        let store = test_store("estimate-warning");
        let manifest = test_manifest_with_weight(
            ModelFormat::SafeTensors,
            Some("llama"),
            "files/model.safetensors",
            5 * GIB,
        );
        let report = estimate_model_memory(
            &store,
            &manifest,
            SystemMemory {
                total_bytes: Some(16 * GIB),
                available_bytes: Some(10 * GIB),
            },
        );

        assert_eq!(report.result, EstimateResult::Warning);
    }

    #[test]
    fn estimate_above_limit_is_likely_oom() {
        let store = test_store("estimate-oom");
        let manifest = test_manifest_with_weight(
            ModelFormat::SafeTensors,
            Some("llama"),
            "files/model.safetensors",
            5 * GIB,
        );
        let report = estimate_model_memory(
            &store,
            &manifest,
            SystemMemory {
                total_bytes: Some(16 * GIB),
                available_bytes: Some(8 * GIB),
            },
        );

        assert_eq!(report.result, EstimateResult::LikelyOom);
    }

    #[test]
    fn estimate_memory_heavy_architectures_use_higher_kv_cache() {
        let normal = kv_cache_fallback_bytes(10 * GIB, Some("llama"));
        for architecture in ["jamba", "mamba", "mixtral", "deepseek-moe"] {
            assert!(
                kv_cache_fallback_bytes(10 * GIB, Some(architecture)) > normal,
                "{architecture} should use the memory-heavy KV estimate"
            );
        }
    }

    #[test]
    fn estimate_unknown_available_memory_falls_back_to_total_thresholds() {
        let store = test_store("estimate-total-fallback");
        let manifest = test_manifest_with_weight(
            ModelFormat::SafeTensors,
            Some("llama"),
            "files/model.safetensors",
            6 * GIB,
        );
        let report = estimate_model_memory(
            &store,
            &manifest,
            SystemMemory {
                total_bytes: Some(16 * GIB),
                available_bytes: None,
            },
        );

        assert_eq!(report.result, EstimateResult::Warning);
    }

    #[test]
    fn estimate_format_bytes_uses_gib_style_output() {
        assert_eq!(format_bytes(GIB), "1.00 GiB");
    }

    #[test]
    fn estimate_gguf_counts_selected_model_file_only() {
        let mut manifest = test_manifest(ModelFormat::Gguf, Some("llama"));
        manifest.model_path = Some("files/model.Q4_K_M.gguf".to_string());
        manifest.files = vec![
            model_file("files/model.Q4_K_M.gguf", 4 * GIB),
            model_file("files/model.Q8_0.gguf", 8 * GIB),
        ];

        assert_eq!(estimate_model_files_bytes(&manifest), 4 * GIB);
    }

    #[test]
    fn estimate_weight_filtering_ignores_metadata_files() {
        let mut manifest = test_manifest(ModelFormat::SafeTensors, Some("llama"));
        manifest.model_path = Some("files/model.safetensors".to_string());
        manifest.files = vec![
            model_file("files/model.safetensors", 3 * GIB),
            model_file("files/tokenizer.json", 1_000),
            model_file("files/tokenizer_config.json", 1_000),
            model_file("files/special_tokens_map.json", 1_000),
            model_file("files/generation_config.json", 1_000),
            model_file("files/config.json", 1_000),
            model_file("files/README.md", 1_000),
            model_file("files/LICENSE", 1_000),
            model_file("files/merges.txt", 1_000),
            model_file("files/vocab.json", 1_000),
            model_file("files/chat_template.jinja", 1_000),
        ];

        let accounting = estimate_weight_accounting_without_store(&manifest);

        assert_eq!(accounting.total_bytes(), 3 * GIB);
        assert_eq!(accounting.counted.len(), 1);
        assert!(
            accounting
                .ignored
                .iter()
                .any(|file| file.path.ends_with("tokenizer.json"))
        );
        assert!(
            accounting
                .ignored
                .iter()
                .any(|file| file.path.ends_with("README.md"))
        );
    }

    #[test]
    fn estimate_safetensors_index_counts_referenced_shards() {
        let store = test_store("estimate-safetensors-index");
        let mut manifest = test_manifest(ModelFormat::SafeTensors, Some("llama"));
        manifest.model_path = Some("files/model.safetensors".to_string());
        manifest.files = vec![
            model_file("files/model.safetensors.index.json", 128),
            model_file("files/model-00001-of-00002.safetensors", 2 * GIB),
            model_file("files/model-00002-of-00002.safetensors", 2 * GIB),
            model_file("files/unreferenced.safetensors", 10 * GIB),
            model_file("files/tokenizer.json", 256),
        ];
        write_store_file(
            &store,
            &manifest,
            "files/model.safetensors.index.json",
            r#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#,
        );

        let accounting = estimate_weight_accounting(&store, &manifest);

        assert_eq!(accounting.total_bytes(), 4 * GIB);
        assert_eq!(accounting.confidence, EstimateConfidence::High);
        assert!(
            accounting
                .ignored
                .iter()
                .any(|file| file.path == "files/unreferenced.safetensors")
        );
    }

    #[test]
    fn estimate_kv_cache_formula_computes_expected_bytes() {
        let config = EstimateConfig {
            hidden_size: Some(2048),
            num_hidden_layers: Some(24),
            num_attention_heads: Some(32),
            num_key_value_heads: Some(8),
            head_dim: None,
            max_position_embeddings: Some(8192),
            dtype: Some("bfloat16".to_string()),
            ..EstimateConfig::default()
        };

        let estimate = kv_cache_estimate(4 * GIB, Some("llama"), &Some(config));

        assert_eq!(estimate.bytes, 24 * 8 * 64 * 2 * 4096 * 2);
        assert_eq!(estimate.confidence, EstimateConfidence::High);
        assert!(estimate.config_used);
    }

    #[test]
    fn estimate_kv_cache_formula_with_defaults_is_medium_confidence() {
        let config = EstimateConfig {
            hidden_size: Some(2048),
            num_hidden_layers: Some(24),
            num_attention_heads: Some(32),
            ..EstimateConfig::default()
        };

        let estimate = kv_cache_estimate(4 * GIB, Some("llama"), &Some(config));

        assert_eq!(estimate.confidence, EstimateConfidence::Medium);
        assert!(estimate.config_used);
    }

    #[test]
    fn estimate_fallback_heuristic_marks_confidence_low() {
        let estimate = kv_cache_estimate(4 * GIB, Some("llama"), &None);

        assert_eq!(estimate.bytes, scale_bytes(4 * GIB, 0.35));
        assert_eq!(estimate.confidence, EstimateConfidence::Low);
        assert!(!estimate.config_used);
    }

    #[test]
    fn estimate_memory_heavy_config_keeps_low_confidence() {
        let config = EstimateConfig {
            hidden_size: Some(4096),
            num_hidden_layers: Some(32),
            num_attention_heads: Some(32),
            num_key_value_heads: Some(8),
            max_position_embeddings: Some(4096),
            dtype: Some("bfloat16".to_string()),
            model_type: Some("jamba".to_string()),
            ..EstimateConfig::default()
        };

        let estimate = kv_cache_estimate(10 * GIB, Some("jamba"), &Some(config));

        assert_eq!(estimate.bytes, scale_bytes(10 * GIB, 0.60));
        assert_eq!(estimate.confidence, EstimateConfidence::Low);
        assert!(!estimate.config_used);
    }

    #[test]
    fn estimate_smol_lm_like_config_uses_formula_not_weight_fraction() {
        let store = test_store("estimate-smollm");
        let mut manifest = test_manifest_with_weight(
            ModelFormat::SafeTensors,
            Some("llama"),
            "files/model.safetensors",
            3 * GIB,
        );
        manifest.config_path = Some("files/config.json".to_string());
        write_store_file(
            &store,
            &manifest,
            "files/config.json",
            r#"{
                "model_type": "llama",
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "max_position_embeddings": 8192,
                "torch_dtype": "bfloat16"
            }"#,
        );

        let report = estimate_model_memory(
            &store,
            &manifest,
            SystemMemory {
                total_bytes: Some(48 * GIB),
                available_bytes: Some(32 * GIB),
            },
        );

        assert!(report.config_used);
        assert_eq!(report.confidence, EstimateConfidence::High);
        assert!(report.kv_cache_bytes < scale_bytes(3 * GIB, 0.35));
        assert!(report.estimated_total_bytes < 5 * GIB);
    }

    #[test]
    fn estimate_output_formatting_ends_with_newline() {
        let store = test_store("estimate-output-newline");
        let manifest = test_manifest_with_weight(
            ModelFormat::Gguf,
            Some("llama"),
            "files/model.Q4_K_M.gguf",
            GIB,
        );
        let report = estimate_model_memory(
            &store,
            &manifest,
            SystemMemory {
                total_bytes: Some(16 * GIB),
                available_bytes: Some(8 * GIB),
            },
        );

        assert!(format_estimate_report(&report, true).ends_with('\n'));
    }

    #[test]
    fn remote_hf_metadata_parses_lfs_file_sizes() {
        let metadata = serde_json::json!({
            "siblings": [
                {"rfilename": "model.safetensors", "lfs": {"size": 1234}},
                {"rfilename": "tokenizer.json", "size": "5678"}
            ]
        });

        let files = parse_remote_hf_files(&metadata);

        assert_eq!(files.len(), 2);
        assert_eq!(files[0].path, "model.safetensors");
        assert_eq!(files[0].size, 1234);
        assert_eq!(files[1].path, "tokenizer.json");
        assert_eq!(files[1].size, 5678);
    }

    #[test]
    fn remote_hf_manifest_prefers_balanced_gguf_quant() {
        let remote = remote_hf_test_model(
            "unsloth/Tiny-GGUF",
            Some(serde_json::json!({"model_type": "llama"})),
            &[
                ("tiny.Q2_K.gguf", 2 * GIB),
                ("tiny.Q4_K_M.gguf", 4 * GIB),
                ("tiny.Q8_0.gguf", 8 * GIB),
                ("tokenizer.json", 1024),
            ],
        );

        let manifest = remote_hf_manifest(&remote, None).unwrap();

        assert_eq!(manifest.format, ModelFormat::Gguf);
        assert_eq!(manifest.architecture.as_deref(), Some("llama"));
        assert_eq!(
            manifest.model_path.as_deref(),
            Some("files/tiny.Q4_K_M.gguf")
        );
        assert!(matches!(
            manifest.source,
            ModelSource::HuggingFace { ref repo } if repo == "unsloth/Tiny-GGUF"
        ));
    }

    #[test]
    fn remote_hf_manifest_respects_explicit_file() {
        let remote = remote_hf_test_model(
            "unsloth/Tiny-GGUF",
            Some(serde_json::json!({"model_type": "llama"})),
            &[("tiny.Q4_K_M.gguf", 4 * GIB), ("tiny.Q8_0.gguf", 8 * GIB)],
        );

        let manifest = remote_hf_manifest(&remote, Some("files/tiny.Q8_0.gguf")).unwrap();

        assert_eq!(manifest.format, ModelFormat::Gguf);
        assert_eq!(manifest.model_path.as_deref(), Some("files/tiny.Q8_0.gguf"));
    }

    #[test]
    fn remote_hf_safetensors_index_counts_referenced_shards() {
        let remote = remote_hf_test_model(
            "org/sharded",
            Some(serde_json::json!({"model_type": "llama"})),
            &[
                ("model.safetensors.index.json", 128),
                ("model-00001-of-00002.safetensors", 2 * GIB),
                ("model-00002-of-00002.safetensors", 2 * GIB),
                ("unreferenced.safetensors", 10 * GIB),
                ("tokenizer.json", 1024),
            ],
        );
        let manifest = remote_hf_manifest(&remote, None).unwrap();
        let index = serde_json::json!({
            "weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors"
            }
        });

        let accounting = safetensors_index_weight_accounting_from_value(
            &manifest,
            "files/model.safetensors.index.json",
            &index,
        )
        .unwrap();

        assert_eq!(accounting.total_bytes(), 4 * GIB);
        assert_eq!(accounting.confidence, EstimateConfidence::High);
        assert!(
            accounting
                .ignored
                .iter()
                .any(|file| file.path == "files/unreferenced.safetensors")
        );
    }

    #[test]
    fn estimate_source_url_reports_huggingface_source() {
        let mut manifest = test_manifest(ModelFormat::SafeTensors, Some("llama"));
        manifest.source = ModelSource::HuggingFace {
            repo: "org/model".to_string(),
        };

        assert_eq!(
            estimate_source_url(&manifest).as_deref(),
            Some("https://huggingface.co/org/model")
        );
    }

    fn test_store(name: &str) -> ModelStore {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "werk1112-cli-{name}-{}-{nanos}",
            std::process::id()
        ));
        ModelStore::resolve(Some(root)).unwrap()
    }

    fn test_manifest(format: ModelFormat, architecture: Option<&str>) -> ModelManifest {
        ModelManifest {
            id: "test-model".to_string(),
            source: ModelSource::LocalPath {
                path: "test".to_string(),
            },
            format,
            architecture: architecture.map(str::to_string),
            tokenizer_path: None,
            config_path: None,
            model_path: Some("files/model.bin".to_string()),
            backend: "test".to_string(),
            created_unix: 1,
            files: Vec::new(),
            artifacts: Vec::new(),
            metadata: Default::default(),
        }
    }

    fn write_store_file(store: &ModelStore, manifest: &ModelManifest, path: &str, data: &str) {
        let path = store.model_dir(&manifest.id).join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, data).unwrap();
    }

    fn test_manifest_with_weight(
        format: ModelFormat,
        architecture: Option<&str>,
        path: &str,
        size: u64,
    ) -> ModelManifest {
        let mut manifest = test_manifest(format, architecture);
        manifest.model_path = Some(path.to_string());
        manifest.files = vec![model_file(path, size)];
        manifest
    }

    fn remote_hf_test_model(
        repo: &str,
        config: Option<Value>,
        files: &[(&str, u64)],
    ) -> RemoteHfModel {
        RemoteHfModel {
            repo: repo.to_string(),
            config,
            files: files
                .iter()
                .map(|(path, size)| RemoteHfFile {
                    path: (*path).to_string(),
                    size: *size,
                })
                .collect(),
            gated: false,
        }
    }

    fn model_file(path: &str, size: u64) -> ModelFile {
        ModelFile {
            path: path.to_string(),
            size,
            checksum: "crc32:00000000".to_string(),
        }
    }

    fn install_fake_managed_llama_server(store: &ModelStore, mode: LlamaCppMode) {
        let path = managed_backend_dir(store, mode).join("llama-server");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"#!/bin/sh\nexit 0\n").unwrap();
        make_executable(&path);
    }

    fn make_executable(path: &std::path::Path) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(path).unwrap().permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(path, permissions).unwrap();
        }
    }
}
