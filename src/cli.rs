use anyhow::{Result, anyhow, bail};
use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;
use std::{
    collections::HashMap,
    io::{self, IsTerminal, Write},
    net::{IpAddr, SocketAddr},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio_stream::StreamExt;

use crate::{
    api::{ApiState, serve},
    backend::{
        BackendAccelerator, BackendRuntime, BurnBackend, BurnMode, CandleBackend, CandleDeviceMode,
        ChatGenerationSession, GenerateRequest, GenerateStreamEvent, GenerationBackend,
        GenerationTimings, LlamaCppBackend, LlamaCppMode, LlamaFastBackend, LlamaFastRuntimeReport,
        LlamaKvCacheType, LlamaRuntimeOptions, LlamaServerBackend, LlamaServerDiscovery,
        MlxBackend, OnnxProvisionOptions, OnnxRuntimeAvailability, OnnxRuntimeBackend,
        OnnxRuntimeMode, RuntimeId, StreamGranularity, VllmBackend, backend_doctor_checks,
        backend_supports_accelerator, backend_supports_format,
        backend_supports_images as runtime_supports_images, install_managed_llama_server,
        install_managed_onnx_runtime, install_managed_vllm, llama_server_help_ok,
        managed_backend_dir, managed_burn_runner_path,
        managed_runner_path as managed_onnx_runner_path, managed_vllm_dir, probe_device,
        runtime_descriptor, runtime_registry, runtime_supports_model, vllm_doctor_checks,
    },
    banner::print_banner,
    model_store::{
        ArtifactStatus, ModelArtifact, ModelFormat, ModelManifest, ModelStore, PullProgress,
    },
    openai::{ChatMessage, MessageContent, messages_to_prompt_for_model},
    runtime_planner::{
        RequestCapabilities, RequestedBackend, RuntimeAvailability, RuntimeDecisionStatus,
        plan_runtime, runtime_candidate_ids, select_runtime,
    },
};

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
        help = "Backend for this process: auto, cpu, cuda, vulkan, metal, mlx, burn, onnx, vllm, candle, llama-highlevel, or llama-legacy"
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum StreamGranularityArg {
    Token,
    Chunk,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BackendArg {
    Auto,
    Burn,
    Candle,
    Cpu,
    Cuda,
    LlamaHighlevel,
    LlamaLegacy,
    Metal,
    Mlx,
    Onnx,
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
    pub batch_size: Option<usize>,

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
            batch_size: self.batch_size,
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
    LlamaVulkan,
    LlamaCpu,
    OnnxCuda,
    OnnxCpu,
    #[value(name = "vllm")]
    Vllm,
}

impl BackendInstallArg {
    fn mode(self) -> Option<LlamaCppMode> {
        match self {
            Self::LlamaCuda => Some(LlamaCppMode::Cuda),
            Self::LlamaVulkan => Some(LlamaCppMode::Vulkan),
            Self::LlamaCpu => Some(LlamaCppMode::Cpu),
            Self::OnnxCuda | Self::OnnxCpu | Self::Vllm => None,
        }
    }

    fn onnx_mode(self) -> Option<OnnxRuntimeMode> {
        match self {
            Self::OnnxCuda => Some(OnnxRuntimeMode::Cuda),
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
    },

    #[command(about = "Run one prompt against an installed model and print the response")]
    Run {
        #[arg(help = "Installed model id")]
        model: String,

        #[arg(required = true, num_args = 1.., help = "Prompt text")]
        prompt: Vec<String>,

        #[arg(long, default_value_t = 128, help = "Maximum generated tokens")]
        max_tokens: usize,

        #[arg(long, help = "Sampling temperature")]
        temperature: Option<f64>,

        #[arg(long, help = "Nucleus sampling top-p")]
        top_p: Option<f64>,

        #[arg(long, help = "RNG seed")]
        seed: Option<u64>,

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
            default_value_t = 256,
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
        command: DoctorCommands,
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
    List,

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
    });

    if should_print_startup_banner(&command) {
        print_banner();
    }

    match command {
        Commands::Serve { host, port, model } => {
            let store = ModelStore::resolve(model_home)?;
            store.ensure()?;
            let backend_choice = resolve_backend(backend_override, device_override)?;
            let ip: IpAddr = host.parse()?;
            let addr = SocketAddr::new(ip, port);
            let backend = build_generation_backend(
                store.clone(),
                backend_choice,
                llama_options.clone(),
                selection_options,
            )?;
            if let Some(model) = model.as_deref() {
                let manifest = store.get(model)?;
                backend.prepare(&manifest)?;
                println!("Default model available: {model}");
            }
            serve(
                addr,
                ApiState::new_with_default_model(store, backend, model),
            )
            .await
        }
        Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            top_p,
            seed,
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
            let prompt = messages_to_prompt_for_model(&manifest, &messages);
            let request = GenerateRequest {
                prompt: prompt.prompt,
                messages,
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
            let response = backend.generate(&manifest, request)?;
            println!("{}", response.text.trim());
            if verbose {
                let mut stderr = io::stderr().lock();
                writeln!(stderr)?;
                write_verbose_stats(
                    &mut stderr,
                    Some(verbose_backend_label(selected_backend)),
                    response.prompt_tokens,
                    response.completion_tokens,
                    response.timings,
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
                verbose_backend_label(selected_backend),
                max_tokens,
                temperature,
                top_p,
                seed,
                images,
                stream_granularity.into(),
                verbose,
                debug,
            )
            .await
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
        Commands::Doctor { command } => match command {
            DoctorCommands::Perf { model } => {
                let store = ModelStore::resolve(model_home)?;
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
        },
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
        Commands::List => {
            let store = ModelStore::resolve(model_home)?;
            let manifests = store.list()?;
            if manifests.is_empty() {
                println!("No models installed in {}", store.home().display());
            } else {
                println!(
                    "{:<32} {:<14} {:<18} PATH",
                    "MODEL", "FORMAT", "ARCHITECTURE"
                );
                for manifest in manifests {
                    println!(
                        "{:<32} {:<14} {:<18} {}",
                        manifest.id,
                        format!("{:?}", manifest.format).to_lowercase(),
                        manifest.architecture.unwrap_or_else(|| "-".to_string()),
                        store.model_dir(&manifest.id).display()
                    );
                }
            }
            Ok(())
        }
        Commands::Inspect { id } => {
            let store = ModelStore::resolve(model_home)?;
            let manifest = store.get(&id)?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
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

fn should_print_startup_banner(command: &Commands) -> bool {
    should_print_startup_banner_for(
        command,
        io::stdout().is_terminal(),
        io::stdin().is_terminal(),
    )
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
        | Commands::Bench { .. }
        | Commands::Doctor { .. }
        | Commands::Backend { .. }
        | Commands::Artifacts { .. }
        | Commands::List
        | Commands::Inspect { .. }
        | Commands::SelectFile { .. } => false,
    }
}

async fn chat_loop(
    backend: Arc<dyn GenerationBackend>,
    manifest: ModelManifest,
    backend_name: &'static str,
    max_tokens: usize,
    temperature: Option<f64>,
    top_p: Option<f64>,
    seed: Option<u64>,
    images: Vec<String>,
    stream_granularity: StreamGranularity,
    verbose: bool,
    debug: bool,
) -> Result<()> {
    let chat_session = prepare_backend_for_chat(backend.as_ref(), &manifest, seed)?;

    println!(
        "Chatting with {}. Type /exit or /quit to stop.",
        manifest.id
    );
    let mut messages = Vec::new();
    let stdin = io::stdin();

    loop {
        print!("you> ");
        io::stdout().flush()?;

        let mut input = String::new();
        let n = stdin.read_line(&mut input)?;
        if n == 0 {
            println!();
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if matches!(input, "/exit" | "/quit") {
            break;
        }

        messages.push(ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text(input.to_string())),
            name: None,
        });

        let prompt = messages_to_prompt_for_model(&manifest, &messages);
        let request = GenerateRequest {
            prompt: prompt.prompt,
            messages: messages.clone(),
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
        let mut timings = None;
        let mut last_flush = Instant::now();
        let mut stream = if let Some(session) = chat_session.as_ref() {
            session.generate_stream(request)
        } else {
            backend.generate_stream(manifest.clone(), request)
        };
        while let Some(event) = stream.next().await {
            match event {
                Ok(GenerateStreamEvent::TextChunk(chunk)) => {
                    print!("{chunk}");
                    if chunk.contains('\n') || last_flush.elapsed() >= Duration::from_millis(16) {
                        io::stdout().flush()?;
                        last_flush = Instant::now();
                    }
                    assistant.push_str(&chunk);
                }
                Ok(GenerateStreamEvent::Done {
                    prompt_tokens: tokens_in,
                    completion_tokens: tokens,
                    timings: response_timings,
                    ..
                }) => {
                    prompt_tokens = tokens_in;
                    completion_tokens = tokens;
                    timings = Some(response_timings);
                    break;
                }
                Err(message) => {
                    println!("\nerror: {message}");
                    break;
                }
            }
        }
        io::stdout().flush()?;
        println!();
        if verbose && let Some(timings) = timings {
            let mut stdout = io::stdout().lock();
            writeln!(stdout)?;
            write_verbose_stats(
                &mut stdout,
                Some(backend_name),
                prompt_tokens,
                completion_tokens,
                timings,
            )?;
        }

        if !assistant.trim().is_empty() {
            messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: Some(MessageContent::Text(assistant)),
                name: None,
            });
        }
    }

    Ok(())
}

fn prepare_backend_for_chat(
    backend: &dyn GenerationBackend,
    manifest: &ModelManifest,
    seed: Option<u64>,
) -> Result<Option<Box<dyn ChatGenerationSession>>> {
    backend.prepare(manifest)?;
    backend.start_chat_session(manifest, seed)
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
    for mode in [LlamaCppMode::Cuda, LlamaCppMode::Vulkan, LlamaCppMode::Cpu] {
        let discovery = LlamaServerBackend::discover(store, mode);
        print_backend_discovery(&discovery);
    }

    println!();
    println!("Burn discovery");
    println!(
        "{:<8} {:<16} {:<7} {:<7} PATH",
        "BACKEND", "SOURCE", "EXISTS", "HELP"
    );
    for mode in [BurnMode::Cuda, BurnMode::Cpu] {
        print_burn_discovery(store, mode);
    }

    println!();
    println!("ONNX Runtime discovery");
    println!(
        "{:<8} {:<16} {:<7} {:<7} PATH",
        "BACKEND", "SOURCE", "EXISTS", "HELP"
    );
    for mode in [OnnxRuntimeMode::Cuda, OnnxRuntimeMode::Cpu] {
        print_onnxruntime_discovery(store, mode);
    }

    println!();
    println!("vLLM discovery");
    let discovery = VllmBackend::discover(store);
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
        "{:<8} {:<16} {:<7} {}",
        "BACKEND", "SOURCE", "EXISTS", "PATH"
    );
    println!(
        "{:<8} {:<16} {:<7} {}",
        "vLLM",
        discovery.source,
        yes_no(discovery.command.is_some()),
        vllm_path
    );

    println!();
    println!(
        "{:<24} {:<12} {:<12} {:<8} INSTALL",
        "RUNTIME", "STATE", "ACCEL", "VLM"
    );
    for runtime in runtime_registry() {
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

fn print_burn_discovery(store: &ModelStore, mode: BurnMode) {
    let discovery = BurnBackend::discover(store, mode);
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
        .unwrap_or_else(|| managed_burn_runner_path(store, mode).display().to_string());
    let help = discovery
        .attempts
        .iter()
        .find(|attempt| attempt.usable)
        .map(|attempt| attempt.usable)
        .unwrap_or(false);
    println!(
        "{:<8} {:<16} {:<7} {:<7} {}",
        match mode {
            BurnMode::Cuda => "CUDA",
            BurnMode::Cpu => "CPU",
        },
        discovery.source,
        yes_no(discovery.path.is_some()),
        yes_no(help),
        path
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
    println!("managed cache: {}", store.home().join("backends").display());
    println!(
        "CUDA cache: {}",
        managed_backend_dir(store, LlamaCppMode::Cuda).display()
    );
    println!();
    for check in backend_doctor_checks(store) {
        println!(
            "{:<24} {:<7} {}",
            check.name,
            if check.ok { "ok" } else { "missing" },
            check.detail
        );
    }
    for check in vllm_doctor_checks(store) {
        println!(
            "{:<24} {:<7} {}",
            check.name,
            if check.ok { "ok" } else { "missing" },
            check.detail
        );
    }
    println!();
    println!("{:<24} {:<12} DETAIL", "RUNTIME", "STATUS");
    for mode in [BurnMode::Cuda, BurnMode::Cpu] {
        let discovery = BurnBackend::discover(store, mode);
        let status = if discovery.path.is_some() {
            "ready"
        } else {
            "unavailable"
        };
        let detail = discovery
            .path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| BurnBackend::unavailable_reason(store, mode));
        println!("{:<24} {:<12} {}", mode.display(), status, detail);
        if debug {
            print_burn_debug_details(store, mode);
        }
    }
    for mode in [OnnxRuntimeMode::Cuda, OnnxRuntimeMode::Cpu] {
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
    if values.len() % 2 == 0 {
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
    timings: GenerationTimings,
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
        format_duration(timings.prompt_seconds)
    )?;
    writeln!(
        writer,
        "{:<22}{:.2} tokens/s",
        "prompt eval rate:",
        rate(prompt_tokens, timings.prompt_seconds)
    )?;
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
    )
}

fn rate(tokens: usize, seconds: f64) -> f64 {
    if seconds <= 0.0 {
        0.0
    } else {
        tokens as f64 / seconds
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
    OnnxRuntime(OnnxRuntimeMode),
    Vllm,
}

#[derive(Debug, Clone, Copy)]
struct SelectionOptions {
    provision_missing_backends: bool,
}

impl SelectionOptions {
    fn from_cli(backend: BackendArg, auto_install: bool, no_auto_install: bool) -> Self {
        let default_provision = matches!(backend, BackendArg::Auto);
        Self {
            provision_missing_backends: !no_auto_install && (auto_install || default_provision),
        }
    }
}

impl Default for SelectionOptions {
    fn default() -> Self {
        Self {
            provision_missing_backends: false,
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
        BackendArg::Metal => BackendChoice::Candle(CandleDeviceMode::Metal),
        BackendArg::Mlx => BackendChoice::Mlx,
        BackendArg::Onnx => BackendChoice::OnnxRuntime(preferred_onnx_mode()),
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
    if cfg!(feature = "cuda") {
        LlamaCppMode::Cuda
    } else if cfg!(feature = "vulkan") {
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
    if cfg!(any(windows, target_os = "linux")) {
        BurnMode::Cuda
    } else {
        BurnMode::Cpu
    }
}

fn burn_mode_for_request(requested: RequestedBackend) -> BurnMode {
    match requested {
        RequestedBackend::Cpu => BurnMode::Cpu,
        RequestedBackend::Cuda => BurnMode::Cuda,
        _ => preferred_burn_mode(),
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
        BackendChoice::OnnxRuntime(mode) => Ok(Arc::new(OnnxRuntimeBackend::new(store, mode))),
        BackendChoice::Vllm => Ok(Arc::new(VllmBackend::new(store))),
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
        RuntimeId::Burn => Some(BackendChoice::Burn(preferred_burn_mode())),
        RuntimeId::LlamaServerCuda => Some(BackendChoice::LlamaServer(LlamaCppMode::Cuda)),
        RuntimeId::LlamaServerVulkan => Some(BackendChoice::LlamaServer(LlamaCppMode::Vulkan)),
        RuntimeId::LlamaServerCpu => Some(BackendChoice::LlamaServer(LlamaCppMode::Cpu)),
        RuntimeId::CandleCuda => Some(BackendChoice::Candle(CandleDeviceMode::Cuda)),
        RuntimeId::CandleMetal => Some(BackendChoice::Candle(CandleDeviceMode::Metal)),
        RuntimeId::CandleCpu => Some(BackendChoice::Candle(CandleDeviceMode::Cpu)),
        RuntimeId::Mlx => Some(BackendChoice::Mlx),
        RuntimeId::OnnxRuntimeCuda => Some(BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cuda)),
        RuntimeId::OnnxRuntimeCpu => Some(BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cpu)),
        RuntimeId::VllmCuda => Some(BackendChoice::Vllm),
    }
}

fn runtime_id_to_backend_for_request(
    id: RuntimeId,
    requested: RequestedBackend,
) -> Option<BackendChoice> {
    match id {
        RuntimeId::Burn => Some(BackendChoice::Burn(burn_mode_for_request(requested))),
        _ => runtime_id_to_backend(id),
    }
}

fn backend_to_runtime_id(backend: BackendChoice) -> Option<RuntimeId> {
    match backend {
        BackendChoice::Burn(_) => Some(RuntimeId::Burn),
        BackendChoice::LlamaServer(LlamaCppMode::Cuda) => Some(RuntimeId::LlamaServerCuda),
        BackendChoice::LlamaServer(LlamaCppMode::Vulkan) => Some(RuntimeId::LlamaServerVulkan),
        BackendChoice::LlamaServer(LlamaCppMode::Cpu) => Some(RuntimeId::LlamaServerCpu),
        BackendChoice::Candle(CandleDeviceMode::Cuda) => Some(RuntimeId::CandleCuda),
        BackendChoice::Candle(CandleDeviceMode::Metal) => Some(RuntimeId::CandleMetal),
        BackendChoice::Candle(CandleDeviceMode::Cpu)
        | BackendChoice::Candle(CandleDeviceMode::Auto) => Some(RuntimeId::CandleCpu),
        BackendChoice::Mlx => Some(RuntimeId::Mlx),
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cuda) => Some(RuntimeId::OnnxRuntimeCuda),
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cpu) => Some(RuntimeId::OnnxRuntimeCpu),
        BackendChoice::Vllm => Some(RuntimeId::VllmCuda),
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
        BackendChoice::OnnxRuntime(_) => BackendRuntime::OnnxRuntime,
        BackendChoice::Vllm => BackendRuntime::Vllm,
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
        BackendChoice::LlamaServer(LlamaCppMode::Vulkan)
        | BackendChoice::LlamaFast(LlamaCppMode::Vulkan)
        | BackendChoice::LlamaHighlevel(LlamaCppMode::Vulkan) => BackendAccelerator::Vulkan,
        BackendChoice::Candle(CandleDeviceMode::Metal) => BackendAccelerator::Metal,
        BackendChoice::Mlx => BackendAccelerator::Mlx,
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cuda) => BackendAccelerator::Cuda,
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cpu) => BackendAccelerator::Cpu,
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
        | BackendChoice::Candle(CandleDeviceMode::Auto)
        | BackendChoice::Candle(CandleDeviceMode::Cpu) => None,
        BackendChoice::Candle(mode) => probe_device(mode).err().map(|_| match mode {
            CandleDeviceMode::Cuda => candle_cuda_rejection_reason(),
            CandleDeviceMode::Metal => "Candle Metal is unavailable".to_string(),
            CandleDeviceMode::Auto | CandleDeviceMode::Cpu => "Candle is unavailable".to_string(),
        }),
        BackendChoice::Mlx => MlxBackend::probe()
            .err()
            .map(|_| "mlx-lm is unavailable".to_string()),
        BackendChoice::Burn(mode) => BurnBackend::probe(store, mode)
            .err()
            .map(|_| BurnBackend::unavailable_reason(store, mode)),
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
        BackendChoice::LlamaServer(mode) => LlamaServerBackend::probe(store, mode)
            .err()
            .map(|_| LlamaServerBackend::missing_message(store, mode)),
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
                selection_options,
            )?;
            ensure_backend_supports_images(selected, has_images)?;
            Ok(selected)
        }
        BackendChoice::LlamaServer(_)
        | BackendChoice::Burn(_)
        | BackendChoice::Mlx
        | BackendChoice::OnnxRuntime(_)
        | BackendChoice::Vllm => {
            let Some(runtime_id) = backend_to_runtime_id(backend) else {
                bail!(
                    "backend {} is not represented in the runtime planner",
                    backend_label(backend)
                );
            };
            let selected = select_backend_from_runtime_candidates(
                store,
                &[runtime_id],
                manifest,
                requested_backend_for_choice(backend),
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
    if manifest.format != ModelFormat::SafeTensors
        || !matches!(
            requested_choice,
            BackendChoice::Auto | BackendChoice::GgufPreferred { .. }
        )
        || !matches!(selected, BackendChoice::Candle(_))
    {
        return None;
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
    let burn_rejection = plan.candidates.iter().find(|decision| {
        decision.runtime_id == RuntimeId::Burn && decision.status == RuntimeDecisionStatus::Rejected
    })?;
    Some(format!(
        "note: {} unavailable; using {} fallback.",
        burn_rejection.display_name,
        verbose_backend_label(selected)
    ))
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
    runtime_candidate_ids(manifest, requested)
        .into_iter()
        .map(|runtime_id| {
            if let Some(backend) = runtime_id_to_backend_for_request(runtime_id, requested) {
                let reason =
                    backend_unavailability_reason(store, backend, manifest, selection_options);
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
            llama: LlamaCppMode::Vulkan,
            ..
        }
        | BackendChoice::LlamaServer(LlamaCppMode::Vulkan) => RequestedBackend::Vulkan,
        BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Cpu,
            ..
        }
        | BackendChoice::LlamaServer(LlamaCppMode::Cpu) => RequestedBackend::Cpu,
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cpu) => RequestedBackend::Cpu,
        BackendChoice::Burn(_) => RequestedBackend::Burn,
        BackendChoice::Candle(_) => RequestedBackend::Candle,
        BackendChoice::Mlx => RequestedBackend::Mlx,
        BackendChoice::Vllm => RequestedBackend::Vllm,
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
            BurnBackend::missing_message(store, mode)
        }
        (BackendChoice::LlamaServer(mode), ModelFormat::Gguf) => {
            LlamaServerBackend::missing_message(store, mode)
        }
        (BackendChoice::OnnxRuntime(mode), ModelFormat::SafeTensors | ModelFormat::Onnx) => {
            OnnxRuntimeBackend::missing_message(store, mode)
        }
        (BackendChoice::Vllm, ModelFormat::SafeTensors) => VllmBackend::missing_message(store),
        (BackendChoice::Mlx, _) => "mlx-lm is unavailable".to_string(),
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
        let status = match decision.status {
            RuntimeDecisionStatus::Accepted => "accepted",
            RuntimeDecisionStatus::Rejected => "rejected",
        };
        eprintln!(
            "candidate: {} -> {status}: {}",
            decision.display_name, decision.reason
        );
        if decision.runtime_id == RuntimeId::Burn
            && decision.status == RuntimeDecisionStatus::Rejected
        {
            let mode = burn_mode_for_request(requested_backend);
            print_burn_debug_details(store, mode);
        }
        if matches!(
            decision.runtime_id,
            RuntimeId::OnnxRuntimeCuda | RuntimeId::OnnxRuntimeCpu
        ) && decision.status == RuntimeDecisionStatus::Rejected
        {
            let mode = match decision.runtime_id {
                RuntimeId::OnnxRuntimeCuda => OnnxRuntimeMode::Cuda,
                RuntimeId::OnnxRuntimeCpu => OnnxRuntimeMode::Cpu,
                _ => unreachable!(),
            };
            print_onnxruntime_debug_details(store, mode);
        }
    }
    eprintln!("selected runtime: {}", verbose_backend_label(selected));
    if let Some(selected) = plan.selected {
        eprintln!("reason: {}", selected.reason);
    }
}

fn print_burn_debug_details(store: &ModelStore, mode: BurnMode) {
    let discovery = BurnBackend::discover(store, mode);
    eprintln!("  status: unavailable");
    eprintln!("  reason: {}", BurnBackend::unavailable_reason(store, mode));
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
            llama: LlamaCppMode::Cpu,
            ..
        } => "cpu",
        BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Vulkan,
            ..
        } => "vulkan",
        BackendChoice::Candle(CandleDeviceMode::Auto) => "candle-auto",
        BackendChoice::Burn(BurnMode::Cuda) => "burn-cuda",
        BackendChoice::Burn(BurnMode::Cpu) => "burn-cpu",
        BackendChoice::Candle(CandleDeviceMode::Cpu) => "candle-cpu",
        BackendChoice::Candle(CandleDeviceMode::Cuda) => "candle-cuda",
        BackendChoice::Candle(CandleDeviceMode::Metal) => "metal",
        BackendChoice::Mlx => "mlx",
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cuda) => "onnxruntime-cuda",
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cpu) => "onnxruntime-cpu",
        BackendChoice::Vllm => "vllm-cuda",
        BackendChoice::LlamaServer(LlamaCppMode::Cuda) => "llama-server-cuda",
        BackendChoice::LlamaServer(LlamaCppMode::Vulkan) => "llama-server-vulkan",
        BackendChoice::LlamaServer(LlamaCppMode::Cpu) => "llama-server-cpu",
        BackendChoice::LlamaFast(LlamaCppMode::Cuda) => "llama-legacy-cuda",
        BackendChoice::LlamaFast(LlamaCppMode::Vulkan) => "llama-legacy-vulkan",
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
            llama: LlamaCppMode::Cpu,
            ..
        } => "llama.cpp server CPU",
        BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Vulkan,
            ..
        } => "llama.cpp server Vulkan",
        BackendChoice::Candle(CandleDeviceMode::Auto) => "Candle auto",
        BackendChoice::Burn(BurnMode::Cuda) => "Burn CUDA",
        BackendChoice::Burn(BurnMode::Cpu) => "Burn CPU",
        BackendChoice::Candle(CandleDeviceMode::Cpu) => "Candle CPU",
        BackendChoice::Candle(CandleDeviceMode::Cuda) => "Candle CUDA",
        BackendChoice::Candle(CandleDeviceMode::Metal) => "Candle Metal",
        BackendChoice::Mlx => "MLX",
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cuda) => "ONNX Runtime CUDA",
        BackendChoice::OnnxRuntime(OnnxRuntimeMode::Cpu) => "ONNX Runtime CPU",
        BackendChoice::Vllm => "vLLM CUDA",
        BackendChoice::LlamaServer(LlamaCppMode::Cuda) => "llama.cpp server CUDA",
        BackendChoice::LlamaServer(LlamaCppMode::Vulkan) => "llama.cpp server Vulkan",
        BackendChoice::LlamaServer(LlamaCppMode::Cpu) => "llama.cpp server CPU",
        BackendChoice::LlamaFast(LlamaCppMode::Cuda) => "llama.cpp legacy FFI CUDA",
        BackendChoice::LlamaFast(LlamaCppMode::Vulkan) => "llama.cpp legacy FFI Vulkan",
        BackendChoice::LlamaFast(LlamaCppMode::Cpu) => "llama.cpp legacy FFI CPU",
        BackendChoice::LlamaHighlevel(LlamaCppMode::Cuda) => "llama.cpp high-level CUDA",
        BackendChoice::LlamaHighlevel(LlamaCppMode::Vulkan) => "llama.cpp high-level Vulkan",
        BackendChoice::LlamaHighlevel(LlamaCppMode::Cpu) => "llama.cpp high-level CPU",
    }
}

fn display_llama_mode(mode: LlamaCppMode) -> &'static str {
    match mode {
        LlamaCppMode::Cuda => "CUDA",
        LlamaCppMode::Vulkan => "Vulkan",
        LlamaCppMode::Cpu => "CPU",
    }
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
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
        PullProgress::LfsStarted => {
            progress.set_position(0);
            progress.set_message("downloading");
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
                progress.set_message(format!(
                    "downloading {} / {} @ {}/s",
                    format_bytes(bytes.min(total_bytes)),
                    format_bytes(total_bytes),
                    format_bytes_per_second(bytes_per_second)
                ));
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
    use crate::model_store::ModelSource;
    use std::fs;
    use std::sync::{Arc as StdArc, Mutex as StdMutex};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parses_cli_commands() {
        let cli = Cli::try_parse_from([
            "werk", "--device", "cuda", "serve", "--host", "0.0.0.0", "--port", "8080", "--model",
            "m",
        ])
        .unwrap();
        assert_eq!(cli.device, Some(DeviceArg::Cuda));
        match cli.command.unwrap() {
            Commands::Serve { host, port, model } => {
                assert_eq!(host, "0.0.0.0");
                assert_eq!(port, 8080);
                assert_eq!(model.as_deref(), Some("m"));
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
                images,
                debug,
                ..
            } => {
                assert_eq!(model, "gemma-2b-it");
                assert_eq!(prompt, vec!["hello"]);
                assert_eq!(images, vec!["image.png"]);
                assert!(debug);
            }
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

        let cli = Cli::try_parse_from(["werk", "backend", "install", "vllm"]).unwrap();
        match cli.command.unwrap() {
            Commands::Backend {
                command: BackendCommands::Install { target },
            } => assert_eq!(target, BackendInstallArg::Vllm),
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
            Commands::Chat { model, .. } => assert_eq!(model, "tiny"),
            command => panic!("unexpected command: {command:?}"),
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
    fn auto_safetensors_prefers_burn_then_candle_on_linux_and_windows() {
        if cfg!(any(windows, target_os = "linux")) {
            let manifest = test_manifest(ModelFormat::SafeTensors, Some("phi3"));
            let order = auto_candidates_for_manifest(&manifest);
            assert!(matches!(order[0], BackendChoice::Burn(BurnMode::Cuda)));
            assert!(matches!(
                order[1],
                BackendChoice::Candle(CandleDeviceMode::Cuda)
            ));
        }
    }

    #[test]
    fn runtime_registry_exposes_real_burn_runtime() {
        let burn = runtime_descriptor(RuntimeId::Burn);
        assert_eq!(burn.display_name, "Burn");
        assert!(burn.implemented);
        assert_eq!(burn.install_target, None);
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
    fn safetensors_runtime_candidates_include_burn_before_candle_for_any_architecture() {
        if cfg!(any(windows, target_os = "linux")) {
            let manifest = test_manifest(ModelFormat::SafeTensors, Some("unknown"));
            let candidates = auto_runtime_candidates_for_manifest(&manifest);
            assert_eq!(candidates[0], RuntimeId::Burn);
            assert!(
                candidates
                    .iter()
                    .position(|id| *id == RuntimeId::CandleCuda)
                    .unwrap()
                    > candidates
                        .iter()
                        .position(|id| *id == RuntimeId::Burn)
                        .unwrap()
            );

            let concrete = auto_candidates_for_manifest(&manifest);
            assert!(matches!(concrete[0], BackendChoice::Burn(BurnMode::Cuda)));
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
    fn backend_selection_routes_safetensors_cuda_to_burn_when_available() {
        let store = test_store("safetensors-cuda");
        install_fake_managed_burn_runner(&store, BurnMode::Cuda);
        let manifest = test_manifest(ModelFormat::SafeTensors, Some("phi3"));
        let selected = selected_backend_for_manifest(
            &store,
            BackendChoice::GgufPreferred {
                llama: LlamaCppMode::Cuda,
                candle: CandleDeviceMode::Cuda,
            },
            &manifest,
        )
        .unwrap();
        assert!(matches!(selected, BackendChoice::Burn(BurnMode::Cuda)));
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
        assert!(!message.contains("Candle CUDA"));
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
    fn auto_safetensors_fallback_note_is_concise() {
        let store = test_store("auto-burn-fallback-note");
        let manifest = test_manifest(ModelFormat::SafeTensors, Some("phi3"));
        let note = verbose_fallback_note(
            &store,
            BackendChoice::Auto,
            &manifest,
            false,
            BackendChoice::Candle(CandleDeviceMode::Cuda),
        )
        .unwrap();
        assert_eq!(note, "note: Burn unavailable; using Candle CUDA fallback.");
        assert!(!note.contains(&store.home().display().to_string()));
    }

    #[test]
    fn auto_safetensors_can_fallback_to_candle_with_burn_note() {
        let store = test_store("auto-burn-fallback");
        let manifest = test_manifest(ModelFormat::SafeTensors, Some("phi3"));
        let selected =
            selected_backend_for_manifest(&store, BackendChoice::Auto, &manifest).unwrap();
        assert!(matches!(selected, BackendChoice::Candle(_)));
        let note =
            verbose_fallback_note(&store, BackendChoice::Auto, &manifest, false, selected).unwrap();
        assert!(note.starts_with("note: Burn unavailable;"));
        assert!(note.contains("using Candle"));
    }

    #[test]
    fn backend_selection_falls_back_to_candle_cuda_when_burn_missing() {
        let store = test_store("safetensors-cuda-fallback");
        let manifest = test_manifest(ModelFormat::SafeTensors, Some("phi3"));
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
                assert!(message.contains("Burn"));
                assert!(message.contains("Candle CUDA"));
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

        let safetensors = test_manifest(ModelFormat::SafeTensors, Some("phi3"));
        install_fake_managed_burn_runner(&store, BurnMode::Cuda);
        let result = selected_backend_for_manifest(
            &store,
            BackendChoice::GgufPreferred {
                llama: LlamaCppMode::Cuda,
                candle: CandleDeviceMode::Cuda,
            },
            &safetensors,
        );
        match result {
            Ok(selected) => assert!(matches!(selected, BackendChoice::Burn(BurnMode::Cuda))),
            Err(err) => {
                let message = err.to_string();
                assert!(message.contains("Burn"));
                assert!(message.contains("Candle CUDA"));
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
        let _ = prepare_backend_for_chat(&backend, &manifest, None).unwrap();
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
            &Commands::List,
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
        }
    }

    fn install_fake_managed_llama_server(store: &ModelStore, mode: LlamaCppMode) {
        let path = managed_backend_dir(store, mode).join("llama-server");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"#!/bin/sh\nexit 0\n").unwrap();
        make_executable(&path);
    }

    fn install_fake_managed_burn_runner(store: &ModelStore, mode: BurnMode) {
        let path = managed_burn_runner_path(store, mode);
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
