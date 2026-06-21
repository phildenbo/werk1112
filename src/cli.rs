use anyhow::{Result, anyhow, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
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
        CandleBackend, CandleDeviceMode, ChatGenerationSession, GenerateRequest,
        GenerateStreamEvent, GenerationBackend, GenerationTimings, LlamaCppBackend, LlamaCppMode,
        LlamaFastBackend, LlamaFastRuntimeReport, LlamaKvCacheType, LlamaRuntimeOptions,
        MlxBackend, StreamGranularity, probe_device,
    },
    banner::print_banner,
    model_store::{ModelFormat, ModelManifest, ModelStore, PullProgress},
    openai::{ChatMessage, MessageContent, messages_to_prompt_for_model},
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
        help = "Backend for this process: auto, cpu, cuda, llama-highlevel, metal, mlx, or vulkan"
    )]
    pub backend: BackendArg,

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
    Cpu,
    Cuda,
    LlamaHighlevel,
    Metal,
    Mlx,
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
    },

    #[command(about = "Inspect Werk runtime diagnostics")]
    Doctor {
        #[command(subcommand)]
        command: DoctorCommands,
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

pub async fn run_from_env() -> Result<()> {
    run(Cli::parse()).await
}

pub async fn run(cli: Cli) -> Result<()> {
    let model_home = cli.model_home;
    let device_override = cli.device;
    let backend_override = cli.backend;
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
            let backend =
                build_generation_backend(store.clone(), backend_choice, llama_options.clone())?;
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
        } => {
            let prompt = prompt.join(" ");
            let store = ModelStore::resolve(model_home)?;
            let backend_choice = resolve_backend(backend_override, device_override)?;
            let manifest = store.get(&model)?;
            let backend = build_generation_backend(store, backend_choice, llama_options.clone())?;
            let prompt = messages_to_prompt_for_model(
                &manifest,
                &[ChatMessage {
                    role: "user".to_string(),
                    content: Some(MessageContent::Text(prompt)),
                    name: None,
                }],
            );
            let request = GenerateRequest {
                prompt: prompt.prompt,
                image_urls: images,
                max_tokens,
                temperature,
                top_p,
                stop: prompt.stop,
                seed,
                stream_granularity: StreamGranularity::Chunk,
            };
            let response = backend.generate(&manifest, request)?;
            println!("{}", response.text.trim());
            if verbose {
                let mut stderr = io::stderr().lock();
                writeln!(stderr)?;
                write_verbose_stats(
                    &mut stderr,
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
        } => {
            let store = ModelStore::resolve(model_home)?;
            let backend_choice = resolve_backend(backend_override, device_override)?;
            let manifest = store.get(&model)?;
            let backend = build_generation_backend(store, backend_choice, llama_options.clone())?;
            chat_loop(
                backend,
                manifest,
                max_tokens,
                temperature,
                top_p,
                seed,
                images,
                stream_granularity.into(),
                verbose,
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
                print_perf_doctor(&store, &manifest, backend_choice, &llama_options)
            }
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
        | Commands::Bench { .. }
        | Commands::Doctor { .. }
        | Commands::List
        | Commands::Inspect { .. }
        | Commands::SelectFile { .. } => false,
    }
}

async fn chat_loop(
    backend: Arc<dyn GenerationBackend>,
    manifest: ModelManifest,
    max_tokens: usize,
    temperature: Option<f64>,
    top_p: Option<f64>,
    seed: Option<u64>,
    images: Vec<String>,
    stream_granularity: StreamGranularity,
    verbose: bool,
) -> Result<()> {
    let chat_session = backend.start_chat_session(&manifest, seed)?;
    if chat_session.is_none() {
        backend.prepare(&manifest)?;
    }

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
            image_urls: images.clone(),
            max_tokens,
            temperature,
            top_p,
            stop: prompt.stop,
            seed,
            stream_granularity,
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
            write_verbose_stats(&mut stdout, prompt_tokens, completion_tokens, timings)?;
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
) -> Result<Vec<BenchSample>> {
    let backend = build_concrete_backend(store, choice, runtime_options)?;
    backend.prepare(manifest)?;
    let session = backend.start_chat_session(manifest, Some(seed))?;

    let mut samples = Vec::with_capacity(runs);
    for index in 0..warmups + runs {
        let request = GenerateRequest {
            prompt: prompt.to_string(),
            image_urls: Vec::new(),
            max_tokens,
            temperature: Some(temperature),
            top_p,
            stop: stop.to_vec(),
            seed: Some(seed),
            stream_granularity: StreamGranularity::Chunk,
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
        BackendChoice::GgufPreferred { llama, .. } | BackendChoice::LlamaFast(llama) => {
            benchmark_llama_choices(llama, compare)
        }
        BackendChoice::LlamaHighlevel(_) => vec![choice],
        _ => vec![choice],
    }
}

fn benchmark_llama_choices(mode: LlamaCppMode, compare: BenchCompareArg) -> Vec<BackendChoice> {
    let mut choices = vec![BackendChoice::LlamaFast(mode)];
    if compare == BenchCompareArg::Legacy {
        choices.push(BackendChoice::LlamaHighlevel(mode));
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
        BackendChoice::LlamaFast(mode) | BackendChoice::GgufPreferred { llama: mode, .. } => {
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
) -> Result<()> {
    let selected = selected_backend_for_manifest(store, backend_choice, manifest)?;
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
        println!("note: detailed llama.cpp diagnostics are available only for llama-fast backends");
    }

    Ok(())
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
    prompt_tokens: usize,
    completion_tokens: usize,
    timings: GenerationTimings,
) -> io::Result<()> {
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
    LlamaFast(LlamaCppMode),
    LlamaHighlevel(LlamaCppMode),
    Mlx,
}

struct AutoBackend {
    store: ModelStore,
    runtime_options: LlamaRuntimeOptions,
    backends: Mutex<HashMap<&'static str, Arc<dyn GenerationBackend>>>,
}

struct GgufPreferredBackend {
    store: ModelStore,
    gguf_backend: BackendChoice,
    fallback_backend: BackendChoice,
    runtime_options: LlamaRuntimeOptions,
    backends: Mutex<HashMap<&'static str, Arc<dyn GenerationBackend>>>,
}

impl AutoBackend {
    fn new(store: ModelStore, runtime_options: LlamaRuntimeOptions) -> Self {
        Self {
            store,
            runtime_options,
            backends: Mutex::new(HashMap::new()),
        }
    }

    fn backend_for(&self, manifest: &ModelManifest) -> Result<Arc<dyn GenerationBackend>> {
        let mut unavailable = Vec::new();
        for backend in target_default_order().iter().copied() {
            if !backend_supports_manifest(backend, manifest) {
                continue;
            }
            if !backend_available_for_store(&self.store, backend) {
                unavailable.push(backend_label(backend));
                continue;
            }
            return self.cached_backend(backend);
        }

        let format = format!("{:?}", manifest.format);
        if unavailable.is_empty() {
            bail!(
                "no backend in this build supports model '{}' with format {format}",
                manifest.id
            );
        }
        bail!(
            "no available backend for model '{}' with format {format}; unavailable candidates: {}",
            manifest.id,
            unavailable.join(", ")
        );
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

        let backend =
            build_concrete_backend(self.store.clone(), backend, self.runtime_options.clone())?;
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
        self.backend_for(manifest)?.generate(manifest, request)
    }

    fn generate_stream(
        &self,
        manifest: ModelManifest,
        request: GenerateRequest,
    ) -> crate::backend::GenerateStream {
        match self.backend_for(&manifest) {
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
    ) -> Self {
        Self {
            store,
            gguf_backend: BackendChoice::LlamaFast(llama),
            fallback_backend: BackendChoice::Candle(candle),
            runtime_options,
            backends: Mutex::new(HashMap::new()),
        }
    }

    fn backend_for(&self, manifest: &ModelManifest) -> Result<Arc<dyn GenerationBackend>> {
        let backend = if manifest.format == ModelFormat::Gguf {
            self.gguf_backend
        } else {
            self.fallback_backend
        };
        if !backend_supports_manifest(backend, manifest) {
            bail!(
                "backend {} does not support model '{}' with format {:?}",
                backend_label(backend),
                manifest.id,
                manifest.format
            );
        }
        self.cached_backend(backend)
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

        let backend =
            build_concrete_backend(self.store.clone(), backend, self.runtime_options.clone())?;
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
        self.backend_for(manifest)?.generate(manifest, request)
    }

    fn generate_stream(
        &self,
        manifest: ModelManifest,
        request: GenerateRequest,
    ) -> crate::backend::GenerateStream {
        match self.backend_for(&manifest) {
            Ok(backend) => backend.generate_stream(manifest, request),
            Err(err) => Box::pin(tokio_stream::iter(vec![Err(err.to_string())])),
        }
    }
}

fn backend_arg_to_choice(backend: BackendArg) -> BackendChoice {
    match backend {
        BackendArg::Auto => BackendChoice::Auto,
        BackendArg::Cpu => BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Cpu,
            candle: CandleDeviceMode::Cpu,
        },
        BackendArg::Cuda => BackendChoice::GgufPreferred {
            llama: LlamaCppMode::Cuda,
            candle: candle_cuda_fallback_mode(),
        },
        BackendArg::Metal => BackendChoice::Candle(CandleDeviceMode::Metal),
        BackendArg::Mlx => BackendChoice::Mlx,
        BackendArg::Vulkan => BackendChoice::LlamaFast(LlamaCppMode::Vulkan),
        BackendArg::LlamaHighlevel => BackendChoice::LlamaHighlevel(preferred_llama_mode()),
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

fn candle_cuda_fallback_mode() -> CandleDeviceMode {
    if cfg!(feature = "candle-cuda") {
        CandleDeviceMode::Cuda
    } else {
        CandleDeviceMode::Cpu
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
) -> Result<Arc<dyn GenerationBackend>> {
    match backend {
        BackendChoice::Auto => Ok(Arc::new(AutoBackend::new(store, runtime_options))),
        backend => build_concrete_backend(store, backend, runtime_options),
    }
}

fn build_concrete_backend(
    store: ModelStore,
    backend: BackendChoice,
    runtime_options: LlamaRuntimeOptions,
) -> Result<Arc<dyn GenerationBackend>> {
    match backend {
        BackendChoice::Auto => bail!("auto backend cannot be built as a concrete backend"),
        BackendChoice::GgufPreferred { llama, candle } => Ok(Arc::new(GgufPreferredBackend::new(
            store,
            llama,
            candle,
            runtime_options,
        ))),
        BackendChoice::Candle(mode) => Ok(Arc::new(CandleBackend::new_with_device(store, mode)?)),
        BackendChoice::LlamaFast(mode) => Ok(Arc::new(LlamaFastBackend::new_with_options(
            store,
            mode,
            runtime_options,
        ))),
        BackendChoice::LlamaHighlevel(mode) => Ok(Arc::new(LlamaCppBackend::new(store, mode))),
        BackendChoice::Mlx => Ok(Arc::new(MlxBackend::new(store))),
    }
}

fn target_default_order() -> &'static [BackendChoice] {
    if cfg!(any(windows, target_os = "linux")) {
        &[
            BackendChoice::LlamaFast(LlamaCppMode::Cuda),
            BackendChoice::LlamaFast(LlamaCppMode::Vulkan),
            BackendChoice::LlamaFast(LlamaCppMode::Cpu),
            BackendChoice::LlamaHighlevel(LlamaCppMode::Cuda),
            BackendChoice::LlamaHighlevel(LlamaCppMode::Cpu),
            BackendChoice::Candle(CandleDeviceMode::Cuda),
            BackendChoice::Candle(CandleDeviceMode::Cpu),
        ]
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        &[
            BackendChoice::Mlx,
            BackendChoice::Candle(CandleDeviceMode::Cpu),
        ]
    } else if cfg!(target_os = "macos") {
        &[BackendChoice::Candle(CandleDeviceMode::Cpu)]
    } else {
        &[
            BackendChoice::Candle(CandleDeviceMode::Cpu),
            BackendChoice::Candle(CandleDeviceMode::Cuda),
            BackendChoice::LlamaFast(LlamaCppMode::Vulkan),
            BackendChoice::LlamaHighlevel(LlamaCppMode::Vulkan),
        ]
    }
}

fn backend_supports_manifest(backend: BackendChoice, manifest: &ModelManifest) -> bool {
    match backend {
        BackendChoice::Auto => false,
        BackendChoice::GgufPreferred { .. } => matches!(
            manifest.format,
            ModelFormat::Gguf | ModelFormat::SafeTensors
        ),
        BackendChoice::Candle(_) => matches!(
            manifest.format,
            ModelFormat::Gguf | ModelFormat::SafeTensors
        ),
        BackendChoice::Mlx => {
            matches!(manifest.format, ModelFormat::Mlx | ModelFormat::SafeTensors)
        }
        BackendChoice::LlamaFast(_) | BackendChoice::LlamaHighlevel(_) => {
            manifest.format == ModelFormat::Gguf
        }
    }
}

fn backend_available_for_store(_store: &ModelStore, backend: BackendChoice) -> bool {
    match backend {
        BackendChoice::Auto => true,
        BackendChoice::GgufPreferred { .. } => true,
        BackendChoice::Candle(CandleDeviceMode::Auto) => true,
        BackendChoice::Candle(CandleDeviceMode::Cpu) => true,
        BackendChoice::Candle(mode) => probe_device(mode).is_ok(),
        BackendChoice::Mlx => MlxBackend::probe().is_ok(),
        BackendChoice::LlamaFast(mode) => LlamaFastBackend::probe(mode).is_ok(),
        BackendChoice::LlamaHighlevel(mode) => LlamaCppBackend::probe(mode).is_ok(),
    }
}

fn selected_backend_for_manifest(
    store: &ModelStore,
    backend: BackendChoice,
    manifest: &ModelManifest,
) -> Result<BackendChoice> {
    match backend {
        BackendChoice::Auto => {
            let mut unavailable = Vec::new();
            for candidate in target_default_order().iter().copied() {
                if !backend_supports_manifest(candidate, manifest) {
                    continue;
                }
                if backend_available_for_store(store, candidate) {
                    return Ok(candidate);
                }
                unavailable.push(backend_label(candidate));
            }
            bail!(
                "no available backend for model '{}' with format {:?}; unavailable candidates: {}",
                manifest.id,
                manifest.format,
                unavailable.join(", ")
            )
        }
        BackendChoice::GgufPreferred { llama, candle } => {
            let selected = if manifest.format == ModelFormat::Gguf {
                BackendChoice::LlamaFast(llama)
            } else {
                BackendChoice::Candle(candle)
            };
            if backend_supports_manifest(selected, manifest) {
                Ok(selected)
            } else {
                bail!(
                    "backend {} does not support model '{}' with format {:?}",
                    backend_label(selected),
                    manifest.id,
                    manifest.format
                )
            }
        }
        selected => {
            if backend_supports_manifest(selected, manifest) {
                Ok(selected)
            } else {
                bail!(
                    "backend {} does not support model '{}' with format {:?}",
                    backend_label(selected),
                    manifest.id,
                    manifest.format
                )
            }
        }
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
        BackendChoice::Candle(CandleDeviceMode::Cpu) => "candle-cpu",
        BackendChoice::Candle(CandleDeviceMode::Cuda) => "candle-cuda",
        BackendChoice::Candle(CandleDeviceMode::Metal) => "metal",
        BackendChoice::Mlx => "mlx",
        BackendChoice::LlamaFast(LlamaCppMode::Cuda) => "llama-fast-cuda",
        BackendChoice::LlamaFast(LlamaCppMode::Vulkan) => "llama-fast-vulkan",
        BackendChoice::LlamaFast(LlamaCppMode::Cpu) => "llama-fast-cpu",
        BackendChoice::LlamaHighlevel(mode) => mode.label(),
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

        let cli = Cli::try_parse_from([
            "werk",
            "run",
            "gemma-2b-it",
            "hello",
            "--image",
            "image.png",
        ])
        .unwrap();
        match cli.command.unwrap() {
            Commands::Run {
                model,
                prompt,
                images,
                ..
            } => {
                assert_eq!(model, "gemma-2b-it");
                assert_eq!(prompt, vec!["hello"]);
                assert_eq!(images, vec!["image.png"]);
            }
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from([
            "werk",
            "chat",
            "gemma-2b-it",
            "--stream-granularity",
            "chunk",
        ])
        .unwrap();
        match cli.command.unwrap() {
            Commands::Chat {
                model,
                stream_granularity,
                ..
            } => {
                assert_eq!(model, "gemma-2b-it");
                assert_eq!(stream_granularity, StreamGranularityArg::Chunk);
            }
            command => panic!("unexpected command: {command:?}"),
        }

        let cli = Cli::try_parse_from(["werk", "--backend", "vulkan", "chat", "tiny"]).unwrap();
        assert_eq!(cli.backend, BackendArg::Vulkan);
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
            BackendChoice::LlamaFast(LlamaCppMode::Vulkan)
        ));
        assert!(matches!(
            backend_arg_to_choice(BackendArg::LlamaHighlevel),
            BackendChoice::LlamaHighlevel(_)
        ));
        assert!(matches!(
            backend_arg_to_choice(BackendArg::Cuda),
            BackendChoice::GgufPreferred {
                llama: LlamaCppMode::Cuda,
                candle
            } if candle == candle_cuda_fallback_mode()
        ));
    }

    #[test]
    fn linux_and_windows_auto_prefer_llama_fast_for_gguf() {
        if cfg!(any(windows, target_os = "linux")) {
            let order = target_default_order();
            assert!(matches!(
                order[0],
                BackendChoice::LlamaFast(LlamaCppMode::Cuda)
            ));
            assert!(matches!(
                order[1],
                BackendChoice::LlamaFast(LlamaCppMode::Vulkan)
            ));
            assert!(matches!(
                order[2],
                BackendChoice::LlamaFast(LlamaCppMode::Cpu)
            ));
        }
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
}
