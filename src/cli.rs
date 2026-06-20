use anyhow::{Result, anyhow, bail};
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    collections::HashMap,
    io::{self, IsTerminal, Write},
    net::{IpAddr, SocketAddr},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio_stream::StreamExt;

use crate::{
    api::{ApiState, serve},
    backend::{
        CandleBackend, CandleDeviceMode, GenerateRequest, GenerateStreamEvent, GenerationBackend,
        GenerationTimings, LlamaCppBackend, MlxBackend, StreamGranularity, probe_device,
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
        help = "Backend for this process: auto, cpu, cuda, metal, mlx, or vulkan"
    )]
    pub backend: BackendArg,

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
    Metal,
    Mlx,
    Vulkan,
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

pub async fn run_from_env() -> Result<()> {
    run(Cli::parse()).await
}

pub async fn run(cli: Cli) -> Result<()> {
    let model_home = cli.model_home;
    let device_override = cli.device;
    let backend_override = cli.backend;
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
            if let Some(model) = model.as_deref() {
                store.get(&model)?;
                println!("Default model available: {model}");
            }
            let ip: IpAddr = host.parse()?;
            let addr = SocketAddr::new(ip, port);
            let backend = build_generation_backend(store.clone(), backend_choice)?;
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
            let backend = build_generation_backend(store, backend_choice)?;
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
            let backend = build_generation_backend(store, backend_choice)?;
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
        let mut stream = backend.generate_stream(manifest.clone(), request);
        while let Some(event) = stream.next().await {
            match event {
                Ok(GenerateStreamEvent::TextChunk(chunk)) => {
                    print!("{chunk}");
                    io::stdout().flush()?;
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
    Candle(CandleDeviceMode),
    Mlx,
    Vulkan,
}

struct AutoBackend {
    store: ModelStore,
    backends: Mutex<HashMap<&'static str, Arc<dyn GenerationBackend>>>,
}

impl AutoBackend {
    fn new(store: ModelStore) -> Self {
        Self {
            store,
            backends: Mutex::new(HashMap::new()),
        }
    }

    fn backend_for(&self, manifest: &ModelManifest) -> Result<Arc<dyn GenerationBackend>> {
        let mut unavailable = Vec::new();
        for backend in target_default_order().iter().copied() {
            if !backend_supports_manifest(backend, manifest) {
                continue;
            }
            if !backend_available(backend) {
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

        let backend = build_concrete_backend(self.store.clone(), backend)?;
        self.backends
            .lock()
            .map_err(|_| anyhow!("auto backend cache mutex poisoned"))?
            .insert(key, backend.clone());
        Ok(backend)
    }
}

impl GenerationBackend for AutoBackend {
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
        BackendArg::Cpu => BackendChoice::Candle(CandleDeviceMode::Cpu),
        BackendArg::Cuda => BackendChoice::Candle(CandleDeviceMode::Cuda),
        BackendArg::Metal => BackendChoice::Candle(CandleDeviceMode::Metal),
        BackendArg::Mlx => BackendChoice::Mlx,
        BackendArg::Vulkan => BackendChoice::Vulkan,
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
) -> Result<Arc<dyn GenerationBackend>> {
    match backend {
        BackendChoice::Auto => Ok(Arc::new(AutoBackend::new(store))),
        backend => build_concrete_backend(store, backend),
    }
}

fn build_concrete_backend(
    store: ModelStore,
    backend: BackendChoice,
) -> Result<Arc<dyn GenerationBackend>> {
    match backend {
        BackendChoice::Auto => bail!("auto backend cannot be built as a concrete backend"),
        BackendChoice::Candle(mode) => Ok(Arc::new(CandleBackend::new_with_device(store, mode)?)),
        BackendChoice::Mlx => Ok(Arc::new(MlxBackend::new(store))),
        BackendChoice::Vulkan => Ok(Arc::new(LlamaCppBackend::new_vulkan(store))),
    }
}

fn target_default_order() -> &'static [BackendChoice] {
    if cfg!(windows) {
        &[
            BackendChoice::Candle(CandleDeviceMode::Cuda),
            BackendChoice::Vulkan,
            BackendChoice::Candle(CandleDeviceMode::Cpu),
        ]
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        &[
            BackendChoice::Mlx,
            BackendChoice::Candle(CandleDeviceMode::Metal),
            BackendChoice::Candle(CandleDeviceMode::Cpu),
        ]
    } else if cfg!(target_os = "macos") {
        &[
            BackendChoice::Candle(CandleDeviceMode::Metal),
            BackendChoice::Candle(CandleDeviceMode::Cpu),
        ]
    } else {
        &[
            BackendChoice::Vulkan,
            BackendChoice::Candle(CandleDeviceMode::Cuda),
            BackendChoice::Candle(CandleDeviceMode::Cpu),
        ]
    }
}

fn backend_supports_manifest(backend: BackendChoice, manifest: &ModelManifest) -> bool {
    match backend {
        BackendChoice::Auto => false,
        BackendChoice::Candle(_) => matches!(
            manifest.format,
            ModelFormat::Gguf | ModelFormat::SafeTensors
        ),
        BackendChoice::Mlx => {
            matches!(manifest.format, ModelFormat::Mlx | ModelFormat::SafeTensors)
        }
        BackendChoice::Vulkan => manifest.format == ModelFormat::Gguf,
    }
}

fn backend_available(backend: BackendChoice) -> bool {
    match backend {
        BackendChoice::Auto => true,
        BackendChoice::Candle(CandleDeviceMode::Auto) => true,
        BackendChoice::Candle(CandleDeviceMode::Cpu) => true,
        BackendChoice::Candle(mode) => probe_device(mode).is_ok(),
        BackendChoice::Mlx => MlxBackend::probe().is_ok(),
        BackendChoice::Vulkan => LlamaCppBackend::probe_vulkan().is_ok(),
    }
}

fn backend_label(backend: BackendChoice) -> &'static str {
    match backend {
        BackendChoice::Auto => "auto",
        BackendChoice::Candle(CandleDeviceMode::Auto) => "candle-auto",
        BackendChoice::Candle(CandleDeviceMode::Cpu) => "cpu",
        BackendChoice::Candle(CandleDeviceMode::Cuda) => "cuda",
        BackendChoice::Candle(CandleDeviceMode::Metal) => "metal",
        BackendChoice::Mlx => "mlx",
        BackendChoice::Vulkan => "vulkan",
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
            BackendChoice::Vulkan
        ));
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
