use anyhow::{Context, Result, anyhow, bail};
use candle_core::quantized::gguf_file;
use crc32fast::Hasher;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::VecDeque,
    env,
    ffi::OsStr,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::mpsc,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub const MANIFEST_FILE: &str = "manifest.json";

#[derive(Debug, Clone)]
pub struct ModelStore {
    home: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelFormat {
    Gguf,
    SafeTensors,
    PyTorch,
    Onnx,
    Mlx,
    TensorRt,
    OpenVino,
    TensorFlow,
    CoreMl,
    Unknown,
}

impl ModelFormat {
    pub fn backend_hint(&self) -> &'static str {
        match self {
            Self::Gguf | Self::SafeTensors => "candle",
            Self::PyTorch => "pytorch",
            Self::Onnx => "onnxruntime",
            Self::Mlx => "mlx",
            Self::TensorRt => "tensorrt",
            Self::OpenVino => "openvino",
            Self::TensorFlow => "tensorflow",
            Self::CoreMl => "coreml",
            Self::Unknown => "unknown",
        }
    }

    pub fn backend_status(&self) -> &'static str {
        match self {
            Self::Gguf => {
                "implemented via Candle quantized GGUF loaders for selected architectures"
            }
            Self::SafeTensors => {
                "implemented via Candle safetensors loaders for selected architectures"
            }
            Self::PyTorch => "catalog/import only; PyTorch backend is not wired yet",
            Self::Onnx => "catalog/import only; ONNX Runtime backend is not wired yet",
            Self::Mlx => "implemented through external mlx-lm backend when configured",
            Self::TensorRt => "catalog/import only; TensorRT backend is not wired yet",
            Self::OpenVino => "catalog/import only; OpenVINO backend is not wired yet",
            Self::TensorFlow => "catalog/import only; TensorFlow backend is not wired yet",
            Self::CoreMl => "catalog/import only; CoreML backend is not wired yet",
            Self::Unknown => "catalog/import only; format could not be detected",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelSource {
    LocalPath { path: String },
    HuggingFace { repo: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFile {
    pub path: String,
    pub size: u64,
    pub checksum: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub id: String,
    pub source: ModelSource,
    pub format: ModelFormat,
    pub architecture: Option<String>,
    pub tokenizer_path: Option<String>,
    pub config_path: Option<String>,
    pub model_path: Option<String>,
    pub backend: String,
    pub created_unix: u64,
    pub files: Vec<ModelFile>,
}

#[derive(Debug, Clone)]
pub enum PullProgress {
    Started {
        url: String,
    },
    GitProgress {
        line: String,
    },
    CloneFinished,
    LfsStarted,
    LfsProgress {
        line: String,
    },
    TransferStats {
        bytes: u64,
        total_bytes: Option<u64>,
        bytes_per_second: f64,
    },
    LfsFinished,
    Importing,
    Finished {
        files: usize,
        bytes: u64,
    },
}

impl ModelStore {
    pub fn resolve(home: Option<PathBuf>) -> Result<Self> {
        let home = match home {
            Some(home) => home,
            None => default_home()?,
        };
        Ok(Self { home })
    }

    pub fn home(&self) -> &Path {
        &self.home
    }

    pub fn ensure(&self) -> Result<()> {
        fs::create_dir_all(self.models_dir())?;
        fs::create_dir_all(self.tmp_dir())?;
        Ok(())
    }

    pub fn models_dir(&self) -> PathBuf {
        self.home.join("models")
    }

    fn tmp_dir(&self) -> PathBuf {
        self.home.join("tmp")
    }

    pub fn model_dir(&self, id: &str) -> PathBuf {
        self.models_dir().join(sanitize_id(id))
    }

    pub fn import_path(&self, source: &Path, id: &str) -> Result<ModelManifest> {
        self.import_path_with_source(
            source,
            id,
            ModelSource::LocalPath {
                path: source.display().to_string(),
            },
        )
    }

    fn import_path_with_source(
        &self,
        source: &Path,
        id: &str,
        model_source: ModelSource,
    ) -> Result<ModelManifest> {
        self.ensure()?;
        validate_id(id)?;
        let dest = self.model_dir(id);
        if dest.exists() {
            bail!("model '{id}' already exists at {}", dest.display());
        }

        let files_dir = dest.join("files");
        fs::create_dir_all(&files_dir)?;
        if source.is_dir() {
            copy_dir_contents(source, &files_dir)?;
        } else if source.is_file() {
            let name = source
                .file_name()
                .ok_or_else(|| anyhow!("cannot determine file name for {}", source.display()))?;
            fs::copy(source, files_dir.join(name))?;
        } else {
            bail!("import path does not exist: {}", source.display());
        }

        let manifest = self.build_manifest(id, model_source, &dest)?;
        write_json_pretty(&dest.join(MANIFEST_FILE), &manifest)?;
        Ok(manifest)
    }

    pub fn pull_from_huggingface(&self, repo: &str, name: Option<&str>) -> Result<ModelManifest> {
        self.pull_from_huggingface_with_progress(repo, name, None, |_| {})
    }

    pub fn pull_from_huggingface_with_progress<F>(
        &self,
        repo: &str,
        name: Option<&str>,
        file: Option<&str>,
        mut progress: F,
    ) -> Result<ModelManifest>
    where
        F: FnMut(PullProgress),
    {
        self.ensure()?;
        validate_hf_repo(repo)?;
        let id = name.unwrap_or(repo);
        validate_id(id)?;
        if self.model_dir(id).exists() {
            bail!("model '{id}' already exists");
        }

        let tmp = self
            .tmp_dir()
            .join(format!("pull-{}-{}", sanitize_id(id), std::process::id()));
        if tmp.exists() {
            fs::remove_dir_all(&tmp).with_context(|| {
                format!(
                    "failed to remove stale temporary directory {}",
                    tmp.display()
                )
            })?;
        }

        let url = format!("https://huggingface.co/{repo}");
        progress(PullProgress::Started { url: url.clone() });

        run_git_with_progress(
            Command::new("git")
                .env("GIT_LFS_SKIP_SMUDGE", "1")
                .args(["clone", "--progress", "--depth", "1", &url])
                .arg(&tmp),
            &format!("git clone failed for {url}"),
            None,
            None,
            |event| match event {
                GitCommandProgress::Line(line) => progress(PullProgress::GitProgress { line }),
                GitCommandProgress::Stats {
                    bytes,
                    total_bytes,
                    bytes_per_second,
                } => progress(PullProgress::TransferStats {
                    bytes,
                    total_bytes,
                    bytes_per_second,
                }),
            },
        )?;
        progress(PullProgress::CloneFinished);

        let include_file = file.map(|file| resolve_pull_file(&tmp, file)).transpose()?;

        if tmp.join(".gitattributes").is_file() {
            let total_bytes = lfs_pointer_total(&tmp, include_file.as_deref())?;
            progress(PullProgress::LfsStarted);
            let mut lfs_command = Command::new("git");
            lfs_command.arg("-C").arg(&tmp).args(["lfs", "pull"]);
            if let Some(include_file) = include_file.as_deref() {
                lfs_command
                    .arg("--include")
                    .arg(include_file)
                    .arg("--exclude")
                    .arg("");
            }
            run_git_with_progress(
                &mut lfs_command,
                "git lfs pull failed; install git-lfs and run `git lfs install`",
                Some(&tmp),
                total_bytes,
                |event| match event {
                    GitCommandProgress::Line(line) => progress(PullProgress::LfsProgress { line }),
                    GitCommandProgress::Stats {
                        bytes,
                        total_bytes,
                        bytes_per_second,
                    } => progress(PullProgress::TransferStats {
                        bytes,
                        total_bytes,
                        bytes_per_second,
                    }),
                },
            )?;
            progress(PullProgress::LfsFinished);
        }

        if let Some(include_file) = include_file.as_deref() {
            ensure_lfs_file_downloaded(&tmp, include_file)?;
            remove_lfs_pointer_files(&tmp)?;
        }

        progress(PullProgress::Importing);

        let manifest = self.import_path_with_source(
            &tmp,
            id,
            ModelSource::HuggingFace {
                repo: repo.to_string(),
            },
        );
        let _ = fs::remove_dir_all(&tmp);
        let manifest = manifest?;
        progress(PullProgress::Finished {
            files: manifest.files.len(),
            bytes: manifest.files.iter().map(|file| file.size).sum(),
        });
        Ok(manifest)
    }

    pub fn list(&self) -> Result<Vec<ModelManifest>> {
        self.ensure()?;
        let mut manifests = Vec::new();
        for entry in fs::read_dir(self.models_dir())? {
            let entry = entry?;
            let manifest_path = entry.path().join(MANIFEST_FILE);
            if manifest_path.is_file() {
                manifests.push(read_manifest(&manifest_path)?);
            }
        }
        manifests.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(manifests)
    }

    pub fn get(&self, id: &str) -> Result<ModelManifest> {
        self.ensure()?;
        let direct = self.model_dir(id).join(MANIFEST_FILE);
        if direct.is_file() {
            return read_manifest(&direct);
        }

        self.list()?
            .into_iter()
            .find(|manifest| manifest.id == id)
            .ok_or_else(|| anyhow!("model '{id}' is not installed"))
    }

    pub fn absolute_model_file(&self, manifest: &ModelManifest, relative_path: &str) -> PathBuf {
        self.model_dir(&manifest.id).join(relative_path)
    }

    pub fn set_model_file(&self, id: &str, file: &str) -> Result<ModelManifest> {
        self.ensure()?;
        let mut manifest = self.get(id)?;
        let selected_path = normalize_model_file_path(file)?;
        if !manifest
            .files
            .iter()
            .any(|model_file| model_file.path == selected_path)
        {
            bail!(
                "file '{selected_path}' is not tracked by model '{}'; run `werk inspect {}` to see available files",
                manifest.id,
                manifest.id
            );
        }

        let absolute_path = self.absolute_model_file(&manifest, &selected_path);
        if !absolute_path.is_file() {
            bail!("model file does not exist: {}", absolute_path.display());
        }

        validate_selected_model_file(&manifest.format, &selected_path)?;
        manifest.model_path = Some(selected_path);
        manifest.architecture = detect_architecture(
            &self.model_dir(&manifest.id),
            &manifest.format,
            manifest.model_path.as_deref(),
            manifest.config_path.as_deref(),
        );
        write_json_pretty(&self.model_dir(&manifest.id).join(MANIFEST_FILE), &manifest)?;
        Ok(manifest)
    }

    fn build_manifest(
        &self,
        id: &str,
        source: ModelSource,
        model_dir: &Path,
    ) -> Result<ModelManifest> {
        let files_root = model_dir.join("files");
        let mut file_paths = Vec::new();
        collect_files(&files_root, &mut file_paths)?;
        file_paths.sort();

        let mut files = Vec::with_capacity(file_paths.len());
        for path in &file_paths {
            let rel = path
                .strip_prefix(model_dir)
                .context("model file is not inside model directory")?
                .to_string_lossy()
                .replace('\\', "/");
            let metadata = fs::metadata(path)?;
            files.push(ModelFile {
                path: rel,
                size: metadata.len(),
                checksum: format!("crc32:{:08x}", crc32(path)?),
            });
        }

        let format = detect_format(&file_paths);
        let model_path = first_model_path(model_dir, &file_paths, &format);
        let tokenizer_path = first_relative_by_name(model_dir, &file_paths, "tokenizer.json");
        let config_path = first_relative_by_name(model_dir, &file_paths, "config.json");
        let architecture = detect_architecture(
            model_dir,
            &format,
            model_path.as_deref(),
            config_path.as_deref(),
        );
        let backend = format.backend_hint().to_string();

        Ok(ModelManifest {
            id: id.to_string(),
            source,
            format,
            architecture,
            tokenizer_path,
            config_path,
            model_path,
            backend,
            created_unix: unix_ts(),
            files,
        })
    }
}

pub fn default_home() -> Result<PathBuf> {
    if let Ok(home) = env::var("WERK_HOME") {
        return Ok(PathBuf::from(home));
    }
    if let Ok(data_home) = env::var("XDG_DATA_HOME") {
        return Ok(PathBuf::from(data_home).join("werk1112"));
    }
    if cfg!(windows) {
        if let Ok(local_app_data) = env::var("LOCALAPPDATA") {
            return Ok(PathBuf::from(local_app_data).join("werk1112"));
        }
        if let Ok(user_profile) = env::var("USERPROFILE") {
            return Ok(PathBuf::from(user_profile).join("AppData/Local/werk1112"));
        }
    }
    let home = env::var("HOME").context("HOME is not set; set WERK_HOME explicitly")?;
    Ok(PathBuf::from(home).join(".local/share/werk1112"))
}

pub fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

fn validate_id(id: &str) -> Result<()> {
    if id.trim().is_empty() {
        bail!("model id cannot be empty");
    }
    if id.contains("..") || id.starts_with('-') {
        bail!("model id contains unsupported path-like syntax: {id}");
    }
    Ok(())
}

fn validate_hf_repo(repo: &str) -> Result<()> {
    if repo.trim().is_empty() || repo.starts_with('-') || repo.contains("..") {
        bail!("invalid Hugging Face repo id: {repo}");
    }
    Ok(())
}

fn run_git_with_progress<F>(
    command: &mut Command,
    error_context: &str,
    watch_path: Option<&Path>,
    total_bytes: Option<u64>,
    mut progress: F,
) -> Result<()>
where
    F: FnMut(GitCommandProgress),
{
    let mut child = command
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to execute git; install git and git-lfs for Hugging Face pulls")?;

    let (line_tx, line_rx) = mpsc::channel();
    let reader = child.stderr.take().map(|stderr| {
        thread::spawn(move || -> Result<VecDeque<String>, String> {
            let mut stderr_tail = VecDeque::new();
            read_git_progress(stderr, &mut stderr_tail, &mut |line| {
                let _ = line_tx.send(line);
            })
            .map_err(|err| err.to_string())?;
            Ok(stderr_tail)
        })
    });

    let mut stderr_tail = VecDeque::<String>::new();
    let mut last_stats_at = Instant::now();
    let mut last_bytes = watch_path.and_then(|path| dir_size(path).ok()).unwrap_or(0);

    loop {
        while let Ok(line) = line_rx.try_recv() {
            push_tail(&mut stderr_tail, line.clone());
            progress(GitCommandProgress::Line(line));
        }

        if let Some(status) = child.try_wait().context("failed to wait for git command")? {
            while let Ok(line) = line_rx.try_recv() {
                push_tail(&mut stderr_tail, line.clone());
                progress(GitCommandProgress::Line(line));
            }

            if let Some(reader) = reader {
                match reader.join() {
                    Ok(Ok(reader_tail)) => {
                        for line in reader_tail {
                            push_tail(&mut stderr_tail, line);
                        }
                    }
                    Ok(Err(err)) => bail!("{error_context}: {err}"),
                    Err(_) => bail!("{error_context}: failed to read git progress output"),
                }
            }

            if !status.success() {
                let stderr = stderr_tail.into_iter().collect::<Vec<_>>().join("\n");
                bail!("{error_context}: {}", stderr.trim());
            }

            return Ok(());
        }

        if let Some(path) = watch_path
            && last_stats_at.elapsed() >= Duration::from_millis(750)
            && let Ok(bytes) = dir_size(path)
        {
            let elapsed = last_stats_at.elapsed().as_secs_f64().max(0.001);
            let bytes_per_second = bytes.saturating_sub(last_bytes) as f64 / elapsed;
            progress(GitCommandProgress::Stats {
                bytes,
                total_bytes,
                bytes_per_second,
            });
            last_bytes = bytes;
            last_stats_at = Instant::now();
        }

        thread::sleep(Duration::from_millis(100));
    }
}

enum GitCommandProgress {
    Line(String),
    Stats {
        bytes: u64,
        total_bytes: Option<u64>,
        bytes_per_second: f64,
    },
}

fn push_tail(stderr_tail: &mut VecDeque<String>, line: String) {
    stderr_tail.push_back(line);
    while stderr_tail.len() > 20 {
        stderr_tail.pop_front();
    }
}

fn dir_size(path: &Path) -> Result<u64> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.is_file() {
        return Ok(metadata.len());
    }
    if !metadata.is_dir() {
        return Ok(0);
    }

    let mut bytes = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        bytes = bytes.saturating_add(dir_size(&entry.path())?);
    }
    Ok(bytes)
}

fn resolve_pull_file(root: &Path, file: &str) -> Result<String> {
    let file = normalize_relative_repo_path(file)?;
    if root.join(&file).exists() {
        return Ok(file);
    }

    if !file.contains('/') {
        let mut files = Vec::new();
        collect_files(root, &mut files)?;
        let mut matches = files
            .into_iter()
            .filter(|path| path.file_name().and_then(OsStr::to_str) == Some(file.as_str()))
            .filter_map(|path| relative_string(root, &path))
            .collect::<Vec<_>>();
        matches.sort();
        match matches.len() {
            0 => {}
            1 => return Ok(matches.remove(0)),
            _ => bail!(
                "file name '{file}' is ambiguous in the repository; pass one of these relative paths: {}",
                matches.join(", ")
            ),
        }
    }

    bail!("file '{file}' was not found in the Hugging Face repository")
}

fn normalize_relative_repo_path(path: &str) -> Result<String> {
    let mut path = path.trim().replace('\\', "/");
    while let Some(rest) = path.strip_prefix("./") {
        path = rest.to_string();
    }
    if path.is_empty() {
        bail!("file cannot be empty");
    }
    if path.starts_with('/') || path.split('/').any(|part| part.is_empty() || part == "..") {
        bail!("file must be a relative path inside the repository");
    }
    Ok(path)
}

fn ensure_lfs_file_downloaded(root: &Path, file: &str) -> Result<()> {
    let path = root.join(file);
    if !path.is_file() {
        bail!("selected file was not downloaded: {file}");
    }
    if lfs_pointer_size(&path)?.is_some() {
        bail!(
            "selected file is still a Git LFS pointer after download: {file}; check the filename and git-lfs setup"
        );
    }
    Ok(())
}

fn remove_lfs_pointer_files(path: &Path) -> Result<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_name() == ".git" {
            continue;
        }

        let path = entry.path();
        if path.is_dir() {
            remove_lfs_pointer_files(&path)?;
        } else if path.is_file() && lfs_pointer_size(&path)?.is_some() {
            fs::remove_file(&path)
                .with_context(|| format!("failed to remove LFS pointer {}", path.display()))?;
        }
    }
    Ok(())
}

fn lfs_pointer_total(path: &Path, include_file: Option<&str>) -> Result<Option<u64>> {
    let mut total = 0u64;
    collect_lfs_pointer_total(path, path, include_file, &mut total)?;
    Ok((total > 0).then_some(total))
}

fn collect_lfs_pointer_total(
    root: &Path,
    path: &Path,
    include_file: Option<&str>,
    total: &mut u64,
) -> Result<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_name() == ".git" {
            continue;
        }

        let path = entry.path();
        if path.is_dir() {
            collect_lfs_pointer_total(root, &path, include_file, total)?;
        } else if path.is_file()
            && include_file
                .map(|include_file| relative_string(root, &path).as_deref() == Some(include_file))
                .unwrap_or(true)
            && let Some(size) = lfs_pointer_size(&path)?
        {
            *total = total.saturating_add(size);
        }
    }
    Ok(())
}

fn lfs_pointer_size(path: &Path) -> Result<Option<u64>> {
    let metadata = fs::metadata(path)?;
    if metadata.len() > 4096 {
        return Ok(None);
    }

    let mut file = fs::File::open(path)?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.read_to_end(&mut bytes)?;
    let text = String::from_utf8_lossy(&bytes);
    if !text.starts_with("version https://git-lfs.github.com/spec/v1") {
        return Ok(None);
    }

    Ok(text
        .lines()
        .find_map(|line| line.strip_prefix("size ")?.parse::<u64>().ok()))
}

fn read_git_progress<R, F>(
    mut reader: R,
    stderr_tail: &mut VecDeque<String>,
    progress: &mut F,
) -> Result<()>
where
    R: Read,
    F: FnMut(String),
{
    let mut buffer = Vec::new();
    let mut byte = [0u8; 1];

    loop {
        match reader.read(&mut byte) {
            Ok(0) => break,
            Ok(_) if matches!(byte[0], b'\r' | b'\n') => {
                emit_git_progress_line(&mut buffer, stderr_tail, progress);
            }
            Ok(_) => buffer.push(byte[0]),
            Err(err) => return Err(err).context("failed to read git progress output"),
        }
    }
    emit_git_progress_line(&mut buffer, stderr_tail, progress);
    Ok(())
}

fn emit_git_progress_line<F>(
    buffer: &mut Vec<u8>,
    stderr_tail: &mut VecDeque<String>,
    progress: &mut F,
) where
    F: FnMut(String),
{
    if buffer.is_empty() {
        return;
    }

    let line = String::from_utf8_lossy(buffer)
        .chars()
        .filter(|ch| !ch.is_control() || matches!(ch, '\t'))
        .collect::<String>()
        .trim()
        .to_string();
    buffer.clear();

    if line.is_empty() {
        return;
    }

    push_tail(stderr_tail, line.clone());
    progress(line);
}

fn copy_dir_contents(source: &Path, dest: &Path) -> Result<()> {
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        if entry.file_name() == ".git" {
            continue;
        }
        let src_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        if src_path.is_dir() {
            fs::create_dir_all(&dest_path)?;
            copy_dir_contents(&src_path, &dest_path)?;
        } else if src_path.is_file() {
            fs::copy(&src_path, &dest_path).with_context(|| {
                format!("copy {} -> {}", src_path.display(), dest_path.display())
            })?;
        }
    }
    Ok(())
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_name() == ".git" {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else if path.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

fn crc32(path: &Path) -> Result<u32> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Hasher::new();
    let mut buf = [0_u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize())
}

fn extension_eq(path: &Path, ext: &str) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .map(|value| value.eq_ignore_ascii_case(ext))
        .unwrap_or(false)
}

fn detect_format(files: &[PathBuf]) -> ModelFormat {
    if any_extension(files, "gguf") {
        ModelFormat::Gguf
    } else if any_extension(files, "safetensors") {
        ModelFormat::SafeTensors
    } else if any_extension(files, "onnx") {
        ModelFormat::Onnx
    } else if any_extension(files, "engine") || any_extension(files, "plan") {
        ModelFormat::TensorRt
    } else if any_extension(files, "xml") && any_extension(files, "bin") {
        ModelFormat::OpenVino
    } else if any_extension(files, "mlmodel") || any_path_contains(files, ".mlpackage") {
        ModelFormat::CoreMl
    } else if any_extension(files, "npz") || any_path_contains(files, "mlx") {
        ModelFormat::Mlx
    } else if any_extension(files, "pt")
        || any_extension(files, "pth")
        || any_file_name(files, "pytorch_model.bin")
    {
        ModelFormat::PyTorch
    } else if any_extension(files, "pb")
        || any_extension(files, "ckpt")
        || any_file_name(files, "checkpoint")
        || any_path_contains(files, ".ckpt-")
    {
        ModelFormat::TensorFlow
    } else {
        ModelFormat::Unknown
    }
}

fn any_extension(files: &[PathBuf], ext: &str) -> bool {
    files.iter().any(|path| extension_eq(path, ext))
}

fn any_file_name(files: &[PathBuf], name: &str) -> bool {
    files
        .iter()
        .any(|path| path.file_name().and_then(OsStr::to_str) == Some(name))
}

fn any_path_contains(files: &[PathBuf], needle: &str) -> bool {
    files
        .iter()
        .any(|path| path.to_string_lossy().to_ascii_lowercase().contains(needle))
}

fn first_model_path(model_dir: &Path, files: &[PathBuf], format: &ModelFormat) -> Option<String> {
    match format {
        ModelFormat::Gguf => first_gguf_model_path(model_dir, files),
        ModelFormat::SafeTensors => first_relative_by_extension(model_dir, files, "safetensors"),
        ModelFormat::PyTorch => first_relative_by_extension(model_dir, files, "pt")
            .or_else(|| first_relative_by_extension(model_dir, files, "pth"))
            .or_else(|| first_relative_by_name(model_dir, files, "pytorch_model.bin")),
        ModelFormat::Onnx => first_relative_by_extension(model_dir, files, "onnx"),
        ModelFormat::Mlx => first_relative_by_extension(model_dir, files, "npz")
            .or_else(|| first_relative_by_extension(model_dir, files, "safetensors")),
        ModelFormat::TensorRt => first_relative_by_extension(model_dir, files, "engine")
            .or_else(|| first_relative_by_extension(model_dir, files, "plan")),
        ModelFormat::OpenVino => first_relative_by_extension(model_dir, files, "xml"),
        ModelFormat::TensorFlow => first_relative_by_extension(model_dir, files, "pb")
            .or_else(|| first_relative_by_extension(model_dir, files, "ckpt")),
        ModelFormat::CoreMl => first_relative_by_extension(model_dir, files, "mlmodel"),
        ModelFormat::Unknown => None,
    }
}

fn first_gguf_model_path(model_dir: &Path, files: &[PathBuf]) -> Option<String> {
    let mut candidates = files
        .iter()
        .filter(|path| extension_eq(path, "gguf"))
        .filter_map(|path| relative_string(model_dir, path).map(|rel| (gguf_priority(&rel), rel)))
        .collect::<Vec<_>>();
    candidates.sort_by(|(left_priority, left), (right_priority, right)| {
        left_priority
            .cmp(right_priority)
            .then_with(|| left.cmp(right))
    });
    candidates.into_iter().map(|(_, path)| path).next()
}

fn gguf_priority(path: &str) -> usize {
    let lower = path.to_ascii_lowercase();
    [
        "q4_k_m", "q5_k_m", "q4_k_s", "q5_k_s", "q6_k", "q8_0", "q3_k_m", "q3_k_l", "q3_k_s",
        "q4_0", "q5_0", "q2_k",
    ]
    .iter()
    .position(|quant| lower.contains(quant))
    .unwrap_or(usize::MAX)
}

fn first_relative_by_extension(model_dir: &Path, files: &[PathBuf], ext: &str) -> Option<String> {
    files
        .iter()
        .find(|path| extension_eq(path, ext))
        .and_then(|path| relative_string(model_dir, path))
}

fn first_relative_by_name(model_dir: &Path, files: &[PathBuf], name: &str) -> Option<String> {
    files
        .iter()
        .find(|path| path.file_name().and_then(OsStr::to_str) == Some(name))
        .and_then(|path| relative_string(model_dir, path))
}

fn relative_string(root: &Path, path: &Path) -> Option<String> {
    path.strip_prefix(root)
        .ok()
        .map(|path| path.to_string_lossy().replace('\\', "/"))
}

fn normalize_model_file_path(file: &str) -> Result<String> {
    let mut path = file.trim().replace('\\', "/");
    while let Some(rest) = path.strip_prefix("./") {
        path = rest.to_string();
    }
    if path.is_empty() {
        bail!("model file cannot be empty");
    }
    if path.starts_with('/') || path.split('/').any(|part| part.is_empty() || part == "..") {
        bail!("model file must be a relative path inside the installed model files directory");
    }
    if !path.starts_with("files/") {
        path = format!("files/{path}");
    }
    Ok(path)
}

fn validate_selected_model_file(format: &ModelFormat, path: &str) -> Result<()> {
    let extension = Path::new(path)
        .extension()
        .and_then(OsStr::to_str)
        .unwrap_or_default();
    let valid = match format {
        ModelFormat::Gguf => extension.eq_ignore_ascii_case("gguf"),
        ModelFormat::SafeTensors => extension.eq_ignore_ascii_case("safetensors"),
        ModelFormat::PyTorch => ["pt", "pth", "bin"]
            .iter()
            .any(|candidate| extension.eq_ignore_ascii_case(candidate)),
        ModelFormat::Onnx => extension.eq_ignore_ascii_case("onnx"),
        ModelFormat::Mlx => ["npz", "safetensors"]
            .iter()
            .any(|candidate| extension.eq_ignore_ascii_case(candidate)),
        ModelFormat::TensorRt => ["engine", "plan"]
            .iter()
            .any(|candidate| extension.eq_ignore_ascii_case(candidate)),
        ModelFormat::OpenVino => extension.eq_ignore_ascii_case("xml"),
        ModelFormat::TensorFlow => ["pb", "ckpt"]
            .iter()
            .any(|candidate| extension.eq_ignore_ascii_case(candidate)),
        ModelFormat::CoreMl => ["mlmodel", "mlpackage"]
            .iter()
            .any(|candidate| extension.eq_ignore_ascii_case(candidate)),
        ModelFormat::Unknown => true,
    };
    if !valid {
        bail!("file '{path}' is not a valid {:?} model file", format);
    }
    Ok(())
}

fn detect_architecture(
    model_dir: &Path,
    format: &ModelFormat,
    model_path: Option<&str>,
    config_path: Option<&str>,
) -> Option<String> {
    match format {
        ModelFormat::Gguf => model_path.and_then(|path| {
            detect_gguf_architecture(&model_dir.join(path))
                .ok()
                .flatten()
        }),
        ModelFormat::SafeTensors => config_path.and_then(|path| {
            detect_config_architecture(&model_dir.join(path))
                .ok()
                .flatten()
        }),
        ModelFormat::PyTorch
        | ModelFormat::Onnx
        | ModelFormat::Mlx
        | ModelFormat::TensorRt
        | ModelFormat::OpenVino
        | ModelFormat::TensorFlow
        | ModelFormat::CoreMl => config_path.and_then(|path| {
            detect_config_architecture(&model_dir.join(path))
                .ok()
                .flatten()
        }),
        ModelFormat::Unknown => None,
    }
}

fn detect_gguf_architecture(path: &Path) -> Result<Option<String>> {
    let mut file = fs::File::open(path)?;
    let content = gguf_file::Content::read(&mut file)?;
    Ok(content
        .metadata
        .get("general.architecture")
        .and_then(|value| value.to_string().ok())
        .cloned())
}

fn detect_config_architecture(path: &Path) -> Result<Option<String>> {
    let data = fs::read_to_string(path)?;
    let value: Value = serde_json::from_str(&data)?;
    if let Some(model_type) = value.get("model_type").and_then(Value::as_str) {
        return Ok(Some(model_type.to_string()));
    }
    Ok(value
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(Value::as_str)
        .map(ToString::to_string))
}

fn read_manifest(path: &Path) -> Result<ModelManifest> {
    let data = fs::read_to_string(path)?;
    serde_json::from_str(&data).with_context(|| format!("invalid manifest {}", path.display()))
}

fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut file = fs::File::create(path)?;
    let data = serde_json::to_vec_pretty(value)?;
    file.write_all(&data)?;
    file.write_all(b"\n")?;
    Ok(())
}

pub fn unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_copies_files_and_writes_manifest() {
        let tmp = test_dir("store-import");
        let source = tmp.join("source");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("tokenizer.json"), "{}").unwrap();
        fs::write(
            source.join("config.json"),
            r#"{"model_type":"llama","architectures":["LlamaForCausalLM"]}"#,
        )
        .unwrap();
        fs::write(source.join("model.safetensors"), b"fake").unwrap();

        let store = ModelStore::resolve(Some(tmp.join("store"))).unwrap();
        let manifest = store.import_path(&source, "test-model").unwrap();

        assert_eq!(manifest.id, "test-model");
        assert_eq!(manifest.format, ModelFormat::SafeTensors);
        assert_eq!(manifest.architecture.as_deref(), Some("llama"));
        assert!(store.model_dir("test-model").join(MANIFEST_FILE).is_file());
        assert!(store.list().unwrap().iter().any(|m| m.id == "test-model"));

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn detects_common_model_formats() {
        let tmp = test_dir("format-detection");
        let store = ModelStore::resolve(Some(tmp.join("store"))).unwrap();

        assert_import_format(
            &store,
            &tmp,
            "safetensors",
            &[("model.safetensors", b"safe")],
            ModelFormat::SafeTensors,
        );
        assert_import_format(
            &store,
            &tmp,
            "gguf",
            &[("model.gguf", b"gguf")],
            ModelFormat::Gguf,
        );
        assert_import_format(
            &store,
            &tmp,
            "pytorch-pt",
            &[("model.pt", b"pt")],
            ModelFormat::PyTorch,
        );
        assert_import_format(
            &store,
            &tmp,
            "pytorch-pth",
            &[("model.pth", b"pth")],
            ModelFormat::PyTorch,
        );
        assert_import_format(
            &store,
            &tmp,
            "pytorch-bin",
            &[("pytorch_model.bin", b"bin")],
            ModelFormat::PyTorch,
        );
        assert_import_format(
            &store,
            &tmp,
            "onnx",
            &[("model.onnx", b"onnx")],
            ModelFormat::Onnx,
        );
        assert_import_format(
            &store,
            &tmp,
            "tensorrt-engine",
            &[("model.engine", b"engine")],
            ModelFormat::TensorRt,
        );
        assert_import_format(
            &store,
            &tmp,
            "tensorrt-plan",
            &[("model.plan", b"plan")],
            ModelFormat::TensorRt,
        );
        assert_import_format(
            &store,
            &tmp,
            "openvino",
            &[("model.xml", b"<xml/>"), ("model.bin", b"weights")],
            ModelFormat::OpenVino,
        );
        assert_import_format(
            &store,
            &tmp,
            "tensorflow-pb",
            &[("saved_model.pb", b"pb")],
            ModelFormat::TensorFlow,
        );
        assert_import_format(
            &store,
            &tmp,
            "tensorflow-ckpt",
            &[("model.ckpt", b"ckpt")],
            ModelFormat::TensorFlow,
        );
        assert_import_format(
            &store,
            &tmp,
            "coreml-mlmodel",
            &[("model.mlmodel", b"coreml")],
            ModelFormat::CoreMl,
        );
        assert_import_format(
            &store,
            &tmp,
            "coreml-mlpackage",
            &[("model.mlpackage/Manifest.json", b"{}")],
            ModelFormat::CoreMl,
        );
        assert_import_format(
            &store,
            &tmp,
            "mlx",
            &[("weights.npz", b"npz")],
            ModelFormat::Mlx,
        );

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn gguf_manifest_prefers_balanced_quant_when_multiple_files_exist() {
        let tmp = test_dir("gguf-quant-selection");
        let source = tmp.join("source");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("tinyllama.Q2_K.gguf"), b"q2").unwrap();
        fs::write(source.join("tinyllama.Q4_K_M.gguf"), b"q4").unwrap();
        fs::write(source.join("tinyllama.Q5_K_M.gguf"), b"q5").unwrap();

        let store = ModelStore::resolve(Some(tmp.join("store"))).unwrap();
        let manifest = store.import_path(&source, "tiny").unwrap();

        assert_eq!(
            manifest.model_path.as_deref(),
            Some("files/tinyllama.Q4_K_M.gguf")
        );

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn set_model_file_updates_manifest_selection() {
        let tmp = test_dir("set-model-file");
        let source = tmp.join("source");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("tinyllama.Q2_K.gguf"), b"q2").unwrap();
        fs::write(source.join("tinyllama.Q4_K_M.gguf"), b"q4").unwrap();

        let store = ModelStore::resolve(Some(tmp.join("store"))).unwrap();
        store.import_path(&source, "tiny").unwrap();
        let manifest = store.set_model_file("tiny", "tinyllama.Q2_K.gguf").unwrap();
        assert_eq!(
            manifest.model_path.as_deref(),
            Some("files/tinyllama.Q2_K.gguf")
        );

        let persisted = store.get("tiny").unwrap();
        assert_eq!(
            persisted.model_path.as_deref(),
            Some("files/tinyllama.Q2_K.gguf")
        );

        let _ = fs::remove_dir_all(tmp);
    }

    fn assert_import_format(
        store: &ModelStore,
        tmp: &Path,
        id: &str,
        files: &[(&str, &[u8])],
        format: ModelFormat,
    ) {
        let source = tmp.join(format!("source-{id}"));
        fs::create_dir_all(&source).unwrap();
        for (name, data) in files {
            let path = source.join(name);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(path, data).unwrap();
        }
        let manifest = store.import_path(&source, id).unwrap();
        assert_eq!(manifest.format, format);
    }

    fn test_dir(name: &str) -> PathBuf {
        let path = env::temp_dir().join(format!(
            "werk1112-{name}-{}-{}",
            std::process::id(),
            unix_ts()
        ));
        if path.exists() {
            fs::remove_dir_all(&path).unwrap();
        }
        path
    }
}
