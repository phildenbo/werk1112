use anyhow::{Context, Result, anyhow, bail};
use candle_core::quantized::gguf_file;
use crc32fast::Hasher;
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{
    collections::VecDeque,
    env,
    ffi::OsStr,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
    sync::mpsc,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub const MANIFEST_FILE: &str = "manifest.json";

#[derive(Debug, Clone)]
pub struct ModelStore {
    home: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HuggingFaceAuthStatus {
    pub source: Option<String>,
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
            Self::Gguf => "llama-server",
            Self::SafeTensors => "onnxruntime",
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
                "implemented via persistent llama.cpp server; Candle is legacy/fallback for selected architectures"
            }
            Self::SafeTensors => {
                "optimized artifacts through ONNX Runtime when available; Candle is compatibility fallback"
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Onnx,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactStatus {
    Ready,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArtifact {
    pub kind: ArtifactKind,
    pub path: String,
    pub status: ArtifactStatus,
    pub created_unix: u64,
    pub detail: Option<String>,
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
    #[serde(default)]
    pub artifacts: Vec<ModelArtifact>,
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
    LfsStarted {
        file: Option<String>,
        total_bytes: Option<u64>,
    },
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

    pub fn huggingface_auth_status(&self) -> Result<HuggingFaceAuthStatus> {
        let source = huggingface_token_with_source(&self.huggingface_token_path())?
            .map(|(_, source)| source);
        Ok(HuggingFaceAuthStatus { source })
    }

    pub fn save_huggingface_token(&self, token: &str) -> Result<PathBuf> {
        let token = token.trim();
        if token.is_empty() {
            bail!("Hugging Face token cannot be empty");
        }

        let path = self.huggingface_token_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create Hugging Face auth directory {}",
                    parent.display()
                )
            })?;
        }
        fs::write(&path, format!("{token}\n"))
            .with_context(|| format!("failed to write Hugging Face token {}", path.display()))?;
        restrict_file_permissions(&path)?;
        Ok(path)
    }

    pub fn delete_huggingface_token(&self) -> Result<bool> {
        let path = self.huggingface_token_path();
        if !path.exists() {
            return Ok(false);
        }
        fs::remove_file(&path)
            .with_context(|| format!("failed to remove Hugging Face token {}", path.display()))?;
        Ok(true)
    }

    pub fn huggingface_token_path(&self) -> PathBuf {
        self.home.join("auth").join("huggingface.token")
    }

    fn huggingface_token(&self) -> Result<Option<String>> {
        Ok(huggingface_token_with_source(&self.huggingface_token_path())?.map(|(token, _)| token))
    }

    pub fn huggingface_http_token(&self) -> Result<Option<String>> {
        self.huggingface_token()
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

    pub fn artifacts_dir(&self, id: &str) -> PathBuf {
        self.model_dir(id).join("artifacts")
    }

    pub fn onnx_artifact_dir(&self, id: &str) -> PathBuf {
        self.artifacts_dir(id).join("onnx")
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

        let url = format!("https://huggingface.co/{repo}");
        let auth_token = self.huggingface_token()?;
        ensure_huggingface_repo_reachable(repo, &url, auth_token.as_deref())?;

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

        progress(PullProgress::Started { url: url.clone() });

        let mut clone_command = Command::new("git");
        clone_command
            .env("GIT_LFS_SKIP_SMUDGE", "1")
            .args(["clone", "--progress", "--depth", "1", &url])
            .arg(&tmp);
        configure_huggingface_git_auth(&mut clone_command, auth_token.as_deref());
        run_git_with_progress(
            &mut clone_command,
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

        let explicit_file = file.map(|file| resolve_pull_file(&tmp, file)).transpose()?;
        let auto_file = if explicit_file.is_none() {
            default_lfs_include_file(&tmp)?
        } else {
            None
        };
        let include_file = explicit_file.or(auto_file);

        if tmp.join(".gitattributes").is_file() {
            let mut lfs_install_command = Command::new("git");
            lfs_install_command
                .arg("-C")
                .arg(&tmp)
                .args(["lfs", "install", "--local"]);
            configure_huggingface_git_auth(&mut lfs_install_command, auth_token.as_deref());
            run_git_with_progress(
                &mut lfs_install_command,
                "git lfs install --local failed; install git-lfs and run `git lfs install`",
                None,
                None,
                |_| {},
            )?;

            let total_bytes = lfs_pointer_total(&tmp, include_file.as_deref())?;
            progress(PullProgress::LfsStarted {
                file: include_file.clone(),
                total_bytes,
            });
            let mut lfs_command = Command::new("git");
            lfs_command
                .env("GIT_LFS_FORCE_PROGRESS", "1")
                .arg("-C")
                .arg(&tmp)
                .args(["lfs", "pull"]);
            configure_huggingface_git_auth(&mut lfs_command, auth_token.as_deref());
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

            let mut lfs_checkout_command = Command::new("git");
            lfs_checkout_command
                .arg("-C")
                .arg(&tmp)
                .args(["lfs", "checkout"]);
            configure_huggingface_git_auth(&mut lfs_checkout_command, auth_token.as_deref());
            run_git_with_progress(
                &mut lfs_checkout_command,
                "git lfs checkout failed after pull",
                None,
                None,
                |_| {},
            )?;
            progress(PullProgress::LfsFinished);
        }

        if let Some(include_file) = include_file.as_deref() {
            ensure_lfs_file_downloaded(&tmp, include_file)?;
        }
        let import_tmp = tmp.with_file_name(format!(
            "{}-import",
            tmp.file_name()
                .and_then(OsStr::to_str)
                .unwrap_or("pull-import")
        ));
        let import_source = if let Some(include_file) = include_file.as_deref() {
            prepare_included_file_import_tree(&tmp, include_file, &import_tmp)?;
            import_tmp.as_path()
        } else {
            tmp.as_path()
        };
        ensure_no_lfs_pointers_remaining(import_source)?;

        progress(PullProgress::Importing);

        let manifest = self.import_path_with_source(
            import_source,
            id,
            ModelSource::HuggingFace {
                repo: repo.to_string(),
            },
        );
        let _ = fs::remove_dir_all(&tmp);
        let _ = fs::remove_dir_all(&import_tmp);
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

    pub fn remove(&self, id: &str) -> Result<ModelManifest> {
        self.ensure()?;
        validate_id(id)?;
        let manifest = self.get(id)?;
        let dir = self.model_dir(&manifest.id);
        if !dir.is_dir() {
            bail!(
                "model '{}' is not installed at {}",
                manifest.id,
                dir.display()
            );
        }
        fs::remove_dir_all(&dir).with_context(|| {
            format!(
                "failed to remove model '{}' at {}",
                manifest.id,
                dir.display()
            )
        })?;
        Ok(manifest)
    }

    pub fn list_artifacts(&self, id: &str) -> Result<Vec<ModelArtifact>> {
        Ok(self.get(id)?.artifacts)
    }

    pub fn ready_onnx_artifact(&self, manifest: &ModelManifest) -> Option<ModelArtifact> {
        manifest
            .artifacts
            .iter()
            .find(|artifact| {
                artifact.kind == ArtifactKind::Onnx && artifact.status == ArtifactStatus::Ready
            })
            .cloned()
            .filter(|artifact| self.model_dir(&manifest.id).join(&artifact.path).is_dir())
    }

    pub fn can_build_onnx_artifact(&self, manifest: &ModelManifest) -> bool {
        manifest.format == ModelFormat::SafeTensors
            && is_supported_onnx_architecture(manifest.architecture.as_deref())
            && find_onnx_exporter().is_some()
    }

    pub fn build_onnx_artifact(&self, id: &str, rebuild: bool) -> Result<ModelArtifact> {
        self.ensure()?;
        let mut manifest = self.get(id)?;
        if manifest.format != ModelFormat::SafeTensors {
            bail!(
                "ONNX artifacts can only be built for safetensors models; '{}' is {:?}",
                manifest.id,
                manifest.format
            );
        }
        let architecture = manifest.architecture.as_deref().unwrap_or("unknown");
        if !is_supported_onnx_architecture(Some(architecture)) {
            bail!("ONNX artifact generation is not supported for architecture '{architecture}'");
        }

        let artifact_dir = self.onnx_artifact_dir(&manifest.id);
        let artifact_rel = "artifacts/onnx".to_string();
        if artifact_dir.is_dir() && !rebuild && onnx_files_exist(&artifact_dir)? {
            let artifact = ModelArtifact {
                kind: ArtifactKind::Onnx,
                path: artifact_rel,
                status: ArtifactStatus::Ready,
                created_unix: unix_ts(),
                detail: Some("existing ONNX artifact".to_string()),
            };
            upsert_artifact(&mut manifest, artifact.clone());
            self.write_manifest(&manifest)?;
            return Ok(artifact);
        }
        if rebuild && artifact_dir.exists() {
            fs::remove_dir_all(&artifact_dir).with_context(|| {
                format!("failed to remove ONNX artifact {}", artifact_dir.display())
            })?;
        }
        fs::create_dir_all(&artifact_dir)?;

        let exporter = find_onnx_exporter().ok_or_else(|| {
            anyhow!("no ONNX exporter found; install optimum-cli or set WERK_ONNX_EXPORTER")
        })?;
        let source_dir = self.model_dir(&manifest.id).join("files");
        let result = run_onnx_exporter(&exporter, &source_dir, &artifact_dir);
        match result {
            Ok(()) if onnx_files_exist(&artifact_dir)? => {
                let artifact = ModelArtifact {
                    kind: ArtifactKind::Onnx,
                    path: artifact_rel,
                    status: ArtifactStatus::Ready,
                    created_unix: unix_ts(),
                    detail: Some(format!("built with {}", exporter.label())),
                };
                write_json_pretty(&artifact_dir.join("artifact.json"), &artifact)?;
                upsert_artifact(&mut manifest, artifact.clone());
                self.write_manifest(&manifest)?;
                Ok(artifact)
            }
            Ok(()) => {
                let detail = "ONNX exporter completed but did not create an .onnx file".to_string();
                let artifact = failed_onnx_artifact(artifact_rel, detail.clone());
                write_json_pretty(&artifact_dir.join("artifact.json"), &artifact)?;
                upsert_artifact(&mut manifest, artifact);
                self.write_manifest(&manifest)?;
                bail!("{detail}");
            }
            Err(err) => {
                let detail = err.to_string();
                let artifact = failed_onnx_artifact(artifact_rel, detail.clone());
                let _ = write_json_pretty(&artifact_dir.join("artifact.json"), &artifact);
                upsert_artifact(&mut manifest, artifact);
                self.write_manifest(&manifest)?;
                bail!("ONNX artifact generation failed: {detail}");
            }
        }
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

        let selected_format = detect_format_for_model_path(&selected_path);
        validate_selected_model_file(&selected_format, &selected_path)?;
        manifest.format = selected_format;
        manifest.model_path = Some(selected_path);
        manifest.architecture = detect_architecture(
            &self.model_dir(&manifest.id),
            &manifest.format,
            manifest.model_path.as_deref(),
            manifest.config_path.as_deref(),
        );
        manifest.backend = manifest.format.backend_hint().to_string();
        write_json_pretty(&self.model_dir(&manifest.id).join(MANIFEST_FILE), &manifest)?;
        Ok(manifest)
    }

    pub fn write_manifest(&self, manifest: &ModelManifest) -> Result<()> {
        write_json_pretty(&self.model_dir(&manifest.id).join(MANIFEST_FILE), manifest)
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
            artifacts: Vec::new(),
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

fn ensure_huggingface_repo_reachable(repo: &str, url: &str, token: Option<&str>) -> Result<()> {
    let metadata = fetch_huggingface_model_metadata(repo, token);
    if metadata.as_ref().map(|metadata| metadata.gated) == Some(true) && token.is_none() {
        bail!("{}", hf_gated_repo_message(repo, url, false, ""));
    }

    let mut command = Command::new("git");
    configure_noninteractive_git(&mut command).args(["ls-remote", "--exit-code", url, "HEAD"]);
    configure_huggingface_git_auth(&mut command, token);
    let output = run_command_with_timeout(&mut command, Duration::from_secs(20))
        .context("failed to execute git; install git and git-lfs for Hugging Face pulls")?;

    if output.timed_out {
        bail!("timed out checking Hugging Face repo {url}; check your network and retry");
    }
    let status = output
        .status
        .context("git repo check exited without a status")?;
    if !status.success() {
        let detail = if output.stderr.trim().is_empty() {
            output.stdout.trim()
        } else {
            output.stderr.trim()
        };
        bail!(
            "{}",
            hf_repo_unreachable_message(repo, url, detail, metadata.as_ref(), token.is_some())
        );
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HuggingFaceModelMetadata {
    gated: bool,
}

fn fetch_huggingface_model_metadata(
    repo: &str,
    token: Option<&str>,
) -> Option<HuggingFaceModelMetadata> {
    let _ = token;
    let api_url = format!("https://huggingface.co/api/models/{repo}");
    let mut command = Command::new("curl");
    command.args(["-fsSL", "--max-time", "8", "-A", "werk1112", &api_url]);
    let output = run_command_with_timeout(&mut command, Duration::from_secs(10)).ok()?;
    if output.timed_out
        || !output
            .status
            .map(|status| status.success())
            .unwrap_or(false)
    {
        return None;
    }
    let value = serde_json::from_str::<Value>(&output.stdout).ok()?;
    Some(parse_huggingface_model_metadata(&value))
}

fn parse_huggingface_model_metadata(value: &Value) -> HuggingFaceModelMetadata {
    HuggingFaceModelMetadata {
        gated: value_is_gated(value.get("gated")),
    }
}

fn value_is_gated(value: Option<&Value>) -> bool {
    match value {
        Some(Value::Bool(gated)) => *gated,
        Some(Value::String(gated)) => !matches!(gated.as_str(), "" | "false" | "False" | "none"),
        _ => false,
    }
}

fn configure_noninteractive_git(command: &mut Command) -> &mut Command {
    command
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_ASKPASS", "")
        .env("GCM_INTERACTIVE", "never")
}

fn configure_huggingface_git_auth(command: &mut Command, token: Option<&str>) {
    let Some(token) = token.map(str::trim).filter(|token| !token.is_empty()) else {
        return;
    };
    let basic_auth = base64_encode(format!("hf_user:{token}").as_bytes());
    command
        .env("GIT_CONFIG_COUNT", "1")
        .env(
            "GIT_CONFIG_KEY_0",
            "http.https://huggingface.co/.extraheader",
        )
        .env(
            "GIT_CONFIG_VALUE_0",
            format!("Authorization: Basic {basic_auth}"),
        );
}

fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut encoded = String::with_capacity(bytes.len().div_ceil(3) * 4);

    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);

        encoded.push(TABLE[(b0 >> 2) as usize] as char);
        encoded.push(TABLE[(((b0 & 0b0000_0011) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            encoded.push(TABLE[(((b1 & 0b0000_1111) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            encoded.push('=');
        }
        if chunk.len() > 2 {
            encoded.push(TABLE[(b2 & 0b0011_1111) as usize] as char);
        } else {
            encoded.push('=');
        }
    }

    encoded
}

fn huggingface_token_with_source(store_token_path: &Path) -> Result<Option<(String, String)>> {
    for name in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(token) = env::var(name)
            && let Some(token) = normalize_huggingface_token(&token)
        {
            return Ok(Some((token, format!("environment variable {name}"))));
        }
    }

    if let Some(token) = read_token_file(store_token_path)? {
        return Ok(Some((
            token,
            format!("Werk token file {}", store_token_path.display()),
        )));
    }

    if let Some(path) = huggingface_cli_token_path()
        && let Some(token) = read_token_file(&path)?
    {
        return Ok(Some((
            token,
            format!("Hugging Face CLI token file {}", path.display()),
        )));
    }

    Ok(None)
}

fn normalize_huggingface_token(token: &str) -> Option<String> {
    let token = token.trim();
    (!token.is_empty()).then(|| token.to_string())
}

fn read_token_file(path: &Path) -> Result<Option<String>> {
    if !path.is_file() {
        return Ok(None);
    }
    let token = fs::read_to_string(path)
        .with_context(|| format!("failed to read Hugging Face token {}", path.display()))?;
    Ok(normalize_huggingface_token(&token))
}

fn huggingface_cli_token_path() -> Option<PathBuf> {
    if let Ok(hf_home) = env::var("HF_HOME")
        && !hf_home.trim().is_empty()
    {
        return Some(PathBuf::from(hf_home).join("token"));
    }
    if let Ok(home) = env::var("HOME")
        && !home.trim().is_empty()
    {
        return Some(PathBuf::from(home).join(".cache/huggingface/token"));
    }
    None
}

fn restrict_file_permissions(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
            .with_context(|| format!("failed to restrict permissions on {}", path.display()))?;
    }
    let _ = path;
    Ok(())
}

struct TimedCommandOutput {
    status: Option<ExitStatus>,
    stdout: String,
    stderr: String,
    timed_out: bool,
}

fn run_command_with_timeout(
    command: &mut Command,
    timeout: Duration,
) -> Result<TimedCommandOutput> {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let started = Instant::now();

    loop {
        if let Some(status) = child.try_wait()? {
            let (stdout, stderr) = read_child_output(&mut child)?;
            return Ok(TimedCommandOutput {
                status: Some(status),
                stdout,
                stderr,
                timed_out: false,
            });
        }

        if started.elapsed() >= timeout {
            let _ = child.kill();
            let status = child.wait().ok();
            let (stdout, stderr) = read_child_output(&mut child)?;
            return Ok(TimedCommandOutput {
                status,
                stdout,
                stderr,
                timed_out: true,
            });
        }

        thread::sleep(Duration::from_millis(100));
    }
}

fn read_child_output(child: &mut std::process::Child) -> Result<(String, String)> {
    let mut stdout = String::new();
    if let Some(mut pipe) = child.stdout.take() {
        pipe.read_to_string(&mut stdout)?;
    }

    let mut stderr = String::new();
    if let Some(mut pipe) = child.stderr.take() {
        pipe.read_to_string(&mut stderr)?;
    }

    Ok((stdout, stderr))
}

fn hf_repo_unreachable_message(
    repo: &str,
    url: &str,
    detail: &str,
    metadata: Option<&HuggingFaceModelMetadata>,
    token_present: bool,
) -> String {
    if metadata.map(|metadata| metadata.gated) == Some(true) {
        return hf_gated_repo_message(repo, url, token_present, detail);
    }

    let mut message = format!(
        "Hugging Face repo not found or inaccessible: {repo} ({url}). Check the repo id and your access."
    );
    message.push_str(
        " If this is a gated model, Werk cannot accept the conditions for you through the Hugging Face API. Open the model page in your browser, accept the conditions, then run `werk auth huggingface login` or set HF_TOKEN.",
    );
    if let Some(rest) = repo.strip_prefix("icrosoft/") {
        message.push_str(&format!(" Did you mean `microsoft/{rest}`?"));
    }
    if !detail.trim().is_empty() {
        message.push_str(&format!(" git said: {}", detail.trim()));
    }
    message
}

fn hf_gated_repo_message(repo: &str, url: &str, token_present: bool, detail: &str) -> String {
    let mut message = format!(
        "Hugging Face gated model requires browser agreement: {repo} ({url}). Werk cannot accept model conditions for you through the Hugging Face API."
    );
    if token_present {
        message.push_str(" Your token is configured, but access still failed; open the model page, accept the conditions with the same Hugging Face account, then retry.");
    } else {
        message.push_str(" Open the model page, accept the conditions, then run `werk auth huggingface login` or set HF_TOKEN and retry.");
    }
    if !detail.trim().is_empty() {
        message.push_str(&format!(" git said: {}", detail.trim()));
    }
    message
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
    configure_noninteractive_git(command);
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to execute git; install git and git-lfs for Hugging Face pulls")?;

    let (line_tx, line_rx) = mpsc::channel();
    let stdout_reader = child.stdout.take().map(|stdout| {
        let line_tx = line_tx.clone();
        thread::spawn(move || -> Result<VecDeque<String>, String> {
            let mut output_tail = VecDeque::new();
            read_git_progress(stdout, &mut output_tail, &mut |line| {
                let _ = line_tx.send(line);
            })
            .map_err(|err| err.to_string())?;
            Ok(output_tail)
        })
    });
    let stderr_reader = child.stderr.take().map(|stderr| {
        thread::spawn(move || -> Result<VecDeque<String>, String> {
            let mut output_tail = VecDeque::new();
            read_git_progress(stderr, &mut output_tail, &mut |line| {
                let _ = line_tx.send(line);
            })
            .map_err(|err| err.to_string())?;
            Ok(output_tail)
        })
    });

    let mut output_tail = VecDeque::<String>::new();
    let baseline_bytes = watch_path.and_then(|path| dir_size(path).ok()).unwrap_or(0);
    let mut last_stats_at = Instant::now();
    let mut last_bytes = 0u64;
    let mut saw_parsed_lfs_progress = false;

    loop {
        while let Ok(line) = line_rx.try_recv() {
            push_tail(&mut output_tail, line.clone());
            if let Some(stats) = parse_git_lfs_progress_line(&line, total_bytes) {
                saw_parsed_lfs_progress = true;
                last_stats_at = Instant::now();
                progress(GitCommandProgress::Stats {
                    bytes: stats.progress_bytes(),
                    total_bytes: stats.total_bytes,
                    bytes_per_second: stats.bytes_per_second,
                });
            } else {
                progress(GitCommandProgress::Line(line));
            }
        }

        if let Some(status) = child.try_wait().context("failed to wait for git command")? {
            while let Ok(line) = line_rx.try_recv() {
                push_tail(&mut output_tail, line.clone());
                if let Some(stats) = parse_git_lfs_progress_line(&line, total_bytes) {
                    progress(GitCommandProgress::Stats {
                        bytes: stats.progress_bytes(),
                        total_bytes: stats.total_bytes,
                        bytes_per_second: stats.bytes_per_second,
                    });
                } else {
                    progress(GitCommandProgress::Line(line));
                }
            }

            if let Some(reader) = stdout_reader {
                match reader.join() {
                    Ok(Ok(reader_tail)) => {
                        for line in reader_tail {
                            push_tail(&mut output_tail, line);
                        }
                    }
                    Ok(Err(err)) => bail!("{error_context}: {err}"),
                    Err(_) => bail!("{error_context}: failed to read git progress output"),
                }
            }
            if let Some(reader) = stderr_reader {
                match reader.join() {
                    Ok(Ok(reader_tail)) => {
                        for line in reader_tail {
                            push_tail(&mut output_tail, line);
                        }
                    }
                    Ok(Err(err)) => bail!("{error_context}: {err}"),
                    Err(_) => bail!("{error_context}: failed to read git progress output"),
                }
            }
            while let Ok(line) = line_rx.try_recv() {
                push_tail(&mut output_tail, line.clone());
                if let Some(stats) = parse_git_lfs_progress_line(&line, total_bytes) {
                    progress(GitCommandProgress::Stats {
                        bytes: stats.progress_bytes(),
                        total_bytes: stats.total_bytes,
                        bytes_per_second: stats.bytes_per_second,
                    });
                } else {
                    progress(GitCommandProgress::Line(line));
                }
            }

            if !status.success() {
                let output = output_tail.into_iter().collect::<Vec<_>>().join("\n");
                bail!("{error_context}: {}", output.trim());
            }

            return Ok(());
        }

        if let Some(path) = watch_path
            && !saw_parsed_lfs_progress
            && last_stats_at.elapsed() >= Duration::from_millis(750)
            && let Ok(current_bytes) = dir_size(path)
        {
            let bytes = current_bytes.saturating_sub(baseline_bytes);
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

#[derive(Debug, Clone)]
struct TransferStats {
    percent: Option<u64>,
    bytes: u64,
    total_bytes: Option<u64>,
    bytes_per_second: f64,
}

impl TransferStats {
    fn progress_bytes(&self) -> u64 {
        // Git LFS percent is object-count progress, so large single-object downloads
        // can stay at 0% while bytes are moving. Werk's bar is byte-based.
        let _object_percent = self.percent;
        self.bytes
    }
}

fn push_tail(stderr_tail: &mut VecDeque<String>, line: String) {
    stderr_tail.push_back(line);
    while stderr_tail.len() > 20 {
        stderr_tail.pop_front();
    }
}

fn parse_git_lfs_progress_line(line: &str, total_hint: Option<u64>) -> Option<TransferStats> {
    let (_, progress_text) = line.split_once("Downloading LFS objects:")?;
    let (percent_text, progress_text) = progress_text.trim_start().split_once('%')?;
    let percent = percent_text.trim().parse::<u64>().ok()?.min(100);
    let (_, byte_and_speed_text) = progress_text.split_once(',')?;
    let (bytes_text, speed_text) = byte_and_speed_text.trim().split_once('|')?;

    let bytes = parse_byte_value(bytes_text.trim())?;
    let speed_text = speed_text.trim();
    let speed_text = speed_text.strip_suffix("/s").unwrap_or(speed_text).trim();
    let bytes_per_second = parse_byte_value_f64(speed_text)?;
    let total_bytes = total_hint.or_else(|| {
        (percent > 0).then(|| {
            ((bytes as u128)
                .saturating_mul(100)
                .checked_div(percent as u128)
                .unwrap_or(0)) as u64
        })
    });

    Some(TransferStats {
        percent: Some(percent),
        bytes,
        total_bytes,
        bytes_per_second,
    })
}

fn parse_byte_value(text: &str) -> Option<u64> {
    Some(parse_byte_value_f64(text)?.round() as u64)
}

fn parse_byte_value_f64(text: &str) -> Option<f64> {
    let mut parts = text.split_whitespace();
    let value = parts.next()?.replace(',', "").parse::<f64>().ok()?;
    let unit = parts.next()?;
    if parts.next().is_some() {
        return None;
    }

    let multiplier = match unit {
        "B" => 1.0,
        "KB" => 1_000.0,
        "MB" => 1_000_000.0,
        "GB" => 1_000_000_000.0,
        "TB" => 1_000_000_000_000.0,
        "KiB" => 1024.0,
        "MiB" => 1024.0 * 1024.0,
        "GiB" => 1024.0 * 1024.0 * 1024.0,
        "TiB" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };

    Some(value * multiplier)
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
        if entry.file_name() == ".git" {
            continue;
        }
        bytes = bytes.saturating_add(dir_size(&entry.path())?);
    }
    Ok(bytes)
}

fn default_lfs_include_file(root: &Path) -> Result<Option<String>> {
    let mut files = Vec::new();
    collect_files(root, &mut files)?;
    let format = detect_format(&files);
    Ok(match format {
        ModelFormat::Gguf => first_gguf_model_path(root, &files),
        _ => None,
    })
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

fn ensure_no_lfs_pointers_remaining(root: &Path) -> Result<()> {
    ensure_no_lfs_pointers_remaining_in(root, root)
}

fn ensure_no_lfs_pointers_remaining_in(root: &Path, path: &Path) -> Result<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();

        if entry.file_name() == ".git" {
            continue;
        }

        if entry_path.is_dir() {
            ensure_no_lfs_pointers_remaining_in(root, &entry_path)?;
        } else if entry_path.is_file() && lfs_pointer_size(&entry_path)?.is_some() {
            let relative_path = relative_string(root, &entry_path)
                .unwrap_or_else(|| entry_path.display().to_string());

            bail!(
                "Git LFS download incomplete: {relative_path} is still a Git LFS pointer. Retry `werk pull` or run `git lfs pull` manually."
            );
        }
    }
    Ok(())
}

fn prepare_included_file_import_tree(root: &Path, include_file: &str, dest: &Path) -> Result<()> {
    if dest.exists() {
        fs::remove_dir_all(dest).with_context(|| {
            format!(
                "failed to remove stale filtered import directory {}",
                dest.display()
            )
        })?;
    }
    fs::create_dir_all(dest)?;

    let include_path = root.join(include_file);
    if !include_path.is_file() {
        bail!("selected file was not downloaded: {include_file}");
    }
    copy_repo_file(root, &include_path, dest)?;
    copy_included_import_metadata(root, root, dest, include_file)?;
    Ok(())
}

fn copy_included_import_metadata(
    root: &Path,
    path: &Path,
    dest: &Path,
    include_file: &str,
) -> Result<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_name() == ".git" {
            continue;
        }

        let entry_path = entry.path();
        if entry_path.is_dir() {
            copy_included_import_metadata(root, &entry_path, dest, include_file)?;
            continue;
        }
        if !entry_path.is_file() {
            continue;
        }

        let Some(relative_path) = relative_string(root, &entry_path) else {
            continue;
        };
        if relative_path == include_file {
            continue;
        }
        if should_copy_included_import_metadata(&entry_path, &relative_path)? {
            copy_repo_file(root, &entry_path, dest)?;
        }
    }

    Ok(())
}

fn should_copy_included_import_metadata(path: &Path, relative_path: &str) -> Result<bool> {
    if lfs_pointer_size(path)?.is_some() || is_likely_model_artifact_path(relative_path) {
        return Ok(false);
    }

    Ok(is_huggingface_metadata_path(relative_path))
}

fn copy_repo_file(root: &Path, source: &Path, dest_root: &Path) -> Result<()> {
    let relative_path = source.strip_prefix(root).with_context(|| {
        format!(
            "file {} is not inside repository {}",
            source.display(),
            root.display()
        )
    })?;
    let dest = dest_root.join(relative_path);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(source, &dest)
        .with_context(|| format!("copy {} -> {}", source.display(), dest.display()))?;
    Ok(())
}

fn is_huggingface_metadata_path(relative_path: &str) -> bool {
    let path = Path::new(relative_path);
    let Some(file_name) = path
        .file_name()
        .and_then(OsStr::to_str)
        .map(|name| name.to_ascii_lowercase())
    else {
        return false;
    };
    let extension = path
        .extension()
        .and_then(OsStr::to_str)
        .map(|extension| extension.to_ascii_lowercase());

    matches!(
        file_name.as_str(),
        ".gitattributes"
            | "added_tokens.json"
            | "chat_template.jinja"
            | "config.json"
            | "generation_config.json"
            | "image_processor_config.json"
            | "merges.txt"
            | "preprocessor_config.json"
            | "processor_config.json"
            | "special_tokens_map.json"
            | "spiece.model"
            | "tokenizer.json"
            | "tokenizer.model"
            | "tokenizer_config.json"
            | "vocab.json"
    ) || matches!(
        extension.as_deref(),
        Some("json" | "jinja" | "model" | "txt")
    )
}

fn is_likely_model_artifact_path(relative_path: &str) -> bool {
    let path = Path::new(relative_path);
    let extension = path
        .extension()
        .and_then(OsStr::to_str)
        .map(|extension| extension.to_ascii_lowercase());
    if matches!(
        extension.as_deref(),
        Some(
            "bin"
                | "ckpt"
                | "engine"
                | "gguf"
                | "mlmodel"
                | "npz"
                | "onnx"
                | "pb"
                | "plan"
                | "pt"
                | "pth"
                | "safetensors"
                | "tflite"
        )
    ) {
        return true;
    }

    relative_path
        .to_ascii_lowercase()
        .split('/')
        .any(|part| part.ends_with(".mlpackage"))
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
    } else if any_mlx_safetensors(files) || any_extension(files, "npz") {
        ModelFormat::Mlx
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
    } else if any_path_contains(files, "mlx") {
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

fn any_mlx_safetensors(files: &[PathBuf]) -> bool {
    files
        .iter()
        .filter(|path| extension_eq(path, "safetensors"))
        .any(|path| safetensors_declares_mlx(path).unwrap_or(false))
}

fn safetensors_declares_mlx(path: &Path) -> Result<bool> {
    let mut file = fs::File::open(path)?;
    let mut len_bytes = [0_u8; 8];
    if file.read_exact(&mut len_bytes).is_err() {
        return Ok(false);
    }

    let header_len = u64::from_le_bytes(len_bytes);
    if header_len == 0 || header_len > 16 * 1024 * 1024 {
        return Ok(false);
    }

    let mut header = vec![0_u8; header_len as usize];
    file.read_exact(&mut header)?;
    let header = serde_json::from_slice::<Value>(&header)?;
    Ok(header
        .get("__metadata__")
        .and_then(|metadata| metadata.get("format"))
        .and_then(Value::as_str)
        .map(|format| format.eq_ignore_ascii_case("mlx"))
        .unwrap_or(false))
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

fn detect_format_for_model_path(path: &str) -> ModelFormat {
    let extension = Path::new(path)
        .extension()
        .and_then(OsStr::to_str)
        .unwrap_or_default();
    if extension.eq_ignore_ascii_case("gguf") {
        ModelFormat::Gguf
    } else if extension.eq_ignore_ascii_case("safetensors") {
        ModelFormat::SafeTensors
    } else if ["pt", "pth", "bin"]
        .iter()
        .any(|candidate| extension.eq_ignore_ascii_case(candidate))
    {
        ModelFormat::PyTorch
    } else if extension.eq_ignore_ascii_case("onnx") {
        ModelFormat::Onnx
    } else if ["engine", "plan"]
        .iter()
        .any(|candidate| extension.eq_ignore_ascii_case(candidate))
    {
        ModelFormat::TensorRt
    } else if extension.eq_ignore_ascii_case("xml") {
        ModelFormat::OpenVino
    } else if ["pb", "ckpt"]
        .iter()
        .any(|candidate| extension.eq_ignore_ascii_case(candidate))
    {
        ModelFormat::TensorFlow
    } else if ["mlmodel", "mlpackage"]
        .iter()
        .any(|candidate| extension.eq_ignore_ascii_case(candidate))
    {
        ModelFormat::CoreMl
    } else if extension.eq_ignore_ascii_case("npz") {
        ModelFormat::Mlx
    } else {
        ModelFormat::Unknown
    }
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
    let mut manifest = serde_json::from_str::<ModelManifest>(&data)
        .with_context(|| format!("invalid manifest {}", path.display()))?;
    reconcile_mlx_safetensors_manifest(path, &mut manifest);
    Ok(manifest)
}

fn reconcile_mlx_safetensors_manifest(manifest_path: &Path, manifest: &mut ModelManifest) {
    if manifest.format != ModelFormat::SafeTensors {
        return;
    }
    let Some(model_path) = manifest.model_path.as_deref() else {
        return;
    };
    let Some(model_dir) = manifest_path.parent() else {
        return;
    };

    if safetensors_declares_mlx(&model_dir.join(model_path)).unwrap_or(false) {
        manifest.format = ModelFormat::Mlx;
        manifest.backend = manifest.format.backend_hint().to_string();
    }
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

#[derive(Debug, Clone)]
enum OnnxExporter {
    Env(PathBuf),
    OptimumCli(PathBuf),
    PythonModule(PathBuf),
}

impl OnnxExporter {
    fn label(&self) -> String {
        match self {
            Self::Env(path) => path.display().to_string(),
            Self::OptimumCli(path) => path.display().to_string(),
            Self::PythonModule(path) => format!("{} -m optimum.exporters.onnx", path.display()),
        }
    }
}

fn is_supported_onnx_architecture(architecture: Option<&str>) -> bool {
    let Some(architecture) = architecture else {
        return false;
    };
    let normalized = architecture
        .to_ascii_lowercase()
        .replace('-', "")
        .replace('_', "");
    matches!(
        normalized.as_str(),
        "phi3"
            | "qwen2"
            | "qwen3"
            | "gemma"
            | "gemma2"
            | "gemma3"
            | "mistral"
            | "mixtral"
            | "llama"
    )
}

fn failed_onnx_artifact(path: String, detail: String) -> ModelArtifact {
    ModelArtifact {
        kind: ArtifactKind::Onnx,
        path,
        status: ArtifactStatus::Failed,
        created_unix: unix_ts(),
        detail: Some(detail),
    }
}

fn upsert_artifact(manifest: &mut ModelManifest, artifact: ModelArtifact) {
    manifest
        .artifacts
        .retain(|existing| existing.kind != artifact.kind);
    manifest.artifacts.push(artifact);
}

fn onnx_files_exist(dir: &Path) -> Result<bool> {
    let mut files = Vec::new();
    if !dir.is_dir() {
        return Ok(false);
    }
    collect_files(dir, &mut files)?;
    Ok(files.iter().any(|path| extension_eq(path, "onnx")))
}

fn find_onnx_exporter() -> Option<OnnxExporter> {
    env::var_os("WERK_ONNX_EXPORTER")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .map(OnnxExporter::Env)
        .or_else(|| find_in_path("optimum-cli").map(OnnxExporter::OptimumCli))
        .or_else(|| {
            find_in_path("python3")
                .or_else(|| find_in_path("python"))
                .map(OnnxExporter::PythonModule)
        })
}

fn run_onnx_exporter(
    exporter: &OnnxExporter,
    source_dir: &Path,
    artifact_dir: &Path,
) -> Result<()> {
    let mut command = match exporter {
        OnnxExporter::Env(path) => {
            let mut command = Command::new(path);
            command
                .arg("--model")
                .arg(source_dir)
                .arg("--output")
                .arg(artifact_dir);
            command
        }
        OnnxExporter::OptimumCli(path) => {
            let mut command = Command::new(path);
            command
                .args([
                    "export",
                    "onnx",
                    "--task",
                    "text-generation-with-past",
                    "--model",
                ])
                .arg(source_dir)
                .arg(artifact_dir);
            command
        }
        OnnxExporter::PythonModule(path) => {
            let mut command = Command::new(path);
            command
                .args([
                    "-m",
                    "optimum.exporters.onnx",
                    "--task",
                    "text-generation-with-past",
                    "--model",
                ])
                .arg(source_dir)
                .arg(artifact_dir);
            command
        }
    };
    let output = command
        .output()
        .with_context(|| format!("failed to run ONNX exporter {}", exporter.label()))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let detail = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        output.status.to_string()
    };
    bail!("{} failed: {}", exporter.label(), detail)
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
    fn import_detects_mlx_safetensors_metadata() {
        let tmp = test_dir("store-import-mlx-safetensors");
        let source = tmp.join("source");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("config.json"), r#"{"model_type":"qwen3"}"#).unwrap();
        write_safetensors_header(
            &source.join("model.safetensors"),
            r#"{"__metadata__":{"format":"mlx"},"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        );

        let store = ModelStore::resolve(Some(tmp.join("store"))).unwrap();
        let manifest = store.import_path(&source, "mlx-model").unwrap();

        assert_eq!(manifest.format, ModelFormat::Mlx);
        assert_eq!(manifest.backend, "mlx");
        assert_eq!(
            manifest.model_path.as_deref(),
            Some("files/model.safetensors")
        );

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn get_reclassifies_legacy_mlx_safetensors_manifest() {
        let tmp = test_dir("legacy-mlx-safetensors-manifest");
        let source = tmp.join("source");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("config.json"), r#"{"model_type":"qwen3"}"#).unwrap();
        fs::write(source.join("model.safetensors"), b"fake").unwrap();

        let store = ModelStore::resolve(Some(tmp.join("store"))).unwrap();
        let manifest = store.import_path(&source, "legacy-mlx").unwrap();
        assert_eq!(manifest.format, ModelFormat::SafeTensors);

        write_safetensors_header(
            &store
                .model_dir("legacy-mlx")
                .join("files")
                .join("model.safetensors"),
            r#"{"__metadata__":{"format":"mlx"},"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        );

        let manifest = store.get("legacy-mlx").unwrap();
        assert_eq!(manifest.format, ModelFormat::Mlx);
        assert_eq!(manifest.backend, "mlx");

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn remove_deletes_managed_model_directory() {
        let tmp = test_dir("store-remove");
        let source = tmp.join("source");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("model.gguf"), b"gguf").unwrap();

        let store = ModelStore::resolve(Some(tmp.join("store"))).unwrap();
        store.import_path(&source, "test-model").unwrap();
        let model_dir = store.model_dir("test-model");
        assert!(model_dir.is_dir());

        let removed = store.remove("test-model").unwrap();
        assert_eq!(removed.id, "test-model");
        assert!(!model_dir.exists());
        assert!(store.get("test-model").is_err());
        assert!(source.join("model.gguf").is_file());

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
    fn default_lfs_include_file_prefers_balanced_gguf() {
        let tmp = test_dir("gguf-pull-default-file");
        let source = tmp.join("source");
        fs::create_dir_all(&source).unwrap();
        write_lfs_pointer(&source.join("tiny.Q2_K.gguf"), 2);
        write_lfs_pointer(&source.join("tiny.Q4_K_M.gguf"), 4);
        write_lfs_pointer(&source.join("tiny.Q5_K_M.gguf"), 5);
        fs::write(source.join("README.md"), b"readme").unwrap();

        assert_eq!(
            default_lfs_include_file(&source).unwrap().as_deref(),
            Some("tiny.Q4_K_M.gguf")
        );

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn dir_size_skips_git_metadata_for_transfer_progress() {
        let tmp = test_dir("transfer-dir-size");
        let source = tmp.join("source");
        fs::create_dir_all(source.join(".git/lfs/objects")).unwrap();
        fs::write(source.join("model.gguf"), vec![0_u8; 32]).unwrap();
        fs::write(source.join(".git/lfs/objects/object"), vec![0_u8; 1024]).unwrap();

        assert_eq!(dir_size(&source).unwrap(), 32);

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn parses_git_lfs_progress_with_mb_units() {
        let stats = parse_git_lfs_progress_line(
            "Downloading LFS objects:  50% (1/2), 28 MB | 3.2 MB/s",
            Some(2_u64 * 1024 * 1024 * 1024),
        )
        .unwrap();

        assert_eq!(stats.percent, Some(50));
        assert_eq!(stats.bytes, 28_000_000);
        assert_eq!(stats.total_bytes, Some(2_u64 * 1024 * 1024 * 1024));
        assert_eq!(stats.bytes_per_second.round() as u64, 3_200_000);
        assert_eq!(stats.progress_bytes(), 28_000_000);
    }

    #[test]
    fn parses_git_lfs_progress_with_gib_units() {
        let stats = parse_git_lfs_progress_line(
            "Downloading LFS objects: 100% (2/2), 2.00 GiB | 12.3 MiB/s",
            None,
        )
        .unwrap();

        assert_eq!(stats.percent, Some(100));
        assert_eq!(stats.bytes, 2_u64 * 1024 * 1024 * 1024);
        assert_eq!(stats.total_bytes, Some(2_u64 * 1024 * 1024 * 1024));
        assert_eq!(
            stats.bytes_per_second.round() as u64,
            (12.3_f64 * 1024.0_f64 * 1024.0_f64).round() as u64
        );
    }

    #[test]
    fn parses_git_lfs_zero_percent_progress() {
        let stats = parse_git_lfs_progress_line(
            "Downloading LFS objects:   0% (0/1), 65 MB | 8.0 MB/s",
            Some(2_u64 * 1024 * 1024 * 1024),
        )
        .unwrap();

        assert_eq!(stats.percent, Some(0));
        assert_eq!(stats.bytes, 65_000_000);
        assert_eq!(stats.total_bytes, Some(2_u64 * 1024 * 1024 * 1024));
        assert_eq!(stats.bytes_per_second.round() as u64, 8_000_000);
        assert_eq!(stats.progress_bytes(), 65_000_000);
    }

    #[test]
    fn non_git_lfs_progress_text_is_ignored() {
        assert!(parse_git_lfs_progress_line("Updated Git hooks.", Some(1024)).is_none());
        assert!(parse_git_lfs_progress_line("Git LFS initialized.", Some(1024)).is_none());
    }

    #[test]
    fn lfs_pointer_size_detects_unresolved_pointer() {
        let tmp = test_dir("lfs-pointer-detection");
        let source = tmp.join("source");
        fs::create_dir_all(&source).unwrap();
        let pointer = source.join("model.safetensors");
        write_lfs_pointer(&pointer, 1234);

        assert_eq!(lfs_pointer_size(&pointer).unwrap(), Some(1234));

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn lfs_pointer_validation_reports_relative_path() {
        let tmp = test_dir("lfs-pointer-validation");
        let source = tmp.join("source");
        fs::create_dir_all(source.join("nested")).unwrap();
        write_lfs_pointer(&source.join("nested/model.safetensors"), 4096);

        let err = ensure_no_lfs_pointers_remaining(&source).unwrap_err();
        let message = err.to_string();

        assert!(message.contains("Git LFS download incomplete"));
        assert!(message.contains("nested/model.safetensors"));

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn lfs_pointer_validation_skips_git_metadata() {
        let tmp = test_dir("lfs-pointer-validation-git");
        let source = tmp.join("source");
        fs::create_dir_all(source.join(".git/lfs/objects")).unwrap();
        fs::write(source.join("model.safetensors"), b"materialized").unwrap();
        write_lfs_pointer(&source.join(".git/lfs/objects/object"), 4096);

        ensure_no_lfs_pointers_remaining(&source).unwrap();

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn hf_repo_unreachable_message_suggests_microsoft_owner_typo() {
        let message = hf_repo_unreachable_message(
            "icrosoft/Phi-3-mini-4k-instruct-onnx",
            "https://huggingface.co/icrosoft/Phi-3-mini-4k-instruct-onnx",
            "Repository not found",
            None,
            false,
        );

        assert!(message.contains("not found or inaccessible"));
        assert!(message.contains("microsoft/Phi-3-mini-4k-instruct-onnx"));
        assert!(message.contains("Repository not found"));
    }

    #[test]
    fn hf_repo_unreachable_message_keeps_generic_repos_generic() {
        let message = hf_repo_unreachable_message(
            "unknown/model",
            "https://huggingface.co/unknown/model",
            "",
            None,
            false,
        );

        assert!(message.contains("unknown/model"));
        assert!(!message.contains("Did you mean"));
    }

    #[test]
    fn hf_repo_unreachable_message_mentions_gated_model_login() {
        let message = hf_repo_unreachable_message(
            "ai21labs/AI21-Jamba-Mini-1.7",
            "https://huggingface.co/ai21labs/AI21-Jamba-Mini-1.7",
            "could not read Username",
            Some(&HuggingFaceModelMetadata { gated: true }),
            false,
        );

        assert!(message.contains("gated model"));
        assert!(message.contains("cannot accept model conditions"));
        assert!(message.contains("werk auth huggingface login"));
        assert!(message.contains("HF_TOKEN"));
    }

    #[test]
    fn hf_repo_unreachable_message_mentions_token_account_for_gated_model() {
        let message = hf_repo_unreachable_message(
            "ai21labs/AI21-Jamba-Mini-1.7",
            "https://huggingface.co/ai21labs/AI21-Jamba-Mini-1.7",
            "Authentication failed",
            Some(&HuggingFaceModelMetadata { gated: true }),
            true,
        );

        assert!(message.contains("Your token is configured"));
        assert!(message.contains("accept the conditions with the same Hugging Face account"));
        assert!(message.contains("Authentication failed"));
    }

    #[test]
    fn parses_huggingface_gated_metadata() {
        assert!(
            parse_huggingface_model_metadata(&serde_json::json!({
                "gated": true
            }))
            .gated
        );
        assert!(
            parse_huggingface_model_metadata(&serde_json::json!({
                "gated": "auto"
            }))
            .gated
        );
        assert!(
            parse_huggingface_model_metadata(&serde_json::json!({
                "gated": "manual"
            }))
            .gated
        );
        assert!(
            !parse_huggingface_model_metadata(&serde_json::json!({
                "gated": false
            }))
            .gated
        );
    }

    #[test]
    fn huggingface_token_round_trips_through_werk_store() {
        let tmp = test_dir("hf-token-store");
        let store = ModelStore::resolve(Some(tmp.join("store"))).unwrap();

        let path = store.save_huggingface_token("hf_test_token").unwrap();
        assert_eq!(path, store.huggingface_token_path());
        assert_eq!(
            read_token_file(&path).unwrap().as_deref(),
            Some("hf_test_token")
        );
        assert!(store.huggingface_auth_status().unwrap().source.is_some());
        assert!(store.delete_huggingface_token().unwrap());
        assert!(!store.delete_huggingface_token().unwrap());

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn huggingface_git_auth_uses_extra_header_env_not_args() {
        let mut command = Command::new("git");
        command.args(["ls-remote", "https://huggingface.co/org/repo"]);

        configure_huggingface_git_auth(&mut command, Some("hf_secret"));

        let envs = command
            .get_envs()
            .map(|(key, value)| {
                (
                    key.to_string_lossy().to_string(),
                    value.map(|value| value.to_string_lossy().to_string()),
                )
            })
            .collect::<Vec<_>>();
        assert!(envs.iter().any(|(key, value)| {
            key == "GIT_CONFIG_VALUE_0"
                && value.as_deref()
                    == Some(&format!(
                        "Authorization: Basic {}",
                        base64_encode(b"hf_user:hf_secret")
                    ))
        }));
        let args = command
            .get_args()
            .map(|arg| arg.to_string_lossy().to_string())
            .collect::<Vec<_>>();
        assert!(!args.iter().any(|arg| arg.contains("hf_secret")));
    }

    #[test]
    fn base64_encode_handles_padding() {
        assert_eq!(
            base64_encode(b"hf_user:hf_secret"),
            "aGZfdXNlcjpoZl9zZWNyZXQ="
        );
        assert_eq!(base64_encode(b"ab"), "YWI=");
        assert_eq!(base64_encode(b"abc"), "YWJj");
    }

    #[test]
    fn included_file_import_tree_skips_unselected_lfs_pointers() {
        let tmp = test_dir("included-file-import-tree");
        let source = tmp.join("source");
        let filtered = tmp.join("filtered");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("Qwen3.5-4B-Q4_K_M.gguf"), b"materialized").unwrap();
        write_lfs_pointer(&source.join("Qwen3.5-4B-Q4_K_S.gguf"), 4_000_000_000);
        fs::write(source.join("config.json"), br#"{"model_type":"qwen3"}"#).unwrap();
        fs::write(source.join("README.md"), b"not required metadata").unwrap();
        fs::create_dir_all(source.join(".git/lfs/objects")).unwrap();
        write_lfs_pointer(&source.join(".git/lfs/objects/object"), 1024);

        prepare_included_file_import_tree(&source, "Qwen3.5-4B-Q4_K_M.gguf", &filtered).unwrap();
        ensure_no_lfs_pointers_remaining(&filtered).unwrap();

        assert!(filtered.join("Qwen3.5-4B-Q4_K_M.gguf").is_file());
        assert!(filtered.join("config.json").is_file());
        assert!(!filtered.join("Qwen3.5-4B-Q4_K_S.gguf").exists());
        assert!(!filtered.join("README.md").exists());
        assert!(!filtered.join(".git").exists());

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn included_file_import_tree_preserves_nested_selected_file_path() {
        let tmp = test_dir("included-file-import-tree-nested");
        let source = tmp.join("source");
        let filtered = tmp.join("filtered");
        fs::create_dir_all(source.join("quantized")).unwrap();
        fs::write(source.join("quantized/model.Q4_K_M.gguf"), b"materialized").unwrap();
        write_lfs_pointer(&source.join("quantized/model.Q5_K_M.gguf"), 5_000_000_000);
        fs::write(source.join("tokenizer_config.json"), b"{}").unwrap();

        prepare_included_file_import_tree(&source, "quantized/model.Q4_K_M.gguf", &filtered)
            .unwrap();
        ensure_no_lfs_pointers_remaining(&filtered).unwrap();

        assert!(filtered.join("quantized/model.Q4_K_M.gguf").is_file());
        assert!(filtered.join("tokenizer_config.json").is_file());
        assert!(!filtered.join("quantized/model.Q5_K_M.gguf").exists());

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

    #[test]
    fn set_model_file_can_switch_mixed_repo_between_gguf_and_safetensors() {
        let tmp = test_dir("mixed-format-selection");
        let source = tmp.join("source");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
        fs::write(source.join("model.safetensors"), b"safe").unwrap();
        fs::write(source.join("model.Q4_0.gguf"), b"gguf").unwrap();

        let store = ModelStore::resolve(Some(tmp.join("store"))).unwrap();
        let manifest = store.import_path(&source, "mixed").unwrap();
        assert_eq!(manifest.format, ModelFormat::Gguf);

        let manifest = store.set_model_file("mixed", "model.safetensors").unwrap();
        assert_eq!(manifest.format, ModelFormat::SafeTensors);
        assert_eq!(manifest.backend, "onnxruntime");
        assert_eq!(manifest.architecture.as_deref(), Some("llama"));
        assert_eq!(
            manifest.model_path.as_deref(),
            Some("files/model.safetensors")
        );

        let manifest = store.set_model_file("mixed", "model.Q4_0.gguf").unwrap();
        assert_eq!(manifest.format, ModelFormat::Gguf);
        assert_eq!(
            manifest.model_path.as_deref(),
            Some("files/model.Q4_0.gguf")
        );

        let _ = fs::remove_dir_all(tmp);
    }

    fn write_lfs_pointer(path: &Path, size: u64) {
        fs::write(
            path,
            format!(
                "version https://git-lfs.github.com/spec/v1\n\
                 oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n\
                 size {size}\n"
            ),
        )
        .unwrap();
    }

    fn write_safetensors_header(path: &Path, header: &str) {
        let mut data = Vec::new();
        data.extend_from_slice(&(header.len() as u64).to_le_bytes());
        data.extend_from_slice(header.as_bytes());
        data.extend_from_slice(&[0_u8; 4]);
        fs::write(path, data).unwrap();
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
