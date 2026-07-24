use anyhow::{Context, Result, anyhow, bail};
use std::{
    env, fs,
    path::{Component, Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use super::{
    backend::BackendOutput,
    helpers::validate_safe_name,
    types::{InferenceResult, OutputMetadata},
};
use crate::{inference::EffectiveInferenceRequest, model_store::ModelManifest};

const DEFAULT_OUTPUT_LIMIT_BYTES: u64 = 20 * 1024 * 1024 * 1024;
const DEFAULT_OUTPUT_RETENTION_SECONDS: u64 = 30 * 24 * 60 * 60;

#[derive(Debug, Clone)]
pub struct OutputStore {
    pub(super) root: PathBuf,
    pub(super) max_bytes: u64,
    max_age_seconds: u64,
}

impl OutputStore {
    pub fn new(home: &Path) -> Self {
        let max_bytes = env::var("WERK_OUTPUT_MAX_BYTES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_OUTPUT_LIMIT_BYTES);
        let max_age_seconds = env::var("WERK_OUTPUT_RETENTION_SECONDS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_OUTPUT_RETENTION_SECONDS);
        Self {
            root: home.join("outputs"),
            max_bytes,
            max_age_seconds,
        }
    }

    pub fn with_limits(home: &Path, max_bytes: u64, max_age_seconds: u64) -> Self {
        Self {
            root: home.join("outputs"),
            max_bytes,
            max_age_seconds,
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn ensure(&self) -> Result<()> {
        fs::create_dir_all(&self.root)?;
        Ok(())
    }

    pub fn create_output_dir(&self, id: &str) -> Result<PathBuf> {
        validate_safe_name(id)?;
        self.ensure()?;
        let path = self.root.join(id);
        fs::create_dir(&path)
            .with_context(|| format!("failed to create output directory {}", path.display()))?;
        Ok(path)
    }

    pub fn get_output(&self, output_id: &str) -> Result<OutputMetadata> {
        validate_safe_name(output_id)?;
        let (result_id, _) = output_id
            .rsplit_once('-')
            .ok_or_else(|| anyhow!("invalid output id '{output_id}'"))?;
        validate_safe_name(result_id)?;
        let metadata_path = self.root.join(result_id).join("metadata.json");
        let data = fs::read(&metadata_path)
            .with_context(|| format!("output '{output_id}' was not found"))?;
        let result: InferenceResult = serde_json::from_slice(&data)
            .with_context(|| format!("invalid output metadata {}", metadata_path.display()))?;
        let output = result
            .outputs
            .into_iter()
            .find(|output| output.id == output_id)
            .ok_or_else(|| anyhow!("output '{output_id}' was not found"))?;
        ensure_output_path(&self.root.join(result_id), Path::new(&output.path))?;
        Ok(output)
    }

    pub fn remove_result(&self, result_id: &str) -> Result<()> {
        validate_safe_name(result_id)?;
        let path = self.root.join(result_id);
        if !path.exists() {
            return Ok(());
        }
        remove_output_dir(&self.root, &path)
    }

    pub fn enforce_retention(&self) -> Result<()> {
        self.enforce_retention_internal(None).map(|_| ())
    }

    pub(super) fn enforce_retention_preserving(&self, preserve: &Path) -> Result<bool> {
        self.enforce_retention_internal(Some(preserve))
    }

    fn enforce_retention_internal(&self, preserve: Option<&Path>) -> Result<bool> {
        self.ensure()?;
        let now = SystemTime::now();
        let mut entries = fs::read_dir(&self.root)?
            .filter_map(std::result::Result::ok)
            .filter_map(|entry| {
                let metadata = entry.metadata().ok()?;
                if !metadata.is_dir() {
                    return None;
                }
                let modified = metadata.modified().ok().unwrap_or(UNIX_EPOCH);
                let size = directory_size(&entry.path()).ok()?;
                Some((entry.path(), modified, size))
            })
            .collect::<Vec<_>>();
        entries.sort_by_key(|(_, modified, _)| *modified);

        for (path, modified, _) in &entries {
            if preserve.is_some_and(|preserve| preserve == path) {
                continue;
            }
            if now
                .duration_since(*modified)
                .ok()
                .is_some_and(|age| age.as_secs() > self.max_age_seconds)
            {
                remove_output_dir(&self.root, path)?;
            }
        }
        entries.retain(|(path, _, _)| path.exists());
        let mut total = entries
            .iter()
            .map(|(_, _, size)| *size)
            .fold(0_u64, u64::saturating_add);
        for (path, _, size) in entries {
            if total <= self.max_bytes {
                break;
            }
            if preserve.is_some_and(|preserve| preserve == path) {
                continue;
            }
            remove_output_dir(&self.root, &path)?;
            total = total.saturating_sub(size);
        }
        Ok(total > self.max_bytes)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn output_metadata(
    result_id: &str,
    index: usize,
    manifest: &ModelManifest,
    effective: &EffectiveInferenceRequest,
    runtime: &str,
    output: BackendOutput,
    seed: Option<u64>,
    created_unix: u64,
    output_dir: &Path,
) -> Result<OutputMetadata> {
    ensure_output_path(output_dir, &output.path)?;
    let metadata = fs::metadata(&output.path)
        .with_context(|| format!("backend output does not exist: {}", output.path.display()))?;
    if !metadata.is_file() {
        bail!("backend output is not a file: {}", output.path.display());
    }
    Ok(OutputMetadata {
        id: format!("{result_id}-{index}"),
        task: effective.task,
        model: manifest.id.clone(),
        runtime: runtime.to_string(),
        path: output.path.display().to_string(),
        mime_type: output
            .mime_type
            .unwrap_or_else(|| mime_for_path(&output.path).to_string()),
        size_bytes: metadata.len(),
        width: output.width,
        height: output.height,
        duration: output.duration,
        seed,
        effective_parameters: effective.parameters.clone(),
        created_unix,
        backend_metadata: output.metadata,
    })
}

pub(super) fn ensure_output_path(root: &Path, path: &Path) -> Result<()> {
    if path
        .components()
        .any(|component| component == Component::ParentDir)
    {
        bail!(
            "backend output path contains parent traversal: {}",
            path.display()
        );
    }
    let normalized_root = root
        .canonicalize()
        .with_context(|| format!("cannot resolve output root {}", root.display()))?;
    let normalized_path = if path.exists() {
        path.canonicalize()?
    } else {
        let parent = path
            .parent()
            .ok_or_else(|| anyhow!("output path has no parent"))?
            .canonicalize()?;
        parent.join(
            path.file_name()
                .ok_or_else(|| anyhow!("output path has no file name"))?,
        )
    };
    if !normalized_path.starts_with(&normalized_root) {
        bail!(
            "backend output escaped the Werk output directory: {}",
            path.display()
        );
    }
    Ok(())
}

pub(super) fn remove_output_dir(root: &Path, path: &Path) -> Result<()> {
    let root = root.canonicalize()?;
    let path = path.canonicalize()?;
    if path.parent() != Some(root.as_path()) {
        bail!("refusing to remove non-output path {}", path.display());
    }
    fs::remove_dir_all(&path)
        .with_context(|| format!("failed to remove expired output {}", path.display()))
}

fn directory_size(path: &Path) -> Result<u64> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.is_file() {
        return Ok(metadata.len());
    }
    if !metadata.is_dir() {
        return Ok(0);
    }
    let mut total = 0_u64;
    for entry in fs::read_dir(path)? {
        total = total.saturating_add(directory_size(&entry?.path())?);
    }
    Ok(total)
}

fn mime_for_path(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        "wav" => "audio/wav",
        "flac" => "audio/flac",
        "mp3" => "audio/mpeg",
        "ogg" => "audio/ogg",
        "json" => "application/json",
        "txt" => "text/plain",
        _ => "application/octet-stream",
    }
}
