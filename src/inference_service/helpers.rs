use anyhow::{Result, anyhow, bail};
use getrandom::getrandom;
use serde::Serialize;
use std::{fs, path::Path};

use crate::model_store::unix_ts;

pub(super) fn validate_safe_name(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || value
            .chars()
            .any(|ch| !(ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.')))
        || value == "."
        || value == ".."
    {
        bail!("unsafe storage identifier '{value}'");
    }
    Ok(())
}

pub(super) fn new_id(prefix: &str) -> Result<String> {
    let mut random = [0_u8; 8];
    getrandom(&mut random).map_err(|error| anyhow!("failed to generate id: {error}"))?;
    let suffix = random
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(format!("{prefix}-{}-{suffix}", unix_ts()))
}

pub(super) fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)?;
    let temporary = path.with_extension(new_id("tmp")?);
    let data = serde_json::to_vec_pretty(value)?;
    fs::write(&temporary, data)?;
    fs::rename(&temporary, path)?;
    Ok(())
}
