use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use crate::{
    capabilities::InferenceTask,
    inference::{
        EffectiveInferenceRequest, InferenceRuntimeCandidate, ParameterSupportStatus,
        WorkloadEstimate,
    },
    model_store::{ModelManifest, ModelStore},
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendProbe {
    pub available: bool,
    pub detail: String,
    pub candidates: Vec<InferenceRuntimeCandidate>,
    #[serde(default)]
    pub parameter_support: BTreeMap<String, ParameterSupportStatus>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendExecution {
    pub runtime: String,
    pub outputs: Vec<BackendOutput>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendOutput {
    pub path: PathBuf,
    pub mime_type: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub duration: Option<f64>,
    #[serde(default)]
    pub metadata: Value,
}

pub trait MediaInferenceBackend: Send + Sync {
    fn probe(
        &self,
        store: &ModelStore,
        manifest: &ModelManifest,
        task: InferenceTask,
        schema_paths: &[String],
    ) -> BackendProbe;

    fn execute(
        &self,
        store: &ModelStore,
        manifest: &ModelManifest,
        request: &EffectiveInferenceRequest,
        output_dir: &Path,
        runtime: &str,
    ) -> Result<BackendExecution>;

    fn estimate(
        &self,
        _store: &ModelStore,
        _manifest: &ModelManifest,
        _request: &EffectiveInferenceRequest,
    ) -> Result<Option<WorkloadEstimate>> {
        Ok(None)
    }
}
