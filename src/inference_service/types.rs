use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

use crate::{
    capabilities::InferenceTask,
    inference::{EffectiveInferenceRequest, ExecutionPlan, ResolvedParameter, WorkloadEstimate},
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceResult {
    pub id: String,
    pub task: InferenceTask,
    pub model: String,
    pub runtime: String,
    pub outputs: Vec<OutputMetadata>,
    pub effective_request: EffectiveInferenceRequest,
    pub estimate: WorkloadEstimate,
    pub plan: ExecutionPlan,
    #[serde(default)]
    pub warnings: Vec<String>,
    pub created_unix: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputMetadata {
    pub id: String,
    pub task: InferenceTask,
    pub model: String,
    pub runtime: String,
    pub path: String,
    pub mime_type: String,
    pub size_bytes: u64,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub duration: Option<f64>,
    pub seed: Option<u64>,
    pub effective_parameters: BTreeMap<String, ResolvedParameter>,
    pub created_unix: u64,
    #[serde(default)]
    pub backend_metadata: Value,
}
