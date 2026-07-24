use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::{
    capabilities::{InferenceTask, OutputModality, RepositoryLayout},
    model_store::{ModelFormat, ModelManifest},
};

use super::{
    estimate::{FitAssessment, WorkloadEstimate},
    types::{EffectiveInferenceRequest, ParameterPolicy, ParameterSupportStatus},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeAccelerator {
    Cpu,
    Cuda,
    Rocm,
    Mps,
    Mlx,
    Other,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceRuntimeCandidate {
    pub id: String,
    pub backend: String,
    pub accelerator: RuntimeAccelerator,
    pub available: bool,
    pub availability_reason: Option<String>,
    pub supported_tasks: Vec<InferenceTask>,
    pub supported_layouts: Vec<RepositoryLayout>,
    #[serde(default)]
    pub supported_formats: Vec<ModelFormat>,
    #[serde(default)]
    pub supported_families: Vec<String>,
    #[serde(default)]
    pub supported_architectures: Vec<String>,
    #[serde(default)]
    pub parameter_support: BTreeMap<String, ParameterSupportStatus>,
    pub supports_offloading: bool,
    pub supports_compile: bool,
    pub supports_batching: bool,
    pub priority: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanCandidateStatus {
    Accepted,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanCandidateDecision {
    pub runtime_id: String,
    pub backend: String,
    pub status: PlanCandidateStatus,
    pub score: Option<i32>,
    pub reasons: Vec<String>,
    pub degradations: Vec<ExecutionDegradation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutionDegradation {
    CpuOffload,
    SequentialOffload,
    ComponentOffload,
    VaeTiling,
    TemporalWindowing,
    SlowerAttention { backend: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub task: InferenceTask,
    pub selected_runtime: Option<String>,
    pub selected_backend: Option<String>,
    pub score: Option<i32>,
    pub candidates: Vec<PlanCandidateDecision>,
    pub backend_fallback: bool,
    pub degradations: Vec<ExecutionDegradation>,
    pub model_or_quality_downgrades: Vec<String>,
}

pub fn plan_execution(
    manifest: &ModelManifest,
    request: &EffectiveInferenceRequest,
    estimate: &WorkloadEstimate,
    candidates: &[InferenceRuntimeCandidate],
) -> ExecutionPlan {
    let requested_backend = request
        .string_parameter("routing.backend")
        .filter(|value| *value != "auto");
    let requested_accelerator = request
        .string_parameter("routing.accelerator")
        .filter(|value| *value != "auto");
    let fallback_policy = request
        .string_parameter("routing.fallback_policy")
        .unwrap_or("backend");
    let allow_degradation = fallback_policy == "degrade";
    let mut decisions = Vec::new();

    for candidate in candidates {
        let mut reasons = Vec::new();
        let mut degradations = Vec::new();
        if !candidate.available {
            reasons.push(
                candidate
                    .availability_reason
                    .clone()
                    .unwrap_or_else(|| "runtime is unavailable".to_string()),
            );
        }
        if !candidate.supported_tasks.contains(&request.task) {
            reasons.push(format!("runtime does not support task {}", request.task));
        }
        if !candidate.supported_layouts.is_empty()
            && !candidate
                .supported_layouts
                .contains(&manifest.metadata.repository_layout)
        {
            reasons.push(format!(
                "runtime does not support {:?} repository layout",
                manifest.metadata.repository_layout
            ));
        }
        if !candidate.supported_formats.is_empty()
            && !candidate.supported_formats.contains(&manifest.format)
        {
            reasons.push(format!(
                "runtime does not support {:?} model format",
                manifest.format
            ));
        }
        if let Some(family) = manifest.metadata.family.as_deref()
            && !candidate.supported_families.is_empty()
            && !candidate
                .supported_families
                .iter()
                .any(|supported| supported.eq_ignore_ascii_case(family))
        {
            reasons.push(format!("runtime does not support model family '{family}'"));
        }
        if let Some(architecture) = manifest.architecture.as_deref()
            && !candidate.supported_architectures.is_empty()
            && !candidate
                .supported_architectures
                .iter()
                .any(|supported| supported.eq_ignore_ascii_case(architecture))
        {
            reasons.push(format!(
                "runtime does not support architecture '{architecture}'"
            ));
        }
        if let Some(backend) = requested_backend
            && fallback_policy == "none"
            && !candidate.backend.eq_ignore_ascii_case(backend)
        {
            reasons.push(format!("explicit backend '{backend}' was requested"));
        }
        if let Some(accelerator) = requested_accelerator
            && !runtime_accelerator_matches(candidate.accelerator, accelerator)
        {
            reasons.push(format!("accelerator '{accelerator}' was requested"));
        }
        for path in &request.explicit_parameters {
            if matches!(
                candidate
                    .parameter_support
                    .get(path)
                    .copied()
                    .unwrap_or(ParameterSupportStatus::ModelDependent),
                ParameterSupportStatus::Ignored | ParameterSupportStatus::Unsupported
            ) && request.parameter_policy == ParameterPolicy::Strict
            {
                reasons.push(format!("explicit parameter '{path}' is unsupported"));
            }
        }
        if estimate.fit == FitAssessment::LikelyOom {
            if allow_degradation && candidate.supports_offloading {
                let accelerator_can_offload = matches!(
                    candidate.accelerator,
                    RuntimeAccelerator::Cuda | RuntimeAccelerator::Rocm
                );
                if accelerator_can_offload
                    && request.bool_parameter("routing.allow_cpu_offload") == Some(true)
                {
                    degradations.push(ExecutionDegradation::CpuOffload);
                }
                if accelerator_can_offload
                    && request.bool_parameter("routing.allow_sequential_offload") == Some(true)
                {
                    degradations.push(ExecutionDegradation::SequentialOffload);
                }
                if accelerator_can_offload
                    && request.bool_parameter("routing.allow_component_offload") == Some(true)
                {
                    degradations.push(ExecutionDegradation::ComponentOffload);
                }
                if request.task.output_modality() == OutputModality::Image
                    && request.bool_parameter("image.vae_tiling") == Some(true)
                {
                    degradations.push(ExecutionDegradation::VaeTiling);
                }
                if request.task.output_modality() == OutputModality::Video
                    && (request.bool_parameter("video.temporal_vae_tiling") == Some(true)
                        || request.u64_parameter("video.window_size").is_some())
                {
                    degradations.push(ExecutionDegradation::TemporalWindowing);
                }
                if degradations.is_empty() {
                    reasons.push(
                        "workload is likely out of memory and no permitted degradation is enabled"
                            .to_string(),
                    );
                }
            } else {
                reasons.push("workload is likely out of memory".to_string());
            }
        }

        let accepted = reasons.is_empty();
        let mut score = candidate.priority;
        if requested_backend.is_some_and(|backend| candidate.backend.eq_ignore_ascii_case(backend))
        {
            score += 500;
        }
        score -= i32::try_from(degradations.len()).unwrap_or(i32::MAX) * 35;
        if candidate.accelerator == RuntimeAccelerator::Cpu {
            score -= 120;
        }
        decisions.push(PlanCandidateDecision {
            runtime_id: candidate.id.clone(),
            backend: candidate.backend.clone(),
            status: if accepted {
                PlanCandidateStatus::Accepted
            } else {
                PlanCandidateStatus::Rejected
            },
            score: accepted.then_some(score),
            reasons,
            degradations,
        });
    }

    decisions.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| left.runtime_id.cmp(&right.runtime_id))
    });
    let selected = decisions
        .iter()
        .find(|decision| decision.status == PlanCandidateStatus::Accepted)
        .cloned();
    let selected_backend = selected.as_ref().map(|decision| decision.backend.clone());
    let backend_fallback = match (requested_backend, selected_backend.as_deref()) {
        (Some(requested), Some(selected)) => !requested.eq_ignore_ascii_case(selected),
        _ => false,
    };
    let model_or_quality_downgrades = if estimate.fit == FitAssessment::LikelyOom {
        vec![
            "consider a smaller model or stronger quantization".to_string(),
            match request.task.output_modality() {
                OutputModality::Image => "consider a lower image resolution".to_string(),
                OutputModality::Video => {
                    "consider fewer frames or a lower video resolution".to_string()
                }
                OutputModality::Audio => "consider a shorter duration".to_string(),
                OutputModality::Text | OutputModality::Embedding => {
                    "consider a shorter context".to_string()
                }
            },
        ]
    } else {
        Vec::new()
    };

    ExecutionPlan {
        task: request.task,
        selected_runtime: selected
            .as_ref()
            .map(|decision| decision.runtime_id.clone()),
        selected_backend,
        score: selected.as_ref().and_then(|decision| decision.score),
        candidates: decisions,
        backend_fallback,
        degradations: selected
            .map(|decision| decision.degradations)
            .unwrap_or_default(),
        model_or_quality_downgrades,
    }
}

fn runtime_accelerator_matches(accelerator: RuntimeAccelerator, requested: &str) -> bool {
    matches!(
        (accelerator, requested.to_ascii_lowercase().as_str()),
        (RuntimeAccelerator::Cpu, "cpu")
            | (RuntimeAccelerator::Cuda, "cuda")
            | (RuntimeAccelerator::Rocm, "rocm" | "hip")
            | (RuntimeAccelerator::Mps, "mps" | "metal")
            | (RuntimeAccelerator::Mlx, "mlx" | "metal")
    )
}
