use std::{collections::HashMap, fmt};

use crate::{
    backend::{
        BackendAccelerator, BackendRuntime, RuntimeId, backend_supports_images,
        explain_backend_rejection, runtime_descriptor, runtime_supports_model,
    },
    model_store::{ModelFormat, ModelManifest},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestedBackend {
    Auto,
    Cpu,
    Cuda,
    Vulkan,
    Metal,
    Mlx,
    Burn,
    Candle,
    Vllm,
    LlamaLegacy,
    LlamaHighlevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestCapabilities {
    pub text_generation: bool,
    pub image_input: bool,
    pub embeddings: bool,
    pub streaming: bool,
}

impl RequestCapabilities {
    pub fn text(streaming: bool) -> Self {
        Self {
            text_generation: true,
            image_input: false,
            embeddings: false,
            streaming,
        }
    }

    pub fn text_with_images(streaming: bool, image_input: bool) -> Self {
        Self {
            image_input,
            ..Self::text(streaming)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeAvailability {
    pub runtime_id: RuntimeId,
    pub available: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeCandidate {
    pub runtime_id: RuntimeId,
    pub priority: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeDecisionStatus {
    Accepted,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeDecision {
    pub runtime_id: RuntimeId,
    pub display_name: &'static str,
    pub status: RuntimeDecisionStatus,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedRuntime {
    pub runtime_id: RuntimeId,
    pub display_name: &'static str,
    pub accelerator: BackendAccelerator,
    pub reason: String,
    pub fallback_chain: Vec<RuntimeDecision>,
    pub rejection_reasons: Vec<RuntimeDecision>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimePlan {
    pub requested_backend: RequestedBackend,
    pub request_capabilities: RequestCapabilities,
    pub candidates: Vec<RuntimeDecision>,
    pub selected: Option<SelectedRuntime>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimePlanError {
    pub requested_backend: RequestedBackend,
    pub decisions: Vec<RuntimeDecision>,
}

impl fmt::Display for RuntimePlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "no available runtime for requested backend {:?}",
            self.requested_backend
        )?;
        if self.decisions.is_empty() {
            return write!(f, "no runtime candidates matched this request");
        }
        writeln!(f, "tried:")?;
        for decision in &self.decisions {
            writeln!(f, "- {}: {}", decision.display_name, decision.reason)?;
        }
        Ok(())
    }
}

impl std::error::Error for RuntimePlanError {}

pub fn select_runtime(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    request_capabilities: RequestCapabilities,
    available_runtimes: &[RuntimeAvailability],
) -> Result<SelectedRuntime, RuntimePlanError> {
    let plan = plan_runtime(
        manifest,
        requested_backend,
        request_capabilities,
        available_runtimes,
    );
    plan.selected.clone().ok_or(RuntimePlanError {
        requested_backend,
        decisions: plan.candidates,
    })
}

pub fn plan_runtime(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    request_capabilities: RequestCapabilities,
    available_runtimes: &[RuntimeAvailability],
) -> RuntimePlan {
    let availability = availability_map(available_runtimes);
    let mut candidates = Vec::new();
    let mut selected = None;
    let mut rejections = Vec::new();

    for candidate in runtime_candidates(manifest, requested_backend) {
        let descriptor = runtime_descriptor(candidate.runtime_id);
        let decision = candidate_decision(
            manifest,
            requested_backend,
            request_capabilities,
            candidate.runtime_id,
            availability.get(&candidate.runtime_id),
        );
        if decision.status == RuntimeDecisionStatus::Accepted {
            selected = Some(SelectedRuntime {
                runtime_id: candidate.runtime_id,
                display_name: descriptor.display_name,
                accelerator: descriptor
                    .accelerators
                    .first()
                    .copied()
                    .unwrap_or(BackendAccelerator::Auto),
                reason: selection_reason(manifest, requested_backend, descriptor.runtime),
                fallback_chain: rejections.clone(),
                rejection_reasons: rejections.clone(),
            });
            candidates.push(decision);
            break;
        }
        rejections.push(decision.clone());
        candidates.push(decision);
    }

    RuntimePlan {
        requested_backend,
        request_capabilities,
        candidates,
        selected,
    }
}

pub fn runtime_candidates(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
) -> Vec<RuntimeCandidate> {
    runtime_candidate_ids(manifest, requested_backend)
        .into_iter()
        .map(|runtime_id| RuntimeCandidate {
            priority: runtime_descriptor(runtime_id).priority,
            runtime_id,
        })
        .collect()
}

pub fn runtime_candidate_ids(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
) -> Vec<RuntimeId> {
    match requested_backend {
        RequestedBackend::Auto => auto_candidates(manifest),
        RequestedBackend::Cpu => cpu_candidates(manifest),
        RequestedBackend::Cuda => cuda_candidates(manifest),
        RequestedBackend::Vulkan => vulkan_candidates(manifest),
        RequestedBackend::Metal => metal_candidates(manifest),
        RequestedBackend::Mlx => vec![RuntimeId::Mlx],
        RequestedBackend::Burn => vec![RuntimeId::Burn],
        RequestedBackend::Candle => candle_candidates(manifest),
        RequestedBackend::Vllm => vec![RuntimeId::VllmCuda],
        RequestedBackend::LlamaLegacy | RequestedBackend::LlamaHighlevel => Vec::new(),
    }
}

fn auto_candidates(manifest: &ModelManifest) -> Vec<RuntimeId> {
    match manifest.format {
        ModelFormat::Gguf => gguf_auto_candidates(),
        ModelFormat::SafeTensors => safetensors_auto_candidates(),
        ModelFormat::Onnx => onnx_auto_candidates(),
        ModelFormat::Mlx => {
            if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                vec![RuntimeId::Mlx, RuntimeId::CandleMetal, RuntimeId::CandleCpu]
            } else {
                vec![RuntimeId::Mlx]
            }
        }
        ModelFormat::TensorRt
        | ModelFormat::OpenVino
        | ModelFormat::CoreMl
        | ModelFormat::PyTorch
        | ModelFormat::TensorFlow
        | ModelFormat::Unknown => Vec::new(),
    }
}

fn onnx_auto_candidates() -> Vec<RuntimeId> {
    if cfg!(any(windows, target_os = "linux")) {
        vec![RuntimeId::OnnxRuntimeCuda, RuntimeId::OnnxRuntimeCpu]
    } else {
        vec![RuntimeId::OnnxRuntimeCpu]
    }
}

fn gguf_auto_candidates() -> Vec<RuntimeId> {
    if cfg!(any(windows, target_os = "linux")) {
        vec![
            RuntimeId::LlamaServerCuda,
            RuntimeId::LlamaServerVulkan,
            RuntimeId::LlamaServerCpu,
            RuntimeId::CandleCuda,
            RuntimeId::CandleCpu,
        ]
    } else if cfg!(target_os = "macos") {
        vec![
            RuntimeId::LlamaServerCpu,
            RuntimeId::CandleMetal,
            RuntimeId::CandleCpu,
        ]
    } else {
        vec![RuntimeId::LlamaServerCpu, RuntimeId::CandleCpu]
    }
}

fn safetensors_auto_candidates() -> Vec<RuntimeId> {
    append_candle_fallbacks(vec![RuntimeId::Burn])
}

fn append_candle_fallbacks(mut candidates: Vec<RuntimeId>) -> Vec<RuntimeId> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        candidates.extend([RuntimeId::Mlx, RuntimeId::CandleMetal, RuntimeId::CandleCpu]);
    } else if cfg!(target_os = "macos") {
        candidates.extend([RuntimeId::CandleMetal, RuntimeId::CandleCpu]);
    } else if cfg!(any(windows, target_os = "linux")) {
        candidates.extend([RuntimeId::CandleCuda, RuntimeId::CandleCpu]);
    } else {
        candidates.push(RuntimeId::CandleCpu);
    }
    dedupe(candidates)
}

fn cpu_candidates(manifest: &ModelManifest) -> Vec<RuntimeId> {
    match manifest.format {
        ModelFormat::Gguf => vec![RuntimeId::LlamaServerCpu, RuntimeId::CandleCpu],
        ModelFormat::SafeTensors => vec![RuntimeId::Burn, RuntimeId::CandleCpu],
        ModelFormat::Onnx => vec![RuntimeId::OnnxRuntimeCpu],
        _ => Vec::new(),
    }
}

fn cuda_candidates(manifest: &ModelManifest) -> Vec<RuntimeId> {
    match manifest.format {
        ModelFormat::Gguf => vec![RuntimeId::LlamaServerCuda],
        ModelFormat::SafeTensors => vec![RuntimeId::Burn, RuntimeId::CandleCuda],
        ModelFormat::Onnx => vec![RuntimeId::OnnxRuntimeCuda],
        _ => Vec::new(),
    }
}

fn vulkan_candidates(manifest: &ModelManifest) -> Vec<RuntimeId> {
    match manifest.format {
        ModelFormat::Gguf => vec![RuntimeId::LlamaServerVulkan],
        ModelFormat::SafeTensors => Vec::new(),
        _ => Vec::new(),
    }
}

fn metal_candidates(manifest: &ModelManifest) -> Vec<RuntimeId> {
    match manifest.format {
        ModelFormat::SafeTensors | ModelFormat::Gguf => vec![RuntimeId::CandleMetal],
        _ => Vec::new(),
    }
}

fn candle_candidates(manifest: &ModelManifest) -> Vec<RuntimeId> {
    match manifest.format {
        ModelFormat::Gguf | ModelFormat::SafeTensors => {
            if cfg!(target_os = "macos") {
                vec![RuntimeId::CandleMetal, RuntimeId::CandleCpu]
            } else if cfg!(any(windows, target_os = "linux")) {
                vec![RuntimeId::CandleCuda, RuntimeId::CandleCpu]
            } else {
                vec![RuntimeId::CandleCpu]
            }
        }
        _ => Vec::new(),
    }
}

fn candidate_decision(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    request_capabilities: RequestCapabilities,
    runtime_id: RuntimeId,
    availability: Option<&RuntimeAvailability>,
) -> RuntimeDecision {
    let descriptor = runtime_descriptor(runtime_id);
    let reason = rejection_reason(
        manifest,
        requested_backend,
        request_capabilities,
        runtime_id,
        availability,
    );
    match reason {
        Some(reason) => RuntimeDecision {
            runtime_id,
            display_name: descriptor.display_name,
            status: RuntimeDecisionStatus::Rejected,
            reason,
        },
        None => RuntimeDecision {
            runtime_id,
            display_name: descriptor.display_name,
            status: RuntimeDecisionStatus::Accepted,
            reason: selection_reason(manifest, requested_backend, descriptor.runtime),
        },
    }
}

fn rejection_reason(
    manifest: &ModelManifest,
    _requested_backend: RequestedBackend,
    request_capabilities: RequestCapabilities,
    runtime_id: RuntimeId,
    availability: Option<&RuntimeAvailability>,
) -> Option<String> {
    let descriptor = runtime_descriptor(runtime_id);
    if !runtime_supports_model(
        descriptor,
        &manifest.format,
        manifest.architecture.as_deref(),
    ) {
        return Some(model_support_rejection(manifest, descriptor.runtime));
    }
    if request_capabilities.text_generation && !descriptor.capabilities.text_generation {
        return Some("runtime does not support text generation".to_string());
    }
    if request_capabilities.image_input && !descriptor.capabilities.vision_language {
        return Some("runtime is not VLM-capable".to_string());
    }
    if request_capabilities.embeddings && !descriptor.capabilities.embeddings {
        return Some("runtime does not support embeddings".to_string());
    }
    if request_capabilities.streaming && !descriptor.capabilities.streaming {
        return Some("runtime does not support streaming".to_string());
    }
    if let Some(reason) = explain_backend_rejection(
        descriptor.runtime,
        &manifest.format,
        request_capabilities.image_input,
    ) {
        return Some(reason.to_string());
    }
    if request_capabilities.image_input && !backend_supports_images(descriptor.runtime) {
        return Some("runtime is not VLM-capable".to_string());
    }
    if !descriptor.implemented {
        return Some(unimplemented_runtime_rejection(
            manifest,
            descriptor.runtime,
        ));
    }
    match availability {
        Some(availability) if availability.available => None,
        Some(availability) => Some(
            availability
                .reason
                .clone()
                .unwrap_or_else(|| "runtime is unavailable".to_string()),
        ),
        None => Some("runtime availability was not reported".to_string()),
    }
}

fn model_support_rejection(manifest: &ModelManifest, runtime: BackendRuntime) -> String {
    match (runtime, &manifest.format) {
        (BackendRuntime::Vllm, ModelFormat::SafeTensors) => {
            "vLLM is not selected for this architecture".to_string()
        }
        _ => "model format or architecture is not supported".to_string(),
    }
}

fn unimplemented_runtime_rejection(manifest: &ModelManifest, runtime: BackendRuntime) -> String {
    let _ = (manifest, runtime);
    "runtime integration is not implemented yet".to_string()
}

fn selection_reason(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    runtime: BackendRuntime,
) -> String {
    match (manifest.format.clone(), runtime, requested_backend) {
        (ModelFormat::Gguf, BackendRuntime::LlamaServer, _) => {
            "GGUF hot path uses persistent llama.cpp server".to_string()
        }
        (ModelFormat::SafeTensors, BackendRuntime::OnnxRuntime, _) => {
            "HF safetensors hot path uses managed ONNX Runtime artifacts".to_string()
        }
        (ModelFormat::SafeTensors, BackendRuntime::Burn, _) => {
            "HF safetensors hot path uses Burn".to_string()
        }
        (ModelFormat::SafeTensors, BackendRuntime::Vllm, _) => {
            "explicit vLLM route requested".to_string()
        }
        (_, BackendRuntime::Candle, RequestedBackend::Candle) => {
            "explicit Candle route requested".to_string()
        }
        (_, BackendRuntime::Candle, _) => {
            "fallback runtime supports the selected model architecture".to_string()
        }
        (ModelFormat::Mlx, BackendRuntime::Mlx, _) => "MLX model uses mlx-lm".to_string(),
        (_, BackendRuntime::Mlx, _) => "MLX runtime selected for compatible model".to_string(),
        (_, _, RequestedBackend::Cpu) => "best CPU runtime for this model".to_string(),
        (_, _, RequestedBackend::Cuda) => "best CUDA runtime for this model".to_string(),
        (_, _, RequestedBackend::Vulkan) => "best Vulkan runtime for this model".to_string(),
        _ => "best available runtime for this model".to_string(),
    }
}

fn availability_map(
    available_runtimes: &[RuntimeAvailability],
) -> HashMap<RuntimeId, RuntimeAvailability> {
    available_runtimes
        .iter()
        .cloned()
        .map(|availability| (availability.runtime_id, availability))
        .collect()
}

fn dedupe(runtime_ids: Vec<RuntimeId>) -> Vec<RuntimeId> {
    let mut out = Vec::with_capacity(runtime_ids.len());
    for runtime_id in runtime_ids {
        if !out.contains(&runtime_id) {
            out.push(runtime_id);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_store::{ModelManifest, ModelSource};

    #[test]
    fn gguf_auto_prefers_llama_server_before_candle() {
        let manifest = manifest(ModelFormat::Gguf, Some("llama"));
        let candidates = runtime_candidate_ids(&manifest, RequestedBackend::Auto);
        assert_eq!(candidates[0], RuntimeId::LlamaServerCuda);
        assert_eq!(candidates[1], RuntimeId::LlamaServerVulkan);
        assert_eq!(candidates[2], RuntimeId::LlamaServerCpu);
        assert!(candidates.contains(&RuntimeId::CandleCpu));
    }

    #[test]
    fn safetensors_cuda_uses_burn_then_candle_without_cpu_fallback() {
        let manifest = manifest(ModelFormat::SafeTensors, Some("phi3"));
        let candidates = runtime_candidate_ids(&manifest, RequestedBackend::Cuda);
        assert_eq!(candidates[0], RuntimeId::Burn);
        assert!(candidates.contains(&RuntimeId::CandleCuda));
        assert!(!candidates.contains(&RuntimeId::CandleCpu));
    }

    #[test]
    fn safetensors_auto_tries_burn_before_candle_for_any_architecture() {
        let qwen = manifest(ModelFormat::SafeTensors, Some("qwen2"));
        let candidates = runtime_candidate_ids(&qwen, RequestedBackend::Auto);
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
        let unknown = manifest(ModelFormat::SafeTensors, Some("unknown"));
        let candidates = runtime_candidate_ids(&unknown, RequestedBackend::Auto);
        assert_eq!(candidates[0], RuntimeId::Burn);
        assert!(candidates.contains(&RuntimeId::CandleCuda));
    }

    #[test]
    fn explicit_vllm_request_has_no_candle_fallback_candidates() {
        let manifest = manifest(ModelFormat::SafeTensors, Some("phi3"));
        let candidates = runtime_candidate_ids(&manifest, RequestedBackend::Vllm);
        assert_eq!(candidates, vec![RuntimeId::VllmCuda]);
    }

    #[test]
    fn explicit_burn_request_has_no_candle_fallback_candidates() {
        let manifest = manifest(ModelFormat::SafeTensors, Some("unknown"));
        let candidates = runtime_candidate_ids(&manifest, RequestedBackend::Burn);
        assert_eq!(candidates, vec![RuntimeId::Burn]);
    }

    #[test]
    fn onnx_auto_routes_to_onnxruntime() {
        let manifest = manifest(ModelFormat::Onnx, None);
        let candidates = runtime_candidate_ids(&manifest, RequestedBackend::Auto);
        assert!(matches!(
            candidates.first(),
            Some(RuntimeId::OnnxRuntimeCuda | RuntimeId::OnnxRuntimeCpu)
        ));
        assert!(!candidates.contains(&RuntimeId::CandleCpu));
        let selected = select_runtime(
            &manifest,
            RequestedBackend::Auto,
            RequestCapabilities::text(true),
            &[RuntimeAvailability {
                runtime_id: candidates[0],
                available: true,
                reason: None,
            }],
        )
        .unwrap();
        assert_eq!(selected.runtime_id, candidates[0]);
    }

    #[test]
    fn image_request_rejects_text_only_runtime() {
        let manifest = manifest(ModelFormat::SafeTensors, Some("phi3"));
        let available = [RuntimeAvailability {
            runtime_id: RuntimeId::CandleCuda,
            available: true,
            reason: None,
        }];
        let plan = plan_runtime(
            &manifest,
            RequestedBackend::Candle,
            RequestCapabilities::text_with_images(true, true),
            &available,
        );
        assert!(plan.selected.is_none());
        assert!(
            plan.candidates
                .iter()
                .any(|decision| decision.reason.contains("VLM"))
        );
    }

    #[test]
    fn candle_can_be_selected_as_explicit_route() {
        let manifest = manifest(ModelFormat::SafeTensors, Some("phi3"));
        let available = [RuntimeAvailability {
            runtime_id: RuntimeId::CandleCpu,
            available: true,
            reason: None,
        }];
        let selected = select_runtime(
            &manifest,
            RequestedBackend::Candle,
            RequestCapabilities::text(true),
            &available,
        )
        .unwrap();
        assert_eq!(selected.runtime_id, RuntimeId::CandleCpu);
    }

    fn manifest(format: ModelFormat, architecture: Option<&str>) -> ModelManifest {
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
            created_unix: 0,
            files: Vec::new(),
            artifacts: Vec::new(),
        }
    }
}
