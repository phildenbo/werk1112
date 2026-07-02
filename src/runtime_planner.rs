use std::{collections::HashMap, fmt};

use crate::{
    backend::{
        BackendAccelerator, BackendRuntime, RuntimeId, backend_supports_images,
        explain_backend_rejection, is_transformers_compat_model, runtime_descriptor,
        runtime_supports_model,
    },
    model_store::{ModelFormat, ModelManifest},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestedBackend {
    Auto,
    Cpu,
    Cuda,
    Rocm,
    Vulkan,
    Metal,
    Mlx,
    Burn,
    Candle,
    Transformers,
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

    for runtime_id in
        runtime_candidate_ids_for_plan(manifest, requested_backend, available_runtimes)
    {
        let descriptor = runtime_descriptor(runtime_id);
        let decision = candidate_decision(
            manifest,
            requested_backend,
            request_capabilities,
            runtime_id,
            availability.get(&runtime_id),
        );
        if decision.status == RuntimeDecisionStatus::Accepted {
            selected = Some(SelectedRuntime {
                runtime_id,
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
        RequestedBackend::Rocm => rocm_candidates(manifest),
        RequestedBackend::Vulkan => vulkan_candidates(manifest),
        RequestedBackend::Metal => metal_candidates(manifest),
        RequestedBackend::Mlx => vec![RuntimeId::MlxVlm, RuntimeId::Mlx],
        RequestedBackend::Burn => burn_candidates(),
        RequestedBackend::Candle => candle_candidates(manifest),
        RequestedBackend::Transformers => vec![RuntimeId::TransformersCompat],
        RequestedBackend::Vllm => vec![RuntimeId::VllmCuda],
        RequestedBackend::LlamaLegacy | RequestedBackend::LlamaHighlevel => Vec::new(),
    }
}

fn runtime_candidate_ids_for_plan(
    manifest: &ModelManifest,
    requested_backend: RequestedBackend,
    available_runtimes: &[RuntimeAvailability],
) -> Vec<RuntimeId> {
    let mut candidates = runtime_candidate_ids(manifest, requested_backend);
    if requested_backend == RequestedBackend::Auto
        && manifest.format == ModelFormat::Gguf
        && available_runtimes
            .iter()
            .any(|availability| availability.runtime_id == RuntimeId::LlamaServerRocm)
        && !candidates.contains(&RuntimeId::LlamaServerRocm)
    {
        let insert_at = llama_rocm_insert_position(&candidates);
        candidates.insert(insert_at, RuntimeId::LlamaServerRocm);
    }
    candidates
}

fn llama_rocm_insert_position(candidates: &[RuntimeId]) -> usize {
    candidates
        .iter()
        .position(|id| {
            matches!(
                id,
                RuntimeId::LlamaServerVulkan
                    | RuntimeId::LlamaServerMetal
                    | RuntimeId::LlamaServerCpu
            )
        })
        .unwrap_or(candidates.len())
}

fn auto_candidates(manifest: &ModelManifest) -> Vec<RuntimeId> {
    match manifest.format {
        ModelFormat::Gguf => gguf_auto_candidates(),
        ModelFormat::SafeTensors => safetensors_auto_candidates(manifest),
        ModelFormat::Onnx => onnx_auto_candidates(),
        ModelFormat::Mlx => {
            if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                vec![
                    RuntimeId::MlxVlm,
                    RuntimeId::Mlx,
                    RuntimeId::CandleMetal,
                    RuntimeId::CandleCpu,
                ]
            } else {
                vec![RuntimeId::MlxVlm, RuntimeId::Mlx]
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
            // ROCm is added by the CLI selection layer only when a ROCm/HIP
            // llama-server is detected or explicitly signaled. Keeping it out
            // of the default pure planner avoids noisy NVIDIA-only auto output.
            RuntimeId::LlamaServerVulkan,
            RuntimeId::LlamaServerCpu,
            RuntimeId::CandleCuda,
            RuntimeId::CandleCpu,
        ]
    } else if cfg!(target_os = "macos") {
        vec![
            RuntimeId::LlamaServerMetal,
            RuntimeId::LlamaServerCpu,
            RuntimeId::CandleMetal,
            RuntimeId::CandleCpu,
        ]
    } else {
        vec![RuntimeId::LlamaServerCpu, RuntimeId::CandleCpu]
    }
}

fn safetensors_auto_candidates(manifest: &ModelManifest) -> Vec<RuntimeId> {
    if is_transformers_compat_model(manifest) {
        return vec![RuntimeId::TransformersCompat];
    }
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        vec![
            RuntimeId::MlxVlm,
            RuntimeId::Mlx,
            RuntimeId::CandleMetal,
            RuntimeId::CandleCpu,
        ]
    } else if cfg!(target_os = "macos") {
        vec![RuntimeId::CandleMetal, RuntimeId::CandleCpu]
    } else if cfg!(any(windows, target_os = "linux")) {
        let mut candidates = Vec::new();
        candidates.extend(vllm_auto_candidates(manifest));
        candidates.extend([RuntimeId::CandleCuda, RuntimeId::CandleCpu]);
        candidates
    } else {
        vec![RuntimeId::CandleCpu]
    }
}

fn vllm_auto_candidates(manifest: &ModelManifest) -> Vec<RuntimeId> {
    if runtime_supports_model(
        runtime_descriptor(RuntimeId::VllmCuda),
        &manifest.format,
        manifest.architecture.as_deref(),
    ) {
        vec![RuntimeId::VllmCuda]
    } else {
        Vec::new()
    }
}

fn cpu_candidates(manifest: &ModelManifest) -> Vec<RuntimeId> {
    match manifest.format {
        ModelFormat::Gguf => vec![RuntimeId::LlamaServerCpu],
        ModelFormat::SafeTensors => vec![RuntimeId::CandleCpu],
        ModelFormat::Onnx => vec![RuntimeId::OnnxRuntimeCpu],
        _ => Vec::new(),
    }
}

fn cuda_candidates(manifest: &ModelManifest) -> Vec<RuntimeId> {
    match manifest.format {
        ModelFormat::Gguf => vec![RuntimeId::LlamaServerCuda],
        ModelFormat::SafeTensors => {
            let mut candidates = Vec::new();
            candidates.extend(vllm_auto_candidates(manifest));
            candidates.push(RuntimeId::CandleCuda);
            candidates
        }
        ModelFormat::Onnx => vec![RuntimeId::OnnxRuntimeCuda],
        _ => Vec::new(),
    }
}

fn rocm_candidates(manifest: &ModelManifest) -> Vec<RuntimeId> {
    match manifest.format {
        ModelFormat::Gguf => vec![RuntimeId::LlamaServerRocm],
        ModelFormat::SafeTensors => vec![RuntimeId::VllmRocm],
        ModelFormat::Onnx => vec![RuntimeId::OnnxRuntimeRocm],
        _ => Vec::new(),
    }
}

fn burn_candidates() -> Vec<RuntimeId> {
    if cfg!(feature = "burn-cuda") && cfg!(any(windows, target_os = "linux")) {
        vec![RuntimeId::BurnCuda]
    } else if cfg!(feature = "burn-cpu") {
        vec![RuntimeId::BurnCpu]
    } else {
        Vec::new()
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
        ModelFormat::Gguf if cfg!(target_os = "macos") => {
            vec![RuntimeId::LlamaServerMetal, RuntimeId::CandleMetal]
        }
        ModelFormat::SafeTensors => vec![RuntimeId::CandleMetal],
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
    if runtime_id == RuntimeId::MlxVlm && !request_capabilities.image_input {
        return Some(
            "MLX-VLM is reserved for image requests; text-only MLX uses mlx-lm".to_string(),
        );
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
        (BackendRuntime::MlxVlm, ModelFormat::Mlx | ModelFormat::SafeTensors) => {
            "MLX-VLM is selected for supported VLM architectures".to_string()
        }
        (BackendRuntime::TransformersCompat, ModelFormat::SafeTensors) => {
            "Transformers compatibility is selected for raw ChatGLM/GLM repositories".to_string()
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
        (ModelFormat::SafeTensors, BackendRuntime::TransformersCompat, _) => {
            "raw ChatGLM/GLM compatibility route uses Transformers trust_remote_code".to_string()
        }
        (ModelFormat::SafeTensors, BackendRuntime::Vllm, RequestedBackend::Vllm) => {
            "explicit vLLM route requested".to_string()
        }
        (ModelFormat::SafeTensors, BackendRuntime::Vllm, _) => {
            "HF safetensors CUDA hot path uses vLLM for supported architectures".to_string()
        }
        (_, BackendRuntime::Candle, RequestedBackend::Candle) => {
            "explicit Candle route requested".to_string()
        }
        (_, BackendRuntime::Candle, _) => {
            "fallback runtime supports the selected model architecture".to_string()
        }
        (ModelFormat::Mlx, BackendRuntime::MlxVlm, _) => {
            "MLX VLM image request uses mlx-vlm".to_string()
        }
        (ModelFormat::Mlx, BackendRuntime::Mlx, _) => "MLX model uses mlx-lm".to_string(),
        (_, BackendRuntime::MlxVlm, _) => {
            "MLX VLM runtime selected for compatible model".to_string()
        }
        (_, BackendRuntime::Mlx, _) => "MLX runtime selected for compatible model".to_string(),
        (_, _, RequestedBackend::Cpu) => "best CPU runtime for this model".to_string(),
        (_, _, RequestedBackend::Cuda) => "best CUDA runtime for this model".to_string(),
        (_, _, RequestedBackend::Rocm) => "best ROCm runtime for this model".to_string(),
        (_, _, RequestedBackend::Vulkan) => "best Vulkan runtime for this model".to_string(),
        (_, _, RequestedBackend::Metal) => "best Metal runtime for this model".to_string(),
        (_, _, RequestedBackend::Transformers) => {
            "explicit Transformers compatibility route requested".to_string()
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_store::{ModelManifest, ModelSource};

    #[test]
    fn gguf_auto_prefers_llama_server_before_candle() {
        let manifest = manifest(ModelFormat::Gguf, Some("llama"));
        let candidates = runtime_candidate_ids(&manifest, RequestedBackend::Auto);
        if cfg!(any(windows, target_os = "linux")) {
            assert_eq!(candidates[0], RuntimeId::LlamaServerCuda);
            assert_eq!(candidates[1], RuntimeId::LlamaServerVulkan);
            assert_eq!(candidates[2], RuntimeId::LlamaServerCpu);
        } else if cfg!(target_os = "macos") {
            assert_eq!(candidates[0], RuntimeId::LlamaServerMetal);
            assert_eq!(candidates[1], RuntimeId::LlamaServerCpu);
        } else {
            assert_eq!(candidates[0], RuntimeId::LlamaServerCpu);
        }
        assert!(candidates.contains(&RuntimeId::CandleCpu));
    }

    #[test]
    fn safetensors_cuda_uses_vllm_then_candle_without_cpu_fallback() {
        let manifest = manifest(ModelFormat::SafeTensors, Some("phi3"));
        let candidates = runtime_candidate_ids(&manifest, RequestedBackend::Cuda);
        assert_eq!(candidates[0], RuntimeId::VllmCuda);
        assert!(
            candidates
                .iter()
                .position(|id| *id == RuntimeId::VllmCuda)
                .unwrap()
                < candidates
                    .iter()
                    .position(|id| *id == RuntimeId::CandleCuda)
                    .unwrap()
        );
        assert!(candidates.contains(&RuntimeId::CandleCuda));
        assert!(!candidates.contains(&RuntimeId::CandleCpu));
        assert!(!candidates.contains(&RuntimeId::BurnCuda));
        assert!(!candidates.contains(&RuntimeId::BurnCpu));
    }

    #[test]
    fn safetensors_auto_tries_vllm_before_candle_for_supported_architectures() {
        let qwen = manifest(ModelFormat::SafeTensors, Some("qwen2"));
        let candidates = runtime_candidate_ids(&qwen, RequestedBackend::Auto);
        if cfg!(any(windows, target_os = "linux")) {
            assert_eq!(candidates[0], RuntimeId::VllmCuda);
            assert!(
                candidates
                    .iter()
                    .position(|id| *id == RuntimeId::VllmCuda)
                    .unwrap()
                    < candidates
                        .iter()
                        .position(|id| *id == RuntimeId::CandleCuda)
                        .unwrap()
            );
            assert!(candidates.contains(&RuntimeId::CandleCpu));
        }
        assert!(!candidates.contains(&RuntimeId::BurnCuda));
        assert!(!candidates.contains(&RuntimeId::BurnCpu));
    }

    #[test]
    fn safetensors_auto_omits_burn_and_keeps_cpu_only_as_auto_fallback() {
        let unknown = manifest(ModelFormat::SafeTensors, Some("unknown"));
        let candidates = runtime_candidate_ids(&unknown, RequestedBackend::Auto);
        if cfg!(any(windows, target_os = "linux")) {
            assert_eq!(
                candidates,
                vec![RuntimeId::CandleCuda, RuntimeId::CandleCpu]
            );
        }
        assert!(!candidates.contains(&RuntimeId::BurnCuda));
        assert!(!candidates.contains(&RuntimeId::BurnCpu));
    }

    #[test]
    fn chatglm_safetensors_auto_uses_transformers_compatibility_route() {
        let chatglm = manifest(ModelFormat::SafeTensors, Some("chatglm"));
        let candidates = runtime_candidate_ids(&chatglm, RequestedBackend::Auto);

        assert_eq!(candidates, vec![RuntimeId::TransformersCompat]);
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
        if cfg!(feature = "burn-cuda") && cfg!(any(windows, target_os = "linux")) {
            assert_eq!(candidates, vec![RuntimeId::BurnCuda]);
        } else if cfg!(feature = "burn-cpu") {
            assert_eq!(candidates, vec![RuntimeId::BurnCpu]);
        } else {
            assert!(candidates.is_empty());
        }
    }

    #[test]
    fn explicit_rocm_routes_to_rocm_candidates_only_for_compatible_formats() {
        let safetensors = manifest(ModelFormat::SafeTensors, Some("qwen3"));
        assert_eq!(
            runtime_candidate_ids(&safetensors, RequestedBackend::Rocm),
            vec![RuntimeId::VllmRocm]
        );
        let unknown_safetensors = manifest(ModelFormat::SafeTensors, Some("unknown"));
        assert_eq!(
            runtime_candidate_ids(&unknown_safetensors, RequestedBackend::Rocm),
            vec![RuntimeId::VllmRocm]
        );

        let gguf = manifest(ModelFormat::Gguf, Some("llama"));
        assert_eq!(
            runtime_candidate_ids(&gguf, RequestedBackend::Rocm),
            vec![RuntimeId::LlamaServerRocm]
        );

        let onnx = manifest(ModelFormat::Onnx, None);
        assert_eq!(
            runtime_candidate_ids(&onnx, RequestedBackend::Rocm),
            vec![RuntimeId::OnnxRuntimeRocm]
        );
    }

    #[test]
    fn gguf_auto_adds_rocm_only_when_availability_mentions_it() {
        let manifest = manifest(ModelFormat::Gguf, Some("llama"));
        let plain = plan_runtime(
            &manifest,
            RequestedBackend::Auto,
            RequestCapabilities::text(true),
            &[],
        );
        assert!(
            !plain
                .candidates
                .iter()
                .any(|decision| decision.runtime_id == RuntimeId::LlamaServerRocm)
        );

        let gated = plan_runtime(
            &manifest,
            RequestedBackend::Auto,
            RequestCapabilities::text(true),
            &[RuntimeAvailability {
                runtime_id: RuntimeId::LlamaServerRocm,
                available: false,
                reason: Some("ROCm probe unavailable".to_string()),
            }],
        );
        let ids = gated
            .candidates
            .iter()
            .map(|decision| decision.runtime_id)
            .collect::<Vec<_>>();
        let rocm = ids
            .iter()
            .position(|id| *id == RuntimeId::LlamaServerRocm)
            .unwrap();
        let first_lower_priority_llama = ids
            .iter()
            .position(|id| {
                matches!(
                    id,
                    RuntimeId::LlamaServerVulkan
                        | RuntimeId::LlamaServerMetal
                        | RuntimeId::LlamaServerCpu
                )
            })
            .unwrap();
        assert!(rocm <= first_lower_priority_llama);
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
    fn gemma4_unified_image_request_prefers_mlx_vlm() {
        let manifest = manifest(ModelFormat::Mlx, Some("gemma4_unified"));
        let available = [
            RuntimeAvailability {
                runtime_id: RuntimeId::MlxVlm,
                available: true,
                reason: None,
            },
            RuntimeAvailability {
                runtime_id: RuntimeId::Mlx,
                available: true,
                reason: None,
            },
        ];

        let selected = select_runtime(
            &manifest,
            RequestedBackend::Auto,
            RequestCapabilities::text_with_images(true, true),
            &available,
        )
        .unwrap();

        assert_eq!(selected.runtime_id, RuntimeId::MlxVlm);
    }

    #[test]
    fn gemma4_unified_text_request_uses_mlx_not_mlx_vlm() {
        let manifest = manifest(ModelFormat::Mlx, Some("gemma4_unified"));
        let available = [
            RuntimeAvailability {
                runtime_id: RuntimeId::MlxVlm,
                available: true,
                reason: None,
            },
            RuntimeAvailability {
                runtime_id: RuntimeId::Mlx,
                available: true,
                reason: None,
            },
        ];

        let plan = plan_runtime(
            &manifest,
            RequestedBackend::Auto,
            RequestCapabilities::text(true),
            &available,
        );

        assert_eq!(plan.selected.unwrap().runtime_id, RuntimeId::Mlx);
        assert!(
            plan.candidates
                .iter()
                .any(|decision| decision.runtime_id == RuntimeId::MlxVlm
                    && decision.reason.contains("text-only MLX uses mlx-lm"))
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
