//! Canonical, backend-neutral inference requests.
//!
//! CLI and HTTP frontends intentionally meet in this module before a runtime is
//! selected. Requests contain only overrides; [`resolve_request`] produces a
//! complete, validated request with provenance for every parameter.

mod conversation;
mod estimate;
mod planner;
mod resolution;
mod schema;
mod types;

pub use conversation::{ConversationContent, MediaContent, ToolCallContent, ToolResultContent};
pub use estimate::{
    EstimateConfidence, FitAssessment, HostResources, WorkloadEstimate, estimate_workload,
};
pub use planner::{
    ExecutionDegradation, ExecutionPlan, InferenceRuntimeCandidate, PlanCandidateDecision,
    PlanCandidateStatus, RuntimeAccelerator, plan_execution,
};
pub use resolution::resolve_request;
pub use schema::{parameter_schema, parameter_schema_for_manifest};
pub use types::{
    AudioGenerationOverrides, EffectiveAudioGenerationOptions, EffectiveImageGenerationOptions,
    EffectiveInferenceRequest, EffectiveVideoGenerationOptions, ImageGenerationOverrides,
    InferenceInput, InferenceInputSource, InferenceRequest, ListOverride, OverrideBool,
    ParameterDescriptor, ParameterPolicy, ParameterSource, ParameterSupport,
    ParameterSupportStatus, ParameterType, ParameterValue, ResolutionContext, ResolvedParameter,
    RoutingOverrides, VideoGenerationOverrides,
};

pub(crate) use estimate::classify_workload_fit;

#[cfg(test)]
mod tests;
