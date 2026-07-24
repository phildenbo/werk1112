//! Shared media inference façade used by both the CLI and HTTP API.
//!
//! Implementation details live in focused submodules. Public re-exports keep
//! the historical `crate::inference_service::*` API stable.

mod backend;
mod companion;
mod helpers;
mod jobs;
mod output;
mod resources;
mod service;
mod types;

pub use backend::{BackendExecution, BackendOutput, BackendProbe, MediaInferenceBackend};
pub use companion::CompanionMediaBackend;
pub use jobs::{JobManager, JobRecord, JobStatus, JobStore};
pub use output::OutputStore;
pub use resources::detect_host_resources;
pub use service::InferenceService;
pub use types::{InferenceResult, OutputMetadata};

#[cfg(test)]
mod tests;
