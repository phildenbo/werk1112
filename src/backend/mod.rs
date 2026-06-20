mod candle;
mod external;

use anyhow::Result;
use std::pin::Pin;
use tokio_stream::Stream;

pub use candle::{CandleBackend, CandleDeviceMode, probe_device};
pub use external::{LlamaCppBackend, MlxBackend};

use crate::model_store::ModelManifest;

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub image_urls: Vec<String>,
    pub max_tokens: usize,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub stop: Vec<String>,
    pub seed: Option<u64>,
    pub stream_granularity: StreamGranularity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamGranularity {
    Token,
    Chunk,
}

#[derive(Debug, Clone)]
pub struct GenerateResponse {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
    pub timings: GenerationTimings,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GenerationTimings {
    pub load_seconds: f64,
    pub prompt_seconds: f64,
    pub decode_seconds: f64,
    pub total_seconds: f64,
}

#[derive(Debug, Clone)]
pub enum GenerateStreamEvent {
    TextChunk(String),
    Done {
        finish_reason: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        timings: GenerationTimings,
    },
}

pub type GenerateStream =
    Pin<Box<dyn Stream<Item = Result<GenerateStreamEvent, String>> + Send + 'static>>;

pub trait GenerationBackend: Send + Sync {
    fn generate(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> Result<GenerateResponse>;
    fn generate_stream(&self, manifest: ModelManifest, request: GenerateRequest) -> GenerateStream;
}
