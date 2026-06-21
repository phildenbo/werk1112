mod candle;
mod external;
mod llama_fast;

use anyhow::Result;
use std::pin::Pin;
use tokio_stream::Stream;

pub use candle::{CandleBackend, CandleDeviceMode, probe_device};
pub use external::{LlamaCppBackend, LlamaCppMode, MlxBackend};
pub use llama_fast::{LlamaFastBackend, LlamaFastRuntimeReport};

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
    pub warmup_seconds: f64,
    pub first_token_seconds: f64,
    pub prompt_seconds: f64,
    pub decode_seconds: f64,
    pub total_seconds: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum LlamaKvCacheType {
    F16,
    F32,
    Q8_0,
}

impl LlamaKvCacheType {
    pub fn label(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::Q8_0 => "q8_0",
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LlamaRuntimeOptions {
    pub ctx_size: Option<usize>,
    pub batch_size: Option<usize>,
    pub ubatch_size: Option<u32>,
    pub gpu_layers: Option<i32>,
    pub main_gpu: Option<i32>,
    pub kv_cache_type: Option<LlamaKvCacheType>,
    pub flash_attn: Option<bool>,
    pub kv_offload: Option<bool>,
    pub warmup_tokens: Option<usize>,
    pub threads: Option<u32>,
    pub threads_batch: Option<u32>,
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

pub trait ChatGenerationSession: Send + Sync {
    fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse>;
    fn generate_stream(&self, request: GenerateRequest) -> GenerateStream;
}

pub trait GenerationBackend: Send + Sync {
    fn prepare(&self, _manifest: &ModelManifest) -> Result<()> {
        Ok(())
    }

    fn start_chat_session(
        &self,
        _manifest: &ModelManifest,
        _seed: Option<u64>,
    ) -> Result<Option<Box<dyn ChatGenerationSession>>> {
        Ok(None)
    }

    fn generate(
        &self,
        manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> Result<GenerateResponse>;
    fn generate_stream(&self, manifest: ModelManifest, request: GenerateRequest) -> GenerateStream;
}
