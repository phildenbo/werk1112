use anyhow::{Result, bail};

use super::{GenerateRequest, GenerateResponse, GenerateStream, GenerationBackend};
use crate::model_store::{ModelManifest, ModelStore};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BurnRuntimeMode {
    Cuda,
    Wgpu,
    Cpu,
}

#[derive(Clone)]
pub struct BurnBackend {
    _store: ModelStore,
    mode: BurnRuntimeMode,
}

impl BurnBackend {
    pub fn new(store: ModelStore, mode: BurnRuntimeMode) -> Self {
        Self {
            _store: store,
            mode,
        }
    }

    pub fn probe(mode: BurnRuntimeMode) -> Result<String> {
        bail!("{}", pending_message(mode))
    }
}

impl GenerationBackend for BurnBackend {
    fn generate(
        &self,
        _manifest: &ModelManifest,
        _request: GenerateRequest,
    ) -> Result<GenerateResponse> {
        bail!("{}", pending_message(self.mode))
    }

    fn generate_stream(
        &self,
        _manifest: ModelManifest,
        _request: GenerateRequest,
    ) -> GenerateStream {
        Box::pin(tokio_stream::iter(vec![Err(pending_message(self.mode))]))
    }
}

fn pending_message(mode: BurnRuntimeMode) -> String {
    format!(
        "Burn {} backend is not implemented yet",
        match mode {
            BurnRuntimeMode::Cuda => "CUDA",
            BurnRuntimeMode::Wgpu => "WGPU/Vulkan",
            BurnRuntimeMode::Cpu => "CPU",
        }
    )
}
