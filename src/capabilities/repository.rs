string_enum! {
    pub enum RepositoryLayout {
        SingleFile => "single-file",
        Gguf => "gguf",
        Transformers => "transformers",
        Diffusers => "diffusers",
        Mlx => "mlx",
        OnnxBundle => "onnx-bundle",
        #[serde(rename = "tensorrt_engine")]
        TensorRtEngine => "tensorrt-engine",
        Custom => "custom"
    }
}

#[allow(clippy::derivable_impls)]
impl Default for RepositoryLayout {
    fn default() -> Self {
        Self::Custom
    }
}
