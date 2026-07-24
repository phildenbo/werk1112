use serde::{Deserialize, Serialize};

string_enum! {
    pub enum ModelComponentKind {
        MainModel => "main-model",
        Transformer => "transformer",
        Unet => "unet",
        Vae => "vae",
        TextEncoder => "text-encoder",
        #[serde(rename = "text_encoder_2")]
        TextEncoder2 => "text-encoder-2",
        Tokenizer => "tokenizer",
        #[serde(rename = "tokenizer_2")]
        Tokenizer2 => "tokenizer-2",
        Scheduler => "scheduler",
        Encoder => "encoder",
        Decoder => "decoder",
        Vocoder => "vocoder",
        FeatureExtractor => "feature-extractor",
        #[serde(rename = "controlnet")]
        ControlNet => "controlnet",
        Adapter => "adapter"
    }
}

/// A repository component, kept separate from the flat file inventory so
/// multi-component repositories remain one installed model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelComponent {
    pub kind: ModelComponentKind,
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<String>,
}

impl ModelComponent {
    pub fn new(kind: ModelComponentKind, path: impl Into<String>) -> Self {
        Self {
            kind,
            path: path.into(),
            format: None,
            precision: None,
            quantization: None,
            files: Vec::new(),
        }
    }
}
