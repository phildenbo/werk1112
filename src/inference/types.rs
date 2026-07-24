use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    str::FromStr,
};

use crate::capabilities::{InferenceTask, InputModality, OutputModality};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParameterValue {
    Null,
    Boolean(bool),
    Integer(i64),
    Number(f64),
    String(String),
    List(Vec<ParameterValue>),
    Object(BTreeMap<String, ParameterValue>),
}

impl ParameterValue {
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Boolean(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Integer(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        self.as_i64().and_then(|value| value.try_into().ok())
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Integer(value) => Some(*value as f64),
            Self::Number(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn from_json(value: Value) -> Result<Self> {
        serde_json::from_value(value).map_err(Into::into)
    }
}

impl From<bool> for ParameterValue {
    fn from(value: bool) -> Self {
        Self::Boolean(value)
    }
}

impl From<u32> for ParameterValue {
    fn from(value: u32) -> Self {
        Self::Integer(i64::from(value))
    }
}

impl From<u64> for ParameterValue {
    fn from(value: u64) -> Self {
        Self::Integer(i64::try_from(value).unwrap_or(i64::MAX))
    }
}

impl From<i64> for ParameterValue {
    fn from(value: i64) -> Self {
        Self::Integer(value)
    }
}

impl From<f64> for ParameterValue {
    fn from(value: f64) -> Self {
        Self::Number(value)
    }
}

impl From<String> for ParameterValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for ParameterValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OverrideBool {
    #[default]
    Inherit,
    Enabled,
    Disabled,
}

impl OverrideBool {
    pub fn resolve(self, inherited: bool) -> bool {
        match self {
            Self::Inherit => inherited,
            Self::Enabled => true,
            Self::Disabled => false,
        }
    }

    pub fn explicit(self) -> Option<bool> {
        match self {
            Self::Inherit => None,
            Self::Enabled => Some(true),
            Self::Disabled => Some(false),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "operation", content = "values", rename_all = "snake_case")]
pub enum ListOverride<T> {
    #[default]
    Inherit,
    Replace(Vec<T>),
    Add(Vec<T>),
    Clear,
}

impl<T: Clone> ListOverride<T> {
    pub fn resolve(&self, inherited: &[T]) -> Vec<T> {
        match self {
            Self::Inherit => inherited.to_vec(),
            Self::Replace(values) => values.clone(),
            Self::Add(values) => inherited.iter().cloned().chain(values.clone()).collect(),
            Self::Clear => Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ImageGenerationOverrides {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub steps: Option<u32>,
    pub guidance: Option<f64>,
    pub seed: Option<u64>,
    pub vae_tiling: OverrideBool,
    pub loras: ListOverride<ParameterValue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectiveImageGenerationOptions {
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: f64,
    pub seed: u64,
    pub vae_tiling: bool,
    pub loras: Vec<ParameterValue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct VideoGenerationOverrides {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub frames: Option<u32>,
    pub fps: Option<f64>,
    pub steps: Option<u32>,
    pub seed: Option<u64>,
    pub temporal_vae_tiling: OverrideBool,
    pub prompt_keyframes: ListOverride<ParameterValue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectiveVideoGenerationOptions {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: f64,
    pub steps: u32,
    pub seed: u64,
    pub temporal_vae_tiling: bool,
    pub prompt_keyframes: Vec<ParameterValue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AudioGenerationOverrides {
    pub duration: Option<f64>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u32>,
    pub seed: Option<u64>,
    pub instrumental: OverrideBool,
    pub instruments: ListOverride<ParameterValue>,
    pub stems: ListOverride<ParameterValue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectiveAudioGenerationOptions {
    pub duration: f64,
    pub sample_rate: u32,
    pub channels: u32,
    pub seed: u64,
    pub instrumental: bool,
    pub instruments: Vec<ParameterValue>,
    pub stems: Vec<ParameterValue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterType {
    Boolean,
    Integer,
    Number,
    String,
    Path,
    Enumeration,
    List,
    Object,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterDescriptor {
    pub path: String,
    pub cli_flag: String,
    pub value_type: ParameterType,
    pub label: String,
    pub description: String,
    pub category: String,
    pub default: Option<ParameterValue>,
    pub minimum: Option<ParameterValue>,
    pub maximum: Option<ParameterValue>,
    pub step: Option<ParameterValue>,
    #[serde(default)]
    pub allowed_values: Vec<ParameterValue>,
    pub repeatable: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub list_override_operations: Vec<String>,
    pub advanced: bool,
    pub affects_memory: bool,
    pub affects_quality: bool,
    pub affects_runtime: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterSource {
    SystemDefault,
    TaskDefault,
    ModelFamilyDefault,
    ModelDefault,
    RuntimeDefault,
    HardwareProfile,
    UserProfile,
    RequestOverride,
    BackendAdjustment,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedParameter {
    pub value: ParameterValue,
    pub source: ParameterSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ParameterPolicy {
    #[default]
    Strict,
    Warn,
    Permissive,
}

impl fmt::Display for ParameterPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Strict => "strict",
            Self::Warn => "warn",
            Self::Permissive => "permissive",
        })
    }
}

impl FromStr for ParameterPolicy {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "strict" => Ok(Self::Strict),
            "warn" => Ok(Self::Warn),
            "permissive" => Ok(Self::Permissive),
            _ => bail!("unknown parameter policy '{value}'"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterSupportStatus {
    Native,
    Translated,
    Emulated,
    Ignored,
    Unsupported,
    ModelDependent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterSupport {
    pub path: String,
    pub status: ParameterSupportStatus,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InferenceInputSource {
    Path { path: String },
    Url { url: String },
    Base64 { data: String },
    Text { text: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceInput {
    pub modality: InputModality,
    pub role: String,
    pub source: InferenceInputSource,
    pub mime_type: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct RoutingOverrides {
    pub backend: Option<String>,
    pub accelerator: Option<String>,
    pub device: Option<String>,
    pub precision: Option<String>,
    pub quantization: Option<String>,
    pub profile: Option<String>,
    pub quality: Option<String>,
    pub performance_preference: Option<String>,
    pub fallback_policy: Option<String>,
    pub parameter_policy: ParameterPolicy,
    pub allow_cpu_offload: OverrideBool,
    pub allow_sequential_offload: OverrideBool,
    pub allow_component_offload: OverrideBool,
    pub allow_disk_offload: OverrideBool,
    pub attention_backend: Option<String>,
    pub compile: OverrideBool,
    pub timeout_seconds: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub model: String,
    pub task: InferenceTask,
    pub prompt: Option<String>,
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub inputs: Vec<InferenceInput>,
    #[serde(default)]
    pub parameters: BTreeMap<String, ParameterValue>,
    #[serde(default)]
    pub routing: RoutingOverrides,
}

impl InferenceRequest {
    pub fn new(model: impl Into<String>, task: InferenceTask) -> Self {
        Self {
            model: model.into(),
            task,
            prompt: None,
            negative_prompt: None,
            inputs: Vec::new(),
            parameters: BTreeMap::new(),
            routing: RoutingOverrides::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectiveInferenceRequest {
    pub model: String,
    pub task: InferenceTask,
    pub prompt: Option<String>,
    pub negative_prompt: Option<String>,
    pub inputs: Vec<InferenceInput>,
    pub output_modality: OutputModality,
    pub parameters: BTreeMap<String, ResolvedParameter>,
    pub explicit_parameters: BTreeSet<String>,
    pub parameter_policy: ParameterPolicy,
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl EffectiveInferenceRequest {
    pub fn parameter(&self, path: &str) -> Option<&ParameterValue> {
        self.parameters.get(path).map(|value| &value.value)
    }

    pub fn bool_parameter(&self, path: &str) -> Option<bool> {
        self.parameter(path).and_then(ParameterValue::as_bool)
    }

    pub fn u64_parameter(&self, path: &str) -> Option<u64> {
        self.parameter(path).and_then(ParameterValue::as_u64)
    }

    pub fn f64_parameter(&self, path: &str) -> Option<f64> {
        self.parameter(path).and_then(ParameterValue::as_f64)
    }

    pub fn string_parameter(&self, path: &str) -> Option<&str> {
        self.parameter(path).and_then(ParameterValue::as_str)
    }

    pub fn values_only(&self) -> BTreeMap<String, ParameterValue> {
        self.parameters
            .iter()
            .map(|(path, value)| (path.clone(), value.value.clone()))
            .collect()
    }

    pub fn image_generation_options(&self) -> Option<EffectiveImageGenerationOptions> {
        (self.task.parameter_namespace() == "image").then_some(())?;
        Some(EffectiveImageGenerationOptions {
            width: self.u64_parameter("image.width")?.try_into().ok()?,
            height: self.u64_parameter("image.height")?.try_into().ok()?,
            steps: self.u64_parameter("image.steps")?.try_into().ok()?,
            guidance: self.f64_parameter("image.guidance")?,
            seed: self.u64_parameter("image.seed")?,
            vae_tiling: self.bool_parameter("image.vae_tiling")?,
            loras: list_parameter(self.parameter("image.loras"))?,
        })
    }

    pub fn video_generation_options(&self) -> Option<EffectiveVideoGenerationOptions> {
        (self.task.parameter_namespace() == "video").then_some(())?;
        Some(EffectiveVideoGenerationOptions {
            width: self.u64_parameter("video.width")?.try_into().ok()?,
            height: self.u64_parameter("video.height")?.try_into().ok()?,
            frames: self.u64_parameter("video.frames")?.try_into().ok()?,
            fps: self.f64_parameter("video.fps")?,
            steps: self.u64_parameter("video.steps")?.try_into().ok()?,
            seed: self.u64_parameter("video.seed")?,
            temporal_vae_tiling: self.bool_parameter("video.temporal_vae_tiling")?,
            prompt_keyframes: list_parameter(self.parameter("video.prompt_keyframes"))?,
        })
    }

    pub fn audio_generation_options(&self) -> Option<EffectiveAudioGenerationOptions> {
        (self.task.parameter_namespace() == "audio").then_some(())?;
        Some(EffectiveAudioGenerationOptions {
            duration: self.f64_parameter("audio.duration")?,
            sample_rate: self.u64_parameter("audio.sample_rate")?.try_into().ok()?,
            channels: self.u64_parameter("audio.channels")?.try_into().ok()?,
            seed: self.u64_parameter("audio.seed")?,
            instrumental: self.bool_parameter("audio.instrumental")?,
            instruments: list_parameter(self.parameter("audio.instruments"))?,
            stems: list_parameter(self.parameter("audio.stems"))?,
        })
    }
}

fn list_parameter(value: Option<&ParameterValue>) -> Option<Vec<ParameterValue>> {
    match value? {
        ParameterValue::List(values) => Some(values.clone()),
        _ => None,
    }
}

#[derive(Debug, Clone, Default)]
pub struct ResolutionContext {
    pub runtime_defaults: BTreeMap<String, ParameterValue>,
    pub hardware_profile: BTreeMap<String, ParameterValue>,
    pub user_profile: BTreeMap<String, ParameterValue>,
    pub parameter_support: BTreeMap<String, ParameterSupportStatus>,
}
