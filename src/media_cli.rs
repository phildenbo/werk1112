//! Clap contracts for image, video, and audio inference.
//!
//! This module deliberately contains transport-level CLI overrides only. It does
//! not resolve defaults or depend on the media inference core, so it can be
//! shared by the eventual CLI-to-request translation layer without creating a
//! second execution pipeline.

use clap::{ArgAction, Args, Subcommand, ValueEnum};
use serde::Serialize;
use serde_json::Value;
use std::{collections::BTreeMap, path::PathBuf};

/// A backend-neutral map of explicitly supplied CLI values.
pub type RawOverrideMap = BTreeMap<String, Value>;

/// Converts a serializable argument struct into dotted raw override paths.
///
/// `None`, `false`, empty arrays, and empty objects are omitted. Consequently,
/// positive and negative boolean flags only appear when they were explicitly
/// selected.
pub fn collect_raw_overrides<T: Serialize>(value: &T) -> Result<RawOverrideMap, serde_json::Error> {
    let value = serde_json::to_value(value)?;
    let mut overrides = BTreeMap::new();
    flatten_value("", &value, &mut overrides);
    Ok(overrides)
}

/// Parses repeatable `--set path=value` values.
///
/// Values that are valid JSON retain their JSON type; all other values become
/// strings. Repeating a path uses last-write-wins semantics.
pub fn parse_set_overrides(values: &[String]) -> Result<RawOverrideMap, String> {
    let mut overrides = BTreeMap::new();
    for entry in values {
        let Some((path, raw_value)) = entry.split_once('=') else {
            return Err(format!(
                "invalid --set value '{entry}': expected path=value"
            ));
        };
        let path = path.trim();
        if path.is_empty() {
            return Err(format!(
                "invalid --set value '{entry}': path must not be empty"
            ));
        }
        let raw_value = raw_value.trim();
        let value = serde_json::from_str(raw_value)
            .unwrap_or_else(|_| Value::String(raw_value.to_string()));
        overrides.insert(path.to_string(), value);
    }
    Ok(overrides)
}

fn flatten_value(prefix: &str, value: &Value, overrides: &mut RawOverrideMap) {
    match value {
        Value::Null | Value::Bool(false) => {}
        Value::Array(items) if items.is_empty() => {}
        Value::Object(fields) if fields.is_empty() => {}
        Value::Object(fields) => {
            for (name, value) in fields {
                let path = if prefix.is_empty() {
                    name.clone()
                } else {
                    format!("{prefix}.{name}")
                };
                flatten_value(&path, value, overrides);
            }
        }
        value if !prefix.is_empty() => {
            overrides.insert(prefix.to_string(), value.clone());
        }
        _ => {}
    }
}

/// The resolved meaning of a positive/negative flag pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOverride {
    Inherit,
    Enabled,
    Disabled,
}

pub fn bool_override(enabled: bool, disabled: bool) -> BoolOverride {
    match (enabled, disabled) {
        (true, false) => BoolOverride::Enabled,
        (false, true) => BoolOverride::Disabled,
        _ => BoolOverride::Inherit,
    }
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct RoutingArgs {
    #[arg(
        long,
        help = "Requested accelerator, for example cpu, cuda, rocm, metal, or auto"
    )]
    pub accelerator: Option<String>,

    #[arg(long, help = "Requested compute precision")]
    pub precision: Option<String>,

    #[arg(long, help = "Requested quantization")]
    pub quantization: Option<String>,

    #[arg(long, help = "Stored or built-in execution profile")]
    pub profile: Option<String>,

    #[arg(long, help = "Quality profile")]
    pub quality: Option<String>,

    #[arg(
        long,
        help = "Performance preference, for example speed, balanced, memory, or quality"
    )]
    pub performance_preference: Option<String>,

    #[arg(long, help = "Backend fallback policy")]
    pub fallback_policy: Option<String>,

    #[arg(long, help = "Parameter support policy: strict, warn, or permissive")]
    pub parameter_policy: Option<String>,

    #[arg(
        long = "allow-cpu-offload",
        action = ArgAction::SetTrue,
        conflicts_with = "no_allow_cpu_offload"
    )]
    pub allow_cpu_offload: bool,

    #[arg(
        long = "no-allow-cpu-offload",
        action = ArgAction::SetTrue,
        conflicts_with = "allow_cpu_offload"
    )]
    pub no_allow_cpu_offload: bool,

    #[arg(
        long = "allow-sequential-offload",
        action = ArgAction::SetTrue,
        conflicts_with = "no_allow_sequential_offload"
    )]
    pub allow_sequential_offload: bool,

    #[arg(
        long = "no-allow-sequential-offload",
        action = ArgAction::SetTrue,
        conflicts_with = "allow_sequential_offload"
    )]
    pub no_allow_sequential_offload: bool,

    #[arg(
        long = "allow-component-offload",
        action = ArgAction::SetTrue,
        conflicts_with = "no_allow_component_offload"
    )]
    pub allow_component_offload: bool,

    #[arg(
        long = "no-allow-component-offload",
        action = ArgAction::SetTrue,
        conflicts_with = "allow_component_offload"
    )]
    pub no_allow_component_offload: bool,

    #[arg(
        long = "allow-disk-offload",
        action = ArgAction::SetTrue,
        conflicts_with = "no_allow_disk_offload"
    )]
    pub allow_disk_offload: bool,

    #[arg(
        long = "no-allow-disk-offload",
        action = ArgAction::SetTrue,
        conflicts_with = "allow_disk_offload"
    )]
    pub no_allow_disk_offload: bool,

    #[arg(long, help = "Requested attention implementation")]
    pub attention_backend: Option<String>,

    #[arg(
        long,
        action = ArgAction::SetTrue,
        conflicts_with = "no_compile",
        help = "Explicitly enable runtime compilation"
    )]
    pub compile: bool,

    #[arg(
        long = "no-compile",
        action = ArgAction::SetTrue,
        conflicts_with = "compile",
        help = "Explicitly disable runtime compilation"
    )]
    pub no_compile: bool,

    #[arg(long, value_name = "SECONDS", help = "Execution timeout in seconds")]
    pub timeout: Option<u64>,

    #[arg(
        long = "set",
        value_name = "PATH=VALUE",
        action = ArgAction::Append,
        help = "Set an advanced raw override; may be repeated"
    )]
    pub set: Vec<String>,
}

impl RoutingArgs {
    pub fn cpu_offload(&self) -> BoolOverride {
        bool_override(self.allow_cpu_offload, self.no_allow_cpu_offload)
    }

    pub fn sequential_offload(&self) -> BoolOverride {
        bool_override(
            self.allow_sequential_offload,
            self.no_allow_sequential_offload,
        )
    }

    pub fn component_offload(&self) -> BoolOverride {
        bool_override(
            self.allow_component_offload,
            self.no_allow_component_offload,
        )
    }

    pub fn disk_offload(&self) -> BoolOverride {
        bool_override(self.allow_disk_offload, self.no_allow_disk_offload)
    }

    pub fn compilation(&self) -> BoolOverride {
        bool_override(self.compile, self.no_compile)
    }
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct PromptArgs {
    #[arg(long, help = "Prompt text")]
    pub prompt: Option<String>,

    #[arg(long, value_name = "PATH", help = "Read prompt text from a file")]
    pub prompt_file: Option<PathBuf>,

    #[arg(long, help = "Negative prompt text")]
    pub negative_prompt: Option<String>,

    #[arg(
        long,
        value_name = "PATH",
        help = "Read negative prompt text from a file"
    )]
    pub negative_prompt_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct TextInputArgs {
    #[arg(long, help = "Input text")]
    pub text: Option<String>,

    #[arg(long, value_name = "PATH", help = "Read input text from a file")]
    pub text_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct LyricsArgs {
    #[arg(long, help = "Song lyrics")]
    pub lyrics: Option<String>,

    #[arg(long, value_name = "PATH", help = "Read song lyrics from a file")]
    pub lyrics_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct OutputArgs {
    #[arg(long = "output-format", help = "Output file format")]
    pub output_format: Option<String>,

    #[arg(
        long = "output",
        alias = "output-path",
        value_name = "PATH",
        help = "Output file or directory"
    )]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct ImageDimensionsArgs {
    #[arg(long)]
    pub width: Option<u32>,

    #[arg(long)]
    pub height: Option<u32>,

    #[arg(long)]
    pub aspect_ratio: Option<String>,

    #[arg(long)]
    pub resize_mode: Option<String>,

    #[arg(long)]
    pub crop_mode: Option<String>,

    #[arg(long)]
    pub batch_size: Option<u32>,

    #[arg(long, help = "Number of output images")]
    pub num_images: Option<u32>,
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct ImageSamplingArgs {
    #[arg(long)]
    pub seed: Option<u64>,

    #[arg(long)]
    pub subseed: Option<u64>,

    #[arg(long)]
    pub variation_strength: Option<f64>,

    #[arg(long)]
    pub steps: Option<u32>,

    #[arg(long, alias = "cfg")]
    pub guidance: Option<f64>,

    #[arg(long)]
    pub guidance_rescale: Option<f64>,

    #[arg(long)]
    pub true_cfg: Option<f64>,

    #[arg(long)]
    pub sampler: Option<String>,

    #[arg(long)]
    pub scheduler: Option<String>,

    #[arg(long)]
    pub prediction_type: Option<String>,

    #[arg(long)]
    pub eta: Option<f64>,

    #[arg(long)]
    pub denoise_strength: Option<f64>,

    #[arg(long)]
    pub noise_strength: Option<f64>,

    #[arg(long)]
    pub sigma_min: Option<f64>,

    #[arg(long)]
    pub sigma_max: Option<f64>,

    #[arg(long)]
    pub rho: Option<f64>,

    #[arg(long = "sigma", action = ArgAction::Append)]
    pub sigmas: Vec<f64>,

    #[arg(long)]
    pub shift: Option<f64>,

    #[arg(
        long,
        action = ArgAction::SetTrue,
        conflicts_with = "no_dynamic_shift"
    )]
    pub dynamic_shift: bool,

    #[arg(
        long = "no-dynamic-shift",
        action = ArgAction::SetTrue,
        conflicts_with = "dynamic_shift"
    )]
    pub no_dynamic_shift: bool,

    #[arg(long)]
    pub clip_skip: Option<u32>,

    #[arg(long)]
    pub prompt_weighting: Option<String>,

    #[arg(long)]
    pub prompt_token_limit: Option<u32>,
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct ImageConditioningArgs {
    #[arg(long, value_name = "PATH_OR_URL")]
    pub init_image: Option<String>,

    #[arg(long)]
    pub image_strength: Option<f64>,

    #[arg(long, value_name = "PATH")]
    pub mask: Option<PathBuf>,

    #[arg(long)]
    pub mask_blur: Option<u32>,

    #[arg(long)]
    pub mask_expand: Option<i32>,

    #[arg(
        long,
        action = ArgAction::SetTrue,
        conflicts_with = "no_mask_invert"
    )]
    pub mask_invert: bool,

    #[arg(
        long = "no-mask-invert",
        action = ArgAction::SetTrue,
        conflicts_with = "mask_invert"
    )]
    pub no_mask_invert: bool,

    #[arg(long)]
    pub mask_fill: Option<String>,

    #[arg(long)]
    pub inpaint_area: Option<String>,

    #[arg(long)]
    pub padding: Option<u32>,

    #[arg(
        long,
        action = ArgAction::SetTrue,
        conflicts_with = "no_preserve_unmasked"
    )]
    pub preserve_unmasked: bool,

    #[arg(
        long = "no-preserve-unmasked",
        action = ArgAction::SetTrue,
        conflicts_with = "preserve_unmasked"
    )]
    pub no_preserve_unmasked: bool,

    #[arg(long = "image-control", value_name = "SPEC", action = ArgAction::Append)]
    pub controls: Vec<String>,

    #[arg(long = "image-control-json", value_name = "JSON")]
    pub controls_json: Option<String>,

    #[arg(long = "image-control-file", value_name = "PATH")]
    pub controls_file: Option<PathBuf>,

    #[arg(long)]
    pub control_type: Option<String>,

    #[arg(long)]
    pub control_model: Option<String>,

    #[arg(long)]
    pub control_weight: Option<f64>,

    #[arg(long)]
    pub control_start: Option<f64>,

    #[arg(long)]
    pub control_end: Option<f64>,

    #[arg(long)]
    pub control_preprocessor: Option<String>,

    #[arg(long = "reference-image", value_name = "PATH_OR_URL", action = ArgAction::Append)]
    pub reference_images: Vec<String>,

    #[arg(long = "reference-image-json", value_name = "JSON")]
    pub reference_images_json: Option<String>,

    #[arg(long = "reference-image-file", value_name = "PATH")]
    pub reference_images_file: Option<PathBuf>,

    #[arg(long)]
    pub reference_weight: Option<f64>,

    #[arg(long)]
    pub reference_start: Option<f64>,

    #[arg(long)]
    pub reference_end: Option<f64>,

    #[arg(long)]
    pub identity_preservation: Option<f64>,

    #[arg(long)]
    pub color_preservation: Option<f64>,

    #[arg(long)]
    pub composition_preservation: Option<f64>,

    #[arg(long = "image-lora", value_name = "SPEC", action = ArgAction::Append)]
    pub loras: Vec<String>,

    #[arg(long = "image-lora-json", value_name = "JSON")]
    pub loras_json: Option<String>,

    #[arg(long = "image-lora-file", value_name = "PATH")]
    pub loras_file: Option<PathBuf>,

    #[arg(long = "image-adapter", value_name = "SPEC", action = ArgAction::Append)]
    pub adapters: Vec<String>,

    #[arg(long = "image-adapter-json", value_name = "JSON")]
    pub adapters_json: Option<String>,

    #[arg(long = "image-adapter-file", value_name = "PATH")]
    pub adapters_file: Option<PathBuf>,

    #[arg(long)]
    pub adapter_weight: Option<f64>,

    #[arg(long)]
    pub text_encoder_weight: Option<f64>,

    #[arg(long)]
    pub transformer_weight: Option<f64>,

    #[arg(long)]
    pub adapter_start: Option<f64>,

    #[arg(long)]
    pub adapter_end: Option<f64>,
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct ImageRefinementArgs {
    #[arg(
        long = "high-resolution-fix",
        action = ArgAction::SetTrue,
        conflicts_with = "no_high_resolution_fix"
    )]
    pub high_resolution_fix: bool,

    #[arg(
        long = "no-high-resolution-fix",
        action = ArgAction::SetTrue,
        conflicts_with = "high_resolution_fix"
    )]
    pub no_high_resolution_fix: bool,

    #[arg(long)]
    pub upscale_scale: Option<f64>,

    #[arg(long)]
    pub target_width: Option<u32>,

    #[arg(long)]
    pub target_height: Option<u32>,

    #[arg(long)]
    pub second_sampler: Option<String>,

    #[arg(long)]
    pub second_scheduler: Option<String>,

    #[arg(long)]
    pub second_denoise: Option<f64>,

    #[arg(long)]
    pub refiner: Option<String>,

    #[arg(long)]
    pub refiner_switch_point: Option<f64>,
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct ImagePostProcessingArgs {
    #[arg(long, value_name = "MODEL_OR_PATH")]
    pub vae: Option<String>,

    #[arg(long)]
    pub vae_precision: Option<String>,

    #[arg(
        long = "image-vae-tiling",
        action = ArgAction::SetTrue,
        conflicts_with = "no_image_vae_tiling"
    )]
    pub image_vae_tiling: bool,

    #[arg(
        long = "no-image-vae-tiling",
        action = ArgAction::SetTrue,
        conflicts_with = "image_vae_tiling"
    )]
    pub no_image_vae_tiling: bool,

    #[arg(
        long = "image-vae-slicing",
        action = ArgAction::SetTrue,
        conflicts_with = "no_image_vae_slicing"
    )]
    pub image_vae_slicing: bool,

    #[arg(
        long = "no-image-vae-slicing",
        action = ArgAction::SetTrue,
        conflicts_with = "image_vae_slicing"
    )]
    pub no_image_vae_slicing: bool,

    #[arg(
        long = "tiled-encode",
        action = ArgAction::SetTrue,
        conflicts_with = "no_tiled_encode"
    )]
    pub tiled_encode: bool,

    #[arg(
        long = "no-tiled-encode",
        action = ArgAction::SetTrue,
        conflicts_with = "tiled_encode"
    )]
    pub no_tiled_encode: bool,

    #[arg(
        long = "tiled-decode",
        action = ArgAction::SetTrue,
        conflicts_with = "no_tiled_decode"
    )]
    pub tiled_decode: bool,

    #[arg(
        long = "no-tiled-decode",
        action = ArgAction::SetTrue,
        conflicts_with = "tiled_decode"
    )]
    pub no_tiled_decode: bool,

    #[arg(long)]
    pub safety_checker_policy: Option<String>,

    #[arg(long)]
    pub watermark_policy: Option<String>,

    #[arg(
        long = "face-restoration",
        action = ArgAction::SetTrue,
        conflicts_with = "no_face_restoration"
    )]
    pub face_restoration: bool,

    #[arg(
        long = "no-face-restoration",
        action = ArgAction::SetTrue,
        conflicts_with = "face_restoration"
    )]
    pub no_face_restoration: bool,

    #[arg(
        long = "color-correction",
        action = ArgAction::SetTrue,
        conflicts_with = "no_color_correction"
    )]
    pub color_correction: bool,

    #[arg(
        long = "no-color-correction",
        action = ArgAction::SetTrue,
        conflicts_with = "color_correction"
    )]
    pub no_color_correction: bool,

    #[arg(long)]
    pub post_upscale: Option<String>,
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct ImageGenerateArgs {
    #[arg(value_name = "MODEL")]
    pub model: String,

    #[command(flatten)]
    #[serde(flatten)]
    pub prompt: PromptArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub routing: RoutingArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub dimensions: ImageDimensionsArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub sampling: ImageSamplingArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub conditioning: ImageConditioningArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub refinement: ImageRefinementArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub post_processing: ImagePostProcessingArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub output: OutputArgs,
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct ImageEditArgs {
    #[arg(value_name = "MODEL")]
    pub model: String,

    #[arg(long = "image", value_name = "PATH", required = true)]
    pub image: PathBuf,

    #[command(flatten)]
    #[serde(flatten)]
    pub prompt: PromptArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub routing: RoutingArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub dimensions: ImageDimensionsArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub sampling: ImageSamplingArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub conditioning: ImageConditioningArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub refinement: ImageRefinementArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub post_processing: ImagePostProcessingArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub output: OutputArgs,
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct ImageUpscaleArgs {
    #[arg(value_name = "MODEL")]
    pub model: String,

    #[arg(long = "image", value_name = "PATH", required = true)]
    pub image: PathBuf,

    #[command(flatten)]
    #[serde(flatten)]
    pub prompt: PromptArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub routing: RoutingArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub dimensions: ImageDimensionsArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub refinement: ImageRefinementArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub post_processing: ImagePostProcessingArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub output: OutputArgs,
}

#[derive(Debug, Clone, Subcommand, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageCommands {
    #[command(about = "Generate images with an installed model")]
    Generate(ImageGenerateArgs),

    #[command(about = "Edit or inpaint an image with an installed model")]
    Edit(ImageEditArgs),

    #[command(about = "Upscale an image with an installed model")]
    Upscale(ImageUpscaleArgs),
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct VideoCoreArgs {
    #[arg(long)]
    pub width: Option<u32>,

    #[arg(long)]
    pub height: Option<u32>,

    #[arg(long)]
    pub aspect_ratio: Option<String>,

    #[arg(long)]
    pub frames: Option<u32>,

    #[arg(long, value_name = "SECONDS")]
    pub duration: Option<f64>,

    #[arg(long)]
    pub fps: Option<f64>,

    #[arg(long)]
    pub batch_size: Option<u32>,

    #[arg(long)]
    pub num_videos: Option<u32>,

    #[arg(long)]
    pub seed: Option<u64>,

    #[arg(long)]
    pub steps: Option<u32>,

    #[arg(long, alias = "cfg")]
    pub guidance: Option<f64>,

    #[arg(long)]
    pub sampler: Option<String>,

    #[arg(long)]
    pub scheduler: Option<String>,

    #[arg(long)]
    pub eta: Option<f64>,

    #[arg(long)]
    pub denoise_strength: Option<f64>,

    #[arg(long)]
    pub noise_augmentation: Option<f64>,

    #[arg(long)]
    pub sigma_min: Option<f64>,

    #[arg(long)]
    pub sigma_max: Option<f64>,

    #[arg(long)]
    pub rho: Option<f64>,

    #[arg(long = "sigma", action = ArgAction::Append)]
    pub sigmas: Vec<f64>,

    #[arg(long)]
    pub shift: Option<f64>,

    #[arg(long)]
    pub motion_strength: Option<f64>,

    #[arg(long)]
    pub motion_bucket: Option<u32>,

    #[arg(long)]
    pub temporal_guidance: Option<f64>,

    #[arg(long)]
    pub temporal_consistency: Option<f64>,

    #[arg(long)]
    pub flow_shift: Option<f64>,
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct VideoReferenceArgs {
    #[arg(long, value_name = "PATH")]
    pub final_image: Option<PathBuf>,

    #[arg(long, value_name = "PATH")]
    pub mask_video: Option<PathBuf>,

    #[arg(long, value_name = "PATH")]
    pub pose_video: Option<PathBuf>,

    #[arg(long, value_name = "PATH")]
    pub depth_video: Option<PathBuf>,

    #[arg(long, value_name = "PATH")]
    pub control_video: Option<PathBuf>,

    #[arg(long = "video-control", value_name = "SPEC", action = ArgAction::Append)]
    pub controls: Vec<String>,

    #[arg(long = "video-control-json", value_name = "JSON")]
    pub controls_json: Option<String>,

    #[arg(long = "video-control-file", value_name = "PATH")]
    pub controls_file: Option<PathBuf>,

    #[arg(long = "reference-image", value_name = "PATH_OR_URL", action = ArgAction::Append)]
    pub reference_images: Vec<String>,

    #[arg(long = "reference-image-json", value_name = "JSON")]
    pub reference_images_json: Option<String>,

    #[arg(long = "reference-image-file", value_name = "PATH")]
    pub reference_images_file: Option<PathBuf>,

    #[arg(long, value_name = "PATH_OR_URL")]
    pub reference_audio: Option<String>,

    #[arg(long)]
    pub image_strength: Option<f64>,

    #[arg(long)]
    pub video_strength: Option<f64>,

    #[arg(long)]
    pub first_frame_strength: Option<f64>,

    #[arg(long)]
    pub last_frame_strength: Option<f64>,
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct VideoScheduleArgs {
    #[arg(long)]
    pub camera_motion: Option<String>,

    #[arg(long)]
    pub camera_strength: Option<f64>,

    #[arg(
        long = "video-camera-keyframe",
        value_name = "SPEC",
        action = ArgAction::Append
    )]
    pub camera_keyframes: Vec<String>,

    #[arg(long = "video-camera-keyframe-json", value_name = "JSON")]
    pub camera_keyframes_json: Option<String>,

    #[arg(long = "video-camera-keyframe-file", value_name = "PATH")]
    pub camera_keyframes_file: Option<PathBuf>,

    #[arg(
        long = "video-prompt-keyframe",
        value_name = "SPEC",
        action = ArgAction::Append
    )]
    pub prompt_keyframes: Vec<String>,

    #[arg(long = "video-prompt-keyframe-json", value_name = "JSON")]
    pub prompt_keyframes_json: Option<String>,

    #[arg(long = "video-prompt-keyframe-file", value_name = "PATH")]
    pub prompt_keyframes_file: Option<PathBuf>,

    #[arg(
        long = "video-guidance-schedule",
        value_name = "SPEC",
        action = ArgAction::Append
    )]
    pub guidance_schedule: Vec<String>,

    #[arg(long = "video-guidance-schedule-json", value_name = "JSON")]
    pub guidance_schedule_json: Option<String>,

    #[arg(long = "video-guidance-schedule-file", value_name = "PATH")]
    pub guidance_schedule_file: Option<PathBuf>,

    #[arg(
        long = "video-denoise-schedule",
        value_name = "SPEC",
        action = ArgAction::Append
    )]
    pub denoise_schedule: Vec<String>,

    #[arg(long = "video-denoise-schedule-json", value_name = "JSON")]
    pub denoise_schedule_json: Option<String>,

    #[arg(long = "video-denoise-schedule-file", value_name = "PATH")]
    pub denoise_schedule_file: Option<PathBuf>,

    #[arg(
        long = "video-adapter",
        value_name = "SPEC",
        action = ArgAction::Append
    )]
    pub adapters: Vec<String>,

    #[arg(long = "video-adapter-json", value_name = "JSON")]
    pub adapters_json: Option<String>,

    #[arg(long = "video-adapter-file", value_name = "PATH")]
    pub adapters_file: Option<PathBuf>,

    #[arg(
        long = "video-adapter-schedule",
        value_name = "SPEC",
        action = ArgAction::Append
    )]
    pub adapter_schedule: Vec<String>,

    #[arg(long = "video-adapter-schedule-json", value_name = "JSON")]
    pub adapter_schedule_json: Option<String>,

    #[arg(long = "video-adapter-schedule-file", value_name = "PATH")]
    pub adapter_schedule_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct VideoProcessingArgs {
    #[arg(long)]
    pub context_frames: Option<u32>,

    #[arg(long)]
    pub context_overlap: Option<u32>,

    #[arg(long)]
    pub context_stride: Option<u32>,

    #[arg(long)]
    pub window_size: Option<u32>,

    #[arg(long)]
    pub window_overlap: Option<u32>,

    #[arg(
        long = "loop",
        action = ArgAction::SetTrue,
        conflicts_with = "no_loop"
    )]
    pub looping: bool,

    #[arg(
        long = "no-loop",
        action = ArgAction::SetTrue,
        conflicts_with = "looping"
    )]
    pub no_loop: bool,

    #[arg(long)]
    pub loop_blend_frames: Option<u32>,

    #[arg(long)]
    pub extension_frames: Option<u32>,

    #[arg(long)]
    pub extension_direction: Option<String>,

    #[arg(
        long = "temporal-vae-tiling",
        action = ArgAction::SetTrue,
        conflicts_with = "no_temporal_vae_tiling"
    )]
    pub temporal_vae_tiling: bool,

    #[arg(
        long = "no-temporal-vae-tiling",
        action = ArgAction::SetTrue,
        conflicts_with = "temporal_vae_tiling"
    )]
    pub no_temporal_vae_tiling: bool,

    #[arg(long)]
    pub tile_width: Option<u32>,

    #[arg(long)]
    pub tile_height: Option<u32>,

    #[arg(long)]
    pub tile_frames: Option<u32>,

    #[arg(long)]
    pub tile_overlap: Option<u32>,

    #[arg(long)]
    pub decode_chunk_size: Option<u32>,

    #[arg(
        long = "frame-interpolation",
        action = ArgAction::SetTrue,
        conflicts_with = "no_frame_interpolation"
    )]
    pub frame_interpolation: bool,

    #[arg(
        long = "no-frame-interpolation",
        action = ArgAction::SetTrue,
        conflicts_with = "frame_interpolation"
    )]
    pub no_frame_interpolation: bool,

    #[arg(long)]
    pub interpolation_factor: Option<f64>,

    #[arg(
        long = "upscaling",
        action = ArgAction::SetTrue,
        conflicts_with = "no_upscaling"
    )]
    pub upscaling: bool,

    #[arg(
        long = "no-upscaling",
        action = ArgAction::SetTrue,
        conflicts_with = "upscaling"
    )]
    pub no_upscaling: bool,

    #[arg(long)]
    pub upscale_scale: Option<f64>,

    #[arg(
        long = "stabilization",
        action = ArgAction::SetTrue,
        conflicts_with = "no_stabilization"
    )]
    pub stabilization: bool,

    #[arg(
        long = "no-stabilization",
        action = ArgAction::SetTrue,
        conflicts_with = "stabilization"
    )]
    pub no_stabilization: bool,
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct VideoEncodingArgs {
    #[arg(long)]
    pub codec: Option<String>,

    #[arg(long)]
    pub pixel_format: Option<String>,

    #[arg(long)]
    pub bitrate: Option<String>,

    #[arg(long)]
    pub crf: Option<u8>,

    #[arg(long)]
    pub encoding_preset: Option<String>,

    #[arg(
        long = "include-audio",
        action = ArgAction::SetTrue,
        conflicts_with = "exclude_audio"
    )]
    pub include_audio: bool,

    #[arg(
        long = "exclude-audio",
        action = ArgAction::SetTrue,
        conflicts_with = "include_audio"
    )]
    pub exclude_audio: bool,

    #[arg(long = "output-format")]
    pub output_format: Option<String>,

    #[arg(long = "output", alias = "output-path", value_name = "PATH")]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct VideoGenerateArgs {
    #[arg(value_name = "MODEL")]
    pub model: String,

    #[arg(long, value_name = "PATH")]
    pub initial_image: Option<PathBuf>,

    #[command(flatten)]
    #[serde(flatten)]
    pub prompt: PromptArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub routing: RoutingArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub core: VideoCoreArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub references: VideoReferenceArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub schedules: VideoScheduleArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub processing: VideoProcessingArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub encoding: VideoEncodingArgs,
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct VideoAnimateArgs {
    #[arg(value_name = "MODEL")]
    pub model: String,

    #[arg(long = "image", value_name = "PATH", required = true)]
    pub image: PathBuf,

    #[command(flatten)]
    #[serde(flatten)]
    pub prompt: PromptArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub routing: RoutingArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub core: VideoCoreArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub references: VideoReferenceArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub schedules: VideoScheduleArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub processing: VideoProcessingArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub encoding: VideoEncodingArgs,
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct VideoTransformArgs {
    #[arg(value_name = "MODEL")]
    pub model: String,

    #[arg(long = "video", value_name = "PATH", required = true)]
    pub video: PathBuf,

    #[command(flatten)]
    #[serde(flatten)]
    pub prompt: PromptArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub routing: RoutingArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub core: VideoCoreArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub references: VideoReferenceArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub schedules: VideoScheduleArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub processing: VideoProcessingArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub encoding: VideoEncodingArgs,
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct VideoUpscaleArgs {
    #[arg(value_name = "MODEL")]
    pub model: String,

    #[arg(long = "video", value_name = "PATH", required = true)]
    pub video: PathBuf,

    #[command(flatten)]
    #[serde(flatten)]
    pub routing: RoutingArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub core: VideoCoreArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub processing: VideoProcessingArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub encoding: VideoEncodingArgs,
}

#[derive(Debug, Clone, Subcommand, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoCommands {
    #[command(about = "Generate a video with an installed model")]
    Generate(VideoGenerateArgs),

    #[command(about = "Animate an image with an installed model")]
    Animate(VideoAnimateArgs),

    #[command(about = "Transform an existing video with an installed model")]
    Transform(VideoTransformArgs),

    #[command(about = "Upscale an existing video with an installed model")]
    Upscale(VideoUpscaleArgs),
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct MusicCompositionArgs {
    #[arg(long)]
    pub title: Option<String>,

    #[arg(
        long,
        action = ArgAction::SetTrue,
        conflicts_with = "no_instrumental"
    )]
    pub instrumental: bool,

    #[arg(
        long = "no-instrumental",
        action = ArgAction::SetTrue,
        conflicts_with = "instrumental"
    )]
    pub no_instrumental: bool,

    #[arg(
        long = "generate-lyrics",
        action = ArgAction::SetTrue,
        conflicts_with = "no_generate_lyrics"
    )]
    pub generate_lyrics: bool,

    #[arg(
        long = "no-generate-lyrics",
        action = ArgAction::SetTrue,
        conflicts_with = "generate_lyrics"
    )]
    pub no_generate_lyrics: bool,

    #[arg(long)]
    pub lyrics_language: Option<String>,

    #[arg(long, value_name = "SECONDS")]
    pub duration: Option<f64>,

    #[arg(long)]
    pub num_variations: Option<u32>,

    #[arg(long)]
    pub seed: Option<u64>,

    #[arg(long = "genre", action = ArgAction::Append)]
    pub genres: Vec<String>,

    #[arg(long = "subgenre", action = ArgAction::Append)]
    pub subgenres: Vec<String>,

    #[arg(long = "style", action = ArgAction::Append)]
    pub styles: Vec<String>,

    #[arg(long = "era", action = ArgAction::Append)]
    pub eras: Vec<String>,

    #[arg(long = "influence", action = ArgAction::Append)]
    pub influences: Vec<String>,

    #[arg(long = "mood", action = ArgAction::Append)]
    pub moods: Vec<String>,

    #[arg(long = "theme", action = ArgAction::Append)]
    pub themes: Vec<String>,

    #[arg(long = "descriptor", action = ArgAction::Append)]
    pub descriptors: Vec<String>,

    #[arg(long, alias = "tempo")]
    pub bpm: Option<f64>,

    #[arg(long)]
    pub bpm_min: Option<f64>,

    #[arg(long)]
    pub bpm_max: Option<f64>,

    #[arg(long)]
    pub tempo_mode: Option<String>,

    #[arg(long)]
    pub time_signature: Option<String>,

    #[arg(long)]
    pub key: Option<String>,

    #[arg(long)]
    pub scale: Option<String>,

    #[arg(long)]
    pub tuning: Option<String>,

    #[arg(long)]
    pub chord_progression: Option<String>,

    #[arg(long)]
    pub chord_complexity: Option<f64>,

    #[arg(long)]
    pub harmonic_tension: Option<f64>,

    #[arg(long)]
    pub song_structure: Option<String>,

    #[arg(long)]
    pub arrangement_prompt: Option<String>,

    #[arg(
        long = "music-instrument",
        value_name = "SPEC",
        action = ArgAction::Append
    )]
    pub instruments: Vec<String>,

    #[arg(long = "music-instrument-json", value_name = "JSON")]
    pub instruments_json: Option<String>,

    #[arg(long = "music-instrument-file", value_name = "PATH")]
    pub instruments_file: Option<PathBuf>,

    #[arg(long = "excluded-instrument", action = ArgAction::Append)]
    pub excluded_instruments: Vec<String>,

    #[arg(long)]
    pub lead_instrument: Option<String>,

    #[arg(long = "rhythm-instrument", action = ArgAction::Append)]
    pub rhythm_instruments: Vec<String>,

    #[arg(long)]
    pub bass_instrument: Option<String>,

    #[arg(long)]
    pub acoustic_electronic_balance: Option<f64>,

    #[arg(long)]
    pub instrument_density: Option<f64>,
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct MusicVocalArgs {
    #[arg(
        long = "vocals",
        action = ArgAction::SetTrue,
        conflicts_with = "no_vocals"
    )]
    pub vocals: bool,

    #[arg(
        long = "no-vocals",
        action = ArgAction::SetTrue,
        conflicts_with = "vocals"
    )]
    pub no_vocals: bool,

    #[arg(long, value_name = "PATH_OR_URL")]
    pub voice_reference: Option<String>,

    #[arg(long)]
    pub speaker_id: Option<String>,

    #[arg(long)]
    pub vocal_presentation: Option<String>,

    #[arg(long)]
    pub vocal_register: Option<String>,

    #[arg(long)]
    pub vocal_range: Option<String>,

    #[arg(long)]
    pub vocal_language: Option<String>,

    #[arg(long)]
    pub vocal_accent: Option<String>,

    #[arg(long)]
    pub vocal_style: Option<String>,

    #[arg(long)]
    pub vocal_delivery: Option<String>,

    #[arg(long)]
    pub vocal_emotion: Option<String>,

    #[arg(long)]
    pub breathiness: Option<f64>,

    #[arg(long)]
    pub rasp: Option<f64>,

    #[arg(long)]
    pub vibrato: Option<f64>,

    #[arg(long)]
    pub vocal_power: Option<f64>,

    #[arg(long)]
    pub intimacy: Option<f64>,

    #[arg(long)]
    pub articulation: Option<f64>,

    #[arg(long)]
    pub pronunciation_strength: Option<f64>,

    #[arg(long)]
    pub vocal_presence: Option<f64>,

    #[arg(long)]
    pub harmony_amount: Option<f64>,

    #[arg(long)]
    pub choir_amount: Option<f64>,

    #[arg(
        long,
        action = ArgAction::SetTrue,
        conflicts_with = "no_duet"
    )]
    pub duet: bool,

    #[arg(
        long = "no-duet",
        action = ArgAction::SetTrue,
        conflicts_with = "duet"
    )]
    pub no_duet: bool,

    #[arg(
        long = "backing-vocals",
        action = ArgAction::SetTrue,
        conflicts_with = "no_backing_vocals"
    )]
    pub backing_vocals: bool,

    #[arg(
        long = "no-backing-vocals",
        action = ArgAction::SetTrue,
        conflicts_with = "backing_vocals"
    )]
    pub no_backing_vocals: bool,
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct MusicConditioningArgs {
    #[arg(long, value_name = "PATH")]
    pub source_audio: Option<PathBuf>,

    #[arg(long, value_name = "PATH")]
    pub reference_audio: Option<PathBuf>,

    #[arg(long, value_name = "PATH")]
    pub instrumental_audio: Option<PathBuf>,

    #[arg(long, value_name = "PATH")]
    pub vocal_audio: Option<PathBuf>,

    #[arg(long, value_name = "PATH")]
    pub melody_audio: Option<PathBuf>,

    #[arg(long, value_name = "PATH")]
    pub rhythm_audio: Option<PathBuf>,

    #[arg(long, value_name = "PATH")]
    pub chord_audio: Option<PathBuf>,

    #[arg(long)]
    pub audio_strength: Option<f64>,

    #[arg(long)]
    pub melody_adherence: Option<f64>,

    #[arg(long)]
    pub rhythm_adherence: Option<f64>,

    #[arg(long)]
    pub harmony_adherence: Option<f64>,

    #[arg(long, value_name = "SECONDS")]
    pub continuation_start: Option<f64>,

    #[arg(long, value_name = "SECONDS")]
    pub continuation_duration: Option<f64>,
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct MusicSamplingArgs {
    #[arg(long)]
    pub prompt_adherence: Option<f64>,

    #[arg(long)]
    pub lyrics_adherence: Option<f64>,

    #[arg(long)]
    pub style_adherence: Option<f64>,

    #[arg(long)]
    pub creativity: Option<f64>,

    #[arg(long)]
    pub variation_strength: Option<f64>,

    #[arg(long)]
    pub originality: Option<f64>,

    #[arg(long)]
    pub guidance: Option<f64>,

    #[arg(long)]
    pub steps: Option<u32>,

    #[arg(long)]
    pub temperature: Option<f64>,

    #[arg(long)]
    pub top_k: Option<u32>,

    #[arg(long)]
    pub top_p: Option<f64>,

    #[arg(
        long = "music-mix-control",
        value_name = "SPEC",
        action = ArgAction::Append
    )]
    pub mix_controls: Vec<String>,

    #[arg(long = "music-mix-control-json", value_name = "JSON")]
    pub mix_controls_json: Option<String>,

    #[arg(long = "music-mix-control-file", value_name = "PATH")]
    pub mix_controls_file: Option<PathBuf>,

    #[arg(
        long = "music-mastering-control",
        value_name = "SPEC",
        action = ArgAction::Append
    )]
    pub mastering_controls: Vec<String>,

    #[arg(long = "music-mastering-control-json", value_name = "JSON")]
    pub mastering_controls_json: Option<String>,

    #[arg(long = "music-mastering-control-file", value_name = "PATH")]
    pub mastering_controls_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct AudioExportArgs {
    #[arg(long)]
    pub sample_rate: Option<u32>,

    #[arg(long)]
    pub bit_depth: Option<u16>,

    #[arg(long)]
    pub channels: Option<u16>,

    #[arg(long)]
    pub format: Option<String>,

    #[arg(long)]
    pub codec: Option<String>,

    #[arg(long)]
    pub bitrate: Option<String>,

    #[arg(
        long = "normalization",
        action = ArgAction::SetTrue,
        conflicts_with = "no_normalization"
    )]
    pub normalization: bool,

    #[arg(
        long = "no-normalization",
        action = ArgAction::SetTrue,
        conflicts_with = "normalization"
    )]
    pub no_normalization: bool,

    #[arg(long, allow_hyphen_values = true)]
    pub loudness_lufs: Option<f64>,

    #[arg(
        long = "export-stems",
        action = ArgAction::SetTrue,
        conflicts_with = "no_export_stems"
    )]
    pub export_stems: bool,

    #[arg(
        long = "no-export-stems",
        action = ArgAction::SetTrue,
        conflicts_with = "export_stems"
    )]
    pub no_export_stems: bool,

    #[arg(long = "stem", action = ArgAction::Append)]
    pub stems: Vec<String>,

    #[arg(long = "stems-json", value_name = "JSON")]
    pub stems_json: Option<String>,

    #[arg(long = "stems-file", value_name = "PATH")]
    pub stems_file: Option<PathBuf>,

    #[arg(long = "output", alias = "output-path", value_name = "PATH")]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct AudioGenerateArgs {
    #[arg(value_name = "MODEL")]
    pub model: String,

    #[command(flatten)]
    #[serde(flatten)]
    pub prompt: PromptArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub lyrics: LyricsArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub routing: RoutingArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub composition: MusicCompositionArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub vocals: MusicVocalArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub conditioning: MusicConditioningArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub sampling: MusicSamplingArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub export: AudioExportArgs,
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct TextToSpeechArgs {
    #[arg(long)]
    pub voice: Option<String>,

    #[arg(long, value_name = "PATH_OR_URL")]
    pub voice_reference: Option<String>,

    #[arg(long)]
    pub language: Option<String>,

    #[arg(long)]
    pub accent: Option<String>,

    #[arg(long)]
    pub dialect: Option<String>,

    #[arg(long)]
    pub speed: Option<f64>,

    #[arg(long, allow_hyphen_values = true)]
    pub pitch: Option<f64>,

    #[arg(long, allow_hyphen_values = true)]
    pub volume: Option<f64>,

    #[arg(long)]
    pub emotion: Option<String>,

    #[arg(long)]
    pub emotion_strength: Option<f64>,

    #[arg(long)]
    pub speaking_style: Option<String>,

    #[arg(long)]
    pub stability: Option<f64>,

    #[arg(long)]
    pub similarity: Option<f64>,

    #[arg(long)]
    pub expressiveness: Option<f64>,

    #[arg(
        long = "pronunciation-dictionary",
        value_name = "PATH",
        action = ArgAction::Append
    )]
    pub pronunciation_dictionaries: Vec<PathBuf>,

    #[arg(
        long = "phoneme-input",
        action = ArgAction::SetTrue,
        conflicts_with = "no_phoneme_input"
    )]
    pub phoneme_input: bool,

    #[arg(
        long = "no-phoneme-input",
        action = ArgAction::SetTrue,
        conflicts_with = "phoneme_input"
    )]
    pub no_phoneme_input: bool,

    #[arg(long)]
    pub pause_scale: Option<f64>,

    #[arg(long)]
    pub sentence_silence: Option<f64>,

    #[arg(long)]
    pub seed: Option<u64>,

    #[arg(long)]
    pub sample_rate: Option<u32>,

    #[arg(long)]
    pub channels: Option<u16>,

    #[arg(long)]
    pub format: Option<String>,

    #[arg(long, allow_hyphen_values = true)]
    pub loudness_lufs: Option<f64>,

    #[arg(
        long = "streaming",
        action = ArgAction::SetTrue,
        conflicts_with = "no_streaming"
    )]
    pub streaming: bool,

    #[arg(
        long = "no-streaming",
        action = ArgAction::SetTrue,
        conflicts_with = "streaming"
    )]
    pub no_streaming: bool,

    #[arg(long)]
    pub chunk_size: Option<u32>,

    #[arg(long = "output", alias = "output-path", value_name = "PATH")]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct AudioSpeakArgs {
    #[arg(value_name = "MODEL")]
    pub model: String,

    #[command(flatten)]
    #[serde(flatten)]
    pub text: TextInputArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub routing: RoutingArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub speech: TextToSpeechArgs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechToTextTask {
    Transcribe,
    Translate,
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct SpeechToTextArgs {
    #[arg(long)]
    pub language: Option<String>,

    #[arg(long = "task", value_enum)]
    pub task: Option<SpeechToTextTask>,

    #[arg(
        long = "segment-timestamps",
        action = ArgAction::SetTrue,
        conflicts_with = "no_segment_timestamps"
    )]
    pub segment_timestamps: bool,

    #[arg(
        long = "no-segment-timestamps",
        action = ArgAction::SetTrue,
        conflicts_with = "segment_timestamps"
    )]
    pub no_segment_timestamps: bool,

    #[arg(
        long = "word-timestamps",
        action = ArgAction::SetTrue,
        conflicts_with = "no_word_timestamps"
    )]
    pub word_timestamps: bool,

    #[arg(
        long = "no-word-timestamps",
        action = ArgAction::SetTrue,
        conflicts_with = "word_timestamps"
    )]
    pub no_word_timestamps: bool,

    #[arg(
        long = "diarization",
        action = ArgAction::SetTrue,
        conflicts_with = "no_diarization"
    )]
    pub diarization: bool,

    #[arg(
        long = "no-diarization",
        action = ArgAction::SetTrue,
        conflicts_with = "diarization"
    )]
    pub no_diarization: bool,

    #[arg(long)]
    pub min_speakers: Option<u32>,

    #[arg(long)]
    pub max_speakers: Option<u32>,

    #[arg(long)]
    pub beam_size: Option<u32>,

    #[arg(long)]
    pub best_of: Option<u32>,

    #[arg(long)]
    pub temperature: Option<f64>,

    #[arg(long = "temperature-fallback", action = ArgAction::Append)]
    pub temperature_fallbacks: Vec<f64>,

    #[arg(long)]
    pub initial_prompt: Option<String>,

    #[arg(long = "hotword", action = ArgAction::Append)]
    pub hotwords: Vec<String>,

    #[arg(
        long = "vad",
        action = ArgAction::SetTrue,
        conflicts_with = "no_vad"
    )]
    pub vad: bool,

    #[arg(
        long = "no-vad",
        action = ArgAction::SetTrue,
        conflicts_with = "vad"
    )]
    pub no_vad: bool,

    #[arg(
        long = "suppress-token",
        action = ArgAction::Append,
        allow_hyphen_values = true
    )]
    pub suppress_tokens: Vec<String>,

    #[arg(
        long = "condition-on-previous-text",
        action = ArgAction::SetTrue,
        conflicts_with = "no_condition_on_previous_text"
    )]
    pub condition_on_previous_text: bool,

    #[arg(
        long = "no-condition-on-previous-text",
        action = ArgAction::SetTrue,
        conflicts_with = "condition_on_previous_text"
    )]
    pub no_condition_on_previous_text: bool,

    #[arg(long)]
    pub hallucination_silence_threshold: Option<f64>,

    #[arg(long = "output-format")]
    pub output_format: Option<String>,

    #[arg(long = "output", alias = "output-path", value_name = "PATH")]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct AudioTranscribeArgs {
    #[arg(value_name = "MODEL")]
    pub model: String,

    #[arg(long = "input", value_name = "AUDIO", required = true)]
    pub input_audio: PathBuf,

    #[command(flatten)]
    #[serde(flatten)]
    pub routing: RoutingArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub transcription: SpeechToTextArgs,
}

#[derive(Debug, Clone, Args, Serialize, Default)]
pub struct AudioSeparationArgs {
    #[arg(long = "stem", action = ArgAction::Append)]
    pub stems: Vec<String>,

    #[arg(long = "stems-json", value_name = "JSON")]
    pub stems_json: Option<String>,

    #[arg(long = "stems-file", value_name = "PATH")]
    pub stems_file: Option<PathBuf>,

    #[arg(long)]
    pub num_stems: Option<u32>,

    #[arg(long)]
    pub segment_duration: Option<f64>,

    #[arg(long)]
    pub segment_overlap: Option<f64>,

    #[arg(long)]
    pub sample_rate: Option<u32>,

    #[arg(long)]
    pub channels: Option<u16>,

    #[arg(long)]
    pub format: Option<String>,

    #[arg(
        long = "normalization",
        action = ArgAction::SetTrue,
        conflicts_with = "no_normalization"
    )]
    pub normalization: bool,

    #[arg(
        long = "no-normalization",
        action = ArgAction::SetTrue,
        conflicts_with = "normalization"
    )]
    pub no_normalization: bool,

    #[arg(long = "output", alias = "output-path", value_name = "PATH")]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct AudioSeparateArgs {
    #[arg(value_name = "MODEL")]
    pub model: String,

    #[arg(long = "input", value_name = "AUDIO", required = true)]
    pub input_audio: PathBuf,

    #[command(flatten)]
    #[serde(flatten)]
    pub routing: RoutingArgs,

    #[command(flatten)]
    #[serde(flatten)]
    pub separation: AudioSeparationArgs,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Subcommand, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioCommands {
    #[command(about = "Generate audio or music with an installed model")]
    Generate(AudioGenerateArgs),

    #[command(about = "Synthesize speech with an installed model")]
    Speak(AudioSpeakArgs),

    #[command(about = "Transcribe or translate speech with an installed model")]
    Transcribe(AudioTranscribeArgs),

    #[command(about = "Separate audio into stems with an installed model")]
    Separate(AudioSeparateArgs),
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Debug, Parser)]
    struct ImageCli {
        #[command(subcommand)]
        command: ImageCommands,
    }

    #[derive(Debug, Parser)]
    struct VideoCli {
        #[command(subcommand)]
        command: VideoCommands,
    }

    #[derive(Debug, Parser)]
    struct AudioCli {
        #[command(subcommand)]
        command: AudioCommands,
    }

    #[test]
    fn parses_image_generate_with_routing_and_structured_overrides() {
        let cli = ImageCli::try_parse_from([
            "werk-image",
            "generate",
            "flux-dev",
            "--prompt",
            "an abandoned orbital station",
            "--negative-prompt-file",
            "negative.txt",
            "--accelerator",
            "cuda",
            "--precision",
            "bf16",
            "--quantization",
            "int8",
            "--profile",
            "workstation",
            "--quality",
            "high",
            "--performance-preference",
            "quality",
            "--fallback-policy",
            "backend",
            "--parameter-policy",
            "strict",
            "--allow-cpu-offload",
            "--no-compile",
            "--attention-backend",
            "flash-attention-2",
            "--timeout",
            "600",
            "--width",
            "1024",
            "--height",
            "768",
            "--batch-size",
            "2",
            "--num-images",
            "4",
            "--seed",
            "42",
            "--subseed",
            "7",
            "--steps",
            "28",
            "--guidance",
            "3.5",
            "--sigma",
            "1.0",
            "--sigma",
            "0.5",
            "--dynamic-shift",
            "--image-control",
            "type=depth,image=depth.png,weight=0.8,start=0,end=0.7",
            "--image-control",
            "type=canny,image=edges.png,weight=0.4",
            "--image-control-json",
            "[{\"type\":\"pose\"}]",
            "--reference-image",
            "face.png",
            "--image-lora",
            "model=style.safetensors,weight=0.6,start=0.2,end=0.8",
            "--image-adapter",
            "model=adapter.safetensors,weight=0.5",
            "--high-resolution-fix",
            "--image-vae-tiling",
            "--no-image-vae-slicing",
            "--output-format",
            "png",
            "--output",
            "out/",
            "--set",
            "image.custom_strength=0.25",
            "--set",
            "runtime.experimental=true",
        ])
        .unwrap();

        let ImageCommands::Generate(args) = cli.command else {
            panic!("expected image generate");
        };
        assert_eq!(args.model, "flux-dev");
        assert_eq!(
            args.prompt.prompt.as_deref(),
            Some("an abandoned orbital station")
        );
        assert_eq!(
            args.prompt.negative_prompt_file.as_deref(),
            Some(PathBuf::from("negative.txt").as_path())
        );
        assert_eq!(args.routing.accelerator.as_deref(), Some("cuda"));
        assert_eq!(args.routing.cpu_offload(), BoolOverride::Enabled);
        assert_eq!(args.routing.compilation(), BoolOverride::Disabled);
        assert_eq!(args.dimensions.width, Some(1024));
        assert_eq!(args.sampling.sigmas, vec![1.0, 0.5]);
        assert_eq!(args.conditioning.controls.len(), 2);
        assert_eq!(args.conditioning.loras.len(), 1);
        assert!(args.refinement.high_resolution_fix);
        assert!(args.post_processing.image_vae_tiling);
        assert!(args.post_processing.no_image_vae_slicing);
        assert_eq!(args.routing.set.len(), 2);
    }

    #[test]
    fn parses_image_edit_and_upscale_inputs_and_negative_booleans() {
        let edit = ImageCli::try_parse_from([
            "werk-image",
            "edit",
            "sdxl-inpaint",
            "--image",
            "source.png",
            "--mask",
            "mask.png",
            "--prompt-file",
            "edit.txt",
            "--image-strength",
            "0.7",
            "--mask-blur",
            "12",
            "--no-mask-invert",
            "--no-preserve-unmasked",
            "--no-allow-disk-offload",
            "--output",
            "edited.webp",
        ])
        .unwrap();
        let ImageCommands::Edit(args) = edit.command else {
            panic!("expected image edit");
        };
        assert_eq!(args.model, "sdxl-inpaint");
        assert_eq!(args.image, PathBuf::from("source.png"));
        assert_eq!(
            args.conditioning.mask.as_deref(),
            Some(PathBuf::from("mask.png").as_path())
        );
        assert!(args.conditioning.no_mask_invert);
        assert!(args.conditioning.no_preserve_unmasked);
        assert_eq!(args.routing.disk_offload(), BoolOverride::Disabled);

        let upscale = ImageCli::try_parse_from([
            "werk-image",
            "upscale",
            "realesrgan",
            "--image",
            "small.png",
            "--upscale-scale",
            "4",
            "--target-width",
            "4096",
            "--target-height",
            "4096",
            "--tiled-decode",
            "--no-face-restoration",
            "--output",
            "large.png",
        ])
        .unwrap();
        let ImageCommands::Upscale(args) = upscale.command else {
            panic!("expected image upscale");
        };
        assert_eq!(args.image, PathBuf::from("small.png"));
        assert_eq!(args.refinement.upscale_scale, Some(4.0));
        assert_eq!(args.refinement.target_width, Some(4096));
        assert!(args.post_processing.tiled_decode);
        assert!(args.post_processing.no_face_restoration);
    }

    #[test]
    fn image_boolean_pairs_conflict() {
        assert!(
            ImageCli::try_parse_from([
                "werk-image",
                "generate",
                "flux",
                "--compile",
                "--no-compile",
            ])
            .is_err()
        );
        assert!(
            ImageCli::try_parse_from([
                "werk-image",
                "generate",
                "flux",
                "--image-vae-tiling",
                "--no-image-vae-tiling",
            ])
            .is_err()
        );
    }

    #[test]
    fn parses_all_video_subcommands() {
        let generate = VideoCli::try_parse_from([
            "werk-video",
            "generate",
            "wan-2.1",
            "--prompt",
            "a storm over a desert",
            "--negative-prompt",
            "text, watermark",
            "--initial-image",
            "start.png",
            "--width",
            "832",
            "--height",
            "480",
            "--frames",
            "81",
            "--duration",
            "5",
            "--fps",
            "16",
            "--steps",
            "30",
            "--motion-strength",
            "0.8",
            "--temporal-guidance",
            "1.2",
            "--reference-image",
            "style.png",
            "--reference-audio",
            "beat.wav",
            "--video-control",
            "type=depth,video=depth.mp4,weight=0.7",
            "--video-prompt-keyframe",
            "frame=48,prompt=violent storm",
            "--video-camera-keyframe",
            "frame=0,yaw=0",
            "--video-guidance-schedule",
            "start=0,value=4.0",
            "--video-adapter",
            "model=motion.safetensors,weight=0.6",
            "--context-frames",
            "24",
            "--context-overlap",
            "8",
            "--loop",
            "--temporal-vae-tiling",
            "--frame-interpolation",
            "--interpolation-factor",
            "2",
            "--upscaling",
            "--stabilization",
            "--codec",
            "h264",
            "--pixel-format",
            "yuv420p",
            "--crf",
            "18",
            "--include-audio",
            "--output-format",
            "mp4",
            "--output",
            "storm.mp4",
        ])
        .unwrap();
        let VideoCommands::Generate(args) = generate.command else {
            panic!("expected video generate");
        };
        assert_eq!(args.model, "wan-2.1");
        assert_eq!(
            args.initial_image.as_deref(),
            Some(PathBuf::from("start.png").as_path())
        );
        assert_eq!(args.core.frames, Some(81));
        assert_eq!(args.schedules.prompt_keyframes.len(), 1);
        assert!(args.processing.looping);
        assert!(args.processing.temporal_vae_tiling);
        assert!(args.encoding.include_audio);

        let animate = VideoCli::try_parse_from([
            "werk-video",
            "animate",
            "stable-video",
            "--image",
            "portrait.png",
            "--prompt-file",
            "motion.txt",
            "--first-frame-strength",
            "0.9",
            "--last-frame-strength",
            "0.5",
            "--no-loop",
            "--exclude-audio",
            "--output",
            "portrait.mp4",
        ])
        .unwrap();
        let VideoCommands::Animate(args) = animate.command else {
            panic!("expected video animate");
        };
        assert_eq!(args.image, PathBuf::from("portrait.png"));
        assert!(args.processing.no_loop);
        assert!(args.encoding.exclude_audio);

        let transform = VideoCli::try_parse_from([
            "werk-video",
            "transform",
            "video-control-model",
            "--video",
            "source.mp4",
            "--prompt",
            "turn it into watercolor",
            "--mask-video",
            "mask.mp4",
            "--video-strength",
            "0.65",
            "--video-denoise-schedule",
            "frame=0,value=0.7",
            "--no-frame-interpolation",
        ])
        .unwrap();
        let VideoCommands::Transform(args) = transform.command else {
            panic!("expected video transform");
        };
        assert_eq!(args.video, PathBuf::from("source.mp4"));
        assert_eq!(args.references.video_strength, Some(0.65));
        assert!(args.processing.no_frame_interpolation);

        let upscale = VideoCli::try_parse_from([
            "werk-video",
            "upscale",
            "video-upscaler",
            "--video",
            "source.mp4",
            "--upscale-scale",
            "2",
            "--tile-width",
            "512",
            "--tile-height",
            "512",
            "--no-stabilization",
            "--codec",
            "hevc",
            "--output",
            "upscaled.mkv",
        ])
        .unwrap();
        let VideoCommands::Upscale(args) = upscale.command else {
            panic!("expected video upscale");
        };
        assert_eq!(args.video, PathBuf::from("source.mp4"));
        assert_eq!(args.processing.upscale_scale, Some(2.0));
        assert!(args.processing.no_stabilization);
    }

    #[test]
    fn video_commands_require_their_source_media() {
        assert!(VideoCli::try_parse_from(["werk-video", "animate", "model"]).is_err());
        assert!(VideoCli::try_parse_from(["werk-video", "transform", "model"]).is_err());
        assert!(VideoCli::try_parse_from(["werk-video", "upscale", "model"]).is_err());
    }

    #[test]
    fn parses_audio_generate_music_surface() {
        let cli = AudioCli::try_parse_from([
            "werk-audio",
            "generate",
            "musicgen-large",
            "--prompt",
            "cinematic progressive metal",
            "--lyrics-file",
            "lyrics.txt",
            "--title",
            "Orbit",
            "--no-instrumental",
            "--generate-lyrics",
            "--lyrics-language",
            "en",
            "--duration",
            "180",
            "--num-variations",
            "3",
            "--seed",
            "99",
            "--genre",
            "metal",
            "--genre",
            "soundtrack",
            "--subgenre",
            "progressive-metal",
            "--style",
            "cinematic",
            "--era",
            "modern",
            "--influence",
            "post-rock",
            "--mood",
            "triumphant",
            "--theme",
            "space",
            "--descriptor",
            "wide",
            "--bpm",
            "132",
            "--time-signature",
            "4/4",
            "--key",
            "D",
            "--scale",
            "minor",
            "--chord-progression",
            "Dm-Bb-F-C",
            "--music-instrument",
            "name=distorted-guitar,role=rhythm,presence=0.8",
            "--music-instrument-json",
            "[{\"name\":\"drums\"}]",
            "--excluded-instrument",
            "banjo",
            "--lead-instrument",
            "electric-guitar",
            "--vocals",
            "--voice-reference",
            "voice.wav",
            "--vocal-presentation",
            "androgynous",
            "--vocal-emotion",
            "intense",
            "--breathiness",
            "0.2",
            "--vibrato",
            "0.4",
            "--backing-vocals",
            "--reference-audio",
            "reference.wav",
            "--melody-audio",
            "melody.wav",
            "--audio-strength",
            "0.5",
            "--continuation-start",
            "30",
            "--continuation-duration",
            "60",
            "--prompt-adherence",
            "0.8",
            "--lyrics-adherence",
            "0.9",
            "--creativity",
            "0.6",
            "--guidance",
            "4",
            "--steps",
            "50",
            "--temperature",
            "0.9",
            "--top-k",
            "250",
            "--top-p",
            "0.95",
            "--music-mix-control",
            "track=vocals,gain=-2",
            "--music-mastering-control-file",
            "mastering.json",
            "--sample-rate",
            "48000",
            "--bit-depth",
            "24",
            "--channels",
            "2",
            "--format",
            "wav",
            "--normalization",
            "--loudness-lufs",
            "-14",
            "--export-stems",
            "--stem",
            "vocals",
            "--stem",
            "drums",
            "--output",
            "orbit.wav",
        ])
        .unwrap();

        let AudioCommands::Generate(args) = cli.command else {
            panic!("expected audio generate");
        };
        assert_eq!(args.model, "musicgen-large");
        assert_eq!(
            args.lyrics.lyrics_file.as_deref(),
            Some(PathBuf::from("lyrics.txt").as_path())
        );
        assert_eq!(args.composition.genres, vec!["metal", "soundtrack"]);
        assert_eq!(args.composition.instruments.len(), 1);
        assert!(args.vocals.vocals);
        assert!(args.vocals.backing_vocals);
        assert_eq!(
            args.conditioning.reference_audio.as_deref(),
            Some(PathBuf::from("reference.wav").as_path())
        );
        assert_eq!(args.sampling.top_k, Some(250));
        assert!(args.export.normalization);
        assert!(args.export.export_stems);
        assert_eq!(args.export.stems, vec!["vocals", "drums"]);
    }

    #[test]
    fn parses_audio_speak_transcribe_and_separate() {
        let speak = AudioCli::try_parse_from([
            "werk-audio",
            "speak",
            "xtts",
            "--text-file",
            "speech.txt",
            "--voice",
            "narrator",
            "--voice-reference",
            "voice.wav",
            "--language",
            "de",
            "--accent",
            "de-DE",
            "--speed",
            "1.05",
            "--pitch",
            "-1",
            "--emotion",
            "calm",
            "--emotion-strength",
            "0.7",
            "--stability",
            "0.6",
            "--similarity",
            "0.8",
            "--pronunciation-dictionary",
            "names.dict",
            "--no-phoneme-input",
            "--sample-rate",
            "24000",
            "--streaming",
            "--chunk-size",
            "1024",
            "--format",
            "wav",
            "--output",
            "speech.wav",
        ])
        .unwrap();
        let AudioCommands::Speak(args) = speak.command else {
            panic!("expected audio speak");
        };
        assert_eq!(args.model, "xtts");
        assert_eq!(
            args.text.text_file.as_deref(),
            Some(PathBuf::from("speech.txt").as_path())
        );
        assert!(args.speech.no_phoneme_input);
        assert!(args.speech.streaming);

        let transcribe = AudioCli::try_parse_from([
            "werk-audio",
            "transcribe",
            "whisper-large-v3",
            "--input",
            "meeting.wav",
            "--language",
            "de",
            "--task",
            "translate",
            "--segment-timestamps",
            "--word-timestamps",
            "--diarization",
            "--min-speakers",
            "2",
            "--max-speakers",
            "6",
            "--beam-size",
            "5",
            "--best-of",
            "3",
            "--temperature",
            "0",
            "--temperature-fallback",
            "0.2",
            "--temperature-fallback",
            "0.4",
            "--initial-prompt",
            "Werk1112 project meeting",
            "--hotword",
            "Werk1112",
            "--vad",
            "--suppress-token",
            "-1",
            "--no-condition-on-previous-text",
            "--hallucination-silence-threshold",
            "2",
            "--output-format",
            "json",
            "--output",
            "meeting.json",
        ])
        .unwrap();
        let AudioCommands::Transcribe(args) = transcribe.command else {
            panic!("expected audio transcribe");
        };
        assert_eq!(args.input_audio, PathBuf::from("meeting.wav"));
        assert_eq!(args.transcription.task, Some(SpeechToTextTask::Translate));
        assert!(args.transcription.word_timestamps);
        assert!(args.transcription.diarization);
        assert_eq!(args.transcription.temperature_fallbacks, vec![0.2, 0.4]);
        assert!(args.transcription.no_condition_on_previous_text);

        let separate = AudioCli::try_parse_from([
            "werk-audio",
            "separate",
            "demucs",
            "--input",
            "song.flac",
            "--stem",
            "vocals",
            "--stem",
            "drums",
            "--stems-json",
            "[\"bass\",\"other\"]",
            "--num-stems",
            "4",
            "--segment-duration",
            "12",
            "--segment-overlap",
            "0.25",
            "--sample-rate",
            "44100",
            "--channels",
            "2",
            "--format",
            "flac",
            "--no-normalization",
            "--output",
            "stems/",
        ])
        .unwrap();
        let AudioCommands::Separate(args) = separate.command else {
            panic!("expected audio separate");
        };
        assert_eq!(args.input_audio, PathBuf::from("song.flac"));
        assert_eq!(args.separation.stems, vec!["vocals", "drums"]);
        assert!(args.separation.no_normalization);
    }

    #[test]
    fn audio_source_commands_require_input() {
        assert!(AudioCli::try_parse_from(["werk-audio", "transcribe", "whisper"]).is_err());
        assert!(AudioCli::try_parse_from(["werk-audio", "separate", "demucs"]).is_err());
    }

    #[test]
    fn raw_override_collection_omits_inherited_values_and_parses_set_values() {
        let cli = ImageCli::try_parse_from([
            "werk-image",
            "generate",
            "flux",
            "--width",
            "1024",
            "--no-compile",
            "--set",
            "image.steps=32",
            "--set",
            "image.enabled=true",
            "--set",
            "image.label=cinematic",
        ])
        .unwrap();
        let ImageCommands::Generate(args) = cli.command else {
            panic!("expected image generate");
        };

        let raw = collect_raw_overrides(&args).unwrap();
        assert_eq!(raw.get("model"), Some(&Value::String("flux".to_string())));
        assert_eq!(raw.get("width"), Some(&Value::from(1024)));
        assert_eq!(raw.get("no_compile"), Some(&Value::Bool(true)));
        assert!(!raw.contains_key("compile"));
        assert!(!raw.contains_key("allow_cpu_offload"));

        let set = parse_set_overrides(&args.routing.set).unwrap();
        assert_eq!(set.get("image.steps"), Some(&Value::from(32)));
        assert_eq!(set.get("image.enabled"), Some(&Value::Bool(true)));
        assert_eq!(
            set.get("image.label"),
            Some(&Value::String("cinematic".to_string()))
        );
        assert!(parse_set_overrides(&["missing-equals".to_string()]).is_err());
    }
}
