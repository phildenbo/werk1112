use super::super::types::{ParameterDescriptor, ParameterType, ParameterValue};

pub(super) fn add_named_descriptors(
    values: &mut Vec<ParameterDescriptor>,
    namespace: &str,
    category: &str,
    specs: &[(&str, ParameterType)],
) {
    for (name, value_type) in specs {
        let cli_flag = cli_flag_for_parameter(namespace, name);
        let repeatable = matches!(value_type, ParameterType::List);
        let default = match value_type {
            ParameterType::Boolean => Some(ParameterValue::Boolean(false)),
            ParameterType::List => Some(ParameterValue::List(Vec::new())),
            _ => Some(ParameterValue::Null),
        };
        values.push(ParameterDescriptor {
            path: format!("{namespace}.{name}"),
            cli_flag,
            value_type: *value_type,
            label: humanize(name),
            description: format!("Normalized {namespace} parameter {}", humanize(name)),
            category: category.to_string(),
            default,
            minimum: None,
            maximum: None,
            step: None,
            allowed_values: Vec::new(),
            repeatable,
            list_override_operations: if repeatable {
                ["inherit", "replace", "add", "clear"]
                    .into_iter()
                    .map(str::to_string)
                    .collect()
            } else {
                Vec::new()
            },
            advanced: true,
            affects_memory: affects_memory(name),
            affects_quality: affects_quality(name),
            affects_runtime: affects_runtime(name),
        });
    }
}

fn cli_flag_for_parameter(namespace: &str, name: &str) -> String {
    let special = match (namespace, name) {
        ("image", "controls") => Some("image-control"),
        ("image", "reference_images") => Some("reference-image"),
        ("image", "loras") => Some("image-lora"),
        ("image", "adapters") => Some("image-adapter"),
        ("image", "sigmas") => Some("sigma"),
        ("image", "post_upscaling") => Some("post-upscale"),
        ("video", "controls") => Some("video-control"),
        ("video", "reference_images") => Some("reference-image"),
        ("video", "camera_keyframes") => Some("video-camera-keyframe"),
        ("video", "prompt_keyframes") => Some("video-prompt-keyframe"),
        ("video", "guidance_schedule") => Some("video-guidance-schedule"),
        ("video", "denoise_schedule") => Some("video-denoise-schedule"),
        ("video", "adapters") => Some("video-adapter"),
        ("video", "adapter_schedule") => Some("video-adapter-schedule"),
        ("video", "sigmas") => Some("sigma"),
        ("video", "looping") => Some("loop"),
        ("audio", "genres") => Some("genre"),
        ("audio", "subgenres") => Some("subgenre"),
        ("audio", "styles") => Some("style"),
        ("audio", "eras") => Some("era"),
        ("audio", "influences") => Some("influence"),
        ("audio", "moods") => Some("mood"),
        ("audio", "themes") => Some("theme"),
        ("audio", "descriptors") => Some("descriptor"),
        ("audio", "tempo_min") => Some("bpm-min"),
        ("audio", "tempo_max") => Some("bpm-max"),
        ("audio", "instruments") => Some("music-instrument"),
        ("audio", "excluded_instruments") => Some("excluded-instrument"),
        ("audio", "rhythm_instruments") => Some("rhythm-instrument"),
        ("audio", "register") => Some("vocal-register"),
        ("audio", "range") => Some("vocal-range"),
        ("audio", "language") => Some("vocal-language"),
        ("audio", "accent") => Some("vocal-accent"),
        ("audio", "delivery") => Some("vocal-delivery"),
        ("audio", "emotion") => Some("vocal-emotion"),
        ("audio", "power") => Some("vocal-power"),
        ("audio", "mix_controls") => Some("music-mix-control"),
        ("audio", "mastering_controls") => Some("music-mastering-control"),
        ("audio", "stems") => Some("stem"),
        ("tts", "pronunciation_dictionaries") => Some("pronunciation-dictionary"),
        ("tts", "loudness") => Some("loudness-lufs"),
        ("stt", "temperature_fallbacks") => Some("temperature-fallback"),
        ("stt", "hotwords") => Some("hotword"),
        ("stt", "suppress_tokens") => Some("suppress-token"),
        ("stt", "operation") => Some("task"),
        ("stt", "input_audio") => Some("input"),
        (_, "output_path") => Some("output"),
        _ => None,
    };
    format!("--{}", special.unwrap_or(name).replace('_', "-"))
}

pub(super) fn string_descriptor(
    path: &str,
    cli_flag: &str,
    label: &str,
    description: &str,
    category: &str,
    default: Option<&str>,
) -> ParameterDescriptor {
    ParameterDescriptor {
        path: path.to_string(),
        cli_flag: cli_flag.to_string(),
        value_type: ParameterType::String,
        label: label.to_string(),
        description: description.to_string(),
        category: category.to_string(),
        default: default.map(ParameterValue::from),
        minimum: None,
        maximum: None,
        step: None,
        allowed_values: Vec::new(),
        repeatable: false,
        list_override_operations: Vec::new(),
        advanced: category == "routing",
        affects_memory: affects_memory(path),
        affects_quality: affects_quality(path),
        affects_runtime: affects_runtime(path),
    }
}

pub(super) fn enum_descriptor(
    path: &str,
    cli_flag: &str,
    label: &str,
    description: &str,
    category: &str,
    default: &str,
    allowed: &[&str],
) -> ParameterDescriptor {
    let mut descriptor =
        string_descriptor(path, cli_flag, label, description, category, Some(default));
    descriptor.value_type = ParameterType::Enumeration;
    descriptor.allowed_values = allowed.iter().copied().map(ParameterValue::from).collect();
    descriptor
}

pub(super) fn bool_descriptor(
    path: &str,
    cli_flag: &str,
    label: &str,
    description: &str,
    category: &str,
    default: bool,
) -> ParameterDescriptor {
    ParameterDescriptor {
        path: path.to_string(),
        cli_flag: cli_flag.to_string(),
        value_type: ParameterType::Boolean,
        label: label.to_string(),
        description: description.to_string(),
        category: category.to_string(),
        default: Some(default.into()),
        minimum: None,
        maximum: None,
        step: None,
        allowed_values: Vec::new(),
        repeatable: false,
        list_override_operations: Vec::new(),
        advanced: category == "routing" || category == "memory",
        affects_memory: affects_memory(path),
        affects_quality: affects_quality(path),
        affects_runtime: affects_runtime(path),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn number_descriptor(
    path: &str,
    cli_flag: &str,
    label: &str,
    description: &str,
    category: &str,
    default: f64,
    minimum: f64,
    maximum: f64,
    step: f64,
) -> ParameterDescriptor {
    ParameterDescriptor {
        path: path.to_string(),
        cli_flag: cli_flag.to_string(),
        value_type: ParameterType::Number,
        label: label.to_string(),
        description: description.to_string(),
        category: category.to_string(),
        default: Some(default.into()),
        minimum: Some(minimum.into()),
        maximum: Some(maximum.into()),
        step: Some(step.into()),
        allowed_values: Vec::new(),
        repeatable: false,
        list_override_operations: Vec::new(),
        advanced: false,
        affects_memory: affects_memory(path),
        affects_quality: affects_quality(path),
        affects_runtime: affects_runtime(path),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn integer_descriptor(
    path: &str,
    cli_flag: &str,
    label: &str,
    description: &str,
    category: &str,
    default: i64,
    minimum: i64,
    maximum: i64,
) -> ParameterDescriptor {
    ParameterDescriptor {
        path: path.to_string(),
        cli_flag: cli_flag.to_string(),
        value_type: ParameterType::Integer,
        label: label.to_string(),
        description: description.to_string(),
        category: category.to_string(),
        default: Some(default.into()),
        minimum: Some(minimum.into()),
        maximum: Some(maximum.into()),
        step: Some(1_i64.into()),
        allowed_values: Vec::new(),
        repeatable: false,
        list_override_operations: Vec::new(),
        advanced: false,
        affects_memory: affects_memory(path),
        affects_quality: affects_quality(path),
        affects_runtime: affects_runtime(path),
    }
}

fn humanize(name: &str) -> String {
    let mut value = name.replace('_', " ");
    if let Some(first) = value.get_mut(0..1) {
        first.make_ascii_uppercase();
    }
    value
}

fn affects_memory(name: &str) -> bool {
    [
        "width",
        "height",
        "frames",
        "batch",
        "tile",
        "offload",
        "precision",
        "quantization",
        "context",
        "chunk",
        "stems",
        "compile",
    ]
    .iter()
    .any(|needle| name.contains(needle))
}

fn affects_quality(name: &str) -> bool {
    [
        "steps",
        "guidance",
        "sampler",
        "scheduler",
        "strength",
        "adherence",
        "quality",
        "precision",
        "bitrate",
        "sample_rate",
        "crf",
    ]
    .iter()
    .any(|needle| name.contains(needle))
}

fn affects_runtime(name: &str) -> bool {
    affects_memory(name)
        || [
            "steps",
            "sampler",
            "scheduler",
            "duration",
            "fps",
            "interpolation",
            "upscal",
            "compile",
        ]
        .iter()
        .any(|needle| name.contains(needle))
}
