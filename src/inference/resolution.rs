use anyhow::{Result, anyhow, bail};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    capabilities::{InferenceTask, InputModality},
    model_store::ModelManifest,
};

use super::{
    schema::{
        family_defaults, json_parameter_layer, normalize_parameter_layer, normalize_parameter_path,
        parameter_schema_for_manifest, task_defaults, validate_parameter,
    },
    types::{
        EffectiveInferenceRequest, InferenceRequest, ParameterDescriptor, ParameterPolicy,
        ParameterSource, ParameterSupportStatus, ParameterType, ParameterValue, ResolutionContext,
        ResolvedParameter, RoutingOverrides,
    },
};

pub fn resolve_request(
    manifest: &ModelManifest,
    request: InferenceRequest,
    context: &ResolutionContext,
) -> Result<EffectiveInferenceRequest> {
    if request.model != manifest.id {
        bail!(
            "request model '{}' does not match manifest '{}'",
            request.model,
            manifest.id
        );
    }
    if !manifest.supports_task(request.task) {
        bail!(
            "model '{}' does not declare support for task {}",
            manifest.id,
            request.task
        );
    }

    validate_required_inputs(&request)?;
    let schema = parameter_schema_for_manifest(request.task, manifest)?;
    let descriptors = schema
        .iter()
        .map(|descriptor| (descriptor.path.clone(), descriptor))
        .collect::<BTreeMap<_, _>>();
    let mut parameters = BTreeMap::new();

    apply_descriptor_defaults(&schema, &mut parameters);
    apply_layer(
        schema_parameter_layer(task_defaults(request.task), &descriptors),
        ParameterSource::TaskDefault,
        &mut parameters,
    );
    apply_layer(
        schema_parameter_layer(
            family_defaults(manifest.metadata.family.as_deref(), request.task),
            &descriptors,
        ),
        ParameterSource::ModelFamilyDefault,
        &mut parameters,
    );
    apply_layer(
        schema_parameter_layer(
            json_parameter_layer(
                request.task,
                &manifest.metadata.generation_defaults,
                "model generation default",
            )?,
            &descriptors,
        ),
        ParameterSource::ModelDefault,
        &mut parameters,
    );
    apply_layer(
        schema_parameter_layer(
            normalize_parameter_layer(
                request.task,
                context.runtime_defaults.clone(),
                "runtime default",
            )?,
            &descriptors,
        ),
        ParameterSource::RuntimeDefault,
        &mut parameters,
    );
    apply_layer(
        schema_parameter_layer(
            normalize_parameter_layer(
                request.task,
                context.hardware_profile.clone(),
                "hardware profile",
            )?,
            &descriptors,
        ),
        ParameterSource::HardwareProfile,
        &mut parameters,
    );
    apply_layer(
        schema_parameter_layer(
            normalize_parameter_layer(request.task, context.user_profile.clone(), "user profile")?,
            &descriptors,
        ),
        ParameterSource::UserProfile,
        &mut parameters,
    );

    let mut explicit_parameters = BTreeSet::new();
    for (raw_path, value) in request.parameters {
        let path = normalize_parameter_path(request.task, &raw_path);
        let descriptor = descriptors
            .get(&path)
            .ok_or_else(|| anyhow!("unknown parameter '{raw_path}' for task {}", request.task))?;
        let Some(value) = resolve_list_parameter_override(
            descriptor,
            value,
            parameters.get(&path).map(|resolved| &resolved.value),
        )?
        else {
            continue;
        };
        validate_parameter(descriptor, &value)?;
        parameters.insert(
            path.clone(),
            ResolvedParameter {
                value,
                source: ParameterSource::RequestOverride,
            },
        );
        explicit_parameters.insert(path);
    }

    apply_routing_overrides(&request.routing, &mut parameters, &mut explicit_parameters);

    for (path, resolved) in &parameters {
        let descriptor = descriptors.get(path).ok_or_else(|| {
            anyhow!(
                "resolved {:?} parameter '{}' is not part of the schema for task {}",
                resolved.source,
                path,
                request.task
            )
        })?;
        validate_parameter(descriptor, &resolved.value)?;
    }
    let parameter_policy = parameters
        .get("routing.parameter_policy")
        .and_then(|resolved| resolved.value.as_str())
        .unwrap_or("strict")
        .parse::<ParameterPolicy>()?;

    let mut warnings = Vec::new();
    for path in &explicit_parameters {
        let support = context
            .parameter_support
            .get(path)
            .copied()
            .unwrap_or(ParameterSupportStatus::ModelDependent);
        match support {
            ParameterSupportStatus::Ignored | ParameterSupportStatus::Unsupported => {
                let message = format!("explicit parameter '{path}' is {support:?} by the backend")
                    .to_ascii_lowercase();
                match parameter_policy {
                    ParameterPolicy::Strict => bail!("{message}"),
                    ParameterPolicy::Warn => warnings.push(message),
                    ParameterPolicy::Permissive => {}
                }
            }
            _ => {}
        }
    }

    for descriptor in &schema {
        if !parameters.contains_key(&descriptor.path) {
            parameters.insert(
                descriptor.path.clone(),
                ResolvedParameter {
                    value: ParameterValue::Null,
                    source: ParameterSource::SystemDefault,
                },
            );
        }
    }

    Ok(EffectiveInferenceRequest {
        model: request.model,
        task: request.task,
        prompt: request.prompt,
        negative_prompt: request.negative_prompt,
        inputs: request.inputs,
        output_modality: request.task.output_modality(),
        parameters,
        explicit_parameters,
        parameter_policy,
        warnings,
    })
}

fn resolve_list_parameter_override(
    descriptor: &ParameterDescriptor,
    value: ParameterValue,
    inherited: Option<&ParameterValue>,
) -> Result<Option<ParameterValue>> {
    if descriptor.value_type != ParameterType::List {
        return Ok(Some(value));
    }
    let ParameterValue::Object(mut operation) = value else {
        return Ok(Some(value));
    };
    let operation_name = operation
        .remove("operation")
        .and_then(|value| match value {
            ParameterValue::String(value) => Some(value),
            _ => None,
        })
        .ok_or_else(|| {
            anyhow!(
                "list override '{}' requires a string operation",
                descriptor.path
            )
        })?
        .trim()
        .to_ascii_lowercase();
    let values = operation
        .remove("values")
        .map(|value| match value {
            ParameterValue::List(values) => Ok(values),
            _ => bail!(
                "list override '{}' values must be an array",
                descriptor.path
            ),
        })
        .transpose()?
        .unwrap_or_default();
    if !operation.is_empty() {
        bail!(
            "list override '{}' contains unknown fields: {:?}",
            descriptor.path,
            operation.keys().collect::<Vec<_>>()
        );
    }
    let inherited = match inherited {
        Some(ParameterValue::List(values)) => values.clone(),
        Some(ParameterValue::Null) | None => Vec::new(),
        Some(other) => {
            bail!(
                "inherited value for list parameter '{}' is not a list: {other:?}",
                descriptor.path
            )
        }
    };
    match operation_name.as_str() {
        "inherit" => {
            if !values.is_empty() {
                bail!(
                    "list override '{}' operation 'inherit' does not accept values",
                    descriptor.path
                );
            }
            Ok(None)
        }
        "replace" => Ok(Some(ParameterValue::List(values))),
        "add" | "append" => Ok(Some(ParameterValue::List(
            inherited.into_iter().chain(values).collect(),
        ))),
        "clear" => {
            if !values.is_empty() {
                bail!(
                    "list override '{}' operation 'clear' does not accept values",
                    descriptor.path
                );
            }
            Ok(Some(ParameterValue::List(Vec::new())))
        }
        operation => bail!(
            "unknown list override operation '{operation}' for '{}'; expected inherit, replace, add, or clear",
            descriptor.path
        ),
    }
}

fn validate_required_inputs(request: &InferenceRequest) -> Result<()> {
    if request.task.requires_prompt()
        && request
            .prompt
            .as_deref()
            .map(str::trim)
            .unwrap_or_default()
            .is_empty()
    {
        bail!("task {} requires a non-empty prompt or text", request.task);
    }
    for required in request.task.required_input_modalities() {
        if *required == InputModality::Text {
            continue;
        }
        if !request
            .inputs
            .iter()
            .any(|input| input.modality == *required)
        {
            bail!("task {} requires a {required} input", request.task);
        }
    }
    use InferenceTask::*;
    match request.task {
        ImageUnderstanding | ImageEditing | ImageVariation | ImageUpscaling => {
            require_input_role(
                request,
                InputModality::Image,
                &["image", "input_image", "initial_image"],
                "image",
            )?;
        }
        ImageInpainting | ImageOutpainting => {
            require_input_role(
                request,
                InputModality::Image,
                &["image", "input_image", "initial_image"],
                "source image",
            )?;
            require_input_role(
                request,
                InputModality::Image,
                &["mask", "mask_image"],
                "mask image",
            )?;
        }
        ImageToVideo => {
            require_input_role(
                request,
                InputModality::Image,
                &["initial_image", "image", "input_image"],
                "initial image",
            )?;
        }
        VideoToVideo | VideoExtension | VideoUpscaling | FrameInterpolation => {
            require_input_role(
                request,
                InputModality::Video,
                &["source_video", "input_video", "video"],
                "source video",
            )?;
        }
        VideoInpainting => {
            require_input_role(
                request,
                InputModality::Video,
                &["source_video", "input_video", "video"],
                "source video",
            )?;
            require_input_role(
                request,
                InputModality::Video,
                &["mask_video", "mask"],
                "mask video",
            )?;
        }
        SongContinuation | SongVariation | SpeechToText | StemGeneration | StemSeparation
        | AudioEnhancement => {
            require_input_role(
                request,
                InputModality::Audio,
                &["input_audio", "source_audio", "audio"],
                "input audio",
            )?;
        }
        VoiceConversion => {
            require_input_role(
                request,
                InputModality::Audio,
                &["input_audio", "source_audio", "audio"],
                "source audio",
            )?;
            require_input_role(
                request,
                InputModality::Audio,
                &["reference_audio", "voice_reference"],
                "reference audio",
            )?;
        }
        TextGeneration | TextEmbedding | ImageGeneration | VideoGeneration | AudioGeneration
        | MusicGeneration | TextToSpeech => {}
    }
    Ok(())
}

fn require_input_role(
    request: &InferenceRequest,
    modality: InputModality,
    accepted_roles: &[&str],
    label: &str,
) -> Result<()> {
    if request.inputs.iter().any(|input| {
        input.modality == modality
            && accepted_roles.iter().any(|role| {
                input
                    .role
                    .trim()
                    .replace('-', "_")
                    .eq_ignore_ascii_case(role)
            })
    }) {
        return Ok(());
    }
    bail!(
        "task {} requires a {label} input with role {}",
        request.task,
        accepted_roles.join("|")
    )
}

fn apply_descriptor_defaults(
    schema: &[ParameterDescriptor],
    parameters: &mut BTreeMap<String, ResolvedParameter>,
) {
    for descriptor in schema {
        parameters.insert(
            descriptor.path.clone(),
            ResolvedParameter {
                value: descriptor.default.clone().unwrap_or(ParameterValue::Null),
                source: ParameterSource::SystemDefault,
            },
        );
    }
}

fn apply_layer(
    layer: BTreeMap<String, ParameterValue>,
    source: ParameterSource,
    parameters: &mut BTreeMap<String, ResolvedParameter>,
) {
    for (path, value) in layer {
        parameters.insert(path, ResolvedParameter { value, source });
    }
}

fn schema_parameter_layer(
    layer: BTreeMap<String, ParameterValue>,
    descriptors: &BTreeMap<String, &ParameterDescriptor>,
) -> BTreeMap<String, ParameterValue> {
    layer
        .into_iter()
        .filter(|(path, _)| descriptors.contains_key(path))
        .collect()
}

fn apply_routing_overrides(
    routing: &RoutingOverrides,
    parameters: &mut BTreeMap<String, ResolvedParameter>,
    explicit: &mut BTreeSet<String>,
) {
    let strings = [
        ("routing.backend", routing.backend.as_ref()),
        ("routing.accelerator", routing.accelerator.as_ref()),
        ("routing.device", routing.device.as_ref()),
        ("routing.precision", routing.precision.as_ref()),
        ("routing.quantization", routing.quantization.as_ref()),
        ("routing.profile", routing.profile.as_ref()),
        ("routing.quality", routing.quality.as_ref()),
        (
            "routing.performance_preference",
            routing.performance_preference.as_ref(),
        ),
        ("routing.fallback_policy", routing.fallback_policy.as_ref()),
        (
            "routing.attention_backend",
            routing.attention_backend.as_ref(),
        ),
    ];
    for (path, value) in strings {
        if let Some(value) = value {
            insert_request_override(parameters, explicit, path, value.clone().into());
        }
    }
    let bools = [
        ("routing.allow_cpu_offload", routing.allow_cpu_offload),
        (
            "routing.allow_sequential_offload",
            routing.allow_sequential_offload,
        ),
        (
            "routing.allow_component_offload",
            routing.allow_component_offload,
        ),
        ("routing.allow_disk_offload", routing.allow_disk_offload),
        ("routing.compile", routing.compile),
    ];
    for (path, value) in bools {
        if let Some(value) = value.explicit() {
            insert_request_override(parameters, explicit, path, value.into());
        }
    }
    if let Some(timeout) = routing.timeout_seconds {
        insert_request_override(parameters, explicit, "routing.timeout", timeout.into());
    }
    if routing.parameter_policy != ParameterPolicy::Strict {
        insert_request_override(
            parameters,
            explicit,
            "routing.parameter_policy",
            routing.parameter_policy.to_string().into(),
        );
    }
}

fn insert_request_override(
    parameters: &mut BTreeMap<String, ResolvedParameter>,
    explicit: &mut BTreeSet<String>,
    path: &str,
    value: ParameterValue,
) {
    parameters.insert(
        path.to_string(),
        ResolvedParameter {
            value,
            source: ParameterSource::RequestOverride,
        },
    );
    explicit.insert(path.to_string());
}
