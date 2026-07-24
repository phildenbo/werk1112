mod audio;
mod builders;
mod defaults;
mod image;
mod routing;
mod video;

use anyhow::{Result, anyhow, bail};
use serde_json::Value;
use std::collections::BTreeMap;

use crate::{
    capabilities::{InferenceTask, OutputModality},
    model_store::ModelManifest,
};

use self::{
    audio::{audio_descriptors, stt_descriptors, tts_descriptors},
    image::image_descriptors,
    routing::routing_descriptors,
    video::video_descriptors,
};
use super::types::{ParameterDescriptor, ParameterType, ParameterValue};

pub(super) use defaults::{family_defaults, task_defaults};

pub(super) fn normalize_parameter_path(task: InferenceTask, path: &str) -> String {
    let normalized = path
        .trim()
        .trim_start_matches("--")
        .replace('-', "_")
        .replace("__", ".");
    if normalized.contains('.') || normalized.starts_with("routing_") {
        return normalized.replacen("routing_", "routing.", 1);
    }
    format!("{}.{}", task.parameter_namespace(), normalized)
}

pub(super) fn normalize_parameter_layer(
    task: InferenceTask,
    layer: BTreeMap<String, ParameterValue>,
    label: &str,
) -> Result<BTreeMap<String, ParameterValue>> {
    let mut normalized = BTreeMap::new();
    for (raw_path, value) in layer {
        let path = normalize_parameter_path(task, &raw_path);
        if normalized.insert(path.clone(), value).is_some() {
            bail!("{label} contains duplicate parameter '{path}' after normalization");
        }
    }
    Ok(normalized)
}

pub(super) fn json_parameter_layer(
    task: InferenceTask,
    layer: &BTreeMap<String, Value>,
    label: &str,
) -> Result<BTreeMap<String, ParameterValue>> {
    let mut converted = BTreeMap::new();
    for (raw_path, value) in layer {
        let path = normalize_parameter_path(task, raw_path);
        let value = ParameterValue::from_json(value.clone()).map_err(|error| {
            anyhow!("{label} '{raw_path}' cannot be converted to a parameter value: {error}")
        })?;
        if converted.insert(path.clone(), value).is_some() {
            bail!("{label} contains duplicate parameter '{path}' after normalization");
        }
    }
    Ok(converted)
}

pub(super) fn validate_parameter(
    descriptor: &ParameterDescriptor,
    value: &ParameterValue,
) -> Result<()> {
    let valid_type = match descriptor.value_type {
        ParameterType::Boolean => matches!(value, ParameterValue::Boolean(_)),
        ParameterType::Integer => matches!(value, ParameterValue::Integer(_)),
        ParameterType::Number => value.as_f64().is_some(),
        ParameterType::String | ParameterType::Path | ParameterType::Enumeration => {
            matches!(value, ParameterValue::String(_))
        }
        ParameterType::List => matches!(value, ParameterValue::List(_)),
        ParameterType::Object => matches!(value, ParameterValue::Object(_)),
    };
    if !valid_type && !matches!(value, ParameterValue::Null) {
        bail!(
            "parameter '{}' expects {:?}, got {value:?}",
            descriptor.path,
            descriptor.value_type
        );
    }
    if let Some(number) = value.as_f64() {
        if descriptor
            .minimum
            .as_ref()
            .and_then(ParameterValue::as_f64)
            .is_some_and(|minimum| number < minimum)
        {
            bail!("parameter '{}' is below its minimum", descriptor.path);
        }
        if descriptor
            .maximum
            .as_ref()
            .and_then(ParameterValue::as_f64)
            .is_some_and(|maximum| number > maximum)
        {
            bail!("parameter '{}' is above its maximum", descriptor.path);
        }
    }
    if !descriptor.allowed_values.is_empty()
        && !descriptor
            .allowed_values
            .iter()
            .any(|allowed| allowed == value)
    {
        bail!(
            "parameter '{}' must be one of {:?}",
            descriptor.path,
            descriptor.allowed_values
        );
    }
    Ok(())
}

pub fn parameter_schema(task: InferenceTask) -> Vec<ParameterDescriptor> {
    let mut descriptors = BTreeMap::new();
    for descriptor in routing_descriptors()
        .into_iter()
        .chain(task_descriptors(task))
    {
        descriptors.insert(descriptor.path.clone(), descriptor);
    }
    descriptors.into_values().collect()
}

pub fn parameter_schema_for_manifest(
    task: InferenceTask,
    manifest: &ModelManifest,
) -> Result<Vec<ParameterDescriptor>> {
    let mut schema = parameter_schema(task);
    let by_path = schema
        .iter()
        .enumerate()
        .map(|(index, descriptor)| (descriptor.path.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let model_defaults = json_parameter_layer(
        task,
        &manifest.metadata.generation_defaults,
        "model generation default",
    )?;
    for layer in [
        task_defaults(task),
        family_defaults(manifest.metadata.family.as_deref(), task),
        model_defaults,
    ] {
        for (path, value) in layer {
            let Some(index) = by_path.get(&path).copied() else {
                continue;
            };
            schema[index].default = Some(value);
        }
    }
    for (raw_path, constraint) in &manifest.metadata.parameter_constraints {
        let path = normalize_parameter_path(task, raw_path);
        let Some(index) = by_path.get(&path).copied() else {
            continue;
        };
        apply_manifest_constraint(&mut schema[index], constraint).map_err(|error| {
            anyhow!(
                "invalid parameter constraint '{}' in model '{}': {error}",
                raw_path,
                manifest.id
            )
        })?;
    }
    Ok(schema)
}

fn apply_manifest_constraint(
    descriptor: &mut ParameterDescriptor,
    constraint: &Value,
) -> Result<()> {
    let Some(object) = constraint.as_object() else {
        if !constraint.is_null() {
            let value = ParameterValue::from_json(constraint.clone())?;
            validate_parameter(descriptor, &value)?;
            descriptor.default = Some(value.clone());
            descriptor.allowed_values = vec![value];
        }
        return Ok(());
    };
    let recognized = [
        "default",
        "minimum",
        "min",
        "maximum",
        "max",
        "step",
        "allowed_values",
        "enum",
    ];
    if !object.keys().any(|key| recognized.contains(&key.as_str())) {
        return Ok(());
    }
    let unknown = object
        .keys()
        .filter(|key| !recognized.contains(&key.as_str()))
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        bail!("unknown constraint fields: {unknown:?}");
    }
    let convert = |name: &str| -> Result<Option<ParameterValue>> {
        object
            .get(name)
            .cloned()
            .map(ParameterValue::from_json)
            .transpose()
    };
    if let Some(default) = convert("default")? {
        descriptor.default = Some(default);
    }
    descriptor.minimum = convert("minimum")?.or(convert("min")?);
    descriptor.maximum = convert("maximum")?.or(convert("max")?);
    if let Some(step) = convert("step")? {
        descriptor.step = Some(step);
    }
    if let Some(values) = object.get("allowed_values").or_else(|| object.get("enum")) {
        let values = values
            .as_array()
            .ok_or_else(|| anyhow!("allowed_values must be an array"))?;
        descriptor.allowed_values = values
            .iter()
            .cloned()
            .map(ParameterValue::from_json)
            .collect::<Result<Vec<_>>>()?;
    }
    if let (Some(minimum), Some(maximum)) = (
        descriptor.minimum.as_ref().and_then(ParameterValue::as_f64),
        descriptor.maximum.as_ref().and_then(ParameterValue::as_f64),
    ) && minimum > maximum
    {
        bail!("minimum exceeds maximum");
    }
    for (name, bound) in [
        ("minimum", descriptor.minimum.as_ref()),
        ("maximum", descriptor.maximum.as_ref()),
        ("step", descriptor.step.as_ref()),
    ] {
        if let Some(bound) = bound
            && bound.as_f64().is_none()
        {
            bail!("{name} must be numeric");
        }
    }
    if descriptor
        .step
        .as_ref()
        .and_then(ParameterValue::as_f64)
        .is_some_and(|step| step <= 0.0)
    {
        bail!("step must be positive");
    }
    if let Some(default) = descriptor.default.as_ref() {
        validate_parameter(descriptor, default)?;
    }
    for allowed in &descriptor.allowed_values {
        validate_parameter(descriptor, allowed)?;
    }
    Ok(())
}

fn task_descriptors(task: InferenceTask) -> Vec<ParameterDescriptor> {
    match task.output_modality() {
        OutputModality::Image => image_descriptors(task),
        OutputModality::Video => video_descriptors(task),
        OutputModality::Audio => match task {
            InferenceTask::TextToSpeech => tts_descriptors(),
            InferenceTask::AudioGeneration
            | InferenceTask::MusicGeneration
            | InferenceTask::SongContinuation
            | InferenceTask::SongVariation
            | InferenceTask::StemSeparation => audio_descriptors(task),
            InferenceTask::VoiceConversion
            | InferenceTask::StemGeneration
            | InferenceTask::AudioEnhancement => Vec::new(),
            _ => Vec::new(),
        },
        OutputModality::Text if task == InferenceTask::SpeechToText => stt_descriptors(),
        OutputModality::Text | OutputModality::Embedding => Vec::new(),
    }
}
