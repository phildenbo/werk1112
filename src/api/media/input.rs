use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

use crate::{
    capabilities::InputModality,
    inference::{
        InferenceInput, InferenceInputSource, OverrideBool, ParameterPolicy, ParameterValue,
        RoutingOverrides,
    },
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(super) struct WerkRequestOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    routing: Option<RoutingOverrides>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    accelerator: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    device: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    precision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    quantization: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    quality: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    performance_preference: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    fallback_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parameter_policy: Option<ParameterPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    allow_cpu_offload: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    allow_sequential_offload: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    allow_component_offload: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    allow_disk_offload: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    attention_backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compile: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    timeout_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    user: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    parameters: BTreeMap<String, Value>,
    #[serde(default, flatten)]
    extra_parameters: BTreeMap<String, Value>,
}

impl WerkRequestOptions {
    pub(super) fn into_parts(
        self,
        namespace: &str,
    ) -> Result<(BTreeMap<String, ParameterValue>, RoutingOverrides), String> {
        let mut routing = self.routing.unwrap_or_default();
        replace_some(&mut routing.backend, self.backend);
        replace_some(&mut routing.accelerator, self.accelerator);
        replace_some(&mut routing.device, self.device);
        replace_some(&mut routing.precision, self.precision);
        replace_some(&mut routing.quantization, self.quantization);
        replace_some(&mut routing.profile, self.profile);
        replace_some(
            &mut routing.quality,
            self.quality.map(|value| normalize_quality(&value)),
        );
        replace_some(
            &mut routing.performance_preference,
            self.performance_preference,
        );
        replace_some(&mut routing.fallback_policy, self.fallback_policy);
        replace_some(&mut routing.attention_backend, self.attention_backend);
        replace_some(&mut routing.timeout_seconds, self.timeout_seconds);
        if let Some(value) = self.parameter_policy {
            routing.parameter_policy = value;
        }
        replace_override(&mut routing.allow_cpu_offload, self.allow_cpu_offload);
        replace_override(
            &mut routing.allow_sequential_offload,
            self.allow_sequential_offload,
        );
        replace_override(
            &mut routing.allow_component_offload,
            self.allow_component_offload,
        );
        replace_override(&mut routing.allow_disk_offload, self.allow_disk_offload);
        replace_override(&mut routing.compile, self.compile);

        // `user` is an OpenAI compatibility field. It intentionally remains
        // transport metadata rather than becoming a backend parameter.
        let _user = self.user;
        let mut raw_parameters = self.parameters;
        raw_parameters.extend(self.extra_parameters);
        let mut parameters = BTreeMap::new();
        for (name, value) in raw_parameters {
            if name == namespace {
                let Value::Object(nested) = value else {
                    return Err(format!("'{namespace}' must be a JSON object"));
                };
                for (nested_name, nested_value) in nested {
                    insert_json_parameter(
                        &mut parameters,
                        format!("{namespace}.{nested_name}"),
                        nested_value,
                    )?;
                }
                continue;
            }
            let path = if name.contains('.') {
                name
            } else {
                format!("{namespace}.{name}")
            };
            insert_json_parameter(&mut parameters, path, value)?;
        }
        Ok((parameters, routing))
    }
}

fn replace_some<T>(target: &mut Option<T>, replacement: Option<T>) {
    if replacement.is_some() {
        *target = replacement;
    }
}

fn replace_override(target: &mut OverrideBool, replacement: Option<bool>) {
    if let Some(value) = replacement {
        *target = if value {
            OverrideBool::Enabled
        } else {
            OverrideBool::Disabled
        };
    }
}

fn normalize_quality(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "standard" => "balanced".to_string(),
        "hd" => "high".to_string(),
        _ => value.to_string(),
    }
}

fn insert_json_parameter(
    parameters: &mut BTreeMap<String, ParameterValue>,
    path: String,
    value: Value,
) -> Result<(), String> {
    let value = ParameterValue::from_json(value)
        .map_err(|error| format!("invalid parameter '{path}': {error}"))?;
    parameters.insert(path, value);
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub(in crate::api) enum ApiMediaInput {
    String(String),
    Object(ApiMediaInputObject),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(in crate::api) struct ApiMediaInputObject {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(in crate::api) path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(in crate::api) url: Option<String>,
    #[serde(
        default,
        alias = "b64_json",
        alias = "data",
        skip_serializing_if = "Option::is_none"
    )]
    pub(in crate::api) base64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(in crate::api) mime_type: Option<String>,
}

impl ApiMediaInput {
    pub(super) fn into_inference(
        self,
        modality: InputModality,
        role: &str,
    ) -> Result<InferenceInput, String> {
        let (source, mime_type) = match self {
            Self::String(value) => classify_input_string(value)?,
            Self::Object(input) => {
                let source_count = usize::from(input.path.is_some())
                    + usize::from(input.url.is_some())
                    + usize::from(input.base64.is_some());
                if source_count != 1 {
                    return Err(
                        "media input must contain exactly one of path, url, or base64".to_string(),
                    );
                }
                let (source, detected_mime_type) = if let Some(path) = input.path {
                    if path.trim().is_empty() {
                        return Err("media input path must not be empty".to_string());
                    }
                    (InferenceInputSource::Path { path }, None)
                } else if let Some(url) = input.url {
                    classify_input_string(url)?
                } else {
                    let data = input.base64.unwrap_or_default();
                    if data.trim().is_empty() {
                        return Err("media input base64 must not be empty".to_string());
                    }
                    (InferenceInputSource::Base64 { data }, None)
                };
                (source, input.mime_type.or(detected_mime_type))
            }
        };
        Ok(InferenceInput {
            modality,
            role: role.to_string(),
            source,
            mime_type,
        })
    }
}

fn classify_input_string(value: String) -> Result<(InferenceInputSource, Option<String>), String> {
    if value.trim().is_empty() {
        return Err("media input must not be empty".to_string());
    }
    if value.starts_with("http://") || value.starts_with("https://") {
        return Ok((InferenceInputSource::Url { url: value }, None));
    }
    if let Some(path) = value.strip_prefix("file://") {
        return Ok((
            InferenceInputSource::Path {
                path: path.to_string(),
            },
            None,
        ));
    }
    if let Some(data_url) = value.strip_prefix("data:") {
        let (metadata, data) = data_url
            .split_once(',')
            .ok_or_else(|| "invalid data URL input".to_string())?;
        if !metadata
            .split(';')
            .any(|component| component.eq_ignore_ascii_case("base64"))
        {
            return Err("data URL input must use base64 encoding".to_string());
        }
        if data.trim().is_empty() {
            return Err("data URL input must contain base64 data".to_string());
        }
        let mime_type = metadata
            .split(';')
            .next()
            .filter(|value| !value.is_empty())
            .map(ToString::to_string);
        return Ok((
            InferenceInputSource::Base64 {
                data: data.to_string(),
            },
            mime_type,
        ));
    }
    Ok((InferenceInputSource::Path { path: value }, None))
}
