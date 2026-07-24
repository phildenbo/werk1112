use serde::Deserialize;
use std::str::FromStr;

use crate::{
    capabilities::{InferenceTask, InputModality},
    inference::{InferenceInput, InferenceRequest},
};

use super::input::{ApiMediaInput, WerkRequestOptions};

#[derive(Debug, Clone, Copy)]
pub(in crate::api) enum DirectResponseFormat {
    Url,
    Base64,
}

#[derive(Debug, Clone, Deserialize)]
pub(in crate::api) struct ImageGenerationApiRequest {
    model: String,
    prompt: String,
    #[serde(default)]
    negative_prompt: Option<String>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(flatten)]
    werk: WerkRequestOptions,
}

impl ImageGenerationApiRequest {
    pub(in crate::api) fn into_inference(
        self,
    ) -> Result<(InferenceRequest, DirectResponseFormat), String> {
        let response_format = image_response_format(self.response_format.as_deref())?;
        let mut request = media_request(
            self.model,
            InferenceTask::ImageGeneration,
            Some(self.prompt),
            self.negative_prompt,
            self.werk,
        )?;
        if let Some(n) = self.n {
            request
                .parameters
                .insert("image.num_images".to_string(), n.into());
        }
        apply_size(&mut request, "image", self.size.as_deref())?;
        Ok((request, response_format))
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(in crate::api) struct ImageEditApiRequest {
    model: String,
    prompt: String,
    image: ApiMediaInput,
    #[serde(default)]
    mask: Option<ApiMediaInput>,
    #[serde(default)]
    negative_prompt: Option<String>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(flatten)]
    werk: WerkRequestOptions,
}

impl ImageEditApiRequest {
    pub(in crate::api) fn into_inference(
        self,
    ) -> Result<(InferenceRequest, DirectResponseFormat), String> {
        let response_format = image_response_format(self.response_format.as_deref())?;
        let task = if self.mask.is_some() {
            InferenceTask::ImageInpainting
        } else {
            InferenceTask::ImageEditing
        };
        let mut request = media_request(
            self.model,
            task,
            Some(self.prompt),
            self.negative_prompt,
            self.werk,
        )?;
        request
            .inputs
            .push(self.image.into_inference(InputModality::Image, "image")?);
        if let Some(mask) = self.mask {
            request
                .inputs
                .push(mask.into_inference(InputModality::Image, "mask")?);
        }
        if let Some(n) = self.n {
            request
                .parameters
                .insert("image.num_images".to_string(), n.into());
        }
        apply_size(&mut request, "image", self.size.as_deref())?;
        Ok((request, response_format))
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(in crate::api) struct VideoGenerationApiRequest {
    model: String,
    prompt: String,
    #[serde(default, alias = "image")]
    initial_image: Option<ApiMediaInput>,
    #[serde(default)]
    negative_prompt: Option<String>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(flatten)]
    werk: WerkRequestOptions,
}

impl VideoGenerationApiRequest {
    pub(super) fn into_inference(self) -> Result<InferenceRequest, String> {
        let task = if self.initial_image.is_some() {
            InferenceTask::ImageToVideo
        } else {
            InferenceTask::VideoGeneration
        };
        let mut request = media_request(
            self.model,
            task,
            Some(self.prompt),
            self.negative_prompt,
            self.werk,
        )?;
        if let Some(initial_image) = self.initial_image {
            request
                .inputs
                .push(initial_image.into_inference(InputModality::Image, "initial_image")?);
        }
        if let Some(n) = self.n {
            request
                .parameters
                .insert("video.num_videos".to_string(), n.into());
        }
        apply_size(&mut request, "video", self.size.as_deref())?;
        apply_output_format(&mut request, "video", self.response_format.as_deref());
        Ok(request)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(in crate::api) struct AudioGenerationApiRequest {
    model: String,
    prompt: String,
    #[serde(default)]
    negative_prompt: Option<String>,
    #[serde(default)]
    task: Option<String>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(flatten)]
    werk: WerkRequestOptions,
}

impl AudioGenerationApiRequest {
    pub(super) fn into_inference(self) -> Result<InferenceRequest, String> {
        let task = match self.task.as_deref() {
            None => InferenceTask::AudioGeneration,
            Some(value) => {
                let task = InferenceTask::from_str(value)?;
                if !matches!(
                    task,
                    InferenceTask::AudioGeneration | InferenceTask::MusicGeneration
                ) {
                    return Err(
                        "audio generations supports only audio-generation or music-generation"
                            .to_string(),
                    );
                }
                task
            }
        };
        let mut request = media_request(
            self.model,
            task,
            Some(self.prompt),
            self.negative_prompt,
            self.werk,
        )?;
        if let Some(n) = self.n {
            request
                .parameters
                .insert("audio.variations".to_string(), n.into());
        }
        apply_output_format(&mut request, "audio", self.response_format.as_deref());
        Ok(request)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(in crate::api) struct AudioSpeechApiRequest {
    model: String,
    #[serde(alias = "text")]
    input: String,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    speed: Option<f64>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default, rename = "async", alias = "background", alias = "job")]
    asynchronous: bool,
    #[serde(flatten)]
    werk: WerkRequestOptions,
}

impl AudioSpeechApiRequest {
    pub(super) fn into_inference(self) -> Result<(InferenceRequest, bool), String> {
        let mut request = media_request(
            self.model,
            InferenceTask::TextToSpeech,
            Some(self.input),
            None,
            self.werk,
        )?;
        if let Some(voice) = self.voice {
            request
                .parameters
                .insert("tts.voice".to_string(), voice.into());
        }
        if let Some(speed) = self.speed {
            request
                .parameters
                .insert("tts.speed".to_string(), speed.into());
        }
        apply_output_format(&mut request, "tts", self.response_format.as_deref());
        Ok((request, self.asynchronous))
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(in crate::api) struct AudioTranscriptionApiRequest {
    model: String,
    #[serde(alias = "audio", alias = "input_audio")]
    file: ApiMediaInput,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(flatten)]
    werk: WerkRequestOptions,
}

impl AudioTranscriptionApiRequest {
    pub(super) fn into_inference(self) -> Result<InferenceRequest, String> {
        let mut request = media_request(
            self.model,
            InferenceTask::SpeechToText,
            self.prompt,
            None,
            self.werk,
        )?;
        request.inputs.push(
            self.file
                .into_inference(InputModality::Audio, "input_audio")?,
        );
        if let Some(language) = self.language {
            request
                .parameters
                .insert("stt.language".to_string(), language.into());
        }
        if let Some(temperature) = self.temperature {
            request
                .parameters
                .insert("stt.temperature".to_string(), temperature.into());
        }
        apply_output_format(&mut request, "stt", self.response_format.as_deref());
        Ok(request)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(in crate::api) struct JobCreateApiRequest {
    model: String,
    task: String,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    negative_prompt: Option<String>,
    #[serde(default)]
    inputs: Vec<InferenceInput>,
    #[serde(flatten)]
    werk: WerkRequestOptions,
}

impl JobCreateApiRequest {
    pub(super) fn into_inference(self) -> Result<InferenceRequest, String> {
        let task = InferenceTask::from_str(&self.task)?;
        let mut request = media_request(
            self.model,
            task,
            self.prompt,
            self.negative_prompt,
            self.werk,
        )?;
        request.inputs = self.inputs;
        Ok(request)
    }
}

fn media_request(
    model: String,
    task: InferenceTask,
    prompt: Option<String>,
    negative_prompt: Option<String>,
    werk: WerkRequestOptions,
) -> Result<InferenceRequest, String> {
    let (parameters, routing) = werk.into_parts(task.parameter_namespace())?;
    Ok(InferenceRequest {
        model,
        task,
        prompt,
        negative_prompt,
        inputs: Vec::new(),
        parameters,
        routing,
    })
}

fn apply_size(
    request: &mut InferenceRequest,
    namespace: &str,
    size: Option<&str>,
) -> Result<(), String> {
    let Some(size) = size else {
        return Ok(());
    };
    let normalized = size.trim().to_ascii_lowercase();
    let (width, height) = normalized
        .split_once('x')
        .ok_or_else(|| "size must use WIDTHxHEIGHT, for example 1024x1024".to_string())?;
    let width = width
        .trim()
        .parse::<u32>()
        .map_err(|_| "size width must be a positive integer".to_string())?;
    let height = height
        .trim()
        .parse::<u32>()
        .map_err(|_| "size height must be a positive integer".to_string())?;
    if width == 0 || height == 0 {
        return Err("size dimensions must be greater than zero".to_string());
    }
    request
        .parameters
        .insert(format!("{namespace}.width"), width.into());
    request
        .parameters
        .insert(format!("{namespace}.height"), height.into());
    Ok(())
}

fn image_response_format(value: Option<&str>) -> Result<DirectResponseFormat, String> {
    match value.unwrap_or("url").trim().to_ascii_lowercase().as_str() {
        "url" => Ok(DirectResponseFormat::Url),
        "b64_json" | "base64" => Ok(DirectResponseFormat::Base64),
        value => Err(format!(
            "image response_format must be 'url' or 'b64_json', got '{value}'"
        )),
    }
}

fn apply_output_format(
    request: &mut InferenceRequest,
    namespace: &str,
    output_format: Option<&str>,
) {
    if let Some(output_format) = output_format {
        request.parameters.insert(
            format!("{namespace}.output_format"),
            output_format.to_string().into(),
        );
    }
}
