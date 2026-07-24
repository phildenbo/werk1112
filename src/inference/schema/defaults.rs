use std::collections::BTreeMap;

use crate::capabilities::{InferenceTask, OutputModality};

use super::super::types::ParameterValue;

pub(in crate::inference) fn task_defaults(task: InferenceTask) -> BTreeMap<String, ParameterValue> {
    let values: &[(&str, ParameterValue)] = match task.output_modality() {
        OutputModality::Image => &[
            ("image.width", ParameterValue::Integer(1024)),
            ("image.height", ParameterValue::Integer(1024)),
            ("image.steps", ParameterValue::Integer(28)),
            ("image.guidance", ParameterValue::Number(7.0)),
            ("image.batch_size", ParameterValue::Integer(1)),
            ("image.num_images", ParameterValue::Integer(1)),
        ],
        OutputModality::Video => &[
            ("video.width", ParameterValue::Integer(832)),
            ("video.height", ParameterValue::Integer(480)),
            ("video.frames", ParameterValue::Integer(81)),
            ("video.fps", ParameterValue::Number(24.0)),
            ("video.steps", ParameterValue::Integer(30)),
            ("video.batch_size", ParameterValue::Integer(1)),
            ("video.num_videos", ParameterValue::Integer(1)),
        ],
        OutputModality::Audio if task == InferenceTask::TextToSpeech => &[
            ("tts.speed", ParameterValue::Number(1.0)),
            ("tts.sample_rate", ParameterValue::Integer(24_000)),
            ("tts.channels", ParameterValue::Integer(1)),
        ],
        OutputModality::Text if task == InferenceTask::SpeechToText => &[
            ("stt.beam_size", ParameterValue::Integer(5)),
            ("stt.best_of", ParameterValue::Integer(5)),
            ("stt.temperature", ParameterValue::Number(0.0)),
        ],
        OutputModality::Audio => &[
            ("audio.duration", ParameterValue::Number(30.0)),
            ("audio.variations", ParameterValue::Integer(1)),
            ("audio.sample_rate", ParameterValue::Integer(44_100)),
            ("audio.channels", ParameterValue::Integer(2)),
        ],
        OutputModality::Text | OutputModality::Embedding => &[],
    };
    values
        .iter()
        .cloned()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
}

pub(in crate::inference) fn family_defaults(
    family: Option<&str>,
    task: InferenceTask,
) -> BTreeMap<String, ParameterValue> {
    let family = family.unwrap_or_default().to_ascii_lowercase();
    let mut values = BTreeMap::new();
    if task.output_modality() == OutputModality::Image {
        if family.contains("flux") {
            values.insert("image.steps".to_string(), 28_i64.into());
            values.insert("image.guidance".to_string(), 3.5_f64.into());
            values.insert("image.sampler".to_string(), "flow_match".into());
        } else if family.contains("stable-diffusion-xl") || family.contains("sdxl") {
            values.insert("image.steps".to_string(), 30_i64.into());
            values.insert("image.guidance".to_string(), 5.0_f64.into());
        }
    }
    if task.output_modality() == OutputModality::Video {
        if family.contains("wan") {
            values.insert("video.frames".to_string(), 81_i64.into());
            values.insert("video.fps".to_string(), 16.0_f64.into());
        } else if family.contains("cogvideo") {
            values.insert("video.frames".to_string(), 49_i64.into());
            values.insert("video.fps".to_string(), 8.0_f64.into());
        }
    }
    if matches!(
        task,
        InferenceTask::MusicGeneration
            | InferenceTask::SongContinuation
            | InferenceTask::SongVariation
    ) && family.contains("musicgen")
    {
        values.insert("audio.sample_rate".to_string(), 32_000_i64.into());
    }
    values
}
