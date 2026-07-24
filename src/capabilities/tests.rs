use super::*;

#[test]
fn task_and_layout_names_support_cli_and_manifest_forms() {
    assert_eq!(
        "image-generation".parse::<InferenceTask>().unwrap(),
        InferenceTask::ImageGeneration
    );
    assert_eq!(
        "image_generation".parse::<InferenceTask>().unwrap(),
        InferenceTask::ImageGeneration
    );
    assert_eq!(
        "single_file".parse::<RepositoryLayout>().unwrap(),
        RepositoryLayout::SingleFile
    );
    assert_eq!(
        serde_json::to_string(&InferenceTask::SpeechToText).unwrap(),
        "\"speech_to_text\""
    );
    assert_eq!(
        serde_json::to_string(&ModelComponentKind::TextEncoder2).unwrap(),
        "\"text_encoder_2\""
    );
    assert_eq!(
        serde_json::to_string(&RepositoryLayout::TensorRtEngine).unwrap(),
        "\"tensorrt_engine\""
    );
    assert_eq!(
        InferenceTask::ImageToVideo.required_input_modalities(),
        &[InputModality::Image]
    );
    assert_eq!(
        InferenceTask::SpeechToText.output_modality(),
        OutputModality::Text
    );
    assert_eq!(InferenceTask::TextToSpeech.parameter_namespace(), "tts");
    assert_eq!(InferenceTask::SpeechToText.parameter_namespace(), "stt");
    assert!(!InferenceTask::ImageUpscaling.requires_prompt());
}
