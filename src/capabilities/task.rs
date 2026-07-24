use super::{InputModality, OutputModality};

string_enum! {
    /// A normalized inference operation supported by a model and runtime.
    pub enum InferenceTask {
        TextGeneration => "text-generation",
        TextEmbedding => "text-embedding",
        ImageUnderstanding => "image-understanding",
        ImageGeneration => "image-generation",
        ImageEditing => "image-editing",
        ImageVariation => "image-variation",
        ImageInpainting => "image-inpainting",
        ImageOutpainting => "image-outpainting",
        ImageUpscaling => "image-upscaling",
        VideoGeneration => "video-generation",
        ImageToVideo => "image-to-video",
        VideoToVideo => "video-to-video",
        VideoInpainting => "video-inpainting",
        VideoExtension => "video-extension",
        VideoUpscaling => "video-upscaling",
        FrameInterpolation => "frame-interpolation",
        AudioGeneration => "audio-generation",
        MusicGeneration => "music-generation",
        SongContinuation => "song-continuation",
        SongVariation => "song-variation",
        TextToSpeech => "text-to-speech",
        SpeechToText => "speech-to-text",
        VoiceConversion => "voice-conversion",
        StemGeneration => "stem-generation",
        StemSeparation => "stem-separation",
        AudioEnhancement => "audio-enhancement"
    }
}

impl InferenceTask {
    pub const fn output_modality(self) -> OutputModality {
        match self {
            Self::TextGeneration | Self::ImageUnderstanding | Self::SpeechToText => {
                OutputModality::Text
            }
            Self::TextEmbedding => OutputModality::Embedding,
            Self::ImageGeneration
            | Self::ImageEditing
            | Self::ImageVariation
            | Self::ImageInpainting
            | Self::ImageOutpainting
            | Self::ImageUpscaling => OutputModality::Image,
            Self::VideoGeneration
            | Self::ImageToVideo
            | Self::VideoToVideo
            | Self::VideoInpainting
            | Self::VideoExtension
            | Self::VideoUpscaling
            | Self::FrameInterpolation => OutputModality::Video,
            Self::AudioGeneration
            | Self::MusicGeneration
            | Self::SongContinuation
            | Self::SongVariation
            | Self::TextToSpeech
            | Self::VoiceConversion
            | Self::StemGeneration
            | Self::StemSeparation
            | Self::AudioEnhancement => OutputModality::Audio,
        }
    }

    pub const fn required_input_modalities(self) -> &'static [InputModality] {
        match self {
            Self::TextGeneration
            | Self::TextEmbedding
            | Self::ImageGeneration
            | Self::VideoGeneration
            | Self::AudioGeneration
            | Self::MusicGeneration
            | Self::TextToSpeech => &[InputModality::Text],
            Self::ImageUnderstanding
            | Self::ImageEditing
            | Self::ImageVariation
            | Self::ImageInpainting
            | Self::ImageOutpainting
            | Self::ImageUpscaling
            | Self::ImageToVideo => &[InputModality::Image],
            Self::VideoToVideo
            | Self::VideoInpainting
            | Self::VideoExtension
            | Self::VideoUpscaling
            | Self::FrameInterpolation => &[InputModality::Video],
            Self::SongContinuation
            | Self::SongVariation
            | Self::SpeechToText
            | Self::VoiceConversion
            | Self::StemGeneration
            | Self::StemSeparation
            | Self::AudioEnhancement => &[InputModality::Audio],
        }
    }

    pub const fn requires_prompt(self) -> bool {
        matches!(
            self,
            Self::TextGeneration
                | Self::TextEmbedding
                | Self::ImageUnderstanding
                | Self::ImageGeneration
                | Self::ImageEditing
                | Self::ImageInpainting
                | Self::ImageOutpainting
                | Self::VideoGeneration
                | Self::ImageToVideo
                | Self::VideoToVideo
                | Self::VideoInpainting
                | Self::VideoExtension
                | Self::AudioGeneration
                | Self::MusicGeneration
                | Self::TextToSpeech
        )
    }

    pub const fn parameter_namespace(self) -> &'static str {
        match self {
            Self::TextGeneration | Self::TextEmbedding | Self::ImageUnderstanding => "text",
            Self::ImageGeneration
            | Self::ImageEditing
            | Self::ImageVariation
            | Self::ImageInpainting
            | Self::ImageOutpainting
            | Self::ImageUpscaling => "image",
            Self::VideoGeneration
            | Self::ImageToVideo
            | Self::VideoToVideo
            | Self::VideoInpainting
            | Self::VideoExtension
            | Self::VideoUpscaling
            | Self::FrameInterpolation => "video",
            Self::AudioGeneration
            | Self::MusicGeneration
            | Self::SongContinuation
            | Self::SongVariation
            | Self::VoiceConversion
            | Self::StemGeneration
            | Self::StemSeparation
            | Self::AudioEnhancement => "audio",
            Self::TextToSpeech => "tts",
            Self::SpeechToText => "stt",
        }
    }
}
