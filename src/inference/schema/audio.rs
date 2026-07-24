use super::super::types::{ParameterDescriptor, ParameterType};
use super::builders::{
    add_named_descriptors, bool_descriptor, enum_descriptor, integer_descriptor, number_descriptor,
};
use crate::capabilities::InferenceTask;

pub(super) fn audio_descriptors(task: InferenceTask) -> Vec<ParameterDescriptor> {
    let mut values = vec![
        number_descriptor(
            "audio.duration",
            "--duration",
            "Duration",
            "Requested duration in seconds",
            "composition",
            30.0,
            0.1,
            86_400.0,
            0.1,
        ),
        integer_descriptor(
            "audio.variations",
            "--num-variations",
            "Variations",
            "Number of generated variations",
            "generation",
            1,
            1,
            1024,
        ),
        integer_descriptor(
            "audio.seed",
            "--seed",
            "Seed",
            "Random seed",
            "sampling",
            0,
            0,
            i64::MAX,
        ),
        integer_descriptor(
            "audio.sample_rate",
            "--sample-rate",
            "Sample rate",
            "Output sample rate",
            "output",
            44_100,
            8_000,
            384_000,
        ),
        integer_descriptor(
            "audio.bit_depth",
            "--bit-depth",
            "Bit depth",
            "PCM output bit depth",
            "output",
            16,
            8,
            64,
        ),
        integer_descriptor(
            "audio.channels",
            "--channels",
            "Channels",
            "Output channel count",
            "output",
            2,
            1,
            32,
        ),
        bool_descriptor(
            "audio.instrumental",
            "--instrumental",
            "Instrumental",
            "Generate without vocals",
            "vocals",
            false,
        ),
        enum_descriptor(
            "audio.output_format",
            "--format",
            "Output format",
            "Audio container",
            "output",
            "wav",
            &["wav", "flac", "ogg"],
        ),
    ];
    add_named_descriptors(
        &mut values,
        "audio",
        "prompt",
        &[
            ("title", ParameterType::String),
            ("lyrics", ParameterType::String),
            ("generate_lyrics", ParameterType::Boolean),
            ("lyrics_language", ParameterType::String),
        ],
    );
    add_named_descriptors(
        &mut values,
        "audio",
        "style",
        &[
            ("genres", ParameterType::List),
            ("subgenres", ParameterType::List),
            ("styles", ParameterType::List),
            ("eras", ParameterType::List),
            ("influences", ParameterType::List),
            ("moods", ParameterType::List),
            ("themes", ParameterType::List),
            ("descriptors", ParameterType::List),
        ],
    );
    add_named_descriptors(
        &mut values,
        "audio",
        "harmony",
        &[
            ("bpm", ParameterType::Number),
            ("tempo_min", ParameterType::Number),
            ("tempo_max", ParameterType::Number),
            ("tempo_mode", ParameterType::Enumeration),
            ("time_signature", ParameterType::String),
            ("key", ParameterType::String),
            ("scale", ParameterType::String),
            ("tuning", ParameterType::String),
            ("chord_progression", ParameterType::String),
            ("chord_complexity", ParameterType::Number),
            ("harmonic_tension", ParameterType::Number),
            ("song_structure", ParameterType::String),
            ("arrangement_prompt", ParameterType::String),
        ],
    );
    add_named_descriptors(
        &mut values,
        "audio",
        "instruments",
        &[
            ("instruments", ParameterType::List),
            ("excluded_instruments", ParameterType::List),
            ("lead_instrument", ParameterType::String),
            ("rhythm_instruments", ParameterType::List),
            ("bass_instrument", ParameterType::String),
            ("acoustic_electronic_balance", ParameterType::Number),
            ("instrument_density", ParameterType::Number),
        ],
    );
    add_named_descriptors(
        &mut values,
        "audio",
        "vocals",
        &[
            ("vocals", ParameterType::Boolean),
            ("voice_reference", ParameterType::Path),
            ("speaker_id", ParameterType::String),
            ("vocal_presentation", ParameterType::String),
            ("register", ParameterType::String),
            ("range", ParameterType::String),
            ("language", ParameterType::String),
            ("accent", ParameterType::String),
            ("vocal_style", ParameterType::String),
            ("delivery", ParameterType::String),
            ("emotion", ParameterType::String),
            ("breathiness", ParameterType::Number),
            ("rasp", ParameterType::Number),
            ("vibrato", ParameterType::Number),
            ("power", ParameterType::Number),
            ("intimacy", ParameterType::Number),
            ("articulation", ParameterType::Number),
            ("pronunciation_strength", ParameterType::Number),
            ("vocal_presence", ParameterType::Number),
            ("harmony_amount", ParameterType::Number),
            ("choir_amount", ParameterType::Number),
            ("duet", ParameterType::Boolean),
            ("backing_vocals", ParameterType::Boolean),
        ],
    );
    add_named_descriptors(
        &mut values,
        "audio",
        "conditioning",
        &[
            ("prompt_adherence", ParameterType::Number),
            ("lyrics_adherence", ParameterType::Number),
            ("style_adherence", ParameterType::Number),
            ("creativity", ParameterType::Number),
            ("variation_strength", ParameterType::Number),
            ("originality", ParameterType::Number),
            ("source_audio", ParameterType::Path),
            ("reference_audio", ParameterType::Path),
            ("instrumental_audio", ParameterType::Path),
            ("vocal_audio", ParameterType::Path),
            ("melody_audio", ParameterType::Path),
            ("rhythm_audio", ParameterType::Path),
            ("chord_audio", ParameterType::Path),
            ("audio_strength", ParameterType::Number),
            ("melody_adherence", ParameterType::Number),
            ("rhythm_adherence", ParameterType::Number),
            ("harmony_adherence", ParameterType::Number),
            ("continuation_start", ParameterType::Number),
            ("continuation_duration", ParameterType::Number),
        ],
    );
    add_named_descriptors(
        &mut values,
        "audio",
        "sampling",
        &[
            ("guidance", ParameterType::Number),
            ("steps", ParameterType::Integer),
            ("temperature", ParameterType::Number),
            ("top_k", ParameterType::Integer),
            ("top_p", ParameterType::Number),
        ],
    );
    add_named_descriptors(
        &mut values,
        "audio",
        "mixing",
        &[
            ("mix_controls", ParameterType::List),
            ("mastering_controls", ParameterType::List),
            ("codec", ParameterType::String),
            ("bitrate", ParameterType::String),
            ("normalization", ParameterType::Boolean),
            ("loudness_lufs", ParameterType::Number),
            ("export_stems", ParameterType::Boolean),
            ("stems", ParameterType::List),
            ("num_stems", ParameterType::Integer),
            ("segment_duration", ParameterType::Number),
            ("segment_overlap", ParameterType::Number),
            ("output_path", ParameterType::Path),
        ],
    );
    if task == InferenceTask::StemSeparation {
        values.retain(|descriptor| {
            matches!(
                descriptor.path.as_str(),
                "audio.stems"
                    | "audio.num_stems"
                    | "audio.segment_duration"
                    | "audio.segment_overlap"
                    | "audio.sample_rate"
                    | "audio.channels"
                    | "audio.output_format"
                    | "audio.normalization"
                    | "audio.output_path"
            )
        });
    } else {
        values.retain(|descriptor| {
            !matches!(
                descriptor.path.as_str(),
                "audio.num_stems" | "audio.segment_duration" | "audio.segment_overlap"
            )
        });
    }
    values
}

pub(super) fn tts_descriptors() -> Vec<ParameterDescriptor> {
    let mut values = vec![
        number_descriptor(
            "tts.speed",
            "--speed",
            "Speed",
            "Speech speed multiplier",
            "voice",
            1.0,
            0.1,
            10.0,
            0.01,
        ),
        number_descriptor(
            "tts.pitch",
            "--pitch",
            "Pitch",
            "Pitch shift",
            "voice",
            0.0,
            -48.0,
            48.0,
            0.1,
        ),
        integer_descriptor(
            "tts.seed",
            "--seed",
            "Seed",
            "Random seed",
            "sampling",
            0,
            0,
            i64::MAX,
        ),
        integer_descriptor(
            "tts.sample_rate",
            "--sample-rate",
            "Sample rate",
            "Output sample rate",
            "output",
            24_000,
            8_000,
            384_000,
        ),
        integer_descriptor(
            "tts.channels",
            "--channels",
            "Channels",
            "Output channels",
            "output",
            1,
            1,
            32,
        ),
        enum_descriptor(
            "tts.output_format",
            "--format",
            "Output format",
            "Audio output format",
            "output",
            "wav",
            &["wav", "flac", "ogg"],
        ),
        bool_descriptor(
            "tts.streaming",
            "--streaming",
            "Streaming",
            "Stream audio chunks when supported",
            "output",
            false,
        ),
    ];
    add_named_descriptors(
        &mut values,
        "tts",
        "voice",
        &[
            ("voice", ParameterType::String),
            ("voice_reference", ParameterType::Path),
            ("language", ParameterType::String),
            ("accent", ParameterType::String),
            ("dialect", ParameterType::String),
            ("volume", ParameterType::Number),
            ("emotion", ParameterType::String),
            ("emotion_strength", ParameterType::Number),
            ("speaking_style", ParameterType::String),
            ("stability", ParameterType::Number),
            ("similarity", ParameterType::Number),
            ("expressiveness", ParameterType::Number),
            ("pronunciation_dictionaries", ParameterType::List),
            ("phoneme_input", ParameterType::Boolean),
            ("pause_scale", ParameterType::Number),
            ("sentence_silence", ParameterType::Number),
        ],
    );
    add_named_descriptors(
        &mut values,
        "tts",
        "output",
        &[
            ("loudness", ParameterType::Number),
            ("chunk_size", ParameterType::Integer),
            ("output_path", ParameterType::Path),
        ],
    );
    values
}

pub(super) fn stt_descriptors() -> Vec<ParameterDescriptor> {
    let mut values = vec![
        enum_descriptor(
            "stt.operation",
            "--task",
            "Operation",
            "Transcribe or translate",
            "transcription",
            "transcribe",
            &["transcribe", "translate"],
        ),
        integer_descriptor(
            "stt.beam_size",
            "--beam-size",
            "Beam size",
            "Beam search width",
            "decoding",
            5,
            1,
            1000,
        ),
        integer_descriptor(
            "stt.best_of",
            "--best-of",
            "Best of",
            "Candidates sampled",
            "decoding",
            5,
            1,
            1000,
        ),
        number_descriptor(
            "stt.temperature",
            "--temperature",
            "Temperature",
            "Decoding temperature",
            "decoding",
            0.0,
            0.0,
            10.0,
            0.01,
        ),
        bool_descriptor(
            "stt.segment_timestamps",
            "--segment-timestamps",
            "Segment timestamps",
            "Return segment timestamps",
            "output",
            true,
        ),
        bool_descriptor(
            "stt.word_timestamps",
            "--word-timestamps",
            "Word timestamps",
            "Return word timestamps",
            "output",
            false,
        ),
        bool_descriptor(
            "stt.diarization",
            "--diarization",
            "Diarization",
            "Identify speakers",
            "transcription",
            false,
        ),
        enum_descriptor(
            "stt.output_format",
            "--output-format",
            "Output format",
            "Transcription output format",
            "output",
            "json",
            &["json", "text", "srt", "vtt", "tsv"],
        ),
    ];
    add_named_descriptors(
        &mut values,
        "stt",
        "transcription",
        &[
            ("input_audio", ParameterType::Path),
            ("language", ParameterType::String),
            ("min_speakers", ParameterType::Integer),
            ("max_speakers", ParameterType::Integer),
            ("initial_prompt", ParameterType::String),
            ("hotwords", ParameterType::List),
            ("vad", ParameterType::Boolean),
            ("suppress_tokens", ParameterType::List),
            ("condition_on_previous_text", ParameterType::Boolean),
            ("hallucination_silence_threshold", ParameterType::Number),
            ("temperature_fallbacks", ParameterType::List),
            ("output_path", ParameterType::Path),
        ],
    );
    values
}
