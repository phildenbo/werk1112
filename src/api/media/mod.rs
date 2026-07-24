pub(super) mod handlers;
pub(super) mod input;
pub(super) mod output;
pub(super) mod requests;

pub(in crate::api) use handlers::{
    audio_generations_handler, audio_speech_handler, audio_transcriptions_handler,
    cancel_job_handler, capabilities_handler, create_job_handler, get_job_handler,
    image_edits_handler, image_generations_handler, output_handler, parameters_handler,
    video_generations_handler,
};
