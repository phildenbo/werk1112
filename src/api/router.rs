use axum::{
    Router,
    routing::{get, post},
};
use std::net::SocketAddr;
use tokio::net::TcpListener;

use super::{
    chat::{chat_completions_handler, models_handler},
    media::{
        audio_generations_handler, audio_speech_handler, audio_transcriptions_handler,
        cancel_job_handler, capabilities_handler, create_job_handler, get_job_handler,
        image_edits_handler, image_generations_handler, output_handler, parameters_handler,
        video_generations_handler,
    },
    state::ApiState,
};

pub fn router(state: ApiState) -> Router {
    Router::new()
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/images/generations", post(image_generations_handler))
        .route("/v1/images/edits", post(image_edits_handler))
        .route("/v1/videos/generations", post(video_generations_handler))
        .route("/v1/audio/generations", post(audio_generations_handler))
        .route("/v1/audio/speech", post(audio_speech_handler))
        .route(
            "/v1/audio/transcriptions",
            post(audio_transcriptions_handler),
        )
        .route("/v1/capabilities", get(capabilities_handler))
        .route("/v1/parameters", get(parameters_handler))
        .route("/v1/outputs/{id}", get(output_handler))
        .route("/v1/jobs", post(create_job_handler))
        .route(
            "/v1/jobs/{id}",
            get(get_job_handler).delete(cancel_job_handler),
        )
        .with_state(state)
}

pub async fn serve(addr: SocketAddr, state: ApiState) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    println!("Server running at http://{addr}");
    if state.api_key_auth_enabled() {
        println!("API key auth enabled; clients must send Authorization: Bearer <key>");
    }
    axum::serve(listener, router(state)).await?;
    Ok(())
}
