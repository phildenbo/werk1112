use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use serde_json::json;
use std::convert::Infallible;
use tokio_stream::{StreamExt, once};

use crate::{
    backend::{GenerateRequest, GenerateResponse, GenerateStreamEvent, StreamGranularity},
    model_store::{ModelManifest, unix_ts},
    openai::{
        AssistantMessage, ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse,
        ModelListResponse, ModelObject, Usage, image_urls_from_messages,
        messages_to_prompt_for_model_with_template,
    },
};

use super::{response::api_error, state::ApiState};

pub(super) async fn models_handler(State(state): State<ApiState>, headers: HeaderMap) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    match state.store.list() {
        Ok(manifests) => {
            state.log_verbose(format!(
                "[werk serve] GET /v1/models -> {} model(s)",
                manifests.len()
            ));
            let data = manifests
                .into_iter()
                .map(|manifest| ModelObject {
                    id: manifest.id,
                    object: "model",
                    created: manifest.created_unix,
                    owned_by: "local",
                })
                .collect();
            Json(ModelListResponse {
                object: "list",
                data,
            })
            .into_response()
        }
        Err(err) => api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string(), None),
    }
}

pub(super) async fn chat_completions_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    let model_id = match request.model.as_deref().or(state.default_model.as_deref()) {
        Some(model) => model,
        None => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "request must include model, or start the server with --model <id>".to_string(),
                Some("model".to_string()),
            );
        }
    };

    let manifest = match state.store.get(model_id) {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("[werk serve] POST /v1/chat/completions model={model_id} -> 404");
            return api_error(
                StatusCode::NOT_FOUND,
                err.to_string(),
                Some("model".to_string()),
            );
        }
    };

    let image_urls = image_urls_from_messages(&request.messages);
    let stream = request.stream.unwrap_or(false);
    state.log_verbose(format!(
        "[werk serve] POST /v1/chat/completions model={} stream={} messages={} images={} max_tokens={}",
        manifest.id,
        yes_no(stream),
        request.messages.len(),
        image_urls.len(),
        request.max_completion_tokens()
    ));
    let prompt_options = match state.prompt_options(&manifest, !image_urls.is_empty()) {
        Ok(options) => options,
        Err(err) => {
            eprintln!(
                "[werk serve] POST /v1/chat/completions model={} -> routing error: {err}",
                manifest.id
            );
            return api_error(StatusCode::BAD_REQUEST, err.to_string(), None);
        }
    };
    let prompt =
        messages_to_prompt_for_model_with_template(&manifest, &request.messages, prompt_options);
    let max_tokens = request.max_completion_tokens();
    let mut stop = prompt.stop;
    stop.extend(request.stop_strings());

    let generate_request = GenerateRequest {
        prompt: prompt.prompt,
        messages: request.messages,
        image_urls,
        max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        stop,
        seed: request.seed,
        stream_granularity: StreamGranularity::Chunk,
        verbose: state.verbose,
        debug: false,
    };

    if stream {
        stream_chat_response(state, manifest, generate_request)
    } else {
        complete_chat_response(state, manifest, generate_request).await
    }
}

async fn complete_chat_response(
    state: ApiState,
    manifest: ModelManifest,
    generate_request: GenerateRequest,
) -> Response {
    let backend = state.backend.clone();
    let verbose = state.verbose;
    let model = manifest.id.clone();
    let chat_session = match state.chat_session(&manifest, generate_request.seed) {
        Ok(session) => session,
        Err(err) => {
            eprintln!("[werk serve] complete model={model} -> session error: {err}");
            return api_error(StatusCode::BAD_REQUEST, err.to_string(), None);
        }
    };
    let result = tokio::task::spawn_blocking(move || {
        if let Some(session) = chat_session.as_ref() {
            session.generate(generate_request)
        } else {
            backend.generate(&manifest, generate_request)
        }
    })
    .await
    .map_err(|err| anyhow::anyhow!("generation task failed: {err}"))
    .and_then(|inner| inner);

    match result {
        Ok(response) => {
            if verbose {
                eprintln!(
                    "[werk serve] complete model={} finish={} prompt_tokens={} completion_tokens={} total={} load={} eval_rate={}",
                    model,
                    response.finish_reason,
                    response.prompt_tokens,
                    response.completion_tokens,
                    format_duration(response.timings.total_seconds),
                    format_duration(response.timings.load_seconds),
                    format_token_rate(response.completion_tokens, response.timings.decode_seconds)
                );
                log_backend_diagnostics(&response.backend_diagnostics);
            }
            Json(to_chat_completion(model, response)).into_response()
        }
        Err(err) => {
            eprintln!("[werk serve] complete model={model} -> error: {err}");
            api_error(StatusCode::BAD_REQUEST, err.to_string(), None)
        }
    }
}

fn stream_chat_response(
    state: ApiState,
    manifest: ModelManifest,
    generate_request: GenerateRequest,
) -> Response {
    let model = manifest.id.clone();
    let created = unix_ts();
    let id = format!("chatcmpl-{created}");

    let role_id = id.clone();
    let role_model = model.clone();
    let role = once(Ok::<Event, Infallible>(
        Event::default().data(
            json!({
                "id": role_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": role_model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": null
                }]
            })
            .to_string(),
        ),
    ));

    let body_id = id.clone();
    let body_model = model.clone();
    let body_model_for_log = model.clone();
    let verbose = state.verbose;
    let body_stream = match state.chat_session(&manifest, generate_request.seed) {
        Ok(Some(session)) => session.generate_stream(generate_request),
        Ok(None) => state.backend.generate_stream(manifest, generate_request),
        Err(err) => Box::pin(tokio_stream::iter(vec![Err(err.to_string())])),
    };
    let body = body_stream.map(move |event| {
            let data = match event {
                Ok(GenerateStreamEvent::TextChunk(text)) => json!({
                    "id": body_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body_model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": null
                    }]
                }),
                Ok(GenerateStreamEvent::Done {
                    finish_reason,
                    prompt_tokens,
                    completion_tokens,
                    timings,
                    backend_diagnostics,
                }) => {
                    if verbose {
                        eprintln!(
                            "[werk serve] stream model={} finish={} prompt_tokens={} completion_tokens={} total={} load={} eval_rate={}",
                            body_model_for_log,
                            finish_reason,
                            prompt_tokens,
                            completion_tokens,
                            format_duration(timings.total_seconds),
                            format_duration(timings.load_seconds),
                            format_token_rate(completion_tokens, timings.decode_seconds)
                        );
                        log_backend_diagnostics(&backend_diagnostics);
                    }
                    json!({
                        "id": body_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": body_model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason
                        }]
                    })
                }
                Err(message) => {
                    eprintln!("[werk serve] stream model={} -> error: {message}", body_model_for_log);
                    json!({
                        "error": {
                            "message": message,
                            "type": "invalid_request_error",
                            "param": null,
                            "code": null
                        }
                    })
                }
            };
            Ok::<Event, Infallible>(Event::default().data(data.to_string()))
        });

    let done = once(Ok::<Event, Infallible>(Event::default().data("[DONE]")));
    let stream = role.chain(body).chain(done);

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn format_duration(seconds: f64) -> String {
    let seconds = seconds.max(0.0);
    if seconds >= 1.0 {
        trim_float(format!("{seconds:.6}")) + "s"
    } else if seconds >= 0.001 {
        trim_float(format!("{:.4}", seconds * 1000.0)) + "ms"
    } else {
        trim_float(format!("{:.3}", seconds * 1_000_000.0)) + "us"
    }
}

fn format_token_rate(tokens: usize, seconds: f64) -> String {
    if seconds <= 0.0 {
        return "-".to_string();
    }
    format!("{:.2} tok/s", tokens as f64 / seconds)
}

fn trim_float(mut value: String) -> String {
    while value.contains('.') && value.ends_with('0') {
        value.pop();
    }
    if value.ends_with('.') {
        value.pop();
    }
    value
}

fn log_backend_diagnostics(diagnostics: &[String]) {
    for diagnostic in diagnostics {
        eprintln!("[werk serve]   {diagnostic}");
    }
}

fn to_chat_completion(model: String, response: GenerateResponse) -> ChatCompletionResponse {
    let created = unix_ts();
    ChatCompletionResponse {
        id: format!("chatcmpl-{created}"),
        object: "chat.completion",
        created,
        model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content: response.text,
            },
            finish_reason: response.finish_reason,
        }],
        usage: Usage {
            prompt_tokens: response.prompt_tokens,
            completion_tokens: response.completion_tokens,
            total_tokens: response.prompt_tokens + response.completion_tokens,
        },
    }
}
