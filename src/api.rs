use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use serde_json::json;
use std::{convert::Infallible, net::SocketAddr, sync::Arc};
use tokio::net::TcpListener;
use tokio_stream::{StreamExt, once};

use crate::{
    backend::{
        GenerateRequest, GenerateResponse, GenerateStreamEvent, GenerationBackend,
        StreamGranularity,
    },
    model_store::{ModelManifest, ModelStore, unix_ts},
    openai::{
        AssistantMessage, ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse,
        ErrorObject, ErrorResponse, ModelListResponse, ModelObject, Usage,
        image_urls_from_messages, messages_to_prompt_for_model,
    },
};

#[derive(Clone)]
pub struct ApiState {
    store: Arc<ModelStore>,
    backend: Arc<dyn GenerationBackend>,
    default_model: Option<String>,
}

impl ApiState {
    pub fn new(store: ModelStore, backend: Arc<dyn GenerationBackend>) -> Self {
        Self::new_with_default_model(store, backend, None)
    }

    pub fn new_with_default_model(
        store: ModelStore,
        backend: Arc<dyn GenerationBackend>,
        default_model: Option<String>,
    ) -> Self {
        Self {
            store: Arc::new(store),
            backend,
            default_model,
        }
    }
}

pub fn router(state: ApiState) -> Router {
    Router::new()
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .with_state(state)
}

pub async fn serve(addr: SocketAddr, state: ApiState) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    println!("Server running at http://{addr}");
    axum::serve(listener, router(state)).await?;
    Ok(())
}

async fn models_handler(State(state): State<ApiState>) -> Response {
    match state.store.list() {
        Ok(manifests) => {
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

async fn chat_completions_handler(
    State(state): State<ApiState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
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
            return api_error(
                StatusCode::NOT_FOUND,
                err.to_string(),
                Some("model".to_string()),
            );
        }
    };

    let prompt = messages_to_prompt_for_model(&manifest, &request.messages);
    let image_urls = image_urls_from_messages(&request.messages);
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
        verbose: false,
        debug: false,
    };

    if request.stream.unwrap_or(false) {
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
    let model = manifest.id.clone();
    let result = tokio::task::spawn_blocking(move || backend.generate(&manifest, generate_request))
        .await
        .map_err(|err| anyhow::anyhow!("generation task failed: {err}"))
        .and_then(|inner| inner);

    match result {
        Ok(response) => Json(to_chat_completion(model, response)).into_response(),
        Err(err) => api_error(StatusCode::BAD_REQUEST, err.to_string(), None),
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
    let body = state
        .backend
        .generate_stream(manifest, generate_request)
        .map(move |event| {
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
                Ok(GenerateStreamEvent::Done { finish_reason, .. }) => json!({
                    "id": body_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body_model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason
                    }]
                }),
                Err(message) => json!({
                    "error": {
                        "message": message,
                        "type": "invalid_request_error",
                        "param": null,
                        "code": null
                    }
                }),
            };
            Ok::<Event, Infallible>(Event::default().data(data.to_string()))
        });

    let done = once(Ok::<Event, Infallible>(Event::default().data("[DONE]")));
    let stream = role.chain(body).chain(done);

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
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

fn api_error(status: StatusCode, message: String, param: Option<String>) -> Response {
    (
        status,
        Json(ErrorResponse {
            error: ErrorObject {
                message,
                kind: "invalid_request_error".to_string(),
                param,
                code: None,
            },
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::{GenerateStream, GenerateStreamEvent, GenerationTimings},
        model_store::{ModelFormat, ModelSource},
    };
    use axum::{body, body::Body, http::Request};
    use std::fs;
    use tower::ServiceExt;

    #[derive(Clone)]
    struct MockBackend;

    impl GenerationBackend for MockBackend {
        fn generate(
            &self,
            _manifest: &ModelManifest,
            _request: GenerateRequest,
        ) -> anyhow::Result<GenerateResponse> {
            Ok(GenerateResponse {
                text: "hello".to_string(),
                prompt_tokens: 2,
                completion_tokens: 1,
                finish_reason: "stop".to_string(),
                timings: GenerationTimings {
                    load_seconds: 0.0,
                    warmup_seconds: 0.0,
                    first_token_seconds: 0.0,
                    prompt_seconds: 0.01,
                    decode_seconds: 0.01,
                    total_seconds: 0.02,
                },
                backend_diagnostics: Vec::new(),
            })
        }

        fn generate_stream(
            &self,
            _manifest: ModelManifest,
            _request: GenerateRequest,
        ) -> GenerateStream {
            let events = vec![
                Ok(GenerateStreamEvent::TextChunk("hello".to_string())),
                Ok(GenerateStreamEvent::Done {
                    finish_reason: "stop".to_string(),
                    prompt_tokens: 2,
                    completion_tokens: 1,
                    timings: GenerationTimings {
                        load_seconds: 0.0,
                        warmup_seconds: 0.0,
                        first_token_seconds: 0.0,
                        prompt_seconds: 0.01,
                        decode_seconds: 0.01,
                        total_seconds: 0.02,
                    },
                    backend_diagnostics: Vec::new(),
                }),
            ];
            Box::pin(tokio_stream::iter(events))
        }
    }

    #[tokio::test]
    async fn models_and_chat_routes_use_openai_shapes() {
        let store = test_store();
        let manifest = ModelManifest {
            id: "mock".to_string(),
            source: ModelSource::LocalPath {
                path: "test".to_string(),
            },
            format: ModelFormat::Unknown,
            architecture: None,
            tokenizer_path: None,
            config_path: None,
            model_path: None,
            backend: "mock".to_string(),
            created_unix: 1,
            files: Vec::new(),
            artifacts: Vec::new(),
        };
        fs::create_dir_all(store.model_dir("mock")).unwrap();
        fs::write(
            store
                .model_dir("mock")
                .join(crate::model_store::MANIFEST_FILE),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let app = router(ApiState::new(store, Arc::new(MockBackend)));

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"mock","messages":[{"role":"user","content":"hi"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["object"], "chat.completion");
        assert_eq!(value["choices"][0]["message"]["role"], "assistant");

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"mock","stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let stream = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(stream.contains("\"object\":\"chat.completion.chunk\""));
        assert!(stream.contains("\"content\":\"hello\""));
        assert!(stream.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn server_default_model_is_used_when_request_omits_model() {
        let store = test_store();
        let manifest = ModelManifest {
            id: "mock".to_string(),
            source: ModelSource::LocalPath {
                path: "test".to_string(),
            },
            format: ModelFormat::Unknown,
            architecture: None,
            tokenizer_path: None,
            config_path: None,
            model_path: None,
            backend: "mock".to_string(),
            created_unix: 1,
            files: Vec::new(),
            artifacts: Vec::new(),
        };
        fs::create_dir_all(store.model_dir("mock")).unwrap();
        fs::write(
            store
                .model_dir("mock")
                .join(crate::model_store::MANIFEST_FILE),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let app = router(ApiState::new_with_default_model(
            store,
            Arc::new(MockBackend),
            Some("mock".to_string()),
        ));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"messages":[{"role":"user","content":"hi"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let bytes = body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["model"], "mock");
    }

    fn test_store() -> ModelStore {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("werk1112-api-test-{}-{nanos}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        ModelStore::resolve(Some(dir)).unwrap()
    }
}
