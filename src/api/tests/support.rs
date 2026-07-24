pub(super) use super::super::{
    ApiState, PromptOptionsResolver,
    media::{
        input::{ApiMediaInput, ApiMediaInputObject},
        output::encode_base64,
        requests::{DirectResponseFormat, ImageEditApiRequest, ImageGenerationApiRequest},
    },
    router,
};
pub(super) use crate::{
    backend::{GenerateRequest, GenerateResponse, GenerationBackend},
    capabilities::{InferenceTask, InputModality},
    inference::{InferenceInputSource, OverrideBool, ParameterValue},
    inference_service::InferenceService,
    model_store::{ModelManifest, ModelStore},
    openai::ChatTemplateOptions,
};
pub(super) use axum::{Router, http::StatusCode, response::Response};
pub(super) use serde_json::{Value, json};
pub(super) use std::{collections::BTreeMap, sync::Arc};

pub(super) use crate::{
    backend::{GenerateStream, GenerateStreamEvent, GenerationTimings},
    capabilities::{OutputModality, RepositoryLayout},
    inference::{
        EffectiveInferenceRequest, InferenceRuntimeCandidate, ParameterSupportStatus,
        RuntimeAccelerator,
    },
    inference_service::{BackendExecution, BackendOutput, BackendProbe, MediaInferenceBackend},
    model_store::{ModelFormat, ModelMetadata, ModelSource},
    openai::ChatTemplateSource,
};
pub(super) use axum::{
    body,
    body::Body,
    http::{Request, header},
};
pub(super) use std::{fs, path::Path};
pub(super) use tower::ServiceExt;

#[derive(Clone)]
pub(super) struct MockBackend;

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

#[derive(Clone)]
pub(super) struct PromptEchoBackend;

impl GenerationBackend for PromptEchoBackend {
    fn generate(
        &self,
        _manifest: &ModelManifest,
        request: GenerateRequest,
    ) -> anyhow::Result<GenerateResponse> {
        Ok(GenerateResponse {
            text: request.prompt,
            prompt_tokens: 1,
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
        request: GenerateRequest,
    ) -> GenerateStream {
        let events = vec![
            Ok(GenerateStreamEvent::TextChunk(request.prompt)),
            Ok(GenerateStreamEvent::Done {
                finish_reason: "stop".to_string(),
                prompt_tokens: 1,
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

#[derive(Clone)]
pub(super) struct MockMediaBackend;

impl MediaInferenceBackend for MockMediaBackend {
    fn probe(
        &self,
        _store: &ModelStore,
        _manifest: &ModelManifest,
        task: InferenceTask,
        schema_paths: &[String],
    ) -> BackendProbe {
        let parameter_support = schema_paths
            .iter()
            .cloned()
            .map(|path| (path, ParameterSupportStatus::Native))
            .collect::<BTreeMap<_, _>>();
        BackendProbe {
            available: true,
            detail: "mock media backend".to_string(),
            candidates: vec![InferenceRuntimeCandidate {
                id: "mock-media-cpu".to_string(),
                backend: "mock-media".to_string(),
                accelerator: RuntimeAccelerator::Cpu,
                available: true,
                availability_reason: None,
                supported_tasks: vec![task],
                supported_layouts: vec![RepositoryLayout::Custom],
                supported_formats: Vec::new(),
                supported_families: Vec::new(),
                supported_architectures: Vec::new(),
                parameter_support: parameter_support.clone(),
                supports_offloading: true,
                supports_compile: true,
                supports_batching: true,
                priority: 1_000,
            }],
            parameter_support,
        }
    }

    #[allow(clippy::type_complexity)]
    fn execute(
        &self,
        _store: &ModelStore,
        _manifest: &ModelManifest,
        request: &EffectiveInferenceRequest,
        output_dir: &Path,
        runtime: &str,
    ) -> anyhow::Result<BackendExecution> {
        let (name, mime_type, contents, width, height, duration): (
            &str,
            &str,
            &[u8],
            Option<u32>,
            Option<u32>,
            Option<f64>,
        ) = match request.task {
            InferenceTask::SpeechToText => (
                "transcript.txt",
                "text/plain",
                b"mock transcript",
                None,
                None,
                Some(1.0),
            ),
            task if task.output_modality() == OutputModality::Image => (
                "image.png",
                "image/png",
                b"mock image",
                Some(512),
                Some(512),
                None,
            ),
            task if task.output_modality() == OutputModality::Video => (
                "video.mp4",
                "video/mp4",
                b"mock video",
                Some(832),
                Some(480),
                Some(2.0),
            ),
            _ => (
                "audio.wav",
                "audio/wav",
                b"mock audio",
                None,
                None,
                Some(1.0),
            ),
        };
        let path = output_dir.join(name);
        fs::write(&path, contents)?;
        Ok(BackendExecution {
            runtime: runtime.to_string(),
            outputs: vec![BackendOutput {
                path,
                mime_type: Some(mime_type.to_string()),
                width,
                height,
                duration,
                metadata: json!({"mock": true}),
            }],
            warnings: Vec::new(),
            metadata: json!({"backend": "mock-media"}),
        })
    }
}

pub(super) fn media_app(api_keys: Vec<String>) -> Router {
    let store = test_store();
    install_media_model(&store);
    let inference_service =
        InferenceService::with_backend(store.clone(), Arc::new(MockMediaBackend));
    let state = ApiState::new(store, Arc::new(MockBackend))
        .with_inference_service(inference_service)
        .with_api_keys(api_keys);
    router(state)
}

pub(super) fn install_media_model(store: &ModelStore) {
    let metadata = ModelMetadata {
        tasks: vec![
            InferenceTask::ImageGeneration,
            InferenceTask::ImageEditing,
            InferenceTask::VideoGeneration,
            InferenceTask::AudioGeneration,
            InferenceTask::MusicGeneration,
            InferenceTask::TextToSpeech,
            InferenceTask::SpeechToText,
        ],
        input_modalities: vec![
            InputModality::Text,
            InputModality::Image,
            InputModality::Audio,
        ],
        output_modalities: vec![
            OutputModality::Text,
            OutputModality::Image,
            OutputModality::Video,
            OutputModality::Audio,
        ],
        ..ModelMetadata::default()
    };
    let manifest = ModelManifest {
        id: "media".to_string(),
        source: ModelSource::LocalPath {
            path: "test".to_string(),
        },
        format: ModelFormat::Unknown,
        architecture: None,
        tokenizer_path: None,
        config_path: None,
        model_path: None,
        backend: "mock-media".to_string(),
        created_unix: 1,
        files: Vec::new(),
        artifacts: Vec::new(),
        metadata,
    };
    fs::create_dir_all(store.model_dir("media")).unwrap();
    fs::write(
        store
            .model_dir("media")
            .join(crate::model_store::MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
}

pub(super) async fn post_json(
    app: &Router,
    uri: &str,
    value: Value,
    bearer_token: Option<&str>,
) -> Response {
    let mut builder = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(token) = bearer_token {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {token}"));
    }
    app.clone()
        .oneshot(builder.body(Body::from(value.to_string())).unwrap())
        .await
        .unwrap()
}

pub(super) async fn response_json(response: Response) -> Value {
    let bytes = body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

pub(super) fn test_store() -> ModelStore {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("werk1112-api-test-{}-{nanos}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    ModelStore::resolve(Some(dir)).unwrap()
}
