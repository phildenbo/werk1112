use axum::{
    Json,
    extract::{Path as AxumPath, Query, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use serde_json::json;
use std::str::FromStr;

use crate::{
    capabilities::InferenceTask,
    inference::{parameter_schema, parameter_schema_for_manifest},
};

use super::{
    output::{execute_audio_bytes, execute_direct, streaming_file_body, submit_job},
    requests::{
        AudioGenerationApiRequest, AudioSpeechApiRequest, AudioTranscriptionApiRequest,
        DirectResponseFormat, ImageEditApiRequest, ImageGenerationApiRequest, JobCreateApiRequest,
        VideoGenerationApiRequest,
    },
};
use crate::api::{response::api_error, state::ApiState};

pub(in crate::api) async fn image_generations_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<ImageGenerationApiRequest>,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    let (request, response_format) = match request.into_inference() {
        Ok(request) => request,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, error, None),
    };
    execute_direct(state, request, response_format).await
}

pub(in crate::api) async fn image_edits_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<ImageEditApiRequest>,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    let (request, response_format) = match request.into_inference() {
        Ok(request) => request,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, error, None),
    };
    execute_direct(state, request, response_format).await
}

pub(in crate::api) async fn video_generations_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<VideoGenerationApiRequest>,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    let request = match request.into_inference() {
        Ok(request) => request,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, error, None),
    };
    submit_job(state, request).await
}

pub(in crate::api) async fn audio_generations_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<AudioGenerationApiRequest>,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    let request = match request.into_inference() {
        Ok(request) => request,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, error, None),
    };
    submit_job(state, request).await
}

pub(in crate::api) async fn audio_speech_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<AudioSpeechApiRequest>,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    let (request, asynchronous) = match request.into_inference() {
        Ok(request) => request,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, error, None),
    };
    if asynchronous {
        submit_job(state, request).await
    } else {
        execute_audio_bytes(state, request).await
    }
}

pub(in crate::api) async fn audio_transcriptions_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<AudioTranscriptionApiRequest>,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    let request = match request.into_inference() {
        Ok(request) => request,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, error, None),
    };
    execute_direct(state, request, DirectResponseFormat::Url).await
}

pub(in crate::api) async fn capabilities_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    let service = state.inference_service.clone();
    match tokio::task::spawn_blocking(move || service.capabilities()).await {
        Ok(Ok(capabilities)) => Json(capabilities).into_response(),
        Ok(Err(error)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, error.to_string(), None),
        Err(error) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("capability discovery task failed: {error}"),
            None,
        ),
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(in crate::api) struct ParametersQuery {
    #[serde(default)]
    task: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    backend: Option<String>,
}

pub(in crate::api) async fn parameters_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Query(query): Query<ParametersQuery>,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    let Some(task_name) = query.task.as_deref() else {
        return api_error(
            StatusCode::BAD_REQUEST,
            "query parameter 'task' is required".to_string(),
            Some("task".to_string()),
        );
    };
    let task = match InferenceTask::from_str(task_name) {
        Ok(task) => task,
        Err(error) => {
            return api_error(StatusCode::BAD_REQUEST, error, Some("task".to_string()));
        }
    };
    let manifest = if let Some(model) = query.model.as_deref() {
        let manifest = match state.store.get(model) {
            Ok(manifest) => manifest,
            Err(error) => {
                return api_error(
                    StatusCode::NOT_FOUND,
                    error.to_string(),
                    Some("model".to_string()),
                );
            }
        };
        if !manifest.supports_task(task) {
            return api_error(
                StatusCode::BAD_REQUEST,
                format!("model '{}' does not declare task {task}", manifest.id),
                Some("task".to_string()),
            );
        }
        Some(manifest)
    } else {
        None
    };
    let parameters = match manifest.as_ref() {
        Some(manifest) => match parameter_schema_for_manifest(task, manifest) {
            Ok(parameters) => parameters,
            Err(error) => {
                return api_error(StatusCode::BAD_REQUEST, error.to_string(), None);
            }
        },
        None => parameter_schema(task),
    };
    let mut runtimes = if let Some(manifest) = manifest.as_ref() {
        let service = state.inference_service.clone();
        let manifest = manifest.clone();
        match tokio::task::spawn_blocking(move || service.parameter_probe(&manifest, task)).await {
            Ok(Ok(probe)) => probe.candidates,
            Ok(Err(error)) => {
                return api_error(StatusCode::BAD_REQUEST, error.to_string(), None);
            }
            Err(error) => {
                return api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("parameter capability probe failed: {error}"),
                    None,
                );
            }
        }
    } else {
        Vec::new()
    };
    if let Some(filter) = query
        .backend
        .as_deref()
        .filter(|filter| !filter.eq_ignore_ascii_case("auto"))
    {
        runtimes.retain(|candidate| {
            candidate.id.eq_ignore_ascii_case(filter)
                || candidate.backend.eq_ignore_ascii_case(filter)
                || format!("{:?}", candidate.accelerator).eq_ignore_ascii_case(filter)
                || (filter.eq_ignore_ascii_case("metal")
                    && candidate.accelerator == crate::inference::RuntimeAccelerator::Mps)
        });
        // A task-only schema has no concrete model to probe. Keep the
        // requested backend as context, matching the CLI, and only reject an
        // empty filtered result when a model-specific probe actually ran.
        if runtimes.is_empty() && manifest.is_some() {
            return api_error(
                StatusCode::BAD_REQUEST,
                format!("no runtime matching backend '{filter}' supports model/task"),
                Some("backend".to_string()),
            );
        }
    }
    let selected_support = runtimes
        .iter()
        .filter(|candidate| candidate.available)
        .max_by_key(|candidate| candidate.priority)
        .or_else(|| runtimes.iter().max_by_key(|candidate| candidate.priority))
        .map(|candidate| &candidate.parameter_support);
    Json(json!({
        "object": "werk.parameter_schema",
        "task": task,
        "model": query.model,
        "backend": query.backend,
        "parameters": parameters,
        "parameter_support": selected_support,
        "runtime_candidates": runtimes,
        "model_constraints": manifest
            .as_ref()
            .map(|manifest| &manifest.metadata.parameter_constraints)
    }))
    .into_response()
}

pub(in crate::api) async fn create_job_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<JobCreateApiRequest>,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    let request = match request.into_inference() {
        Ok(request) => request,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, error, None),
    };
    submit_job(state, request).await
}

pub(in crate::api) async fn get_job_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    match state.job_manager.store().get(&id) {
        Ok(record) => Json(record).into_response(),
        Err(error) => api_error(
            StatusCode::NOT_FOUND,
            error.to_string(),
            Some("id".to_string()),
        ),
    }
}

pub(in crate::api) async fn cancel_job_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    match state.job_manager.store().cancel(&id) {
        Ok(record) => Json(record).into_response(),
        Err(error) => api_error(
            StatusCode::NOT_FOUND,
            error.to_string(),
            Some("id".to_string()),
        ),
    }
}

pub(in crate::api) async fn output_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if let Err(response) = state.authorize(&headers) {
        return response;
    }
    let output = match state.inference_service.output_store().get_output(&id) {
        Ok(output) => output,
        Err(error) => {
            return api_error(
                StatusCode::NOT_FOUND,
                error.to_string(),
                Some("id".to_string()),
            );
        }
    };
    let body = match streaming_file_body(&output.path).await {
        Ok(body) => body,
        Err(error) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to read output '{id}': {error}"),
                Some("id".to_string()),
            );
        }
    };
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    if let Ok(content_type) = HeaderValue::from_str(&output.mime_type) {
        response
            .headers_mut()
            .insert(header::CONTENT_TYPE, content_type);
    }
    if let Ok(content_length) = HeaderValue::from_str(&output.size_bytes.to_string()) {
        response
            .headers_mut()
            .insert(header::CONTENT_LENGTH, content_length);
    }
    response
}
