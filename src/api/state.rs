use axum::{
    http::{HeaderMap, header},
    response::Response,
};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::{
    backend::{ChatGenerationSession, GenerationBackend},
    inference_service::{InferenceService, JobManager},
    model_store::{ModelManifest, ModelStore},
    openai::ChatTemplateOptions,
};

use super::response::{auth_error, constant_time_eq};

pub type PromptOptionsResolver = Arc<
    dyn Fn(&ModelStore, &ModelManifest, bool) -> anyhow::Result<ChatTemplateOptions<'static>>
        + Send
        + Sync,
>;

#[derive(Clone)]
pub struct ApiState {
    pub(super) store: Arc<ModelStore>,
    pub(super) backend: Arc<dyn GenerationBackend>,
    pub(super) default_model: Option<String>,
    prompt_options_resolver: Option<PromptOptionsResolver>,
    chat_sessions: Arc<Mutex<HashMap<String, Arc<dyn ChatGenerationSession>>>>,
    api_keys: Arc<Vec<String>>,
    pub(super) verbose: bool,
    pub(super) inference_service: Arc<InferenceService>,
    pub(super) job_manager: Arc<JobManager>,
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
        Self::new_with_default_model_and_prompt_options(store, backend, default_model, None)
    }

    pub fn new_with_default_model_and_prompt_options(
        store: ModelStore,
        backend: Arc<dyn GenerationBackend>,
        default_model: Option<String>,
        prompt_options_resolver: Option<PromptOptionsResolver>,
    ) -> Self {
        Self::new_with_default_model_prompt_options_and_verbose(
            store,
            backend,
            default_model,
            prompt_options_resolver,
            false,
        )
    }

    pub fn new_with_default_model_prompt_options_and_verbose(
        store: ModelStore,
        backend: Arc<dyn GenerationBackend>,
        default_model: Option<String>,
        prompt_options_resolver: Option<PromptOptionsResolver>,
        verbose: bool,
    ) -> Self {
        let inference_service = Arc::new(InferenceService::new(store.clone()));
        let job_manager = Arc::new(JobManager::new(inference_service.as_ref().clone()));
        Self {
            store: Arc::new(store),
            backend,
            default_model,
            prompt_options_resolver,
            chat_sessions: Arc::new(Mutex::new(HashMap::new())),
            api_keys: Arc::new(Vec::new()),
            verbose,
            inference_service,
            job_manager,
        }
    }

    pub fn with_api_keys(mut self, api_keys: Vec<String>) -> Self {
        self.api_keys = Arc::new(api_keys);
        self
    }

    /// Replaces the media inference pipeline while keeping the chat backend
    /// untouched. This is useful for embedders and deterministic test backends.
    pub fn with_inference_service(mut self, inference_service: InferenceService) -> Self {
        let inference_service = Arc::new(inference_service);
        self.job_manager = Arc::new(JobManager::new(inference_service.as_ref().clone()));
        self.inference_service = inference_service;
        self
    }

    pub fn api_key_auth_enabled(&self) -> bool {
        !self.api_keys.is_empty()
    }

    #[allow(clippy::result_large_err)]
    pub(super) fn authorize(&self, headers: &HeaderMap) -> Result<(), Response> {
        if self.api_keys.is_empty() {
            return Ok(());
        }

        let Some(header_value) = headers.get(header::AUTHORIZATION) else {
            return Err(auth_error("missing bearer token"));
        };
        let Ok(header_value) = header_value.to_str() else {
            return Err(auth_error("invalid authorization header"));
        };
        let Some((scheme, token)) = header_value.split_once(' ') else {
            return Err(auth_error("expected Authorization: Bearer <token>"));
        };
        if !scheme.eq_ignore_ascii_case("bearer") {
            return Err(auth_error("expected Authorization: Bearer <token>"));
        }
        let token = token.trim();
        if token.is_empty() {
            return Err(auth_error("empty bearer token"));
        }
        if self
            .api_keys
            .iter()
            .any(|key| constant_time_eq(key.as_bytes(), token.as_bytes()))
        {
            Ok(())
        } else {
            Err(auth_error("invalid bearer token"))
        }
    }

    pub(super) fn prompt_options(
        &self,
        manifest: &ModelManifest,
        has_images: bool,
    ) -> anyhow::Result<ChatTemplateOptions<'static>> {
        self.prompt_options_resolver
            .as_ref()
            .map(|resolver| resolver(&self.store, manifest, has_images))
            .unwrap_or_else(|| Ok(ChatTemplateOptions::default()))
    }

    pub(super) fn log_verbose(&self, message: impl AsRef<str>) {
        if self.verbose {
            eprintln!("{}", message.as_ref());
        }
    }

    pub(super) fn chat_session(
        &self,
        manifest: &ModelManifest,
        seed: Option<u64>,
    ) -> anyhow::Result<Option<Arc<dyn ChatGenerationSession>>> {
        let key = format!("{}:{seed:?}", manifest.id);
        if let Some(session) = self
            .chat_sessions
            .lock()
            .map_err(|_| anyhow::anyhow!("chat session cache mutex poisoned"))?
            .get(&key)
            .cloned()
        {
            return Ok(Some(session));
        }

        let Some(session) = self.backend.start_chat_session(manifest, seed)? else {
            return Ok(None);
        };
        let session: Arc<dyn ChatGenerationSession> = Arc::from(session);
        self.chat_sessions
            .lock()
            .map_err(|_| anyhow::anyhow!("chat session cache mutex poisoned"))?
            .insert(key, session.clone());
        Ok(Some(session))
    }
}
