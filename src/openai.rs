use serde::{Deserialize, Serialize};

use crate::model_store::{ModelManifest, ModelSource};

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub stop: Option<StopSpec>,
    #[serde(default)]
    pub seed: Option<u64>,
}

impl ChatCompletionRequest {
    pub fn max_completion_tokens(&self) -> usize {
        self.max_completion_tokens
            .or(self.max_tokens)
            .unwrap_or(256)
            .min(4096)
    }

    pub fn stop_strings(&self) -> Vec<String> {
        match &self.stop {
            None => Vec::new(),
            Some(StopSpec::One(stop)) => vec![stop.clone()],
            Some(StopSpec::Many(stops)) => stops.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<MessageContent>,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    pub fn as_text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter(|part| part.kind == "text")
                .filter_map(|part| part.text.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    pub fn image_urls(&self) -> Vec<String> {
        match self {
            Self::Text(_) => Vec::new(),
            Self::Parts(parts) => parts
                .iter()
                .filter(|part| matches!(part.kind.as_str(), "image_url" | "input_image"))
                .filter_map(|part| part.image_url.as_ref())
                .map(ImageUrlSpec::url)
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub image_url: Option<ImageUrlSpec>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ImageUrlSpec {
    Object(ImageUrlPart),
    Url(String),
}

impl ImageUrlSpec {
    pub fn url(&self) -> String {
        match self {
            Self::Object(image) => image.url.clone(),
            Self::Url(url) => url.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageUrlPart {
    pub url: String,
    #[serde(default)]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopSpec {
    One(String),
    Many(Vec<String>),
}

pub fn messages_to_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        let content = message
            .content
            .as_ref()
            .map(MessageContent::as_text)
            .unwrap_or_default();
        if content.trim().is_empty() {
            continue;
        }
        prompt.push_str(message.role.trim());
        prompt.push_str(": ");
        prompt.push_str(content.trim());
        prompt.push('\n');
    }
    prompt.push_str("assistant: ");
    prompt
}

#[derive(Debug, Clone)]
pub struct PromptSpec {
    pub prompt: String,
    pub stop: Vec<String>,
    pub chat_template: ChatTemplateConfig,
    pub assistant_end_token: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplateSource {
    Model,
    Werk,
    None,
}

impl ChatTemplateSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Werk => "werk",
            Self::None => "none",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplateConfig {
    pub source: ChatTemplateSource,
    pub name: String,
    pub applied_by_werk: bool,
    pub override_from_cli: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct ChatTemplateOptions<'a> {
    pub default_source: ChatTemplateSource,
    pub model_template_preferred: bool,
    pub override_name: Option<&'a str>,
}

impl Default for ChatTemplateOptions<'_> {
    fn default() -> Self {
        Self {
            default_source: ChatTemplateSource::Werk,
            model_template_preferred: false,
            override_name: None,
        }
    }
}

pub fn messages_to_prompt_for_model(
    manifest: &ModelManifest,
    messages: &[ChatMessage],
) -> PromptSpec {
    messages_to_prompt_for_model_with_template(manifest, messages, ChatTemplateOptions::default())
}

pub fn messages_to_prompt_for_model_with_template(
    manifest: &ModelManifest,
    messages: &[ChatMessage],
    options: ChatTemplateOptions<'_>,
) -> PromptSpec {
    if let Some(template) = options.override_name {
        return messages_to_prompt_with_template_override(messages, template);
    }

    if options.model_template_preferred && options.default_source == ChatTemplateSource::Model {
        return model_template_prompt(messages, None);
    }

    if uses_qwen_chat_template(manifest) {
        return PromptSpec {
            prompt: messages_to_qwen_chatml_prompt(
                messages,
                uses_qwen3_non_thinking_prompt(manifest),
            ),
            stop: qwen_stop_strings(),
            chat_template: werk_template_config("qwen-chatml", None),
            assistant_end_token: Some("<|im_end|>"),
        };
    }

    if uses_phi3_chat_template(manifest) {
        return PromptSpec {
            prompt: messages_to_phi3_prompt(messages),
            stop: phi3_stop_strings(),
            chat_template: werk_template_config("phi3", None),
            assistant_end_token: Some("<|end|>"),
        };
    }

    if uses_llama3_chat_template(manifest) {
        return PromptSpec {
            prompt: messages_to_llama3_prompt(messages),
            stop: llama3_stop_strings(),
            chat_template: werk_template_config("llama3", None),
            assistant_end_token: Some("<|eot_id|>"),
        };
    }

    if uses_gemma_chat_template(manifest) {
        return PromptSpec {
            prompt: messages_to_gemma_prompt(messages),
            stop: gemma_stop_strings(),
            chat_template: werk_template_config("gemma", None),
            assistant_end_token: Some("<end_of_turn>"),
        };
    }

    if uses_tinyllama_chat_template(manifest) {
        return PromptSpec {
            prompt: messages_to_chatml_prompt(messages),
            stop: vec![
                "<|user|>".to_string(),
                "<|system|>".to_string(),
                "</s>".to_string(),
            ],
            chat_template: werk_template_config("chatml", None),
            assistant_end_token: Some("</s>"),
        };
    }

    if options.default_source == ChatTemplateSource::Model {
        return model_template_prompt(messages, None);
    }

    if options.default_source == ChatTemplateSource::None {
        return no_template_prompt(messages, None);
    }

    generic_template_prompt(messages, None)
}

fn messages_to_prompt_with_template_override(
    messages: &[ChatMessage],
    template: &str,
) -> PromptSpec {
    let normalized = normalize_template_name(template);
    match normalized.as_str() {
        "model" => model_template_prompt(messages, Some(template)),
        "none" => no_template_prompt(messages, Some(template)),
        "generic" => generic_template_prompt(messages, Some(template)),
        "phi3" | "phi-3" => PromptSpec {
            prompt: messages_to_phi3_prompt(messages),
            stop: phi3_stop_strings(),
            chat_template: werk_template_config("phi3", Some(template)),
            assistant_end_token: Some("<|end|>"),
        },
        "llama3" | "llama-3" => PromptSpec {
            prompt: messages_to_llama3_prompt(messages),
            stop: llama3_stop_strings(),
            chat_template: werk_template_config("llama3", Some(template)),
            assistant_end_token: Some("<|eot_id|>"),
        },
        "gemma" => PromptSpec {
            prompt: messages_to_gemma_prompt(messages),
            stop: gemma_stop_strings(),
            chat_template: werk_template_config("gemma", Some(template)),
            assistant_end_token: Some("<end_of_turn>"),
        },
        "chatml" => PromptSpec {
            prompt: messages_to_chatml_prompt(messages),
            stop: vec![
                "<|user|>".to_string(),
                "<|system|>".to_string(),
                "</s>".to_string(),
            ],
            chat_template: werk_template_config("chatml", Some(template)),
            assistant_end_token: Some("</s>"),
        },
        "qwen" | "qwen-chatml" => PromptSpec {
            prompt: messages_to_qwen_chatml_prompt(messages, false),
            stop: qwen_stop_strings(),
            chat_template: werk_template_config("qwen-chatml", Some(template)),
            assistant_end_token: Some("<|im_end|>"),
        },
        _ => generic_template_prompt(messages, Some(template)),
    }
}

fn normalize_template_name(template: &str) -> String {
    template.trim().to_ascii_lowercase().replace('_', "-")
}

fn werk_template_config(name: &str, override_from_cli: Option<&str>) -> ChatTemplateConfig {
    ChatTemplateConfig {
        source: ChatTemplateSource::Werk,
        name: name.to_string(),
        applied_by_werk: true,
        override_from_cli: override_from_cli.map(str::to_string),
    }
}

fn model_template_prompt(messages: &[ChatMessage], override_from_cli: Option<&str>) -> PromptSpec {
    PromptSpec {
        prompt: messages_to_last_user_text(messages),
        stop: Vec::new(),
        chat_template: ChatTemplateConfig {
            source: ChatTemplateSource::Model,
            name: "model".to_string(),
            applied_by_werk: false,
            override_from_cli: override_from_cli.map(str::to_string),
        },
        assistant_end_token: None,
    }
}

fn no_template_prompt(messages: &[ChatMessage], override_from_cli: Option<&str>) -> PromptSpec {
    PromptSpec {
        prompt: messages_to_last_user_text(messages),
        stop: Vec::new(),
        chat_template: ChatTemplateConfig {
            source: ChatTemplateSource::None,
            name: "none".to_string(),
            applied_by_werk: false,
            override_from_cli: override_from_cli.map(str::to_string),
        },
        assistant_end_token: None,
    }
}

fn generic_template_prompt(
    messages: &[ChatMessage],
    override_from_cli: Option<&str>,
) -> PromptSpec {
    PromptSpec {
        prompt: messages_to_prompt(messages),
        stop: vec!["\nuser:".to_string()],
        chat_template: werk_template_config("generic", override_from_cli),
        assistant_end_token: None,
    }
}

fn messages_to_last_user_text(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .rev()
        .find(|message| message.role.trim().eq_ignore_ascii_case("user"))
        .or_else(|| {
            messages
                .iter()
                .rev()
                .find(|message| message.content.is_some())
        })
        .and_then(|message| message.content.as_ref())
        .map(MessageContent::as_text)
        .unwrap_or_default()
}

pub fn image_urls_from_messages(messages: &[ChatMessage]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| message.content.as_ref())
        .flat_map(MessageContent::image_urls)
        .collect()
}

fn uses_qwen_chat_template(manifest: &ModelManifest) -> bool {
    manifest_text_contains(manifest, "qwen")
}

fn uses_qwen3_non_thinking_prompt(manifest: &ModelManifest) -> bool {
    manifest_text_contains(manifest, "qwen3")
}

fn uses_tinyllama_chat_template(manifest: &ModelManifest) -> bool {
    manifest_text_contains(manifest, "tinyllama")
}

fn uses_phi3_chat_template(manifest: &ModelManifest) -> bool {
    manifest_text_contains(manifest, "phi3") || manifest_text_contains(manifest, "phi-3")
}

fn uses_llama3_chat_template(manifest: &ModelManifest) -> bool {
    manifest_text_contains(manifest, "llama-3") || manifest_text_contains(manifest, "llama3")
}

fn uses_gemma_chat_template(manifest: &ModelManifest) -> bool {
    manifest_text_contains(manifest, "gemma")
}

fn manifest_text_contains(manifest: &ModelManifest, needle: &str) -> bool {
    if manifest.id.to_ascii_lowercase().contains(needle) {
        return true;
    }
    if manifest
        .architecture
        .as_deref()
        .map(|architecture| architecture.to_ascii_lowercase().contains(needle))
        .unwrap_or(false)
    {
        return true;
    }
    match &manifest.source {
        ModelSource::HuggingFace { repo } => repo.to_ascii_lowercase().contains(needle),
        ModelSource::LocalPath { path } => path.to_ascii_lowercase().contains(needle),
    }
}

fn phi3_stop_strings() -> Vec<String> {
    vec![
        "<|end|>".to_string(),
        "<|endoftext|>".to_string(),
        "</s>".to_string(),
        "<|user|>".to_string(),
        "<|system|>".to_string(),
        "<|assistant|>".to_string(),
    ]
}

fn qwen_stop_strings() -> Vec<String> {
    vec![
        "<|im_end|>".to_string(),
        "<|endoftext|>".to_string(),
        "</s>".to_string(),
        "<|im_start|>user".to_string(),
        "<|im_start|>system".to_string(),
        "\nHuman:".to_string(),
        "\nUser:".to_string(),
        "\nAssistant:".to_string(),
    ]
}

fn llama3_stop_strings() -> Vec<String> {
    vec![
        "<|eot_id|>".to_string(),
        "<|end_of_text|>".to_string(),
        "<|start_header_id|>user<|end_header_id|>".to_string(),
        "<|start_header_id|>system<|end_header_id|>".to_string(),
    ]
}

fn gemma_stop_strings() -> Vec<String> {
    vec![
        "<end_of_turn>".to_string(),
        "<start_of_turn>user".to_string(),
        "<start_of_turn>model".to_string(),
    ]
}

fn messages_to_phi3_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        let content = message
            .content
            .as_ref()
            .map(MessageContent::as_text)
            .unwrap_or_default();
        if content.trim().is_empty() {
            continue;
        }

        let role = match message.role.trim() {
            "system" => "system",
            "assistant" => "assistant",
            _ => "user",
        };
        prompt.push_str("<|");
        prompt.push_str(role);
        prompt.push_str("|>\n");
        prompt.push_str(content.trim());
        prompt.push_str("<|end|>\n");
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}

fn messages_to_llama3_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::from("<|begin_of_text|>");
    for message in messages {
        let content = message
            .content
            .as_ref()
            .map(MessageContent::as_text)
            .unwrap_or_default();
        if content.trim().is_empty() {
            continue;
        }

        let role = match message.role.trim() {
            "system" => "system",
            "assistant" => "assistant",
            _ => "user",
        };
        prompt.push_str("<|start_header_id|>");
        prompt.push_str(role);
        prompt.push_str("<|end_header_id|>\n\n");
        prompt.push_str(content.trim());
        prompt.push_str("<|eot_id|>");
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

fn messages_to_gemma_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        let content = message
            .content
            .as_ref()
            .map(MessageContent::as_text)
            .unwrap_or_default();
        if content.trim().is_empty() {
            continue;
        }

        let role = match message.role.trim() {
            "assistant" => "model",
            _ => "user",
        };
        prompt.push_str("<start_of_turn>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(content.trim());
        prompt.push_str("<end_of_turn>\n");
    }
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

fn messages_to_qwen_chatml_prompt(messages: &[ChatMessage], non_thinking: bool) -> String {
    let mut prompt = String::new();
    for message in messages {
        let content = message
            .content
            .as_ref()
            .map(MessageContent::as_text)
            .unwrap_or_default();
        if content.trim().is_empty() {
            continue;
        }

        let role = match message.role.trim() {
            "system" => "system",
            "assistant" => "assistant",
            _ => "user",
        };
        prompt.push_str("<|im_start|>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(content.trim());
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    if non_thinking {
        prompt.push_str("<think>\n\n</think>\n\n");
    }
    prompt
}

fn messages_to_chatml_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        let content = message
            .content
            .as_ref()
            .map(MessageContent::as_text)
            .unwrap_or_default();
        if content.trim().is_empty() {
            continue;
        }

        let role = match message.role.trim() {
            "system" => "system",
            "assistant" => "assistant",
            _ => "user",
        };
        prompt.push_str("<|");
        prompt.push_str(role);
        prompt.push_str("|>\n");
        prompt.push_str(content.trim());
        prompt.push_str("</s>\n");
    }
    prompt.push_str("<|assistant|>");
    prompt
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: AssistantMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelListResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorObject,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorObject {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_store::{ModelFile, ModelFormat};

    #[test]
    fn deserializes_openai_message_content_shapes() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "model": "local-model",
                "messages": [
                    {"role": "system", "content": "Be terse."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "image_url", "image_url": {"url": "file:///tmp/a.png"}},
                        {"type": "input_image", "image_url": "data:image/png;base64,abc"}
                    ]}
                ],
                "stream": true,
                "max_completion_tokens": 12,
                "stop": ["\nuser:"]
            }"#,
        )
        .unwrap();

        assert_eq!(request.model.as_deref(), Some("local-model"));
        assert_eq!(request.max_completion_tokens(), 12);
        assert_eq!(request.stop_strings(), vec!["\nuser:".to_string()]);
        assert!(request.stream.unwrap());
        assert!(messages_to_prompt(&request.messages).contains("user: Hello"));
        assert_eq!(
            image_urls_from_messages(&request.messages),
            vec!["file:///tmp/a.png", "data:image/png;base64,abc"]
        );
    }

    #[test]
    fn tinyllama_uses_chatml_prompt_shape() {
        let manifest = ModelManifest {
            id: "TinyLLama-1B-GGUF".to_string(),
            source: ModelSource::HuggingFace {
                repo: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".to_string(),
            },
            format: ModelFormat::Gguf,
            architecture: Some("llama".to_string()),
            tokenizer_path: None,
            config_path: None,
            model_path: Some("files/model.gguf".to_string()),
            backend: "candle".to_string(),
            created_unix: 1,
            files: Vec::<ModelFile>::new(),
            artifacts: Vec::new(),
            metadata: Default::default(),
        };
        let prompt = messages_to_prompt_for_model(
            &manifest,
            &[
                ChatMessage {
                    role: "system".to_string(),
                    content: Some(MessageContent::Text("Be accurate.".to_string())),
                    name: None,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: Some(MessageContent::Text("Write one sentence.".to_string())),
                    name: None,
                },
            ],
        );

        assert!(prompt.prompt.contains("<|system|>\nBe accurate.</s>"));
        assert!(prompt.prompt.contains("<|user|>\nWrite one sentence.</s>"));
        assert!(prompt.prompt.ends_with("<|assistant|>"));
        assert!(prompt.stop.contains(&"<|user|>".to_string()));
    }

    #[test]
    fn qwen3_uses_chatml_prompt_shape_and_stops() {
        let manifest = ModelManifest {
            id: "Qwen3-14B".to_string(),
            source: ModelSource::HuggingFace {
                repo: "Qwen/Qwen3-14B".to_string(),
            },
            format: ModelFormat::SafeTensors,
            architecture: Some("qwen3".to_string()),
            tokenizer_path: None,
            config_path: None,
            model_path: Some("files/model.safetensors".to_string()),
            backend: "candle".to_string(),
            created_unix: 1,
            files: Vec::<ModelFile>::new(),
            artifacts: Vec::new(),
            metadata: Default::default(),
        };
        let prompt = messages_to_prompt_for_model(
            &manifest,
            &[
                ChatMessage {
                    role: "system".to_string(),
                    content: Some(MessageContent::Text("Be accurate.".to_string())),
                    name: None,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: Some(MessageContent::Text("Write one sentence.".to_string())),
                    name: None,
                },
            ],
        );

        assert!(prompt.prompt.contains("<|im_start|>user"));
        assert!(prompt.prompt.contains("<|im_end|>"));
        assert!(
            prompt
                .prompt
                .ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        );
        assert!(prompt.stop.contains(&"<|im_end|>".to_string()));
        assert!(prompt.stop.contains(&"\nHuman:".to_string()));
    }

    #[test]
    fn phi3_uses_expected_chat_template_and_assistant_turn_end() {
        let manifest = ModelManifest {
            id: "microsoft/Phi-3-mini-4k-instruct-onnx".to_string(),
            source: ModelSource::HuggingFace {
                repo: "microsoft/Phi-3-mini-4k-instruct-onnx".to_string(),
            },
            format: ModelFormat::Onnx,
            architecture: Some("phi3".to_string()),
            tokenizer_path: None,
            config_path: None,
            model_path: Some("files/model.onnx".to_string()),
            backend: "onnxruntime".to_string(),
            created_unix: 1,
            files: Vec::<ModelFile>::new(),
            artifacts: Vec::new(),
            metadata: Default::default(),
        };
        let prompt = messages_to_prompt_for_model(
            &manifest,
            &[
                ChatMessage {
                    role: "user".to_string(),
                    content: Some(MessageContent::Text(
                        "write a sentence about Rust.".to_string(),
                    )),
                    name: None,
                },
                ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(MessageContent::Text(
                        "Rust is a systems programming language.".to_string(),
                    )),
                    name: None,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: Some(MessageContent::Text(
                        "write a sentence about Rust.".to_string(),
                    )),
                    name: None,
                },
            ],
        );

        assert!(prompt.chat_template.applied_by_werk);
        assert_eq!(prompt.chat_template.source, ChatTemplateSource::Werk);
        assert_eq!(prompt.chat_template.name, "phi3");
        assert_eq!(prompt.assistant_end_token, Some("<|end|>"));
        assert!(
            prompt.prompt.contains(
                "<|assistant|>\nRust is a systems programming language.<|end|>\n<|user|>"
            )
        );
        assert!(prompt.prompt.ends_with("<|assistant|>\n"));
        assert_eq!(prompt.prompt.matches("<|user|>").count(), 2);
        assert_eq!(prompt.prompt.matches("<|assistant|>").count(), 2);
        assert!(prompt.stop.contains(&"<|end|>".to_string()));
        assert!(prompt.stop.contains(&"<|endoftext|>".to_string()));
    }

    #[test]
    fn phi3_repeated_short_prompt_history_keeps_turns_separate() {
        let manifest = ModelManifest {
            id: "Phi3".to_string(),
            source: ModelSource::LocalPath {
                path: "phi3".to_string(),
            },
            format: ModelFormat::Onnx,
            architecture: Some("phi3".to_string()),
            tokenizer_path: None,
            config_path: None,
            model_path: Some("files/model.onnx".to_string()),
            backend: "onnxruntime".to_string(),
            created_unix: 1,
            files: Vec::<ModelFile>::new(),
            artifacts: Vec::new(),
            metadata: Default::default(),
        };
        let prompt = messages_to_prompt_for_model(
            &manifest,
            &[
                ChatMessage {
                    role: "user".to_string(),
                    content: Some(MessageContent::Text("Say one fact.".to_string())),
                    name: None,
                },
                ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(MessageContent::Text("Rust has ownership.".to_string())),
                    name: None,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: Some(MessageContent::Text("Say one fact.".to_string())),
                    name: None,
                },
            ],
        );

        assert!(
            prompt
                .prompt
                .contains("Rust has ownership.<|end|>\n<|user|>")
        );
        assert!(
            !prompt
                .prompt
                .contains("assistant: Rust has ownership.\nuser:")
        );
        assert!(prompt.prompt.ends_with("<|assistant|>\n"));
    }
}
