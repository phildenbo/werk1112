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
}

pub fn messages_to_prompt_for_model(
    manifest: &ModelManifest,
    messages: &[ChatMessage],
) -> PromptSpec {
    if uses_tinyllama_chat_template(manifest) {
        return PromptSpec {
            prompt: messages_to_chatml_prompt(messages),
            stop: vec![
                "<|user|>".to_string(),
                "<|system|>".to_string(),
                "</s>".to_string(),
            ],
        };
    }

    PromptSpec {
        prompt: messages_to_prompt(messages),
        stop: vec!["\nuser:".to_string()],
    }
}

pub fn image_urls_from_messages(messages: &[ChatMessage]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| message.content.as_ref())
        .flat_map(MessageContent::image_urls)
        .collect()
}

fn uses_tinyllama_chat_template(manifest: &ModelManifest) -> bool {
    let id = manifest.id.to_ascii_lowercase();
    if id.contains("tinyllama") {
        return true;
    }

    match &manifest.source {
        ModelSource::HuggingFace { repo } => repo.to_ascii_lowercase().contains("tinyllama"),
        ModelSource::LocalPath { path } => path.to_ascii_lowercase().contains("tinyllama"),
    }
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
}
