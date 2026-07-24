use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "content", rename_all = "snake_case")]
pub enum ConversationContent {
    Text(String),
    Image(MediaContent),
    Video(MediaContent),
    Audio(MediaContent),
    ToolCall(ToolCallContent),
    ToolResult(ToolResultContent),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaContent {
    pub url: Option<String>,
    pub path: Option<String>,
    pub mime_type: String,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallContent {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResultContent {
    pub call_id: String,
    pub result: Value,
    pub content: Vec<ConversationContent>,
}
