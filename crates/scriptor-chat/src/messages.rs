//! OpenAI-compatible chat message types and their conversion to template context values.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::error::{ChatError, Result};

/// A message content value: plain text or a list of typed parts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    /// Plain text.
    Text(String),
    /// Typed parts such as `{"type":"text","text":"..."}`.
    Parts(Vec<ContentPart>),
}

/// An image reference carried by an `image_url` content part.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageUrl {
    /// The image location, typically a `data:` URL.
    pub url: String,
    /// Optional detail hint (`auto`, `low`, `high`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// One typed element of a multi-part message content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ContentPart {
    /// `{"type":"text","text":"..."}`.
    Text {
        /// Part discriminator, always `text`.
        #[serde(rename = "type", with = "text_tag")]
        kind: TextTag,
        /// The text.
        text: String,
    },
    /// `{"type":"image_url","image_url":{"url":"...","detail":"..."}}`.
    ImageUrl {
        /// Part discriminator, always `image_url`.
        #[serde(rename = "type", with = "image_tag")]
        kind: ImageTag,
        /// The image reference.
        image_url: ImageUrl,
    },
    /// Any other part, passed through to the template unchanged.
    Other(Value),
}

/// Marker for the `text` discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextTag;

/// Marker for the `image_url` discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageTag;

macro_rules! tag_module {
    ($module:ident, $ty:ident, $literal:literal) => {
        mod $module {
            use super::$ty;
            use serde::{Deserialize, Deserializer, Serializer, de::Error};

            pub fn serialize<S: Serializer>(_: &$ty, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_str($literal)
            }

            pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<$ty, D::Error> {
                let v = String::deserialize(d)?;
                if v == $literal {
                    Ok($ty)
                } else {
                    Err(D::Error::custom(concat!("expected type ", $literal)))
                }
            }
        }
    };
}
tag_module!(text_tag, TextTag, "text");
tag_module!(image_tag, ImageTag, "image_url");

impl ContentPart {
    /// Builds a text part.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            kind: TextTag,
            text: text.into(),
        }
    }

    /// Builds an image part from a URL and optional detail hint.
    pub fn image_url(url: impl Into<String>, detail: Option<String>) -> Self {
        Self::ImageUrl {
            kind: ImageTag,
            image_url: ImageUrl {
                url: url.into(),
                detail,
            },
        }
    }

    /// Converts the part to the JSON value the chat template receives.
    pub fn to_value(&self) -> Value {
        match self {
            Self::Text { text, .. } => json!({"type": "text", "text": text}),
            Self::ImageUrl { image_url, .. } => match &image_url.detail {
                Some(d) => {
                    json!({"type": "image_url", "image_url": {"url": image_url.url, "detail": d}})
                }
                None => json!({"type": "image_url", "image_url": {"url": image_url.url}}),
            },
            Self::Other(v) => v.clone(),
        }
    }
}

/// The function half of a tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Function name.
    pub name: String,
    /// Arguments as a JSON object, or a string holding one.
    #[serde(default)]
    pub arguments: Value,
}

/// A tool call issued by the assistant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Call identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Call type, always `function`.
    #[serde(default = "function_type", rename = "type")]
    pub kind: String,
    /// The called function.
    pub function: FunctionCall,
}

fn function_type() -> String {
    "function".to_owned()
}

/// One chat message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// `system`, `user`, `assistant` or `tool`.
    pub role: String,
    /// Message content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
    /// Assistant reasoning text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Assistant tool calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Identifier of the call a tool message answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional participant name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    /// Converts the message to the value the chat template receives. String tool-call arguments
    /// are parsed into JSON objects; a missing content becomes an empty string.
    pub fn to_template_value(&self) -> Result<Value> {
        let mut obj = Map::new();
        obj.insert("role".into(), json!(self.role));
        let content = match &self.content {
            None => json!(""),
            Some(Content::Text(t)) => json!(t),
            Some(Content::Parts(p)) => Value::Array(p.iter().map(ContentPart::to_value).collect()),
        };
        obj.insert("content".into(), content);
        if let Some(r) = &self.reasoning_content {
            obj.insert("reasoning_content".into(), json!(r));
        }
        if let Some(calls) = &self.tool_calls {
            let mut out = Vec::with_capacity(calls.len());
            for call in calls {
                let arguments = match &call.function.arguments {
                    Value::String(s) if s.trim().is_empty() => json!({}),
                    Value::String(s) => serde_json::from_str::<Value>(s).map_err(|e| {
                        ChatError::InvalidMessage(format!(
                            "tool call arguments for {} are not valid JSON: {e}",
                            call.function.name
                        ))
                    })?,
                    Value::Null => json!({}),
                    other => other.clone(),
                };
                out.push(json!({
                    "type": call.kind,
                    "function": {"name": call.function.name, "arguments": arguments},
                }));
            }
            obj.insert("tool_calls".into(), Value::Array(out));
        }
        Ok(Value::Object(obj))
    }
}

/// Converts a message list to the template `messages` value.
pub fn messages_to_template(messages: &[ChatMessage]) -> Result<Value> {
    messages
        .iter()
        .map(ChatMessage::to_template_value)
        .collect::<Result<Vec<_>>>()
        .map(Value::Array)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_round_trip_typed() {
        let src = r#"[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,AA==","detail":"high"}},{"type":"text","text":"Text Recognition:"},{"type":"video_url","video_url":{"url":"x"}}]"#;
        let content: Content = serde_json::from_str(src).unwrap();
        let Content::Parts(parts) = &content else {
            panic!("expected parts");
        };
        assert_eq!(
            parts[0],
            ContentPart::image_url("data:image/jpeg;base64,AA==", Some("high".into()))
        );
        assert_eq!(parts[1], ContentPart::text("Text Recognition:"));
        assert!(matches!(parts[2], ContentPart::Other(_)));
        let again: Value = serde_json::to_value(&content).unwrap();
        assert_eq!(again, serde_json::from_str::<Value>(src).unwrap());
    }

    #[test]
    fn template_value_keeps_part_order() {
        let msg = ChatMessage {
            role: "user".into(),
            content: Some(Content::Parts(vec![
                ContentPart::image_url("u", None),
                ContentPart::text("t"),
            ])),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let v = msg.to_template_value().unwrap();
        assert_eq!(v["content"][0]["type"], "image_url");
        assert_eq!(v["content"][1]["text"], "t");
    }
}
