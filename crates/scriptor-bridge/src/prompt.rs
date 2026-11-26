//! The recognition prompts, the request sampling constants, and the
//! OpenAI-compatible chat body one region request carries.

use serde_json::{Value, json};

/// The reply limit of a region request.
pub const MAX_TOKENS: u32 = 8_192;

/// The sampling temperature.
pub const TEMPERATURE: f64 = 0.0;

/// The nucleus mass.
pub const TOP_P: f64 = 0.00001;

/// The candidate count.
pub const TOP_K: u32 = 1;

/// The repetition penalty.
pub const REPETITION_PENALTY: f64 = 1.1;

/// What a region is recognised as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    Text,
    Table,
    Formula,
}

impl Task {
    /// The task named by a layout task type (`text`, `table`, `formula`);
    /// `None` for `skip` and any other name, which are not sent to the seat.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "text" => Some(Self::Text),
            "table" => Some(Self::Table),
            "formula" => Some(Self::Formula),
            _ => None,
        }
    }

    /// The prompt text of the task.
    pub const fn prompt(self) -> &'static str {
        match self {
            Self::Text => "Text Recognition:",
            Self::Table => "Table Recognition:",
            Self::Formula => "Formula Recognition:",
        }
    }
}

/// The chat-completions body of one region request: the image part then the
/// prompt part in a single user message, the sampling fields, `stop_on_repetition: true` (honoured by engines that
/// implement it, ignored by servers that do not), and `model`
/// when it is not empty.
pub fn request_body(model: &str, task: Task, image_data_url: &str) -> Value {
    let mut body = json!({
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": task.prompt()},
            ],
        }],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "stop_on_repetition": true,
    });
    if !model.is_empty() {
        body["model"] = json!(model);
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    fn golden(prompt: &str) -> Value {
        json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": 8192,
            "temperature": 0.0,
            "top_p": 0.00001,
            "top_k": 1,
            "repetition_penalty": 1.1,
            "stop_on_repetition": true,
            "model": "glm-ocr",
        })
    }

    #[test]
    fn each_task_builds_its_golden_body() {
        let url = "data:image/png;base64,AAAA";
        for (task, prompt) in [
            (Task::Text, "Text Recognition:"),
            (Task::Table, "Table Recognition:"),
            (Task::Formula, "Formula Recognition:"),
        ] {
            assert_eq!(request_body("glm-ocr", task, url), golden(prompt));
        }
    }

    #[test]
    fn an_empty_model_is_omitted() {
        let body = request_body("", Task::Text, "data:image/png;base64,AAAA");
        assert!(body.get("model").is_none());
    }

    #[test]
    fn task_names_map_and_skip_is_not_sent() {
        assert_eq!(Task::from_name("text"), Some(Task::Text));
        assert_eq!(Task::from_name("table"), Some(Task::Table));
        assert_eq!(Task::from_name("formula"), Some(Task::Formula));
        assert_eq!(Task::from_name("skip"), None);
    }

    #[test]
    fn sampling_fields_serialize_with_their_values() {
        let text = request_body("m", Task::Text, "u").to_string();
        let parsed: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["top_p"].as_f64(), Some(1e-5));
        assert_eq!(parsed["temperature"].as_f64(), Some(0.0));
        assert_eq!(parsed["repetition_penalty"].as_f64(), Some(1.1));
        assert_eq!(parsed["stop_on_repetition"].as_bool(), Some(true));
    }
}
