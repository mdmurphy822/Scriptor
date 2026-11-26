//! Prompt building and incremental detokenisation over the tiny synthetic tokenizer.

use std::path::PathBuf;

use scriptor_chat::{
    ByteTable, ChatMessage, ChatTemplate, Content, ContentPart, ImageGrid, PromptBuilder,
    RenderOptions, Tokenizer,
};

fn chat_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/chat")
}

fn builder() -> PromptBuilder {
    let tokenizer = Tokenizer::from_file(chat_dir().join("tiny-tokenizer.json")).unwrap();
    let template = ChatTemplate::from_file(chat_dir().join("chat_template.jinja")).unwrap();
    PromptBuilder::new(tokenizer, template, "<|image|>", 2).unwrap()
}

fn semantik_message() -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: "user".into(),
        content: Some(Content::Parts(vec![
            ContentPart::image_url("data:image/jpeg;base64,AA==", None),
            ContentPart::text("Text Recognition:"),
        ])),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }]
}

fn generation_options() -> RenderOptions {
    RenderOptions {
        add_generation_prompt: true,
        ..RenderOptions::default()
    }
}

#[test]
fn semantik_body_matches_golden_and_expands_the_image() {
    let b = builder();
    let golden: serde_json::Value = serde_json::from_slice(
        &std::fs::read(chat_dir().join("goldens/semantik_text.json")).unwrap(),
    )
    .unwrap();
    let prepared = b
        .prepare(&semantik_message(), &generation_options())
        .unwrap();
    assert_eq!(prepared.text, golden["rendered"].as_str().unwrap());
    assert_eq!(prepared.images.len(), 1);
    assert_eq!(prepared.images[0].url, "data:image/jpeg;base64,AA==");

    let tok = b.tokenizer();
    let gmask = tok.require_token_id("[gMASK]").unwrap();
    let sop = tok.require_token_id("<sop>").unwrap();
    assert_eq!(&prepared.token_ids[..2], &[gmask, sop]);

    let prompt = b
        .expand(&prepared, &[ImageGrid { t: 1, h: 6, w: 8 }])
        .unwrap();
    assert_eq!(prompt.image_ranges.len(), 1);
    let range = prompt.image_ranges[0].clone();
    assert_eq!(range.len(), 12);
    assert!(
        prompt.token_ids[range.clone()]
            .iter()
            .all(|&id| id == b.image_token_id())
    );
    assert_eq!(prompt.token_ids.len(), prepared.token_ids.len() - 1 + 12);
    let begin = tok.require_token_id("<|begin_of_image|>").unwrap();
    let end = tok.require_token_id("<|end_of_image|>").unwrap();
    assert_eq!(prompt.token_ids[range.start - 1], begin);
    assert_eq!(prompt.token_ids[range.end], end);

    let positions = prompt.positions().unwrap();
    assert_eq!(positions.len(), prompt.token_ids.len());
    assert_eq!(
        positions.rows[0][range.start],
        positions.rows[0][range.end - 1]
    );
    assert!(positions.delta < 0);
}

#[test]
fn grid_count_mismatch_is_rejected() {
    let b = builder();
    let prepared = b
        .prepare(&semantik_message(), &generation_options())
        .unwrap();
    assert!(b.expand(&prepared, &[]).is_err());
    assert!(
        b.expand(&prepared, &[ImageGrid { t: 1, h: 3, w: 4 }])
            .is_err(),
        "odd merge grid"
    );
}

#[test]
fn byte_detokenizer_matches_tokenizer_decode_and_holds_partial_characters() {
    let bytes = std::fs::read(chat_dir().join("tiny-tokenizer.json")).unwrap();
    let table = ByteTable::from_tokenizer_json(&bytes).unwrap();
    let tok = Tokenizer::from_bytes(&bytes).unwrap();
    let text = "the other thing \u{65e5}\u{672c}\u{8a9e} caf\u{e9} \u{1f600} done";
    let ids = tok.encode(text).unwrap();
    assert_eq!(
        table.decode(&ids, false).unwrap(),
        tok.decode(&ids, false).unwrap()
    );
    assert_eq!(table.decode(&ids, false).unwrap(), text);

    let mut decoder = table.decoder(false);
    let mut streamed = String::new();
    for &id in &ids {
        let piece = decoder.push(id).unwrap();
        assert!(!piece.contains('\u{fffd}'));
        streamed.push_str(&piece);
    }
    streamed.push_str(&decoder.finish());
    assert_eq!(streamed, text);

    // Byte-level fallback tokens split a multi-byte character across pushes.
    let emoji = tok.encode("\u{1f600}").unwrap();
    assert!(emoji.len() > 1);
    let mut decoder = table.decoder(false);
    for &id in &emoji[..emoji.len() - 1] {
        assert_eq!(decoder.push(id).unwrap(), "");
    }
    assert_eq!(decoder.push(*emoji.last().unwrap()).unwrap(), "\u{1f600}");
}

#[test]
fn special_tokens_are_skipped_on_request() {
    let bytes = std::fs::read(chat_dir().join("tiny-tokenizer.json")).unwrap();
    let table = ByteTable::from_tokenizer_json(&bytes).unwrap();
    let tok = Tokenizer::from_bytes(&bytes).unwrap();
    let ids = tok.encode("a<|user|>b").unwrap();
    assert_eq!(table.decode(&ids, true).unwrap(), "ab");
    assert_eq!(table.decode(&ids, false).unwrap(), "a<|user|>b");
    assert_eq!(tok.decode(&ids, true).unwrap(), "ab");
}
