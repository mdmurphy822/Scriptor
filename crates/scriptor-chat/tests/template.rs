//! Byte-for-byte comparison of the template renderer against Python jinja2 goldens.

use std::path::PathBuf;

use scriptor_chat::{ChatTemplate, RenderOptions};
use serde_json::Value;

fn chat_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/chat")
}

#[test]
fn goldens_match_python_jinja() {
    let template = ChatTemplate::from_file(chat_dir().join("chat_template.jinja")).unwrap();
    let mut count = 0;
    let mut failures = Vec::new();
    for entry in std::fs::read_dir(chat_dir().join("goldens")).unwrap() {
        let path = entry.unwrap().path();
        let golden: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let options = RenderOptions::from_kwargs(&golden["kwargs"]).unwrap();
        let got = template.render(&golden["messages"], &options);
        let want = golden["rendered"].as_str().unwrap();
        match got {
            Ok(text) if text == want => {}
            Ok(text) => failures.push(format!("{}: got {text:?}\nwant {want:?}", golden["name"])),
            Err(e) => failures.push(format!("{}: {e}", golden["name"])),
        }
        count += 1;
    }
    assert!(count >= 12, "expected at least 12 goldens, found {count}");
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}
