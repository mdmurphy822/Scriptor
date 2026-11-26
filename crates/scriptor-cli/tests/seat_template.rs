//! The seat template follows Semantik's render rules and the CLI's flags.

#[path = "../src/args.rs"]
#[allow(dead_code)]
mod args;

use clap::CommandFactory;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn seat_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../seat")
}

fn read(name: &str) -> String {
    std::fs::read_to_string(seat_dir().join(name)).unwrap()
}

fn params() -> BTreeMap<String, String> {
    read("params.toml")
        .lines()
        .filter(|line| !line.trim_start().starts_with('#') && line.contains('='))
        .map(|line| {
            let (name, value) = line.split_once('=').unwrap();
            let value = value.trim().trim_matches('"');
            (name.trim().to_string(), value.to_string())
        })
        .collect()
}

fn safe(value: &str) -> bool {
    value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "._:/@+=,_-".contains(c))
}

/// Substitutes `{{name}}` placeholders with the Semantik render rules.
fn render(source: &str, params: &BTreeMap<String, String>) -> Result<String, String> {
    for (name, value) in params {
        if !safe(value) {
            return Err(format!("unsafe value for {name}"));
        }
    }
    let mut out = String::new();
    let mut rest = source;
    while let Some(open) = rest.find("{{") {
        out.push_str(&rest[..open]);
        let after = &rest[open + 2..];
        let close = after.find("}}").ok_or("unterminated placeholder")?;
        let name = &after[..close];
        let valid = !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
        let value = params
            .get(name)
            .filter(|_| valid)
            .ok_or_else(|| format!("unresolved placeholder {name}"))?;
        out.push_str(value);
        rest = &after[close + 2..];
    }
    out.push_str(rest);
    Ok(out)
}

fn placeholders(source: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut rest = source;
    while let Some(open) = rest.find("{{") {
        let after = &rest[open + 2..];
        let close = after.find("}}").unwrap();
        names.push(after[..close].to_string());
        rest = &after[close + 2..];
    }
    names
}

#[test]
fn defaults_are_render_safe() {
    for (name, value) in params() {
        assert!(safe(&value), "{name} has unsafe characters");
    }
}

#[test]
fn every_placeholder_resolves_from_params() {
    let params = params();
    for file in ["scriptor.start.sh", "scriptor.stop.sh"] {
        let source = read(file);
        for name in placeholders(&source) {
            assert!(
                params.contains_key(&name),
                "{file}: {name} not in params.toml"
            );
        }
        let mut filled = params.clone();
        filled.insert("binary".into(), "/opt/engine/bin".into());
        filled.insert("model_dir".into(), "/opt/model".into());
        filled.insert("state_dir".into(), "/var/seat".into());
        let rendered = render(&source, &filled).unwrap();
        assert!(!rendered.contains("{{"));
    }
}

#[test]
fn every_param_is_used() {
    let used: Vec<String> = ["scriptor.start.sh", "scriptor.stop.sh"]
        .iter()
        .flat_map(|f| placeholders(&read(f)))
        .collect();
    for name in params().keys() {
        assert!(used.contains(name), "{name} is unused");
    }
}

#[test]
fn render_rejects_unsafe_and_unknown() {
    let mut params = params();
    assert!(render("{{nope}}", &params).is_err());
    assert!(render("{{port", &params).is_err());
    assert!(render("{{}}", &params).is_err());
    params.insert("port".into(), "80; rm".into());
    assert!(render("{{port}}", &params).is_err());
}

#[test]
fn start_script_flags_exist_in_the_cli() {
    let command = args::Cli::command();
    let serve = command.find_subcommand("serve").unwrap();
    let longs: Vec<String> = serve
        .get_arguments()
        .filter_map(|a| a.get_long().map(str::to_string))
        .collect();
    let source = read("scriptor.start.sh");
    let flags: Vec<&str> = source
        .split_whitespace()
        .filter(|w| w.starts_with("--") && w.len() > 2)
        .collect();
    assert!(!flags.is_empty());
    for flag in flags {
        let name = flag.trim_start_matches("--");
        if name == "model-dir" || name == "out" {
            continue;
        }
        let name = name.trim_end_matches(')');
        assert!(longs.iter().any(|l| l == name), "unknown flag --{name}");
    }
}

#[test]
fn rendered_scripts_parse_and_hold_no_machine_paths() {
    let mut filled = params();
    filled.insert("binary".into(), "/opt/engine/bin".into());
    filled.insert("model_dir".into(), "/opt/model".into());
    filled.insert("state_dir".into(), "/var/seat".into());
    for file in ["scriptor.start.sh", "scriptor.stop.sh"] {
        let rendered = render(&read(file), &filled).unwrap();
        let path =
            std::env::temp_dir().join(format!("scriptor-seat-{}-{file}", std::process::id()));
        std::fs::write(&path, rendered).unwrap();
        let status = std::process::Command::new("bash")
            .arg("-n")
            .arg(&path)
            .status()
            .unwrap();
        std::fs::remove_file(&path).ok();
        assert!(status.success(), "{file} fails bash -n");
    }
    for file in [
        "scriptor.start.sh",
        "scriptor.stop.sh",
        "params.toml",
        "README.md",
        "scriptor.service.example",
    ] {
        assert!(
            !read(file).contains("/home/"),
            "{file} holds a machine path"
        );
    }
}
