//! `inspect reference-dump`: runs the f32 CPU reference on a real checkpoint and writes the greedy
//! tokens with their top-2 logit margins.

use std::error::Error;
use std::path::{Path, PathBuf};

use clap::Args;
use scriptor_chat::{ChatMessage, ChatTemplate, Content, ContentPart, ImageGrid, PromptBuilder, RenderOptions, Tokenizer};
use scriptor_image::{SmartResizeConfig, preprocess};
use scriptor_kernels::reference::sampling::Penalties;
use scriptor_model::ReferenceModel;
use scriptor_spec::{ImageProcessorConfig, ModelSpec};
use serde_json::json;

/// Arguments of `inspect reference-dump`.
#[derive(Debug, Args)]
pub struct ReferenceDumpArgs {
    /// Checkpoint snapshot directory (config, tokenizer, chat template and safetensors).
    #[arg(long)]
    pub model_dir: PathBuf,
    /// Image file (JPEG, PNG or WebP).
    #[arg(long)]
    pub image: PathBuf,
    /// Text part of the user message.
    #[arg(long)]
    pub prompt: String,
    /// Maximum number of tokens to generate.
    #[arg(long, default_value_t = 32)]
    pub steps: usize,
    /// Repetition penalty applied before the greedy pick.
    #[arg(long, default_value_t = 1.1)]
    pub repetition_penalty: f32,
    /// Write the JSON dump here instead of standard output.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

fn read(dir: &Path, name: &str) -> Result<String, Box<dyn Error>> {
    std::fs::read_to_string(dir.join(name)).map_err(|e| format!("reading {name}: {e}").into())
}

/// Runs the reference model and emits the dump.
pub fn run(args: &ReferenceDumpArgs) -> Result<(), Box<dyn Error>> {
    let dir = &args.model_dir;
    let generation = std::fs::read_to_string(dir.join("generation_config.json")).ok();
    let spec = ModelSpec::from_json(&read(dir, "config.json")?, generation.as_deref())?;
    let processor = ImageProcessorConfig::from_preprocessor_json(&read(dir, "preprocessor_config.json")?)?;

    let resize = SmartResizeConfig {
        min_pixels: processor.min_pixels as u64,
        max_pixels: processor.max_pixels as u64,
        ..SmartResizeConfig::default()
    };
    let image = preprocess(&std::fs::read(&args.image)?, &resize)?;
    let pixels: Vec<f32> = image.pixels.iter().map(|v| v.to_f32()).collect();
    let grid = (image.grid_thw.t as usize, image.grid_thw.h as usize, image.grid_thw.w as usize);

    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
    let template = ChatTemplate::new(&read(dir, "chat_template.jinja")?)?;
    let builder = PromptBuilder::new(tokenizer, template, "<|image|>", spec.vision.spatial_merge_size)?;
    let message = ChatMessage {
        role: "user".into(),
        content: Some(Content::Parts(vec![
            ContentPart::image_url("image", None),
            ContentPart::text(args.prompt.clone()),
        ])),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    };
    let options = RenderOptions {
        add_generation_prompt: true,
        ..RenderOptions::default()
    };
    let prompt = builder.build(
        &[message],
        &options,
        &[ImageGrid {
            t: grid.0,
            h: grid.1,
            w: grid.2,
        }],
    )?;
    let mrope = prompt.positions()?;
    let positions: Vec<[i32; 3]> = (0..mrope.len())
        .map(|i| [mrope.rows[0][i] as i32, mrope.rows[1][i] as i32, mrope.rows[2][i] as i32])
        .collect();

    let model = ReferenceModel::load(dir, &spec)?;
    let embeds = model.encode_images(&pixels, &[grid])?;
    let penalties = Penalties {
        repetition: args.repetition_penalty,
        ..Penalties::default()
    };
    let out = model.generate(&prompt.token_ids, &embeds, &positions, mrope.delta, args.steps, penalties)?;

    let tokens: Vec<u32> = out.steps.iter().map(|s| s.token).collect();
    let text = builder.tokenizer().decode(&tokens, true)?;
    let report = json!({
        "prompt_tokens": prompt.token_ids.len(),
        "image_grid_thw": [grid.0, grid.1, grid.2],
        "image_tokens": embeds.len() / model.text.hidden,
        "repetition_penalty": args.repetition_penalty,
        "steps": out.steps.iter().map(|s| json!({"token": s.token, "margin": s.margin})).collect::<Vec<_>>(),
        "stopped_on_eos": out.hit_eos,
        "text": text,
    });
    let rendered = serde_json::to_string_pretty(&report)?;
    match &args.output {
        Some(path) => std::fs::write(path, rendered + "\n")?,
        None => println!("{rendered}"),
    }
    Ok(())
}
