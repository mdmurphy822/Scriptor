//! GPU text executor against the CPU reference on a synthetic decoder with 128-wide heads:
//! mixed prefill and decode rows, image embedding scatter, multi-row decode and the MTP layer.
//! Weight sharing between the MTP head and the decoder is covered by the `gpu_mtp_` tests.
//! Run with `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-model --features cuda gpu_ -- --ignored`.
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use scriptor_cuda::{Device, DeviceArena};
use scriptor_kernels::cublaslt::GemmOptions;
use scriptor_kernels::heuristics::TuningTable;
use scriptor_kernels::reference::sampling::argmax_with_margin;
use scriptor_kernels::reference::scatter::{embedding_gather, image_scatter};
use scriptor_model::gpu::{
    DecodeSeq, Fusion, GpuDecoderWeights, GpuKvPool, GpuMtpWeights, PrefillSeq, StepInput,
    TextExecutor, TextGeometry,
};
use scriptor_model::reference::{KvCache, MtpHead, TensorSource, TextModel};
use scriptor_spec::ModelSpec;
use scriptor_weights::SafetensorsDir;
use std::sync::Arc;

const HIDDEN: usize = 256;
const INTER: usize = 768;
const VOCAB: usize = 512;
const LAYERS: usize = 2;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 128;
const BLOCK: usize = 16;
const BLOCKS: usize = 32;
const SM_COUNT: usize = 48;
const IMAGE_ID: u32 = 510;
const TOLERANCE: f32 = 0.05;
const DECODE_STEPS: usize = 6;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn below(&mut self, n: usize) -> usize {
        (((self.next() + 1.0) * 0.5 * n as f32) as usize).min(n - 1)
    }
}

fn bf16_bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| bf16::from_f32(x).to_bits()).collect()
}

fn bf16_round(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| bf16::from_f32(x).to_f32()).collect()
}

fn gpu_enabled() -> bool {
    std::env::var("SCRIPTOR_GPU_TESTS").as_deref() == Ok("1")
        || std::env::var("QE_GPU_TESTS").as_deref() == Ok("1")
}

fn config_json() -> String {
    format!(
        r#"{{
  "architectures": ["GlmOcrForConditionalGeneration"],
  "image_end_token_id": 509, "image_start_token_id": 508, "image_token_id": {IMAGE_ID},
  "model_type": "glm_ocr",
  "text_config": {{
    "attention_bias": false, "eos_token_id": [500, 501], "head_dim": {HEAD_DIM},
    "hidden_act": "silu", "hidden_size": {HIDDEN}, "intermediate_size": {INTER},
    "max_position_embeddings": 4096, "model_type": "glm_ocr_text",
    "num_attention_heads": {HEADS}, "num_hidden_layers": {LAYERS},
    "num_key_value_heads": {KV_HEADS}, "num_nextn_predict_layers": 1, "pad_token_id": 500,
    "rms_norm_eps": 1e-05,
    "rope_parameters": {{"mrope_section": [16, 24, 24], "partial_rotary_factor": 1.0,
      "rope_theta": 10000, "rope_type": "default"}},
    "tie_word_embeddings": false, "vocab_size": {VOCAB}
  }},
  "tie_word_embeddings": false,
  "vision_config": {{
    "attention_bias": true, "depth": 2, "hidden_act": "silu", "hidden_size": 32,
    "in_channels": 3, "intermediate_size": 64, "model_type": "glm_ocr_vision", "num_heads": 2,
    "out_hidden_size": {HIDDEN}, "patch_size": 14,
    "rms_norm_eps": 1e-05, "rope_parameters": {{"rope_theta": 10000.0, "rope_type": "axial"}},
    "spatial_merge_size": 2, "temporal_patch_size": 2
  }}
}}"#
    )
}

/// Writes a deterministic BF16 safetensors checkpoint of the text decoder and MTP layer.
fn write_checkpoint(dir: &Path, rng: &mut Rng) {
    let mut tensors: Vec<(String, Vec<usize>, Vec<u16>)> = Vec::new();
    let mut matrix = |name: String, out: usize, inp: usize, rng: &mut Rng| {
        let amp = 1.7 / (inp as f32).sqrt();
        let v: Vec<f32> = (0..out * inp).map(|_| rng.next() * amp).collect();
        tensors.push((name, vec![out, inp], bf16_bits(&v)));
    };
    let norm = |name: String, rng: &mut Rng| {
        let v: Vec<f32> = (0..HIDDEN).map(|_| 1.0 + 0.1 * rng.next()).collect();
        (name, vec![HIDDEN], bf16_bits(&v))
    };
    let (qd, kd) = (HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
    let layer_names = [
        "input_layernorm",
        "post_self_attn_layernorm",
        "post_attention_layernorm",
        "post_mlp_layernorm",
    ];
    let layer = |tensors_prefix: String,
                 rng: &mut Rng,
                 matrix: &mut dyn FnMut(String, usize, usize, &mut Rng)|
     -> Vec<(String, Vec<usize>, Vec<u16>)> {
        let mut norms = Vec::new();
        for n in layer_names {
            norms.push(norm(format!("{tensors_prefix}.{n}.weight"), rng));
        }
        matrix(
            format!("{tensors_prefix}.self_attn.q_proj.weight"),
            qd,
            HIDDEN,
            rng,
        );
        matrix(
            format!("{tensors_prefix}.self_attn.k_proj.weight"),
            kd,
            HIDDEN,
            rng,
        );
        matrix(
            format!("{tensors_prefix}.self_attn.v_proj.weight"),
            kd,
            HIDDEN,
            rng,
        );
        matrix(
            format!("{tensors_prefix}.self_attn.o_proj.weight"),
            HIDDEN,
            qd,
            rng,
        );
        matrix(
            format!("{tensors_prefix}.mlp.gate_up_proj.weight"),
            2 * INTER,
            HIDDEN,
            rng,
        );
        matrix(
            format!("{tensors_prefix}.mlp.down_proj.weight"),
            HIDDEN,
            INTER,
            rng,
        );
        norms
    };
    let mut extra = Vec::new();
    matrix(
        "model.language_model.embed_tokens.weight".into(),
        VOCAB,
        HIDDEN,
        rng,
    );
    matrix("lm_head.weight".into(), VOCAB, HIDDEN, rng);
    extra.push(norm("model.language_model.norm.weight".into(), rng));
    for l in 0..=LAYERS {
        let prefix = format!("model.language_model.layers.{l}");
        extra.extend(layer(prefix.clone(), rng, &mut matrix));
        if l == LAYERS {
            matrix(format!("{prefix}.embed_tokens.weight"), VOCAB, HIDDEN, rng);
            matrix(format!("{prefix}.eh_proj.weight"), HIDDEN, 2 * HIDDEN, rng);
            matrix(
                format!("{prefix}.shared_head.head.weight"),
                VOCAB,
                HIDDEN,
                rng,
            );
            extra.push(norm(format!("{prefix}.enorm.weight"), rng));
            extra.push(norm(format!("{prefix}.hnorm.weight"), rng));
            extra.push(norm(format!("{prefix}.shared_head.norm.weight"), rng));
        }
    }
    tensors.extend(extra);

    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, shape, bits) in &tensors {
        let start = data.len();
        for b in bits {
            data.extend_from_slice(&b.to_le_bytes());
        }
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "BF16", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut text = serde_json::to_string(&serde_json::Value::Object(header)).unwrap();
    while !text.len().is_multiple_of(8) {
        text.push(' ');
    }
    let mut file = (text.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(text.as_bytes());
    file.extend_from_slice(&data);
    std::fs::write(dir.join("model.safetensors"), file).unwrap();
}

fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("scriptor-gpu-text-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

struct Seq {
    ids: Vec<u32>,
    positions: Vec<[i32; 3]>,
    table: Vec<u32>,
    /// Rows of `ids` replaced by image embeddings, with the (bf16-rounded) embeddings.
    image: Option<(usize, Vec<f32>)>,
}

impl Seq {
    fn text(rng: &mut Rng, len: usize, table: &[u32]) -> Self {
        Self {
            ids: (0..len).map(|_| rng.below(490) as u32).collect(),
            positions: (0..len as i32).map(|i| [i, i, i]).collect(),
            table: table.to_vec(),
            image: None,
        }
    }

    /// Three text rows, a 2x2 image block, then text resuming after the largest position.
    fn with_image(rng: &mut Rng, len: usize, table: &[u32]) -> Self {
        let mut s = Self::text(rng, len, table);
        for (i, (h, w)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
            s.ids[3 + i] = IMAGE_ID;
            s.positions[3 + i] = [3, 3 + h, 3 + w];
        }
        for (i, p) in s.positions.iter_mut().enumerate().skip(7) {
            *p = [i as i32 - 2; 3];
        }
        let embeds: Vec<f32> = (0..4 * HIDDEN).map(|_| rng.next()).collect();
        s.image = Some((3, bf16_round(&embeds)));
        s
    }

    fn next_position(&self) -> i32 {
        self.positions.iter().flatten().copied().max().unwrap() + 1
    }
}

struct Cpu {
    text: TextModel,
    mtp: MtpHead,
}

impl Cpu {
    fn load(dir: &Path, spec: &ModelSpec) -> Self {
        let files = SafetensorsDir::open(dir).unwrap();
        let src = TensorSource::new(&files);
        Self {
            text: TextModel::load(&src, &spec.text).unwrap(),
            mtp: MtpHead::load(&src, &spec.text).unwrap(),
        }
    }

    /// Post-norm hidden rows of the prompt, leaving its cache filled.
    fn prefill(&self, seq: &Seq, cache: &mut KvCache) -> Vec<f32> {
        let mut embeds = embedding_gather(&self.text.embed_tokens, HIDDEN, &seq.ids);
        if let Some((start, rows)) = &seq.image {
            let slots: Vec<usize> = (*start..*start + rows.len() / HIDDEN).collect();
            image_scatter(&mut embeds, HIDDEN, rows, &slots);
        }
        self.text.forward(&embeds, &seq.positions, cache)
    }

    fn step(&self, tokens: &[u32], positions: &[[i32; 3]], cache: &mut KvCache) -> Vec<f32> {
        let embeds = embedding_gather(&self.text.embed_tokens, HIDDEN, tokens);
        self.text.forward(&embeds, positions, cache)
    }
}

fn rel_rms(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let num: f32 = got.iter().zip(want).map(|(g, w)| (g - w) * (g - w)).sum();
    let den: f32 = want.iter().map(|w| w * w).sum();
    (num / den.max(1e-20)).sqrt()
}

/// Checks fp32 logits rows against the reference: relative error and, where the reference is
/// decisive, the chosen token.
fn assert_logits(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.chunks(VOCAB).zip(want.chunks(VOCAB)).enumerate() {
        let err = rel_rms(g, w);
        assert!(err < TOLERANCE, "{label} row {i}: relative error {err}");
        let pick = argmax_with_margin(w);
        if pick.margin > 0.25 {
            assert_eq!(argmax_with_margin(g).token, pick.token, "{label} row {i}");
        }
    }
}

fn last_row(hidden: &[f32]) -> &[f32] {
    &hidden[hidden.len() - HIDDEN..]
}

struct Rig {
    stream: Arc<CudaStream>,
    _dev: Device,
    _arena: DeviceArena,
    exec: TextExecutor,
    weights: GpuDecoderWeights,
    mtp: GpuMtpWeights,
    kv: GpuKvPool,
    mtp_kv: GpuKvPool,
}

impl Rig {
    fn new(dir: &Path, spec: &ModelSpec) -> Self {
        assert!(gpu_enabled(), "set SCRIPTOR_GPU_TESTS=1 to run GPU tests");
        let dev = Device::open().expect("gpu");
        let stream = dev.new_stream().expect("stream");
        let mut arena = dev.arena(64 << 20, &stream).expect("arena");
        let files = SafetensorsDir::open(dir).unwrap();
        let geometry = TextGeometry::from_spec(&spec.text).unwrap();
        let exec = TextExecutor::new(
            dev.context(),
            &stream,
            &mut arena,
            geometry,
            SM_COUNT,
            TuningTable::default(),
            GemmOptions {
                fp8: false,
                workspace_bytes: 16 << 20,
            },
        )
        .expect("executor");
        let weights = GpuDecoderWeights::load(&stream, &files, &spec.text).unwrap();
        let mtp = GpuMtpWeights::load(&stream, &files, &spec.text).unwrap();
        let kv = GpuKvPool::new(&stream, LAYERS, BLOCKS, BLOCK, &geometry).unwrap();
        let mtp_kv = GpuKvPool::new(&stream, 1, BLOCKS, BLOCK, &geometry).unwrap();
        Self {
            stream,
            _dev: dev,
            _arena: arena,
            exec,
            weights,
            mtp,
            kv,
            mtp_kv,
        }
    }

    fn upload_bf16(&self, values: &[f32]) -> CudaSlice<u16> {
        self.stream.clone_htod(&bf16_bits(values)).expect("htod")
    }

    fn read_f32(&self, buf: &CudaSlice<f32>) -> Vec<f32> {
        self.stream.synchronize().expect("sync");
        self.stream.clone_dtoh(buf).expect("dtoh")
    }
}

fn prefill_seq(s: &Seq) -> PrefillSeq<'_> {
    PrefillSeq {
        tokens: &s.ids,
        positions: &s.positions,
        block_table: &s.table,
    }
}

fn setup(tag: &str) -> (PathBuf, ModelSpec, Rng) {
    let dir = scratch_dir(tag);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    write_checkpoint(&dir, &mut rng);
    let spec = ModelSpec::from_json(&config_json(), None).unwrap();
    (dir, spec, rng)
}

#[test]
#[ignore = "needs GPU"]
fn gpu_text_prefill_then_decode_matches_reference() {
    let (dir, spec, mut rng) = setup("decode");
    let cpu = Cpu::load(&dir, &spec);
    let mut rig = Rig::new(&dir, &spec);
    let mut seqs = [
        Seq::text(&mut rng, 20, &[7, 3, 12, 1, 20, 21]),
        Seq::text(&mut rng, 37, &[4, 9, 0, 15, 2, 22]),
        Seq::with_image(&mut rng, 12, &[30, 11, 5, 8, 23, 24]),
    ];
    let mut caches: Vec<KvCache> = seqs.iter().map(|_| cpu.text.new_cache()).collect();
    let mut want = Vec::new();
    for (seq, cache) in seqs.iter().zip(&mut caches) {
        want.extend(cpu.text.logits(last_row(&cpu.prefill(seq, cache))));
    }

    let prefill: Vec<PrefillSeq> = seqs.iter().map(prefill_seq).collect();
    let image_start = seqs[0].ids.len() + seqs[1].ids.len() + 3;
    let image_rows: Vec<u32> = (image_start..image_start + 4).map(|r| r as u32).collect();
    let sample_rows = [19, 19 + 37, 19 + 37 + 12];
    let image_embeds = rig.upload_bf16(&seqs[2].image.as_ref().unwrap().1);
    let out = rig
        .exec
        .forward(
            &rig.weights,
            &mut rig.kv,
            &StepInput {
                prefill: &prefill,
                decode: &[],
                image_rows: &image_rows,
                sample_rows: &sample_rows,
                keep_hidden: false,
            },
            Some(&image_embeds),
        )
        .unwrap();
    assert_logits("prefill", &rig.read_f32(&out.logits), &want);

    let mut tokens: Vec<u32> = want
        .chunks(VOCAB)
        .map(|row| argmax_with_margin(row).token)
        .collect();
    let mut cached: Vec<usize> = seqs.iter().map(|s| s.ids.len()).collect();
    for step in 0..DECODE_STEPS {
        let positions: Vec<[[i32; 3]; 1]> = seqs.iter().map(|s| [[s.next_position(); 3]]).collect();
        let mut want = Vec::new();
        for (i, cache) in caches.iter_mut().enumerate() {
            want.extend(
                cpu.text
                    .logits(&cpu.step(&tokens[i..=i], &positions[i], cache)),
            );
        }
        let token_rows: Vec<[u32; 1]> = tokens.iter().map(|&t| [t]).collect();
        let decode: Vec<DecodeSeq> = (0..3)
            .map(|i| DecodeSeq {
                tokens: &token_rows[i],
                positions: &positions[i],
                block_table: &seqs[i].table,
                cached: cached[i],
            })
            .collect();
        let out = rig
            .exec
            .forward(
                &rig.weights,
                &mut rig.kv,
                &StepInput {
                    prefill: &[],
                    decode: &decode,
                    image_rows: &[],
                    sample_rows: &[0, 1, 2],
                    keep_hidden: false,
                },
                None,
            )
            .unwrap();
        assert_logits(
            &format!("decode step {step}"),
            &rig.read_f32(&out.logits),
            &want,
        );
        tokens = want
            .chunks(VOCAB)
            .map(|row| argmax_with_margin(row).token)
            .collect();
        for (i, seq) in seqs.iter_mut().enumerate() {
            seq.positions.push([positions[i][0][0]; 3]);
            cached[i] += 1;
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore = "needs GPU"]
fn gpu_text_mixed_prefill_and_multi_row_decode_matches_reference() {
    let (dir, spec, mut rng) = setup("mixed");
    let cpu = Cpu::load(&dir, &spec);
    let mut rig = Rig::new(&dir, &spec);
    let a = Seq::text(&mut rng, 25, &[7, 3, 12, 1, 20, 21]);
    let b = Seq::text(&mut rng, 40, &[4, 9, 0, 15, 2, 22]);
    let c = Seq::with_image(&mut rng, 12, &[30, 11, 5, 8, 23, 24]);
    let (mut cache_a, mut cache_b, mut cache_c) = (
        cpu.text.new_cache(),
        cpu.text.new_cache(),
        cpu.text.new_cache(),
    );
    cpu.prefill(&a, &mut cache_a);
    cpu.prefill(&b, &mut cache_b);
    let want_c = cpu.text.logits(last_row(&cpu.prefill(&c, &mut cache_c)));

    let first = [prefill_seq(&a), prefill_seq(&b)];
    rig.exec
        .forward(
            &rig.weights,
            &mut rig.kv,
            &StepInput {
                prefill: &first,
                decode: &[],
                image_rows: &[],
                sample_rows: &[],
                keep_hidden: false,
            },
            None,
        )
        .unwrap();

    let draft_a: Vec<u32> = (0..3).map(|_| rng.below(490) as u32).collect();
    let draft_b: Vec<u32> = (0..3).map(|_| rng.below(490) as u32).collect();
    let pos_a: Vec<[i32; 3]> = (0..3).map(|i| [a.next_position() + i; 3]).collect();
    let pos_b: Vec<[i32; 3]> = (0..3).map(|i| [b.next_position() + i; 3]).collect();
    let mut want = Vec::new();
    want.extend(cpu.text.logits(&cpu.step(&draft_a, &pos_a, &mut cache_a)));
    want.extend(cpu.text.logits(&cpu.step(&draft_b, &pos_b, &mut cache_b)));
    want.extend(want_c);

    let prefill = [prefill_seq(&c)];
    let decode = [
        DecodeSeq {
            tokens: &draft_a,
            positions: &pos_a,
            block_table: &a.table,
            cached: a.ids.len(),
        },
        DecodeSeq {
            tokens: &draft_b,
            positions: &pos_b,
            block_table: &b.table,
            cached: b.ids.len(),
        },
    ];
    let image_embeds = rig.upload_bf16(&c.image.as_ref().unwrap().1);
    let out = rig
        .exec
        .forward(
            &rig.weights,
            &mut rig.kv,
            &StepInput {
                prefill: &prefill,
                decode: &decode,
                image_rows: &[3, 4, 5, 6],
                sample_rows: &[12, 13, 14, 15, 16, 17, 11],
                keep_hidden: false,
            },
            Some(&image_embeds),
        )
        .unwrap();
    assert_logits("mixed", &rig.read_f32(&out.logits), &want);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore = "needs GPU"]
fn gpu_text_mtp_draft_logits_match_reference() {
    let (dir, spec, mut rng) = setup("mtp");
    let cpu = Cpu::load(&dir, &spec);
    let mut rig = Rig::new(&dir, &spec);
    let seqs = [
        Seq::text(&mut rng, 22, &[7, 3, 12, 1, 20, 21]),
        Seq::text(&mut rng, 35, &[4, 9, 0, 15, 2, 22]),
    ];
    let mut mtp_caches = [cpu.mtp.new_cache(), cpu.mtp.new_cache()];
    let mut hidden_rows: Vec<f32> = Vec::new();
    let mut next_tokens: Vec<Vec<u32>> = Vec::new();
    let mut want = Vec::new();
    for (seq, mtp_cache) in seqs.iter().zip(&mut mtp_caches) {
        let mut cache = cpu.text.new_cache();
        let hidden = bf16_round(&cpu.prefill(seq, &mut cache));
        let mut next: Vec<u32> = seq.ids[1..].to_vec();
        next.push(rng.below(490) as u32);
        let layer = cpu.mtp.forward(&next, &hidden, &seq.positions, mtp_cache);
        want.extend(cpu.mtp.logits(last_row(&layer)));
        hidden_rows.extend(hidden);
        next_tokens.push(next);
    }

    let prefill = [
        PrefillSeq {
            tokens: &next_tokens[0],
            positions: &seqs[0].positions,
            block_table: &seqs[0].table,
        },
        PrefillSeq {
            tokens: &next_tokens[1],
            positions: &seqs[1].positions,
            block_table: &seqs[1].table,
        },
    ];
    let hidden = rig.upload_bf16(&hidden_rows);
    let out = rig
        .exec
        .mtp_forward(
            &rig.mtp,
            &mut rig.mtp_kv,
            &StepInput {
                prefill: &prefill,
                decode: &[],
                image_rows: &[],
                sample_rows: &[21, 21 + 35],
                keep_hidden: false,
            },
            &hidden,
        )
        .unwrap();
    assert_logits("mtp prefill", &rig.read_f32(&out.logits), &want);

    let tokens: Vec<[u32; 1]> = (0..2).map(|_| [rng.below(490) as u32]).collect();
    let positions: Vec<[[i32; 3]; 1]> = seqs.iter().map(|s| [[s.next_position(); 3]]).collect();
    let step_hidden: Vec<f32> =
        bf16_round(&(0..2 * HIDDEN).map(|_| rng.next()).collect::<Vec<_>>());
    let mut want = Vec::new();
    for (i, cache) in mtp_caches.iter_mut().enumerate() {
        let layer = cpu.mtp.forward(
            &tokens[i],
            &step_hidden[i * HIDDEN..(i + 1) * HIDDEN],
            &positions[i],
            cache,
        );
        want.extend(cpu.mtp.logits(&layer));
    }
    let decode: Vec<DecodeSeq> = (0..2)
        .map(|i| DecodeSeq {
            tokens: &tokens[i],
            positions: &positions[i],
            block_table: &seqs[i].table,
            cached: seqs[i].ids.len(),
        })
        .collect();
    let hidden = rig.upload_bf16(&step_hidden);
    let out = rig
        .exec
        .mtp_forward(
            &rig.mtp,
            &mut rig.mtp_kv,
            &StepInput {
                prefill: &[],
                decode: &decode,
                image_rows: &[],
                sample_rows: &[0, 1],
                keep_hidden: false,
            },
            &hidden,
        )
        .unwrap();
    assert_logits("mtp decode", &rig.read_f32(&out.logits), &want);
    std::fs::remove_dir_all(&dir).ok();
}

/// Prefills two text prompts, then runs one forward mixing an image prefill with multi-row
/// decode of the first two; returns the logits and the hidden rows of both forwards as bits.
fn fusion_scenario(
    rig: &mut Rig,
    fusion: Fusion,
    seqs: &[Seq; 3],
    drafts: &[Vec<u32>; 2],
) -> Vec<Vec<u32>> {
    rig.exec.set_fusion(fusion);
    let geometry = *rig.exec.geometry();
    let mut kv = GpuKvPool::new(&rig.stream, LAYERS, BLOCKS, BLOCK, &geometry).unwrap();
    let first = [prefill_seq(&seqs[0]), prefill_seq(&seqs[1])];
    let rows0 = seqs[0].ids.len() + seqs[1].ids.len();
    let sample0: Vec<u32> = (0..rows0 as u32).collect();
    let o1 = rig
        .exec
        .forward(
            &rig.weights,
            &mut kv,
            &StepInput {
                prefill: &first,
                decode: &[],
                image_rows: &[],
                sample_rows: &sample0,
                keep_hidden: true,
            },
            None,
        )
        .unwrap();
    let pos: Vec<Vec<[i32; 3]>> = (0..2)
        .map(|i| (0..3).map(|j| [seqs[i].next_position() + j; 3]).collect())
        .collect();
    let prefill = [prefill_seq(&seqs[2])];
    let decode = [0, 1].map(|i| DecodeSeq {
        tokens: &drafts[i],
        positions: &pos[i],
        block_table: &seqs[i].table,
        cached: seqs[i].ids.len(),
    });
    let image_embeds = rig.upload_bf16(&seqs[2].image.as_ref().unwrap().1);
    let o2 = rig
        .exec
        .forward(
            &rig.weights,
            &mut kv,
            &StepInput {
                prefill: &prefill,
                decode: &decode,
                image_rows: &[3, 4, 5, 6],
                sample_rows: &[12, 13, 14, 15, 16, 17, 11],
                keep_hidden: true,
            },
            Some(&image_embeds),
        )
        .unwrap();
    rig.stream.synchronize().unwrap();
    let mut out = Vec::new();
    for o in [o1, o2] {
        let logits: Vec<f32> = rig.stream.clone_dtoh(&o.logits).unwrap();
        out.push(logits.iter().map(|v| v.to_bits()).collect());
        let hidden: Vec<u16> = rig.stream.clone_dtoh(o.hidden.as_ref().unwrap()).unwrap();
        out.push(hidden.iter().map(|&v| u32::from(v)).collect());
    }
    out
}

#[test]
#[ignore = "needs GPU"]
fn gpu_text_fused_layers_equal_unfused_layers() {
    let (dir, spec, mut rng) = setup("fused");
    let mut rig = Rig::new(&dir, &spec);
    let seqs = [
        Seq::text(&mut rng, 25, &[7, 3, 12, 1, 20, 21]),
        Seq::text(&mut rng, 31, &[4, 9, 0, 15, 2, 22]),
        Seq::with_image(&mut rng, 12, &[30, 11, 5, 8, 23, 24]),
    ];
    let drafts: [Vec<u32>; 2] = [
        (0..3).map(|_| rng.below(490) as u32).collect(),
        (0..3).map(|_| rng.below(490) as u32).collect(),
    ];
    let unfused = fusion_scenario(&mut rig, Fusion::Unfused, &seqs, &drafts);
    let fused = fusion_scenario(&mut rig, Fusion::Fused, &seqs, &drafts);
    for (i, (u, f)) in unfused.iter().zip(&fused).enumerate() {
        assert_eq!(u, f, "output {i} differs between fused and unfused layers");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Overwrites the MTP embedding and shared head in the checkpoint with the decoder's
/// embedding table and language-model head.
fn make_mtp_tensors_equal(dir: &Path) {
    let path = dir.join("model.safetensors");
    let mut file = std::fs::read(&path).unwrap();
    let n = u64::from_le_bytes(file[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&file[8..8 + n]).unwrap();
    let base = 8 + n;
    let range = |name: &str| {
        let o = &header[name]["data_offsets"];
        (
            base + o[0].as_u64().unwrap() as usize,
            base + o[1].as_u64().unwrap() as usize,
        )
    };
    let prefix = format!("model.language_model.layers.{LAYERS}");
    for (src, dst) in [
        (
            "model.language_model.embed_tokens.weight".to_string(),
            format!("{prefix}.embed_tokens.weight"),
        ),
        (
            "lm_head.weight".to_string(),
            format!("{prefix}.shared_head.head.weight"),
        ),
    ] {
        let (s, e) = range(&src);
        let (ds, de) = range(&dst);
        assert_eq!(e - s, de - ds);
        let bytes = file[s..e].to_vec();
        file[ds..de].copy_from_slice(&bytes);
    }
    std::fs::write(&path, file).unwrap();
}

fn device_ptr(stream: &Arc<CudaStream>, slice: &CudaSlice<u16>) -> u64 {
    use cudarc::driver::DevicePtr;
    slice.device_ptr(stream).0
}

fn download(stream: &Arc<CudaStream>, slice: &CudaSlice<u16>) -> Vec<u16> {
    stream.synchronize().expect("sync");
    stream.clone_dtoh(slice).expect("dtoh")
}

#[test]
#[ignore = "needs GPU"]
fn gpu_mtp_head_and_embedding_alias_decoder_when_bytes_equal() {
    assert!(gpu_enabled(), "set SCRIPTOR_GPU_TESTS=1 to run GPU tests");
    let (dir, spec, _) = setup("mtp-alias-equal");
    make_mtp_tensors_equal(&dir);
    let files = SafetensorsDir::open(&dir).unwrap();
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let weights = GpuDecoderWeights::load(&stream, &files, &spec.text).unwrap();

    let separate = GpuMtpWeights::load(&stream, &files, &spec.text).unwrap();
    let shared = GpuMtpWeights::load_sharing(&stream, &files, &spec.text, &weights).unwrap();

    assert_eq!(
        device_ptr(&stream, &shared.embed_tokens),
        device_ptr(&stream, &weights.embed_tokens)
    );
    assert_eq!(
        device_ptr(&stream, &shared.shared_head),
        device_ptr(&stream, &weights.lm_head)
    );
    assert_ne!(
        device_ptr(&stream, &separate.embed_tokens),
        device_ptr(&stream, &weights.embed_tokens)
    );
    assert_eq!(
        download(&stream, &shared.embed_tokens),
        download(&stream, &separate.embed_tokens)
    );
    assert_eq!(
        download(&stream, &shared.shared_head),
        download(&stream, &separate.shared_head)
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore = "needs GPU"]
fn gpu_mtp_head_and_embedding_upload_separately_when_bytes_differ() {
    assert!(gpu_enabled(), "set SCRIPTOR_GPU_TESTS=1 to run GPU tests");
    let (dir, spec, _) = setup("mtp-alias-differ");
    let files = SafetensorsDir::open(&dir).unwrap();
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let weights = GpuDecoderWeights::load(&stream, &files, &spec.text).unwrap();
    let separate = GpuMtpWeights::load(&stream, &files, &spec.text).unwrap();
    let shared = GpuMtpWeights::load_sharing(&stream, &files, &spec.text, &weights).unwrap();

    assert_ne!(
        device_ptr(&stream, &shared.embed_tokens),
        device_ptr(&stream, &weights.embed_tokens)
    );
    assert_ne!(
        device_ptr(&stream, &shared.shared_head),
        device_ptr(&stream, &weights.lm_head)
    );
    assert_eq!(
        download(&stream, &shared.embed_tokens),
        download(&stream, &separate.embed_tokens)
    );
    assert_eq!(
        download(&stream, &shared.shared_head),
        download(&stream, &separate.shared_head)
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Logits of the last token of each sequence from one prefill pass over `seqs`.
fn packed_last_logits(rig: &mut Rig, seqs: &[&Seq]) -> Vec<f32> {
    let prefill: Vec<PrefillSeq> = seqs.iter().map(|s| prefill_seq(s)).collect();
    let mut image_rows: Vec<u32> = Vec::new();
    let mut embeds: Vec<f32> = Vec::new();
    let mut sample_rows = Vec::new();
    let mut offset = 0usize;
    for s in seqs {
        if let Some((start, rows)) = &s.image {
            let n = rows.len() / HIDDEN;
            image_rows.extend((offset + start..offset + start + n).map(|r| r as u32));
            embeds.extend_from_slice(rows);
        }
        offset += s.ids.len();
        sample_rows.push((offset - 1) as u32);
    }
    let image_embeds = (!embeds.is_empty()).then(|| rig.upload_bf16(&embeds));
    let out = rig
        .exec
        .forward(
            &rig.weights,
            &mut rig.kv,
            &StepInput {
                prefill: &prefill,
                decode: &[],
                image_rows: &image_rows,
                sample_rows: &sample_rows,
                keep_hidden: false,
            },
            image_embeds.as_ref(),
        )
        .unwrap();
    rig.read_f32(&out.logits)
}

#[test]
#[ignore = "needs GPU"]
fn gpu_text_packed_prefill_matches_each_request_alone() {
    let (dir, spec, mut rng) = setup("packed");
    let cpu = Cpu::load(&dir, &spec);
    let mut rig = Rig::new(&dir, &spec);
    let seqs = [
        Seq::text(&mut rng, 20, &[7, 3, 12, 1, 20, 21]),
        Seq::with_image(&mut rng, 12, &[30, 11, 5, 8, 23, 24]),
        Seq::text(&mut rng, 37, &[4, 9, 0, 15, 2, 22]),
    ];
    let refs: Vec<&Seq> = seqs.iter().collect();
    let packed = packed_last_logits(&mut rig, &refs);
    for (i, seq) in seqs.iter().enumerate() {
        let alone = packed_last_logits(&mut rig, &[seq]);
        let row = &packed[i * VOCAB..(i + 1) * VOCAB];
        let err = rel_rms(row, &alone);
        assert!(err < TOLERANCE, "request {i}: packed vs alone {err}");
        assert_eq!(
            argmax_with_margin(row).token,
            argmax_with_margin(&alone).token,
            "request {i}"
        );
        let mut cache = cpu.text.new_cache();
        let want = cpu.text.logits(last_row(&cpu.prefill(seq, &mut cache)));
        assert_logits("packed vs reference", row, &want);
    }
    // Reordering the packed requests leaves each request's logits unchanged.
    let reversed: Vec<&Seq> = seqs.iter().rev().collect();
    let flipped = packed_last_logits(&mut rig, &reversed);
    for i in 0..seqs.len() {
        let j = seqs.len() - 1 - i;
        let err = rel_rms(
            &flipped[j * VOCAB..(j + 1) * VOCAB],
            &packed[i * VOCAB..(i + 1) * VOCAB],
        );
        assert!(err < TOLERANCE, "request {i}: reordered {err}");
    }
}
