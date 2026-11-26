# GLM-OCR model facts

Reference facts the engine must match. Source: the zai-org/GLM-OCR checkpoint (snapshot 2e85a62840ccac27daa451df36c736c4636b8628) and the vLLM v0.33.0 reference sources named below (glm_ocr.py, glm4.py, glm4_1v.py, glm_ocr_mtp.py, transformers glm46v/image_processing_glm46v.py, glm_ocr/modular_glm_ocr.py).

## Model facts (verified)

### 1 Checkpoint and snapshot

`zai-org/GLM-OCR` in the HF cache. Two snapshots exist; `refs/main` → **`2e85a62840ccac27daa451df36c736c4636b8628`**,
which is what the seat loads (`--model zai-org/GLM-OCR` with `HF_HUB_OFFLINE=1` resolves
`refs/main`). Both snapshots symlink the **same** `model.safetensors` blob (`a16eb0de…`,
2,650,579,464 bytes) and byte-identical `tokenizer.json`, `chat_template.jinja`,
`preprocessor_config.json`, `generation_config.json`. They differ only in `config.json`
(2e85a62: transformers 5.17 layout, adds explicit `rope_type: "axial"` on the vision config,
`attention_bias`, `dtype`; same values otherwise), `tokenizer_config.json` (2e85a62 drops
`<|image|>`/`<|video|>` from `extra_special_tokens`; they remain special in `tokenizer.json`),
and 2e85a62 adds `processor_config.json` (ca5d8b3e has README/.eval_results instead). The
engine takes an explicit `--model-dir` (the snapshot directory); the fixture set carries
`2e85a62`'s configs.

### 2 Architecture (`config.json`, `model_type: glm_ocr`, `architectures: ["GlmOcrForConditionalGeneration"]`)

Text decoder (`text_config`, `model_type: glm_ocr_text`) — vLLM runs it as `Glm4ForCausalLM` (`glm4.py`):

| Field | Value |
|---|---|
| hidden / intermediate / layers | 1536 / 4608 (SwiGLU, `silu`) / 16 |
| Attention | 16 Q heads × head_dim 128, **8 KV heads** (GQA 2:1), `attention_bias: false`, no q/k norm |
| RoPE | theta 10000, `partial_rotary_factor 1.0` (all 128 dims), **`is_neox_style=False`** in vLLM (GPT-J interleaved pairs), **M-RoPE `mrope_section [16, 24, 24]`** (sums to 64 = head_dim/2 frequency pairs; t/h/w sections, not interleaved). Text tokens get identical t=h=w positions; image tokens get (t, h, w) grid positions offset by the preceding text length; text after an image resumes at max+1 (`Glm4vForConditionalGeneration.get_mrope_input_positions`) |
| Norms | **plain RMSNorm** (`eps 1e-5`), **four per layer** (sandwich): `input_layernorm` → attn → `post_self_attn_layernorm` → residual add → `post_attention_layernorm` → MLP → `post_mlp_layernorm` → residual add (`Glm4DecoderLayer.forward`); final `model.norm` |
| Vocab / embeddings | 59392; `tie_word_embeddings: false`; separate `embed_tokens` and `lm_head` (BF16, 182 MB each) |
| Context | `max_position_embeddings 131072`; seat runs `--max-model-len 16384` |
| Special ids | eos `[59246 <\|endoftext\|>, 59253 <\|user\|>]`, pad 59246; image_start 59256 `<\|begin_of_image\|>`, image_end 59257, **image_token 59280 `<\|image\|>`**; video 59258/59259/59281; `[gMASK]` `<sop>` open every prompt |
| MTP | `num_nextn_predict_layers 1`: layer index **16** holds `embed_tokens` (own copy, 182 MB), `enorm`, `hnorm`, `eh_proj [1536, 3072]` over concat(enorm(embed(tok)), hnorm(hidden)), one full `Glm4DecoderLayer` (own KV cache, 4 KB/token), `shared_head.norm`, `shared_head.head [59392,1536]` (own lm_head copy). vLLM zeroes `inputs_embeds` at position 0 (`glm_ocr_mtp.py`) |

Vision encoder (`vision_config`, `model_type: glm_ocr_vision`) — vLLM `GlmOcrVisionTransformer` (`glm_ocr.py`, over `glm4_1v.py`):

| Field | Value |
|---|---|
| depth / hidden / heads / intermediate | 24 / 1024 / 16 (head_dim 64) / 4096 |
| Patch embed | `Conv3d(3→1024, kernel (2,14,14), stride same, bias)` on `[N, 3·2·14·14 = 1176]` flattened patches (each image frame duplicated along t, `temporal_patch_size 2`) |
| Position encoding | **no learned position embedding** (`del self.embeddings`, no `post_conv_layernorm`); **2-D axial RoPE**: `get_rope(head_size 64, max_position 8192, is_neox_style=True, partial_rotary_factor 0.5)` → 32 of 64 dims rotated, 16 frequency pairs indexed by the patch's h index and 16 by its w index (`rot_pos_emb`: pos_ids laid out in merge-window order) |
| Block | `x = x + attn(RMSNorm(x))`; `x = x + mlp(RMSNorm(x))` (`norm1`,`norm2` RMSNorm eps 1e-5); attention: fused `qkv [3072,1024]+bias`, **per-head RMSNorm q_norm/k_norm [64] (eps 1e-5)** before RoPE, non-causal varlen attention per image (`cu_seqlens`), scale 64^-0.5, `proj [1024,1024]+bias`; MLP `gate_proj`,`up_proj [4096,1024]+bias`, SiLU-and-mul, `down_proj [1024,4096]+bias` |
| Adapter | `post_layernorm` RMSNorm(1024) → view `[N/4, 2, 2, 1024]` → `downsample Conv2d(1024→1536, k=2, s=2, bias)` → `[N/4, 1536]` → merger: `proj [1536,1536]` (no bias) → `LayerNorm(1536)` (weight+bias, eps 1e-5) → **GELU (erf)** → `gate_proj`,`up_proj [4608,1536]` (no bias) → SiLU-and-mul → `down_proj [1536,4608]` → image embeddings `[N/4, 1536]` scattered into the `<\|image\|>` positions |
| Pixel normalisation | vLLM fuses rescale+normalise on device (`mm_input_norm`): `y = x·(1/255)/std − mean/std` with CLIP mean `[0.48145466, 0.4578275, 0.40821073]`, std `[0.26862954, 0.26130258, 0.27577711]` |

Image preprocessing (`Glm46VImageProcessor`, transformers 5.20.2 in the image): RGB convert; `smart_resize(num_frames=2, h, w, factor=28, min_pixels=12544, max_pixels=9633792)` (round-half-even to a multiple of 28; floor-shrink when `2·h̄·w̄ > max`, ceil-grow when `< min`; aspect ratio > 200 refused); BICUBIC antialiased resize (torchvision tensor path); rescale 1/255; normalise; patchify in order `(grid_h/2, grid_w/2, 2, 2)` → consecutive groups of 4 patches form one merge window, each patch flattened as `(c, t, 14, 14)` with the frame repeated twice. **Tokens per image = (h̄/28)·(w̄/28)**, max 12288. The prompt holds one `<|image|>` token that the processor expands to that many copies (`_get_prompt_updates`).

**Semantik pre-conforms the image.** `crop.rs` already resizes every region crop to the same `smart_resize` target (factor 28, `MIN_PIXELS 112·112`, `MAX_PIXELS 28·28·2·6000` — tighter than the model's 6144) with the `image` crate's bicubic filter and encodes JPEG quality 75 → for bridge traffic the server-side resize is the identity and the only server-side pixel work is JPEG decode → normalise → patchify. Non-conforming images still need the resize path (equivalence set must include one).

### 3 Tensor inventory (`model.safetensors`, 526 tensors, all BF16, 2.65 GB; header fixture committed)

| Group | bytes |
|---|---|
| 16 decoder layers (`model.language_model.layers.{0..15}`: `gate_up_proj [9216,1536]`, `down_proj [1536,4608]`, `q_proj [2048,1536]`, `k_proj [1024,1536]`, `v_proj [1024,1536]`, `o_proj [1536,2048]`, 4 norms) | 0.981 GB (61.3 MB/layer) |
| `lm_head.weight`, `model.language_model.embed_tokens.weight` [59392,1536] | 0.182 GB each |
| MTP layer 16 (`embed_tokens`, `eh_proj`, `enorm`, `hnorm`, decoder layer, `shared_head.{norm,head}`) | 0.439 GB (0.254 GB streamed per draft step: layer + head; embed is a gather) |
| `model.visual.*` (24 blocks + `patch_embed`, `downsample`, `merger`, `post_layernorm`) | 0.868 GB |

Weights read per decode step (text): **1.164 GB** (decoder + lm_head) → 4.8 ms at 240 GB/s. KV: **64 KB/token** BF16 (16 layers × 2 × 8 × 128 × 2 B) + 4 KB MTP.

### 4 Reference behaviour the engine must match (vLLM v0.33.0 image)

- Model: `glm_ocr.py` (vision tower), `glm4.py` (decoder), `glm4_1v.py` (patch embed, merger, `rot_pos_emb`, mrope positions, prompt expansion), `glm_ocr_mtp.py`, transformers `glm46v/image_processing_glm46v.py` (`smart_resize`, `patchify`), `glm_ocr/modular_glm_ocr.py`.
- Sampling for the Semantik request (`prompt.rs`): `temperature 0.0`, `top_p 1e-5`, `top_k 1`, **`repetition_penalty 1.1`**, `max_tokens 8192`. vLLM v1 sampler order (`sampler.py`): logits → fp32 → `apply_logits_processors` (**penalties applied before greedy**: `apply_repetition_penalties(logits, prompt_mask, output_mask, p)` = for every token id present in the prompt or in the output so far, `logit>0 ? logit/p : logit·p`) → `greedy_argmax`. Temperature 0 means argmax; top_k/top_p are not applied on the greedy path. The prompt mask includes the image placeholder ids and `[gMASK]<sop>`.
- Chat template (4606 bytes, identical in both snapshots): `[gMASK]<sop>` prefix; user turn renders content parts in order — image part → `\n<|begin_of_image|><|image|><|end_of_image|>`, text part → `\n{text}`; `add_generation_prompt` → `<|assistant|>`; `enable_thinking=false` appends `/nothink` and `<think></think>\n`. vLLM detects the "openai" content format for this template, so the parts reach Jinja as a list in the request order (image first, then `Text Recognition:` / `Table Recognition:` / `Formula Recognition:`).
- Stop: eos ids 59246, 59253; `finish_reason` `stop`/`length`; Semantik reads `choices[0].message.content` (trimmed; `null` = empty), warns on `finish_reason == "length"`, retries 429/5xx, classes other 4xx permanent.
- Metrics the sweep scripts scrape (`csweep2.sh`, `mstat.py`): `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`, `vllm:prompt_tokens_total`, `vllm:generation_tokens_total`, `vllm:request_prefill_time_seconds_sum`, `vllm:request_decode_time_seconds_sum`, `vllm:iteration_tokens_total_count`, `vllm:spec_decode_num_{draft,accepted}_tokens_total`, `vllm:spec_decode_num_accepted_tokens_per_pos_total{position=…}`. The engine exports the same names (aliased) so the existing sweep tooling works unchanged.

### 5 What the Semantik bridge depends on

`crates/semantik-pipeline/src/native_ocr/{prompt.rs,client.rs,crop.rs,serve.rs}` and `request_policy.rs`: `POST {endpoint}` (default `http://127.0.0.1:8002/v1/chat/completions`), body = one user message with `[image_url{data:image/jpeg;base64,…}, text]`, `model` = served name (`glm-ocr`), the sampling fields above; non-streaming JSON reply; optional bearer token; `Retry-After` honoured. The bridge gates concurrency per seat URL (`SeatGate`: halves on 429, regrows on successes; ceiling = `SEMANTIK_GLM_OCR_MAX_NUM_SEQS`, workers = `SEMANTIK_GLM_OCR_WORKERS`). Readiness is `GET /v1/models` listing the served name (`csweep2.sh`, `semantik-seats` `SeatStatus::ModelMissing`). Seat template: `seatconfig/templates/glm-ocr-vllm.start.sh` rendered by `semantik-seats::render` from `TEMPLATES[glm-ocr-vllm]` params (`container, image, model, served_name, port 8002, gpu_memory_utilization 0.15, max_num_seqs 32, max_model_len 16384, hf_cache_dir, allow_download 0, compile_cache_dir, load_strategy prefetch`); values must match `[A-Za-z0-9._:/@+=,-]`; `{{name}}` placeholders only; start script must return after launching, stop script kills by name. The sweep's seat ran `vllm/vllm-openai:v0.33.0` at share 0.60 (`ocr-start-s32.sh`).

---

### Tensor name patterns (every pattern in the header fixture)

Fixture: `tests/fixtures/weights/glm-ocr-header.json` (526 entries; name, dtype, shape, data_offsets). N is a layer or block index. Text layers N = 0..15; the MTP layer is N = 16 and additionally carries the roles marked (MTP).

| Pattern | Count | Shape |
|---|---|---|
| `lm_head.weight` | 1 | [59392, 1536] |
| `model.language_model.embed_tokens.weight` | 1 | [59392, 1536] |
| `model.language_model.layers.N.eh_proj.weight` (MTP) | 1 | [1536, 3072] |
| `model.language_model.layers.N.embed_tokens.weight` (MTP) | 1 | [59392, 1536] |
| `model.language_model.layers.N.enorm.weight` (MTP) | 1 | [1536] |
| `model.language_model.layers.N.hnorm.weight` (MTP) | 1 | [1536] |
| `model.language_model.layers.N.input_layernorm.weight` | 17 | [1536] |
| `model.language_model.layers.N.mlp.down_proj.weight` | 17 | [1536, 4608] |
| `model.language_model.layers.N.mlp.gate_up_proj.weight` | 17 | [9216, 1536] |
| `model.language_model.layers.N.post_attention_layernorm.weight` | 17 | [1536] |
| `model.language_model.layers.N.post_mlp_layernorm.weight` | 17 | [1536] |
| `model.language_model.layers.N.post_self_attn_layernorm.weight` | 17 | [1536] |
| `model.language_model.layers.N.self_attn.k_proj.weight` | 17 | [1024, 1536] |
| `model.language_model.layers.N.self_attn.o_proj.weight` | 17 | [1536, 2048] |
| `model.language_model.layers.N.self_attn.q_proj.weight` | 17 | [2048, 1536] |
| `model.language_model.layers.N.self_attn.v_proj.weight` | 17 | [1024, 1536] |
| `model.language_model.layers.N.shared_head.head.weight` (MTP) | 1 | [59392, 1536] |
| `model.language_model.layers.N.shared_head.norm.weight` (MTP) | 1 | [1536] |
| `model.language_model.norm.weight` | 1 | [1536] |
| `model.visual.blocks.N.attn.k_norm.weight` | 24 | [64] |
| `model.visual.blocks.N.attn.proj.bias` | 24 | [1024] |
| `model.visual.blocks.N.attn.proj.weight` | 24 | [1024, 1024] |
| `model.visual.blocks.N.attn.q_norm.weight` | 24 | [64] |
| `model.visual.blocks.N.attn.qkv.bias` | 24 | [3072] |
| `model.visual.blocks.N.attn.qkv.weight` | 24 | [3072, 1024] |
| `model.visual.blocks.N.mlp.down_proj.bias` | 24 | [1024] |
| `model.visual.blocks.N.mlp.down_proj.weight` | 24 | [1024, 4096] |
| `model.visual.blocks.N.mlp.gate_proj.bias` | 24 | [4096] |
| `model.visual.blocks.N.mlp.gate_proj.weight` | 24 | [4096, 1024] |
| `model.visual.blocks.N.mlp.up_proj.bias` | 24 | [4096] |
| `model.visual.blocks.N.mlp.up_proj.weight` | 24 | [4096, 1024] |
| `model.visual.blocks.N.norm1.weight` | 24 | [1024] |
| `model.visual.blocks.N.norm2.weight` | 24 | [1024] |
| `model.visual.downsample.bias` | 1 | [1536] |
| `model.visual.downsample.weight` | 1 | [1536, 1024, 2, 2] |
| `model.visual.merger.down_proj.weight` | 1 | [1536, 4608] |
| `model.visual.merger.gate_proj.weight` | 1 | [4608, 1536] |
| `model.visual.merger.post_projection_norm.bias` | 1 | [1536] |
| `model.visual.merger.post_projection_norm.weight` | 1 | [1536] |
| `model.visual.merger.proj.weight` | 1 | [1536, 1536] |
| `model.visual.merger.up_proj.weight` | 1 | [4608, 1536] |
| `model.visual.patch_embed.proj.bias` | 1 | [1024] |
| `model.visual.patch_embed.proj.weight` | 1 | [1024, 3, 2, 14, 14] |
| `model.visual.post_layernorm.weight` | 1 | [1024] |

The text checkpoint stores gate and up fused as `mlp.gate_up_proj` ([9216,1536]); the vision MLP and merger store `gate_proj` and `up_proj` separately. Vision `attn.qkv` is already fused. Text attention stores `q_proj`, `k_proj`, `v_proj` separately (fused to qkv by the loader).
