# Test fixtures

Synthetic and configuration-only fixtures. No model weights are stored here.

## config/

- `glm-ocr.config.json`, `glm-ocr.preprocessor_config.json`, `glm-ocr.processor_config.json`, `glm-ocr.generation_config.json`: the GLM-OCR configuration files, copied verbatim from checkpoint snapshot 2e85a62.
- `tiny.config.json`, `tiny.preprocessor_config.json`: synthetic configuration with the same layout as the real files (specified below; authored with the spec crate).

## chat/

- `chat_template.jinja`: the GLM-OCR chat template, verbatim.

## weights/

- `glm-ocr-header.json`: the safetensors header of the checkpoint reduced to 526 entries of `name`, `dtype`, `shape`, `data_offsets`, ordered by offset. Used to check tensor-name classification without weights.

## Tiny synthetic model

| Field | Value |
|---|---|
| vision depth | 2 |
| vision hidden | 32 |
| vision heads | 2 (head_dim 16) |
| vision intermediate | 64 |
| vision out_hidden_size | 48 |
| patch_size | 14 |
| spatial_merge_size | 2 |
| text hidden | 48 |
| text layers | 2 |
| text attention heads | 4, head_dim 12 |
| text KV heads | 2 |
| text intermediate | 96 |
| vocab | 512 |
| mrope_section | [2, 2, 2] |
| MTP | 1 layer (num_nextn_predict_layers 1) |
