# Engine file

The engine file is the converted, kernel-ready form of the GLM-OCR checkpoint.
`scriptor convert --model-dir <snapshot> --out <file>` is the only writer;
the engine never reads `*.safetensors` at serve time.

Code: `crates/scriptor-weights/src/{engine_file,repack,loader,prefetch}.rs`,
`crates/scriptor-cli/src/convert.rs`. Tests: `crates/scriptor-weights/tests/engine_file.rs`.

## Byte layout (little endian)

| bytes | content |
|---|---|
| 0..8 | magic `GLMOCRE\0` |
| 8..12 | layout version (u32, currently 1) |
| 12..16 | zero |
| 16..24 | header JSON length (u64) |
| 24..56 | SHA-256 of the header JSON |
| 56.. | header JSON, zero padded to a 4096 boundary |
| per tensor | data at its recorded offset, a multiple of 4096; gaps are zero |

Header JSON: `layout_version`, `revision` (the snapshot directory name),
`text_layers` (decoder layers excluding the MTP layer), `vision_blocks`,
`sources`, `tensors`, `file_len`.

`sources` has one record per input shard: `name`, `size`, `mtime_ns`,
`header_sha256` (SHA-256 of the safetensors length prefix plus header JSON) and
`sha256` (SHA-256 of the whole shard). `tensors` has one record per stored
tensor: `name`, `unit`, `dtype`, `shape`, `transform`, `sources` (checkpoint
names it derives from), `offset`, `len`.

## Stored layout

Element types are kept as stored (BF16). Every 2-D weight is row-major
`[out, in]`, which is the operand order of `C[m, n] = A[m, k] * W[n, k]^T`
(`docs/gemm.md`): cuBLASLt reads it without a transpose or copy.

| Engine tensor | Transform | Layout |
|---|---|---|
| `model.language_model.layers.N.self_attn.qkv_proj.weight` (every layer including the MTP layer) | concat | `q_proj`, `k_proj`, `v_proj` rows: `[Hq*D + 2*Hkv*D, hidden]` |
| `model.visual.blocks.N.mlp.gate_up_proj.{weight,bias}` | concat | `gate_proj` rows then `up_proj` rows, the `[rows, gate \| up]` layout `silu_mul_fused` reads |
| `model.visual.patch_embed.proj.weight` | reshape | `[hidden, 3*2*14*14]`, columns in the `(c, t, 14, 14)` patchify order |
| `model.visual.downsample.weight` | reshape | `[out, in*2*2]`, columns in `(channel, row, col)` order, matching `downsample_2x2` |
| every other tensor | copy | checkpoint name and shape unchanged (`mlp.gate_up_proj`, `o_proj`, norms, merger, embeddings, `lm_head`, MTP tensors) |

Fused and flattened tensors are byte concatenations of their sources, so the
file is at most one 4 KiB page per tensor plus the header larger than the
checkpoint.

## Load units and order

Entries are written in load order and tagged with a `unit`; consecutive entries
sharing a unit form one group: `vision.patch_embed`, `vision.block.N`,
`vision.adapter`, `text.embed`, `text.layer.N`, `text.head`, `mtp`.

## Conversion

`convert` maps the shard headers, infers the layer counts from the tensor
names, requires every tensor the layout implies and consumes every checkpoint
tensor; an unmapped, missing or unconsumed tensor refuses conversion. The file
is written to `<out>.partial` and renamed into place, so a failed conversion
leaves no engine file. The whole-shard SHA-256 is computed concurrently with the
write.

## Loading and verification

`EngineFile::open` memory-maps the file and refuses a bad magic, a layout
version other than the one this build reads, a header whose checksum does not
match, a total length other than the recorded `file_len`, and any tensor that is
misaligned, out of order or extends past the end (truncation).

`verify_source(<model-dir>)` compares each recorded shard's name, size,
modification time and header hash with the directory and refuses on any
difference, including added or missing shards. `verify_source_contents` also
re-hashes every shard against the recorded `sha256`.

`LayerPrefetcher` issues `posix_fadvise(WILLNEED)` for the current and next unit
and reads each unit with parallel positional reads (8 MiB segments shared by the
reader threads) into buffers obtained from a `Staging` implementation. The
default `HeapStaging` uses heap memory; the CUDA layer supplies pinned memory.
Finished units are queued up to a lookahead depth, so the consumer's
host-to-device copy of unit N overlaps the read of unit N+1.
