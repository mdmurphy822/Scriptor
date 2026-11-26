# Install

Three routes: the install script, a prebuilt release, or building from source. Back to the [README](../README.md).

## Install script

`install.sh` at the repository root fetches Scriptor's dependencies. It is the only networked step: the server itself downloads nothing at runtime. Every step is opt-in and safe to repeat, every download is https, checksums are verified where one exists, and files land in a temporary directory under the prefix before an atomic move.

```bash
./install.sh all          # binary, model, layout, onnxruntime
./install.sh env          # print the environment lines and the serve command
./install.sh check        # driver, GPU, files, checksums, scriptor --version
./install.sh --dry-run all
```

Defaults: data under `~/.local/share/scriptor` (`--prefix`), the binary in `~/.local/bin` (`--bindir`).

| Command | What it does |
| ------- | ------------ |
| `binary` | Downloads `scriptor-v<version>-aarch64-linux-cuda.tar.gz` and its `.sha256` from the [releases page](https://github.com/mdmurphy822/Scriptor/releases) (latest, or `--version`), verifies the digest and installs `scriptor`. Refuses on a host that is not aarch64 Linux. |
| `model` | Fetches `zai-org/GLM-OCR` at revision `2e85a62840ccac27daa451df36c736c4636b8628` into `<prefix>/models/glm-ocr`, using `hf download` or `huggingface-cli download` when present, then `git lfs`, then plain `curl` per file through the Hugging Face API (large files are checked against their recorded sha256). |
| `layout` | Downloads `pp-doclayoutv3-e4f489dc.onnx` from the v0.1.0 release, verifies sha256 `6a99236055d3efed8b5bb9203ebed4c7ed36836d0259add97ffc884adf4346ca` and writes a `.sha256` sidecar the bridge also checks. It is an ONNX export of [PaddlePaddle/PP-DocLayoutV3_safetensors](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors) at revision `e4f489dc536556fc1be02b973ba06756e0d4a2ac` (Apache-2.0), the layout model the published benchmark ran with. |
| `onnxruntime` | Downloads `onnxruntime-1.28.0-linux-aarch64-cuda13-sm121.tar.gz` from the v0.1.0 release and verifies sha256 `1bc031df14da045dddf3e12f1c01afb95a39c08917233179855301fb152009b4`: ONNX Runtime 1.28.0 built from the upstream `v1.28.0` tag with the CUDA execution provider for CUDA 13 and the GB10 (compute capability 12.1), the build the benchmark ran with. `--ort-cpu` installs the official CPU tarball from the microsoft/onnxruntime releases instead. |
| `all` | Runs the four steps above, then prints the environment. |
| `env` | Prints `SCRIPTOR_BRIDGE_LAYOUT_RUNTIME`, `SCRIPTOR_BRIDGE_LAYOUT_WEIGHTS` and the `scriptor serve --model ...` command for the installed paths. |
| `check` | Checks `nvidia-smi`, the GPU name, the installed files, the layout checksum and `scriptor --version`. |

Options: `--prefix DIR`, `--bindir DIR`, `--version VER`, `--ort-cpu`, `--ort-version VER`, `--ort-url URL`, `--ort-sha256 HEX`, `--ort-lib PATH`, `--dry-run`, `--help`.

### onnxruntime and the GPU

The bridge loads onnxruntime dynamically (`ort` 2.0.0-rc.13, `load-dynamic`, API level 27). The microsoft/onnxruntime releases do not include an aarch64 build with the CUDA execution provider, so the Scriptor release carries one. It needs the NVIDIA driver and the CUDA 13 runtime libraries `libcudart.so.13`, `libcublas.so.13` and `libcublasLt.so.13` on the library path; the tarball's `BUILD-INFO.txt` records the build configuration, and its `LICENSE` and `ThirdPartyNotices.txt` are ONNX Runtime's. With `--ort-cpu` the layout step runs on the CPU (`SCRIPTOR_BRIDGE_LAYOUT_DEVICE=cpu`, which `install.sh env` prints). To use another build, pass `--ort-url <https url> --ort-sha256 <hex>`, or point at an existing library with `--ort-lib PATH`.

`tools/gate/test_install.sh` checks the script offline (help, dry-run plan, refusal on a non-aarch64 host).

## Prebuilt release

The [releases page](https://github.com/mdmurphy822/Scriptor/releases) has a prebuilt aarch64 Linux CUDA build for the NVIDIA GB10 (DGX Spark); `./install.sh binary` installs it.

## Build from source

Building from source needs Rust 1.85 or newer (the workspace uses edition 2024; `rust-toolchain.toml` selects the stable channel through rustup), the CUDA 13 toolkit with `nvcc` (kernels are compiled for `sm_121`), and a GLM-OCR checkpoint snapshot directory you supply.

```bash
cargo build --release --locked --workspace
# one binary, target/release/scriptor, runs the engine and the OCR bridge
```

Kernel fatbins compile in parallel and are cached by content in `$SCRIPTOR_KERNEL_CACHE` (default `~/.cache/scriptor/kernels`, `off` disables); `cargo test --profile gpu-test` skips LTO for faster GPU-test builds.

## Model and dependency licenses

The scriptor server never downloads models or runtime libraries; `install.sh` fetches them when you run it. Each keeps its own license, which applies to your use of it.

| Component | Where it comes from | License |
| --------- | ------------------- | ------- |
| GLM-OCR model weights | `zai-org/GLM-OCR` | See the upstream model card |
| PP-DocLayoutV3 layout model (ONNX) | ONNX export of `PaddlePaddle/PP-DocLayoutV3_safetensors`, attached to the v0.1.0 release with `pp-doclayoutv3-NOTICE.txt` | Apache-2.0 |
| onnxruntime shared library | ONNX Runtime 1.28.0 CUDA build attached to the v0.1.0 release, or the upstream release you install | MIT (`LICENSE` and `ThirdPartyNotices.txt` in the tarball) |

The license of the weights you use is not changed by this repository's AGPL-3.0-only or commercial license, which covers Scriptor's own code.
