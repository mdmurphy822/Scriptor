<p align="center"><img src="../assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*面向 NVIDIA GB10 的原生 GLM-OCR 推理服务器:输入扫描件,输出识别文本,通过兼容 OpenAI 的 API 提供服务,不保存任何预热状态。*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[项目主页](https://chironcourseworks.com/scriptor)** • [快速开始](#quick-start) • [安装](../install.md) • [OCR 桥接](../bridge.md) • [基准测试](../benchmark.md) • [文档](#documentation) • [参与贡献](#contributing) • [许可证](#license)

[语言](../languages.md) • [English](../../README.md)

</div>

<a id="introduction"></a>
## 简介

Scriptor(拉丁语,意为“抄写员”)是一款面向吞吐量优化、仅限本地运行的 GLM-OCR 模型推理引擎,使用 Rust 编写,搭载手写 CUDA 内核,专为 DGX Spark(ARM64)中的 NVIDIA GB10 打造。它通过兼容 OpenAI 的 chat-completions 端点提供 GLM-OCR 服务,在请求中接收页面图像,并暴露与 vLLM 兼容的 `vllm:*` Prometheus 指标,因此为 vLLM 服务器构建的客户端或仪表盘无需修改即可使用。它仅在本地运行且采用 `fail-closed` 策略(输入缺失或有误时停止,而不是猜测):模型目录和绑定地址必须显式指定,不会下载任何内容,也不会自动发现任何资源;输入缺失、过期或不匹配时,启动会直接中止,而不是凭猜测绕过。功能列表和崩溃安全约定请参阅 [docs/serving.md](../serving.md)。

## 核心结果

Scriptor 与 vLLM 对比:相同的模型、文档、OCR 桥接协议和 GPU;每个引擎各进行 10 次配对运行,每次运行均重新启动,每次 368 页,比率附 95% 配对 bootstrap 置信区间。

| 指标 | Scriptor p50 | vLLM p50 | 结论 |
| ------ | -----------: | -------: | ------- |
| 耗时,368 页(秒) | 236.7 | 454.8 | 胜出 (0.520x) |
| 吞吐量(页/小时) | 5,598 | 2,913 | 胜出 (1.922x) |
| 启动至就绪,首次启动之后(秒) | 14.25 | 154.3 | 胜出 (0.092x) |
| 首个 token 时间(秒) | 5.070 | 9.167 | 胜出 (0.553x) |
| 端到端请求(秒) | 8.339 | 13.29 | 胜出 (0.627x) |

单台机器、单块 GPU、单次会话、固定的 15 份文档;每个引擎均使用其自身发布的配置。测试环境、全部指标、准确率、局限性及复现方法:[docs/benchmark.md](../benchmark.md)。

<a id="quick-start"></a>
## 快速开始

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

按上述设置导出后,引擎监听 127.0.0.1:8002,OCR 桥接监听 127.0.0.1:8765。将页面图像以图像 data URL 的形式发送给引擎,将整份文档发送给桥接:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` 是一个 chat-completions 请求体,其用户消息中携带 base64 编码的 `image_url` data URL。服务器不会下载任何内容;`install.sh` 是唯一需要联网的步骤。

## 安装

`./install.sh` 会安装发布版二进制文件、GLM-OCR 检查点、版面模型,以及适用于 GB10 的 CUDA 版 onnxruntime,每一步均需主动选择并经过校验和验证;`./install.sh check` 用于验证安装结果。你也可以从源码构建。参阅 [docs/install.md](../install.md)。

## OCR 桥接

`scriptor serve` 同时运行文档 OCR 桥接:输入 PDF 和图像,输出识别文本,并通过 onnxruntime 进行版面检测。参数、设置及内置的桥接检查请参阅 [docs/bridge.md](../bridge.md)。

## 基准测试

完整表格、准确率、局限性和复现步骤:[docs/benchmark.md](../benchmark.md)。

## 崩溃安全

有界的请求接入、内存压力分级、受监督的工作进程,以及供监督程序据以处理的退出码:[docs/serving.md](../serving.md#crash-safety)。

<a id="documentation"></a>
## 文档

文档(英文):

- [docs/install.md](../install.md) — 安装脚本、预构建发布版、从源码构建、模型与依赖许可证
- [docs/serving.md](../serving.md) — `scriptor serve` 参数、请求限制、崩溃安全
- [docs/bridge.md](../bridge.md) — OCR 桥接及其参数和设置
- [docs/benchmark.md](../benchmark.md) — 结果、准确率、局限性、复现基准测试
- [docs/boot.md](../boot.md) — 启动内存规划、启动阶段及启动时间的测量方式
- [docs/drafting.md](../drafting.md) — MTP 推测解码、接受策略及其指标
- [docs/attention.md](../attention.md) — 注意力内核
- [docs/gemm.md](../gemm.md) — GEMM 内核
- [docs/equivalence.md](../equivalence.md) 和 [docs/e2e-equivalence.md](../e2e-equivalence.md) — 内核及端到端等价性检查
- [docs/engine-file.md](../engine-file.md) — 引擎原生权重文件
- [docs/model-facts.md](../model-facts.md) — 引擎所匹配的 GLM-OCR 模型事实
- [docs/vision-overlap.md](../vision-overlap.md) 和 [docs/nvjpeg.md](../nvjpeg.md) — 视觉重叠与 GPU JPEG 解码
- [seat/README.md](../../seat/README.md) — `seat` 模板、就绪与排空约定、退出码及故障注入
- [docs/phase0-protocol.md](../phase0-protocol.md) — 参考测量协议
- [docs/third-party-notices.md](../third-party-notices.md) — 第三方声明
- [docs/languages.md](../languages.md) — 本 README 的 38 种语言版本

<a id="contributing"></a>
## 参与贡献

请先阅读 [CONTRIBUTING.md](../../CONTRIBUTING.md):Conventional Commit 提交标题、格式化、clippy 与测试关卡、仅使用合成测试数据。安全漏洞请按 [SECURITY.md](../../SECURITY.md) 所述私下报告。

---

<a id="license"></a>
## 许可证

采用 AGPL-3.0-only 或商业许可证双重许可(参见
[LICENSE](../../LICENSE) 和 [COMMERCIAL-LICENSE.md](../../COMMERCIAL-LICENSE.md))。
版权所有 2026 Matthew D. Murphy。商业许可证通过
[Chiron Courseworks](https://chironcourseworks.com) 颁发;咨询邮箱:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com)。

第三方声明见 [docs/third-party-notices.md](../third-party-notices.md)。模型权重和运行时库需单独下载,并保留各自的许可证(参见[模型与依赖许可证](../install.md#model-and-dependency-licenses))。
