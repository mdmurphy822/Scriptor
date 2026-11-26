<p align="center"><img src="../assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*Um servidor nativo de inferência do GLM-OCR para a NVIDIA GB10: entram digitalizações, sai texto reconhecido, por uma API compatível com a OpenAI, sem nenhum estado aquecido salvo.*

[![Licença](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Plataforma](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Página do projeto](https://chironcourseworks.com/scriptor)** • [Início rápido](#quick-start) • [Instalação](../install.md) • [Bridge de OCR](../bridge.md) • [Benchmark](../benchmark.md) • [Documentação](#documentation) • [Contribuindo](#contributing) • [Licença](#license)

[Idiomas](../languages.md) • [English](../../README.md)

</div>

## Introdução

Scriptor (do latim, "escriba") é um mecanismo de inferência local, otimizado para throughput, do modelo GLM-OCR, escrito em Rust com kernels CUDA feitos à mão e construído para a NVIDIA GB10 do DGX Spark (ARM64). Ele serve o GLM-OCR por um endpoint de chat completions compatível com a OpenAI, aceita imagens de página na requisição e expõe métricas Prometheus `vllm:*` compatíveis com o vLLM, de modo que um cliente ou dashboard feito para um servidor vLLM funciona sem alterações. Ele é local e `fail-closed` (para em vez de adivinhar quando uma entrada está ausente ou incorreta): o diretório do modelo e o endereço de bind são explícitos, nada é baixado, nada é descoberto automaticamente, e uma entrada ausente, desatualizada ou incompatível interrompe a inicialização em vez de ser contornada por suposição. Veja [docs/serving.md](../serving.md) para a lista de recursos e o contrato de segurança contra falhas.

## Principais resultados

Scriptor contra o vLLM, com o mesmo modelo, os mesmos documentos, o mesmo protocolo da bridge de OCR e a mesma GPU: 10 execuções pareadas por mecanismo, um boot novo a cada execução, 368 páginas por execução, razões com intervalos de confiança de 95% por bootstrap pareado.

| Métrica | Scriptor p50 | vLLM p50 | Veredito |
| ------- | -----------: | -------: | -------- |
| Tempo total, 368 páginas (s) | 236.7 | 454.8 | VITÓRIA (0.520x) |
| Throughput (páginas/hora) | 5,598 | 2,913 | VITÓRIA (1.922x) |
| Boot até ficar pronto, após o primeiro boot (s) | 14.25 | 154.3 | VITÓRIA (0.092x) |
| Tempo até o primeiro token (s) | 5.070 | 9.167 | VITÓRIA (0.553x) |
| Requisição de ponta a ponta (s) | 8.339 | 13.29 | VITÓRIA (0.627x) |

Uma máquina, uma GPU, uma sessão, um conjunto fixo de 15 documentos; cada mecanismo rodou a própria configuração distribuída. Configuração, todas as métricas, acurácia, limitações e como reproduzir: [docs/benchmark.md](../benchmark.md).

<a id="quick-start"></a>
## Início rápido

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Com essas configurações exportadas, o mecanismo escuta em 127.0.0.1:8002 e a bridge de OCR em 127.0.0.1:8765. Envie uma imagem de página ao mecanismo como data URL de imagem, e um documento inteiro à bridge:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` é um corpo de chat completions cuja mensagem do usuário carrega um data URL `image_url` em base64. O servidor não baixa nada; o `install.sh` é a única etapa que usa a rede.

## Instalação

O `./install.sh` instala o binário de release, o checkpoint do GLM-OCR, o modelo de layout e um build CUDA do onnxruntime para a GB10, cada etapa opcional e verificada por checksum; `./install.sh check` confere o resultado. Também é possível compilar a partir do código-fonte. Veja [docs/install.md](../install.md).

## Bridge de OCR

O `scriptor serve` também executa uma bridge de OCR de documentos: entram PDFs e imagens, sai texto reconhecido, com detecção de layout via onnxruntime. Flags, configurações e a verificação embutida da bridge estão em [docs/bridge.md](../bridge.md).

## Benchmark

Tabelas completas, acurácia, limitações e passos de reprodução: [docs/benchmark.md](../benchmark.md).

## Segurança contra falhas

Entrada limitada, níveis de pressão de memória, workers supervisionados e códigos de saída sobre os quais um supervisor consegue agir: [docs/serving.md](../serving.md#crash-safety).

<a id="documentation"></a>
## Documentação

Documentação (em inglês):

- [docs/install.md](../install.md) — script de instalação, release pré-compilado, build a partir do código-fonte, licenças de modelos e dependências
- [docs/serving.md](../serving.md) — flags do `scriptor serve`, limites de requisição, segurança contra falhas
- [docs/bridge.md](../bridge.md) — a bridge de OCR, suas flags e configurações
- [docs/benchmark.md](../benchmark.md) — resultados, acurácia, limitações, como reproduzir o benchmark
- [docs/boot.md](../boot.md) — o plano de memória do boot, as etapas do boot e como o tempo de boot é medido
- [docs/drafting.md](../drafting.md) — decodificação especulativa MTP, política de aceitação e suas métricas
- [docs/attention.md](../attention.md) — kernels de atenção
- [docs/gemm.md](../gemm.md) — kernels GEMM
- [docs/equivalence.md](../equivalence.md) e [docs/e2e-equivalence.md](../e2e-equivalence.md) — verificações de equivalência de kernel e de ponta a ponta
- [docs/engine-file.md](../engine-file.md) — o arquivo de pesos nativo do mecanismo
- [docs/model-facts.md](../model-facts.md) — os fatos do GLM-OCR que o mecanismo reproduz
- [docs/vision-overlap.md](../vision-overlap.md) e [docs/nvjpeg.md](../nvjpeg.md) — sobreposição de visão e decodificação JPEG na GPU
- [seat/README.md](../../seat/README.md) — o template `seat`, os contratos de prontidão e de drenagem, códigos de saída e injeção de falhas
- [docs/phase0-protocol.md](../phase0-protocol.md) — o protocolo de medição de referência
- [docs/third-party-notices.md](../third-party-notices.md) — avisos de terceiros
- [docs/languages.md](../languages.md) — este README em 38 idiomas

<a id="contributing"></a>
## Contribuindo

Leia primeiro o [CONTRIBUTING.md](../../CONTRIBUTING.md): assuntos de commit no padrão Conventional Commits, o gate de format, clippy e testes, somente fixtures sintéticas. Relate vulnerabilidades em particular, conforme descrito no [SECURITY.md](../../SECURITY.md).

---

<a id="license"></a>
## Licença

Licenciado em dupla licença: AGPL-3.0-only ou uma licença comercial (veja
[LICENSE](../../LICENSE) e [COMMERCIAL-LICENSE.md](../../COMMERCIAL-LICENSE.md)).
Copyright 2026 Matthew D. Murphy. As licenças comerciais são emitidas pela
[Chiron Courseworks](https://chironcourseworks.com); consultas:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Os avisos de terceiros estão em [docs/third-party-notices.md](../third-party-notices.md). Os pesos do modelo e as bibliotecas de runtime são downloads separados e mantêm as próprias licenças (veja [Licenças de modelos e dependências](../install.md#model-and-dependency-licenses)).
