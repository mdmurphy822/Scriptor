<div align="center">
<pre>
███████╗ ██████╗██████╗ ██╗██████╗ ████████╗ ██████╗ ██████╗ 
██╔════╝██╔════╝██╔══██╗██║██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗
███████╗██║     ██████╔╝██║██████╔╝   ██║   ██║   ██║██████╔╝
╚════██║██║     ██╔══██╗██║██╔═══╝    ██║   ██║   ██║██╔══██╗
███████║╚██████╗██║  ██║██║██║        ██║   ╚██████╔╝██║  ██║
╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝╚═╝        ╚═╝    ╚═════╝ ╚═╝  ╚═╝
</pre>
</div>

<div align="center">

*Um servidor nativo de inferência do GLM-OCR para a NVIDIA GB10: entram digitalizações, sai texto reconhecido, através de uma API compatível com a OpenAI, sem qualquer estado pré-aquecido guardado.*

[![Licença](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Plataforma](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Página do projeto](https://chironcourseworks.com/scriptor)** • [Início rápido](#quick-start) • [Instalação](docs/install.md) • [Bridge de OCR](docs/bridge.md) • [Benchmark](docs/benchmark.md) • [Documentação](#documentation) • [Contribuir](#contributing) • [Licença](#license)

[Idiomas](docs/languages.md) • [English](README.md)

</div>

## Introdução

O Scriptor (do latim, "escriba") é um motor de inferência local, otimizado para débito (throughput), do modelo GLM-OCR, escrito em Rust com kernels CUDA feitos à mão e concebido para a NVIDIA GB10 do DGX Spark (ARM64). Serve o GLM-OCR através de um endpoint de chat completions compatível com a OpenAI, aceita imagens de página no pedido e expõe métricas Prometheus `vllm:*` compatíveis com o vLLM, pelo que um cliente ou painel de controlo concebido para um servidor vLLM funciona sem alterações. É local e `fail-closed` (pára em vez de adivinhar quando uma entrada está em falta ou incorreta): a pasta do modelo e o endereço de bind são explícitos, nada é descarregado, nada é descoberto automaticamente, e uma entrada em falta, desatualizada ou incompatível interrompe o arranque em vez de ser contornada por dedução. Consulte [docs/serving.md](docs/serving.md) para a lista de funcionalidades e o contrato de segurança contra falhas.

## Principais resultados

O Scriptor face ao vLLM, com o mesmo modelo, os mesmos documentos, o mesmo protocolo da bridge de OCR e a mesma GPU: 10 execuções emparelhadas por motor, um arranque novo em cada execução, 368 páginas por execução, rácios com intervalos de confiança de 95% por bootstrap emparelhado.

| Métrica | Scriptor p50 | vLLM p50 | Veredicto |
| ------- | -----------: | -------: | --------- |
| Tempo total, 368 páginas (s) | 236.7 | 454.8 | VITÓRIA (0.520x) |
| Débito (páginas/hora) | 5,598 | 2,913 | VITÓRIA (1.922x) |
| Arranque até estar pronto, após o primeiro arranque (s) | 14.25 | 154.3 | VITÓRIA (0.092x) |
| Tempo até ao primeiro token (s) | 5.070 | 9.167 | VITÓRIA (0.553x) |
| Pedido de ponta a ponta (s) | 8.339 | 13.29 | VITÓRIA (0.627x) |

Uma máquina, uma GPU, uma sessão, um conjunto fixo de 15 documentos; cada motor correu a sua própria configuração distribuída. Configuração, todas as métricas, precisão, limitações e como reproduzir: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Início rápido

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Com essas definições exportadas, o motor escuta em 127.0.0.1:8002 e a bridge de OCR em 127.0.0.1:8765. Envie uma imagem de página ao motor como data URL de imagem, e um documento inteiro à bridge:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

O `request.json` é um corpo de chat completions cuja mensagem do utilizador transporta um data URL `image_url` em base64. O servidor não descarrega nada; o `install.sh` é o único passo que usa a rede.

## Instalação

O `./install.sh` instala o binário de release, o checkpoint do GLM-OCR, o modelo de layout e uma compilação CUDA do onnxruntime para a GB10, sendo cada passo opcional e verificado por checksum; o `./install.sh check` confirma o resultado. Também pode compilar a partir do código-fonte. Consulte [docs/install.md](docs/install.md).

## Bridge de OCR

O `scriptor serve` executa também uma bridge de OCR de documentos: entram PDF e imagens, sai texto reconhecido, com deteção de layout através do onnxruntime. As flags, as definições e a verificação integrada da bridge estão em [docs/bridge.md](docs/bridge.md).

## Benchmark

Tabelas completas, precisão, limitações e passos de reprodução: [docs/benchmark.md](docs/benchmark.md).

## Segurança contra falhas

Entrada limitada, escalões de pressão de memória, workers supervisionados e códigos de saída sobre os quais um supervisor consegue atuar: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Documentação

Documentação (em inglês):

- [docs/install.md](docs/install.md) — script de instalação, release pré-compilada, compilação a partir do código-fonte, licenças de modelos e dependências
- [docs/serving.md](docs/serving.md) — flags do `scriptor serve`, limites de pedidos, segurança contra falhas
- [docs/bridge.md](docs/bridge.md) — a bridge de OCR, as suas flags e definições
- [docs/benchmark.md](docs/benchmark.md) — resultados, precisão, limitações, como reproduzir o benchmark
- [docs/boot.md](docs/boot.md) — o plano de memória do arranque, as etapas do arranque e como o tempo de arranque é medido
- [docs/drafting.md](docs/drafting.md) — descodificação especulativa MTP, política de aceitação e as suas métricas
- [docs/attention.md](docs/attention.md) — kernels de atenção
- [docs/gemm.md](docs/gemm.md) — kernels GEMM
- [docs/equivalence.md](docs/equivalence.md) e [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — verificações de equivalência de kernel e de ponta a ponta
- [docs/engine-file.md](docs/engine-file.md) — o ficheiro de pesos nativo do motor
- [docs/model-facts.md](docs/model-facts.md) — os factos do GLM-OCR que o motor reproduz
- [docs/vision-overlap.md](docs/vision-overlap.md) e [docs/nvjpeg.md](docs/nvjpeg.md) — sobreposição de visão e descodificação JPEG na GPU
- [seat/README.md](seat/README.md) — o modelo `seat`, os contratos de prontidão e de escoamento, códigos de saída e injeção de falhas
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — o protocolo de medição de referência
- [docs/third-party-notices.md](docs/third-party-notices.md) — avisos de terceiros
- [docs/languages.md](docs/languages.md) — este README em 38 idiomas

<a id="contributing"></a>
## Contribuir

Leia primeiro o [CONTRIBUTING.md](CONTRIBUTING.md): assuntos de commit no formato Conventional Commits, o gate de formatação, clippy e testes, apenas fixtures sintéticas. Comunique vulnerabilidades em privado, conforme descrito no [SECURITY.md](SECURITY.md).

---

<a id="license"></a>
## Licença

Licenciado em licença dupla: AGPL-3.0-only ou uma licença comercial (consulte
[LICENSE](LICENSE) e [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Copyright 2026 Matthew D. Murphy. As licenças comerciais são emitidas pela
[Chiron Courseworks](https://chironcourseworks.com); pedidos de informação:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Os avisos de terceiros encontram-se em [docs/third-party-notices.md](docs/third-party-notices.md). Os pesos do modelo e as bibliotecas de runtime são transferências separadas e mantêm as suas próprias licenças (consulte [Licenças de modelos e dependências](docs/install.md#model-and-dependency-licenses)).
