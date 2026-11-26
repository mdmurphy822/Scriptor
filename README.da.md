<p align="center"><img src="docs/assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*En nativ GLM-OCR-inferensserver til NVIDIA GB10: scanninger ind, genkendt tekst ud, via en OpenAI-kompatibel API og uden gemt varm tilstand.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Projektside](https://chironcourseworks.com/scriptor)** • [Kom godt i gang](#quick-start) • [Installation](docs/install.md) • [OCR-bro](docs/bridge.md) • [Benchmark](docs/benchmark.md) • [Dokumentation](#documentation) • [Bidrag](#contributing) • [Licens](#license)

[Sprog](docs/languages.md) • [English](README.md)

</div>

## Introduktion

Scriptor (latin for "skriver") er en throughput-optimeret, udelukkende lokal inferensmotor til GLM-OCR-modellen, skrevet i Rust med håndskrevne CUDA-kerner og bygget til NVIDIA GB10 i DGX Spark (ARM64). Den serverer GLM-OCR via et OpenAI-kompatibelt chat-completions-endpoint, modtager sidebilleder i forespørgslen og udstiller vLLM-kompatible `vllm:*` Prometheus-metrikker, så en klient eller et dashboard bygget til en vLLM-server fungerer uændret. Den er lokal og `fail-closed` (stopper i stedet for at gætte, når et input mangler eller er forkert): modelmappen og bind-adressen angives eksplicit, intet downloades, intet opdages automatisk, og et manglende, forældet eller uoverensstemmende input stopper opstarten i stedet for at blive gættet sig udenom. Se [docs/serving.md](docs/serving.md) for funktionslisten og kontrakten for nedbrudssikkerhed.

## Hovedresultater

Scriptor mod vLLM, samme model, dokumenter, OCR-bro-protokol og GPU: 10 parrede kørsler pr. motor, en frisk opstart til hver kørsel, 368 sider pr. kørsel, forhold med 95 % parrede bootstrap-konfidensintervaller.

| Metrik | Scriptor p50 | vLLM p50 | Dom |
| ------ | -----------: | -------: | ------- |
| Vægurstid, 368 sider (s) | 236.7 | 454.8 | SEJR (0.520x) |
| Gennemløb (sider/time) | 5,598 | 2,913 | SEJR (1.922x) |
| Opstart til klar, efter første opstart (s) | 14.25 | 154.3 | SEJR (0.092x) |
| Tid til første token (s) | 5.070 | 9.167 | SEJR (0.553x) |
| End-to-end-forespørgsel (s) | 8.339 | 13.29 | SEJR (0.627x) |

Én maskine, én GPU, én session, et fast sæt på 15 dokumenter; hver motor kørte sin egen leverede konfiguration. Opsætning, alle metrikker, nøjagtighed, begrænsninger og sådan reproducerer du resultaterne: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Kom godt i gang

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Når disse indstillinger er eksporteret, lytter motoren på 127.0.0.1:8002 og OCR-broen på 127.0.0.1:8765. Send et sidebillede til motoren som en billed-data-URL, og et helt dokument til broen:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` er en chat-completions-body, hvis brugerbesked indeholder en base64-kodet `image_url`-data-URL. Serveren downloader intet; `install.sh` er det eneste trin, der bruger netværk.

## Installation

`./install.sh` installerer release-binæren, GLM-OCR-checkpointet, layoutmodellen og en CUDA-build af onnxruntime til GB10, hvert trin er opt-in og checksum-verificeret; `./install.sh check` verificerer resultatet. Du kan også bygge fra kildekode. Se [docs/install.md](docs/install.md).

## OCR-bro

`scriptor serve` kører også en dokument-OCR-bro: PDF'er og billeder ind, genkendt tekst ud, med layoutgenkendelse via onnxruntime. Flag, indstillinger og den indbyggede brokontrol står i [docs/bridge.md](docs/bridge.md).

## Benchmark

Fulde tabeller, nøjagtighed, begrænsninger og reproduktionstrin: [docs/benchmark.md](docs/benchmark.md).

## Nedbrudssikkerhed

Afgrænset indtag, hukommelsestryksniveauer, overvågede workere og exitkoder, som en supervisor kan handle på: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Dokumentation

Dokumentation (på engelsk):

- [docs/install.md](docs/install.md) — installationsscript, forhåndsbygget release, bygning fra kildekode, model- og afhængighedslicenser
- [docs/serving.md](docs/serving.md) — `scriptor serve`-flag, forespørgselsgrænser, nedbrudssikkerhed
- [docs/bridge.md](docs/bridge.md) — OCR-broen, dens flag og indstillinger
- [docs/benchmark.md](docs/benchmark.md) — resultater, nøjagtighed, begrænsninger, reproduktion af benchmarken
- [docs/boot.md](docs/boot.md) — boot-hukommelsesplanen, opstartsfaserne og hvordan opstartstiden måles
- [docs/drafting.md](docs/drafting.md) — MTP-spekulativ afkodning, accept-politik og dens metrikker
- [docs/attention.md](docs/attention.md) — attention-kerner
- [docs/gemm.md](docs/gemm.md) — GEMM-kerner
- [docs/equivalence.md](docs/equivalence.md) og [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — ækvivalenstjek af kerner og end-to-end
- [docs/engine-file.md](docs/engine-file.md) — motorens native vægtfil
- [docs/model-facts.md](docs/model-facts.md) — de GLM-OCR-fakta, som motoren matcher
- [docs/vision-overlap.md](docs/vision-overlap.md) og [docs/nvjpeg.md](docs/nvjpeg.md) — vision-overlap og GPU-JPEG-afkodning
- [seat/README.md](seat/README.md) — `seat`-skabelonen, kontrakterne for klarhed og drain, exitkoder og fejlinjektion
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — referencemålingsprotokollen
- [docs/third-party-notices.md](docs/third-party-notices.md) — tredjepartsmeddelelser
- [docs/languages.md](docs/languages.md) — denne README på 38 sprog

<a id="contributing"></a>
## Bidrag

Læs først [CONTRIBUTING.md](CONTRIBUTING.md): Conventional Commit-emnelinjer, format-, clippy- og testgate, kun syntetiske fixtures. Rapportér sårbarheder privat som beskrevet i [SECURITY.md](SECURITY.md).

---

<a id="license"></a>
## Licens

Dobbeltlicenseret under AGPL-3.0-only eller en kommerciel licens (se
[LICENSE](LICENSE) og [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Copyright 2026 Matthew D. Murphy. Kommercielle licenser udstedes gennem
[Chiron Courseworks](https://chironcourseworks.com); henvendelser:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Tredjepartsmeddelelser findes i [docs/third-party-notices.md](docs/third-party-notices.md). Modelvægte og runtime-biblioteker er separate downloads og beholder deres egne licenser (se [Model and dependency licenses](docs/install.md#model-and-dependency-licenses)).
