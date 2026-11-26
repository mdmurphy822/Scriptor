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

*En inbyggd GLM-OCR-inferensserver för NVIDIA GB10: skanningar in, igenkänd text ut, via ett OpenAI-kompatibelt API och utan sparat varmt tillstånd.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Projektsida](https://chironcourseworks.com/scriptor)** • [Kom igång](#quick-start) • [Installation](docs/install.md) • [OCR-brygga](docs/bridge.md) • [Benchmark](docs/benchmark.md) • [Dokumentation](#documentation) • [Bidra](#contributing) • [Licens](#license)

[Språk](docs/languages.md) • [English](README.md)

</div>

## Introduktion

Scriptor (latin för "skrivare") är en throughput-optimerad, enbart lokal inferensmotor för GLM-OCR-modellen, skriven i Rust med handskrivna CUDA-kärnor och byggd för NVIDIA GB10 i DGX Spark (ARM64). Den serverar GLM-OCR via en OpenAI-kompatibel chat-completions-endpoint, tar emot sidbilder i anropet och exponerar vLLM-kompatibla `vllm:*` Prometheus-mätvärden, så en klient eller en dashboard byggd för en vLLM-server fungerar oförändrad. Den är lokal och `fail-closed` (stoppar i stället för att gissa när indata saknas eller är fel): modellkatalogen och bind-adressen anges uttryckligen, inget laddas ned, inget upptäcks automatiskt, och saknad, föråldrad eller felaktig indata stoppar starten i stället för att gissas runt. Se [docs/serving.md](docs/serving.md) för funktionslistan och kontraktet för krasch-säkerhet.

## Huvudresultat

Scriptor mot vLLM, samma modell, dokument, OCR-bryggprotokoll och GPU: 10 parade körningar per motor, en ny uppstart för varje körning, 368 sidor per körning, kvoter med 95 % parade bootstrap-konfidensintervall.

| Mått | Scriptor p50 | vLLM p50 | Utlåtande |
| ---- | -----------: | -------: | --------- |
| Väggklocketid, 368 sidor (s) | 236.7 | 454.8 | VINST (0.520x) |
| Genomströmning (sidor/timme) | 5,598 | 2,913 | VINST (1.922x) |
| Uppstart till redo, efter första uppstarten (s) | 14.25 | 154.3 | VINST (0.092x) |
| Tid till första token (s) | 5.070 | 9.167 | VINST (0.553x) |
| End-to-end-anrop (s) | 8.339 | 13.29 | VINST (0.627x) |

En maskin, en GPU, en session, en fast uppsättning om 15 dokument; varje motor kördes med sin egen levererade konfiguration. Uppsättning, alla mätvärden, noggrannhet, begränsningar och hur du reproducerar resultaten: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Kom igång

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

När dessa inställningar är exporterade lyssnar motorn på 127.0.0.1:8002 och OCR-bryggan på 127.0.0.1:8765. Skicka en sidbild till motorn som en bild-data-URL, och ett helt dokument till bryggan:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` är en chat-completions-body vars användarmeddelande innehåller en base64-kodad `image_url`-data-URL. Servern laddar inte ned något; `install.sh` är det enda steget som använder nätverket.

## Installation

`./install.sh` installerar release-binären, GLM-OCR-checkpointen, layoutmodellen och ett CUDA-bygge av onnxruntime för GB10, varje steg är opt-in och checksummeverifierat; `./install.sh check` verifierar resultatet. Du kan också bygga från källkod. Se [docs/install.md](docs/install.md).

## OCR-brygga

`scriptor serve` kör också en dokument-OCR-brygga: PDF-filer och bilder in, igenkänd text ut, med layoutdetektering via onnxruntime. Flaggor, inställningar och den inbyggda bryggkontrollen finns i [docs/bridge.md](docs/bridge.md).

## Benchmark

Fullständiga tabeller, noggrannhet, begränsningar och steg för att reproducera: [docs/benchmark.md](docs/benchmark.md).

## Krasch-säkerhet

Begränsat intag, minnestrycksnivåer, övervakade workers och exitkoder som en supervisor kan agera på: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Dokumentation

Dokumentation (på engelska):

- [docs/install.md](docs/install.md) — installationsskript, förbyggd release, bygge från källkod, licenser för modeller och beroenden
- [docs/serving.md](docs/serving.md) — `scriptor serve`-flaggor, begäransgränser, krasch-säkerhet
- [docs/bridge.md](docs/bridge.md) — OCR-bryggan, dess flaggor och inställningar
- [docs/benchmark.md](docs/benchmark.md) — resultat, noggrannhet, begränsningar, att reproducera benchmarken
- [docs/boot.md](docs/boot.md) — boot-minnesplanen, uppstartsstegen och hur uppstartstiden mäts
- [docs/drafting.md](docs/drafting.md) — MTP-spekulativ avkodning, accepteringspolicy och dess mätvärden
- [docs/attention.md](docs/attention.md) — attention-kärnor
- [docs/gemm.md](docs/gemm.md) — GEMM-kärnor
- [docs/equivalence.md](docs/equivalence.md) och [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — ekvivalenskontroller för kärnor och end-to-end
- [docs/engine-file.md](docs/engine-file.md) — motorns inbyggda viktfil
- [docs/model-facts.md](docs/model-facts.md) — de GLM-OCR-fakta som motorn matchar
- [docs/vision-overlap.md](docs/vision-overlap.md) och [docs/nvjpeg.md](docs/nvjpeg.md) — vision-överlapp och GPU-JPEG-avkodning
- [seat/README.md](seat/README.md) — `seat`-mallen, kontrakten för beredskap och drain, exitkoder och felinjektion
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — referensmätningsprotokollet
- [docs/third-party-notices.md](docs/third-party-notices.md) — tredjepartsmeddelanden
- [docs/languages.md](docs/languages.md) — denna README på 38 språk

<a id="contributing"></a>
## Bidra

Läs [CONTRIBUTING.md](CONTRIBUTING.md) först: Conventional Commit-ämnesrader, format-, clippy- och testgrind, endast syntetiska fixtures. Rapportera sårbarheter privat enligt beskrivningen i [SECURITY.md](SECURITY.md).

---

<a id="license"></a>
## Licens

Dubbellicensierad under AGPL-3.0-only eller en kommersiell licens (se
[LICENSE](LICENSE) och [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Copyright 2026 Matthew D. Murphy. Kommersiella licenser utfärdas via
[Chiron Courseworks](https://chironcourseworks.com); förfrågningar:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Tredjepartsmeddelanden finns i [docs/third-party-notices.md](docs/third-party-notices.md). Modellvikter och runtime-bibliotek är separata nedladdningar och behåller sina egna licenser (se [Model and dependency licenses](docs/install.md#model-and-dependency-licenses)).
