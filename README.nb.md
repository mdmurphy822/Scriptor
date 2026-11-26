<p align="center"><img src="docs/assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>


<div align="center">

*En innebygd GLM-OCR-inferensserver for NVIDIA GB10: skanninger inn, gjenkjent tekst ut, over et OpenAI-kompatibelt API, uten lagret varm tilstand.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Prosjektside](https://chironcourseworks.com/scriptor)** • [Hurtigstart](#quick-start) • [Installasjon](docs/install.md) • [OCR-bro](docs/bridge.md) • [Ytelsestest](docs/benchmark.md) • [Dokumentasjon](#documentation) • [Bidra](#contributing) • [Lisens](#license)

[Språk](docs/languages.md) • [English](README.md)

</div>

## Introduksjon

Scriptor (latin for «skriver») er en gjennomstrømningsoptimalisert, kun lokal inferensmotor for modellen GLM-OCR, skrevet i Rust med håndskrevne CUDA-kjerner og bygget for NVIDIA GB10 i DGX Spark (ARM64). Den serverer GLM-OCR gjennom et OpenAI-kompatibelt endepunkt for chat-completions, tar imot sidebilder i forespørselen og eksponerer vLLM-kompatible `vllm:*`-metrikker for Prometheus, slik at en klient eller et dashbord bygget for en vLLM-server fungerer uendret. Den er lokal og `fail-closed` (stopper i stedet for å gjette når en inndata mangler eller er feil): modellkatalogen og bindeadressen er eksplisitte, ingenting lastes ned, ingenting oppdages, og en manglende, utdatert eller ikke-samsvarende inndata stopper oppstarten i stedet for å bli gjettet seg rundt. Se [docs/serving.md](docs/serving.md) for funksjonslisten og krasjsikkerhetskontrakten.

## Hovedresultater

Scriptor mot vLLM, samme modell, samme dokumenter, samme OCR-bro-protokoll og samme GPU: 10 parvise kjøringer per motor, en ny oppstart for hver kjøring, 368 sider per kjøring, forholdstall med 95 % parvise bootstrap-konfidensintervaller.

| Måling | Scriptor p50 | vLLM p50 | Dom |
| ------ | -----------: | -------: | --- |
| Veggtid, 368 sider (s) | 236.7 | 454.8 | SEIER (0.520x) |
| Gjennomstrømning (sider/time) | 5,598 | 2,913 | SEIER (1.922x) |
| Oppstart til klar, etter første oppstart (s) | 14.25 | 154.3 | SEIER (0.092x) |
| Tid til første token (s) | 5.070 | 9.167 | SEIER (0.553x) |
| Ende-til-ende-forespørsel (s) | 8.339 | 13.29 | SEIER (0.627x) |

Én maskin, én GPU, én økt, et fast sett på 15 dokumenter; hver motor kjørte sin egen leverte konfigurasjon. Oppsett, alle målinger, nøyaktighet, begrensninger og hvordan resultatene reproduseres: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Hurtigstart

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Med disse innstillingene eksportert lytter motoren på 127.0.0.1:8002 og OCR-broen på 127.0.0.1:8765. Send et sidebilde til motoren som en bilde-data-URL, og et helt dokument til broen:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` er en chat-completions-body der brukermeldingen inneholder en base64-kodet `image_url`-data-URL. Serveren laster ikke ned noe; `install.sh` er det eneste trinnet som bruker nettverk.

## Installasjon

`./install.sh` installerer den ferdige binærfilen, GLM-OCR-sjekkpunktet, layoutmodellen og et CUDA-bygg av onnxruntime for GB10, der hvert trinn er valgfritt og sjekksumverifisert; `./install.sh check` verifiserer resultatet. Du kan også bygge fra kildekode. Se [docs/install.md](docs/install.md).

## OCR-bro

`scriptor serve` kjører også en OCR-bro for dokumenter: PDF-er og bilder inn, gjenkjent tekst ut, med layoutdeteksjon gjennom onnxruntime. Flagg, innstillinger og den innebygde brosjekken står i [docs/bridge.md](docs/bridge.md).

## Ytelsestest

Fullstendige tabeller, nøyaktighet, begrensninger og trinn for reproduksjon: [docs/benchmark.md](docs/benchmark.md).

## Krasjsikkerhet

Avgrenset inntak, minnetrykknivåer, overvåkede arbeidsprosesser og avslutningskoder en veileder kan handle på: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Dokumentasjon

Dokumentasjon (på engelsk):

- [docs/install.md](docs/install.md) — installasjonsskript, ferdig utgivelse, bygging fra kildekode, lisenser for modeller og avhengigheter
- [docs/serving.md](docs/serving.md) — flagg for `scriptor serve`, forespørselsgrenser, krasjsikkerhet
- [docs/bridge.md](docs/bridge.md) — OCR-broen, med flagg og innstillinger
- [docs/benchmark.md](docs/benchmark.md) — resultater, nøyaktighet, begrensninger, reproduksjon av ytelsestesten
- [docs/boot.md](docs/boot.md) — minneplanen ved oppstart, oppstartstrinnene og hvordan oppstartstiden måles
- [docs/drafting.md](docs/drafting.md) — spekulativ MTP-dekoding, akseptpolicy og tilhørende metrikker
- [docs/attention.md](docs/attention.md) — attention-kjerner
- [docs/gemm.md](docs/gemm.md) — GEMM-kjerner
- [docs/equivalence.md](docs/equivalence.md) og [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — ekvivalenssjekker for kjerner og ende-til-ende
- [docs/engine-file.md](docs/engine-file.md) — motorens egen vektfil
- [docs/model-facts.md](docs/model-facts.md) — GLM-OCR-fakta som motoren følger
- [docs/vision-overlap.md](docs/vision-overlap.md) og [docs/nvjpeg.md](docs/nvjpeg.md) — overlapp i bildebehandlingen og JPEG-dekoding på GPU
- [seat/README.md](seat/README.md) — malen `seat`, kontraktene for klarhet og tømming, avslutningskoder og feilinjeksjon
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — referanseprotokollen for måling
- [docs/third-party-notices.md](docs/third-party-notices.md) — merknader om tredjepartskomponenter
- [docs/languages.md](docs/languages.md) — denne README-en på 38 språk

<a id="contributing"></a>
## Bidra

Les [CONTRIBUTING.md](CONTRIBUTING.md) først: Conventional Commit-emner, sjekkene for format, clippy og tester, kun syntetiske testdata. Meld sårbarheter privat som beskrevet i [SECURITY.md](SECURITY.md).

---

<a id="license"></a>
## Lisens

Dobbeltlisensiert under AGPL-3.0-only eller en kommersiell lisens (se
[LICENSE](LICENSE) og [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Opphavsrett 2026 Matthew D. Murphy. Kommersielle lisenser utstedes gjennom
[Chiron Courseworks](https://chironcourseworks.com); henvendelser:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Merknader om tredjepart finnes i [docs/third-party-notices.md](docs/third-party-notices.md). Modellvekter og kjøretidsbiblioteker lastes ned separat og beholder sine egne lisenser (se [Model and dependency licenses](docs/install.md#model-and-dependency-licenses)).
