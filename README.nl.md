<p align="center"><img src="docs/assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*Een native GLM-OCR-inferentieserver voor de NVIDIA GB10: scans erin, herkende tekst eruit, via een OpenAI-compatibele API en zonder opgeslagen warme status.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Projectpagina](https://chironcourseworks.com/scriptor)** • [Snelstart](#quick-start) • [Installatie](docs/install.md) • [OCR-bridge](docs/bridge.md) • [Benchmark](docs/benchmark.md) • [Documentatie](#documentation) • [Bijdragen](#contributing) • [Licentie](#license)

[Talen](docs/languages.md) • [English](README.md)

</div>

<a id="introduction"></a>
## Inleiding

Scriptor (Latijn voor „schrijver”) is een op doorvoer geoptimaliseerde, uitsluitend lokale inferentie-engine voor het GLM-OCR-model, geschreven in Rust met met de hand geschreven CUDA-kernels en gebouwd voor de NVIDIA GB10 in de DGX Spark (ARM64). Het serveert GLM-OCR via een OpenAI-compatibel chat-completions-eindpunt, accepteert paginabeelden in het verzoek en biedt vLLM-compatibele `vllm:*`-Prometheus-metrics, zodat een client of dashboard dat voor een vLLM-server is gebouwd ongewijzigd werkt. Het is lokaal en `fail-closed` (het stopt in plaats van te gokken als een invoer ontbreekt of fout is): de modelmap en het bindadres zijn expliciet, er wordt niets gedownload, niets automatisch ontdekt, en een ontbrekende, verouderde of niet-passende invoer stopt de start in plaats van omzeild te worden. Zie [docs/serving.md](docs/serving.md) voor de functielijst en het crashveiligheidscontract.

## Belangrijkste resultaten

Scriptor tegenover vLLM, met hetzelfde model, dezelfde documenten, hetzelfde OCR-bridgeprotocol en dezelfde GPU: 10 gepaarde runs per engine, bij elke run een verse boot, 368 pagina’s per run, verhoudingen met gepaarde 95%-bootstrap-betrouwbaarheidsintervallen.

| Metric | Scriptor p50 | vLLM p50 | Oordeel |
| ------ | -----------: | -------: | ------- |
| Wandkloktijd, 368 pagina’s (s) | 236.7 | 454.8 | WINST (0.520x) |
| Doorvoer (pagina’s/uur) | 5,598 | 2,913 | WINST (1.922x) |
| Boot tot gereed, na de eerste boot (s) | 14.25 | 154.3 | WINST (0.092x) |
| Tijd tot eerste token (s) | 5.070 | 9.167 | WINST (0.553x) |
| Verzoek end-to-end (s) | 8.339 | 13.29 | WINST (0.627x) |

Eén machine, één GPU, één sessie, een vaste set van 15 documenten; elke engine draaide met zijn eigen meegeleverde configuratie. Opzet, alle metrics, nauwkeurigheid, beperkingen en reproductie: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Snelstart

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Met die instellingen luistert de engine op 127.0.0.1:8002 en de OCR-bridge op 127.0.0.1:8765. Stuur een paginabeeld als image-data-URL naar de engine en een heel document naar de bridge:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` is een chat-completions-body waarvan het gebruikersbericht een base64-`image_url`-data-URL bevat. De server downloadt niets; `install.sh` is de enige stap met netwerktoegang.

## Installatie

`./install.sh` installeert de release-binary, het GLM-OCR-checkpoint, het layoutmodel en een CUDA-build van onnxruntime voor de GB10; elke stap is opt-in en met checksum geverifieerd, en `./install.sh check` verifieert het resultaat. U kunt ook vanaf de broncode bouwen. Zie [docs/install.md](docs/install.md).

## OCR-bridge

`scriptor serve` draait ook een document-OCR-bridge: pdf’s en afbeeldingen erin, herkende tekst eruit, met layoutdetectie via onnxruntime. Flags, instellingen en de ingebouwde bridgecontrole staan in [docs/bridge.md](docs/bridge.md).

## Benchmark

Volledige tabellen, nauwkeurigheid, beperkingen en reproductiestappen: [docs/benchmark.md](docs/benchmark.md).

## Crashveiligheid

Begrensde intake, geheugendrukniveaus, bewaakte workers en exitcodes waarop een supervisor kan reageren: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Documentatie

Documentatie (in het Engels):

- [docs/install.md](docs/install.md) — installatiescript, voorgebouwde release, bouwen vanaf de broncode, model- en afhankelijkheidslicenties
- [docs/serving.md](docs/serving.md) — `scriptor serve`-flags, verzoeklimieten, crashveiligheid
- [docs/bridge.md](docs/bridge.md) — de OCR-bridge, zijn flags en instellingen
- [docs/benchmark.md](docs/benchmark.md) — resultaten, nauwkeurigheid, beperkingen, de benchmark reproduceren
- [docs/boot.md](docs/boot.md) — het boot-geheugenplan, de bootfasen en hoe de boottijd wordt gemeten
- [docs/drafting.md](docs/drafting.md) — MTP-speculatieve decodering, acceptatiebeleid en de bijbehorende metrics
- [docs/attention.md](docs/attention.md) — attention-kernels
- [docs/gemm.md](docs/gemm.md) — GEMM-kernels
- [docs/equivalence.md](docs/equivalence.md) en [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — equivalentiecontroles voor kernels en end-to-end
- [docs/engine-file.md](docs/engine-file.md) — het engine-eigen gewichtenbestand
- [docs/model-facts.md](docs/model-facts.md) — de GLM-OCR-feiten waaraan de engine voldoet
- [docs/vision-overlap.md](docs/vision-overlap.md) en [docs/nvjpeg.md](docs/nvjpeg.md) — vision-overlap en GPU-JPEG-decodering
- [seat/README.md](seat/README.md) — het `seat`-sjabloon, gereedheids- en drain-contracten, exitcodes en foutinjectie
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — het referentiemeetprotocol
- [docs/third-party-notices.md](docs/third-party-notices.md) — vermeldingen van derden
- [docs/languages.md](docs/languages.md) — deze README in 38 talen

<a id="contributing"></a>
## Bijdragen

Lees eerst [CONTRIBUTING.md](CONTRIBUTING.md): Conventional Commit-onderwerpen, de gate van format, clippy en tests, uitsluitend synthetische fixtures. Meld kwetsbaarheden privé zoals beschreven in [SECURITY.md](SECURITY.md).

---

<a id="license"></a>
## Licentie

Dubbel gelicentieerd onder AGPL-3.0-only of een commerciële licentie (zie
[LICENSE](LICENSE) en [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Copyright 2026 Matthew D. Murphy. Commerciële licenties worden verstrekt via
[Chiron Courseworks](https://chironcourseworks.com); aanvragen:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Vermeldingen van derden staan in [docs/third-party-notices.md](docs/third-party-notices.md). Modelgewichten en runtimebibliotheken zijn aparte downloads en behouden hun eigen licenties (zie [Model and dependency licenses](docs/install.md#model-and-dependency-licenses)).
