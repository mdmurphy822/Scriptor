<p align="center"><img src="../assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*Natívny inferenčný server GLM-OCR pre NVIDIA GB10: skeny na vstupe, rozpoznaný text na výstupe, cez API kompatibilné s OpenAI, bez uloženého zahriateho stavu.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Stránka projektu](https://chironcourseworks.com/scriptor)** • [Rýchly štart](#quick-start) • [Inštalácia](../install.md) • [OCR most](../bridge.md) • [Benchmark](../benchmark.md) • [Dokumentácia](#documentation) • [Prispievanie](#contributing) • [Licencia](#license)

[Jazyky](../languages.md) • [English](../../README.md)

</div>

## Úvod

Scriptor (latinsky „pisár“) je lokálny inferenčný engine pre model GLM-OCR optimalizovaný na priepustnosť, napísaný v Ruste s ručne písanými jadrami CUDA a určený pre NVIDIA GB10 v DGX Spark (ARM64). Model GLM-OCR sprístupňuje cez endpoint chat-completions kompatibilný s OpenAI, v požiadavke prijíma obrázky strán a vystavuje metriky Prometheus `vllm:*` kompatibilné s vLLM, takže klient alebo dashboard vytvorený pre server vLLM funguje bez zmien. Je lokálny a `fail-closed` (pri chýbajúcom alebo chybnom vstupe sa zastaví, namiesto aby hádal): adresár modelu a bind adresa sú explicitné, nič sa nesťahuje, nič sa neobjavuje automaticky a chýbajúci, zastaraný alebo nezhodný vstup zastaví štart namiesto toho, aby sa obchádzal odhadom. Zoznam funkcií a zmluvu o odolnosti voči pádom nájdete v [docs/serving.md](../serving.md).

## Hlavné výsledky

Scriptor oproti vLLM, rovnaký model, dokumenty, protokol OCR mosta a GPU: 10 párových behov na engine, čerstvý boot pri každom behu, 368 strán na beh, pomery s 95 % párovými bootstrap intervalmi spoľahlivosti.

| Metrika | Scriptor p50 | vLLM p50 | Verdikt |
| ------- | -----------: | -------: | ------- |
| Celkový čas, 368 strán (s) | 236.7 | 454.8 | VÝHRA (0.520x) |
| Priepustnosť (strán/hodinu) | 5,598 | 2,913 | VÝHRA (1.922x) |
| Boot po pripravenosť, po prvom boote (s) | 14.25 | 154.3 | VÝHRA (0.092x) |
| Čas do prvého tokenu (s) | 5.070 | 9.167 | VÝHRA (0.553x) |
| Požiadavka end-to-end (s) | 8.339 | 13.29 | VÝHRA (0.627x) |

Jeden stroj, jedno GPU, jedna relácia, pevná sada 15 dokumentov; každý engine bežal vo svojej dodávanej konfigurácii. Nastavenie, všetky metriky, presnosť, obmedzenia a postup reprodukcie: [docs/benchmark.md](../benchmark.md).

<a id="quick-start"></a>
## Rýchly štart

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

S exportovaným nastavením engine počúva na 127.0.0.1:8002 a OCR most na 127.0.0.1:8765. Obrázok strany pošlite enginu ako data URL obrázka a celý dokument mostu:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` je telo požiadavky chat-completions, v ktorého používateľskej správe je `image_url` data URL v base64. Server nič nesťahuje; `install.sh` je jediný krok so sieťou.

## Inštalácia

`./install.sh` nainštaluje release binárku, checkpoint GLM-OCR, model rozloženia a build onnxruntime s CUDA pre GB10, pričom každý krok je voliteľný a overený kontrolným súčtom; `./install.sh check` overí výsledok. Môžete tiež zostaviť zo zdrojových kódov. Pozrite [docs/install.md](../install.md).

## OCR most

`scriptor serve` spúšťa aj dokumentový OCR most: PDF a obrázky na vstupe, rozpoznaný text na výstupe, s detekciou rozloženia cez onnxruntime. Príznaky, nastavenia a vstavanú kontrolu mosta nájdete v [docs/bridge.md](../bridge.md).

## Benchmark

Úplné tabuľky, presnosť, obmedzenia a kroky reprodukcie: [docs/benchmark.md](../benchmark.md).

## Odolnosť voči pádom

Ohraničený príjem, úrovne tlaku na pamäť, supervidované workery a kódy ukončenia, na ktoré môže supervízor reagovať: [docs/serving.md](../serving.md#crash-safety).

<a id="documentation"></a>
## Dokumentácia

Dokumentácia (v angličtine):

- [docs/install.md](../install.md) — inštalačný skript, predpripravený release, zostavenie zo zdrojových kódov, licencie modelov a závislostí
- [docs/serving.md](../serving.md) — príznaky `scriptor serve`, limity požiadaviek, odolnosť voči pádom
- [docs/bridge.md](../bridge.md) — OCR most, jeho príznaky a nastavenia
- [docs/benchmark.md](../benchmark.md) — výsledky, presnosť, obmedzenia, reprodukcia benchmarku
- [docs/boot.md](../boot.md) — plán pamäte pri boote, fázy bootu a spôsob merania času bootu
- [docs/drafting.md](../drafting.md) — špekulatívne dekódovanie MTP, politika akceptácie a jej metriky
- [docs/attention.md](../attention.md) — jadrá attention
- [docs/gemm.md](../gemm.md) — jadrá GEMM
- [docs/equivalence.md](../equivalence.md) a [docs/e2e-equivalence.md](../e2e-equivalence.md) — kontroly ekvivalencie jadier a end-to-end
- [docs/engine-file.md](../engine-file.md) — natívny súbor váh enginu
- [docs/model-facts.md](../model-facts.md) — fakty o GLM-OCR, ktorým engine zodpovedá
- [docs/vision-overlap.md](../vision-overlap.md) a [docs/nvjpeg.md](../nvjpeg.md) — prekrytie vision a dekódovanie JPEG na GPU
- [seat/README.md](../../seat/README.md) — šablóna `seat`, zmluvy o pripravenosti a vyprázdnení, kódy ukončenia a injekcia chýb
- [docs/phase0-protocol.md](../phase0-protocol.md) — referenčný protokol merania
- [docs/third-party-notices.md](../third-party-notices.md) — upozornenia tretích strán
- [docs/languages.md](../languages.md) — tento README v 38 jazykoch

<a id="contributing"></a>
## Prispievanie

Najprv si prečítajte [CONTRIBUTING.md](../../CONTRIBUTING.md): predmety commitov v štýle Conventional Commit, brána format, clippy a testy, iba syntetické fixtures. Zraniteľnosti nahlasujte súkromne podľa [SECURITY.md](../../SECURITY.md).

---

<a id="license"></a>
## Licencia

Dvojitá licencia: AGPL-3.0-only alebo komerčná licencia (pozrite
[LICENSE](../../LICENSE) a [COMMERCIAL-LICENSE.md](../../COMMERCIAL-LICENSE.md)).
Copyright 2026 Matthew D. Murphy. Komerčné licencie vydáva
[Chiron Courseworks](https://chironcourseworks.com); dopyty:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Upozornenia tretích strán sú v [docs/third-party-notices.md](../third-party-notices.md). Váhy modelov a runtime knižnice sú samostatné stiahnutia a zachovávajú si vlastné licencie (pozrite [Model and dependency licenses](../install.md#model-and-dependency-licenses)).
