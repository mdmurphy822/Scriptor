<p align="center"><img src="docs/assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*Nativní inferenční server GLM-OCR pro NVIDIA GB10: skeny na vstupu, rozpoznaný text na výstupu, přes API kompatibilní s OpenAI, bez ukládaného zahřátého stavu.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Stránka projektu](https://chironcourseworks.com/scriptor)** • [Rychlý start](#quick-start) • [Instalace](docs/install.md) • [OCR most](docs/bridge.md) • [Benchmark](docs/benchmark.md) • [Dokumentace](#documentation) • [Přispívání](#contributing) • [Licence](#license)

[Jazyky](docs/languages.md) • [English](README.md)

</div>

## Úvod

Scriptor (latinsky „písař“) je lokální inferenční engine pro model GLM-OCR optimalizovaný na propustnost, napsaný v Rustu s ručně psanými jádry CUDA a vytvořený pro NVIDIA GB10 v DGX Spark (ARM64). Model GLM-OCR zpřístupňuje přes endpoint chat-completions kompatibilní s OpenAI, přijímá v požadavku obrázky stránek a vystavuje metriky Prometheus `vllm:*` kompatibilní s vLLM, takže klient nebo dashboard určený pro server vLLM funguje beze změny. Pracuje pouze lokálně a v režimu `fail-closed` (zastavení místo hádání): adresář modelu a adresa pro naslouchání jsou explicitní, nic se nestahuje, nic se automaticky nevyhledává a chybějící, zastaralý nebo nesouhlasný vstup zastaví start, místo aby se obcházel odhady. Seznam funkcí a smlouvu o odolnosti proti pádům najdete v [docs/serving.md](docs/serving.md).

## Hlavní výsledky

Scriptor oproti vLLM, stejný model, dokumenty, protokol OCR mostu a GPU: 10 párových běhů na engine, čerstvý start při každém běhu, 368 stránek na běh, poměry s 95% párovými bootstrapovými intervaly spolehlivosti.

| Metrika | Scriptor p50 | vLLM p50 | Verdikt |
| ------ | -----------: | -------: | ------- |
| Celkový čas, 368 stránek (s) | 236.7 | 454.8 | VÝHRA (0.520x) |
| Propustnost (stránek/hod.) | 5,598 | 2,913 | VÝHRA (1.922x) |
| Start do připravenosti, po prvním startu (s) | 14.25 | 154.3 | VÝHRA (0.092x) |
| Čas do prvního tokenu (s) | 5.070 | 9.167 | VÝHRA (0.553x) |
| Požadavek end-to-end (s) | 8.339 | 13.29 | VÝHRA (0.627x) |

Jeden stroj, jedno GPU, jedna relace, pevná sada 15 dokumentů; každý engine běžel ve své vlastní dodávané konfiguraci. Nastavení, všechny metriky, přesnost, omezení a postup reprodukce: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Rychlý start

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Po exportu těchto nastavení engine naslouchá na 127.0.0.1:8002 a OCR most na 127.0.0.1:8765. Obrázek stránky posílejte enginu jako data URL obrázku a celý dokument mostu:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` je tělo požadavku chat-completions, jehož uživatelská zpráva obsahuje data URL `image_url` v base64. Server nic nestahuje; `install.sh` je jediný krok, který používá síť.

## Instalace

`./install.sh` nainstaluje release binárku, checkpoint GLM-OCR, model rozložení stránky a build CUDA knihovny onnxruntime pro GB10; každý krok je volitelný a ověřený kontrolním součtem a `./install.sh check` výsledek zkontroluje. Lze také sestavit ze zdrojových kódů. Viz [docs/install.md](docs/install.md).

## OCR most

`scriptor serve` spouští také OCR most dokumentů: PDF a obrázky na vstupu, rozpoznaný text na výstupu, s detekcí rozložení stránky přes onnxruntime. Příznaky, nastavení a vestavěnou kontrolu mostu popisuje [docs/bridge.md](docs/bridge.md).

## Benchmark

Úplné tabulky, přesnost, omezení a postup reprodukce: [docs/benchmark.md](docs/benchmark.md).

## Odolnost proti pádům

Omezený příjem požadavků, úrovně tlaku na paměť, dohlížené worker procesy a návratové kódy, na které může reagovat supervizor: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Dokumentace

Dokumentace (anglicky):

- [docs/install.md](docs/install.md) — instalační skript, předpřipravené vydání, sestavení ze zdrojů, licence modelů a závislostí
- [docs/serving.md](docs/serving.md) — příznaky `scriptor serve`, limity požadavků, odolnost proti pádům
- [docs/bridge.md](docs/bridge.md) — OCR most, jeho příznaky a nastavení
- [docs/benchmark.md](docs/benchmark.md) — výsledky, přesnost, omezení, reprodukce benchmarku
- [docs/boot.md](docs/boot.md) — paměťový plán startu, fáze startu a způsob měření doby startu
- [docs/drafting.md](docs/drafting.md) — spekulativní dekódování MTP, politika přijímání a jeho metriky
- [docs/attention.md](docs/attention.md) — jádra attention
- [docs/gemm.md](docs/gemm.md) — jádra GEMM
- [docs/equivalence.md](docs/equivalence.md) a [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — kontroly ekvivalence jader a end-to-end
- [docs/engine-file.md](docs/engine-file.md) — nativní soubor vah enginu
- [docs/model-facts.md](docs/model-facts.md) — fakta o GLM-OCR, kterým engine odpovídá
- [docs/vision-overlap.md](docs/vision-overlap.md) a [docs/nvjpeg.md](docs/nvjpeg.md) — překryv zpracování obrazu a dekódování JPEG na GPU
- [seat/README.md](seat/README.md) — šablona `seat`, smlouvy o připravenosti a drain, návratové kódy a injekce chyb
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — referenční měřicí protokol
- [docs/third-party-notices.md](docs/third-party-notices.md) — oznámení o komponentách třetích stran
- [docs/languages.md](docs/languages.md) — tento README v 38 jazycích

<a id="contributing"></a>
## Přispívání

Začněte v [CONTRIBUTING.md](CONTRIBUTING.md): předměty commitů ve stylu Conventional Commits, brána formátování, clippy a testů, pouze syntetická testovací data. Zranitelnosti hlaste soukromě podle popisu v [SECURITY.md](SECURITY.md).

---

<a id="license"></a>
## Licence

Dvojí licence: AGPL-3.0-only nebo komerční licence (viz
[LICENSE](LICENSE) a [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Copyright 2026 Matthew D. Murphy. Komerční licence uděluje
[Chiron Courseworks](https://chironcourseworks.com); dotazy:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Oznámení třetích stran jsou v [docs/third-party-notices.md](docs/third-party-notices.md). Váhy modelů a běhové knihovny se stahují zvlášť a zachovávají si vlastní licence (viz [Licence modelů a závislostí](docs/install.md#model-and-dependency-licenses)).
