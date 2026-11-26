<p align="center"><img src="docs/assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*Natīvs GLM-OCR secināšanas serveris NVIDIA GB10: skenējumi iekšā, atpazīts teksts ārā, ar OpenAI saderīgu API, bez saglabāta sasildīšanas stāvokļa.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Projekta lapa](https://chironcourseworks.com/scriptor)** • [Ātrā sākšana](#quick-start) • [Instalēšana](docs/install.md) • [OCR tilts](docs/bridge.md) • [Etalontests](docs/benchmark.md) • [Dokumentācija](#documentation) • [Līdzdalība](#contributing) • [Licence](#license)

[Valodas](docs/languages.md) • [English](README.md)

</div>

## Ievads

Scriptor (latīņu valodā „rakstvedis") ir uz caurlaidspēju optimizēts, tikai lokāli darbināms secināšanas dzinējs GLM-OCR modelim. Tas ir uzrakstīts Rust valodā, izmanto ar roku rakstītus CUDA kodolus un ir paredzēts NVIDIA GB10 DGX Spark (ARM64) platformā. Dzinējs apkalpo GLM-OCR ar OpenAI saderīgu chat-completions galapunktu, pieprasījumā pieņem lapu attēlus un nodrošina ar vLLM saderīgas `vllm:*` Prometheus metrikas, tāpēc klients vai informācijas panelis, kas veidots vLLM serverim, darbojas nemainīti. Sistēma ir lokāla un `fail-closed` (apstājas, nevis min, ja ievade trūkst vai ir kļūdaina): modeļa direktorija un saistīšanas adrese ir norādīti tieši, nekas netiek lejupielādēts, nekas netiek automātiski atklāts, un trūkstoša, novecojusi vai neatbilstoša ievade aptur startēšanu, nevis tiek minēta. Funkciju sarakstu un avāriju drošības līgumu skatiet [docs/serving.md](docs/serving.md).

## Galvenie rezultāti

Scriptor pret vLLM, viens un tas pats modelis, dokumenti, OCR tilta protokols un GPU: 10 pārī savienoti darbinājumi katram dzinējam, katram darbinājumam jauna palaišana (boot), 368 lapas darbinājumā, attiecības ar 95% pārī savienotiem bootstrap ticamības intervāliem.

| Metrika | Scriptor p50 | vLLM p50 | Spriedums |
| ------ | -----------: | -------: | ------- |
| Sienas laiks, 368 lapas (s) | 236.7 | 454.8 | UZVARA (0.520x) |
| Caurlaidspēja (lapas/stundā) | 5,598 | 2,913 | UZVARA (1.922x) |
| Palaišana līdz gatavībai, pēc pirmās palaišanas (s) | 14.25 | 154.3 | UZVARA (0.092x) |
| Laiks līdz pirmajam tokenam (s) | 5.070 | 9.167 | UZVARA (0.553x) |
| Pieprasījums no gala līdz galam (s) | 8.339 | 13.29 | UZVARA (0.627x) |

Viena iekārta, viens GPU, viena sesija, fiksēts 15 dokumentu kopums; katrs dzinējs darbojās ar savu piegādāto konfigurāciju. Iestatīšana, visas metrikas, precizitāte, ierobežojumi un atkārtošana: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Ātrā sākšana

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Ar šiem iestatījumiem eksportētiem dzinējs klausās 127.0.0.1:8002, bet OCR tilts 127.0.0.1:8765. Lapas attēlu sūtiet dzinējam kā attēla data URL, bet veselu dokumentu tiltam:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` ir chat-completions pieprasījuma pamatteksts, kura lietotāja ziņojums satur base64 kodētu `image_url` data URL. Serveris neko nelejupielādē; `install.sh` ir vienīgais soli, kas izmanto tīklu.

## Instalēšana

`./install.sh` instalē izlaiduma bināro failu, GLM-OCR kontrolpunktu, izkārtojuma modeli un GB10 paredzētu CUDA onnxruntime būvējumu; katrs solis ir izvēles un pārbaudīts ar kontrolsummu; `./install.sh check` pārbauda rezultātu. Varat arī būvēt no pirmkoda. Skatiet [docs/install.md](docs/install.md).

## OCR tilts

`scriptor serve` palaiž arī dokumentu OCR tiltu: PDF un attēli iekšā, atpazīts teksts ārā, izkārtojuma noteikšana ar onnxruntime. Karodziņi, iestatījumi un iebūvētā tilta pārbaude ir aprakstīti [docs/bridge.md](docs/bridge.md).

## Etalontests

Pilnas tabulas, precizitāte, ierobežojumi un atkārtošanas soļi: [docs/benchmark.md](docs/benchmark.md).

## Avāriju drošība

Ierobežota pieņemšana, atmiņas spiediena līmeņi, uzraudzīti darbinieki un iziešanas kodi, uz kuriem uzraugs var reaģēt: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Dokumentācija

Dokumentācija (angliski):

- [docs/install.md](docs/install.md) — instalēšanas skripts, sagatavots izlaidums, būvēšana no pirmkoda, modeļa un atkarību licences
- [docs/serving.md](docs/serving.md) — `scriptor serve` karodziņi, pieprasījumu ierobežojumi, avāriju drošība
- [docs/bridge.md](docs/bridge.md) — OCR tilts, tā karodziņi un iestatījumi
- [docs/benchmark.md](docs/benchmark.md) — rezultāti, precizitāte, ierobežojumi, etalontesta atkārtošana
- [docs/boot.md](docs/boot.md) — palaišanas atmiņas plāns, palaišanas posmi un palaišanas laika mērīšana
- [docs/drafting.md](docs/drafting.md) — MTP spekulatīvā dekodēšana, pieņemšanas politika un tās metrikas
- [docs/attention.md](docs/attention.md) — uzmanības kodoli
- [docs/gemm.md](docs/gemm.md) — GEMM kodoli
- [docs/equivalence.md](docs/equivalence.md) un [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — kodolu un pilna cikla ekvivalences pārbaudes
- [docs/engine-file.md](docs/engine-file.md) — dzinēja natīvais svaru fails
- [docs/model-facts.md](docs/model-facts.md) — GLM-OCR fakti, kuriem dzinējs atbilst
- [docs/vision-overlap.md](docs/vision-overlap.md) un [docs/nvjpeg.md](docs/nvjpeg.md) — redzes pārklāšanās un GPU JPEG dekodēšana
- [seat/README.md](seat/README.md) — `seat` sagatave, gatavības un iztukšošanas līgumi, iziešanas kodi un kļūdu injicēšana
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — references mērījumu protokols
- [docs/third-party-notices.md](docs/third-party-notices.md) — trešo pušu paziņojumi
- [docs/languages.md](docs/languages.md) — šis README 38 valodās

<a id="contributing"></a>
## Līdzdalība

Vispirms izlasiet [CONTRIBUTING.md](CONTRIBUTING.md): Conventional Commit tēmas, formāta, clippy un testu vārti, tikai sintētiski testa dati. Par ievainojamībām ziņojiet privāti, kā aprakstīts [SECURITY.md](SECURITY.md).

---

<a id="license"></a>
## Licence

Divkārša licence: AGPL-3.0-only vai komerciālā licence (skatiet
[LICENSE](LICENSE) un [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Autortiesības 2026 Matthew D. Murphy. Komerciālās licences izsniedz
[Chiron Courseworks](https://chironcourseworks.com); jautājumi:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Trešo pušu paziņojumi ir [docs/third-party-notices.md](docs/third-party-notices.md). Modeļa svari un izpildlaika bibliotēkas ir atsevišķi lejupielādējamas un saglabā savas licences (skatiet [Modeļa un atkarību licences](docs/install.md#model-and-dependency-licenses)).
