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

*Server nattiv ta' inferenza ta' GLM-OCR għall-NVIDIA GB10: skans fl-input, test rikonoxxut fl-output, permezz ta' API kompatibbli ma' OpenAI, mingħajr stat imsaħħan salvat.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Paġna tal-proġett](https://chironcourseworks.com/scriptor)** • [Bidu mgħaġġel](#quick-start) • [Installazzjoni](docs/install.md) • [Pont OCR](docs/bridge.md) • [Benchmark](docs/benchmark.md) • [Dokumentazzjoni](#documentation) • [Kontribuzzjoni](#contributing) • [Liċenzja](#license)

[Lingwi](docs/languages.md) • [English](README.md)

</div>

## Introduzzjoni

Scriptor (Latin għal "kittieb") huwa magna ta' inferenza lokali biss, ottimizzata għall-throughput, għall-mudell GLM-OCR, miktuba bir-Rust b'kernels CUDA miktuba bl-idejn u mibnija għall-NVIDIA GB10 tad-DGX Spark (ARM64). Iservi GLM-OCR permezz ta' endpoint chat-completions kompatibbli ma' OpenAI, jaċċetta stampi tal-paġni fit-talba, u jesponi metriċi Prometheus `vllm:*` kompatibbli ma' vLLM, biex klijent jew dashboard mibni għal server ta' vLLM jaħdem mingħajr bidliet. Huwa lokali u `fail-closed` (jieqaf minflok jaħseb bil-kjass meta input ikun nieqes jew żbaljat): id-direttorju tal-mudell u l-indirizz tal-bind huma espliċiti, xejn ma jitniżżel, xejn ma jiġi skopert, u input nieqes, skadut jew mhux konsistenti jwaqqaf il-bidu minflok ma jiġi evitat bil-kjass. Ara [docs/serving.md](docs/serving.md) għal-lista tal-karatteristiċi u l-kuntratt tas-sigurtà fil-ħabtiet.

## Riżultati ewlenin

Scriptor kontra vLLM, bl-istess mudell, dokumenti, protokoll tal-pont OCR u GPU: 10 ġirjiet f'pari għal kull magna, boot ġdid għal kull ġirja, 368 paġna għal kull ġirja, proporzjonijiet b'intervalli ta' kunfidenza ta' 95% bil-bootstrap f'pari.

| Metrika | Scriptor p50 | vLLM p50 | Verdett |
| ------ | -----------: | -------: | ------- |
| Ħin totali, 368 paġna (s) | 236.7 | 454.8 | rebħa (0.520x) |
| Throughput (paġni/siegħa) | 5,598 | 2,913 | rebħa (1.922x) |
| Boot sa lest, wara l-ewwel boot (s) | 14.25 | 154.3 | rebħa (0.092x) |
| Ħin sal-ewwel token (s) | 5.070 | 9.167 | rebħa (0.553x) |
| Talba minn tarf sa tarf (s) | 8.339 | 13.29 | rebħa (0.627x) |

Magna waħda, GPU waħda, sessjoni waħda, sett fiss ta' 15-il dokument; kull magna mxiet bil-konfigurazzjoni tagħha stess kif ġiet fornuta. Setup, il-metriċi kollha, l-akkuratezza, il-limitazzjonijiet u kif tirriproduċi: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Bidu mgħaġġel

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

B'dawk is-settings esportati, il-magna tisma' fuq 127.0.0.1:8002 u l-pont OCR fuq 127.0.0.1:8765. Ibgħat stampa ta' paġna lill-magna bħala data URL ta' stampa, u dokument sħiħ lill-pont:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` huwa body ta' chat-completions li l-messaġġ tal-utent fih iġorr data URL `image_url` f'base64. Is-server ma jniżżel xejn; `install.sh` huwa l-uniku pass li juża n-netwerk.

## Installazzjoni

`./install.sh` jinstalla l-binary tar-release, iċ-checkpoint ta' GLM-OCR, il-mudell tal-layout u bini CUDA ta' onnxruntime għall-GB10, kull pass huwa opt-in u vverifikat bi checksum; `./install.sh check` jivverifika r-riżultat. Tista' wkoll tibni mis-sors. Ara [docs/install.md](docs/install.md).

## Pont OCR

`scriptor serve` iħaddem ukoll pont OCR tad-dokumenti: PDFs u stampi fl-input, test rikonoxxut fl-output, b'detezzjoni tal-layout permezz ta' onnxruntime. Il-flags, is-settings u l-kontroll integrat tal-pont jinsabu f'[docs/bridge.md](docs/bridge.md).

## Benchmark

It-tabelli sħaħ, l-akkuratezza, il-limitazzjonijiet u l-passi għar-riproduzzjoni: [docs/benchmark.md](docs/benchmark.md).

## Sigurtà fil-ħabtiet

Intake limitat, livelli ta' pressjoni tal-memorja, workers issorveljati u exit codes li superviżur jista' jaġixxi fuqhom: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Dokumentazzjoni

Dokumentazzjoni (bl-Ingliż):

- [docs/install.md](docs/install.md) — l-iskrip tal-installazzjoni, ir-release lest, il-bini mis-sors, il-liċenzji tal-mudell u tad-dipendenzi
- [docs/serving.md](docs/serving.md) — il-flags ta' `scriptor serve`, il-limiti tat-talbiet, is-sigurtà fil-ħabtiet
- [docs/bridge.md](docs/bridge.md) — il-pont OCR, il-flags u s-settings tiegħu
- [docs/benchmark.md](docs/benchmark.md) — riżultati, akkuratezza, limitazzjonijiet, kif tirriproduċi l-benchmark
- [docs/boot.md](docs/boot.md) — il-pjan tal-memorja tal-boot, il-fażijiet tal-boot u kif jitkejjel il-ħin tal-boot
- [docs/drafting.md](docs/drafting.md) — deċifrar spekulattiv MTP, il-politika ta' aċċettazzjoni u l-metriċi tagħha
- [docs/attention.md](docs/attention.md) — kernels tal-attenzjoni
- [docs/gemm.md](docs/gemm.md) — kernels GEMM
- [docs/equivalence.md](docs/equivalence.md) u [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — kontrolli ta' ekwivalenza tal-kernels u minn tarf sa tarf
- [docs/engine-file.md](docs/engine-file.md) — il-fajl tal-piżijiet nattiv tal-magna
- [docs/model-facts.md](docs/model-facts.md) — il-fatti ta' GLM-OCR li l-magna tqabbel magħhom
- [docs/vision-overlap.md](docs/vision-overlap.md) u [docs/nvjpeg.md](docs/nvjpeg.md) — it-tqegħid fuq xulxin tal-vużjoni u d-dekodifikar JPEG fuq GPU
- [seat/README.md](seat/README.md) — it-template `seat`, il-kuntratti ta' readiness u drain, exit codes u injezzjoni ta' fawls
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — il-protokoll ta' qies ta' referenza
- [docs/third-party-notices.md](docs/third-party-notices.md) — avviżi ta' terzi
- [docs/languages.md](docs/languages.md) — dan ir-README f'38 lingwa

<a id="contributing"></a>
## Kontribuzzjoni

Aqra [CONTRIBUTING.md](CONTRIBUTING.md) l-ewwel: suġġetti ta' Conventional Commit, il-gate tal-format, clippy u t-testijiet, fixtures sintetiċi biss. Irrapporta vulnerabbiltajiet privatament kif deskritt f'[SECURITY.md](SECURITY.md).

---

<a id="license"></a>
## Liċenzja

Liċenzjat doppjament taħt AGPL-3.0-only jew liċenzja kummerċjali (ara
[LICENSE](LICENSE) u [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Copyright 2026 Matthew D. Murphy. Il-liċenzji kummerċjali jinħarġu permezz ta'
[Chiron Courseworks](https://chironcourseworks.com); mistoqsijiet:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

L-avviżi ta' terzi jinsabu f'[docs/third-party-notices.md](docs/third-party-notices.md). Il-piżijiet tal-mudell u l-libreriji tar-runtime huma downloads separati u jżommu l-liċenzji tagħhom stess (ara [Model and dependency licenses](docs/install.md#model-and-dependency-licenses)).
