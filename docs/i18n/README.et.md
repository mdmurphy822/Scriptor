<p align="center"><img src="../assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*Natiivne GLM-OCR järeldusserver NVIDIA GB10 jaoks: skaneeringud sisse, tuvastatud tekst välja, OpenAI-ühilduva API kaudu, ilma salvestatud soojaoleku olekuta.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Projekti leht](https://chironcourseworks.com/scriptor)** • [Kiirstart](#quick-start) • [Paigaldus](../install.md) • [OCR-sild](../bridge.md) • [Võrdlustest](../benchmark.md) • [Dokumentatsioon](#documentation) • [Panustamine](#contributing) • [Litsents](#license)

[Keeled](../languages.md) • [English](../../README.md)

</div>

## Sissejuhatus

Scriptor (ladina keeles „kirjutaja") on läbilaskevõime poole optimeeritud, ainult kohalikult töötav järeldusmootor GLM-OCR mudelile. See on kirjutatud Rustis, kasutab käsitsi kirjutatud CUDA tuumasid ning on mõeldud DGX Sparki (ARM64) NVIDIA GB10 jaoks. Mootor teenindab GLM-OCR-i OpenAI-ühilduva chat-completions lõpp-punkti kaudu, võtab lehepildid vastu päringus ning pakub vLLM-iga ühilduvaid `vllm:*` Prometheus'e mõõdikuid, nii et vLLM-serveri jaoks ehitatud klient või töölaud töötab muutmata kujul. Süsteem on kohalik ja `fail-closed` (peatub, selle asemel et puuduva või vale sisendi puhul arvata): mudelikataloog ja kuulamisaadress on selgesõnalised, midagi ei laadita alla, midagi ei avastata automaatselt ning puuduv, aegunud või mittesobiv sisend peatab käivituse, selle asemel et seda aimata. Funktsioonide loendi ja krahhikindluse lepingu leiad failist [docs/serving.md](../serving.md).

## Peamised tulemused

Scriptor vs vLLM, sama mudel, samad dokumendid, sama OCR-silla protokoll ja sama GPU: 10 paaritatud käivitust mootori kohta, iga käivituse jaoks värske boot, 368 lehte käivituse kohta, suhtarvud 95% paaritatud bootstrap-usaldusvahemikega.

| Mõõdik | Scriptor p50 | vLLM p50 | Hinnang |
| ------ | -----------: | -------: | ------- |
| Seinaaeg, 368 lehte (s) | 236.7 | 454.8 | VÕIT (0.520x) |
| Läbilaskevõime (lehte/tunnis) | 5,598 | 2,913 | VÕIT (1.922x) |
| Boot valmisolekuni, pärast esimest booti (s) | 14.25 | 154.3 | VÕIT (0.092x) |
| Aeg esimese tokenini (s) | 5.070 | 9.167 | VÕIT (0.553x) |
| Päring otsast lõpuni (s) | 8.339 | 13.29 | VÕIT (0.627x) |

Üks masin, üks GPU, üks sessioon, fikseeritud 15 dokumendi komplekt; kumbki mootor kasutas oma tarnitud konfiguratsiooni. Seadistus, kõik mõõdikud, täpsus, piirangud ja reprodutseerimine: [docs/benchmark.md](../benchmark.md).

<a id="quick-start"></a>
## Kiirstart

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Nende seadistustega ekspordituna kuulab mootor aadressil 127.0.0.1:8002 ja OCR-sild aadressil 127.0.0.1:8765. Saada lehepilt mootorile pildi data-URL-ina ja tervik dokument sillale:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` on chat-completions päringu keha, mille kasutajasõnum sisaldab base64-kodeeritud `image_url` data-URL-i. Server ei lae midagi alla; `install.sh` on ainus võrku kasutav samm.

## Paigaldus

`./install.sh` paigaldab väljalaskebinaari, GLM-OCR-i kontrollpunkti, paigutusmudeli ja GB10 jaoks mõeldud CUDA-ga onnxruntime'i ehituse; iga samm on vabatahtlik ja kontrollsummaga verifitseeritud; `./install.sh check` kontrollib tulemust. Võid ka allikast ise ehitada. Vaata [docs/install.md](../install.md).

## OCR-sild

`scriptor serve` käivitab ka dokumentide OCR-silla: PDF-id ja pildid sisse, tuvastatud tekst välja, paigutuse tuvastus onnxruntime'i kaudu. Lipud, seaded ja sisseehitatud sillakontroll on failis [docs/bridge.md](../bridge.md).

## Võrdlustest

Täielikud tabelid, täpsus, piirangud ja reprodutseerimise sammud: [docs/benchmark.md](../benchmark.md).

## Krahhikindlus

Piiratud sissevõtt, mälusurve tasemed, järelevalvega töötajad ja väljumiskoodid, mille põhjal saab järelevaataja tegutseda: [docs/serving.md](../serving.md#crash-safety).

<a id="documentation"></a>
## Dokumentatsioon

Dokumentatsioon (inglise keeles):

- [docs/install.md](../install.md) — paigaldusskript, eelehitatud väljalase, ehitamine allikast, mudeli ja sõltuvuste litsentsid
- [docs/serving.md](../serving.md) — `scriptor serve` lipud, päringupiirangud, krahhikindlus
- [docs/bridge.md](../bridge.md) — OCR-sild, selle lipud ja seaded
- [docs/benchmark.md](../benchmark.md) — tulemused, täpsus, piirangud, võrdlustesti reprodutseerimine
- [docs/boot.md](../boot.md) — boot-mälukava, boot-etapid ja boot-aja mõõtmine
- [docs/drafting.md](../drafting.md) — MTP spekulatiivne dekodeerimine, aktsepteerimispoliitika ja selle mõõdikud
- [docs/attention.md](../attention.md) — tähelepanutuumad
- [docs/gemm.md](../gemm.md) — GEMM-tuumad
- [docs/equivalence.md](../equivalence.md) ja [docs/e2e-equivalence.md](../e2e-equivalence.md) — tuuma- ja otsast-lõpuni ekvivalentsikontrollid
- [docs/engine-file.md](../engine-file.md) — mootori natiivne kaaluide fail
- [docs/model-facts.md](../model-facts.md) — GLM-OCR-i faktid, millega mootor ühtib
- [docs/vision-overlap.md](../vision-overlap.md) ja [docs/nvjpeg.md](../nvjpeg.md) — visiooni kattuvus ja GPU JPEG-dekodeerimine
- [seat/README.md](../../seat/README.md) — `seat`-mall, valmisoleku ja tühjendamise lepingud, väljumiskoodid ja tõrkesüst
- [docs/phase0-protocol.md](../phase0-protocol.md) — võrdlusmõõtmise protokoll
- [docs/third-party-notices.md](../third-party-notices.md) — kolmandate osapoolte teated
- [docs/languages.md](../languages.md) — see README 38 keeles

<a id="contributing"></a>
## Panustamine

Loe kõigepealt [CONTRIBUTING.md](../../CONTRIBUTING.md): Conventional Commit'i teemaread, vormingu-, clippy ja testide värav, ainult sünteetilised testandmed. Teata haavatavustest privaatselt, nagu kirjeldatud failis [SECURITY.md](../../SECURITY.md).

---

<a id="license"></a>
## Litsents

Kahe litsentsiga: AGPL-3.0-only või kommertslitsents (vt
[LICENSE](../../LICENSE) ja [COMMERCIAL-LICENSE.md](../../COMMERCIAL-LICENSE.md)).
Autoriõigus 2026 Matthew D. Murphy. Kommertslitsentse väljastab
[Chiron Courseworks](https://chironcourseworks.com); päringud:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Kolmandate osapoolte teated on failis [docs/third-party-notices.md](../third-party-notices.md). Mudeli kaalud ja käitusaja teegid on eraldi allalaaditavad ja säilitavad oma litsentsid (vt [Mudeli ja sõltuvuste litsentsid](../install.md#model-and-dependency-licenses)).
