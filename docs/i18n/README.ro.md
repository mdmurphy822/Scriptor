<p align="center"><img src="../assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*Un server nativ de inferență GLM-OCR pentru NVIDIA GB10: scanări la intrare, text recunoscut la ieșire, printr-un API compatibil OpenAI, fără nicio stare de warm-up salvată.*

[![Licență](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platformă](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Pagina proiectului](https://chironcourseworks.com/scriptor)** • [Pornire rapidă](#quick-start) • [Instalare](../install.md) • [Bridge OCR](../bridge.md) • [Benchmark](../benchmark.md) • [Documentație](#documentation) • [Contribuții](#contributing) • [Licență](#license)

[Limbi](../languages.md) • [English](../../README.md)

</div>

## Introducere

Scriptor (din latină „scriitor, copist”) este un motor de inferență local pentru modelul GLM-OCR, optimizat pentru throughput, scris în Rust cu kernel-uri CUDA scrise de mână și construit pentru NVIDIA GB10 din DGX Spark (ARM64). Servește GLM-OCR printr-un endpoint chat-completions compatibil OpenAI, acceptă imagini ale paginilor în cerere și expune metrici Prometheus `vllm:*` compatibile vLLM, astfel încât un client sau un dashboard construit pentru un server vLLM funcționează fără modificări. Este local și `fail-closed` (se oprește în loc să ghicească): directorul modelului și adresa de bind sunt explicite, nu se descarcă nimic, nu se descoperă nimic, iar o intrare lipsă, învechită sau neconcordantă oprește pornirea în loc să fie ocolită prin presupuneri. Lista de funcționalități și contractul de crash safety sunt în [docs/serving.md](../serving.md).

<a id="headline-results"></a>
## Rezultate principale

Scriptor față de vLLM, același model, aceleași documente, același protocol al bridge-ului OCR și aceeași GPU: 10 rulări perechi pentru fiecare motor, un boot de la zero la fiecare rulare, 368 de pagini pe rulare, rapoarte cu intervale de încredere bootstrap pereche de 95%.

| Metrică | Scriptor p50 | vLLM p50 | Verdict |
| ------- | -----------: | -------: | ------- |
| Timp total, 368 de pagini (s) | 236.7 | 454.8 | VICTORIE (0.520x) |
| Throughput (pagini/oră) | 5,598 | 2,913 | VICTORIE (1.922x) |
| Boot până la starea gata, după primul boot (s) | 14.25 | 154.3 | VICTORIE (0.092x) |
| Timp până la primul token (s) | 5.070 | 9.167 | VICTORIE (0.553x) |
| Cerere end-to-end (s) | 8.339 | 13.29 | VICTORIE (0.627x) |

O singură mașină, o singură GPU, o singură sesiune, un set fix de 15 documente; fiecare motor a rulat cu propria configurație livrată. Setup, toate metricile, acuratețe, limitări și cum se reproduc rezultatele: [docs/benchmark.md](../benchmark.md).

<a id="quick-start"></a>
## Pornire rapidă

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Cu aceste setări exportate, motorul ascultă pe 127.0.0.1:8002, iar bridge-ul OCR pe 127.0.0.1:8765. Trimite imaginea unei pagini către motor ca data URL de imagine și un document întreg către bridge:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` este un corp chat-completions al cărui mesaj de utilizator conține un data URL `image_url` în base64. Serverul nu descarcă nimic; `install.sh` este singurul pas care folosește rețeaua.

## Instalare

`./install.sh` instalează binarul de release, checkpoint-ul GLM-OCR, modelul de layout și o variantă CUDA de onnxruntime pentru GB10, fiecare pas fiind opțional și cu verificarea sumei de control; `./install.sh check` verifică rezultatul. Se poate compila și din sursă. Vezi [docs/install.md](../install.md).

## Bridge OCR

`scriptor serve` rulează și un bridge OCR pentru documente: PDF-uri și imagini la intrare, text recunoscut la ieșire, cu detecția layout-ului prin onnxruntime. Flag-urile, setările și verificarea integrată a bridge-ului sunt în [docs/bridge.md](../bridge.md).

## Benchmark

Tabele complete, acuratețe, limitări și pașii de reproducere: [docs/benchmark.md](../benchmark.md).

## Crash safety

Admisie limitată, trepte de presiune a memoriei, workeri supravegheați și coduri de ieșire pe baza cărora un supervisor poate acționa: [docs/serving.md](../serving.md#crash-safety).

<a id="documentation"></a>
## Documentație

Documentație (în engleză):

- [docs/install.md](../install.md) — scriptul de instalare, release precompilat, compilare din sursă, licențele modelelor și ale dependențelor
- [docs/serving.md](../serving.md) — flag-urile `scriptor serve`, limitele cererilor, crash safety
- [docs/bridge.md](../bridge.md) — bridge-ul OCR, flag-urile și setările sale
- [docs/benchmark.md](../benchmark.md) — rezultate, acuratețe, limitări, reproducerea benchmarkului
- [docs/boot.md](../boot.md) — planul de memorie la boot, etapele de boot și cum se măsoară timpul de boot
- [docs/drafting.md](../drafting.md) — decodarea speculativă MTP, politica de acceptare și metricile ei
- [docs/attention.md](../attention.md) — kernel-uri de attention
- [docs/gemm.md](../gemm.md) — kernel-uri GEMM
- [docs/equivalence.md](../equivalence.md) și [docs/e2e-equivalence.md](../e2e-equivalence.md) — verificări de echivalență a kernel-urilor și end-to-end
- [docs/engine-file.md](../engine-file.md) — fișierul de ponderi nativ al motorului
- [docs/model-facts.md](../model-facts.md) — datele despre GLM-OCR cu care se aliniază motorul
- [docs/vision-overlap.md](../vision-overlap.md) și [docs/nvjpeg.md](../nvjpeg.md) — suprapunerea vizuală și decodarea JPEG pe GPU
- [seat/README.md](../../seat/README.md) — șablonul `seat`, contractele de readiness și drain, codurile de ieșire și fault injection
- [docs/phase0-protocol.md](../phase0-protocol.md) — protocolul de măsurare de referință
- [docs/third-party-notices.md](../third-party-notices.md) — notificări despre terți
- [docs/languages.md](../languages.md) — acest README în 38 de limbi

<a id="contributing"></a>
## Contribuții

Citește mai întâi [CONTRIBUTING.md](../../CONTRIBUTING.md): subiecte de commit în stilul Conventional Commit, poarta de formatare, clippy și teste, doar fixture-uri sintetice. Raportează vulnerabilitățile în privat, așa cum este descris în [SECURITY.md](../../SECURITY.md).

---

<a id="license"></a>
## Licență

Licențiat dual: AGPL-3.0-only sau o licență comercială (vezi
[LICENSE](../../LICENSE) și [COMMERCIAL-LICENSE.md](../../COMMERCIAL-LICENSE.md)).
Copyright 2026 Matthew D. Murphy. Licențele comerciale se emit prin
[Chiron Courseworks](https://chironcourseworks.com); solicitări:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Notificările despre terți se găsesc în [docs/third-party-notices.md](../third-party-notices.md). Ponderile modelelor și bibliotecile de runtime sunt descărcări separate și își păstrează propriile licențe (vezi [Model and dependency licenses](../install.md#model-and-dependency-licenses)).
