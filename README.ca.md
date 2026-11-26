<p align="center"><img src="docs/assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*Un servidor d'inferència GLM-OCR natiu per al NVIDIA GB10: escanejos d'entrada, text reconegut de sortida, mitjançant una API compatible amb OpenAI, sense cap estat d'escalfament desat.*

[![Llicència](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Plataforma](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Pàgina del projecte](https://chironcourseworks.com/scriptor)** • [Inici ràpid](#quick-start) • [Instal·lació](docs/install.md) • [Pont OCR](docs/bridge.md) • [Benchmark](docs/benchmark.md) • [Documentació](#documentation) • [Contribuir](#contributing) • [Llicència](#license)

[Idiomes](docs/languages.md) • [English](README.md)

</div>

## Introducció

Scriptor («escrivent» en llatí) és un motor d'inferència local, optimitzat per al rendiment, dedicat al model GLM-OCR. Està escrit en Rust amb nuclis CUDA escrits a mà i pensat per al NVIDIA GB10 del DGX Spark (ARM64). Serveix GLM-OCR mitjançant un punt d'accés de chat completions compatible amb OpenAI, accepta imatges de pàgines a la petició i exposa mètriques Prometheus `vllm:*` compatibles amb vLLM, de manera que un client o un tauler fet per a un servidor vLLM funciona sense canvis. És local i `fail-closed` (s'atura en lloc d'endevinar quan falta una entrada o és incorrecta): el directori del model i l'adreça d'escolta són explícits, no es baixa res, no es descobreix res, i una entrada absent, obsoleta o incoherent atura l'arrencada en lloc de resoldre-la per suposició. Vegeu [docs/serving.md](docs/serving.md) per a la llista de funcionalitats i el contracte de resiliència davant de fallades.

## Resultats principals

Scriptor davant de vLLM, amb el mateix model, els mateixos documents, el mateix protocol de pont OCR i la mateixa GPU: 10 execucions aparellades per motor, una arrencada en fred per a cada execució, 368 pàgines per execució, ràtios amb intervals de confiança del 95 % obtinguts per bootstrap aparellat.

| Mètrica | Scriptor p50 | vLLM p50 | Veredicte |
| ------ | -----------: | -------: | ------- |
| Temps total, 368 pàgines (s) | 236.7 | 454.8 | GUANYA (0.520x) |
| Rendiment (pàgines/hora) | 5,598 | 2,913 | GUANYA (1.922x) |
| Arrencada fins a l'estat llest, després de la primera arrencada (s) | 14.25 | 154.3 | GUANYA (0.092x) |
| Temps fins al primer token (s) | 5.070 | 9.167 | GUANYA (0.553x) |
| Petició de punta a punta (s) | 8.339 | 13.29 | GUANYA (0.627x) |

Una màquina, una GPU, una sessió, un conjunt fix de 15 documents; cada motor ha usat la seva pròpia configuració distribuïda. Configuració, totes les mètriques, precisió, limitacions i com reproduir-ho: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Inici ràpid

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Amb aquests paràmetres exportats, el motor escolta a 127.0.0.1:8002 i el pont OCR a 127.0.0.1:8765. Envieu una imatge de pàgina al motor com a URL de dades d'imatge, i un document sencer al pont:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` és un cos de petició chat completions el missatge d'usuari del qual porta una URL de dades `image_url` en base64. El servidor no baixa res; `install.sh` és l'única passa que usa la xarxa.

## Instal·lació

`./install.sh` instal·la el binari de versió, el checkpoint de GLM-OCR, el model de detecció de maquetació i una compilació CUDA d'onnxruntime per al GB10; cada pas és opcional i es verifica amb suma de comprovació; `./install.sh check` en verifica el resultat. També podeu compilar des del codi font. Vegeu [docs/install.md](docs/install.md).

## Pont OCR

`scriptor serve` també executa un pont OCR de documents: PDF i imatges d'entrada, text reconegut de sortida, amb detecció de la maquetació mitjançant onnxruntime. Les opcions, els paràmetres i la comprovació integrada del pont són a [docs/bridge.md](docs/bridge.md).

## Benchmark

Taules completes, precisió, limitacions i passos de reproducció: [docs/benchmark.md](docs/benchmark.md).

## Resiliència davant de fallades

Admissió acotada, nivells de pressió de memòria, processos de treball supervisats i codis de sortida sobre els quals un supervisor pot actuar: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Documentació

Documentació (en anglès):

- [docs/install.md](docs/install.md) — script d'instal·lació, versió precompilada, compilació des del codi font, llicències del model i de les dependències
- [docs/serving.md](docs/serving.md) — opcions de `scriptor serve`, límits de les peticions, resiliència davant de fallades
- [docs/bridge.md](docs/bridge.md) — el pont OCR, les seves opcions i paràmetres
- [docs/benchmark.md](docs/benchmark.md) — resultats, precisió, limitacions, com reproduir el benchmark
- [docs/boot.md](docs/boot.md) — el pla de memòria de l'arrencada, les seves etapes i com es mesura el temps d'arrencada
- [docs/drafting.md](docs/drafting.md) — decodificació especulativa MTP, política d'acceptació i les seves mètriques
- [docs/attention.md](docs/attention.md) — nuclis d'atenció
- [docs/gemm.md](docs/gemm.md) — nuclis GEMM
- [docs/equivalence.md](docs/equivalence.md) i [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — comprovacions d'equivalència dels nuclis i de punta a punta
- [docs/engine-file.md](docs/engine-file.md) — el fitxer de pesos natiu del motor
- [docs/model-facts.md](docs/model-facts.md) — les característiques de GLM-OCR que el motor reprodueix
- [docs/vision-overlap.md](docs/vision-overlap.md) i [docs/nvjpeg.md](docs/nvjpeg.md) — solapament de la visió i descodificació JPEG a la GPU
- [seat/README.md](seat/README.md) — la plantilla `seat`, els contractes de disponibilitat i de buidatge, els codis de sortida i la injecció de fallades
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — el protocol de mesura de referència
- [docs/third-party-notices.md](docs/third-party-notices.md) — avisos de tercers
- [docs/languages.md](docs/languages.md) — aquest README en 38 idiomes

<a id="contributing"></a>
## Contribuir

Llegiu primer [CONTRIBUTING.md](CONTRIBUTING.md): assumptes de commit en format Conventional Commits, la verificació de format, clippy i proves, només dades de prova sintètiques. Informeu de les vulnerabilitats en privat tal com es descriu a [SECURITY.md](SECURITY.md).

---

<a id="license"></a>
## Llicència

Sota doble llicència: AGPL-3.0-only o una llicència comercial (vegeu
[LICENSE](LICENSE) i [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Copyright 2026 Matthew D. Murphy. Les llicències comercials les emet
[Chiron Courseworks](https://chironcourseworks.com); consultes:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Els avisos de tercers són a [docs/third-party-notices.md](docs/third-party-notices.md). Els pesos del model i les biblioteques d'execució són baixades separades i conserven les seves pròpies llicències (vegeu [Llicències del model i de les dependències](docs/install.md#model-and-dependency-licenses)).
