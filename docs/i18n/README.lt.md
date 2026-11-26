<p align="center"><img src="../assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*Vietinis GLM-OCR inferencijos serveris NVIDIA GB10: skenuoti vaizdai įeina, atpažintas tekstas išeina per su OpenAI suderinamą API, be jokios išsaugotos įšilimo būsenos.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Projekto puslapis](https://chironcourseworks.com/scriptor)** • [Greita pradžia](#quick-start) • [Diegimas](../install.md) • [OCR tiltas](../bridge.md) • [Našumo testas](../benchmark.md) • [Dokumentacija](#documentation) • [Prisidėjimas](#contributing) • [Licencija](#license)

[Kalbos](../languages.md) • [English](../../README.md)

</div>

## Įvadas

Scriptor (lotyniškai „perrašinėtojas“) yra pralaidumui optimizuotas, tik vietinis GLM-OCR modelio inferencijos variklis, parašytas Rust kalba su ranka rašytais CUDA branduoliais ir sukurtas NVIDIA GB10, esančiam DGX Spark (ARM64). Jis aptarnauja GLM-OCR per su OpenAI suderinamą chat-completions galinį tašką, užklausoje priima puslapių vaizdus ir pateikia su vLLM suderinamas `vllm:*` Prometheus metrikas, todėl vLLM serveriui sukurtas klientas ar skydelis veikia be pakeitimų. Jis yra vietinis ir `fail-closed` (sustoja, užuot spėliojęs, kai trūksta įvesties ar ji klaidinga): modelio katalogas ir susiejimo adresas nurodomi aiškiai, niekas nesisiunčia, niekas neaptinkama automatiškai, o trūkstama, pasenusi ar neatitinkanti įvestis sustabdo paleidimą, užuot buvusi spėliojama. Funkcijų sąrašą ir atsparumo gedimams sutartį rasite [docs/serving.md](../serving.md).

## Pagrindiniai rezultatai

Scriptor palyginti su vLLM: tas pats modelis, dokumentai, OCR tilto protokolas ir GPU; 10 suporuotų paleidimų kiekvienam varikliui, kiekvieną kartą šviežias paleidimas, 368 puslapiai per paleidimą, santykiai su 95 % suporuotais bootstrap pasikliautinaisiais intervalais.

| Metrika | Scriptor p50 | vLLM p50 | Verdiktas |
| ------ | -----------: | -------: | ------- |
| Visas laikas, 368 puslapiai (s) | 236.7 | 454.8 | LAIMĖJIMAS (0.520x) |
| Pralaidumas (puslapių/val.) | 5,598 | 2,913 | LAIMĖJIMAS (1.922x) |
| Paleidimas iki parengties, po pirmojo paleidimo (s) | 14.25 | 154.3 | LAIMĖJIMAS (0.092x) |
| Laikas iki pirmos žymės (s) | 5.070 | 9.167 | LAIMĖJIMAS (0.553x) |
| Užklausa nuo pradžios iki pabaigos (s) | 8.339 | 13.29 | LAIMĖJIMAS (0.627x) |

Viena mašina, vienas GPU, viena sesija, fiksuotas 15 dokumentų rinkinys; kiekvienas variklis veikė su savo tiekiama konfigūracija. Sąranka, visos metrikos, tikslumas, apribojimai ir kaip atkartoti: [docs/benchmark.md](../benchmark.md).

<a id="quick-start"></a>
## Greita pradžia

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Su šiais nustatymais variklis klauso 127.0.0.1:8002, o OCR tiltas – 127.0.0.1:8765. Puslapio vaizdą varikliui siųskite kaip vaizdo data URL, o visą dokumentą – tiltui:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` yra chat-completions užklausos turinys, kurio vartotojo žinutėje yra base64 `image_url` data URL. Serveris nieko nesisiunčia; `install.sh` yra vienintelis veiksmas, naudojantis tinklą.

## Diegimas

`./install.sh` įdiegia išleistą vykdomąjį failą, GLM-OCR kontrolinį tašką, maketo modelį ir GB10 skirtą CUDA onnxruntime versiją; kiekvienas žingsnis pasirenkamas ir tikrinamas kontrolinėmis sumomis; `./install.sh check` patikrina rezultatą. Taip pat galima kompiliuoti iš šaltinio. Žr. [docs/install.md](../install.md).

## OCR tiltas

`scriptor serve` taip pat paleidžia dokumentų OCR tiltą: PDF ir vaizdai įeina, atpažintas tekstas išeina, o maketas aptinkamas per onnxruntime. Vėliavėlės, nustatymai ir įtaisytasis tilto patikrinimas aprašyti [docs/bridge.md](../bridge.md).

## Našumo testas

Pilnos lentelės, tikslumas, apribojimai ir atkartojimo žingsniai: [docs/benchmark.md](../benchmark.md).

## Atsparumas gedimams

Ribotas priėmimas, atminties apkrovos pakopos, prižiūrimi darbininkai ir išėjimo kodai, kuriais gali remtis prižiūrėtojas: [docs/serving.md](../serving.md#crash-safety).

<a id="documentation"></a>
## Dokumentacija

Dokumentacija (anglų kalba):

- [docs/install.md](../install.md) — diegimo scenarijus, paruoštas leidimas, kompiliavimas iš šaltinio, modelių ir priklausomybių licencijos
- [docs/serving.md](../serving.md) — `scriptor serve` vėliavėlės, užklausų ribos, atsparumas gedimams
- [docs/bridge.md](../bridge.md) — OCR tiltas, jo vėliavėlės ir nustatymai
- [docs/benchmark.md](../benchmark.md) — rezultatai, tikslumas, apribojimai, našumo testo atkartojimas
- [docs/boot.md](../boot.md) — paleidimo atminties planas, paleidimo etapai ir kaip matuojamas paleidimo laikas
- [docs/drafting.md](../drafting.md) — MTP spekuliatyvusis dekodavimas, priėmimo politika ir jos metrikos
- [docs/attention.md](../attention.md) — dėmesio (attention) branduoliai
- [docs/gemm.md](../gemm.md) — GEMM branduoliai
- [docs/equivalence.md](../equivalence.md) ir [docs/e2e-equivalence.md](../e2e-equivalence.md) — branduolių ir visos grandinės ekvivalentiškumo patikros
- [docs/engine-file.md](../engine-file.md) — variklio savasis svorių failas
- [docs/model-facts.md](../model-facts.md) — GLM-OCR faktai, kuriuos atitinka variklis
- [docs/vision-overlap.md](../vision-overlap.md) ir [docs/nvjpeg.md](../nvjpeg.md) — vaizdo apdorojimo persidengimas ir GPU JPEG dekodavimas
- [seat/README.md](../../seat/README.md) — `seat` šablonas, parengties ir ištuštinimo sutartys, išėjimo kodai ir gedimų injekcija
- [docs/phase0-protocol.md](../phase0-protocol.md) — etaloninis matavimo protokolas
- [docs/third-party-notices.md](../third-party-notices.md) — trečiųjų šalių pranešimai
- [docs/languages.md](../languages.md) — šis README 38 kalbomis

<a id="contributing"></a>
## Prisidėjimas

Pirmiausia perskaitykite [CONTRIBUTING.md](../../CONTRIBUTING.md): Conventional Commit antraštės, formatavimo, clippy ir testų vartai, tik sintetiniai bandymų duomenys. Apie pažeidžiamumus pranešykite privačiai, kaip aprašyta [SECURITY.md](../../SECURITY.md).

---

<a id="license"></a>
## Licencija

Licencijuojama pagal dvigubą licenciją: AGPL-3.0-only arba komercinė licencija (žr.
[LICENSE](../../LICENSE) ir [COMMERCIAL-LICENSE.md](../../COMMERCIAL-LICENSE.md)).
Autorių teisės 2026 Matthew D. Murphy. Komercinės licencijos išduodamos per
[Chiron Courseworks](https://chironcourseworks.com); užklausoms:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Trečiųjų šalių pranešimai pateikti [docs/third-party-notices.md](../third-party-notices.md). Modelių svoriai ir vykdymo bibliotekos atsisiunčiami atskirai ir išlaiko savo licencijas (žr. [Model and dependency licenses](../install.md#model-and-dependency-licenses)).
