<p align="center"><img src="docs/assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*Izvorni strežnik za sklepanje GLM-OCR za NVIDIA GB10: skeni na vhodu, prepoznano besedilo na izhodu, prek API-ja, združljivega z OpenAI, brez shranjenega ogretega stanja.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Stran projekta](https://chironcourseworks.com/scriptor)** • [Hitri začetek](#quick-start) • [Namestitev](docs/install.md) • [OCR most](docs/bridge.md) • [Primerjalni test](docs/benchmark.md) • [Dokumentacija](#documentation) • [Prispevanje](#contributing) • [Licenca](#license)

[Jeziki](docs/languages.md) • [English](README.md)

</div>

## Uvod

Scriptor (latinsko »pisar«) je lokalni pogon za sklepanje modela GLM-OCR, optimiziran za prepustnost, napisan v Rustu z ročno napisanimi jedri CUDA in zgrajen za NVIDIA GB10 v DGX Spark (ARM64). GLM-OCR streže prek končne točke chat-completions, združljive z OpenAI, v zahtevku sprejema slike strani in izpostavlja metrike Prometheus `vllm:*`, združljive z vLLM, zato odjemalec ali nadzorna plošča, narejena za strežnik vLLM, deluje brez sprememb. Je lokalen in `fail-closed` (ob manjkajočem ali napačnem vhodu se ustavi, namesto da bi ugibal): imenik modela in naslov za vezavo sta izrecna, nič se ne prenaša, nič se ne odkriva samodejno, manjkajoč, zastarel ali neujemajoč se vhod pa ustavi zagon, namesto da bi se ga obšlo z ugibanjem. Seznam funkcij in pogodbo o odpornosti na sesutja najdete v [docs/serving.md](docs/serving.md).

## Ključni rezultati

Scriptor proti vLLM, isti model, dokumenti, protokol OCR mostu in GPU: 10 parnih zagonov na pogon, svež zagon (boot) pri vsakem, 368 strani na zagon, razmerja z 95-odstotnimi parnimi bootstrap intervali zaupanja.

| Metrika | Scriptor p50 | vLLM p50 | Ugotovitev |
| ------- | -----------: | -------: | ---------- |
| Celotni čas, 368 strani (s) | 236.7 | 454.8 | ZMAGA (0.520x) |
| Prepustnost (strani/uro) | 5,598 | 2,913 | ZMAGA (1.922x) |
| Boot do pripravljenosti, po prvem bootu (s) | 14.25 | 154.3 | ZMAGA (0.092x) |
| Čas do prvega žetona (s) | 5.070 | 9.167 | ZMAGA (0.553x) |
| Zahtevek end-to-end (s) | 8.339 | 13.29 | ZMAGA (0.627x) |

En stroj, ena GPU, ena seja, fiksen nabor 15 dokumentov; vsak pogon je tekel v svoji dobavljeni konfiguraciji. Postavitev, vse metrike, natančnost, omejitve in postopek reprodukcije: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Hitri začetek

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Ko so te nastavitve izvožene, pogon posluša na 127.0.0.1:8002, OCR most pa na 127.0.0.1:8765. Sliko strani pošljite pogonu kot podatkovni URL slike, celoten dokument pa mostu:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` je telo zahtevka chat-completions, katerega uporabniško sporočilo vsebuje podatkovni URL `image_url` v base64. Strežnik ničesar ne prenaša; `install.sh` je edini korak z omrežjem.

## Namestitev

`./install.sh` namesti izdajno binarno datoteko, kontrolno točko GLM-OCR, model postavitve in različico onnxruntime s CUDA za GB10; vsak korak je neobvezen in preverjen s kontrolno vsoto; `./install.sh check` preveri rezultat. Gradite lahko tudi iz izvorne kode. Glejte [docs/install.md](docs/install.md).

## OCR most

`scriptor serve` zažene tudi dokumentni OCR most: PDF-ji in slike na vhodu, prepoznano besedilo na izhodu, z zaznavanjem postavitve prek onnxruntime. Zastavice, nastavitve in vgrajeni preizkus mostu so v [docs/bridge.md](docs/bridge.md).

## Primerjalni test

Celotne tabele, natančnost, omejitve in koraki reprodukcije: [docs/benchmark.md](docs/benchmark.md).

## Odpornost na sesutja

Omejen sprejem, stopnje pritiska na pomnilnik, nadzorovani delavci in izhodne kode, na katere se lahko odzove nadzornik: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Dokumentacija

Dokumentacija (v angleščini):

- [docs/install.md](docs/install.md) — namestitveni skript, vnaprej zgrajena izdaja, gradnja iz izvorne kode, licence modelov in odvisnosti
- [docs/serving.md](docs/serving.md) — zastavice `scriptor serve`, omejitve zahtevkov, odpornost na sesutja
- [docs/bridge.md](docs/bridge.md) — OCR most, njegove zastavice in nastavitve
- [docs/benchmark.md](docs/benchmark.md) — rezultati, natančnost, omejitve, reprodukcija primerjalnega testa
- [docs/boot.md](docs/boot.md) — načrt pomnilnika ob zagonu, stopnje zagona in način merjenja časa zagona
- [docs/drafting.md](docs/drafting.md) — špekulativno dekodiranje MTP, pravilnik sprejemanja in njegove metrike
- [docs/attention.md](docs/attention.md) — jedra attention
- [docs/gemm.md](docs/gemm.md) — jedra GEMM
- [docs/equivalence.md](docs/equivalence.md) in [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — preverjanja enakovrednosti jeder in end-to-end
- [docs/engine-file.md](docs/engine-file.md) — izvorna datoteka uteži pogona
- [docs/model-facts.md](docs/model-facts.md) — dejstva o GLM-OCR, ki jim pogon ustreza
- [docs/vision-overlap.md](docs/vision-overlap.md) in [docs/nvjpeg.md](docs/nvjpeg.md) — prekrivanje vision in dekodiranje JPEG na GPU
- [seat/README.md](seat/README.md) — predloga `seat`, pogodbi o pripravljenosti in izpraznitvi, izhodne kode in vbrizgavanje napak
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — referenčni protokol merjenja
- [docs/third-party-notices.md](docs/third-party-notices.md) — obvestila tretjih oseb
- [docs/languages.md](docs/languages.md) — ta README v 38 jezikih

<a id="contributing"></a>
## Prispevanje

Najprej preberite [CONTRIBUTING.md](CONTRIBUTING.md): zadeve commitov v slogu Conventional Commit, vrata format, clippy in testi, samo sintetični testni podatki. Ranljivosti prijavite zasebno, kot je opisano v [SECURITY.md](SECURITY.md).

---

<a id="license"></a>
## Licenca

Dvojna licenca: AGPL-3.0-only ali komercialna licenca (glejte
[LICENSE](LICENSE) in [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Copyright 2026 Matthew D. Murphy. Komercialne licence izdaja
[Chiron Courseworks](https://chironcourseworks.com); poizvedbe:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Obvestila tretjih oseb so v [docs/third-party-notices.md](docs/third-party-notices.md). Uteži modelov in izvajalne knjižnice so ločeni prenosi in ohranijo svoje licence (glejte [Model and dependency licenses](docs/install.md#model-and-dependency-licenses)).
