<p align="center"><img src="docs/assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>


<div align="center">

*Nativni poslužitelj za GLM-OCR inferenciju na NVIDIA GB10: skenovi na ulazu, prepoznati tekst na izlazu, preko OpenAI-kompatibilnog API-ja, bez spremljenog toplog stanja.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Stranica projekta](https://chironcourseworks.com/scriptor)** • [Brzi početak](#quick-start) • [Instalacija](docs/install.md) • [OCR most](docs/bridge.md) • [Benchmark](docs/benchmark.md) • [Dokumentacija](#documentation) • [Doprinos](#contributing) • [Licenca](#license)

[Jezici](docs/languages.md) • [English](README.md)

</div>

## Uvod

Scriptor (latinski za „pisara”) je lokalni inferencijski engine za model GLM-OCR, optimiziran za propusnost, napisan u Rustu s ručno pisanim CUDA kernelima i izgrađen za NVIDIA GB10 u DGX Sparku (ARM64). Poslužuje GLM-OCR preko OpenAI-kompatibilne chat-completions krajnje točke, prihvaća slike stranica u zahtjevu i izlaže `vllm:*` Prometheus metrike kompatibilne s vLLM-om, pa klijent ili nadzorna ploča izrađeni za vLLM poslužitelj rade bez izmjena. Radi lokalno i `fail-closed` (zaustavlja se umjesto da pogađa kad ulaz nedostaje ili je pogrešan): direktorij modela i adresa vezanja su eksplicitni, ništa se ne preuzima, ništa se ne otkriva samo, a nedostajući, zastarjeli ili neusklađeni ulaz zaustavlja pokretanje umjesto da se zaobiđe pogađanjem. Popis značajki i ugovor o otpornosti na rušenja nalaze se u [docs/serving.md](docs/serving.md).

## Glavni rezultati

Scriptor naspram vLLM-a, isti model, isti dokumenti, isti protokol OCR mosta i isti GPU: 10 uparenih pokretanja po engineu, svako pokretanje s novim bootom, 368 stranica po pokretanju, omjeri s 95 % uparenim bootstrap intervalima pouzdanosti.

| Metrika | Scriptor p50 | vLLM p50 | Ishod |
| ------- | -----------: | -------: | ----- |
| Ukupno vrijeme, 368 stranica (s) | 236.7 | 454.8 | POBJEDA (0.520x) |
| Propusnost (stranica/sat) | 5,598 | 2,913 | POBJEDA (1.922x) |
| Boot do spremnosti, nakon prvog boota (s) | 14.25 | 154.3 | POBJEDA (0.092x) |
| Vrijeme do prvog tokena (s) | 5.070 | 9.167 | POBJEDA (0.553x) |
| Zahtjev od početka do kraja (s) | 8.339 | 13.29 | POBJEDA (0.627x) |

Jedan stroj, jedan GPU, jedna sesija, fiksni skup od 15 dokumenata; svaki engine pokretan je u vlastitoj isporučenoj konfiguraciji. Postavke, sve metrike, točnost, ograničenja i način reprodukcije: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Brzi početak

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Kad su te postavke izvezene, engine sluša na 127.0.0.1:8002, a OCR most na 127.0.0.1:8765. Sliku stranice šaljite engineu kao data URL slike, a cijeli dokument mostu:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` je tijelo chat-completions zahtjeva čija korisnička poruka nosi base64 `image_url` data URL. Poslužitelj ništa ne preuzima; `install.sh` jedini je korak koji koristi mrežu.

## Instalacija

`./install.sh` instalira izdanu binarnu datoteku, GLM-OCR checkpoint, model rasporeda stranice i CUDA izgradnju onnxruntimea za GB10; svaki korak je opcionalan i provjeren kontrolnim zbrojem; `./install.sh check` provjerava rezultat. Možete i izgraditi iz izvornog koda. Pogledajte [docs/install.md](docs/install.md).

## OCR most

`scriptor serve` pokreće i OCR most za dokumente: PDF-ovi i slike na ulazu, prepoznati tekst na izlazu, s detekcijom rasporeda preko onnxruntimea. Zastavice, postavke i ugrađena provjera mosta nalaze se u [docs/bridge.md](docs/bridge.md).

## Benchmark

Potpune tablice, točnost, ograničenja i koraci reprodukcije: [docs/benchmark.md](docs/benchmark.md).

## Otpornost na rušenja

Ograničen prihvat zahtjeva, razine pritiska na memoriju, nadzirani workeri i izlazni kodovi na koje nadzornik može reagirati: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Dokumentacija

Dokumentacija (na engleskom):

- [docs/install.md](docs/install.md) — instalacijska skripta, gotovo izdanje, izgradnja iz izvornog koda, licence modela i ovisnosti
- [docs/serving.md](docs/serving.md) — zastavice `scriptor serve`, ograničenja zahtjeva, otpornost na rušenja
- [docs/bridge.md](docs/bridge.md) — OCR most, njegove zastavice i postavke
- [docs/benchmark.md](docs/benchmark.md) — rezultati, točnost, ograničenja, reprodukcija benchmarka
- [docs/boot.md](docs/boot.md) — plan memorije pri bootu, faze boota i način mjerenja vremena boota
- [docs/drafting.md](docs/drafting.md) — MTP spekulativno dekodiranje, politika prihvaćanja i njezine metrike
- [docs/attention.md](docs/attention.md) — kerneli pažnje (attention)
- [docs/gemm.md](docs/gemm.md) — GEMM kerneli
- [docs/equivalence.md](docs/equivalence.md) i [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — provjere ekvivalencije kernela i od početka do kraja
- [docs/engine-file.md](docs/engine-file.md) — datoteka težina izvorna za engine
- [docs/model-facts.md](docs/model-facts.md) — činjenice o GLM-OCR-u koje engine poštuje
- [docs/vision-overlap.md](docs/vision-overlap.md) i [docs/nvjpeg.md](docs/nvjpeg.md) — preklapanje obrade slike i GPU JPEG dekodiranje
- [seat/README.md](seat/README.md) — predložak `seat`, ugovori o spremnosti i drainu, izlazni kodovi i injekcija grešaka
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — referentni protokol mjerenja
- [docs/third-party-notices.md](docs/third-party-notices.md) — obavijesti o softveru trećih strana
- [docs/languages.md](docs/languages.md) — ovaj README na 38 jezika

<a id="contributing"></a>
## Doprinos

Prvo pročitajte [CONTRIBUTING.md](CONTRIBUTING.md): Conventional Commit naslovi, provjere formata, clippyja i testova, samo sintetički testni podaci. Ranjivosti prijavite privatno kako je opisano u [SECURITY.md](SECURITY.md).

---

<a id="license"></a>
## Licenca

Dvostruko licencirano: AGPL-3.0-only ili komercijalna licenca (vidi
[LICENSE](LICENSE) i [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Autorska prava 2026 Matthew D. Murphy. Komercijalne licence izdaje
[Chiron Courseworks](https://chironcourseworks.com); upiti:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Obavijesti o softveru trećih strana nalaze se u [docs/third-party-notices.md](docs/third-party-notices.md). Težine modela i runtime biblioteke zasebna su preuzimanja i zadržavaju vlastite licence (vidi [Model and dependency licenses](docs/install.md#model-and-dependency-licenses)).
