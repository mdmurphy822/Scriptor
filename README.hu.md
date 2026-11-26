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

*Natív GLM-OCR inferenciaszerver az NVIDIA GB10-hez: szkennelt oldalak be, felismert szöveg ki, OpenAI-kompatibilis API-n keresztül, mentett meleg állapot nélkül.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Projektoldal](https://chironcourseworks.com/scriptor)** • [Gyors kezdés](#quick-start) • [Telepítés](docs/install.md) • [OCR-híd](docs/bridge.md) • [Benchmark](docs/benchmark.md) • [Dokumentáció](#documentation) • [Közreműködés](#contributing) • [Licenc](#license)

[Nyelvek](docs/languages.md) • [English](README.md)

</div>

## Bevezetés

A Scriptor (latinul „írnok”) áteresztőképességre optimalizált, kizárólag helyi futtatású inferenciamotor a GLM-OCR modellhez, Rustban írva, kézzel írt CUDA-kernelekkel, a DGX Sparkban található NVIDIA GB10-re (ARM64) építve. A GLM-OCR-t OpenAI-kompatibilis chat-completions végponton szolgálja ki, a kérésben fogadja az oldalképeket, és vLLM-kompatibilis `vllm:*` Prometheus-metrikákat tesz elérhetővé, így a vLLM-szerverhez készült kliensek és irányítópultok módosítás nélkül működnek. Helyi és `fail-closed` (leáll ahelyett, hogy találgatna, ha egy bemenet hiányzik vagy hibás): a modellkönyvtár és a kötési cím explicit, semmi nem töltődik le, semmi nem kerül felderítésre, és a hiányzó, elavult vagy nem egyező bemenet megállítja az indítást ahelyett, hogy találgatással kerülnék meg. A funkciólistát és az összeomlásbiztonsági szerződést lásd a [docs/serving.md](docs/serving.md) fájlban.

## Fő eredmények

A Scriptor a vLLM-mel szemben, azonos modell, dokumentumok, OCR-híd protokoll és GPU mellett: motoronként 10 páros futás, minden futás friss bootolással, futásonként 368 oldal, az arányok 95%-os páros bootstrap konfidenciaintervallummal.

| Metrika | Scriptor p50 | vLLM p50 | Eredmény |
| ------- | -----------: | -------: | -------- |
| Falióra-idő, 368 oldal (s) | 236.7 | 454.8 | GYŐZELEM (0.520x) |
| Áteresztőképesség (oldal/óra) | 5,598 | 2,913 | GYŐZELEM (1.922x) |
| Boot a készenlétig, az első boot után (s) | 14.25 | 154.3 | GYŐZELEM (0.092x) |
| Idő az első tokenig (s) | 5.070 | 9.167 | GYŐZELEM (0.553x) |
| Végponttól végpontig tartó kérés (s) | 8.339 | 13.29 | GYŐZELEM (0.627x) |

Egy gép, egy GPU, egy munkamenet, rögzített 15 dokumentumos készlet; mindkét motor a saját szállított konfigurációjával futott. A beállítások, az összes metrika, a pontosság, a korlátok és a reprodukálás módja: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Gyors kezdés

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Ezekkel a beállításokkal exportálva a motor a 127.0.0.1:8002 címen, az OCR-híd a 127.0.0.1:8765 címen figyel. Az oldalképet képi data URL-ként küldje a motornak, a teljes dokumentumot pedig a hídnak:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

A `request.json` egy chat-completions kérés törzse, amelynek felhasználói üzenete base64 kódolású `image_url` data URL-t tartalmaz. A szerver semmit nem tölt le; az `install.sh` az egyetlen hálózatot használó lépés.

## Telepítés

Az `./install.sh` telepíti a kiadási binárist, a GLM-OCR checkpointot, az elrendezésfelismerő modellt és az onnxruntime CUDA-s buildjét a GB10-hez; minden lépés opcionális és ellenőrzőösszeggel védett; az `./install.sh check` ellenőrzi az eredményt. Forrásból is fordíthat. Lásd: [docs/install.md](docs/install.md).

## OCR-híd

A `scriptor serve` dokumentum-OCR-hidat is futtat: PDF-ek és képek be, felismert szöveg ki, az onnxruntime-on keresztüli elrendezésfelismeréssel. A kapcsolók, beállítások és a beépített hídellenőrzés a [docs/bridge.md](docs/bridge.md) fájlban találhatók.

## Benchmark

Teljes táblázatok, pontosság, korlátok és a reprodukálás lépései: [docs/benchmark.md](docs/benchmark.md).

## Összeomlásbiztonság

Korlátozott bemeneti fogadás, memórianyomási szintek, felügyelt workerek és a felügyelő által kezelhető kilépési kódok: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Dokumentáció

Dokumentáció (angolul):

- [docs/install.md](docs/install.md) — telepítőszkript, előre épített kiadás, fordítás forrásból, a modell és a függőségek licencei
- [docs/serving.md](docs/serving.md) — a `scriptor serve` kapcsolói, kérési korlátok, összeomlásbiztonság
- [docs/bridge.md](docs/bridge.md) — az OCR-híd, kapcsolói és beállításai
- [docs/benchmark.md](docs/benchmark.md) — eredmények, pontosság, korlátok, a benchmark reprodukálása
- [docs/boot.md](docs/boot.md) — a boot memóriaterve, a bootfázisok és a bootidő mérése
- [docs/drafting.md](docs/drafting.md) — MTP spekulatív dekódolás, az elfogadási szabályzat és metrikái
- [docs/attention.md](docs/attention.md) — attention-kernelek
- [docs/gemm.md](docs/gemm.md) — GEMM-kernelek
- [docs/equivalence.md](docs/equivalence.md) és [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — kernel- és végponttól végpontig tartó ekvivalenciaellenőrzések
- [docs/engine-file.md](docs/engine-file.md) — a motor natív súlyfájlja
- [docs/model-facts.md](docs/model-facts.md) — a GLM-OCR tulajdonságai, amelyeket a motor követ
- [docs/vision-overlap.md](docs/vision-overlap.md) és [docs/nvjpeg.md](docs/nvjpeg.md) — a képfeldolgozás átfedése és a GPU-s JPEG-dekódolás
- [seat/README.md](seat/README.md) — a `seat` sablon, a készenléti és drain-szerződések, kilépési kódok és hibainjektálás
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — a referencia-mérési protokoll
- [docs/third-party-notices.md](docs/third-party-notices.md) — harmadik féltől származó szoftverek közlései
- [docs/languages.md](docs/languages.md) — ez a README 38 nyelven

<a id="contributing"></a>
## Közreműködés

Először olvassa el a [CONTRIBUTING.md](CONTRIBUTING.md) fájlt: Conventional Commit tárgysorok, formátum-, clippy- és tesztkapu, kizárólag szintetikus tesztadatok. A sérülékenységeket privátban jelentse a [SECURITY.md](SECURITY.md) leírása szerint.

---

<a id="license"></a>
## Licenc

Kettős licenc: AGPL-3.0-only vagy kereskedelmi licenc (lásd:
[LICENSE](LICENSE) és [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Copyright 2026 Matthew D. Murphy. A kereskedelmi licenceket a
[Chiron Courseworks](https://chironcourseworks.com) adja ki; megkeresések:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

A harmadik féltől származó szoftverek közlései a [docs/third-party-notices.md](docs/third-party-notices.md) fájlban találhatók. A modellsúlyok és a futásidejű könyvtárak külön letöltések, és megtartják saját licencüket (lásd: [Model and dependency licenses](docs/install.md#model-and-dependency-licenses)).
