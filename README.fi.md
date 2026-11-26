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

*Natiivi GLM-OCR-päättelypalvelin NVIDIA GB10:lle: skannaukset sisään, tunnistettu teksti ulos, OpenAI-yhteensopivan rajapinnan kautta, ilman tallennettua lämmintä tilaa.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Projektisivu](https://chironcourseworks.com/scriptor)** • [Pikaopas](#quick-start) • [Asennus](docs/install.md) • [OCR-silta](docs/bridge.md) • [Suorituskykytesti](docs/benchmark.md) • [Dokumentaatio](#documentation) • [Osallistuminen](#contributing) • [Lisenssi](#license)

[Kielet](docs/languages.md) • [English](README.md)

</div>

## Johdanto

Scriptor (latinaksi »kirjuri») on läpimenoon optimoitu, vain paikallisesti toimiva päättelymoottori GLM-OCR-mallille. Se on kirjoitettu Rustilla käsin kirjoitetuin CUDA-ytimin ja rakennettu DGX Sparkin NVIDIA GB10:lle (ARM64). Se tarjoaa GLM-OCR:n OpenAI-yhteensopivan chat-completions-päätepisteen kautta, ottaa pyynnössä vastaan sivukuvia ja julkaisee vLLM-yhteensopivat `vllm:*`-Prometheus-mittarit, joten vLLM-palvelimelle rakennettu asiakas tai kojelauta toimii muuttumattomana. Se on paikallinen ja `fail-closed` (pysähtyy arvailun sijaan, kun syöte puuttuu tai on virheellinen): mallihakemisto ja sidontaosoite annetaan eksplisiittisesti, mitään ei ladata, mitään ei etsitä automaattisesti, ja puuttuva, vanhentunut tai epäsopiva syöte pysäyttää käynnistyksen sen sijaan, että sitä arvailtaisiin. Ominaisuusluettelo ja kaatumisturvallisuussopimus ovat tiedostossa [docs/serving.md](docs/serving.md).

## Keskeiset tulokset

Scriptor vLLM:ää vastaan, sama malli, samat dokumentit, sama OCR-siltaprotokolla ja sama GPU: 10 pariutettua ajoa moottoria kohden, jokaiselle ajolle uusi käynnistys, 368 sivua ajoa kohden, suhdeluvut 95 %:n pariutetuin bootstrap-luottamusvälein.

| Mittari | Scriptor p50 | vLLM p50 | Tulos |
| ------- | -----------: | -------: | ----- |
| Seinäkelloaika, 368 sivua (s) | 236.7 | 454.8 | VOITTO (0.520x) |
| Läpimeno (sivua/tunti) | 5,598 | 2,913 | VOITTO (1.922x) |
| Käynnistys valmiiksi, ensimmäisen käynnistyksen jälkeen (s) | 14.25 | 154.3 | VOITTO (0.092x) |
| Aika ensimmäiseen tokeniin (s) | 5.070 | 9.167 | VOITTO (0.553x) |
| Pyyntö päästä päähän (s) | 8.339 | 13.29 | VOITTO (0.627x) |

Yksi kone, yksi GPU, yksi istunto, kiinteä 15 dokumentin joukko; kumpikin moottori ajettiin omalla toimitetulla kokoonpanollaan. Asetelma, kaikki mittarit, tarkkuus, rajoitukset ja toistaminen: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Pikaopas

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Kun nämä asetukset on viety ympäristöön, moottori kuuntelee osoitteessa 127.0.0.1:8002 ja OCR-silta osoitteessa 127.0.0.1:8765. Lähetä sivukuva moottorille kuvan data-URL:nä ja kokonainen dokumentti sillalle:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` on chat-completions-runko, jonka käyttäjäviesti sisältää base64-koodatun `image_url`-data-URL:n. Palvelin ei lataa mitään; `install.sh` on ainoa verkkoa käyttävä vaihe.

## Asennus

`./install.sh` asentaa julkaisubinäärin, GLM-OCR-tarkistuspisteen, asettelumallin ja GB10:lle tehdyn CUDA-version onnxruntimesta. Jokainen vaihe on valinnainen ja tarkistussummalla varmennettu; `./install.sh check` varmistaa lopputuloksen. Voit myös rakentaa lähdekoodista. Katso [docs/install.md](docs/install.md).

## OCR-silta

`scriptor serve` ajaa myös dokumenttien OCR-siltaa: PDF-tiedostot ja kuvat sisään, tunnistettu teksti ulos, asettelun tunnistus onnxruntimen kautta. Liput, asetukset ja sisäänrakennettu siltatarkistus on kuvattu tiedostossa [docs/bridge.md](docs/bridge.md).

## Suorituskykytesti

Täydet taulukot, tarkkuus, rajoitukset ja toistamisen vaiheet: [docs/benchmark.md](docs/benchmark.md).

## Kaatumisturvallisuus

Rajattu vastaanotto, muistipainetasot, valvotut työprosessit ja poistumiskoodit, joihin valvoja voi reagoida: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Dokumentaatio

Dokumentaatio (englanniksi):

- [docs/install.md](docs/install.md) — asennusskripti, valmis julkaisu, rakentaminen lähdekoodista, mallien ja riippuvuuksien lisenssit
- [docs/serving.md](docs/serving.md) — `scriptor serve` -liput, pyyntörajat, kaatumisturvallisuus
- [docs/bridge.md](docs/bridge.md) — OCR-silta, sen liput ja asetukset
- [docs/benchmark.md](docs/benchmark.md) — tulokset, tarkkuus, rajoitukset, suorituskykytestin toistaminen
- [docs/boot.md](docs/boot.md) — käynnistyksen muistisuunnitelma, käynnistysvaiheet ja käynnistysajan mittaaminen
- [docs/drafting.md](docs/drafting.md) — MTP-spekulatiivinen dekoodaus, hyväksymiskäytäntö ja sen mittarit
- [docs/attention.md](docs/attention.md) — attention-ytimet
- [docs/gemm.md](docs/gemm.md) — GEMM-ytimet
- [docs/equivalence.md](docs/equivalence.md) ja [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — ytimien ja päästä päähän -ekvivalenssitarkistukset
- [docs/engine-file.md](docs/engine-file.md) — moottorin oma painotiedosto
- [docs/model-facts.md](docs/model-facts.md) — GLM-OCR-tiedot, joita moottori noudattaa
- [docs/vision-overlap.md](docs/vision-overlap.md) ja [docs/nvjpeg.md](docs/nvjpeg.md) — kuvankäsittelyn limittäminen ja JPEG-dekoodaus GPU:lla
- [seat/README.md](seat/README.md) — `seat`-malli, valmiuden ja tyhjennyksen sopimukset, poistumiskoodit ja vikainjektio
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — mittauksen vertailuprotokolla
- [docs/third-party-notices.md](docs/third-party-notices.md) — kolmannen osapuolen ilmoitukset
- [docs/languages.md](docs/languages.md) — tämä README 38 kielellä

<a id="contributing"></a>
## Osallistuminen

Lue ensin [CONTRIBUTING.md](CONTRIBUTING.md): Conventional Commit -otsikot, muotoilun, clippyn ja testien portti, vain synteettiset testiaineistot. Ilmoita haavoittuvuudet yksityisesti tiedoston [SECURITY.md](SECURITY.md) ohjeen mukaan.

---

<a id="license"></a>
## Lisenssi

Kaksoislisensoitu: AGPL-3.0-only tai kaupallinen lisenssi (katso
[LICENSE](LICENSE) ja [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Tekijänoikeus 2026 Matthew D. Murphy. Kaupalliset lisenssit myöntää
[Chiron Courseworks](https://chironcourseworks.com); tiedustelut:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Kolmannen osapuolen ilmoitukset ovat tiedostossa [docs/third-party-notices.md](docs/third-party-notices.md). Mallipainot ja ajonaikaiset kirjastot ladataan erikseen, ja niihin pätevät niiden omat lisenssit (katso [Model and dependency licenses](docs/install.md#model-and-dependency-licenses)).
