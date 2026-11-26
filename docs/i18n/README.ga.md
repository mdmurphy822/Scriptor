<p align="center"><img src="../assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*Freastalaí tátail GLM-OCR dúchasach don NVIDIA GB10: scananna isteach, téacs aitheanta amach, thar API atá comhoiriúnach le OpenAI, gan aon staid théite shábháilte.*

[![Ceadúnas](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Ardán](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Leathanach an tionscadail](https://chironcourseworks.com/scriptor)** • [Tosú tapa](#quick-start) • [Suiteáil](../install.md) • [Droichead OCR](../bridge.md) • [Tagarmharc](../benchmark.md) • [Cáipéisíocht](#documentation) • [Ag cur leis](#contributing) • [Ceadúnas](#license)

[Teangacha](../languages.md) • [English](../../README.md)

</div>

## Réamhrá

Is inneall tátail áitiúil é Scriptor (Laidin do "scríobhaí") don mhúnla GLM-OCR, optamaithe don tréchur, scríofa i Rust le eithne CUDA lámhscríofa agus tógtha don NVIDIA GB10 sa DGX Spark (ARM64). Freastalaíonn sé GLM-OCR trí chríochphointe chomhoiriúnach le OpenAI do chomhlánuithe comhrá, glacann sé le híomhánna leathanach sa iarratas, agus nochtann sé méadrachtaí Prometheus `vllm:*` atá comhoiriúnach le vLLM, ionas go n-oibríonn cliant nó painéal a tógadh do fhreastalaí vLLM gan athrú. Tá sé áitiúil agus is `fail-closed` é (stopann sé in ionad buille faoi thuairim a dhéanamh nuair a bhíonn ionchur ar iarraidh nó mícheart): tá eolaire an mhúnla agus seoladh an cheangail sainráite, ní íoslódáiltear aon rud, ní aimsítear aon rud go huathoibríoch, agus stopann ionchur atá ar iarraidh, as dáta nó neamhoiriúnach an tosú. Féach [docs/serving.md](../serving.md) le haghaidh liosta na ngnéithe agus conradh na sábháilteachta in aghaidh tuairte.

## Príomhthorthaí

Scriptor i gcomparáid le vLLM, an múnla, na cáipéisí, prótacal droichid OCR agus GPU céanna: 10 rith péireáilte in aghaidh an innill, tosú úr do gach rith, 368 leathanach in aghaidh an rith, comhéifeachtaí le eatraimh muiníne bootstrap péireáilte 95%.

| Méadracht | Scriptor p50 | vLLM p50 | Breithiúnas |
| ------ | -----------: | -------: | ------- |
| Am iomlán, 368 leathanach (s) | 236.7 | 454.8 | BUACHAINT (0.520x) |
| Tréchur (leathanaigh/uair) | 5,598 | 2,913 | BUACHAINT (1.922x) |
| Tosú go réidh, tar éis an chéad tosaithe (s) | 14.25 | 154.3 | BUACHAINT (0.092x) |
| Am go dtí an chéad chomhartha (s) | 5.070 | 9.167 | BUACHAINT (0.553x) |
| Iarratas ó cheann go ceann (s) | 8.339 | 13.29 | BUACHAINT (0.627x) |

Meaisín amháin, GPU amháin, seisiún amháin, tacar seasta de 15 cháipéis; reáchtáil gach inneall a chumraíocht seolta féin. Socrú, gach méadracht, cruinneas, teorainneacha agus conas é a atáirgeadh: [docs/benchmark.md](../benchmark.md).

<a id="quick-start"></a>
## Tosú tapa

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Leis na socruithe sin easpórtáilte, éisteann an t-inneall ar 127.0.0.1:8002 agus an droichead OCR ar 127.0.0.1:8765. Seol íomhá leathanaigh chuig an inneall mar URL sonraí íomhá, agus cáipéis iomlán chuig an droichead:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

Is corp comhlánaithe comhrá é `request.json` a iompraíonn `image_url` base64 mar URL sonraí i dteachtaireacht an úsáideora. Ní íoslódálann an freastalaí aon rud; is é `install.sh` an t-aon chéim líonraithe.

## Suiteáil

Suiteálann `./install.sh` an dénártha scaoilte, seicphointe GLM-OCR, an múnla leagain amach agus tógáil CUDA d'onnxruntime don GB10, gach céim roghnach agus fíoraithe le suim sheiceála; fíoraíonn `./install.sh check` an toradh. Is féidir leat tógáil ón bhfoinse freisin. Féach [docs/install.md](../install.md).

## Droichead OCR

Reáchtálann `scriptor serve` droichead OCR cáipéisí freisin: PDFanna agus íomhánna isteach, téacs aitheanta amach, le brath leagan amach trí onnxruntime. Tá na bratacha, na socruithe agus seic ionsuite an droichid in [docs/bridge.md](../bridge.md).

## Tagarmharc

Táblaí iomlána, cruinneas, teorainneacha agus céimeanna atáirgthe: [docs/benchmark.md](../benchmark.md).

## Sábháilteacht in aghaidh tuairte

Glacadh isteach teoranta, leibhéil brú cuimhne, oibrithe maoirseachta agus cóid scoir ar féidir le maoirseoir gníomhú orthu: [docs/serving.md](../serving.md#crash-safety).

<a id="documentation"></a>
## Cáipéisíocht

Cáipéisíocht (i mBéarla):

- [docs/install.md](../install.md) — script suiteála, scaoileadh réamhthógtha, tógáil ón bhfoinse, ceadúnais múnla agus spleáchais
- [docs/serving.md](../serving.md) — bratacha `scriptor serve`, teorainneacha iarratais, sábháilteacht in aghaidh tuairte
- [docs/bridge.md](../bridge.md) — an droichead OCR, a bhratacha agus a shocruithe
- [docs/benchmark.md](../benchmark.md) — torthaí, cruinneas, teorainneacha, atáirgeadh an tagarmharc
- [docs/boot.md](../boot.md) — plean cuimhne an tosaithe, céimeanna tosaithe agus conas a thomhaistear am tosaithe
- [docs/drafting.md](../drafting.md) — díchódú tuairimiúil MTP, polasaí glactha agus a mhéadrachtaí
- [docs/attention.md](../attention.md) — eithne aire
- [docs/gemm.md](../gemm.md) — eithne GEMM
- [docs/equivalence.md](../equivalence.md) agus [docs/e2e-equivalence.md](../e2e-equivalence.md) — seiceálacha coibhéise eithne agus ó cheann go ceann
- [docs/engine-file.md](../engine-file.md) — comhad meáchan dúchasach an innill
- [docs/model-facts.md](../model-facts.md) — fíricí GLM-OCR a mheaitseálann an t-inneall
- [docs/vision-overlap.md](../vision-overlap.md) agus [docs/nvjpeg.md](../nvjpeg.md) — forluí físe agus díchódú JPEG ar GPU
- [seat/README.md](../../seat/README.md) — teimpléad `seat`, conarthaí réidhe agus draenála, cóid scoir agus instealladh lochtanna
- [docs/phase0-protocol.md](../phase0-protocol.md) — prótacal tagartha an tomhais
- [docs/third-party-notices.md](../third-party-notices.md) — fógraí tríú páirtí
- [docs/languages.md](../languages.md) — an README seo in 38 teanga

<a id="contributing"></a>
## Ag cur leis

Léigh [CONTRIBUTING.md](../../CONTRIBUTING.md) ar dtús: teidil Conventional Commit, an geata formáide, clippy agus tástála, feistis shintéiseacha amháin. Tuairiscigh leochaileachtaí go príobháideach mar a thuairiscítear in [SECURITY.md](../../SECURITY.md).

---

<a id="license"></a>
## Ceadúnas

Ceadúnaithe faoi dhá cheadúnas: AGPL-3.0-only nó ceadúnas tráchtála (féach
[LICENSE](../../LICENSE) agus [COMMERCIAL-LICENSE.md](../../COMMERCIAL-LICENSE.md)).
Cóipcheart 2026 Matthew D. Murphy. Eisítear ceadúnais thráchtála trí
[Chiron Courseworks](https://chironcourseworks.com); fiosruithe:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Tá fógraí tríú páirtí in [docs/third-party-notices.md](../third-party-notices.md). Is íoslódálacha ar leith iad meáchain an mhúnla agus leabharlanna ama rite agus coinníonn siad a gceadúnais féin (féach [Model and dependency licenses](../install.md#model-and-dependency-licenses)).
