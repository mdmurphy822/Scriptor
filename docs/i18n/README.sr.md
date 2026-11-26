<p align="center"><img src="../assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*Изворни сервер за закључивање GLM-OCR за NVIDIA GB10: скенирани странице на улазу, препознат текст на излазу, преко OpenAI-компатибилног API-ја, без сачуваног топлог стања.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Страница пројекта](https://chironcourseworks.com/scriptor)** • [Брзи почетак](#quick-start) • [Инсталација](../install.md) • [OCR мост](../bridge.md) • [Бенчмарк](../benchmark.md) • [Документација](#documentation) • [Допринос](#contributing) • [Лиценца](#license)

[Језици](../languages.md) • [English](../../README.md)

</div>

## Увод

Scriptor (латински „писар") је локални механизам за закључивање модела GLM-OCR, оптимизован за пропусност, написан у Rust-у са ручно писаним CUDA кернелима и направљен за NVIDIA GB10 у DGX Spark-у (ARM64). Он служи GLM-OCR преко OpenAI-компатибилне крајње тачке за chat completions, прихвата слике страница у захтеву и излаже vLLM-компатибилне Prometheus метрике `vllm:*`, па клијент или контролна табла направљени за vLLM сервер раде без измена. Систем је локалан и `fail-closed` (зауставља се уместо да нагађа када улаз недостаје или је погрешан): директоријум модела и адреса за повезивање задају се изричито, ништа се не преузима, ништа се не открива аутоматски, а недостајући, застарели или неусаглашени улаз зауставља покретање уместо да се нагађа. Списак функција и уговор о отпорности на отказ налазе се у [docs/serving.md](../serving.md).

## Кључни резултати

Scriptor наспрам vLLM-а, исти модел, исти документи, исти протокол OCR моста и исти GPU: 10 упарених извршавања по механизму, свеже покретање за свако извршавање, 368 страница по извршавању, односи са 95% упареним bootstrap интервалима поверења.

| Метрика | Scriptor p50 | vLLM p50 | Исход |
| ------ | -----------: | -------: | ------- |
| Укупно време, 368 страница (s) | 236.7 | 454.8 | ПОБЕДА (0.520x) |
| Пропусност (страница/сат) | 5,598 | 2,913 | ПОБЕДА (1.922x) |
| Покретање до спремности, после првог покретања (s) | 14.25 | 154.3 | ПОБЕДА (0.092x) |
| Време до првог токена (s) | 5.070 | 9.167 | ПОБЕДА (0.553x) |
| Захтев од краја до краја (s) | 8.339 | 13.29 | ПОБЕДА (0.627x) |

Једна машина, један GPU, једна сесија, фиксан скуп од 15 докумената; сваки механизам радио је са својом испорученом конфигурацијом. Подешавање, све метрике, тачност, ограничења и понављање мерења: [docs/benchmark.md](../benchmark.md).

<a id="quick-start"></a>
## Брзи почетак

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Са тако постављеним подешавањима, механизам слуша на 127.0.0.1:8002, а OCR мост на 127.0.0.1:8765. Пошаљите слику странице механизму као image data URL, а цео документ мосту:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` је тело захтева за chat completions чија корисничка порука носи base64 `image_url` data URL. Сервер ништа не преузима; `install.sh` је једини корак који користи мрежу.

## Инсталација

`./install.sh` инсталира издање бинарног фајла, GLM-OCR чекпоинт, модел за препознавање распореда и CUDA верзију onnxruntime-а за GB10, при чему је сваки корак опциони и са провером контролних сума; `./install.sh check` проверава резултат. Можете градити и из изворног кода. Погледајте [docs/install.md](../install.md).

## OCR мост

`scriptor serve` покреће и OCR мост за документе: PDF фајлови и слике на улазу, препознат текст на излазу, уз детекцију распореда преко onnxruntime-а. Флагови, подешавања и уграђена провера моста налазе се у [docs/bridge.md](../bridge.md).

## Бенчмарк

Пуне табеле, тачност, ограничења и кораци за понављање мерења: [docs/benchmark.md](../benchmark.md).

## Отпорност на отказ

Ограничен пријем, нивои притиска на меморију, надгледани радни процеси и излазни кодови на које надзорник може да реагује: [docs/serving.md](../serving.md#crash-safety).

<a id="documentation"></a>
## Документација

Документација (на енглеском):

- [docs/install.md](../install.md) — инсталациона скрипта, готово издање, градња из изворног кода, лиценце модела и зависности
- [docs/serving.md](../serving.md) — флагови команде `scriptor serve`, ограничења захтева, отпорност на отказ
- [docs/bridge.md](../bridge.md) — OCR мост, његови флагови и подешавања
- [docs/benchmark.md](../benchmark.md) — резултати, тачност, ограничења, понављање бенчмарка
- [docs/boot.md](../boot.md) — план меморије при покретању, фазе покретања и како се мери време покретања
- [docs/drafting.md](../drafting.md) — спекулативно декодирање MTP, политика прихватања и његове метрике
- [docs/attention.md](../attention.md) — кернели пажње (attention)
- [docs/gemm.md](../gemm.md) — GEMM кернели
- [docs/equivalence.md](../equivalence.md) и [docs/e2e-equivalence.md](../e2e-equivalence.md) — провере еквивалентности кернела и од краја до краја
- [docs/engine-file.md](../engine-file.md) — фајл тежина у изворном формату механизма
- [docs/model-facts.md](../model-facts.md) — чињенице о GLM-OCR-у са којима се механизам поклапа
- [docs/vision-overlap.md](../vision-overlap.md) и [docs/nvjpeg.md](../nvjpeg.md) — преклапање визуелне обраде и GPU декодирање JPEG-а
- [seat/README.md](../../seat/README.md) — шаблон `seat`, уговори о спремности и пражњењу, излазни кодови и убризгавање грешака
- [docs/phase0-protocol.md](../phase0-protocol.md) — референтни протокол мерења
- [docs/third-party-notices.md](../third-party-notices.md) — напомене о трећим странама
- [docs/languages.md](../languages.md) — овај README на 38 језика

<a id="contributing"></a>
## Допринос

Прво прочитајте [CONTRIBUTING.md](../../CONTRIBUTING.md): наслови комитова по Conventional Commit, провере форматирања, clippy-ја и тестова, само синтетички тестни подаци. Рањивости пријављујте приватно, како је описано у [SECURITY.md](../../SECURITY.md).

---

<a id="license"></a>
## Лиценца

Двојно лиценцирано: AGPL-3.0-only или комерцијална лиценца (погледајте
[LICENSE](../../LICENSE) и [COMMERCIAL-LICENSE.md](../../COMMERCIAL-LICENSE.md)).
Ауторско право 2026 Matthew D. Murphy. Комерцијалне лиценце издају се преко
[Chiron Courseworks](https://chironcourseworks.com); упити:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Напомене о трећим странама налазе се у [docs/third-party-notices.md](../third-party-notices.md). Тежине модела и библиотеке за извршавање преузимају се засебно и задржавају сопствене лиценце (погледајте [Model and dependency licenses](../install.md#model-and-dependency-licenses)).
