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

*Роден сървър за извод на GLM-OCR за NVIDIA GB10: сканирани страници на входа, разпознат текст на изхода, през OpenAI-съвместим API, без запазено топло състояние.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Страница на проекта](https://chironcourseworks.com/scriptor)** • [Бърз старт](#quick-start) • [Инсталиране](docs/install.md) • [OCR мост](docs/bridge.md) • [Бенчмарк](docs/benchmark.md) • [Документация](#documentation) • [Принос](#contributing) • [Лиценз](#license)

[Езици](docs/languages.md) • [English](README.md)

</div>

## Въведение

Scriptor (на латински „писар") е локален механизъм за извод на модела GLM-OCR, оптимизиран за пропускателна способност, написан на Rust с ръчно написани CUDA ядра и създаден за NVIDIA GB10 в DGX Spark (ARM64). Той обслужва GLM-OCR през OpenAI-съвместима крайна точка за chat completions, приема изображения на страници в заявката и предоставя vLLM-съвместими Prometheus метрики `vllm:*`, така че клиент или табло, изградени за vLLM сървър, работят без промени. Системата е локална и `fail-closed` (спира, вместо да налучква, когато входът липсва или е грешен): директорията на модела и адресът за свързване се задават изрично, нищо не се изтегля, нищо не се открива автоматично, а липсващ, остарял или несъвместим вход спира стартирането, вместо да бъде отгатван. Списъкът с функции и договорът за устойчивост при срив са в [docs/serving.md](docs/serving.md).

## Основни резултати

Scriptor срещу vLLM, същият модел, същите документи, същият протокол на OCR моста и същият GPU: 10 сдвоени изпълнения за всеки механизъм, свежо стартиране за всяко изпълнение, 368 страници на изпълнение, съотношения с 95% сдвоени bootstrap доверителни интервали.

| Метрика | Scriptor p50 | vLLM p50 | Резултат |
| ------ | -----------: | -------: | ------- |
| Общо време, 368 страници (s) | 236.7 | 454.8 | ПОБЕДА (0.520x) |
| Пропускателна способност (страници/час) | 5,598 | 2,913 | ПОБЕДА (1.922x) |
| Стартиране до готовност, след първото стартиране (s) | 14.25 | 154.3 | ПОБЕДА (0.092x) |
| Време до първия токен (s) | 5.070 | 9.167 | ПОБЕДА (0.553x) |
| Заявка от край до край (s) | 8.339 | 13.29 | ПОБЕДА (0.627x) |

Една машина, един GPU, една сесия, фиксиран набор от 15 документа; всеки механизъм работи със своята доставяна конфигурация. Настройка, всички метрики, точност, ограничения и възпроизвеждане: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Бърз старт

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

При така зададените настройки механизмът слуша на 127.0.0.1:8002, а OCR мостът на 127.0.0.1:8765. Изпратете изображение на страница към механизма като image data URL, а цял документ към моста:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` е тяло на заявка за chat completions, чието потребителско съобщение носи base64 `image_url` data URL. Сървърът не изтегля нищо; `install.sh` е единствената стъпка, която използва мрежата.

## Инсталиране

`./install.sh` инсталира готовия двоичен файл, контролната точка на GLM-OCR, модела за разпознаване на оформление и CUDA изграждане на onnxruntime за GB10, като всяка стъпка е по избор и с проверка на контролните суми; `./install.sh check` проверява резултата. Можете да изградите и от изходен код. Вижте [docs/install.md](docs/install.md).

## OCR мост

`scriptor serve` стартира и OCR мост за документи: PDF файлове и изображения на входа, разпознат текст на изхода, с откриване на оформлението през onnxruntime. Флаговете, настройките и вграденото изпитване на моста са в [docs/bridge.md](docs/bridge.md).

## Бенчмарк

Пълни таблици, точност, ограничения и стъпки за възпроизвеждане: [docs/benchmark.md](docs/benchmark.md).

## Устойчивост при срив

Ограничен приемател, нива на натиск върху паметта, наблюдавани работни процеси и изходни кодове, на които надзорник може да реагира: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Документация

Документация (на английски):

- [docs/install.md](docs/install.md) — скрипт за инсталиране, готова версия, изграждане от изходен код, лицензи на модела и зависимостите
- [docs/serving.md](docs/serving.md) — флагове на `scriptor serve`, ограничения на заявките, устойчивост при срив
- [docs/bridge.md](docs/bridge.md) — OCR мостът, неговите флагове и настройки
- [docs/benchmark.md](docs/benchmark.md) — резултати, точност, ограничения, възпроизвеждане на бенчмарка
- [docs/boot.md](docs/boot.md) — планът на паметта при стартиране, етапите на стартиране и как се измерва времето за стартиране
- [docs/drafting.md](docs/drafting.md) — спекулативно декодиране MTP, политика на приемане и неговите метрики
- [docs/attention.md](docs/attention.md) — ядра за внимание (attention)
- [docs/gemm.md](docs/gemm.md) — GEMM ядра
- [docs/equivalence.md](docs/equivalence.md) и [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — проверки за еквивалентност на ядрата и от край до край
- [docs/engine-file.md](docs/engine-file.md) — файлът с тегла в собствен формат на механизма
- [docs/model-facts.md](docs/model-facts.md) — фактите за GLM-OCR, с които механизмът съвпада
- [docs/vision-overlap.md](docs/vision-overlap.md) и [docs/nvjpeg.md](docs/nvjpeg.md) — припокриване на визуалната обработка и GPU декодиране на JPEG
- [seat/README.md](seat/README.md) — шаблонът `seat`, договорите за готовност и източване, изходните кодове и внасянето на грешки
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — референтният протокол за измерване
- [docs/third-party-notices.md](docs/third-party-notices.md) — бележки за трети страни
- [docs/languages.md](docs/languages.md) — този README на 38 езика

<a id="contributing"></a>
## Принос

Първо прочетете [CONTRIBUTING.md](CONTRIBUTING.md): заглавия на комитите по Conventional Commit, проверките за форматиране, clippy и тестове, само синтетични тестови данни. Съобщавайте за уязвимости поверително, както е описано в [SECURITY.md](SECURITY.md).

---

<a id="license"></a>
## Лиценз

С двоен лиценз: AGPL-3.0-only или комерсиален лиценз (вижте
[LICENSE](LICENSE) и [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Авторско право 2026 Matthew D. Murphy. Комерсиалните лицензи се издават чрез
[Chiron Courseworks](https://chironcourseworks.com); запитвания:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Бележките за трети страни са в [docs/third-party-notices.md](docs/third-party-notices.md). Теглата на модела и библиотеките по време на изпълнение се изтеглят отделно и запазват собствените си лицензи (вижте [Model and dependency licenses](docs/install.md#model-and-dependency-licenses)).
