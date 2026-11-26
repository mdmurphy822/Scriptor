<p align="center"><img src="../assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*Нативний сервер інференсу GLM-OCR для NVIDIA GB10: на вході скани, на виході розпізнаний текст, через OpenAI-сумісний API і без збереженого прогрітого стану.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Сторінка проєкту](https://chironcourseworks.com/scriptor)** • [Швидкий старт](#quick-start) • [Встановлення](../install.md) • [OCR-міст](../bridge.md) • [Бенчмарк](../benchmark.md) • [Документація](#documentation) • [Участь у розробці](#contributing) • [Ліцензія](#license)

[Мови](../languages.md) • [English](../../README.md)

</div>

## Вступ

Scriptor (лат. «писар») — локальний рушій інференсу моделі GLM-OCR, оптимізований за пропускною здатністю. Він написаний на Rust із рукописними ядрами CUDA та створений для NVIDIA GB10 у DGX Spark (ARM64). Рушій віддає GLM-OCR через OpenAI-сумісну кінцеву точку chat-completions, приймає зображення сторінок безпосередньо в запиті та публікує метрики Prometheus `vllm:*`, сумісні з vLLM, тож клієнт або дашборд, розраховані на сервер vLLM, працюють без змін. Він локальний і працює за принципом `fail-closed` (зупинка замість здогадок за відсутніх чи неправильних вхідних даних): каталог моделі та адреса прив'язки задаються явно, нічого не завантажується, нічого не виявляється автоматично, а відсутні, застарілі чи невідповідні вхідні дані зупиняють запуск, а не обходяться здогадками. Перелік можливостей і контракт безпеки під час збоїв див. у [docs/serving.md](../serving.md).

## Основні результати

Scriptor порівняно з vLLM: та сама модель, ті самі документи, той самий протокол OCR-моста і той самий GPU; 10 парних запусків на кожен рушій, свіже завантаження для кожного запуску, 368 сторінок за запуск, співвідношення з 95% парними бутстреп-довірчими інтервалами.

| Метрика | Scriptor p50 | vLLM p50 | Вердикт |
| ------ | -----------: | -------: | ------- |
| Загальний час, 368 сторінок (с) | 236.7 | 454.8 | ПЕРЕМОГА (0.520x) |
| Пропускна здатність (сторінок/год) | 5,598 | 2,913 | ПЕРЕМОГА (1.922x) |
| Завантаження до готовності, після першого завантаження (с) | 14.25 | 154.3 | ПЕРЕМОГА (0.092x) |
| Час до першого токена (с) | 5.070 | 9.167 | ПЕРЕМОГА (0.553x) |
| Запит від початку до кінця (с) | 8.339 | 13.29 | ПЕРЕМОГА (0.627x) |

Одна машина, один GPU, один сеанс, фіксований набір із 15 документів; кожен рушій працював у власній постачальній конфігурації. Налаштування, усі метрики, точність, обмеження та спосіб відтворення: [docs/benchmark.md](../benchmark.md).

<a id="quick-start"></a>
## Швидкий старт

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Із експортованими налаштуваннями рушій слухає 127.0.0.1:8002, а OCR-міст — 127.0.0.1:8765. Надсилайте зображення сторінки рушію як data URL зображення, а документ цілком — мосту:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` — це тіло запиту chat-completions, у користувацькому повідомленні якого передано data URL `image_url` у base64. Сервер нічого не завантажує; `install.sh` — єдиний крок, що використовує мережу.

## Встановлення

`./install.sh` встановлює релізний бінарний файл, контрольну точку GLM-OCR, модель розмітки та збірку onnxruntime з CUDA для GB10; кожен крок виконується лише за явною згодою та супроводжується перевіркою контрольних сум, а `./install.sh check` перевіряє результат. Також можна зібрати проєкт із вихідного коду. Див. [docs/install.md](../install.md).

## OCR-міст

`scriptor serve` запускає також OCR-міст для документів: на вході PDF і зображення, на виході розпізнаний текст, із визначенням розмітки через onnxruntime. Прапорці, налаштування та вбудовану перевірку моста описано в [docs/bridge.md](../bridge.md).

## Бенчмарк

Повні таблиці, точність, обмеження та кроки відтворення: [docs/benchmark.md](../benchmark.md).

## Стійкість до збоїв

Обмежений прийом запитів, рівні тиску на пам'ять, контрольовані робочі процеси та коди завершення, на які може реагувати супервізор: [docs/serving.md](../serving.md#crash-safety).

<a id="documentation"></a>
## Документація

Документація (англійською мовою):

- [docs/install.md](../install.md) — скрипт встановлення, готовий реліз, збірка з вихідного коду, ліцензії моделей і залежностей
- [docs/serving.md](../serving.md) — прапорці `scriptor serve`, обмеження запитів, стійкість до збоїв
- [docs/bridge.md](../bridge.md) — OCR-міст, його прапорці та налаштування
- [docs/benchmark.md](../benchmark.md) — результати, точність, обмеження, відтворення бенчмарка
- [docs/boot.md](../boot.md) — план пам'яті під час завантаження, етапи завантаження та спосіб вимірювання часу завантаження
- [docs/drafting.md](../drafting.md) — спекулятивне декодування MTP, політика прийняття та її метрики
- [docs/attention.md](../attention.md) — ядра уваги
- [docs/gemm.md](../gemm.md) — ядра GEMM
- [docs/equivalence.md](../equivalence.md) та [docs/e2e-equivalence.md](../e2e-equivalence.md) — перевірки еквівалентності ядер і наскрізної еквівалентності
- [docs/engine-file.md](../engine-file.md) — файл ваг у власному форматі рушія
- [docs/model-facts.md](../model-facts.md) — відомості про GLM-OCR, яким відповідає рушій
- [docs/vision-overlap.md](../vision-overlap.md) та [docs/nvjpeg.md](../nvjpeg.md) — перекриття обробки зображень і декодування JPEG на GPU
- [seat/README.md](../../seat/README.md) — шаблон `seat`, контракти готовності та дренування, коди завершення та впровадження збоїв
- [docs/phase0-protocol.md](../phase0-protocol.md) — еталонний протокол вимірювань
- [docs/third-party-notices.md](../third-party-notices.md) — повідомлення про стороннє ПЗ
- [docs/languages.md](../languages.md) — цей README 38 мовами

<a id="contributing"></a>
## Участь у розробці

Спершу прочитайте [CONTRIBUTING.md](../../CONTRIBUTING.md): заголовки комітів за Conventional Commits, перевірки форматування, clippy і тестів, лише синтетичні тестові дані. Про вразливості повідомляйте конфіденційно, як описано в [SECURITY.md](../../SECURITY.md).

---

<a id="license"></a>
## Ліцензія

Подвійне ліцензування: AGPL-3.0-only або комерційна ліцензія (див.
[LICENSE](../../LICENSE) і [COMMERCIAL-LICENSE.md](../../COMMERCIAL-LICENSE.md)).
Copyright 2026 Matthew D. Murphy. Комерційні ліцензії видаються через
[Chiron Courseworks](https://chironcourseworks.com); із запитань:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Повідомлення про стороннє ПЗ наведено в [docs/third-party-notices.md](../third-party-notices.md). Ваги моделей і бібліотеки часу виконання завантажуються окремо та зберігають власні ліцензії (див. [Ліцензії моделей і залежностей](../install.md#model-and-dependency-licenses)).
