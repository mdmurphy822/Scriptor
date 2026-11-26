<p align="center"><img src="docs/assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*Нативный сервер инференса GLM-OCR для NVIDIA GB10: на входе сканы, на выходе распознанный текст, через OpenAI-совместимый API и без сохранённого прогретого состояния.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Страница проекта](https://chironcourseworks.com/scriptor)** • [Быстрый старт](#quick-start) • [Установка](docs/install.md) • [OCR-мост](docs/bridge.md) • [Бенчмарк](docs/benchmark.md) • [Документация](#documentation) • [Участие в разработке](#contributing) • [Лицензия](#license)

[Языки](docs/languages.md) • [English](README.md)

</div>

## Введение

Scriptor (лат. «писец») — локальный движок инференса модели GLM-OCR, оптимизированный по пропускной способности. Он написан на Rust с рукописными ядрами CUDA и создан для NVIDIA GB10 в DGX Spark (ARM64). Движок отдаёт GLM-OCR через OpenAI-совместимую конечную точку chat-completions, принимает изображения страниц прямо в запросе и публикует метрики Prometheus `vllm:*`, совместимые с vLLM, поэтому клиент или дашборд, рассчитанные на сервер vLLM, работают без изменений. Он локален и работает по принципу `fail-closed` (остановка вместо догадок при отсутствующих или неверных входных данных): каталог модели и адрес привязки задаются явно, ничего не скачивается, ничего не обнаруживается автоматически, а отсутствующие, устаревшие или несоответствующие входные данные останавливают запуск, а не обходятся догадками. Список возможностей и контракт безопасности при сбоях см. в [docs/serving.md](docs/serving.md).

## Основные результаты

Scriptor в сравнении с vLLM: та же модель, те же документы, тот же протокол OCR-моста и тот же GPU; 10 парных запусков на каждый движок, свежая загрузка для каждого запуска, 368 страниц за запуск, отношения с 95% парными бутстрэп-доверительными интервалами.

| Метрика | Scriptor p50 | vLLM p50 | Вердикт |
| ------ | -----------: | -------: | ------- |
| Общее время, 368 страниц (с) | 236.7 | 454.8 | ПОБЕДА (0.520x) |
| Пропускная способность (страниц/час) | 5,598 | 2,913 | ПОБЕДА (1.922x) |
| Загрузка до готовности, после первой загрузки (с) | 14.25 | 154.3 | ПОБЕДА (0.092x) |
| Время до первого токена (с) | 5.070 | 9.167 | ПОБЕДА (0.553x) |
| Запрос от начала до конца (с) | 8.339 | 13.29 | ПОБЕДА (0.627x) |

Одна машина, один GPU, один сеанс, фиксированный набор из 15 документов; каждый движок работал в собственной поставляемой конфигурации. Настройка, все метрики, точность, ограничения и способ воспроизведения: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Быстрый старт

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

При экспортированных настройках движок слушает 127.0.0.1:8002, а OCR-мост — 127.0.0.1:8765. Отправляйте изображение страницы движку как data URL изображения, а документ целиком — мосту:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` — это тело запроса chat-completions, в пользовательском сообщении которого передан data URL `image_url` в base64. Сервер ничего не скачивает; `install.sh` — единственный шаг, использующий сеть.

## Установка

`./install.sh` устанавливает релизный бинарный файл, контрольную точку GLM-OCR, модель разметки и сборку onnxruntime с CUDA для GB10; каждый шаг выполняется только по явному согласию и сопровождается проверкой контрольных сумм, а `./install.sh check` проверяет результат. Можно также собрать проект из исходников. См. [docs/install.md](docs/install.md).

## OCR-мост

`scriptor serve` запускает также OCR-мост для документов: на входе PDF и изображения, на выходе распознанный текст, с определением разметки через onnxruntime. Флаги, настройки и встроенная проверка моста описаны в [docs/bridge.md](docs/bridge.md).

## Бенчмарк

Полные таблицы, точность, ограничения и шаги воспроизведения: [docs/benchmark.md](docs/benchmark.md).

## Устойчивость к сбоям

Ограниченный приём запросов, уровни давления на память, контролируемые рабочие процессы и коды завершения, на которые может реагировать супервизор: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Документация

Документация (на английском языке):

- [docs/install.md](docs/install.md) — скрипт установки, готовый релиз, сборка из исходников, лицензии моделей и зависимостей
- [docs/serving.md](docs/serving.md) — флаги `scriptor serve`, ограничения запросов, устойчивость к сбоям
- [docs/bridge.md](docs/bridge.md) — OCR-мост, его флаги и настройки
- [docs/benchmark.md](docs/benchmark.md) — результаты, точность, ограничения, воспроизведение бенчмарка
- [docs/boot.md](docs/boot.md) — план памяти при загрузке, этапы загрузки и способ измерения времени загрузки
- [docs/drafting.md](docs/drafting.md) — спекулятивное декодирование MTP, политика принятия и её метрики
- [docs/attention.md](docs/attention.md) — ядра внимания
- [docs/gemm.md](docs/gemm.md) — ядра GEMM
- [docs/equivalence.md](docs/equivalence.md) и [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — проверки эквивалентности ядер и сквозной эквивалентности
- [docs/engine-file.md](docs/engine-file.md) — файл весов в собственном формате движка
- [docs/model-facts.md](docs/model-facts.md) — сведения о GLM-OCR, которым соответствует движок
- [docs/vision-overlap.md](docs/vision-overlap.md) и [docs/nvjpeg.md](docs/nvjpeg.md) — перекрытие обработки изображений и декодирование JPEG на GPU
- [seat/README.md](seat/README.md) — шаблон `seat`, контракты готовности и дренирования, коды завершения и внедрение сбоев
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — эталонный протокол измерений
- [docs/third-party-notices.md](docs/third-party-notices.md) — уведомления о стороннем ПО
- [docs/languages.md](docs/languages.md) — этот README на 38 языках

<a id="contributing"></a>
## Участие в разработке

Сначала прочитайте [CONTRIBUTING.md](CONTRIBUTING.md): заголовки коммитов по Conventional Commits, проверки форматирования, clippy и тестов, только синтетические тестовые данные. Об уязвимостях сообщайте конфиденциально, как описано в [SECURITY.md](SECURITY.md).

---

<a id="license"></a>
## Лицензия

Двойное лицензирование: AGPL-3.0-only или коммерческая лицензия (см.
[LICENSE](LICENSE) и [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Copyright 2026 Matthew D. Murphy. Коммерческие лицензии выдаются через
[Chiron Courseworks](https://chironcourseworks.com); по вопросам:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Уведомления о стороннем ПО приведены в [docs/third-party-notices.md](docs/third-party-notices.md). Веса моделей и библиотеки времени выполнения загружаются отдельно и сохраняют собственные лицензии (см. [Лицензии моделей и зависимостей](docs/install.md#model-and-dependency-licenses)).
