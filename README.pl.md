<p align="center"><img src="docs/assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*Natywny serwer inferencji GLM-OCR dla NVIDIA GB10: skany na wejściu, rozpoznany tekst na wyjściu, przez API zgodne z OpenAI, bez zapisywanego stanu rozgrzania.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Strona projektu](https://chironcourseworks.com/scriptor)** • [Szybki start](#quick-start) • [Instalacja](docs/install.md) • [Most OCR](docs/bridge.md) • [Benchmark](docs/benchmark.md) • [Dokumentacja](#documentation) • [Współtworzenie](#contributing) • [Licencja](#license)

[Języki](docs/languages.md) • [English](README.md)

</div>

## Wprowadzenie

Scriptor (z łaciny „skryba”) to lokalny silnik inferencji modelu GLM-OCR zoptymalizowany pod kątem przepustowości, napisany w Rust z ręcznie pisanymi jądrami CUDA i zbudowany dla NVIDIA GB10 w DGX Spark (ARM64). Udostępnia GLM-OCR przez endpoint chat-completions zgodny z OpenAI, przyjmuje obrazy stron w żądaniu i wystawia zgodne z vLLM metryki Prometheus `vllm:*`, dzięki czemu klient lub dashboard przygotowany dla serwera vLLM działa bez zmian. Działa wyłącznie lokalnie i w trybie `fail-closed` (zatrzymanie zamiast zgadywania): katalog modelu i adres nasłuchu są jawne, nic nie jest pobierane, nic nie jest wykrywane automatycznie, a brakujące, nieaktualne lub niezgodne dane wejściowe przerywają start zamiast być obchodzone domysłami. Listę funkcji i kontrakt bezpieczeństwa przy awariach znajdziesz w [docs/serving.md](docs/serving.md).

## Najważniejsze wyniki

Scriptor kontra vLLM, ten sam model, dokumenty, protokół mostu OCR i GPU: 10 sparowanych uruchomień na silnik, świeży start przy każdym uruchomieniu, 368 stron na uruchomienie, współczynniki z 95% sparowanymi przedziałami ufności bootstrap.

| Metryka | Scriptor p50 | vLLM p50 | Werdykt |
| ------ | -----------: | -------: | ------- |
| Czas całkowity, 368 stron (s) | 236.7 | 454.8 | WYGRANA (0.520x) |
| Przepustowość (stron/godz.) | 5,598 | 2,913 | WYGRANA (1.922x) |
| Start do gotowości, po pierwszym starcie (s) | 14.25 | 154.3 | WYGRANA (0.092x) |
| Czas do pierwszego tokenu (s) | 5.070 | 9.167 | WYGRANA (0.553x) |
| Żądanie end-to-end (s) | 8.339 | 13.29 | WYGRANA (0.627x) |

Jedna maszyna, jedno GPU, jedna sesja, stały zestaw 15 dokumentów; każdy silnik działał w swojej własnej dostarczanej konfiguracji. Konfiguracja, wszystkie metryki, dokładność, ograniczenia i sposób odtworzenia: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Szybki start

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Po wyeksportowaniu tych ustawień silnik nasłuchuje na 127.0.0.1:8002, a most OCR na 127.0.0.1:8765. Obraz strony wysyłaj do silnika jako data URL obrazu, a cały dokument do mostu:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` to treść żądania chat-completions, w której wiadomość użytkownika zawiera data URL `image_url` w base64. Serwer niczego nie pobiera; `install.sh` jest jedynym krokiem korzystającym z sieci.

## Instalacja

`./install.sh` instaluje binarkę wydania, checkpoint GLM-OCR, model układu strony oraz build CUDA biblioteki onnxruntime dla GB10; każdy krok jest opcjonalny i weryfikowany sumą kontrolną, a `./install.sh check` sprawdza wynik. Można też zbudować ze źródeł. Zobacz [docs/install.md](docs/install.md).

## Most OCR

`scriptor serve` uruchamia także most OCR dokumentów: PDF-y i obrazy na wejściu, rozpoznany tekst na wyjściu, z wykrywaniem układu strony przez onnxruntime. Flagi, ustawienia i wbudowane sprawdzenie mostu opisano w [docs/bridge.md](docs/bridge.md).

## Benchmark

Pełne tabele, dokładność, ograniczenia i kroki odtworzenia: [docs/benchmark.md](docs/benchmark.md).

## Bezpieczeństwo przy awariach

Ograniczony przyjmowany ruch, poziomy presji pamięci, nadzorowane procesy robocze i kody wyjścia, na które może reagować nadzorca: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Dokumentacja

Dokumentacja (po angielsku):

- [docs/install.md](docs/install.md) — skrypt instalacyjny, gotowe wydanie, budowanie ze źródeł, licencje modeli i zależności
- [docs/serving.md](docs/serving.md) — flagi `scriptor serve`, limity żądań, bezpieczeństwo przy awariach
- [docs/bridge.md](docs/bridge.md) — most OCR, jego flagi i ustawienia
- [docs/benchmark.md](docs/benchmark.md) — wyniki, dokładność, ograniczenia, odtwarzanie benchmarku
- [docs/boot.md](docs/boot.md) — plan pamięci przy starcie, etapy startu i sposób pomiaru czasu startu
- [docs/drafting.md](docs/drafting.md) — dekodowanie spekulatywne MTP, polityka akceptacji i jej metryki
- [docs/attention.md](docs/attention.md) — jądra uwagi (attention)
- [docs/gemm.md](docs/gemm.md) — jądra GEMM
- [docs/equivalence.md](docs/equivalence.md) i [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — kontrole równoważności jąder i end-to-end
- [docs/engine-file.md](docs/engine-file.md) — natywny plik wag silnika
- [docs/model-facts.md](docs/model-facts.md) — fakty o GLM-OCR, z którymi silnik jest zgodny
- [docs/vision-overlap.md](docs/vision-overlap.md) i [docs/nvjpeg.md](docs/nvjpeg.md) — nakładanie obliczeń wizyjnych i dekodowanie JPEG na GPU
- [seat/README.md](seat/README.md) — szablon `seat`, kontrakty gotowości i drain, kody wyjścia oraz wstrzykiwanie błędów
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — referencyjny protokół pomiarowy
- [docs/third-party-notices.md](docs/third-party-notices.md) — informacje o komponentach zewnętrznych
- [docs/languages.md](docs/languages.md) — ten README w 38 językach

<a id="contributing"></a>
## Współtworzenie

Zacznij od [CONTRIBUTING.md](CONTRIBUTING.md): tytuły commitów w stylu Conventional Commits, bramka formatowania, clippy i testów, wyłącznie syntetyczne dane testowe. Luki bezpieczeństwa zgłaszaj prywatnie, zgodnie z opisem w [SECURITY.md](SECURITY.md).

---

<a id="license"></a>
## Licencja

Dostępny na podwójnej licencji: AGPL-3.0-only lub licencji komercyjnej (zob.
[LICENSE](LICENSE) i [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Copyright 2026 Matthew D. Murphy. Licencje komercyjne wydaje
[Chiron Courseworks](https://chironcourseworks.com); zapytania:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Informacje o komponentach zewnętrznych znajdują się w [docs/third-party-notices.md](docs/third-party-notices.md). Wagi modeli i biblioteki uruchomieniowe są pobierane osobno i zachowują własne licencje (zob. [Licencje modeli i zależności](docs/install.md#model-and-dependency-licenses)).
