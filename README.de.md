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

*Ein nativer GLM-OCR-Inferenzserver für die NVIDIA GB10: Scans hinein, erkannter Text heraus, über eine OpenAI-kompatible API und ohne gespeicherten Warm-Zustand.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Projektseite](https://chironcourseworks.com/scriptor)** • [Schnellstart](#quick-start) • [Installation](docs/install.md) • [OCR-Bridge](docs/bridge.md) • [Benchmark](docs/benchmark.md) • [Dokumentation](#documentation) • [Mitwirken](#contributing) • [Lizenz](#license)

[Sprachen](docs/languages.md) • [English](README.md)

</div>

<a id="introduction"></a>
## Einführung

Scriptor (lateinisch für „Schreiber“) ist eine auf Durchsatz optimierte, rein lokale Inferenz-Engine für das Modell GLM-OCR, geschrieben in Rust mit handgeschriebenen CUDA-Kernels und gebaut für die NVIDIA GB10 im DGX Spark (ARM64). Sie stellt GLM-OCR über einen OpenAI-kompatiblen Chat-Completions-Endpunkt bereit, nimmt Seitenbilder in der Anfrage entgegen und liefert vLLM-kompatible `vllm:*`-Prometheus-Metriken, sodass ein Client oder Dashboard, das für einen vLLM-Server gebaut wurde, unverändert funktioniert. Sie arbeitet lokal und `fail-closed` (sie stoppt, statt zu raten, wenn eine Eingabe fehlt oder falsch ist): Modellverzeichnis und Bind-Adresse werden explizit angegeben, nichts wird heruntergeladen, nichts wird automatisch gesucht, und eine fehlende, veraltete oder nicht passende Eingabe stoppt den Start, anstatt umgangen zu werden. Die Funktionsliste und den Crash-Safety-Vertrag finden Sie in [docs/serving.md](docs/serving.md).

## Wichtigste Ergebnisse

Scriptor im Vergleich zu vLLM, bei gleichem Modell, gleichen Dokumenten, gleichem OCR-Bridge-Protokoll und gleicher GPU: 10 gepaarte Läufe pro Engine, bei jedem Lauf ein frischer Boot, 368 Seiten pro Lauf, Verhältnisse mit gepaarten 95-%-Bootstrap-Konfidenzintervallen.

| Metrik | Scriptor p50 | vLLM p50 | Urteil |
| ------ | -----------: | -------: | ------- |
| Gesamtzeit, 368 Seiten (s) | 236.7 | 454.8 | GEWINNT (0.520x) |
| Durchsatz (Seiten/Stunde) | 5,598 | 2,913 | GEWINNT (1.922x) |
| Boot bis bereit, nach dem ersten Boot (s) | 14.25 | 154.3 | GEWINNT (0.092x) |
| Zeit bis zum ersten Token (s) | 5.070 | 9.167 | GEWINNT (0.553x) |
| Anfrage Ende-zu-Ende (s) | 8.339 | 13.29 | GEWINNT (0.627x) |

Eine Maschine, eine GPU, eine Sitzung, ein fester Satz aus 15 Dokumenten; jede Engine lief mit ihrer eigenen ausgelieferten Konfiguration. Aufbau, alle Metriken, Genauigkeit, Einschränkungen und Reproduktion: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Schnellstart

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Mit diesen Einstellungen lauscht die Engine auf 127.0.0.1:8002 und die OCR-Bridge auf 127.0.0.1:8765. Senden Sie ein Seitenbild als Image-Data-URL an die Engine und ein ganzes Dokument an die Bridge:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` ist ein Chat-Completions-Body, dessen Benutzernachricht eine base64-kodierte `image_url`-Data-URL enthält. Der Server lädt nichts herunter; `install.sh` ist der einzige Schritt mit Netzwerkzugriff.

## Installation

`./install.sh` installiert das Release-Binary, den GLM-OCR-Checkpoint, das Layoutmodell und einen CUDA-Build von onnxruntime für die GB10; jeder Schritt ist optional und prüfsummenverifiziert, und `./install.sh check` verifiziert das Ergebnis. Sie können auch aus dem Quellcode bauen. Siehe [docs/install.md](docs/install.md).

## OCR-Bridge

`scriptor serve` betreibt außerdem eine Dokument-OCR-Bridge: PDFs und Bilder hinein, erkannter Text heraus, mit Layouterkennung über onnxruntime. Flags, Einstellungen und die integrierte Bridge-Prüfung stehen in [docs/bridge.md](docs/bridge.md).

## Benchmark

Vollständige Tabellen, Genauigkeit, Einschränkungen und Schritte zur Reproduktion: [docs/benchmark.md](docs/benchmark.md).

## Crash-Sicherheit

Begrenzte Annahme von Anfragen, Speicherdruck-Stufen, überwachte Worker und Exit-Codes, auf die ein Supervisor reagieren kann: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Dokumentation

Dokumentation (auf Englisch):

- [docs/install.md](docs/install.md) — Installationsskript, vorgebautes Release, Bauen aus dem Quellcode, Modell- und Abhängigkeitslizenzen
- [docs/serving.md](docs/serving.md) — `scriptor serve`-Flags, Anfragelimits, Crash-Sicherheit
- [docs/bridge.md](docs/bridge.md) — die OCR-Bridge, ihre Flags und Einstellungen
- [docs/benchmark.md](docs/benchmark.md) — Ergebnisse, Genauigkeit, Einschränkungen, Reproduktion des Benchmarks
- [docs/boot.md](docs/boot.md) — der Boot-Speicherplan, die Boot-Phasen und wie die Boot-Zeit gemessen wird
- [docs/drafting.md](docs/drafting.md) — MTP-Speculative-Decoding, Akzeptanzrichtlinie und ihre Metriken
- [docs/attention.md](docs/attention.md) — Attention-Kernels
- [docs/gemm.md](docs/gemm.md) — GEMM-Kernels
- [docs/equivalence.md](docs/equivalence.md) und [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — Äquivalenzprüfungen für Kernels und Ende-zu-Ende
- [docs/engine-file.md](docs/engine-file.md) — die engine-native Gewichtsdatei
- [docs/model-facts.md](docs/model-facts.md) — die GLM-OCR-Fakten, denen die Engine entspricht
- [docs/vision-overlap.md](docs/vision-overlap.md) und [docs/nvjpeg.md](docs/nvjpeg.md) — Vision-Overlap und GPU-JPEG-Dekodierung
- [seat/README.md](seat/README.md) — die `seat`-Vorlage, Bereitschafts- und Drain-Verträge, Exit-Codes und Fehlerinjektion
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — das Referenz-Messprotokoll
- [docs/third-party-notices.md](docs/third-party-notices.md) — Hinweise zu Drittanbietern
- [docs/languages.md](docs/languages.md) — dieses README in 38 Sprachen

<a id="contributing"></a>
## Mitwirken

Lesen Sie zuerst [CONTRIBUTING.md](CONTRIBUTING.md): Conventional-Commit-Betreffzeilen, das Gate aus Format, Clippy und Tests, ausschließlich synthetische Fixtures. Melden Sie Sicherheitslücken vertraulich, wie in [SECURITY.md](SECURITY.md) beschrieben.

---

<a id="license"></a>
## Lizenz

Doppelt lizenziert unter AGPL-3.0-only oder einer kommerziellen Lizenz (siehe
[LICENSE](LICENSE) und [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Copyright 2026 Matthew D. Murphy. Kommerzielle Lizenzen werden über
[Chiron Courseworks](https://chironcourseworks.com) vergeben; Anfragen:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Hinweise zu Drittanbietern stehen in [docs/third-party-notices.md](docs/third-party-notices.md). Modellgewichte und Laufzeitbibliotheken sind separate Downloads und behalten ihre eigenen Lizenzen (siehe [Model and dependency licenses](docs/install.md#model-and-dependency-licenses)).
