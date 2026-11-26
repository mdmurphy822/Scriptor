<p align="center"><img src="docs/assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*Un server di inferenza GLM-OCR nativo per NVIDIA GB10: scansioni in ingresso, testo riconosciuto in uscita, tramite un'API compatibile con OpenAI, senza alcuno stato di warm-up salvato.*

[![Licenza](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Piattaforma](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Pagina del progetto](https://chironcourseworks.com/scriptor)** • [Avvio rapido](#quick-start) • [Installazione](docs/install.md) • [Bridge OCR](docs/bridge.md) • [Benchmark](docs/benchmark.md) • [Documentazione](#documentation) • [Contribuire](#contributing) • [Licenza](#license)

[Lingue](docs/languages.md) • [English](README.md)

</div>

## Introduzione

Scriptor (dal latino «scriba») è un motore di inferenza locale per il modello GLM-OCR, ottimizzato per il throughput, scritto in Rust con kernel CUDA scritti a mano e pensato per la NVIDIA GB10 del DGX Spark (ARM64). Serve GLM-OCR tramite un endpoint chat-completions compatibile con OpenAI, accetta le immagini delle pagine nella richiesta ed espone metriche Prometheus `vllm:*` compatibili con vLLM, così un client o una dashboard pensati per un server vLLM funzionano senza modifiche. È locale e `fail-closed` (si ferma invece di tirare a indovinare): la directory del modello e l'indirizzo di bind sono espliciti, non viene scaricato nulla, non viene rilevato nulla, e un input mancante, obsoleto o non corrispondente interrompe l'avvio invece di essere aggirato con ipotesi. Per l'elenco delle funzionalità e il contratto di crash safety vedi [docs/serving.md](docs/serving.md).

<a id="headline-results"></a>
## Risultati principali

Scriptor a confronto con vLLM, stesso modello, stessi documenti, stesso protocollo del bridge OCR e stessa GPU: 10 esecuzioni appaiate per motore, un boot da zero per ogni esecuzione, 368 pagine per esecuzione, rapporti con intervalli di confidenza bootstrap appaiati al 95%.

| Metrica | Scriptor p50 | vLLM p50 | Verdetto |
| ------- | -----------: | -------: | -------- |
| Tempo totale, 368 pagine (s) | 236.7 | 454.8 | VITTORIA (0.520x) |
| Throughput (pagine/ora) | 5,598 | 2,913 | VITTORIA (1.922x) |
| Boot fino allo stato pronto, dopo il primo boot (s) | 14.25 | 154.3 | VITTORIA (0.092x) |
| Tempo al primo token (s) | 5.070 | 9.167 | VITTORIA (0.553x) |
| Richiesta end-to-end (s) | 8.339 | 13.29 | VITTORIA (0.627x) |

Una macchina, una GPU, una sessione, un insieme fisso di 15 documenti; ogni motore ha usato la propria configurazione distribuita. Setup, tutte le metriche, accuratezza, limitazioni e come riprodurre i risultati: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Avvio rapido

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Con queste impostazioni esportate, il motore resta in ascolto su 127.0.0.1:8002 e il bridge OCR su 127.0.0.1:8765. Invia l'immagine di una pagina al motore come data URL di immagine e un intero documento al bridge:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` è un corpo chat-completions il cui messaggio utente contiene un data URL `image_url` in base64. Il server non scarica nulla; `install.sh` è l'unico passaggio che usa la rete.

## Installazione

`./install.sh` installa il binario di release, il checkpoint GLM-OCR, il modello di layout e una build CUDA di onnxruntime per la GB10, ogni passaggio su richiesta esplicita e con verifica del checksum; `./install.sh check` verifica il risultato. È anche possibile compilare dai sorgenti. Vedi [docs/install.md](docs/install.md).

## Bridge OCR

`scriptor serve` esegue anche un bridge OCR per documenti: PDF e immagini in ingresso, testo riconosciuto in uscita, con rilevamento del layout tramite onnxruntime. Flag, impostazioni e il controllo integrato del bridge sono in [docs/bridge.md](docs/bridge.md).

## Benchmark

Tabelle complete, accuratezza, limitazioni e passaggi per la riproduzione: [docs/benchmark.md](docs/benchmark.md).

## Crash safety

Intake limitato, livelli di pressione sulla memoria, worker supervisionati e codici di uscita su cui un supervisor può agire: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Documentazione

Documentazione (in inglese):

- [docs/install.md](docs/install.md) — script di installazione, release precompilata, compilazione dai sorgenti, licenze di modelli e dipendenze
- [docs/serving.md](docs/serving.md) — flag di `scriptor serve`, limiti delle richieste, crash safety
- [docs/bridge.md](docs/bridge.md) — il bridge OCR, i suoi flag e le sue impostazioni
- [docs/benchmark.md](docs/benchmark.md) — risultati, accuratezza, limitazioni, riproduzione del benchmark
- [docs/boot.md](docs/boot.md) — il piano di memoria del boot, le fasi del boot e come viene misurato il tempo di boot
- [docs/drafting.md](docs/drafting.md) — decodifica speculativa MTP, politica di accettazione e relative metriche
- [docs/attention.md](docs/attention.md) — kernel di attention
- [docs/gemm.md](docs/gemm.md) — kernel GEMM
- [docs/equivalence.md](docs/equivalence.md) e [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — verifiche di equivalenza dei kernel e end-to-end
- [docs/engine-file.md](docs/engine-file.md) — il file dei pesi nativo del motore
- [docs/model-facts.md](docs/model-facts.md) — i dati di GLM-OCR a cui il motore si conforma
- [docs/vision-overlap.md](docs/vision-overlap.md) e [docs/nvjpeg.md](docs/nvjpeg.md) — sovrapposizione della visione e decodifica JPEG su GPU
- [seat/README.md](seat/README.md) — il template `seat`, i contratti di readiness e drain, i codici di uscita e la fault injection
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — il protocollo di misurazione di riferimento
- [docs/third-party-notices.md](docs/third-party-notices.md) — avvisi di terze parti
- [docs/languages.md](docs/languages.md) — questo README in 38 lingue

<a id="contributing"></a>
## Contribuire

Leggi prima [CONTRIBUTING.md](CONTRIBUTING.md): subject dei commit in stile Conventional Commit, il gate di formattazione, clippy e test, solo fixture sintetiche. Segnala le vulnerabilità in privato come descritto in [SECURITY.md](SECURITY.md).

---

<a id="license"></a>
## Licenza

Distribuito con doppia licenza: AGPL-3.0-only oppure licenza commerciale (vedi
[LICENSE](LICENSE) e [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Copyright 2026 Matthew D. Murphy. Le licenze commerciali sono rilasciate tramite
[Chiron Courseworks](https://chironcourseworks.com); richieste:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Gli avvisi di terze parti si trovano in [docs/third-party-notices.md](docs/third-party-notices.md). I pesi dei modelli e le librerie di runtime sono download separati e mantengono le proprie licenze (vedi [Model and dependency licenses](docs/install.md#model-and-dependency-licenses)).
