<p align="center"><img src="docs/assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*Ένας εγγενής διακομιστής εξαγωγής συμπερασμάτων GLM-OCR για το NVIDIA GB10: σαρωμένες σελίδες μπαίνουν, αναγνωρισμένο κείμενο βγαίνει, μέσω API συμβατού με OpenAI, χωρίς αποθηκευμένη κατάσταση προθέρμανσης.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Σελίδα έργου](https://chironcourseworks.com/scriptor)** • [Γρήγορη εκκίνηση](#quick-start) • [Εγκατάσταση](docs/install.md) • [Γέφυρα OCR](docs/bridge.md) • [Δοκιμή επιδόσεων](docs/benchmark.md) • [Τεκμηρίωση](#documentation) • [Συνεισφορά](#contributing) • [Άδεια χρήσης](#license)

[Γλώσσες](docs/languages.md) • [English](README.md)

</div>

## Εισαγωγή

Το Scriptor (στα λατινικά «γραφέας») είναι μια τοπική μηχανή εξαγωγής συμπερασμάτων για το μοντέλο GLM-OCR, βελτιστοποιημένη για διεκπεραίωση (throughput), γραμμένη σε Rust με χειροποίητους πυρήνες CUDA και σχεδιασμένη για το NVIDIA GB10 του DGX Spark (ARM64). Εξυπηρετεί το GLM-OCR μέσω ενός endpoint chat-completions συμβατού με OpenAI, δέχεται εικόνες σελίδων στο αίτημα και εκθέτει μετρικές Prometheus `vllm:*` συμβατές με το vLLM, ώστε ένας πελάτης ή ένας πίνακας ελέγχου φτιαγμένος για διακομιστή vLLM να λειτουργεί αμετάβλητος. Είναι τοπική και `fail-closed` (σταματά αντί να μαντεύει όταν μια εισροή λείπει ή είναι λάθος): ο κατάλογος του μοντέλου και η διεύθυνση σύνδεσης δίνονται ρητά, τίποτα δεν κατεβαίνει, τίποτα δεν ανακαλύπτεται αυτόματα, και μια εισροή που λείπει, είναι παλιά ή δεν ταιριάζει σταματά την εκκίνηση αντί να μαντεύεται. Δείτε το [docs/serving.md](docs/serving.md) για τη λίστα χαρακτηριστικών και το συμβόλαιο ασφάλειας σε περίπτωση κατάρρευσης.

## Βασικά αποτελέσματα

Το Scriptor έναντι του vLLM, με το ίδιο μοντέλο, τα ίδια έγγραφα, το ίδιο πρωτόκολλο γέφυρας OCR και την ίδια GPU: 10 ζευγαρωτές εκτελέσεις ανά μηχανή, νέα εκκίνηση σε κάθε εκτέλεση, 368 σελίδες ανά εκτέλεση, λόγοι με διαστήματα εμπιστοσύνης 95% ζευγαρωτού bootstrap.

| Μετρική | Scriptor p50 | vLLM p50 | Κρίση |
| ------ | -----------: | -------: | ------- |
| Συνολικός χρόνος, 368 σελίδες (s) | 236.7 | 454.8 | ΝΙΚΗ (0.520x) |
| Διεκπεραίωση (σελίδες/ώρα) | 5,598 | 2,913 | ΝΙΚΗ (1.922x) |
| Εκκίνηση έως ετοιμότητα, μετά την πρώτη εκκίνηση (s) | 14.25 | 154.3 | ΝΙΚΗ (0.092x) |
| Χρόνος έως το πρώτο token (s) | 5.070 | 9.167 | ΝΙΚΗ (0.553x) |
| Αίτημα από άκρη σε άκρη (s) | 8.339 | 13.29 | ΝΙΚΗ (0.627x) |

Ένα μηχάνημα, μία GPU, μία συνεδρία, ένα σταθερό σύνολο 15 εγγράφων· κάθε μηχανή έτρεξε με τη δική της διανεμόμενη ρύθμιση. Ρύθμιση, όλες οι μετρικές, ακρίβεια, περιορισμοί και πώς να αναπαραχθούν: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Γρήγορη εκκίνηση

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Με αυτές τις ρυθμίσεις εξαγμένες, η μηχανή ακούει στο 127.0.0.1:8002 και η γέφυρα OCR στο 127.0.0.1:8765. Στείλτε μια εικόνα σελίδας στη μηχανή ως image data URL και ένα ολόκληρο έγγραφο στη γέφυρα:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

Το `request.json` είναι ένα σώμα chat-completions του οποίου το μήνυμα χρήστη φέρει ένα base64 `image_url` data URL. Ο διακομιστής δεν κατεβάζει τίποτα· το `install.sh` είναι το μόνο βήμα που χρησιμοποιεί το δίκτυο.

## Εγκατάσταση

Το `./install.sh` εγκαθιστά το δυαδικό αρχείο έκδοσης, το checkpoint του GLM-OCR, το μοντέλο διάταξης και μια έκδοση CUDA του onnxruntime για το GB10· κάθε βήμα είναι προαιρετικό και επαληθεύεται με αθροίσματα ελέγχου· το `./install.sh check` επαληθεύει το αποτέλεσμα. Μπορείτε επίσης να κάνετε μεταγλώττιση από τον πηγαίο κώδικα. Δείτε το [docs/install.md](docs/install.md).

## Γέφυρα OCR

Το `scriptor serve` εκτελεί επίσης μια γέφυρα OCR εγγράφων: PDF και εικόνες μπαίνουν, αναγνωρισμένο κείμενο βγαίνει, με ανίχνευση διάταξης μέσω onnxruntime. Οι σημαίες, οι ρυθμίσεις και ο ενσωματωμένος έλεγχος της γέφυρας περιγράφονται στο [docs/bridge.md](docs/bridge.md).

## Δοκιμή επιδόσεων

Πλήρεις πίνακες, ακρίβεια, περιορισμοί και βήματα αναπαραγωγής: [docs/benchmark.md](docs/benchmark.md).

## Ασφάλεια σε κατάρρευση

Φραγμένη υποδοχή αιτημάτων, επίπεδα πίεσης μνήμης, εποπτευόμενοι εργάτες και κωδικοί εξόδου στους οποίους μπορεί να αντιδράσει ένας επόπτης: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Τεκμηρίωση

Τεκμηρίωση (στα αγγλικά):

- [docs/install.md](docs/install.md) — σενάριο εγκατάστασης, προκατασκευασμένη έκδοση, μεταγλώττιση από τον πηγαίο κώδικα, άδειες μοντέλων και εξαρτήσεων
- [docs/serving.md](docs/serving.md) — σημαίες του `scriptor serve`, όρια αιτημάτων, ασφάλεια σε κατάρρευση
- [docs/bridge.md](docs/bridge.md) — η γέφυρα OCR, οι σημαίες και οι ρυθμίσεις της
- [docs/benchmark.md](docs/benchmark.md) — αποτελέσματα, ακρίβεια, περιορισμοί, αναπαραγωγή του benchmark
- [docs/boot.md](docs/boot.md) — το σχέδιο μνήμης εκκίνησης, τα στάδια εκκίνησης και πώς μετριέται ο χρόνος εκκίνησης
- [docs/drafting.md](docs/drafting.md) — MTP speculative decoding, πολιτική αποδοχής και οι μετρικές της
- [docs/attention.md](docs/attention.md) — πυρήνες attention
- [docs/gemm.md](docs/gemm.md) — πυρήνες GEMM
- [docs/equivalence.md](docs/equivalence.md) και [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — έλεγχοι ισοδυναμίας πυρήνων και από άκρη σε άκρη
- [docs/engine-file.md](docs/engine-file.md) — το εγγενές αρχείο βαρών της μηχανής
- [docs/model-facts.md](docs/model-facts.md) — τα στοιχεία του GLM-OCR με τα οποία συμφωνεί η μηχανή
- [docs/vision-overlap.md](docs/vision-overlap.md) και [docs/nvjpeg.md](docs/nvjpeg.md) — επικάλυψη επεξεργασίας όρασης και αποκωδικοποίηση JPEG στη GPU
- [seat/README.md](seat/README.md) — το πρότυπο `seat`, τα συμβόλαια ετοιμότητας και αποστράγγισης, οι κωδικοί εξόδου και η έγχυση σφαλμάτων
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — το πρωτόκολλο μέτρησης αναφοράς
- [docs/third-party-notices.md](docs/third-party-notices.md) — ειδοποιήσεις τρίτων
- [docs/languages.md](docs/languages.md) — αυτό το README σε 38 γλώσσες

<a id="contributing"></a>
## Συνεισφορά

Διαβάστε πρώτα το [CONTRIBUTING.md](CONTRIBUTING.md): θέματα Conventional Commit, πύλη μορφοποίησης, clippy και δοκιμών, μόνο συνθετικά δεδομένα δοκιμών. Αναφέρετε τρωτά σημεία ιδιωτικά, όπως περιγράφεται στο [SECURITY.md](SECURITY.md).

---

<a id="license"></a>
## Άδεια χρήσης

Διπλή αδειοδότηση: AGPL-3.0-only ή εμπορική άδεια (βλ.
[LICENSE](LICENSE) και [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Πνευματικά δικαιώματα 2026 Matthew D. Murphy. Οι εμπορικές άδειες εκδίδονται μέσω της
[Chiron Courseworks](https://chironcourseworks.com)· αιτήματα:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Οι ειδοποιήσεις τρίτων βρίσκονται στο [docs/third-party-notices.md](docs/third-party-notices.md). Τα βάρη των μοντέλων και οι βιβλιοθήκες χρόνου εκτέλεσης είναι χωριστές λήψεις και διατηρούν τις δικές τους άδειες (βλ. [Model and dependency licenses](docs/install.md#model-and-dependency-licenses)).
