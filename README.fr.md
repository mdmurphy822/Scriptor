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

*Un serveur d'inférence GLM-OCR natif pour le NVIDIA GB10 : des scans en entrée, du texte reconnu en sortie, via une API compatible OpenAI, sans aucun état de préchauffage sauvegardé.*

[![Licence](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Plateforme](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Page du projet](https://chironcourseworks.com/scriptor)** • [Démarrage rapide](#quick-start) • [Installation](docs/install.md) • [Pont OCR](docs/bridge.md) • [Benchmark](docs/benchmark.md) • [Documentation](#documentation) • [Contribuer](#contributing) • [Licence](#license)

[Langues](docs/languages.md) • [English](README.md)

</div>

## Introduction

Scriptor (« scribe » en latin) est un moteur d'inférence local, optimisé pour le débit, dédié au modèle GLM-OCR. Il est écrit en Rust avec des noyaux CUDA écrits à la main et conçu pour le NVIDIA GB10 du DGX Spark (ARM64). Il sert GLM-OCR via un point d'accès de chat completions compatible OpenAI, accepte des images de pages dans la requête et expose des métriques Prometheus `vllm:*` compatibles vLLM : un client ou un tableau de bord conçu pour un serveur vLLM fonctionne donc sans modification. Il est local et `fail-closed` (il s'arrête plutôt que de deviner quand une entrée manque ou est incorrecte) : le répertoire du modèle et l'adresse d'écoute sont explicites, rien n'est téléchargé, rien n'est découvert, et une entrée manquante, périmée ou incohérente interrompt le démarrage au lieu d'être contournée par déduction. Voir [docs/serving.md](docs/serving.md) pour la liste des fonctionnalités et le contrat de résilience aux pannes.

## Résultats clés

Scriptor face à vLLM, avec le même modèle, les mêmes documents, le même protocole de pont OCR et le même GPU : 10 exécutions appariées par moteur, un démarrage à froid pour chaque exécution, 368 pages par exécution, rapports accompagnés d'intervalles de confiance à 95 % obtenus par bootstrap apparié.

| Métrique | Scriptor p50 | vLLM p50 | Verdict |
| ------ | -----------: | -------: | ------- |
| Temps total, 368 pages (s) | 236.7 | 454.8 | GAGNANT (0.520x) |
| Débit (pages/heure) | 5,598 | 2,913 | GAGNANT (1.922x) |
| Démarrage jusqu'à l'état prêt, après le premier démarrage (s) | 14.25 | 154.3 | GAGNANT (0.092x) |
| Délai avant le premier jeton (s) | 5.070 | 9.167 | GAGNANT (0.553x) |
| Requête de bout en bout (s) | 8.339 | 13.29 | GAGNANT (0.627x) |

Une machine, un GPU, une session, un jeu fixe de 15 documents ; chaque moteur a utilisé sa propre configuration livrée. Configuration, toutes les métriques, précision, limites et méthode de reproduction : [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Démarrage rapide

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Une fois ces réglages exportés, le moteur écoute sur 127.0.0.1:8002 et le pont OCR sur 127.0.0.1:8765. Envoyez une image de page au moteur sous forme d'URL de données image, et un document entier au pont :

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` est un corps de requête chat completions dont le message utilisateur contient une URL de données `image_url` en base64. Le serveur ne télécharge rien ; `install.sh` est la seule étape qui utilise le réseau.

## Installation

`./install.sh` installe le binaire de version, le checkpoint GLM-OCR, le modèle de détection de mise en page et une build CUDA d'onnxruntime pour le GB10, chaque étape étant facultative et vérifiée par somme de contrôle ; `./install.sh check` vérifie le résultat. Vous pouvez aussi compiler depuis les sources. Voir [docs/install.md](docs/install.md).

## Pont OCR

`scriptor serve` exécute aussi un pont OCR de documents : PDF et images en entrée, texte reconnu en sortie, avec détection de la mise en page via onnxruntime. Les options, les réglages et la vérification intégrée du pont se trouvent dans [docs/bridge.md](docs/bridge.md).

## Benchmark

Tableaux complets, précision, limites et étapes de reproduction : [docs/benchmark.md](docs/benchmark.md).

## Résilience aux pannes

Admission bornée, paliers de pression mémoire, processus de travail supervisés et codes de sortie exploitables par un superviseur : [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Documentation

Documentation (en anglais) :

- [docs/install.md](docs/install.md) — script d'installation, version précompilée, compilation depuis les sources, licences du modèle et des dépendances
- [docs/serving.md](docs/serving.md) — options de `scriptor serve`, limites des requêtes, résilience aux pannes
- [docs/bridge.md](docs/bridge.md) — le pont OCR, ses options et ses réglages
- [docs/benchmark.md](docs/benchmark.md) — résultats, précision, limites, reproduction du benchmark
- [docs/boot.md](docs/boot.md) — le plan mémoire du démarrage, ses étapes et la mesure du temps de démarrage
- [docs/drafting.md](docs/drafting.md) — décodage spéculatif MTP, politique d'acceptation et ses métriques
- [docs/attention.md](docs/attention.md) — noyaux d'attention
- [docs/gemm.md](docs/gemm.md) — noyaux GEMM
- [docs/equivalence.md](docs/equivalence.md) et [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — vérifications d'équivalence des noyaux et de bout en bout
- [docs/engine-file.md](docs/engine-file.md) — le fichier de poids natif du moteur
- [docs/model-facts.md](docs/model-facts.md) — les caractéristiques de GLM-OCR que le moteur reproduit
- [docs/vision-overlap.md](docs/vision-overlap.md) et [docs/nvjpeg.md](docs/nvjpeg.md) — recouvrement de la vision et décodage JPEG sur GPU
- [seat/README.md](seat/README.md) — le modèle `seat`, les contrats de disponibilité et de vidage, les codes de sortie et l'injection de pannes
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — le protocole de mesure de référence
- [docs/third-party-notices.md](docs/third-party-notices.md) — mentions relatives aux composants tiers
- [docs/languages.md](docs/languages.md) — ce README en 38 langues

<a id="contributing"></a>
## Contribuer

Lisez d'abord [CONTRIBUTING.md](CONTRIBUTING.md) : sujets de commit au format Conventional Commits, passage obligatoire du formatage, de clippy et des tests, jeux de données synthétiques uniquement. Signalez les vulnérabilités en privé comme décrit dans [SECURITY.md](SECURITY.md).

---

<a id="license"></a>
## Licence

Distribué sous double licence : AGPL-3.0-only ou licence commerciale (voir
[LICENSE](LICENSE) et [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Copyright 2026 Matthew D. Murphy. Les licences commerciales sont délivrées par
[Chiron Courseworks](https://chironcourseworks.com) ; demandes :
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Les mentions relatives aux composants tiers figurent dans [docs/third-party-notices.md](docs/third-party-notices.md). Les poids du modèle et les bibliothèques d'exécution sont des téléchargements distincts et conservent leurs propres licences (voir [Licences des modèles et des dépendances](docs/install.md#model-and-dependency-licenses)).
