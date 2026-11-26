<p align="center"><img src="docs/assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*Un servidor de inferencia nativo de GLM-OCR para la NVIDIA GB10: entran escaneos, sale texto reconocido, a través de una API compatible con OpenAI y sin guardar ningún estado precalentado.*

[![Licencia](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Plataforma](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Página del proyecto](https://chironcourseworks.com/scriptor)** • [Inicio rápido](#quick-start) • [Instalación](docs/install.md) • [Puente OCR](docs/bridge.md) • [Benchmark](docs/benchmark.md) • [Documentación](#documentation) • [Contribuir](#contributing) • [Licencia](#license)

[Idiomas](docs/languages.md) • [English](README.md)

</div>

## Introducción

Scriptor (del latín «escriba») es un motor de inferencia local para el modelo GLM-OCR, optimizado para el throughput, escrito en Rust con kernels CUDA hechos a mano y diseñado para la NVIDIA GB10 del DGX Spark (ARM64). Sirve GLM-OCR mediante un endpoint de chat completions compatible con OpenAI, acepta imágenes de página en la solicitud y expone métricas de Prometheus `vllm:*` compatibles con vLLM, de modo que un cliente o un tablero creado para un servidor de vLLM funciona sin cambios. Es local y es `fail-closed` (se detiene en lugar de adivinar cuando falta una entrada o es incorrecta): el directorio del modelo y la dirección de enlace son explícitos, no se descarga nada, no se descubre nada, y una entrada faltante, obsoleta o inconsistente detiene el inicio en lugar de intentar resolverla por su cuenta. Consulten [docs/serving.md](docs/serving.md) para ver la lista de funciones y el contrato de seguridad ante fallas.

## Resultados destacados

Scriptor frente a vLLM, con el mismo modelo, los mismos documentos, el mismo protocolo del puente OCR y la misma GPU: 10 ejecuciones pareadas por motor, con un inicio en frío en cada ejecución, 368 páginas por ejecución y razones con intervalos de confianza bootstrap pareados de 95 %.

| Métrica | Scriptor p50 | vLLM p50 | Veredicto |
| ------ | -----------: | -------: | ------- |
| Tiempo total, 368 páginas (s) | 236.7 | 454.8 | GANA (0.520x) |
| Rendimiento (páginas/hora) | 5,598 | 2,913 | GANA (1.922x) |
| Inicio hasta estar listo, después del primer inicio (s) | 14.25 | 154.3 | GANA (0.092x) |
| Tiempo hasta el primer token (s) | 5.070 | 9.167 | GANA (0.553x) |
| Solicitud de extremo a extremo (s) | 8.339 | 13.29 | GANA (0.627x) |

Una máquina, una GPU, una sesión y un conjunto fijo de 15 documentos; cada motor usó su propia configuración incluida. Configuración, todas las métricas, precisión, limitaciones y cómo reproducirlo: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Inicio rápido

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Con esos valores exportados, el motor escucha en 127.0.0.1:8002 y el puente OCR en 127.0.0.1:8765. Envíen una imagen de página al motor como URL de datos de imagen, y un documento completo al puente:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` es un cuerpo de chat completions cuyo mensaje de usuario incluye una URL de datos `image_url` en base64. El servidor no descarga nada; `install.sh` es el único paso que usa la red.

## Instalación

`./install.sh` instala el binario de la versión publicada, el checkpoint de GLM-OCR, el modelo de diseño de página y una compilación de onnxruntime con CUDA para la GB10; cada paso es opcional y se verifica con suma de verificación, y `./install.sh check` verifica el resultado. También pueden compilar desde el código fuente. Consulten [docs/install.md](docs/install.md).

## Puente OCR

`scriptor serve` también ejecuta un puente OCR de documentos: entran PDF e imágenes, sale texto reconocido, con detección del diseño de página mediante onnxruntime. Los flags, los valores de configuración y la verificación integrada del puente están en [docs/bridge.md](docs/bridge.md).

## Benchmark

Tablas completas, precisión, limitaciones y pasos para reproducirlo: [docs/benchmark.md](docs/benchmark.md).

## Seguridad ante fallas

Admisión acotada, niveles de presión de memoria, procesos de trabajo supervisados y códigos de salida sobre los que un supervisor puede actuar: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Documentación

Documentación (en inglés):

- [docs/install.md](docs/install.md) — script de instalación, versión precompilada, compilación desde el código fuente, licencias de modelos y dependencias
- [docs/serving.md](docs/serving.md) — flags de `scriptor serve`, límites de petición, seguridad ante fallas
- [docs/bridge.md](docs/bridge.md) — el puente OCR, sus flags y valores de configuración
- [docs/benchmark.md](docs/benchmark.md) — resultados, precisión, limitaciones, cómo reproducir el benchmark
- [docs/boot.md](docs/boot.md) — el plan de memoria del inicio, sus etapas y cómo se mide el tiempo de inicio
- [docs/drafting.md](docs/drafting.md) — decodificación especulativa MTP, política de aceptación y sus métricas
- [docs/attention.md](docs/attention.md) — kernels de atención
- [docs/gemm.md](docs/gemm.md) — kernels GEMM
- [docs/equivalence.md](docs/equivalence.md) y [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — comprobaciones de equivalencia de los kernels y de extremo a extremo
- [docs/engine-file.md](docs/engine-file.md) — el archivo de pesos nativo del motor
- [docs/model-facts.md](docs/model-facts.md) — los datos de GLM-OCR con los que el motor coincide
- [docs/vision-overlap.md](docs/vision-overlap.md) y [docs/nvjpeg.md](docs/nvjpeg.md) — traslape de la visión y decodificación JPEG en GPU
- [seat/README.md](seat/README.md) — la plantilla `seat`, los contratos de disponibilidad y vaciado, los códigos de salida y la inyección de fallas
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — el protocolo de medición de referencia
- [docs/third-party-notices.md](docs/third-party-notices.md) — avisos de terceros
- [docs/languages.md](docs/languages.md) — este README en 38 idiomas

<a id="contributing"></a>
## Contribuir

Lean primero [CONTRIBUTING.md](CONTRIBUTING.md): asuntos de commit según Conventional Commits, la puerta de formato, clippy y pruebas, y solo fixtures sintéticos. Reporten las vulnerabilidades de forma privada, como se describe en [SECURITY.md](SECURITY.md).

---

<a id="license"></a>
## Licencia

Licencia dual: AGPL-3.0-only o una licencia comercial (consulten
[LICENSE](LICENSE) y [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Copyright 2026 Matthew D. Murphy. Las licencias comerciales las emite
[Chiron Courseworks](https://chironcourseworks.com); consultas:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Los avisos de terceros están en [docs/third-party-notices.md](docs/third-party-notices.md). Los pesos del modelo y las bibliotecas de ejecución son descargas independientes y conservan sus propias licencias (consulten [Licencias de modelos y dependencias](docs/install.md#model-and-dependency-licenses)).
