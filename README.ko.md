<p align="center"><img src="docs/assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*NVIDIA GB10용 네이티브 GLM-OCR 추론 서버: 스캔 이미지를 입력하면 인식된 텍스트를 OpenAI 호환 API로 반환하며, 저장된 웜 상태가 없습니다.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[프로젝트 페이지](https://chironcourseworks.com/scriptor)** • [빠른 시작](#quick-start) • [설치](docs/install.md) • [OCR 브리지](docs/bridge.md) • [벤치마크](docs/benchmark.md) • [문서](#documentation) • [기여하기](#contributing) • [라이선스](#license)

[언어](docs/languages.md) • [English](README.md)

</div>

## 소개

Scriptor(라틴어로 '필경사')는 GLM-OCR 모델을 위한, 처리량에 최적화된 로컬 전용 추론 엔진입니다. Rust와 직접 작성한 CUDA 커널로 구현했으며 DGX Spark(ARM64)의 NVIDIA GB10을 대상으로 만들었습니다. OpenAI 호환 chat-completions 엔드포인트로 GLM-OCR를 서비스하고, 요청에 포함된 페이지 이미지를 받으며, vLLM 호환 `vllm:*` Prometheus 메트릭을 노출하므로 vLLM 서버용으로 만든 클라이언트나 대시보드를 그대로 사용할 수 있습니다. 로컬 전용이며 `fail-closed`(입력이 누락되었거나 잘못되면 추측하지 않고 중단)로 동작합니다. 모델 디렉터리와 바인드 주소는 명시적으로 지정해야 하고, 아무것도 다운로드하거나 탐색하지 않으며, 누락되었거나 오래되었거나 일치하지 않는 입력은 추측으로 우회하지 않고 시작 자체를 중단시킵니다. 기능 목록과 크래시 안전성 계약은 [docs/serving.md](docs/serving.md)를 참고하세요.

## 주요 결과

동일한 모델, 문서, OCR 브리지 프로토콜, GPU에서 Scriptor와 vLLM을 비교했습니다. 엔진당 10회씩 짝지은 실행, 매 실행마다 새로 부팅, 실행당 368페이지이며, 비율에는 95% 짝지은 부트스트랩 신뢰구간을 적용했습니다.

| 지표 | Scriptor p50 | vLLM p50 | 판정 |
| ------ | -----------: | -------: | ------- |
| 경과 시간, 368페이지 (초) | 236.7 | 454.8 | 승 (0.520x) |
| 처리량 (페이지/시간) | 5,598 | 2,913 | 승 (1.922x) |
| 준비 완료까지 부팅 시간, 첫 부팅 이후 (초) | 14.25 | 154.3 | 승 (0.092x) |
| 첫 토큰까지의 시간 (초) | 5.070 | 9.167 | 승 (0.553x) |
| 종단 간 요청 (초) | 8.339 | 13.29 | 승 (0.627x) |

머신 1대, GPU 1개, 세션 1회, 고정된 15개 문서 세트이며, 각 엔진은 자체 배포 구성으로 실행했습니다. 설정, 전체 메트릭, 정확도, 한계, 재현 방법은 [docs/benchmark.md](docs/benchmark.md)를 참고하세요.

<a id="quick-start"></a>
## 빠른 시작

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

위 설정을 내보내면 엔진은 127.0.0.1:8002에서, OCR 브리지는 127.0.0.1:8765에서 수신 대기합니다. 페이지 이미지는 이미지 data URL로 엔진에 보내고, 문서 전체는 브리지에 보냅니다.

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json`은 사용자 메시지에 base64 `image_url` data URL을 담은 chat-completions 본문입니다. 서버는 아무것도 다운로드하지 않으며, 네트워크를 사용하는 단계는 `install.sh`뿐입니다.

## 설치

`./install.sh`는 릴리스 바이너리, GLM-OCR 체크포인트, 레이아웃 모델, GB10용 CUDA 빌드 onnxruntime을 설치하며, 각 단계는 선택 사항이고 체크섬으로 검증됩니다. `./install.sh check`로 결과를 확인할 수 있습니다. 소스에서 직접 빌드할 수도 있습니다. [docs/install.md](docs/install.md)를 참고하세요.

## OCR 브리지

`scriptor serve`는 문서 OCR 브리지도 함께 실행합니다. PDF와 이미지를 입력하면 인식된 텍스트를 반환하며, 레이아웃 검출에는 onnxruntime을 사용합니다. 플래그, 설정, 내장 브리지 점검은 [docs/bridge.md](docs/bridge.md)에 있습니다.

## 벤치마크

전체 표, 정확도, 한계, 재현 절차: [docs/benchmark.md](docs/benchmark.md).

## 크래시 안전성

제한된 입력 수용, 메모리 압박 단계, 감독되는 워커, 슈퍼바이저가 대응할 수 있는 종료 코드: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## 문서

문서(영어):

- [docs/install.md](docs/install.md) — 설치 스크립트, 사전 빌드 릴리스, 소스에서 빌드, 모델 및 의존성 라이선스
- [docs/serving.md](docs/serving.md) — `scriptor serve` 플래그, 요청 제한, 크래시 안전성
- [docs/bridge.md](docs/bridge.md) — OCR 브리지, 플래그와 설정
- [docs/benchmark.md](docs/benchmark.md) — 결과, 정확도, 한계, 벤치마크 재현
- [docs/boot.md](docs/boot.md) — 부팅 메모리 계획, 부팅 단계, 부팅 시간 측정 방법
- [docs/drafting.md](docs/drafting.md) — MTP 추측 디코딩(speculative decoding), 수락 정책과 관련 메트릭
- [docs/attention.md](docs/attention.md) — 어텐션 커널
- [docs/gemm.md](docs/gemm.md) — GEMM 커널
- [docs/equivalence.md](docs/equivalence.md) 및 [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — 커널 및 종단 간 동등성 검사
- [docs/engine-file.md](docs/engine-file.md) — 엔진 고유 가중치 파일
- [docs/model-facts.md](docs/model-facts.md) — 엔진이 맞추는 GLM-OCR 사양
- [docs/vision-overlap.md](docs/vision-overlap.md) 및 [docs/nvjpeg.md](docs/nvjpeg.md) — 비전 오버랩과 GPU JPEG 디코딩
- [seat/README.md](seat/README.md) — `seat` 템플릿, 준비(readiness) 및 드레인(drain) 계약, 종료 코드, 장애 주입
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — 기준 측정 프로토콜
- [docs/third-party-notices.md](docs/third-party-notices.md) — 서드파티 고지
- [docs/languages.md](docs/languages.md) — 이 README의 38개 언어 버전

<a id="contributing"></a>
## 기여하기

먼저 [CONTRIBUTING.md](CONTRIBUTING.md)를 읽어 주세요. Conventional Commit 형식의 제목, 포맷·clippy·테스트 게이트, 합성 픽스처만 사용한다는 규칙이 담겨 있습니다. 취약점은 [SECURITY.md](SECURITY.md)에 설명된 대로 비공개로 신고해 주세요.

---

<a id="license"></a>
## 라이선스

AGPL-3.0-only 또는 상용 라이선스의 이중 라이선스로 제공됩니다
([LICENSE](LICENSE) 및 [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) 참조).
Copyright 2026 Matthew D. Murphy. 상용 라이선스는
[Chiron Courseworks](https://chironcourseworks.com)를 통해 발급되며, 문의처:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

서드파티 고지는 [docs/third-party-notices.md](docs/third-party-notices.md)에 있습니다. 모델 가중치와 런타임 라이브러리는 별도로 다운로드하며 각자의 라이선스를 유지합니다([모델 및 의존성 라이선스](docs/install.md#model-and-dependency-licenses) 참조).
