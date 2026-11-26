<p align="center"><img src="../assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*NVIDIA GB10 向けのネイティブ GLM-OCR 推論サーバー:スキャン画像を入力すると認識テキストを出力します。OpenAI 互換 API で提供し、ウォーム状態は一切保存しません。*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[プロジェクトページ](https://chironcourseworks.com/scriptor)** • [クイックスタート](#quick-start) • [インストール](../install.md) • [OCR ブリッジ](../bridge.md) • [ベンチマーク](../benchmark.md) • [ドキュメント](#documentation) • [コントリビュート](#contributing) • [ライセンス](#license)

[言語](../languages.md) • [English](../../README.md)

</div>

<a id="introduction"></a>
## はじめに

Scriptor(ラテン語で「書記」の意)は、GLM-OCR モデル向けのスループット最適化済み・ローカル専用の推論エンジンです。Rust と手書きの CUDA カーネルで実装され、DGX Spark(ARM64)に搭載された NVIDIA GB10 向けに構築されています。OpenAI 互換の chat-completions エンドポイントで GLM-OCR を提供し、リクエストでページ画像を受け付け、vLLM 互換の `vllm:*` Prometheus メトリクスを公開するため、vLLM サーバー向けに作られたクライアントやダッシュボードをそのまま利用できます。動作はローカル限定かつ `fail-closed` です(入力が欠けている、または誤っている場合は推測せずに停止します)。モデルディレクトリとバインドアドレスは明示的に指定する必要があり、何もダウンロードせず、何も自動検出せず、入力が欠落・陳腐化・不一致の場合は推測で回避せず起動を中止します。機能一覧とクラッシュ安全性の契約については [docs/serving.md](../serving.md) を参照してください。

## 主な結果

Scriptor と vLLM の比較です。モデル、ドキュメント、OCR ブリッジプロトコル、GPU は同一で、エンジンごとに 10 回のペア実行を行い、毎回新規に起動し、1 回あたり 368 ページを処理しました。比率には 95% のペア・ブートストラップ信頼区間を付しています。

| 指標 | Scriptor p50 | vLLM p50 | 判定 |
| ------ | -----------: | -------: | ------- |
| 所要時間、368 ページ(秒) | 236.7 | 454.8 | 勝利 (0.520x) |
| スループット(ページ/時間) | 5,598 | 2,913 | 勝利 (1.922x) |
| 準備完了までの起動時間、初回起動後(秒) | 14.25 | 154.3 | 勝利 (0.092x) |
| 最初のトークンまでの時間(秒) | 5.070 | 9.167 | 勝利 (0.553x) |
| エンドツーエンドのリクエスト(秒) | 8.339 | 13.29 | 勝利 (0.627x) |

1 台のマシン、1 基の GPU、1 回のセッション、固定の 15 文書セットで測定しました。各エンジンは自身が出荷する構成で実行しています。セットアップ、全指標、精度、制限事項、再現方法は [docs/benchmark.md](../benchmark.md) を参照してください。

<a id="quick-start"></a>
## クイックスタート

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

上記の設定をエクスポートすると、エンジンは 127.0.0.1:8002、OCR ブリッジは 127.0.0.1:8765 で待ち受けます。ページ画像は画像 data URL としてエンジンに、文書全体はブリッジに送信します。

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` は、ユーザーメッセージに base64 の `image_url` data URL を含む chat-completions のリクエストボディです。サーバーは何もダウンロードしません。ネットワークを使う手順は `install.sh` のみです。

## インストール

`./install.sh` は、リリース版バイナリ、GLM-OCR チェックポイント、レイアウトモデル、GB10 向けの CUDA 対応 onnxruntime をインストールします。各ステップはオプトイン方式でチェックサム検証されます。`./install.sh check` で結果を確認できます。ソースからビルドすることも可能です。詳細は [docs/install.md](../install.md) を参照してください。

## OCR ブリッジ

`scriptor serve` は文書 OCR ブリッジも起動します。PDF や画像を入力すると認識テキストを出力し、レイアウト検出には onnxruntime を使用します。フラグ、設定、組み込みのブリッジチェックは [docs/bridge.md](../bridge.md) にあります。

## ベンチマーク

完全な表、精度、制限事項、再現手順は [docs/benchmark.md](../benchmark.md) を参照してください。

## クラッシュ安全性

上限付きの受け入れ、メモリ圧迫の段階、監視下のワーカー、スーパーバイザーが対処できる終了コードについては [docs/serving.md](../serving.md#crash-safety) を参照してください。

<a id="documentation"></a>
## ドキュメント

ドキュメント(英語):

- [docs/install.md](../install.md) — インストールスクリプト、ビルド済みリリース、ソースからのビルド、モデルと依存ライブラリのライセンス
- [docs/serving.md](../serving.md) — `scriptor serve` のフラグ、リクエスト制限、クラッシュ安全性
- [docs/bridge.md](../bridge.md) — OCR ブリッジ、そのフラグと設定
- [docs/benchmark.md](../benchmark.md) — 結果、精度、制限事項、ベンチマークの再現
- [docs/boot.md](../boot.md) — 起動時のメモリ計画、起動ステージ、起動時間の測定方法
- [docs/drafting.md](../drafting.md) — MTP 投機的デコーディング、受理ポリシーとそのメトリクス
- [docs/attention.md](../attention.md) — アテンションカーネル
- [docs/gemm.md](../gemm.md) — GEMM カーネル
- [docs/equivalence.md](../equivalence.md) と [docs/e2e-equivalence.md](../e2e-equivalence.md) — カーネルおよびエンドツーエンドの等価性チェック
- [docs/engine-file.md](../engine-file.md) — エンジンネイティブの重みファイル
- [docs/model-facts.md](../model-facts.md) — エンジンが一致させる GLM-OCR の仕様
- [docs/vision-overlap.md](../vision-overlap.md) と [docs/nvjpeg.md](../nvjpeg.md) — ビジョン処理のオーバーラップと GPU JPEG デコード
- [seat/README.md](../../seat/README.md) — `seat` テンプレート、準備完了とドレインの契約、終了コード、障害注入
- [docs/phase0-protocol.md](../phase0-protocol.md) — 基準となる測定プロトコル
- [docs/third-party-notices.md](../third-party-notices.md) — サードパーティの通知
- [docs/languages.md](../languages.md) — この README の 38 言語版

<a id="contributing"></a>
## コントリビュート

まず [CONTRIBUTING.md](../../CONTRIBUTING.md) をお読みください。Conventional Commit 形式のコミット件名、フォーマット・clippy・テストのゲート、合成フィクスチャのみの使用が求められます。脆弱性は [SECURITY.md](../../SECURITY.md) の手順に従って非公開で報告してください。

---

<a id="license"></a>
## ライセンス

AGPL-3.0-only または商用ライセンスのデュアルライセンスです
([LICENSE](../../LICENSE) および [COMMERCIAL-LICENSE.md](../../COMMERCIAL-LICENSE.md) を参照)。
Copyright 2026 Matthew D. Murphy。商用ライセンスは
[Chiron Courseworks](https://chironcourseworks.com) を通じて発行されます。お問い合わせ:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com)。

サードパーティの通知は [docs/third-party-notices.md](../third-party-notices.md) にあります。モデルの重みとランタイムライブラリは別途ダウンロードするもので、それぞれのライセンスが適用されます([モデルと依存ライブラリのライセンス](../install.md#model-and-dependency-licenses)を参照)。
