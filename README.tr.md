<p align="center"><img src="docs/assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div align="center">

*NVIDIA GB10 için yerel bir GLM-OCR çıkarım sunucusu: taramalar girer, tanınan metin OpenAI uyumlu bir API üzerinden çıkar; kaydedilmiş sıcak durum yoktur.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[Proje sayfası](https://chironcourseworks.com/scriptor)** • [Hızlı başlangıç](#quick-start) • [Kurulum](docs/install.md) • [OCR köprüsü](docs/bridge.md) • [Kıyaslama](docs/benchmark.md) • [Dokümantasyon](#documentation) • [Katkıda bulunma](#contributing) • [Lisans](#license)

[Diller](docs/languages.md) • [English](README.md)

</div>

## Giriş

Scriptor (Latince "kâtip"), GLM-OCR modeli için üretilmiş, verim odaklı ve yalnızca yerel çalışan bir çıkarım motorudur; Rust ve elle yazılmış CUDA çekirdekleriyle geliştirilmiştir ve DGX Spark'taki (ARM64) NVIDIA GB10 için tasarlanmıştır. GLM-OCR'ı OpenAI uyumlu bir chat-completions uç noktası üzerinden sunar, istekte sayfa görüntülerini kabul eder ve vLLM uyumlu `vllm:*` Prometheus metriklerini yayımlar; böylece bir vLLM sunucusu için yazılmış istemci veya pano değişiklik yapılmadan çalışır. Yerel ve `fail-closed` çalışır (girdi eksik veya hatalıysa tahmin yürütmek yerine durur): model dizini ve bind adresi açıkça verilir, hiçbir şey indirilmez, hiçbir şey keşfedilmez; eksik, eski veya uyumsuz bir girdi tahminle aşılmaz, başlatmayı durdurur. Özellik listesi ve çökme güvenliği sözleşmesi için [docs/serving.md](docs/serving.md) dosyasına bakın.

## Öne çıkan sonuçlar

Aynı model, belgeler, OCR köprüsü protokolü ve GPU üzerinde Scriptor ile vLLM karşılaştırması: motor başına 10 eşleştirilmiş çalıştırma, her çalıştırma için yeni bir açılış, çalıştırma başına 368 sayfa; oranlar %95 eşleştirilmiş bootstrap güven aralıklarıyla verilmiştir.

| Metrik | Scriptor p50 | vLLM p50 | Sonuç |
| ------ | -----------: | -------: | ------- |
| Duvar saati süresi, 368 sayfa (sn) | 236.7 | 454.8 | KAZANÇ (0.520x) |
| Verim (sayfa/saat) | 5,598 | 2,913 | KAZANÇ (1.922x) |
| Hazır olana kadar açılış, ilk açılıştan sonra (sn) | 14.25 | 154.3 | KAZANÇ (0.092x) |
| İlk token'a kadar geçen süre (sn) | 5.070 | 9.167 | KAZANÇ (0.553x) |
| Uçtan uca istek (sn) | 8.339 | 13.29 | KAZANÇ (0.627x) |

Tek makine, tek GPU, tek oturum, sabit 15 belgelik bir küme; her motor kendi sevk edilen yapılandırmasıyla çalıştırıldı. Kurulum, tüm metrikler, doğruluk, sınırlamalar ve nasıl yeniden üretileceği: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## Hızlı başlangıç

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Bu ayarlar dışa aktarıldığında motor 127.0.0.1:8002 adresini, OCR köprüsü ise 127.0.0.1:8765 adresini dinler. Motora sayfa görüntüsünü görüntü data URL'si olarak, köprüye ise belgenin tamamını gönderin:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json`, kullanıcı mesajında base64 biçiminde bir `image_url` data URL'si taşıyan bir chat-completions gövdesidir. Sunucu hiçbir şey indirmez; ağa erişen tek adım `install.sh` betiğidir.

## Kurulum

`./install.sh`; sürüm ikili dosyasını, GLM-OCR kontrol noktasını, yerleşim modelini ve GB10 için CUDA ile derlenmiş bir onnxruntime'ı kurar; her adım isteğe bağlıdır ve sağlama toplamıyla doğrulanır. `./install.sh check` sonucu doğrular. Kaynaktan da derleyebilirsiniz. Bkz. [docs/install.md](docs/install.md).

## OCR köprüsü

`scriptor serve` ayrıca bir belge OCR köprüsü çalıştırır: PDF ve görüntüler girer, tanınan metin çıkar; yerleşim tespiti onnxruntime ile yapılır. Bayraklar, ayarlar ve yerleşik köprü denetimi [docs/bridge.md](docs/bridge.md) içindedir.

## Kıyaslama

Tüm tablolar, doğruluk, sınırlamalar ve yeniden üretim adımları: [docs/benchmark.md](docs/benchmark.md).

## Çökme güvenliği

Sınırlı kabul, bellek baskısı kademeleri, denetlenen işçiler ve bir süpervizörün işlem yapabileceği çıkış kodları: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## Dokümantasyon

Dokümantasyon (İngilizce):

- [docs/install.md](docs/install.md) — kurulum betiği, hazır sürüm, kaynaktan derleme, model ve bağımlılık lisansları
- [docs/serving.md](docs/serving.md) — `scriptor serve` bayrakları, istek sınırları, çökme güvenliği
- [docs/bridge.md](docs/bridge.md) — OCR köprüsü, bayrakları ve ayarları
- [docs/benchmark.md](docs/benchmark.md) — sonuçlar, doğruluk, sınırlamalar, kıyaslamanın yeniden üretilmesi
- [docs/boot.md](docs/boot.md) — açılış bellek planı, açılış aşamaları ve açılış süresinin nasıl ölçüldüğü
- [docs/drafting.md](docs/drafting.md) — MTP spekülatif kod çözme, kabul politikası ve metrikleri
- [docs/attention.md](docs/attention.md) — dikkat (attention) çekirdekleri
- [docs/gemm.md](docs/gemm.md) — GEMM çekirdekleri
- [docs/equivalence.md](docs/equivalence.md) ve [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — çekirdek ve uçtan uca denklik denetimleri
- [docs/engine-file.md](docs/engine-file.md) — motorun yerel ağırlık dosyası
- [docs/model-facts.md](docs/model-facts.md) — motorun uyduğu GLM-OCR bilgileri
- [docs/vision-overlap.md](docs/vision-overlap.md) ve [docs/nvjpeg.md](docs/nvjpeg.md) — görü örtüşmesi ve GPU JPEG kod çözme
- [seat/README.md](seat/README.md) — `seat` şablonu, hazırlık ve boşaltma (drain) sözleşmeleri, çıkış kodları ve hata enjeksiyonu
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — başvuru ölçüm protokolü
- [docs/third-party-notices.md](docs/third-party-notices.md) — üçüncü taraf bildirimleri
- [docs/languages.md](docs/languages.md) — bu README'nin 38 dildeki sürümleri

<a id="contributing"></a>
## Katkıda bulunma

Önce [CONTRIBUTING.md](CONTRIBUTING.md) dosyasını okuyun: Conventional Commit başlıkları, biçim, clippy ve test kapısı, yalnızca sentetik test verileri. Güvenlik açıklarını [SECURITY.md](SECURITY.md) dosyasında anlatıldığı gibi gizli olarak bildirin.

---

<a id="license"></a>
## Lisans

AGPL-3.0-only veya ticari lisans altında çifte lisanslıdır
([LICENSE](LICENSE) ve [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) dosyalarına bakın).
Telif hakkı 2026 Matthew D. Murphy. Ticari lisanslar
[Chiron Courseworks](https://chironcourseworks.com) aracılığıyla verilir; sorular için:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Üçüncü taraf bildirimleri [docs/third-party-notices.md](docs/third-party-notices.md) dosyasındadır. Model ağırlıkları ve çalışma zamanı kütüphaneleri ayrı indirmelerdir ve kendi lisanslarını korur (bkz. [Model ve bağımlılık lisansları](docs/install.md#model-and-dependency-licenses)).
