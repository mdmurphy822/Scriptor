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

*NVIDIA GB10 के लिए एक नेटिव GLM-OCR इन्फ़रेंस सर्वर: स्कैन अंदर, पहचाना गया टेक्स्ट बाहर, OpenAI-संगत API के ज़रिए, बिना किसी सहेजी गई वार्म स्थिति के।*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[प्रोजेक्ट पेज](https://chironcourseworks.com/scriptor)** • [त्वरित शुरुआत](#quick-start) • [इंस्टॉल](docs/install.md) • [OCR ब्रिज](docs/bridge.md) • [बेंचमार्क](docs/benchmark.md) • [दस्तावेज़ीकरण](#documentation) • [योगदान](#contributing) • [लाइसेंस](#license)

[भाषाएँ](docs/languages.md) • [English](README.md)

</div>

## परिचय

Scriptor (लैटिन में "लिपिक") GLM-OCR मॉडल के लिए एक थ्रूपुट-अनुकूलित, केवल-स्थानीय इन्फ़रेंस इंजन है, जो Rust में हाथ से लिखे CUDA कर्नेल के साथ बना है और DGX Spark (ARM64) के NVIDIA GB10 के लिए तैयार किया गया है। यह GLM-OCR को OpenAI-संगत चैट-कम्प्लीशन एंडपॉइंट के ज़रिए परोसता है, अनुरोध में पेज की छवियाँ स्वीकार करता है, और vLLM-संगत `vllm:*` Prometheus मेट्रिक्स उपलब्ध कराता है, इसलिए vLLM सर्वर के लिए बनाया गया क्लाइंट या डैशबोर्ड बिना बदलाव के चलता है। यह स्थानीय और `fail-closed` है (कोई इनपुट गायब या ग़लत हो तो अनुमान लगाने के बजाय रुक जाता है): मॉडल डायरेक्टरी और बाइंड पता स्पष्ट रूप से दिए जाते हैं, कुछ भी डाउनलोड नहीं होता, कुछ भी खोजा नहीं जाता, और कोई इनपुट गायब, पुराना या बेमेल हो तो अनुमान लगाकर आगे बढ़ने के बजाय स्टार्ट रुक जाता है। सुविधाओं की सूची और क्रैश-सुरक्षा अनुबंध के लिए [docs/serving.md](docs/serving.md) देखें।

## मुख्य परिणाम

Scriptor बनाम vLLM, एक ही मॉडल, दस्तावेज़, OCR ब्रिज प्रोटोकॉल और GPU: प्रति इंजन 10 युग्मित रन, हर रन के लिए नया बूट, प्रति रन 368 पेज, अनुपात 95% युग्मित बूटस्ट्रैप विश्वास अंतराल के साथ।

| मेट्रिक | Scriptor p50 | vLLM p50 | निष्कर्ष |
| ------ | -----------: | -------: | ------- |
| कुल समय, 368 पेज (s) | 236.7 | 454.8 | जीत (0.520x) |
| थ्रूपुट (पेज/घंटा) | 5,598 | 2,913 | जीत (1.922x) |
| पहले बूट के बाद, तैयार होने तक बूट (s) | 14.25 | 154.3 | जीत (0.092x) |
| पहले टोकन तक का समय (s) | 5.070 | 9.167 | जीत (0.553x) |
| एंड-टू-एंड अनुरोध (s) | 8.339 | 13.29 | जीत (0.627x) |

एक मशीन, एक GPU, एक सत्र, 15 दस्तावेज़ों का निश्चित सेट; हर इंजन अपने शिप किए गए कॉन्फ़िगरेशन पर चला। सेटअप, सभी मेट्रिक्स, सटीकता, सीमाएँ और इन्हें दोहराने का तरीका: [docs/benchmark.md](docs/benchmark.md)।

<a id="quick-start"></a>
## त्वरित शुरुआत

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

इन सेटिंग्स को एक्सपोर्ट करने के बाद इंजन 127.0.0.1:8002 पर और OCR ब्रिज 127.0.0.1:8765 पर सुनता है। इंजन को पेज की छवि इमेज डेटा URL के रूप में भेजें, और ब्रिज को पूरा दस्तावेज़:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` एक चैट-कम्प्लीशन बॉडी है जिसके यूज़र संदेश में base64 `image_url` डेटा URL होता है। सर्वर कुछ भी डाउनलोड नहीं करता; नेटवर्क का उपयोग करने वाला एकमात्र चरण `install.sh` है।

## इंस्टॉल

`./install.sh` GB10 के लिए रिलीज़ बाइनरी, GLM-OCR चेकपॉइंट, लेआउट मॉडल और onnxruntime का CUDA बिल्ड इंस्टॉल करता है; हर चरण वैकल्पिक (opt-in) और चेकसम से सत्यापित है; `./install.sh check` परिणाम की जाँच करता है। आप सोर्स से भी बिल्ड कर सकते हैं। [docs/install.md](docs/install.md) देखें।

## OCR ब्रिज

`scriptor serve` एक दस्तावेज़ OCR ब्रिज भी चलाता है: PDF और छवियाँ अंदर, पहचाना गया टेक्स्ट बाहर, और लेआउट पहचान onnxruntime के ज़रिए। फ़्लैग, सेटिंग्स और ब्रिज की अंतर्निहित जाँच [docs/bridge.md](docs/bridge.md) में हैं।

## बेंचमार्क

पूरी तालिकाएँ, सटीकता, सीमाएँ और दोहराने के चरण: [docs/benchmark.md](docs/benchmark.md)।

## क्रैश सुरक्षा

सीमित इनटेक, मेमोरी-दबाव के स्तर, निगरानी में चलने वाले वर्कर और ऐसे एग्ज़िट कोड जिन पर सुपरवाइज़र कार्रवाई कर सके: [docs/serving.md](docs/serving.md#crash-safety)।

<a id="documentation"></a>
## दस्तावेज़ीकरण

दस्तावेज़ीकरण (अंग्रेज़ी में):

- [docs/install.md](docs/install.md) — इंस्टॉल स्क्रिप्ट, पहले से बनी रिलीज़, सोर्स से बिल्ड, मॉडल और डिपेंडेंसी के लाइसेंस
- [docs/serving.md](docs/serving.md) — `scriptor serve` के फ़्लैग, अनुरोध सीमाएँ, क्रैश सुरक्षा
- [docs/bridge.md](docs/bridge.md) — OCR ब्रिज, उसके फ़्लैग और सेटिंग्स
- [docs/benchmark.md](docs/benchmark.md) — परिणाम, सटीकता, सीमाएँ, बेंचमार्क दोहराना
- [docs/boot.md](docs/boot.md) — बूट मेमोरी योजना, बूट चरण और बूट समय कैसे मापा जाता है
- [docs/drafting.md](docs/drafting.md) — MTP स्पेकुलेटिव डिकोडिंग, स्वीकृति नीति और उसके मेट्रिक्स
- [docs/attention.md](docs/attention.md) — अटेंशन कर्नेल
- [docs/gemm.md](docs/gemm.md) — GEMM कर्नेल
- [docs/equivalence.md](docs/equivalence.md) और [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — कर्नेल और एंड-टू-एंड समतुल्यता जाँच
- [docs/engine-file.md](docs/engine-file.md) — इंजन-नेटिव वेट फ़ाइल
- [docs/model-facts.md](docs/model-facts.md) — GLM-OCR के वे तथ्य जिनसे इंजन मेल खाता है
- [docs/vision-overlap.md](docs/vision-overlap.md) और [docs/nvjpeg.md](docs/nvjpeg.md) — विज़न ओवरलैप और GPU JPEG डिकोड
- [seat/README.md](seat/README.md) — `seat` टेम्पलेट, रेडीनेस और ड्रेन अनुबंध, एग्ज़िट कोड और फ़ॉल्ट इंजेक्शन
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — संदर्भ माप प्रोटोकॉल
- [docs/third-party-notices.md](docs/third-party-notices.md) — तृतीय-पक्ष सूचनाएँ
- [docs/languages.md](docs/languages.md) — यह README 38 भाषाओं में

<a id="contributing"></a>
## योगदान

पहले [CONTRIBUTING.md](CONTRIBUTING.md) पढ़ें: Conventional Commit विषय, फ़ॉर्मैट, clippy और टेस्ट गेट, केवल कृत्रिम (synthetic) फ़िक्स्चर। सुरक्षा कमज़ोरियों की रिपोर्ट [SECURITY.md](SECURITY.md) में बताए अनुसार निजी तौर पर करें।

---

<a id="license"></a>
## लाइसेंस

AGPL-3.0-only या वाणिज्यिक लाइसेंस के तहत दोहरे-लाइसेंस प्राप्त (देखें
[LICENSE](LICENSE) और [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md))।
Copyright 2026 Matthew D. Murphy. वाणिज्यिक लाइसेंस
[Chiron Courseworks](https://chironcourseworks.com) के माध्यम से जारी किए जाते हैं; पूछताछ:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com)।

तृतीय-पक्ष सूचनाएँ [docs/third-party-notices.md](docs/third-party-notices.md) में हैं। मॉडल वेट और रनटाइम लाइब्रेरी अलग डाउनलोड हैं और अपने-अपने लाइसेंस बनाए रखती हैं (देखें [मॉडल और डिपेंडेंसी लाइसेंस](docs/install.md#model-and-dependency-licenses))।
