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

*NVIDIA GB10-এর জন্য একটি নেটিভ GLM-OCR ইনফারেন্স সার্ভার: স্ক্যান ভেতরে, শনাক্ত করা টেক্সট বাইরে, OpenAI-সামঞ্জস্যপূর্ণ API-র মাধ্যমে, কোনো সংরক্ষিত ওয়ার্ম অবস্থা ছাড়াই।*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[প্রকল্পের পাতা](https://chironcourseworks.com/scriptor)** • [দ্রুত শুরু](#quick-start) • [ইনস্টল](docs/install.md) • [OCR ব্রিজ](docs/bridge.md) • [বেঞ্চমার্ক](docs/benchmark.md) • [ডকুমেন্টেশন](#documentation) • [অবদান](#contributing) • [লাইসেন্স](#license)

[ভাষা](docs/languages.md) • [English](README.md)

</div>

## ভূমিকা

Scriptor (ল্যাটিন ভাষায় "লিপিকার") GLM-OCR মডেলের জন্য একটি থ্রুপুট-অপ্টিমাইজড, শুধুমাত্র-লোকাল ইনফারেন্স ইঞ্জিন, যা Rust-এ হাতে লেখা CUDA কার্নেল দিয়ে তৈরি এবং DGX Spark (ARM64)-এর NVIDIA GB10-এর জন্য নির্মিত। এটি OpenAI-সামঞ্জস্যপূর্ণ চ্যাট-কমপ্লিশন এন্ডপয়েন্টের মাধ্যমে GLM-OCR পরিবেশন করে, অনুরোধে পেজের ছবি গ্রহণ করে এবং vLLM-সামঞ্জস্যপূর্ণ `vllm:*` Prometheus মেট্রিক্স প্রকাশ করে, ফলে vLLM সার্ভারের জন্য তৈরি ক্লায়েন্ট বা ড্যাশবোর্ড অপরিবর্তিত অবস্থায় কাজ করে। এটি লোকাল এবং `fail-closed` (কোনো ইনপুট অনুপস্থিত বা ভুল হলে আন্দাজ না করে থেমে যায়): মডেল ডিরেক্টরি ও বাইন্ড ঠিকানা স্পষ্টভাবে দিতে হয়, কিছু ডাউনলোড হয় না, কিছু খুঁজে বের করা হয় না, এবং কোনো ইনপুট অনুপস্থিত, পুরোনো বা অমিল হলে আন্দাজ করে এগোনোর বদলে স্টার্ট থেমে যায়। ফিচার তালিকা ও ক্র্যাশ-সুরক্ষা চুক্তির জন্য [docs/serving.md](docs/serving.md) দেখুন।

## মূল ফলাফল

Scriptor বনাম vLLM, একই মডেল, নথি, OCR ব্রিজ প্রোটোকল ও GPU: প্রতি ইঞ্জিনে ১০টি জোড়াবদ্ধ রান, প্রতি রানে নতুন বুট, প্রতি রানে ৩৬৮ পেজ, ৯৫% জোড়াবদ্ধ বুটস্ট্র্যাপ আস্থা-ব্যবধানসহ অনুপাত।

| মেট্রিক | Scriptor p50 | vLLM p50 | ফলাফল |
| ------ | -----------: | -------: | ------- |
| মোট সময়, 368 পেজ (s) | 236.7 | 454.8 | জয় (0.520x) |
| থ্রুপুট (পেজ/ঘণ্টা) | 5,598 | 2,913 | জয় (1.922x) |
| প্রথম বুটের পর, প্রস্তুত হওয়া পর্যন্ত বুট (s) | 14.25 | 154.3 | জয় (0.092x) |
| প্রথম টোকেন পর্যন্ত সময় (s) | 5.070 | 9.167 | জয় (0.553x) |
| এন্ড-টু-এন্ড অনুরোধ (s) | 8.339 | 13.29 | জয় (0.627x) |

একটি মেশিন, একটি GPU, একটি সেশন, ১৫টি নথির নির্দিষ্ট সেট; প্রতিটি ইঞ্জিন তার নিজস্ব শিপ করা কনফিগারেশনে চলেছে। সেটআপ, সব মেট্রিক, নির্ভুলতা, সীমাবদ্ধতা এবং পুনরুৎপাদনের উপায়: [docs/benchmark.md](docs/benchmark.md)।

<a id="quick-start"></a>
## দ্রুত শুরু

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

এই সেটিংস এক্সপোর্ট করা থাকলে ইঞ্জিন 127.0.0.1:8002-এ এবং OCR ব্রিজ 127.0.0.1:8765-এ শোনে। ইঞ্জিনে পেজের ছবি ইমেজ ডেটা URL হিসেবে পাঠান, আর ব্রিজে পুরো নথি:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` একটি চ্যাট-কমপ্লিশন বডি, যার ইউজার মেসেজে base64 `image_url` ডেটা URL থাকে। সার্ভার কিছুই ডাউনলোড করে না; নেটওয়ার্ক ব্যবহার করা একমাত্র ধাপ `install.sh`।

## ইনস্টল

`./install.sh` GB10-এর জন্য রিলিজ বাইনারি, GLM-OCR চেকপয়েন্ট, লেআউট মডেল এবং onnxruntime-এর একটি CUDA বিল্ড ইনস্টল করে; প্রতিটি ধাপ ঐচ্ছিক (opt-in) ও চেকসাম দিয়ে যাচাই করা; `./install.sh check` ফলাফল যাচাই করে। আপনি সোর্স থেকেও বিল্ড করতে পারেন। [docs/install.md](docs/install.md) দেখুন।

## OCR ব্রিজ

`scriptor serve` একটি নথি OCR ব্রিজও চালায়: PDF ও ছবি ভেতরে, শনাক্ত করা টেক্সট বাইরে, আর লেআউট শনাক্তকরণ onnxruntime-এর মাধ্যমে। ফ্ল্যাগ, সেটিংস এবং ব্রিজের অন্তর্নির্মিত যাচাই [docs/bridge.md](docs/bridge.md)-এ আছে।

## বেঞ্চমার্ক

পূর্ণ টেবিল, নির্ভুলতা, সীমাবদ্ধতা এবং পুনরুৎপাদনের ধাপ: [docs/benchmark.md](docs/benchmark.md)।

## ক্র্যাশ সুরক্ষা

সীমাবদ্ধ ইনটেক, মেমরি-চাপের স্তর, তত্ত্বাবধানে চলা ওয়ার্কার এবং এমন এক্সিট কোড যার ওপর সুপারভাইজার ব্যবস্থা নিতে পারে: [docs/serving.md](docs/serving.md#crash-safety)।

<a id="documentation"></a>
## ডকুমেন্টেশন

ডকুমেন্টেশন (ইংরেজিতে):

- [docs/install.md](docs/install.md) — ইনস্টল স্ক্রিপ্ট, প্রি-বিল্ট রিলিজ, সোর্স থেকে বিল্ড, মডেল ও ডিপেন্ডেন্সির লাইসেন্স
- [docs/serving.md](docs/serving.md) — `scriptor serve`-এর ফ্ল্যাগ, অনুরোধের সীমা, ক্র্যাশ সুরক্ষা
- [docs/bridge.md](docs/bridge.md) — OCR ব্রিজ, তার ফ্ল্যাগ ও সেটিংস
- [docs/benchmark.md](docs/benchmark.md) — ফলাফল, নির্ভুলতা, সীমাবদ্ধতা, বেঞ্চমার্ক পুনরুৎপাদন
- [docs/boot.md](docs/boot.md) — বুট মেমরি পরিকল্পনা, বুটের ধাপ এবং বুট সময় কীভাবে মাপা হয়
- [docs/drafting.md](docs/drafting.md) — MTP স্পেকুলেটিভ ডিকোডিং, গ্রহণ নীতি ও তার মেট্রিক্স
- [docs/attention.md](docs/attention.md) — অ্যাটেনশন কার্নেল
- [docs/gemm.md](docs/gemm.md) — GEMM কার্নেল
- [docs/equivalence.md](docs/equivalence.md) এবং [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — কার্নেল ও এন্ড-টু-এন্ড সমতুল্যতা যাচাই
- [docs/engine-file.md](docs/engine-file.md) — ইঞ্জিন-নেটিভ ওয়েট ফাইল
- [docs/model-facts.md](docs/model-facts.md) — GLM-OCR-এর যে তথ্যের সাথে ইঞ্জিন মেলে
- [docs/vision-overlap.md](docs/vision-overlap.md) এবং [docs/nvjpeg.md](docs/nvjpeg.md) — ভিশন ওভারল্যাপ ও GPU JPEG ডিকোড
- [seat/README.md](seat/README.md) — `seat` টেমপ্লেট, রেডিনেস ও ড্রেন চুক্তি, এক্সিট কোড এবং ফল্ট ইনজেকশন
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — রেফারেন্স পরিমাপ প্রোটোকল
- [docs/third-party-notices.md](docs/third-party-notices.md) — তৃতীয়-পক্ষের বিজ্ঞপ্তি
- [docs/languages.md](docs/languages.md) — এই README ৩৮টি ভাষায়

<a id="contributing"></a>
## অবদান

প্রথমে [CONTRIBUTING.md](CONTRIBUTING.md) পড়ুন: Conventional Commit সাবজেক্ট, ফরম্যাট, clippy ও টেস্ট গেট, শুধুমাত্র কৃত্রিম (synthetic) ফিক্সচার। নিরাপত্তা দুর্বলতার রিপোর্ট [SECURITY.md](SECURITY.md)-এ বর্ণিত পদ্ধতিতে ব্যক্তিগতভাবে করুন।

---

<a id="license"></a>
## লাইসেন্স

AGPL-3.0-only অথবা একটি বাণিজ্যিক লাইসেন্সের অধীনে দ্বৈত-লাইসেন্সপ্রাপ্ত (দেখুন
[LICENSE](LICENSE) ও [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md))।
Copyright 2026 Matthew D. Murphy. বাণিজ্যিক লাইসেন্স
[Chiron Courseworks](https://chironcourseworks.com)-এর মাধ্যমে দেওয়া হয়; অনুসন্ধান:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com)।

তৃতীয়-পক্ষের বিজ্ঞপ্তি [docs/third-party-notices.md](docs/third-party-notices.md)-এ আছে। মডেল ওয়েট ও রানটাইম লাইব্রেরি আলাদা ডাউনলোড এবং নিজ নিজ লাইসেন্স বজায় রাখে (দেখুন [মডেল ও ডিপেন্ডেন্সি লাইসেন্স](docs/install.md#model-and-dependency-licenses))।
