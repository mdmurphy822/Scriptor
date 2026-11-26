<p align="center"><img src="docs/assets/scriptor-banner.svg" alt="Scriptor" width="630"></p>

<div dir="rtl">

<div align="center">

*خادم استدلال أصلي لنموذج GLM-OCR على NVIDIA GB10: مسح ضوئي كمدخل ونص معترف به كمخرج، عبر واجهة برمجية متوافقة مع OpenAI، ومن دون أي حالة تهيئة محفوظة.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

**[صفحة المشروع](https://chironcourseworks.com/scriptor)** • [البدء السريع](#quick-start) • [التثبيت](docs/install.md) • [جسر OCR](docs/bridge.md) • [اختبار الأداء](docs/benchmark.md) • [التوثيق](#documentation) • [المساهمة](#contributing) • [الترخيص](#license)

[اللغات](docs/languages.md) • [English](README.md)

</div>

## مقدمة

Scriptor (كلمة لاتينية تعني «الكاتب») محرك استدلال محلي فقط ومحسَّن للإنتاجية لنموذج GLM-OCR، مكتوب بلغة Rust مع نوى CUDA مكتوبة يدويًا، ومبني لمعالج NVIDIA GB10 في DGX Spark (بنية ARM64). يقدّم GLM-OCR عبر نقطة نهاية chat-completions متوافقة مع OpenAI، ويقبل صور الصفحات ضمن الطلب، ويعرض مقاييس Prometheus بالصيغة `vllm:*` المتوافقة مع vLLM، ولذلك يعمل أي عميل أو لوحة مراقبة مبنية لخادم vLLM من دون تعديل. وهو محلي و`fail-closed` (يتوقف بدلًا من التخمين عند فقدان مدخل أو خطئه): دليل النموذج وعنوان الربط يُحدَّدان صراحةً، ولا يُنزَّل شيء، ولا يُكتشف شيء، وأي مدخل مفقود أو قديم أو غير مطابق يوقف التشغيل بدلًا من الالتفاف عليه بالتخمين. راجع [docs/serving.md](docs/serving.md) لقائمة الميزات وعقد السلامة عند الأعطال.

## أبرز النتائج

Scriptor مقابل vLLM، بالنموذج والمستندات وبروتوكول جسر OCR ووحدة GPU نفسها: 10 تشغيلات مزدوجة لكل محرك، مع إقلاع جديد في كل تشغيل، و368 صفحة في كل تشغيل، ونسب مع فترات ثقة 95% بطريقة bootstrap المزدوجة.

| المقياس | Scriptor p50 | vLLM p50 | النتيجة |
| ------ | -----------: | -------: | ------- |
| الزمن الكلي، 368 صفحة (ث) | 236.7 | 454.8 | فوز (0.520x) |
| الإنتاجية (صفحة/ساعة) | 5,598 | 2,913 | فوز (1.922x) |
| الإقلاع حتى الجاهزية، بعد الإقلاع الأول (ث) | 14.25 | 154.3 | فوز (0.092x) |
| زمن أول رمز (ث) | 5.070 | 9.167 | فوز (0.553x) |
| الطلب من البداية إلى النهاية (ث) | 8.339 | 13.29 | فوز (0.627x) |

جهاز واحد، ووحدة GPU واحدة، وجلسة واحدة، ومجموعة ثابتة من 15 مستندًا؛ وقد شغّل كل محرك إعداده الخاص المُشحون. الإعداد وجميع المقاييس والدقة والقيود وطريقة إعادة الإنتاج: [docs/benchmark.md](docs/benchmark.md).

<a id="quick-start"></a>
## البدء السريع

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

عند تصدير هذه الإعدادات، يستمع المحرك على 127.0.0.1:8002 ويستمع جسر OCR على 127.0.0.1:8765. أرسل صورة الصفحة إلى المحرك على هيئة data URL للصورة، وأرسل المستند كاملًا إلى الجسر:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` هو جسم طلب chat-completions تحمل رسالة المستخدم فيه data URL بصيغة base64 ضمن `image_url`. لا ينزّل الخادم شيئًا؛ و`install.sh` هو الخطوة الوحيدة التي تستخدم الشبكة.

## التثبيت

يثبّت `./install.sh` الملف التنفيذي الجاهز وحزمة نموذج GLM-OCR ونموذج تحليل التخطيط ونسخة onnxruntime المبنية بـ CUDA لمعالج GB10، وكل خطوة اختيارية ويُتحقق من مجموعها الاختباري؛ ويتحقق `./install.sh check` من النتيجة. ويمكنك أيضًا البناء من المصدر. راجع [docs/install.md](docs/install.md).

## جسر OCR

يشغّل `scriptor serve` أيضًا جسرًا للتعرف الضوئي على المستندات: ملفات PDF وصور كمدخل ونص معترف به كمخرج، مع اكتشاف التخطيط عبر onnxruntime. الخيارات والإعدادات وفحص الجسر المدمج موجودة في [docs/bridge.md](docs/bridge.md).

## اختبار الأداء

الجداول الكاملة والدقة والقيود وخطوات إعادة الإنتاج: [docs/benchmark.md](docs/benchmark.md).

## السلامة عند الأعطال

استقبال محدود، ومستويات ضغط الذاكرة، وعمليات عاملة خاضعة للإشراف، ورموز خروج يستطيع المشرف التصرف بناءً عليها: [docs/serving.md](docs/serving.md#crash-safety).

<a id="documentation"></a>
## التوثيق

التوثيق (بالإنجليزية):

- [docs/install.md](docs/install.md) — سكربت التثبيت، والإصدار الجاهز، والبناء من المصدر، وتراخيص النموذج والاعتماديات
- [docs/serving.md](docs/serving.md) — خيارات `scriptor serve`، وحدود الطلبات، والسلامة عند الأعطال
- [docs/bridge.md](docs/bridge.md) — جسر OCR وخياراته وإعداداته
- [docs/benchmark.md](docs/benchmark.md) — النتائج والدقة والقيود وإعادة إنتاج اختبار الأداء
- [docs/boot.md](docs/boot.md) — خطة ذاكرة الإقلاع ومراحله وكيفية قياس زمنه
- [docs/drafting.md](docs/drafting.md) — فك الترميز التخميني MTP وسياسة القبول ومقاييسه
- [docs/attention.md](docs/attention.md) — نوى الانتباه
- [docs/gemm.md](docs/gemm.md) — نوى GEMM
- [docs/equivalence.md](docs/equivalence.md) و[docs/e2e-equivalence.md](docs/e2e-equivalence.md) — فحوص التكافؤ للنوى ومن البداية إلى النهاية
- [docs/engine-file.md](docs/engine-file.md) — ملف الأوزان الأصلي للمحرك
- [docs/model-facts.md](docs/model-facts.md) — حقائق GLM-OCR التي يطابقها المحرك
- [docs/vision-overlap.md](docs/vision-overlap.md) و[docs/nvjpeg.md](docs/nvjpeg.md) — تداخل الرؤية وفك ترميز JPEG على GPU
- [seat/README.md](seat/README.md) — قالب `seat` وعقود الجاهزية والتفريغ ورموز الخروج وحقن الأعطال
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — بروتوكول القياس المرجعي
- [docs/third-party-notices.md](docs/third-party-notices.md) — إشعارات الأطراف الثالثة
- [docs/languages.md](docs/languages.md) — هذا الملف التمهيدي بـ 38 لغة

<a id="contributing"></a>
## المساهمة

اقرأ [CONTRIBUTING.md](CONTRIBUTING.md) أولًا: عناوين الإيداع بصيغة Conventional Commit، وبوابة التنسيق وclippy والاختبارات، ومعطيات اختبار اصطناعية فقط. أبلغ عن الثغرات الأمنية بسرية كما هو موضح في [SECURITY.md](SECURITY.md).

---

<a id="license"></a>
## الترخيص

مرخَّص بترخيص مزدوج: AGPL-3.0-only أو ترخيص تجاري (راجع
[LICENSE](LICENSE) و[COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
حقوق النشر 2026 Matthew D. Murphy. تُصدَر التراخيص التجارية عبر
[Chiron Courseworks](https://chironcourseworks.com)؛ للاستفسارات:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

إشعارات الأطراف الثالثة موجودة في [docs/third-party-notices.md](docs/third-party-notices.md). أوزان النموذج ومكتبات التشغيل تُنزَّل بشكل منفصل وتحتفظ بتراخيصها الخاصة (راجع [Model and dependency licenses](docs/install.md#model-and-dependency-licenses)).

</div>
