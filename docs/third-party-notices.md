# Third-party notices

scriptor is dual-licensed under AGPL-3.0-only or a commercial license (see
`LICENSE` and `COMMERCIAL-LICENSE.md`). This file lists every
third-party component it builds on. The crate table is generated from
`cargo metadata` by `tools/gate/license-check.sh --write` and checked by the
same script without the flag.

## Rust crates

<!-- crates:begin -->
| Crate | Version | License | Source |
|---|---|---|---|
| ab_glyph | 0.2.32 | Apache-2.0 | crates.io |
| ab_glyph_rasterizer | 0.1.10 | Apache-2.0 | crates.io |
| adler2 | 2.0.1 | 0BSD OR MIT OR Apache-2.0 | crates.io |
| ahash | 0.8.12 | MIT OR Apache-2.0 | crates.io |
| aho-corasick | 1.1.5 | Unlicense OR MIT | crates.io |
| alloc-no-stdlib | 2.0.4 | BSD-3-Clause | crates.io |
| alloc-stdlib | 0.2.4 | BSD-3-Clause | crates.io |
| anstream | 1.0.0 | MIT OR Apache-2.0 | crates.io |
| anstyle | 1.0.14 | MIT OR Apache-2.0 | crates.io |
| anstyle-parse | 1.0.0 | MIT OR Apache-2.0 | crates.io |
| anstyle-query | 1.1.5 | MIT OR Apache-2.0 | crates.io |
| anstyle-wincon | 3.0.11 | MIT OR Apache-2.0 | crates.io |
| anyhow | 1.0.105 | MIT OR Apache-2.0 | crates.io |
| arrayvec | 0.7.8 | MIT OR Apache-2.0 | crates.io |
| atomic-waker | 1.1.2 | Apache-2.0 OR MIT | crates.io |
| autocfg | 1.5.1 | Apache-2.0 OR MIT | crates.io |
| axum | 0.8.9 | MIT | crates.io |
| axum-core | 0.5.6 | MIT | crates.io |
| base64 | 0.13.1 | MIT/Apache-2.0 | crates.io |
| base64 | 0.22.1 | MIT OR Apache-2.0 | crates.io |
| base64 | 0.23.1 | MIT OR Apache-2.0 | crates.io |
| bitflags | 1.3.2 | MIT/Apache-2.0 | crates.io |
| bitflags | 2.13.2 | MIT OR Apache-2.0 | crates.io |
| block-buffer | 0.10.4 | MIT OR Apache-2.0 | crates.io |
| brotli | 8.0.4 | BSD-3-Clause AND MIT | crates.io |
| brotli-decompressor | 5.0.3 | BSD-3-Clause/MIT | crates.io |
| bumpalo | 3.20.4 | MIT OR Apache-2.0 | crates.io |
| bytemuck | 1.25.2 | Zlib OR Apache-2.0 OR MIT | crates.io |
| bytemuck_derive | 1.12.1 | Zlib OR Apache-2.0 OR MIT | crates.io |
| byteorder-lite | 0.1.0 | Unlicense OR MIT | crates.io |
| bytes | 1.12.1 | MIT | crates.io |
| castaway | 0.2.4 | MIT | crates.io |
| cc | 1.11.1 | MIT OR Apache-2.0 | crates.io |
| cfg-if | 1.0.5 | MIT OR Apache-2.0 | crates.io |
| cfg_aliases | 0.2.2 | MIT | crates.io |
| chacha20 | 0.10.2 | MIT OR Apache-2.0 | crates.io |
| clap | 4.6.7 | MIT OR Apache-2.0 | crates.io |
| clap_builder | 4.6.7 | MIT OR Apache-2.0 | crates.io |
| clap_derive | 4.6.7 | MIT OR Apache-2.0 | crates.io |
| clap_lex | 1.1.1 | MIT OR Apache-2.0 | crates.io |
| color | 0.3.3 | Apache-2.0 OR MIT | crates.io |
| colorchoice | 1.0.5 | MIT OR Apache-2.0 | crates.io |
| compact_str | 0.9.1 | MIT | crates.io |
| core-foundation | 0.10.1 | MIT OR Apache-2.0 | crates.io |
| core-foundation-sys | 0.8.7 | MIT OR Apache-2.0 | crates.io |
| cpufeatures | 0.2.17 | MIT OR Apache-2.0 | crates.io |
| cpufeatures | 0.3.1 | MIT OR Apache-2.0 | crates.io |
| crc32fast | 1.5.4 | MIT OR Apache-2.0 | crates.io |
| crossbeam-deque | 0.8.8 | MIT OR Apache-2.0 | crates.io |
| crossbeam-epoch | 0.9.21 | MIT OR Apache-2.0 | crates.io |
| crossbeam-utils | 0.8.23 | MIT OR Apache-2.0 | crates.io |
| crunchy | 0.2.4 | MIT | crates.io |
| crypto-common | 0.1.7 | MIT OR Apache-2.0 | crates.io |
| cudarc | 0.19.10 | MIT OR Apache-2.0 | crates.io |
| darling | 0.20.11 | MIT | crates.io |
| darling_core | 0.20.11 | MIT | crates.io |
| darling_macro | 0.20.11 | MIT | crates.io |
| dary_heap | 0.3.9 | MIT OR Apache-2.0 | crates.io |
| derive_builder | 0.20.2 | MIT OR Apache-2.0 | crates.io |
| derive_builder_core | 0.20.2 | MIT OR Apache-2.0 | crates.io |
| derive_builder_macro | 0.20.2 | MIT OR Apache-2.0 | crates.io |
| digest | 0.10.7 | MIT OR Apache-2.0 | crates.io |
| displaydoc | 0.2.7 | MIT OR Apache-2.0 | crates.io |
| either | 1.19.0 | MIT OR Apache-2.0 | crates.io |
| equivalent | 1.0.3 | Apache-2.0 OR MIT | crates.io |
| errno | 0.3.14 | MIT OR Apache-2.0 | crates.io |
| esaxx-rs | 0.1.10 | Apache-2.0 | crates.io |
| euclid | 0.22.14 | MIT OR Apache-2.0 | crates.io |
| fastrand | 2.5.0 | Apache-2.0 OR MIT | crates.io |
| fax | 0.2.7 | MIT | crates.io |
| fdeflate | 0.3.7 | MIT OR Apache-2.0 | crates.io |
| fearless_simd | 0.7.0 | Apache-2.0 OR MIT | crates.io |
| fearless_simd | 1.3.0 | Apache-2.0 OR MIT | crates.io |
| find-msvc-tools | 0.1.15 | MIT OR Apache-2.0 | crates.io |
| flate2 | 1.1.11 | MIT OR Apache-2.0 | crates.io |
| fnv | 1.0.7 | Apache-2.0 / MIT | crates.io |
| foldhash | 0.2.0 | Zlib | crates.io |
| font-types | 0.12.7 | MIT OR Apache-2.0 | crates.io |
| form_urlencoded | 1.2.3 | MIT OR Apache-2.0 | crates.io |
| futures-channel | 0.3.34 | MIT OR Apache-2.0 | crates.io |
| futures-core | 0.3.34 | MIT OR Apache-2.0 | crates.io |
| futures-io | 0.3.34 | MIT OR Apache-2.0 | crates.io |
| futures-sink | 0.3.34 | MIT OR Apache-2.0 | crates.io |
| futures-task | 0.3.34 | MIT OR Apache-2.0 | crates.io |
| futures-util | 0.3.34 | MIT OR Apache-2.0 | crates.io |
| generic-array | 0.14.7 | MIT | crates.io |
| getrandom | 0.2.17 | MIT OR Apache-2.0 | crates.io |
| getrandom | 0.3.4 | MIT OR Apache-2.0 | crates.io |
| getrandom | 0.4.3 | MIT OR Apache-2.0 | crates.io |
| glifo | 0.4.0 | Apache-2.0 OR MIT | crates.io |
| guillotiere | 0.7.0 | MIT/Apache-2.0 | crates.io |
| half | 2.7.1 | MIT OR Apache-2.0 | crates.io |
| hashbrown | 0.17.1 | MIT OR Apache-2.0 | crates.io |
| hayro | 0.8.0 | Apache-2.0 OR MIT | crates.io |
| hayro-ccitt | 0.4.0 | Apache-2.0 OR MIT | crates.io |
| hayro-cmap | 0.1.1 | Apache-2.0 OR MIT | crates.io |
| hayro-interpret | 0.8.0 | Apache-2.0 OR MIT | crates.io |
| hayro-jbig2 | 0.3.1 | Apache-2.0 OR MIT | crates.io |
| hayro-jpeg2000 | 0.4.1 | Apache-2.0 OR MIT | crates.io |
| hayro-postscript | 0.2.0 | Apache-2.0 OR MIT | crates.io |
| hayro-syntax | 0.8.0 | Apache-2.0 OR MIT | crates.io |
| heck | 0.5.0 | MIT OR Apache-2.0 | crates.io |
| http | 1.5.0 | MIT OR Apache-2.0 | crates.io |
| http-body | 1.1.0 | MIT | crates.io |
| http-body-util | 0.1.5 | MIT | crates.io |
| httparse | 1.10.1 | MIT OR Apache-2.0 | crates.io |
| httpdate | 1.0.3 | MIT OR Apache-2.0 | crates.io |
| hyper | 1.12.0 | MIT | crates.io |
| hyper-rustls | 0.27.10 | Apache-2.0 OR ISC OR MIT | crates.io |
| hyper-util | 0.1.21 | MIT | crates.io |
| icu_collections | 2.4.0 | Unicode-3.0 | crates.io |
| icu_locale_core | 2.4.0 | Unicode-3.0 | crates.io |
| icu_normalizer | 2.4.0 | Unicode-3.0 | crates.io |
| icu_normalizer_data | 2.4.0 | Unicode-3.0 | crates.io |
| icu_properties | 2.4.0 | Unicode-3.0 | crates.io |
| icu_properties_data | 2.4.0 | Unicode-3.0 | crates.io |
| icu_provider | 2.4.0 | Unicode-3.0 | crates.io |
| ident_case | 1.0.1 | MIT/Apache-2.0 | crates.io |
| idna | 1.1.0 | MIT OR Apache-2.0 | crates.io |
| idna_adapter | 1.2.2 | Apache-2.0 OR MIT | crates.io |
| image | 0.25.10 | MIT OR Apache-2.0 | crates.io |
| image-webp | 0.2.4 | MIT OR Apache-2.0 | crates.io |
| indexmap | 2.14.2 | Apache-2.0 OR MIT | crates.io |
| ipnet | 2.12.2 | MIT OR Apache-2.0 | crates.io |
| is_terminal_polyfill | 1.70.2 | MIT OR Apache-2.0 | crates.io |
| itertools | 0.14.0 | MIT OR Apache-2.0 | crates.io |
| itoa | 1.0.18 | MIT OR Apache-2.0 | crates.io |
| js-sys | 0.3.107 | MIT OR Apache-2.0 | crates.io |
| kurbo | 0.13.1 | Apache-2.0 OR MIT | crates.io |
| libc | 0.2.191 | MIT OR Apache-2.0 | crates.io |
| libloading | 0.9.0 | ISC | crates.io |
| linebender_resource_handle | 0.1.1 | Apache-2.0 OR MIT | crates.io |
| linux-raw-sys | 0.12.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | crates.io |
| litemap | 0.8.3 | Unicode-3.0 | crates.io |
| log | 0.4.35 | MIT OR Apache-2.0 | crates.io |
| lru-slab | 0.1.3 | MIT OR Apache-2.0 OR Zlib | crates.io |
| macro_rules_attribute | 0.2.3 | Apache-2.0 OR MIT OR Zlib | crates.io |
| macro_rules_attribute-proc_macro | 0.2.3 | Apache-2.0 OR MIT OR Zlib | crates.io |
| matchit | 0.8.4 | MIT AND BSD-3-Clause | crates.io |
| matrixmultiply | 0.3.11 | MIT/Apache-2.0 | crates.io |
| memchr | 2.8.3 | Unlicense OR MIT | crates.io |
| memmap2 | 0.9.12 | MIT OR Apache-2.0 | crates.io |
| mime | 0.3.17 | MIT OR Apache-2.0 | crates.io |
| minimal-lexical | 0.2.1 | MIT/Apache-2.0 | crates.io |
| miniz_oxide | 0.8.9 | MIT OR Zlib OR Apache-2.0 | crates.io |
| miniz_oxide | 0.9.1 | MIT OR Zlib OR Apache-2.0 | crates.io |
| mio | 1.2.4 | MIT | crates.io |
| monostate | 0.1.18 | MIT OR Apache-2.0 | crates.io |
| monostate-impl | 0.1.18 | MIT OR Apache-2.0 | crates.io |
| moxcms | 0.8.1 | BSD-3-Clause OR Apache-2.0 | crates.io |
| ndarray | 0.17.2 | MIT OR Apache-2.0 | crates.io |
| nom | 7.1.3 | MIT | crates.io |
| num-complex | 0.4.6 | MIT OR Apache-2.0 | crates.io |
| num-integer | 0.1.47 | MIT OR Apache-2.0 | crates.io |
| num-traits | 0.2.19 | MIT OR Apache-2.0 | crates.io |
| once_cell | 1.21.4 | MIT OR Apache-2.0 | crates.io |
| once_cell_polyfill | 1.70.2 | MIT OR Apache-2.0 | crates.io |
| onig | 6.5.3 | MIT | crates.io |
| onig_sys | 69.9.3 | MIT | crates.io |
| openssl-probe | 0.2.2 | MIT OR Apache-2.0 | crates.io |
| ort | 2.0.0-rc.13 | MIT OR Apache-2.0 | crates.io |
| ort-sys | 2.0.0-rc.13 | MIT OR Apache-2.0 | crates.io |
| owned_ttf_parser | 0.25.1 | Apache-2.0 | crates.io |
| paste | 1.0.15 | MIT OR Apache-2.0 | crates.io |
| pastey | 0.2.3 | MIT OR Apache-2.0 | crates.io |
| peniko | 0.6.1 | Apache-2.0 OR MIT | crates.io |
| percent-encoding | 2.3.2 | MIT OR Apache-2.0 | crates.io |
| phf | 0.13.1 | MIT | crates.io |
| phf_generator | 0.13.1 | MIT | crates.io |
| phf_macros | 0.13.1 | MIT | crates.io |
| phf_shared | 0.13.1 | MIT | crates.io |
| pic-scale | 0.7.14 | BSD-3-Clause OR Apache-2.0 | crates.io |
| pin-project-lite | 0.2.17 | Apache-2.0 OR MIT | crates.io |
| pkg-config | 0.3.34 | MIT OR Apache-2.0 | crates.io |
| png | 0.17.16 | MIT OR Apache-2.0 | crates.io |
| png | 0.18.1 | MIT OR Apache-2.0 | crates.io |
| polycool | 0.4.0 | MIT OR Apache-2.0 | crates.io |
| portable-atomic | 1.15.0 | Apache-2.0 OR MIT | crates.io |
| portable-atomic-util | 0.2.8 | Apache-2.0 OR MIT | crates.io |
| potential_utf | 0.1.6 | Unicode-3.0 | crates.io |
| ppv-lite86 | 0.2.21 | MIT OR Apache-2.0 | crates.io |
| proc-macro2 | 1.0.107 | MIT OR Apache-2.0 | crates.io |
| pxfm | 0.1.30 | BSD-3-Clause OR Apache-2.0 | crates.io |
| quick-error | 2.0.1 | MIT/Apache-2.0 | crates.io |
| quinn | 0.11.12 | MIT OR Apache-2.0 | crates.io |
| quinn-proto | 0.11.19 | MIT OR Apache-2.0 | crates.io |
| quinn-udp | 0.5.16 | MIT OR Apache-2.0 | crates.io |
| quote | 1.0.47 | MIT OR Apache-2.0 | crates.io |
| r-efi | 5.3.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later | crates.io |
| r-efi | 6.0.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later | crates.io |
| rand | 0.10.4 | MIT OR Apache-2.0 | crates.io |
| rand | 0.9.5 | MIT OR Apache-2.0 | crates.io |
| rand_chacha | 0.9.0 | MIT OR Apache-2.0 | crates.io |
| rand_core | 0.10.1 | MIT OR Apache-2.0 | crates.io |
| rand_core | 0.9.5 | MIT OR Apache-2.0 | crates.io |
| rand_pcg | 0.10.3 | MIT OR Apache-2.0 | crates.io |
| rawpointer | 0.2.1 | MIT/Apache-2.0 | crates.io |
| rayon | 1.12.0 | MIT OR Apache-2.0 | crates.io |
| rayon-cond | 0.4.0 | Apache-2.0/MIT | crates.io |
| rayon-core | 1.13.0 | MIT OR Apache-2.0 | crates.io |
| read-fonts | 0.41.0 | MIT OR Apache-2.0 | crates.io |
| read-fonts | 0.43.3 | MIT OR Apache-2.0 | crates.io |
| regex | 1.13.1 | MIT OR Apache-2.0 | crates.io |
| regex-automata | 0.4.18 | MIT OR Apache-2.0 | crates.io |
| regex-syntax | 0.8.11 | MIT OR Apache-2.0 | crates.io |
| reqwest | 0.12.28 | MIT OR Apache-2.0 | crates.io |
| ring | 0.17.14 | Apache-2.0 AND ISC | crates.io |
| rustc-hash | 2.1.3 | Apache-2.0 OR MIT | crates.io |
| rustix | 1.1.6 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | crates.io |
| rustls | 0.23.46 | Apache-2.0 OR ISC OR MIT | crates.io |
| rustls-native-certs | 0.8.4 | Apache-2.0 OR ISC OR MIT | crates.io |
| rustls-pki-types | 1.16.0 | MIT OR Apache-2.0 | crates.io |
| rustls-webpki | 0.103.17 | ISC | crates.io |
| rustversion | 1.0.23 | MIT OR Apache-2.0 | crates.io |
| ryu | 1.0.23 | Apache-2.0 OR BSL-1.0 | crates.io |
| schannel | 0.1.30 | MIT | crates.io |
| security-framework | 3.7.0 | MIT OR Apache-2.0 | crates.io |
| security-framework-sys | 2.17.0 | MIT OR Apache-2.0 | crates.io |
| serde | 1.0.229 | MIT OR Apache-2.0 | crates.io |
| serde_core | 1.0.229 | MIT OR Apache-2.0 | crates.io |
| serde_derive | 1.0.229 | MIT OR Apache-2.0 | crates.io |
| serde_json | 1.0.154 | MIT OR Apache-2.0 | crates.io |
| serde_path_to_error | 0.1.20 | MIT OR Apache-2.0 | crates.io |
| serde_urlencoded | 0.7.1 | MIT/Apache-2.0 | crates.io |
| sha2 | 0.10.9 | MIT OR Apache-2.0 | crates.io |
| shlex | 2.0.1 | MIT OR Apache-2.0 | crates.io |
| signal-hook-registry | 1.4.8 | MIT OR Apache-2.0 | crates.io |
| simd-adler32 | 0.3.10 | MIT | crates.io |
| siphasher | 1.0.4 | MIT OR Apache-2.0 | crates.io |
| skrifa | 0.44.0 | MIT OR Apache-2.0 | crates.io |
| skrifa | 0.46.2 | MIT OR Apache-2.0 | crates.io |
| slab | 0.4.12 | MIT | crates.io |
| smallvec | 1.16.5 | MIT OR Apache-2.0 | crates.io |
| socket2 | 0.6.5 | MIT OR Apache-2.0 | crates.io |
| spm_precompiled | 0.1.4 | Apache-2.0 | crates.io |
| stable_deref_trait | 1.2.1 | MIT OR Apache-2.0 | crates.io |
| static_assertions | 1.1.0 | MIT OR Apache-2.0 | crates.io |
| strsim | 0.11.1 | MIT | crates.io |
| subtle | 2.6.1 | BSD-3-Clause | crates.io |
| syn | 2.0.119 | MIT OR Apache-2.0 | crates.io |
| syn | 3.0.9 | MIT OR Apache-2.0 | crates.io |
| sync_wrapper | 1.0.2 | Apache-2.0 | crates.io |
| synstructure | 0.14.0 | MIT | crates.io |
| tempfile | 3.27.0 | MIT OR Apache-2.0 | crates.io |
| thiserror | 2.0.21 | MIT OR Apache-2.0 | crates.io |
| thiserror-impl | 2.0.21 | MIT OR Apache-2.0 | crates.io |
| tiff | 0.11.3 | MIT | crates.io |
| tinystr | 0.8.5 | Unicode-3.0 | crates.io |
| tinyvec | 1.13.3 | Zlib OR Apache-2.0 OR MIT | crates.io |
| tokenizers | 0.22.2 | Apache-2.0 | crates.io |
| tokio | 1.54.1 | MIT | crates.io |
| tokio-macros | 2.7.2 | MIT | crates.io |
| tokio-rustls | 0.26.6 | MIT OR Apache-2.0 | crates.io |
| tokio-stream | 0.1.19 | MIT | crates.io |
| tower | 0.5.3 | MIT | crates.io |
| tower-http | 0.6.11 | MIT | crates.io |
| tower-layer | 0.3.3 | MIT | crates.io |
| tower-service | 0.3.3 | MIT | crates.io |
| tracing | 0.1.44 | MIT | crates.io |
| tracing-core | 0.1.36 | MIT | crates.io |
| try-lock | 0.2.5 | MIT | crates.io |
| ttf-parser | 0.25.1 | MIT OR Apache-2.0 | crates.io |
| typenum | 1.20.1 | MIT OR Apache-2.0 | crates.io |
| unicode-ident | 1.0.27 | (MIT OR Apache-2.0) AND Unicode-3.0 | crates.io |
| unicode-normalization-alignments | 0.1.12 | MIT/Apache-2.0 | crates.io |
| unicode-segmentation | 1.13.3 | MIT OR Apache-2.0 | crates.io |
| unicode_categories | 0.1.1 | MIT OR Apache-2.0 | crates.io |
| untrusted | 0.9.0 | ISC | crates.io |
| ureq | 3.5.1 | MIT OR Apache-2.0 | crates.io |
| ureq-proto | 0.7.1 | MIT OR Apache-2.0 | crates.io |
| url | 2.5.9 | MIT OR Apache-2.0 | crates.io |
| utf8-zero | 0.8.1 | MIT OR Apache-2.0 | crates.io |
| utf8_iter | 1.1.0 | Apache-2.0 OR MIT | crates.io |
| utf8parse | 0.2.2 | Apache-2.0 OR MIT | crates.io |
| utf_types | 1.0.0 | Apache-2.0 OR MIT | crates.io |
| vello_common | 0.3.0 | Apache-2.0 OR MIT | crates.io |
| vello_cpu | 0.3.0 | Apache-2.0 OR MIT | crates.io |
| version_check | 0.9.5 | MIT/Apache-2.0 | crates.io |
| want | 0.3.2 | MIT | crates.io |
| wasi | 0.11.1+wasi-snapshot-preview1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | crates.io |
| wasip2 | 1.0.4+wasi-0.2.12 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | crates.io |
| wasm-bindgen | 0.2.130 | MIT OR Apache-2.0 | crates.io |
| wasm-bindgen-futures | 0.4.80 | MIT OR Apache-2.0 | crates.io |
| wasm-bindgen-macro | 0.2.130 | MIT OR Apache-2.0 | crates.io |
| wasm-bindgen-macro-support | 0.2.130 | MIT OR Apache-2.0 | crates.io |
| wasm-bindgen-shared | 0.2.130 | MIT OR Apache-2.0 | crates.io |
| web-sys | 0.3.107 | MIT OR Apache-2.0 | crates.io |
| web-time | 1.1.0 | MIT OR Apache-2.0 | crates.io |
| webpki-roots | 1.0.12 | CDLA-Permissive-2.0 | crates.io |
| weezl | 0.1.12 | MIT OR Apache-2.0 | crates.io |
| windows-link | 0.2.1 | MIT OR Apache-2.0 | crates.io |
| windows-sys | 0.52.0 | MIT OR Apache-2.0 | crates.io |
| windows-sys | 0.61.2 | MIT OR Apache-2.0 | crates.io |
| windows-targets | 0.52.6 | MIT OR Apache-2.0 | crates.io |
| windows_aarch64_gnullvm | 0.52.6 | MIT OR Apache-2.0 | crates.io |
| windows_aarch64_msvc | 0.52.6 | MIT OR Apache-2.0 | crates.io |
| windows_i686_gnu | 0.52.6 | MIT OR Apache-2.0 | crates.io |
| windows_i686_gnullvm | 0.52.6 | MIT OR Apache-2.0 | crates.io |
| windows_i686_msvc | 0.52.6 | MIT OR Apache-2.0 | crates.io |
| windows_x86_64_gnu | 0.52.6 | MIT OR Apache-2.0 | crates.io |
| windows_x86_64_gnullvm | 0.52.6 | MIT OR Apache-2.0 | crates.io |
| windows_x86_64_msvc | 0.52.6 | MIT OR Apache-2.0 | crates.io |
| wit-bindgen | 0.57.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | crates.io |
| writeable | 0.6.4 | Unicode-3.0 | crates.io |
| yoke | 0.8.4 | Unicode-3.0 | crates.io |
| yoke-derive | 0.8.4 | Unicode-3.0 | crates.io |
| zerocopy | 0.8.63 | BSD-2-Clause OR Apache-2.0 OR MIT | crates.io |
| zerocopy-derive | 0.8.63 | BSD-2-Clause OR Apache-2.0 OR MIT | crates.io |
| zerofrom | 0.1.8 | Unicode-3.0 | crates.io |
| zerofrom-derive | 0.1.8 | Unicode-3.0 | crates.io |
| zeroize | 1.9.2 | Apache-2.0 OR MIT | crates.io |
| zerotrie | 0.2.5 | Unicode-3.0 | crates.io |
| zerovec | 0.11.9 | Unicode-3.0 | crates.io |
| zerovec-derive | 0.11.7 | Unicode-3.0 | crates.io |
| zlib-rs | 0.6.8 | Zlib | crates.io |
| zmij | 1.0.23 | MIT | crates.io |
| zune-core | 0.5.3 | MIT OR Apache-2.0 OR Zlib | crates.io |
| zune-jpeg | 0.5.15 | MIT OR Apache-2.0 OR Zlib | crates.io |
<!-- crates:end -->

## Reused non-crate sources

Adapted source files carry the header from `kernels/NOTICE-template.txt`.

| Source | License | Use | Notice location |
|---|---|---|---|
| vLLM kernels and processors | Apache-2.0 | adapted `.cu` files and GLM-OCR preprocessing behaviour | header of each adapted file under `kernels/`; upstream notice retained verbatim |
| FlashInfer | Apache-2.0 | adapted attention kernels | header of each adapted file under `kernels/` |
| CUTLASS | BSD-3-Clause | vendored only if used | `kernels/` header of each file that includes it; upstream license text retained |
| libjpeg-turbo | IJG AND BSD-3-Clause AND Zlib | JPEG decoding behind the `libjpeg` feature | linked system library; license text retained in distributions |
