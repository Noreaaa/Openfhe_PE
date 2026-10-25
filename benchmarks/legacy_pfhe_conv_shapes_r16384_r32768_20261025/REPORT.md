# Historical PFHE Convolution: Representative ResNet Shapes

Updated UTC: 2026-10-25T09:40:50.279972+00:00

## Scope

The unmodified historical Conv2dBN_P forward implementation in src/cnn/conv2d.cpp is benchmarked, without SparseConv2dBN_P optimizations. Four main-path convolution indices in the project's modified ResNet are selected: 3 (64→64, 56×56), 8 (128→128, 28×28), 12 (256→256, 14×14), 16 (512→512, 7×7). All are 3×3, stride 1, padding 1, with trained weights and folded batch normalization. These represent stages 1–6, 7–10, 11–14 and 15–18 respectively; stem and downsampling convolutions are not separately characterized.

Ring dimensions: 16384 and 32768; slots=N/2 (8192 and 16384). Depth 30, 30/40-bit scaling/first moduli, HYBRID digits 6, 96 threads. Each case runs once in a fresh process. Original input sensitive square: center 10×10 in a 224×224 image. The region is propagated through the preceding sequential Conv+BN+ReLU prefix; by later stages it may cover the entire small feature map. Input feature values are obtained from that plaintext prefix without residual additions, so this is a convolution-shape benchmark rather than full ResNet inference.

Every tested convolution starts independently with freshly encrypted level-0, scale-degree-1 ciphertexts. This holds the ciphertext modulus state fixed across shapes; it does not reproduce the consumed levels of an 18-layer encrypted run. Only Conv+BN is timed. No ReLU, scheme-switching keys or bootstrap keys are evaluated/loaded in these isolated processes. Historical plaintext-to-ciphertext additions, public-key zero encryptions, EvalSum reductions and per-point output temporaries are intentionally retained.

## Measurements

Conv wall time excludes preparation, key generation, encryption and verification. Current RSS is sampled every 0.5 seconds during the operator; it includes resident CKKS keys, reference tensors and allocator retention in the same process. Whole-process peak is obtained independently from GNU time. Neither peak is a full pipeline memory measurement. Output reconstruction is checked against plaintext Conv+BN, including region propagation. No network transfer or allocator trimming.

| Layer group | Representative layer | Ring / slots | Input C×H×W | Output C×H×W | Encrypted output H×W | Status | Conv+BN (s) | Operator RSS mean / max (GiB) | Whole-process peak (GiB) | Max abs error |
|---|---:|---|---|---|---|---|---:|---|---:|---:|
| 1–6 | 3 | 16384 / 8192 | 64×56×56 | 64×56×56 | 7×7 | success | 371.098 | 10.222 / 12.352 | 12.560 | 0.00483491 |
| 7–10 | 8 | 16384 / 8192 | 128×28×28 | 128×28×28 | 9×9 | failed | 1112.812 | 19.459 / 23.216 | 23.287 | 0.0254355 |
| 11–14 | 12 | 16384 / 8192 | — | — | — | pending / running | — | — | — | — |
| 15–18 | 16 | 16384 / 8192 | — | — | — | pending / running | — | — | — | — |
| 1–6 | 3 | 32768 / 16384 | — | — | — | pending / running | — | — | — | — |
| 7–10 | 8 | 32768 / 16384 | — | — | — | pending / running | — | — | — | — |
| 11–14 | 12 | 32768 / 16384 | — | — | — | pending / running | — | — | — | — |
| 15–18 | 16 | 32768 / 16384 | — | — | — | pending / running | — | — | — | — |

Per-case raw files: `run.log`, `measurement.csv`, `rss.csv`, `time.txt`, `result.json`, `command.json`. Binary and source snapshots are retained. Only successful complete cases are valid measured results.

## Scope correction

Stopped after the user clarified fully encrypted channel-wise convolution. These legacy partial-encryption results are excluded from the requested baseline.
