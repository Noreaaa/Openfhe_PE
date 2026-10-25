# Fully Encrypted Channel-Wise Convolution: Representative ResNet Shapes

Updated UTC: 2026-10-25T12:13:01.345770+00:00

## Scope

The channel-wise convolution helper shared with the prior channelwise_prefix_bench baseline is benchmarked. It follows the historical one-ciphertext-per-channel design, with padding, folded BN and cached spatial rotations. There is no partial-encryption bypass or sparse row packing. Four main-path convolution indices in the project's modified ResNet are selected: 3 (64→64, 56×56), 8 (128→128, 28×28), 12 (256→256, 14×14), 16 (512→512, 7×7). All are 3×3, stride 1, padding 1, with trained weights and folded batch normalization. These represent stages 1–6, 7–10, 11–14 and 15–18 respectively; stem and downsampling convolutions are not separately characterized.

Ring dimensions: 16384 and 32768; slots=N/2 (8192 and 16384). Depth 30, 30/40-bit scaling/first moduli, HYBRID digits 6, 96 threads. Each case runs once in a fresh process. Every input feature value is encrypted. Each channel is independently packed into one ciphertext with contiguous H×W spatial slots; unused slots are zero. No channel sharing, spatial compaction or encrypted-region sweep is performed. This fresh per-layer layout differs from the retained 112-pitch grid of the earlier end-to-end prefix; it isolates the convolution shape itself. Input feature values are obtained from that plaintext prefix without residual additions, so this is a convolution-shape benchmark rather than full ResNet inference.

Every tested convolution starts independently with freshly encrypted level-0, scale-degree-1 ciphertexts. This holds the ciphertext modulus state fixed across shapes; it does not reproduce the consumed levels of an 18-layer encrypted run. Only Conv+BN is timed. No ReLU, scheme-switching keys or bootstrap keys are evaluated/loaded in these isolated processes. Only the eight required spatial rotation keys are generated. The shared convolution caches rotated inputs and accumulates ciphertext/plaintext products across all input channels; no ReLU or bootstrapping occurs.

## Measurements

Conv wall time excludes preparation, key generation, encryption and verification. Current RSS is sampled every 0.5 seconds during the operator; it includes resident CKKS keys, reference tensors and allocator retention in the same process. Whole-process peak is obtained independently from GNU time. Neither peak is a full pipeline memory measurement. Output reconstruction is checked against plaintext Conv+BN, at every output feature position. No network transfer or allocator trimming.

| Layer group | Representative layer | Ring / slots | Input C×H×W | Output C×H×W | Fully encrypted output H×W | Status | Conv+BN (s) | Operator RSS mean / max (GiB) | Whole-process peak (GiB) | Max abs error |
|---|---:|---|---|---|---|---|---:|---|---:|---:|
| 1–6 | 3 | 16384 / 8192 | 64×56×56 | 64×56×56 | 56×56 | success | 67.588 | 6.125 / 6.734 | 6.803 | 0.000948513 |
| 7–10 | 8 | 16384 / 8192 | 128×28×28 | 128×28×28 | 28×28 | success | 199.769 | 11.736 / 12.458 | 12.667 | 0.00611684 |
| 11–14 | 12 | 16384 / 8192 | 256×14×14 | 256×14×14 | 14×14 | success | 733.350 | 21.165 / 22.504 | 22.541 | 0.00387463 |
| 15–18 | 16 | 16384 / 8192 | 512×7×7 | 512×7×7 | 7×7 | success | 2916.835 | 39.401 / 40.683 | 40.761 | 0.00935678 |
| 1–6 | 3 | 32768 / 16384 | 64×56×56 | 64×56×56 | 56×56 | success | 216.574 | 13.184 / 14.697 | 14.795 | 0.0070782 |
| 7–10 | 8 | 32768 / 16384 | 128×28×28 | 128×28×28 | 28×28 | failed | 1525.374 | 23.425 / 24.121 | 24.136 | 0.0297416 |
| 11–14 | 12 | 32768 / 16384 | — | — | — | pending / running | — | — | — | — |
| 15–18 | 16 | 32768 / 16384 | — | — | — | pending / running | — | — | — | — |

Per-case raw files: `run.log`, `measurement.csv`, `rss.csv`, `time.txt`, `result.json`, `command.json`. Binary and source snapshots are retained. Only successful complete cases are valid measured results.
