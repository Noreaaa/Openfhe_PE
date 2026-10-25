# Ring 4096: Encrypted-Mask ReLU, Six-Stage Center-Area Sweep

Updated UTC: 2026-10-25T07:05:40.634470+00:00

## Configuration and scope

Ring dimension 4096; logical slots 2048; multiplicative depth 30; 30/40-bit scaling/first moduli; HYBRID digits 6; 96 OpenMP threads. Original input center areas: 10×10, 20×20, 30×30 and 40×40. One run per area, in separate sequential processes. Original convolutions 1–6, BN and ReLU execute on the server. The second convolution replaces max pooling. No client first-layer offload, residual additions, classifier or CKKS bootstrap.

ReLU uses CKKS → FHEW → encrypted sign → CKKS negative indicator → positive mask → multiplication with the original CKKS ciphertext. Activation does not decrypt FHEW signs. Reference decryption occurs only in verification outside operator timers. The normal production executable is used, not the decrypted-sign fixture.

Sparse convolution includes occupied-channel reduction, one-level grouped stride-2 compaction, owned in-place accumulation and working-level plaintext encoding. The latter is newer than earlier ring-scaling experiments, so those archived timings are not an identical-code comparison. Direct rotation keys and unrestricted output-channel concurrency are retained. An explicit experimental-depth option fixes depth at 30 below the conservative default margin; numerical checks validate each completed prefix.

## Measurement

Conv+BN and ReLU operator wall times exclude key generation, input encryption and verification. RSS is sampled every 0.5 seconds with phase markers. Operator-phase RSS includes resident keys and retained allocations from the same process, and is not an isolated cloud-process measurement. GNU time records whole-process wall time and lifetime peak RSS separately. No allocator trimming, network measurement or repeated-run averaging.

## Summary

| Center area | Status | Conv+BN (s) | ReLU (s) | Forward total (s / min) | Operator RSS mean / peak (GiB) | Whole-process peak (GiB) | Max absolute feature error |
|---|---|---:|---:|---|---|---:|---:|
| 10×10 | success | 24.810 | 246.176 | 270.986 / 4.516 | 4.115 / 5.430 | 5.471 | 0.0163894 |
| 20×20 | success | 38.279 | 357.751 | 396.030 / 6.601 | 4.573 / 6.302 | 6.318 | 0.00267803 |
| 30×30 | success | 60.753 | 521.895 | 582.648 / 9.711 | 4.969 / 7.165 | 7.169 | 0.0142904 |
| 40×40 | success | 95.169 | 713.082 | 808.251 / 13.471 | 5.449 / 8.022 | 8.027 | 0.0165188 |

## Per-layer measurements

| Area | Layer | Conv+BN (s) | ReLU (s) | Layer total (s) | Cumulative (s) | Ciphertexts | Level / towers | Operator RSS mean / peak (GiB) |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 10 | 1 | 12.880 | 53.567 | 66.447 | 66.447 | 32 | 12 / 19 | 4.591 / 5.430 |
| 10 | 2 | 3.754 | 16.602 | 20.356 | 86.803 | 10 | 15 / 16 | 4.021 / 4.514 |
| 10 | 3 | 2.462 | 26.252 | 28.714 | 115.517 | 14 | 18 / 13 | 3.944 / 4.211 |
| 10 | 4 | 2.207 | 37.771 | 39.979 | 155.496 | 18 | 21 / 10 | 3.935 / 4.153 |
| 10 | 5 | 2.192 | 50.116 | 52.309 | 207.804 | 22 | 24 / 7 | 3.959 / 4.045 |
| 10 | 6 | 1.314 | 61.868 | 63.182 | 270.986 | 26 | 27 / 4 | 3.956 / 3.969 |
| 20 | 1 | 23.633 | 98.357 | 121.990 | 121.990 | 52 | 12 / 19 | 5.694 / 6.302 |
| 20 | 2 | 5.330 | 27.277 | 32.608 | 154.597 | 14 | 15 / 16 | 4.151 / 4.671 |
| 20 | 3 | 3.135 | 38.258 | 41.393 | 195.990 | 18 | 18 / 13 | 4.108 / 4.373 |
| 20 | 4 | 2.384 | 51.389 | 53.773 | 249.763 | 22 | 21 / 10 | 4.066 / 4.326 |
| 20 | 5 | 2.062 | 62.597 | 64.659 | 314.422 | 26 | 24 / 7 | 4.064 / 4.212 |
| 20 | 6 | 1.735 | 79.873 | 81.608 | 396.030 | 30 | 27 / 4 | 4.017 / 4.040 |
| 30 | 1 | 40.516 | 151.422 | 191.938 | 191.938 | 72 | 12 / 19 | 6.671 / 7.165 |
| 30 | 2 | 7.686 | 43.063 | 50.749 | 242.687 | 20 | 15 / 16 | 4.260 / 4.980 |
| 30 | 3 | 3.831 | 58.109 | 61.940 | 304.627 | 24 | 18 / 13 | 4.145 / 4.582 |
| 30 | 4 | 3.339 | 72.313 | 75.652 | 380.279 | 28 | 21 / 10 | 4.107 / 4.459 |
| 30 | 5 | 3.181 | 88.719 | 91.900 | 472.179 | 32 | 24 / 7 | 4.102 / 4.338 |
| 30 | 6 | 2.199 | 108.270 | 110.469 | 582.648 | 36 | 27 / 4 | 4.108 / 4.114 |
| 40 | 1 | 70.966 | 214.928 | 285.893 | 285.893 | 92 | 12 / 19 | 7.490 / 8.022 |
| 40 | 2 | 9.819 | 64.540 | 74.359 | 360.253 | 26 | 15 / 16 | 4.621 / 5.378 |
| 40 | 3 | 4.706 | 80.265 | 84.971 | 445.224 | 30 | 18 / 13 | 4.343 / 4.878 |
| 40 | 4 | 4.158 | 97.217 | 101.375 | 546.599 | 34 | 21 / 10 | 4.368 / 4.694 |
| 40 | 5 | 3.228 | 118.404 | 121.633 | 668.232 | 38 | 24 / 7 | 4.248 / 4.543 |
| 40 | 6 | 2.292 | 137.728 | 140.020 | 808.251 | 42 | 27 / 4 | 4.217 / 4.229 |

Raw records per area: `layers.csv`, `run.log`, `rss.csv`, `time.txt`, `result.json`, `configuration.json`. Per-layer feature comparisons are retained in logs/results; they are not classifier accuracy. Binary hash and source snapshot are retained at the sweep root. Incomplete layers are not valid completed measurements.
