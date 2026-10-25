# Ring 4096: Encrypted-Mask ReLU, Six-Stage Top-Left-Corner Area Sweep

Updated UTC: 2026-10-25T07:38:43.277795+00:00

## Configuration and scope

Ring dimension 4096; logical slots 2048; multiplicative depth 30; 30/40-bit scaling/first moduli; HYBRID digits 6; 96 OpenMP threads. Original input top-left corner square areas: 10×10, 20×20, 30×30 and 40×40. One run per area, in separate sequential processes. Original convolutions 1–6, BN and ReLU execute on the server. The second convolution replaces max pooling. No client first-layer offload, residual additions, classifier or CKKS bootstrap.

ReLU uses CKKS → FHEW → encrypted sign → CKKS negative indicator → positive mask → multiplication with the original CKKS ciphertext. Activation does not decrypt FHEW signs. Reference decryption occurs only in verification outside operator timers. The normal production executable is used, not the decrypted-sign fixture.

Sparse convolution includes occupied-channel reduction, one-level grouped stride-2 compaction, owned in-place accumulation and working-level plaintext encoding. The latter is newer than earlier ring-scaling experiments, so those archived timings are not an identical-code comparison. Direct rotation keys and unrestricted output-channel concurrency are retained. An explicit experimental-depth option fixes depth at 30 below the conservative default margin; numerical checks validate each completed prefix.

The sensitive square touches both top and left image boundaries: rows 0..area−1 and columns 0..area−1, inclusive. This is a corner experiment, not an edge-centered square or a border band. Image bounds clip receptive-field expansion on two sides. The executable is copied from the completed center sweep to hold the implementation fixed.

## Measurement

Conv+BN and ReLU operator wall times exclude key generation, input encryption and verification. RSS is sampled every 0.5 seconds with phase markers. Operator-phase RSS includes resident keys and retained allocations from the same process, and is not an isolated cloud-process measurement. GNU time records whole-process wall time and lifetime peak RSS separately. No allocator trimming, network measurement or repeated-run averaging.

## Summary

| Corner area | Status | Conv+BN (s) | ReLU (s) | Forward total (s / min) | Operator RSS mean / peak (GiB) | Whole-process peak (GiB) | Max absolute feature error |
|---|---|---:|---:|---|---|---:|---:|
| 10×10 | success | 19.185 | 154.024 | 173.209 / 2.887 | 4.134 / 5.407 | 5.414 | 0.00385544 |
| 20×20 | success | 33.655 | 274.309 | 307.964 / 5.133 | 4.561 / 6.248 | 6.264 | 0.0164468 |
| 30×30 | success | 52.160 | 393.822 | 445.983 / 7.433 | 5.108 / 6.997 | 7.013 | 0.00296395 |
| 40×40 | success | 93.542 | 564.691 | 658.233 / 10.971 | 5.536 / 7.862 | 7.865 | 0.00535548 |

## Per-layer measurements

| Area | Layer | Conv+BN (s) | ReLU (s) | Layer total (s) | Cumulative (s) | Ciphertexts | Level / towers | Operator RSS mean / peak (GiB) |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 10 | 1 | 10.472 | 44.144 | 54.616 | 54.616 | 28 | 12 / 19 | 4.379 / 5.407 |
| 10 | 2 | 2.794 | 13.741 | 16.535 | 71.151 | 8 | 15 / 16 | 3.941 / 4.276 |
| 10 | 3 | 1.887 | 16.803 | 18.690 | 89.841 | 10 | 18 / 13 | 3.966 / 4.062 |
| 10 | 4 | 1.617 | 21.053 | 22.669 | 112.511 | 12 | 21 / 10 | 4.034 / 4.048 |
| 10 | 5 | 1.437 | 26.304 | 27.740 | 140.251 | 14 | 24 / 7 | 4.050 / 4.054 |
| 10 | 6 | 0.978 | 31.980 | 32.958 | 173.209 | 16 | 27 / 4 | 4.048 / 4.054 |
| 20 | 1 | 20.278 | 88.327 | 108.605 | 108.605 | 48 | 12 / 19 | 5.443 / 6.248 |
| 20 | 2 | 4.650 | 26.552 | 31.202 | 139.807 | 14 | 15 / 16 | 4.114 / 4.703 |
| 20 | 3 | 2.963 | 31.244 | 34.206 | 174.013 | 16 | 18 / 13 | 4.092 / 4.355 |
| 20 | 4 | 2.359 | 37.853 | 40.212 | 214.225 | 18 | 21 / 10 | 4.024 / 4.271 |
| 20 | 5 | 2.046 | 41.050 | 43.096 | 257.321 | 20 | 24 / 7 | 4.084 / 4.111 |
| 20 | 6 | 1.359 | 49.284 | 50.643 | 307.964 | 22 | 27 / 4 | 4.082 / 4.089 |
| 30 | 1 | 35.518 | 140.058 | 175.576 | 175.576 | 68 | 12 / 19 | 6.585 / 6.997 |
| 30 | 2 | 6.436 | 37.756 | 44.192 | 219.768 | 18 | 15 / 16 | 4.298 / 5.021 |
| 30 | 3 | 3.636 | 43.649 | 47.285 | 267.054 | 20 | 18 / 13 | 4.141 / 4.517 |
| 30 | 4 | 2.935 | 51.754 | 54.690 | 321.744 | 22 | 21 / 10 | 4.123 / 4.394 |
| 30 | 5 | 2.088 | 57.744 | 59.832 | 381.576 | 24 | 24 / 7 | 4.103 / 4.223 |
| 30 | 6 | 1.546 | 62.861 | 64.407 | 445.983 | 26 | 27 / 4 | 4.093 / 4.111 |
| 40 | 1 | 71.362 | 201.474 | 272.836 | 272.836 | 88 | 12 / 19 | 7.366 / 7.862 |
| 40 | 2 | 8.931 | 57.239 | 66.170 | 339.006 | 24 | 15 / 16 | 4.546 / 5.330 |
| 40 | 3 | 4.659 | 63.806 | 68.465 | 407.471 | 26 | 18 / 13 | 4.257 / 4.722 |
| 40 | 4 | 3.584 | 72.767 | 76.351 | 483.822 | 28 | 21 / 10 | 4.153 / 4.551 |
| 40 | 5 | 2.996 | 81.353 | 84.348 | 568.171 | 30 | 24 / 7 | 4.169 / 4.361 |
| 40 | 6 | 2.009 | 88.053 | 90.063 | 658.233 | 32 | 27 / 4 | 4.137 / 4.144 |

Raw records per area: `layers.csv`, `run.log`, `rss.csv`, `time.txt`, `result.json`, `configuration.json`. Per-layer feature comparisons are retained in logs/results; they are not classifier accuracy. Binary hash and source snapshot are retained at the sweep root. Incomplete layers are not valid completed measurements.

## Center versus top-left corner

Same executable and HE parameters; one run per region.

| Area | Center total (s) | Corner total (s) | Speedup | Center / corner lifetime peak (GiB) |
|---|---:|---:|---:|---|
| 10×10 | 270.986 | 173.209 | 1.565× | 5.471 / 5.414 |
| 20×20 | 396.030 | 307.964 | 1.286× | 6.318 / 6.264 |
| 30×30 | 582.648 | 445.983 | 1.306× | 7.169 / 7.013 |
| 40×40 | 808.251 | 658.233 | 1.228× | 8.027 / 7.865 |
