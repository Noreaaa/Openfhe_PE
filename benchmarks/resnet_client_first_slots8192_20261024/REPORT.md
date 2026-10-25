# Client first layer + 8192 slots

Updated UTC: 2026-10-24T07:55:22.198092+00:00

Parameters: ring 16384, 30/40-bit moduli, depth 40, HYBRID digits 6, 8192 logical slots, 96 OpenMP threads. Direct rotations; no channel concurrency limit; no CKKS bootstrap. Original convolution 1 (Conv+BN+ReLU) runs in plaintext on the simulated client. Original convolutions 2..8 run on mixed plaintext/ciphertext features. The prefix still has no residual additions or classifier.

Client, encryption and server timings are measured separately on the same machine. No network transfer is measured. Payload is ciphertext coefficient storage, not serialized wire size. The original sensitive rectangle is propagated by receptive-field overlap before encryption. Rows retain original full-width coordinates.

RSS is sampled every 0.5 seconds. Server samples include only explicitly marked Conv/ReLU execution, excluding key generation, client processing and reference verification time windows. These remain same-process measurements: resident keys, allocator retention and reference tensors are not isolated into a separate server process. Reported sampled peak is distinct from lifetime whole-process peak. No allocator trimming or peak reset is applied.

The combined client-offload + slot-count change is compared against the completed 2048-slot all-server baseline. This is not a factorial ablation: improvements cannot be attributed to just one of the two changes.

| Area | Status | Client conv1 (s) | Encrypt (s) | Server forward (min) | Client+encrypt+forward (min) | Baseline forward (min) | Server sampled mean/max (GiB) | Whole-process peak (GiB) | Upload ciphertexts |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| 10×10 | success | 0.119 | 0.463 | 14.951 | 14.961 | 19.682 | 35.080 / 40.536 | 40.698 | 8 |
| 20×20 | success | 0.119 | 0.737 | 19.257 | 19.272 | 27.104 | 35.418 / 41.864 | 41.951 | 13 |
| 30×30 | success | 0.119 | 1.023 | 24.299 | 24.318 | 36.704 | 36.200 / 44.357 | 44.428 | 18 |
| 40×40 | success | 0.119 | 1.331 | 31.234 | 31.259 | 48.092 | 36.852 / 45.949 | 46.063 | 23 |
| 50×50 | success | 0.119 | 1.603 | 36.378 | 36.406 | 57.707 | 37.462 / 49.956 | 50.012 | 28 |
| 60×60 | success | 0.118 | 1.872 | 40.596 | 40.629 | 67.818 | 38.246 / 49.080 | 49.202 | 33 |
| 70×70 | success | 0.117 | 2.156 | 48.827 | 48.865 | 81.085 | 39.034 / 51.686 | 51.779 | 38 |
| 80×80 | success | 0.119 | 2.438 | 57.919 | 57.961 | 97.498 | 39.793 / 54.547 | 54.593 | 43 |

## Server layer measurements

| Area | Original layer | Conv+BN (s) | ReLU (s) | Cumulative forward (s) | Output ciphertexts | Level |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2 | 41.329 | 51.408 | 92.737 | 5 | 12 |
| 10 | 3 | 25.838 | 70.121 | 188.696 | 7 | 15 |
| 10 | 4 | 30.072 | 93.305 | 312.072 | 9 | 18 |
| 10 | 5 | 32.079 | 122.204 | 466.355 | 11 | 21 |
| 10 | 6 | 30.562 | 145.357 | 642.275 | 13 | 24 |
| 10 | 7 | 40.618 | 79.556 | 762.449 | 7 | 27 |
| 10 | 8 | 25.842 | 108.783 | 897.075 | 9 | 30 |
| 20 | 2 | 73.562 | 70.286 | 143.848 | 7 | 12 |
| 20 | 3 | 33.593 | 89.931 | 267.371 | 9 | 15 |
| 20 | 4 | 37.294 | 112.862 | 417.527 | 11 | 18 |
| 20 | 5 | 38.939 | 150.051 | 606.517 | 13 | 21 |
| 20 | 6 | 35.629 | 170.996 | 813.141 | 15 | 24 |
| 20 | 7 | 56.977 | 109.991 | 980.109 | 9 | 27 |
| 20 | 8 | 31.018 | 144.317 | 1155.444 | 11 | 30 |
| 30 | 2 | 106.489 | 104.637 | 211.126 | 10 | 12 |
| 30 | 3 | 42.667 | 127.187 | 380.980 | 12 | 15 |
| 30 | 4 | 46.845 | 154.043 | 581.868 | 14 | 18 |
| 30 | 5 | 44.940 | 183.299 | 810.106 | 16 | 21 |
| 30 | 6 | 42.492 | 211.908 | 1064.506 | 18 | 24 |
| 30 | 7 | 68.012 | 126.617 | 1259.136 | 10 | 27 |
| 30 | 8 | 34.051 | 164.752 | 1457.939 | 12 | 30 |
| 40 | 2 | 170.981 | 142.139 | 313.119 | 13 | 12 |
| 40 | 3 | 55.709 | 173.788 | 542.616 | 15 | 15 |
| 40 | 4 | 68.500 | 203.514 | 814.630 | 17 | 18 |
| 40 | 5 | 59.861 | 236.740 | 1111.231 | 19 | 21 |
| 40 | 6 | 47.787 | 266.480 | 1425.498 | 21 | 24 |
| 40 | 7 | 79.814 | 147.816 | 1653.129 | 11 | 27 |
| 40 | 8 | 37.189 | 183.743 | 1874.061 | 13 | 30 |
| 50 | 2 | 203.011 | 169.322 | 372.334 | 15 | 12 |
| 50 | 3 | 64.080 | 197.224 | 633.638 | 17 | 15 |
| 50 | 4 | 65.327 | 233.455 | 932.420 | 19 | 18 |
| 50 | 5 | 65.311 | 266.388 | 1264.119 | 21 | 21 |
| 50 | 6 | 57.497 | 305.604 | 1627.219 | 23 | 24 |
| 50 | 7 | 110.532 | 179.137 | 1916.888 | 13 | 27 |
| 50 | 8 | 44.200 | 221.567 | 2182.655 | 15 | 30 |
| 60 | 2 | 245.335 | 200.076 | 445.411 | 17 | 12 |
| 60 | 3 | 68.295 | 241.092 | 754.799 | 19 | 15 |
| 60 | 4 | 73.046 | 267.530 | 1095.375 | 21 | 18 |
| 60 | 5 | 68.303 | 309.978 | 1473.656 | 23 | 21 |
| 60 | 6 | 61.072 | 341.810 | 1876.537 | 25 | 24 |
| 60 | 7 | 109.726 | 179.137 | 2165.401 | 13 | 27 |
| 60 | 8 | 44.135 | 226.207 | 2435.743 | 15 | 30 |
| 70 | 2 | 319.369 | 254.104 | 573.473 | 20 | 12 |
| 70 | 3 | 79.533 | 287.698 | 940.704 | 22 | 15 |
| 70 | 4 | 81.379 | 321.083 | 1343.166 | 24 | 18 |
| 70 | 5 | 74.515 | 359.844 | 1777.525 | 26 | 21 |
| 70 | 6 | 67.668 | 404.000 | 2249.194 | 28 | 24 |
| 70 | 7 | 139.313 | 219.485 | 2607.992 | 15 | 27 |
| 70 | 8 | 53.089 | 268.548 | 2929.628 | 17 | 30 |
| 80 | 2 | 403.410 | 297.576 | 700.985 | 23 | 12 |
| 80 | 3 | 94.438 | 344.026 | 1139.449 | 25 | 15 |
| 80 | 4 | 106.541 | 375.437 | 1621.428 | 27 | 18 |
| 80 | 5 | 92.664 | 413.242 | 2127.334 | 29 | 21 |
| 80 | 6 | 74.237 | 454.818 | 2656.389 | 31 | 24 |
| 80 | 7 | 169.805 | 265.166 | 3091.359 | 17 | 27 |
| 80 | 8 | 64.496 | 319.265 | 3475.120 | 19 | 30 |
