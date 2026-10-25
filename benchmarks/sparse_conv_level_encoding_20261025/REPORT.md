# Sparse convolution: encode plaintexts at the working level

Only the plaintext encoding levels changed between the frozen executables. Earlier occupied-channel reduction, grouped stride-2 compaction and in-place accumulation are present in both. No change to packing, thread count, convolution weights, ReLU or multiplicative depth.

Filters are encoded at the effective multiplication level (accounting for pending automatic rescaling); selection/compaction masks at the next multiplication level; public values at the destination ciphertext level. The reference path retains its previous default encoding behavior.

## Correctness

The synthetic suite passed with multiple slot sizes, kernels 1/3/7, strides 1/2, partial channel blocks, odd channel counts and chained convolutions. The wide-region fixture additionally uses input level 4 and noise-scale degree 2. Explicit output-level checks passed.

## Matched real-weight observation

Original ResNet convolution 7, input from the same plaintext prefix, original sensitive center 10×10, ring 16384, 8192 slots, depth 40, 30/40-bit moduli, digits 6, 96 threads, input level 24 and noise-scale degree 2. Separate fresh processes; no resident scheme-switch keys. One observation each, not a statistical timing estimate.

| Version | Conv+BN (s) | Output level | Towers | Max absolute error | Whole-process peak (GiB) |
|---|---:|---:|---:|---:|---:|
| before | 35.548 | 26 | 15 | 6.62547e-05 | 16.454 |
| after | 29.140 | 26 | 15 | 5.98636e-05 | 16.477 |

Observed speedup: 1.220×. This single-case result does not establish a full-network speedup. Raw logs include CONV_PROFILE encoding-level fields.
