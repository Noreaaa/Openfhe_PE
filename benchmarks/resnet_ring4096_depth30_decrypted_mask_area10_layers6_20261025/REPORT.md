# Ring 4096 / Depth 30: Decrypted-Sign Mask Baseline

Status: **success**. Finished UTC: 2026-10-25T05:59:44.641236+00:00

## Configuration

Ring dimension 4096; 2048 logical slots; multiplicative depth 30; scaling/first modulus bits 30/40; HYBRID digits 6; 96 OpenMP threads. Initial encrypted center: 10×10 in a 224×224 image. All six original Conv+BN+ReLU stages execute on the server, including conv1 and the convolution replacing max pooling. No client first-layer offload, residual additions, classifier or CKKS bootstrap. One run.

Convolution uses the current optimized SparseConv2dBN_P path (occupied-channel reduction, one-level grouped stride-2 compaction, owned in-place accumulation). ReLU amplifies by 1000, converts CKKS to FHEW, evaluates signs, decrypts those signs with the FHEW secret key, encodes a plaintext positive mask and multiplies the original CKKS ciphertext by that mask. No FHEW-to-CKKS conversion is evaluated in activation. This benchmark explicitly uses the sign-decryption fixture, not the production encrypted activation. Conversion tasks, sign tasks and mask multiplication are parallelized by the fixture; the encrypted-mask comparison processes ciphertext tasks sequentially. Scheme-switch setup/key generation is unchanged for comparability, so resident reverse-conversion keys may remain even though activation does not use them.

An explicit experimental-depth option permits depth 30 below the default conservative minimum 32; no modulus depth is silently increased. The estimated experimental minimum is 27. Per-layer numeric checks determine whether this particular run completes successfully. The run does not reproduce the historical 168-second implementation: convolution, activation and possibly precision/thread settings differ.

## Timing and memory

- Conv+BN total: 42.385 s.
- Decrypted-sign mask ReLU total: 111.522 s.
- Server forward total: 153.907 s (2.565 min).
- Whole-process wall time: 160.500 s.
- Operator-phase sampled RSS mean / peak: 4.755 / 5.565 GiB.
- Whole-process peak RSS: 5.570 GiB.

Operator totals exclude key generation, encryption and verification. Operator-phase RSS is sampled every 0.5 seconds and excludes their time windows, but remains a same-process measure including resident keys and retained allocations. Lifetime peak is separately obtained from GNU time. No network transfer or allocator trimming.

| Original layer | Conv+BN (s) | ReLU (s) | Layer total (s) | Cumulative (s) | Output ciphertexts | Level / remaining towers | Operator RSS mean / peak (GiB) |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 13.288 | 14.585 | 27.872 | 27.872 | 32 | 2 / 29 | 5.252 / 5.565 |
| 2 | 6.897 | 6.205 | 13.103 | 40.975 | 10 | 5 / 26 | 5.059 / 5.224 |
| 3 | 5.074 | 11.570 | 16.643 | 57.619 | 14 | 8 / 23 | 4.693 / 4.845 |
| 4 | 5.440 | 17.746 | 23.187 | 80.805 | 18 | 11 / 20 | 4.619 / 4.733 |
| 5 | 6.095 | 26.280 | 32.374 | 113.180 | 22 | 14 / 17 | 4.591 / 4.715 |
| 6 | 5.591 | 35.136 | 40.727 | 153.907 | 26 | 17 / 14 | 4.563 / 4.698 |

## Numeric checks

Structure: True; packing geometry: True; numerical check thresholds: True. Maximum absolute feature error: 0.00048578; maximum RMSE / max(1, reference maximum absolute value): 1.33267e-06. This is a feature consistency check, not classifier accuracy.

Raw records: `area10/{layers.csv,run.log,rss.csv,time.txt,result.json}`. Binary and source snapshots are retained.

## Comparison with ring dimension 2048 / 1024 slots

| Metric | Ring 2048 / slots 1024 | Ring 4096 / slots 2048 |
|---|---:|---:|
| Server forward (s) | 132.291 | 153.907 |
| Conv+BN (s) | 26.261 | 42.385 |
| ReLU (s) | 106.031 | 111.522 |
| Operator mean RSS (GiB) | 2.237 | 4.755 |
| Whole-process peak (GiB) | 2.729 | 5.570 |

Depth, modulus bit sizes, convolution code, decrypted-sign ReLU, thread count, image and original sensitive area are held fixed. Ring dimension and logical slot count both double; ciphertext block counts and scheme-switch conversion sizes may therefore change. This is a ring-plus-packing comparison, not an isolated ring-dimension microbenchmark.
