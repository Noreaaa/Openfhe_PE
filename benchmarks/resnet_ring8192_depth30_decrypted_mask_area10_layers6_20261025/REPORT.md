# Ring 8192 / Depth 30: Decrypted-Sign Mask Baseline

Status: **success**. Finished UTC: 2026-10-25T06:12:04.331728+00:00

## Configuration

Ring dimension 8192; 4096 logical slots; multiplicative depth 30; scaling/first modulus bits 30/40; HYBRID digits 6; 96 OpenMP threads. Initial encrypted center: 10×10 in a 224×224 image. All six original Conv+BN+ReLU stages execute on the server, including conv1 and the convolution replacing max pooling. No client first-layer offload, residual additions, classifier or CKKS bootstrap. One run.

Convolution uses the current optimized SparseConv2dBN_P path (occupied-channel reduction, one-level grouped stride-2 compaction, owned in-place accumulation). ReLU amplifies by 1000, converts CKKS to FHEW, evaluates signs, decrypts those signs with the FHEW secret key, encodes a plaintext positive mask and multiplies the original CKKS ciphertext by that mask. No FHEW-to-CKKS conversion is evaluated in activation. This benchmark explicitly uses the sign-decryption fixture, not the production encrypted activation. Conversion tasks, sign tasks and mask multiplication are parallelized by the fixture; the encrypted-mask comparison processes ciphertext tasks sequentially. Scheme-switch setup/key generation is unchanged for comparability, so resident reverse-conversion keys may remain even though activation does not use them.

An explicit experimental-depth option permits depth 30 below the default conservative minimum 32; no modulus depth is silently increased. The estimated experimental minimum is 27. Per-layer numeric checks determine whether this particular run completes successfully. The run does not reproduce the historical 168-second implementation: convolution, activation and possibly precision/thread settings differ.

## Timing and memory

- Conv+BN total: 87.869 s.
- Decrypted-sign mask ReLU total: 124.940 s.
- Server forward total: 212.809 s (3.547 min).
- Whole-process wall time: 223.790 s.
- Operator-phase sampled RSS mean / peak: 11.660 / 13.663 GiB.
- Whole-process peak RSS: 13.670 GiB.

Operator totals exclude key generation, encryption and verification. Operator-phase RSS is sampled every 0.5 seconds and excludes their time windows, but remains a same-process measure including resident keys and retained allocations. Lifetime peak is separately obtained from GNU time. No network transfer or allocator trimming.

| Original layer | Conv+BN (s) | ReLU (s) | Layer total (s) | Cumulative (s) | Output ciphertexts | Level / remaining towers | Operator RSS mean / peak (GiB) |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 37.198 | 18.077 | 55.274 | 55.274 | 16 | 2 / 29 | 12.284 / 13.663 |
| 2 | 12.255 | 8.102 | 20.358 | 75.632 | 5 | 5 / 26 | 11.849 / 12.189 |
| 3 | 9.310 | 13.028 | 22.337 | 97.970 | 7 | 8 / 23 | 11.379 / 11.687 |
| 4 | 10.229 | 19.882 | 30.112 | 128.081 | 9 | 11 / 20 | 11.332 / 11.656 |
| 5 | 9.949 | 28.852 | 38.802 | 166.883 | 11 | 14 / 17 | 11.465 / 11.742 |
| 6 | 8.928 | 36.999 | 45.926 | 212.809 | 13 | 17 / 14 | 11.333 / 11.783 |

## Numeric checks

Structure: True; packing geometry: True; numerical check thresholds: True. Maximum absolute feature error: 0.000486106; maximum RMSE / max(1, reference maximum absolute value): 1.3746e-06. This is a feature consistency check, not classifier accuracy.

Raw records: `area10/{layers.csv,run.log,rss.csv,time.txt,result.json}`. Binary and source snapshots are retained.

## Comparison with ring dimension 2048 / 1024 slots

| Metric | Ring 2048 / slots 1024 | Ring 8192 / slots 4096 |
|---|---:|---:|
| Server forward (s) | 132.291 | 212.809 |
| Conv+BN (s) | 26.261 | 87.869 |
| ReLU (s) | 106.031 | 124.940 |
| Operator mean RSS (GiB) | 2.237 | 11.660 |
| Whole-process peak (GiB) | 2.729 | 13.670 |

Depth, modulus bit sizes, convolution code, decrypted-sign ReLU, thread count, image and original sensitive area are held fixed. Compared with the ring-2048 baseline, ring dimension and logical slot count both increase fourfold; ciphertext block counts and scheme-switch conversion sizes may therefore change. This is a ring-plus-packing comparison, not an isolated ring-dimension microbenchmark.

## Three-ring comparison

All cases use a 10×10 original sensitive center, original layers 1–6, depth 30, 30/40-bit moduli, digits 6, 96 threads and the decrypted-sign mask fixture. Logical slots equal half the ring dimension. One run per configuration.

| Ring | Slots | Conv+BN (s) | ReLU (s) | Total server (s) | Operator mean RSS (GiB) | Whole-process peak (GiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 1024 | 26.261 | 106.031 | 132.291 | 2.237 | 2.729 |
| 4096 | 2048 | 42.385 | 111.522 | 153.907 | 4.755 | 5.570 |
| 8192 | 4096 | 87.869 | 124.940 | 212.809 | 11.660 | 13.670 |
