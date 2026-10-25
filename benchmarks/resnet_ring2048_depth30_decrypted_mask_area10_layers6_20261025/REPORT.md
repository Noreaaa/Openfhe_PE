# Ring 2048 / Depth 30: Decrypted-Sign Mask Baseline

Status: **success**. Finished UTC: 2026-10-25T05:48:14.182839+00:00

## Configuration

Ring dimension 2048; 1024 logical slots; multiplicative depth 30; scaling/first modulus bits 30/40; HYBRID digits 6; 96 OpenMP threads. Initial encrypted center: 10×10 in a 224×224 image. All six original Conv+BN+ReLU stages execute on the server, including conv1 and the convolution replacing max pooling. No client first-layer offload, residual additions, classifier or CKKS bootstrap. One run.

Convolution uses the current optimized SparseConv2dBN_P path (occupied-channel reduction, one-level grouped stride-2 compaction, owned in-place accumulation). ReLU amplifies by 1000, converts CKKS to FHEW, evaluates signs, decrypts those signs with the FHEW secret key, encodes a plaintext positive mask and multiplies the original CKKS ciphertext by that mask. No FHEW-to-CKKS conversion is evaluated in activation. This benchmark explicitly uses the sign-decryption fixture, not the production encrypted activation. Conversion tasks, sign tasks and mask multiplication are parallelized by the fixture; the encrypted-mask comparison processes ciphertext tasks sequentially. Scheme-switch setup/key generation is unchanged for comparability, so resident reverse-conversion keys may remain even though activation does not use them.

An explicit experimental-depth option permits depth 30 below the default conservative minimum 32; no modulus depth is silently increased. The estimated experimental minimum is 27. Per-layer numeric checks determine whether this particular run completes successfully. The run does not reproduce the historical 168-second implementation: convolution, activation and possibly precision/thread settings differ.

## Timing and memory

- Conv+BN total: 26.261 s.
- Decrypted-sign mask ReLU total: 106.031 s.
- Server forward total: 132.291 s (2.205 min).
- Whole-process wall time: 136.780 s.
- Operator-phase sampled RSS mean / peak: 2.237 / 2.719 GiB.
- Whole-process peak RSS: 2.729 GiB.

Operator totals exclude key generation, encryption and verification. Operator-phase RSS is sampled every 0.5 seconds and excludes their time windows, but remains a same-process measure including resident keys and retained allocations. Lifetime peak is separately obtained from GNU time. No network transfer or allocator trimming.

| Original layer | Conv+BN (s) | ReLU (s) | Layer total (s) | Cumulative (s) | Output ciphertexts | Level / remaining towers | Operator RSS mean / peak (GiB) |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 6.492 | 14.016 | 20.509 | 20.509 | 64 | 2 / 29 | 2.397 / 2.502 |
| 2 | 5.235 | 5.711 | 10.946 | 31.455 | 20 | 5 / 26 | 2.538 / 2.719 |
| 3 | 3.029 | 10.500 | 13.529 | 44.983 | 28 | 8 / 23 | 2.190 / 2.255 |
| 4 | 3.480 | 16.890 | 20.370 | 65.354 | 36 | 11 / 20 | 2.180 / 2.237 |
| 5 | 4.206 | 24.519 | 28.725 | 94.079 | 44 | 14 / 17 | 2.157 / 2.253 |
| 6 | 3.819 | 34.394 | 38.213 | 132.291 | 52 | 17 / 14 | 2.179 / 2.193 |

## Numeric checks

Structure: True; packing geometry: True; numerical check thresholds: True. Maximum absolute feature error: 0.000513068; maximum RMSE / max(1, reference maximum absolute value): 1.34804e-06. This is a feature consistency check, not classifier accuracy.

Raw records: `area10/{layers.csv,run.log,rss.csv,time.txt,result.json}`. Binary and source snapshots are retained.

## Comparison with encrypted-mask ReLU

| Metric | Encrypted mask | Decrypted-sign plaintext mask |
|---|---:|---:|
| Server forward (s) | 289.379 | 132.291 |
| Conv+BN (s) | 17.527 | 26.261 |
| ReLU (s) | 271.853 | 106.031 |
| Operator mean RSS (GiB) | 1.888 | 2.237 |
| Whole-process peak (GiB) | 2.504 | 2.729 |

The convolution implementation and initial parameters are identical, but later input levels differ because the activation paths consume different depths. Activation parallel scheduling also differs. This comparison therefore measures two pipeline variants rather than isolating reverse-conversion cost alone.
