# Ring 2048 / Depth 30: Six-Stage Partially Encrypted ResNet Prefix

Status: **success**. Finished UTC: 2026-10-25T05:41:08.166859+00:00

## Configuration

Ring dimension 2048; 1024 logical slots; multiplicative depth 30; scaling/first modulus bits 30/40; HYBRID digits 6; 96 OpenMP threads. Initial encrypted center: 10×10 in a 224×224 image. All six original Conv+BN+ReLU stages execute on the server, including conv1 and the convolution replacing max pooling. No client first-layer offload, residual additions, classifier or CKKS bootstrap. One run.

Convolution uses the current optimized SparseConv2dBN_P path (occupied-channel reduction, one-level grouped stride-2 compaction, owned in-place accumulation). ReLU amplifies by 1000, converts CKKS to FHEW, evaluates encrypted signs, converts back to CKKS and multiplies the original ciphertext by the encrypted positive mask. Activation evaluation does not decrypt the sign. Reference decryption is limited to verification outside operator timers.

An explicit experimental-depth option permits depth 30 below the default conservative minimum 32; no modulus depth is silently increased. The estimated experimental minimum is 27. Per-layer numeric checks determine whether this particular run completes successfully. The run does not reproduce the historical 168-second implementation: convolution, activation and possibly precision/thread settings differ.

## Timing and memory

- Conv+BN total: 17.527 s.
- Encrypted ReLU total: 271.853 s.
- Server forward total: 289.379 s (4.823 min).
- Whole-process wall time: 293.620 s.
- Operator-phase sampled RSS mean / peak: 1.888 / 2.496 GiB.
- Whole-process peak RSS: 2.504 GiB.

Operator totals exclude key generation, encryption and verification. Operator-phase RSS is sampled every 0.5 seconds and excludes their time windows, but remains a same-process measure including resident keys and retained allocations. Lifetime peak is separately obtained from GNU time. No network transfer or allocator trimming.

| Original layer | Conv+BN (s) | ReLU (s) | Layer total (s) | Cumulative (s) | Output ciphertexts | Level / remaining towers | Operator RSS mean / peak (GiB) |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 6.045 | 56.981 | 63.026 | 63.026 | 64 | 12 / 19 | 2.357 / 2.496 |
| 2 | 3.670 | 18.269 | 21.939 | 84.965 | 20 | 15 / 16 | 1.818 / 2.047 |
| 3 | 1.974 | 29.380 | 31.353 | 116.318 | 28 | 18 / 13 | 1.768 / 1.878 |
| 4 | 2.107 | 40.017 | 42.125 | 158.443 | 36 | 21 / 10 | 1.748 / 1.837 |
| 5 | 1.903 | 56.540 | 58.443 | 216.886 | 44 | 24 / 7 | 1.749 / 1.789 |
| 6 | 1.828 | 70.665 | 72.493 | 289.379 | 52 | 27 / 4 | 1.740 / 1.752 |

## Numeric checks

Structure: True; packing geometry: True; numerical check thresholds: True. Maximum absolute feature error: 0.0085284; maximum RMSE / max(1, reference maximum absolute value): 1.09451e-05. This is a feature consistency check, not classifier accuracy.

Raw records: `area10/{layers.csv,run.log,rss.csv,time.txt,result.json}`. Binary and source snapshots are retained.
