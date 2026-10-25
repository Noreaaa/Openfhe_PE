# Convolution memory optimization — 2026-10-23

**Disposition: rejected as production defaults because of the latency regression. Production now uses direct rotations, scratch budget 0 (maximum channel concurrency), and 2048 slots. Measurements below describe the rejected policies; they are not current defaults.**

This directory records a fresh-process A/B benchmark of sparse Conv+BN. The cancelled 24-case experiment is not restarted.

Changes:
- Share signed power-of-two rotation keys; compose arbitrary convolution rotations and normalize offsets to the logical slot period. Reuse existing keys, including keys already supplied by scheme switching. No evaluation-key generation during forward.
- Limit concurrent output channels using a 8192 MiB experimental estimate for channel accumulators and key-switch temporaries. This is not a total process memory limit; shared caches, ciphertexts, keys and allocation overhead are additional. Global OpenMP configuration remains 96 threads.

`conv_memory_bench baseline` uses direct rotation keys and unrestricted output-channel parallelism. `conv_memory_bench optimized 8192` uses the experimental balanced scratch policy; `conv_memory_bench optimized 2048` measures a stricter policy. Both use ring 32768, 30/40 bits, depth 40, HYBRID digits 6, 2048 logical slots and 96 OpenMP threads. The synthetic convolution uses 8 input and 32 output channels, input 32x32, an encrypted central 8x8 area, a 3x3 kernel and stride 2. Both results are checked against the same plaintext convolution and BN reference, with max absolute error required below 1e-3.

This isolates convolution memory. It excludes CKKS/FHEW scheme-switch setup and ReLU. Its memory reduction cannot be applied as a measured percentage to the previous full-model 53 GiB peak. No new security claim is implied.

Commands (from repository root):

```sh
LD_LIBRARY_PATH=/root/local/openFHE/lib OMP_NUM_THREADS=96 /usr/bin/time -v ./build-sparse/conv_memory_bench baseline
LD_LIBRARY_PATH=/root/local/openFHE/lib OMP_NUM_THREADS=96 /usr/bin/time -v ./build-sparse/conv_memory_bench optimized
```

## Single-trial measurements

| Policy | Rotation keys | Key coefficient storage (GiB) | Concurrent channels | Conv+BN (s) | Process peak RSS (GiB) | Max absolute error |
|---|---:|---:|---:|---:|---:|---:|
| baseline (0.0 MiB scratch) | 53 | 6.987 | 32 | 47.080 | 18.994 | 1.04819e-05 |
| optimized (2048.0 MiB scratch) | 16 | 2.109 | 7 | 253.326 | 6.011 | 8.77249e-06 |
| balanced (8192.0 MiB scratch) | 16 | 2.109 | 30 | 110.614 | 12.056 | 1.03489e-05 |

The experimental balanced policy reduced peak RSS by 36.5%, at 2.35x the Conv+BN time in this single trial. The stricter 2048 MiB policy reduced peak RSS by 68.4%, at 5.38x the time. These are measurements, not repeat means. Global OpenMP maximum is 96; output-channel teams were explicitly capped at 30 and 7 in the two optimized runs. Inner library operations may use the global pool when not inside an active outer team.

The measured balanced policy used `--conv-scratch-mib 8192`; `2048` selected the stricter policy. Both were rejected as defaults. Production now uses `0` and direct rotations. This is an estimate of concurrent channel scratch, not a total RSS guarantee. The logical slot default and the future sweep configuration are now 2048. The old cancelled sweep and its 1024-slot artifacts remain untouched; no full-model experiment was restarted.

For the eight-layer ResNet prefix, enumerating convolution rotation requirements gives 195 direct offsets versus 20 signed power-of-two basis offsets at 2048 slots (155 versus 18 at 1024 slots). This is an enumeration, not a full-model memory measurement. Scheme switching may already provide overlapping keys and also requires additional keys outside this convolution basis.

Validation: sparse convolution correctness and encrypted ReLU/continuous-convolution correctness passed (2/2 CTest cases). All three fresh-process A/B runs passed the plaintext reference check. Main ResNet executable builds successfully. See raw logs, GNU time records and `summary.json` in this directory.
