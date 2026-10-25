# Channel-Wise Polynomial Approximation with CKKS Bootstrapping

Updated UTC: 2026-10-25T05:10:22.745816+00:00

## Configuration

Same image, weights, channel-wise spatial layout, folded BN and normalized degree-6 polynomial as the completed no-bootstrap approximation baseline. Client executes original Conv1+BN+ReLU. Server features are fully encrypted; original convolutions 2–8 execute once, without residual additions or classifier. Ring dimension 32768; 16384 slots; one ciphertext per channel; 96 OpenMP threads; HYBRID numLargeDigits=6.

Scaling/first modulus bits: 59/60. Multiplicative depth: 30; level budget {3,3}. OpenFHE's GetBootstrapDepth returns 20 reserved levels for UNIFORM_TERNARY; initial depth 30 reserves ten additional levels. These settings do not establish 128-bit security.

A genuine CKKS bootstrap is performed after every activation, including the final stage (64 ciphertexts for each of stages 2–6, 128 for each of stages 7–8: 576 refreshes in total). Each ciphertext is divided by a plaintext-calibrated output bound, its scale degree is reduced to one, and the modulus chain is explicitly reduced to two towers. After EvalBootstrap it is scaled back. Assertions require two towers before bootstrap and more than two afterwards; a skipped/no-op refresh cannot silently pass. The explicit internal level-reduction API is used because the public LevelReduce is a no-op with FLEXIBLEAUTO in this installed OpenFHE version. Normalization, depletion and restoration costs are included in bootstrap time. Ciphertexts are refreshed sequentially, with OpenFHE using the thread pool internally.

The original polynomial is p(z)=0.0229645+0.5z+1.4340124z²−2.089185z⁴+1.1551725z⁶. Per-layer convolution normalization follows R=max(1,1.1×max|plaintext preactivation|), and returns R×p(x/R). Bootstrapping normalization is also calibrated from the matching plaintext polynomial trajectory. Calibration is specific to this benchmark image and is performed outside server timers; it is not a dataset-wide bound.

The installed OpenFHE example uses 59/60-bit moduli for 64-bit bootstrapping. Preliminary genuine-refresh probes with 30/40, 40/50 and 40/41 failed numerical checks; their logs are retained. The selected 59/60-bit two-refresh probe passed with RMSE 8.00e-5 and 1.36e-4, restoring 11 towers each time. The early initial-depletion-check probe is invalid for performance because its public LevelReduce call did not force genuine refresh.

## Measurement and validation

Conv+BN, polynomial evaluation and bootstrap wall times are separate. Per-layer cumulative time includes all three. Key generation, client compute/encryption, plaintext reference construction and verification are excluded from operator totals. RSS is sampled every 0.5 s, including server_bootstrap phases. Operator-phase RSS still includes resident keys and retained allocations from this shared process; lifetime whole-process peak is reported separately. No network measurement or allocator trimming. Each layer is checked against the plaintext polynomial path after refreshing; polynomial-versus-ReLU error is separate from HE arithmetic error and is not classifier accuracy.

## Summary

| Run | Status | Conv+BN (min) | Polynomial (min) | Bootstrap (min) | Total server (min) | Operator RSS mean / peak (GiB) | Lifetime peak (GiB) |
|---|---|---:|---:|---:|---:|---|---:|
| No bootstrap, depth 36 | completed | 13.916 | 0.456 | 0 | 14.372 | 15.010 / 19.488 | 19.898 |
| Bootstrap after each stage | cancelled_by_user (4/7 stages) | 4.207 | 0.170 | 46.015 | 50.392 | 21.364 / 30.338 | 30.563 |

Changing modulus bit sizes (30/40 to 59/60), initial depth and keeping ciphertexts refreshed changes convolution cost; this is a pipeline comparison rather than a matched-level convolution microbenchmark. Running totals cover only completed stages. Until completion, final memory summaries are pending.

## Per-layer results

| Layer | C×H×W | Conv+BN (s) | Polynomial (s) | Bootstrap (s) | Cumulative (s) | Level before / after refresh | Remaining towers | Operator RSS mean / peak (GiB) | Post-refresh arithmetic RMSE | vs ReLU RMSE |
|---:|---|---:|---:|---:|---:|---|---:|---|---:|---:|
| 2 | 64×56×56 | 92.943 | 6.037 | 685.077 | 784.058 | 4 / 20 | 11 | 26.028 / 30.338 | 0.000943119 | 0.170995 |
| 3 | 64×56×56 | 51.913 | 1.409 | 689.616 | 1526.996 | 25 / 20 | 11 | 20.079 / 20.786 | 0.000286426 | 0.0713938 |
| 4 | 64×56×56 | 53.685 | 1.382 | 691.389 | 2273.451 | 25 / 20 | 11 | 20.052 / 20.314 | 0.000552245 | 0.063465 |
| 5 | 64×56×56 | 53.885 | 1.360 | 694.825 | 3023.522 | 25 / 20 | 11 | 20.048 / 20.334 | 0.000241696 | 0.0261847 |

Raw data: `approximation_bootstrap/{layers.csv,rss.csv,run.log,time.txt,result.json}`. The executable, source snapshot and hashes are retained in this directory. The two-refresh parameter probe is archived separately.

## Cancellation

This run was cancelled at the user's request because the polynomial approximation error was unacceptable. Completed layers and all sampled logs are retained. The interrupted stage is not a completed measurement. No automatic restart is scheduled.
