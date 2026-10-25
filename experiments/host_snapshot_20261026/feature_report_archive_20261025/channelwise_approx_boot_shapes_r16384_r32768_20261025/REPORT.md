# Channel-Wise Approximation + Parallel CKKS Bootstrapping: Four Shapes

Updated UTC: 2026-10-25T12:59:10.916945+00:00

## Configuration

Fully encrypted channel-wise inputs; one contiguous H×W feature map per ciphertext. Representative layers 3, 8, 12, 16 cover 64×56×56, 128×28×28, 256×14×14, 512×7×7, respectively. All are same-channel-count 3×3, stride-1, padding-1 Conv+BN operators. Ring dimensions 16384 and 32768; slots N/2; depth 30; scaling/first moduli 59/60 bits; HYBRID digits 6; 96 OpenMP threads. One independent process per case, with fresh level-0 input.

Each measured process includes CKKS context creation, bootstrap precomputation, key generation (including bootstrap and spatial rotation keys), full input encryption, Conv+BN, normalized degree-6 polynomial activation, genuine CKKS bootstrapping and numerical verification. No scheme-switch keys or FHEW operations are included, because this is the approximation method only. Inputs are prepared by the same sequential plaintext prefix without residual additions as the shape-only benchmark; these are independent shape pipelines, not an 18-layer encrypted execution.

The polynomial is p(z)=0.0229645+0.5z+1.4340124z²−2.089185z⁴+1.1551725z⁶. Conv+BN includes normalization R=max(1,1.1 max|plaintext preactivation|) and activation returns R p(x/R). Bootstrap inputs are divided by B=max(1,1.1 max|plaintext polynomial output|), explicitly rescaled and depleted to two towers, bootstrapped, then restored by B. Assertions require two towers before refresh and more than two afterwards. Bounds are calibrated on this single image outside operator timers. Feature arithmetic error and approximation-versus-ReLU error are distinct; this is not a classifier accuracy evaluation.

## Concurrency and comparison scope

Polynomial evaluation and bootstrap both use the historical outer OpenMP channel-parallel policy, up to 96 simultaneous ciphertext tasks. This deliberately differs from the previous sequential-bootstrap experiment. Nested teams are disabled by the process defaults. Large temporary working sets are expected; the monitor stops a case if host MemAvailable falls below 12 GiB. It does not silently reduce threads or ring dimension.

This is an adapted channel-wise baseline with cached rotations, padding and folded BN, not an unmodified historical executable. The 59/60-bit precision was chosen after genuine bootstrap probes failed with lower precision. Therefore these timings cannot be attributed to adding bootstrap alone versus the earlier 30/40-bit shape-only runs. Every case starts in a fresh process, so allocations from previous network layers are not reproduced.

## Time and memory

All phase timings are wall seconds. Operator total = convolution + polynomial + bootstrap. Setup+compute includes key setup and encryption as well; process wall additionally includes data preparation, calibration, numerical verification and other overhead. Current RSS is sampled every 0.5 seconds for all phases. Operator RSS includes resident keys and allocator retention; whole-process peak includes setup and verification too. No network measurement or allocator trimming.

| Group | Layer | Ring / slots | Status | Key setup (s) | Encrypt (s) | Conv+BN (s) | Polynomial (s) | Bootstrap (s) | Operator total (s) | Setup+compute (s) | Process wall (s) | Whole-process peak (GiB) |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–6 | 3 | 16384 / 8192 | success | 6.054 | 0.468 | 68.975 | 3.000 | 42.424 | 114.399 | 120.920 | 126.730 | 21.886 |
| 7–10 | 8 | 16384 / 8192 | success | 6.103 | 1.354 | 269.968 | 4.968 | 78.351 | 353.286 | 360.744 | 371.830 | 28.070 |
| 11–14 | 12 | 16384 / 8192 | cancelled (incomplete) | — | — | — | — | — | — | — | — | — |
| 15–18 | 16 | 16384 / 8192 | not run (cancelled) | — | — | — | — | — | — | — | — | — |
| 1–6 | 3 | 32768 / 16384 | not run (cancelled) | — | — | — | — | — | — | — | — | — |
| 7–10 | 8 | 32768 / 16384 | not run (cancelled) | — | — | — | — | — | — | — | — | — |
| 11–14 | 12 | 32768 / 16384 | not run (cancelled) | — | — | — | — | — | — | — | — | — |
| 15–18 | 16 | 32768 / 16384 | not run (cancelled) | — | — | — | — | — | — | — | — | — |

## Phase RSS and numerical checks

| Layer | Ring | Key setup max (GiB) | Conv max (GiB) | Polynomial max (GiB) | Bootstrap max (GiB) | Operator mean / max (GiB) | Post-bootstrap RMSE | vs ReLU RMSE |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 3 | 16384 | 7.828 | 12.226 | 12.636 | 21.519 | 14.675 / 21.519 | 8.14022e-05 | 0.0706266 |
| 8 | 16384 | 8.126 | 16.752 | 18.360 | 28.013 | 18.195 / 28.013 | 9.47297e-05 | 0.0322999 |

A preliminary 8-output-channel parallel-bootstrap probe is retained under `validation/probe8`; it is not included in the four-shape table. Successful cases must pass convolution, polynomial and post-bootstrap arithmetic checks. Incomplete or failed cases are not valid complete performance measurements. Raw logs, per-phase RSS, numerical metrics, binary and source snapshots are retained.

## Cancelled by user

The experiment was cancelled before Git upload. Successful completed cases are retained. The interrupted case is not a completed performance result. No automatic restart is scheduled. See cancellation.json for the interrupted phase.
