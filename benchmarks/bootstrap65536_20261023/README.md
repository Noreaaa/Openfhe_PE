# OpenFHE 1.2.4: bootstrap at ring dimension 65536

This probe establishes runtime support, not full CNN/CKKS–FHEW correctness or end-to-end security. Each context explicitly requests `HEStd_128_classic`, UNIFORM_TERNARY, HYBRID with three large digits, FLEXIBLEAUTO, 1024 slots, and bootstrap level budgets {3,3}. The input has 1024 deterministic values in [-0.2,0.2] and starts with two towers. OpenMP is configured for 96 threads. Decryption checks all output values and reports maximum absolute error; a successful process does not imply that the error meets a specific application tolerance.

| Scale / first modulus | Total depth | Correction factor | Outcome | Bootstrap time | Maximum absolute error |
|---|---:|---:|---|---:|---:|
| 30 / 40 | 30 | automatic (8) | EvalBootstrap rejects degree 10 > correction factor 8 | — | — |
| 30 / 40 | 30 | 10 | Executes; output level 19, 12 towers | 11.409 s | 0.221395 |
| 40 / 50 | 30 | 10 | Bootstrap returns, but decoding rejects excessive approximation error | — | — |
| 50 / 60 | 24 | 10 | Executes; output level 19, 6 towers | 8.829 s | 0.0236744 |
| 50 / 51 | 24 | automatic (8) | Executes; output level 19, 6 towers | 8.431 s | 0.00655743 |

The final case has noise scale degree 2, so the effective remaining level count is `24 - 19 - (2-1) = 4`. Its process peak RSS is approximately 16.57 GiB, including setup and keys. These samples are not an optimized precision study and have no repeated timing trials. They show that ring 65536 is supported, while low-bit parameter choices still need precision tuning. In particular, the numerical failures cannot be interpreted as lack of ring-dimension support, and none of these tests establishes a six-layer encrypted CNN result.

The application-level check in `tests/sparse_packing/resnet_conv3_bench.cpp` accepts only ring 2048 or 16384. That restriction belongs to the local benchmark, not the OpenFHE library. The main application's parameter setup does not have that restriction.

The default correction-factor heuristic in OpenFHE 1.2.4 can choose a value below `firstModSize - scalingModSize` for our settings. Source `EvalBootstrap` explicitly rejects that case. The application now supplies at least that difference (and at least 10) to avoid this specific configuration error; this is not a guarantee of acceptable precision.

`probe.cpp` accepts `[scale_bits=30] [total_depth=30] [first_bits=scale+10] [correction_factor=10]`. A correction argument of zero selects the OpenFHE heuristic. Raw logs and GNU time output are retained. `run.log`/`run.time` duplicate the 30/40, correction=10 attempt; the named `scale*.log` files distinguish all reported cases. No FHEW keys or scheme switching are used in this probe.
