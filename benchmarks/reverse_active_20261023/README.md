# Active values and FHEW-to-CKKS latency — 2026-10-23

At the current fixed transform size, increasing active values from 6 to 1024 does not materially increase reverse-switch latency. The median changes by -1.03%; sign evaluation increases 18.79 times.

| Active values | Sign median (ms) | Reverse median (ms) | Reverse min–max (ms) |
|---:|---:|---:|---:|
| 6 | 175.94 | 1239.55 | 1223.62–1256.25 |
| 64 | 313.99 | 1226.49 | 1216.99–1252.09 |
| 256 | 946.99 | 1230.14 | 1224.34–1272.97 |
| 1024 | 3305.70 | 1226.78 | 1217.84–1287.15 |

All times are for one CKKS output ciphertext, rather than the two-ciphertext workload in the previous report. Fixed parameters: OpenFHE 1.2.4, ring 16384, logical slots 1024, depth 20, scale/first modulus 50/60 bits, 96 OpenMP threads, no nested teams. Each case has one warmup followed by five measured trials; case order alternates ascending/descending. The same context, keys, and input FHEW ciphertexts are reused. Forward switching is outside the measured region.

The production `EncryptedReluMask` evaluates signs only at active positions, then fills inactive entries by reusing the first encrypted sign. Its `EvalFHEWtoCKKS(signs, slots, slots, 4, -1, 1, 0)` call always receives 1024 FHEW ciphertexts and requests 1024 output values. Thus increasing the active count changes the number of sign evaluations and the data in the matrix, but does not change the reverse transform dimensions. This experiment varies active positions, not the API `numCtxts` parameter. It does not establish the cost of a genuinely smaller reverse transform.

All 1024 output mask slots are checked after each conversion, outside the timers. Inactive masks are expected to equal the filler mask. The maximum absolute error, including warmups, was 1.48e-5, below the 1e-3 threshold. No production code was changed for this experiment.

Raw measurements: `run.log`, GNU time: `run.time`, structured summary: `summary.json`. The time file reports the entire mixed-case process; its peak RSS is not an independent per-active-count memory comparison.

## Reproduction

```bash
cmake -S tests/sparse_packing -B build-sparse -DCMAKE_BUILD_TYPE=Release
cmake --build build-sparse --target reverse_active_bench -j "$(nproc)"
OMP_NUM_THREADS="$(nproc)" OMP_THREAD_LIMIT="$(nproc)" \
  OMP_DYNAMIC=FALSE OMP_MAX_ACTIVE_LEVELS=1 \
  LD_LIBRARY_PATH="$HOME/local/openFHE/lib:$HOME/local/cnpy/lib:${LD_LIBRARY_PATH:-}" \
  ./build-sparse/reverse_active_bench
```
