# Minimal encrypted-ReLU experiment — 2026-10-22

The ResNet prefix uses `SparseConv2dBN_P`, a sparse row-packed convolution/BN implementation. It is a sequential convolution prefix rather than a complete residual ResNet-18 classifier.

The new ReLU computes `x * (1 - SwitchBack(EvalSign(SwitchForward(1000*x), true)))`. Signs are never decrypted during evaluation. OpenFHE's reverse-switch bit encoding uses `p=4, pmin=-1, pmax=1`; simply applying the old default `EvalSign` encoding is not equivalent. The evaluator no longer retains the FHEW secret key. Decryption in the benchmark is only for output validation.

## Environment and workload

- Ubuntu 22.04, Montage Jintide C6348, 96 logical CPUs / 48 physical cores, approximately 188 GiB RAM.
- OpenFHE 1.2.4, C++17 Release build, native 64-bit arithmetic, FLEXIBLEAUTO, HYBRID key switching, three large digits, FHEW TOY.
- All processes configured with 96 OpenMP threads and nested teams disabled. This is a maximum team size, not a claim that every operation keeps 96 CPUs busy.
- Ring dimension 16384, 1024 logical slots, three channels of width 512, one encrypted row, columns 510–511: two ciphertexts and six active values.
- Positive, negative, and zero inputs, last-slot and partial-block coverage. Three forward trials per process; all 1024 slots of each output ciphertext and the public tensor are checked against a `1e-3` absolute tolerance.
- Each variant ran in its own process, serially. Raw `.log` and GNU time `.time` files are retained beside `summary.json`. Stage times are medians of three trials, including the first trial; primitive times are medians of five calls. These are small-sample local measurements, not full-model throughput.

## Before/after comparison

| Variant | Depth | Scale / first modulus bits | ReLU latency, two ciphertexts | Whole-process peak RSS | Max absolute output error |
|---|---:|---:|---:|---:|---:|
| Original decrypted-sign implementation, original small-test parameters | 10 | 30 / 40 | 1520.71 ms | 2.897 GiB | 2.36e-5 |
| Original implementation, parameters matched to new path | 20 | 50 / 60 | 2471.06 ms | 5.314 GiB | 1.51e-11 |
| Encrypted FHEW→CKKS mask and ciphertext product | 20 | 50 / 60 | 4900.22 ms | 5.252 GiB | 1.06e-6 |

Against the original small-test configuration, latency is **3.22×** and peak RSS is **1.81×** (+2.355 GiB). This includes the larger scale/modulus and deeper chain required by the new path. At matched parameters, latency is **1.98×**; peak RSS is approximately 1.2% lower in this run, too small a difference to claim a general memory improvement. Setup and switching keys dominate this tiny workload's process memory. The baseline retains all forward-conversion results and parallelizes ciphertext blocks, whereas the new sparse path processes blocks sequentially and uses internal transform parallelism; scheduling is therefore part of the implementation comparison.

The old 30-bit scale produced an above-tolerance result in an initial reverse-conversion probe. The production path uses 50/60 bits, and the main executable conservatively budgets `max(20, 3*layers+14)` levels (automatic selection: `max(23, 3*layers+16)`). The sweep default is now depth 46 and all available CPUs. The archived depth-35/32-thread results predate this change and cannot be reused as a measured full-model baseline on this server.

## New ReLU stage costs

These times cover **both ciphertexts together**, with six sign evaluations and 2048 logical values extracted/converted in each switching direction. Medians of separate stages need not add exactly to the median total.

| Stage | Median wall time |
|---|---:|
| CKKS scalar multiplication by 1000 | 3.11 ms |
| CKKS→FHEW conversion | 1786.09 ms |
| Six FHEW `EvalSign(..., true)` evaluations | 338.25 ms |
| FHEW→CKKS conversion | 2725.92 ms |
| Encrypted complement `1 - negative` | 1.88 ms |
| CKKS ciphertext×ciphertext products | 38.47 ms |
| Complete ReLU forward | 4900.22 ms |

Reverse conversion accounts for approximately 55.6% of total latency. The final products account for approximately 0.8%. The resulting ciphertexts have level 12 and nine remaining towers at depth 20; the old plaintext-mask output stayed at level 0 with 21 towers. Consecutive layers therefore require a revised depth budget.

## Individual CKKS primitives

One fresh level-0 ciphertext per operand, ring 16384, slots 1024, depth 20, scale/first modulus 50/60 bits. Timing includes each API call and excludes result destruction. `EvalMult_ct_ct` includes the API's key switching/relinearization; these are not separately instrumented inside OpenFHE. `Rescale` uses a multiplication result. These probe inputs differ in level from the final ReLU products above.

| Operation | Median single-call latency |
|---|---:|
| `EvalAdd(ct, ct)` | 1.043 ms |
| `EvalRotate(ct, 1)` | 29.196 ms |
| `EvalMult(ct, scalar)` | 0.573 ms |
| `EvalMult(ct, plaintext)` | 1.214 ms |
| `EvalMult(ct, ct)` | 34.096 ms |
| `EvalSub(1, ct)` | 1.225 ms |
| `Rescale(product)` | 0.424 ms |

Context creation took 86.54 ms, CKKS key generation approximately 437 ms, switching setup/key generation approximately 4103 ms, and forward precomputation approximately 378 ms in the encrypted run. Setup is excluded from the ReLU timings and included in GNU time's peak RSS.

## Validation

The standalone convolution and activation CTests passed (2/2). Activation tests include repeated Conv+BN+encrypted-ReLU stages at nonzero levels: stages 2 and 3 reached levels 15 and 18, with maximum errors 1.76e-5 and 1.84e-5. The real-image third-convolution/BN/ReLU check also passed: 200704 outputs (3136 in the encrypted region), convolution max error 2.29e-11 and ReLU max error 4.46e-4, all within the 1e-3 tolerance. This larger correctness check took 101.30 seconds and reached 9.002 GiB peak RSS; it is a different workload from the two-ciphertext latency/memory comparison. Raw logs, GNU time metrics, and `real-accuracy-errors.csv` are included. It used depth 20, scale/first modulus 50/60 bits, 28 ciphertexts after convolution, and 3136 sign evaluations. A complete ten-convolution encrypted-prefix sweep has not been run for this change.

## Reproduction

From the repository root after installing OpenFHE 1.2.4:

```bash
cmake -S tests/sparse_packing -B build-sparse -DCMAKE_BUILD_TYPE=Release
cmake --build build-sparse -j "$(nproc)"
export OMP_NUM_THREADS="$(nproc)" OMP_THREAD_LIMIT="$(nproc)"
export OMP_DYNAMIC=FALSE OMP_MAX_ACTIVE_LEVELS=1
export LD_LIBRARY_PATH="$HOME/local/openFHE/lib:$HOME/local/cnpy/lib:${LD_LIBRARY_PATH:-}"
/usr/bin/time -v ./build-sparse/relu_roundtrip_bench baseline 10 30
/usr/bin/time -v ./build-sparse/relu_roundtrip_bench baseline 20 50
/usr/bin/time -v ./build-sparse/relu_roundtrip_bench encrypted 20 50
ctest --test-dir build-sparse --output-on-failure
```

The unchanged baseline fixture is linked only into `relu_roundtrip_bench`, never into the production CNN library. The benchmark arguments are variant, depth, and scale bits; the first modulus is scale bits + 10.
