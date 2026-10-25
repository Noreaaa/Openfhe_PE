# Standalone CKKS-to-FHEW sparse packing benchmark

This directory is a separate CMake project. It does not compile `src/main.cpp`,
the CNN, or cnpy, and needs no model/data files. Use the same OpenFHE version
as the server (the parent project documents 1.2.4).

## Build on the server

From `Openfhe_PE`:

```bash
cmake -S tests/sparse_packing -B build-sparse \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-sparse -j 4
./build-sparse/sparse_packing_bench --help
```

The build searches `$HOME/local/openFHE`, matching the existing parent project's
installation convention. Build and run on Linux with OpenFHE installed. The synthetic convolution and
activation correctness tests are available through `ctest --test-dir build-sparse
--output-on-failure`; see the repository README for the complete workflow.

For a different installation, pass `-DCMAKE_PREFIX_PATH=/actual/install/prefix` or
`-DOpenFHE_DIR=/path/to/directory/containing/OpenFHEConfig.cmake`.
If the loader cannot find OpenFHE shared libraries, add the installation's
`lib` directory to `LD_LIBRARY_PATH`. Static installations can use `-DBUILD_STATIC=ON`.

## First comparison: fixed N and K, varying genuine encoding slots S

```bash
mkdir -p results-sparse
for s in 256 1024 4096 8192; do
  ./build-sparse/sparse_packing_bench \
    --ring 16384 --slots "$s" --outputs 64 \
    --threads 4 --warmup 1 --repeats 5 \
    > "results-sparse/N16384-S${s}-K64.csv" \
    2> "results-sparse/N16384-S${s}-K64.log" || break
done
```

8192 slots is full packing for N=16384; the other cases use explicit sparse
encoding and matching switching setup. Each case runs in its own process so
keys, contexts and precomputations from earlier cases do not accumulate.
Full packing precomputation can require substantial time and memory; start
with the small-slot case and monitor memory before running larger cases.

## Separate output count from transform size

Keep `--ring 16384 --slots 1024` fixed and try `--outputs 16`, `64`, `256`,
and `1024`. This measures changing the extraction count without changing
the configured transform size. Outputs must be positive and at most slots.

To study ring dimension, keep `--slots 1024 --outputs 64` fixed and compare
`--ring 16384`, `32768`, and `65536`. Keep other flags and CPU allocations fixed.
Use `--threads` no greater than the CPU allocation of the job.

## What is measured

- `setup_ms`: context creation, key generation, switching setup/key generation,
  and switching precomputation. Excludes input encoding/encryption.
- `min/median/mean/max_ms`: only `EvalCKKStoFHEW`, after warmup. Cloning,
  decrypting, validating, and printing are outside the timer.
- stderr logs every measured trial, actual ring dimension, input level/towers,
  LWE dimension, library version and configured OpenMP thread limit.
- stdout contains a CSV header and summary. Prefer the median for comparisons;
  retain the trial logs to inspect noise. This is single-ciphertext latency,
  not CNN throughput or timing of the internal switching stages.

Input is deterministic signed integers from -8 to 8, repeated across all S
logical slots, with identical first K values between cases. Explicit `slots`
in encoding distinguishes this from full packing with a short zero-padded input.
All input CKKS slots and all output LWE values are checked. LWE error is measured
modulo the plaintext modulus (negative values wrap); tolerance is one integer
unit because switching is approximate. This is a layout/conversion sanity
check, not a guarantee of ReLU sign accuracy near zero. Failed checks produce
a nonzero exit status; do not use a failed result as a valid performance point.

## Security and matching your CNN

CKKS uses `HEStd_NotSet`, uniform ternary secrets, HYBRID switching with
three large digits by default (`--digits 3`), matching `SetParamCKKS` in
`src/main.cpp`. FHEW uses `TOY` and logQ=25, matching `SetParam_TFHE`.
The ring dimension is explicitly set by `--ring`. These are experimental
parameters, without a 128-bit security requirement.

The forward-only microbenchmark retains the historical 30/40-bit settings (the current CNN roundtrip uses 50/60 bits). Its depth is computed as
`3 + FHECKKSRNS::GetBootstrapDepth({3, 3}, UNIFORM_TERNARY)`, with 30-bit scale,
40-bit first modulus and FLEXIBLEAUTO on 64-bit builds. On native 128-bit builds,
use 78/89 bits and FIXEDAUTO (the intended settings of main.cpp's 128-bit branch).
Calling GetBootstrapDepth only computes the chain depth; it does not perform
bootstrapping. The actual default depth is printed in help and the run log.
Optional `--depth`, `--scale-bits`, `--first-bits`, `--digits`, and `--level`
override these defaults. If explicitly choosing `--depth 3`, also use `--digits 2`
to avoid the four-tower/three-digit partition error on FLEXIBLEAUTO builds.
Keep all these parameters fixed across slot-count comparisons.

Unlike main.cpp, this test explicitly sets the encoding/switching slots to S
and generates only the forward switching keys. Input level defaults to 0.
`level` creates an input directly at that
level; it does not reproduce accumulated CNN noise. At least three levels of
depth must remain. No EvalSign, FHEW bootstrapping keys, or reverse conversion
are included. Do not compare these numbers to the complete activation time.

Smaller S may need more ciphertexts for a fixed CNN workload. Evaluate total
layer time including repacking before adopting it in the actual CNN.


## Minimal encrypted-ReLU comparison

`relu_roundtrip_bench` compares the unchanged old implementation (stored only in
`baseline/`) with the current sparse ReLU. Run each command in its own process,
serially, to obtain independent peak-memory measurements:

```bash
mkdir -p /root/benchmarks/relu-roundtrip
/usr/bin/time -v ./build-sparse/relu_roundtrip_bench baseline 10 30
/usr/bin/time -v ./build-sparse/relu_roundtrip_bench baseline 20 50
/usr/bin/time -v ./build-sparse/relu_roundtrip_bench encrypted 20 50
```

Arguments are variant, multiplicative depth, and scale bits; the first modulus is
scale bits + 10. All variants use ring dimension 16384, 1024 logical slots, and all
available OpenMP CPUs. The input has three channels, width 512, one encrypted row,
and two encrypted positions per channel (two ciphertext blocks, six active
values). It includes positive, negative, and zero values, slot 1023, and a partial
last block. Each process measures three forwards, verifies every output slot and
the public tensor with a `1e-3` absolute tolerance, and then measures five calls
per CKKS primitive. Validation decryption is outside the forward timers.

`TRIAL` reports pipeline stage wall times, RSS, and process high-water RSS.
`PRIMITIVE` reports single-call medians for fresh level-0 ciphertexts; those times
are not directly interchangeable with the final product at a deeper level.
Baseline stage timers group amplification with forward switching, sign with
secret-key decryption, and encoding with plaintext multiplication, preserving the
original timing boundaries. The baseline fixture is never linked into production
targets. GNU time includes setup, validation, and primitive probes; the per-trial
high-water marks are also reported. This tiny workload establishes correctness and
local operation costs, rather than full-model throughput or a 96-thread speedup.


## Active sign count versus fixed reverse-transform size

`reverse_active_bench` holds ring=16384, slots=1024, depth=20, scale/first
modulus=50/60 bits, and all available CPUs fixed. It calls the production
`EncryptedReluMask` with 6, 64, 256, or 1024 active positions. Every reverse
conversion still receives 1024 FHEW ciphertexts, exactly as in production;
inactive positions reuse an existing encrypted sign. Forward switching is done
once before timing. One warmup per case precedes five measured rounds, with
case order reversed on alternate rounds. Sign computation and FHEW-to-CKKS
conversion are timed separately. Decryption checks all mask slots outside the
timed region. This isolates active-value count at a fixed API transform size;
it does not measure changing `numCtxts`, logical slots, or ring dimension.

```bash
cmake --build build-sparse --target reverse_active_bench -j "$(nproc)"
OMP_NUM_THREADS="$(nproc)" OMP_THREAD_LIMIT="$(nproc)" \
  OMP_DYNAMIC=FALSE OMP_MAX_ACTIVE_LEVELS=1 ./build-sparse/reverse_active_bench
```

Measurements and raw logs: [active-value experiment](../../benchmarks/reverse_active_20261023/README.md).

## Convolution memory experiments

Production defaults preserve execution speed: direct rotation keys and unrestricted
output-channel concurrency (`--conv-scratch-mib 0`). Existing rotation keys are
reused rather than regenerated. The logical slot default is 2048.

The signed power-of-two rotation basis and scratch-limited policies were measured
but rejected as production defaults because they increased forward latency.
They remain accessible only through explicit API arguments and the standalone
`conv_memory_bench optimized [scratch_mib]` experiment. The baseline measurement
uses `conv_memory_bench baseline`. See the [historical measurement report](../../benchmarks/conv_memory_optimization_20261023/README.md).

The cancelled sweep remains cancelled.

## Sparse convolution rotation optimization

The fast path reduces `min(input_channels, channels_per_ciphertext)` occupied
channel positions and shares baby rotations during stride-2 compaction. The
compaction still uses one ciphertext/plaintext multiplication level and retains
the same row/channel output layout. Direct keys and maximum channel concurrency
remain the defaults. `CONV_PROFILE` separates public arithmetic, preparation,
encrypted work and output insertion timings.

The wide-region correctness fixture covers nonzero-start compaction, odd channel
counts, padding and chained convolutions. The root-build target
`sparse_conv_opt_bench ROOT LAYER AREA INPUT_LEVEL` checks real ResNet weights
against plaintext Conv+BN at a specified input level. Frozen before/after
executables and measurements are kept under
`benchmarks/sparse_conv_optimization_20261025/`; the benchmark runner is
`scripts/benchmark_sparse_conv_optimization.py`. Prior full-prefix results remain
historical measurements of their saved binaries.


## Working-level plaintext encoding

The sparse fast path encodes kernel plaintexts at the effective multiplication
level, accounting for an outstanding automatic rescale. Selection/compaction
plaintexts use the next multiplication level; public-value additions use the
output ciphertext level. The reference path retains its default encoding.
Packing, parallelism and multiplicative-level consumption are unchanged.

The wide-region correctness fixture starts at level 4 with noise-scale degree 2
and checks both numerical output and level consumption through two convolutions.
The real-weight benchmark accepts an optional input noise-scale degree:
`sparse_conv_opt_bench ROOT LAYER AREA INPUT_LEVEL [INPUT_NOISE_DEGREE]`.
The matched before/after observation is archived in
`benchmarks/sparse_conv_level_encoding_20261025/`.
