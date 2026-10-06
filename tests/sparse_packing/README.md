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
installation convention. No local Windows OpenFHE installation is needed:
upload these sources and build/run them on the server. Compilation and runtime
validation must be performed there; they have not been completed locally.

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
three large digits. The ring dimension is set explicitly by `--ring`, without
the CKKS 128-bit security-level check. These runs do not establish 128-bit
security. FHEW still uses `STD128`, not `TOY`.

Defaults are depth 3, 50-bit scale, 60-bit first modulus, level 0 (FLEXIBLEAUTO
for 64-bit native integers; FIXEDAUTO for 128-bit). They intentionally isolate
conversion and are not the original CNN parameters. For an experiment matching
your application, specify `--depth`, `--scale-bits`, `--first-bits`, and `--level`
and retain the parameter record. `level` creates an input directly at that
level; it does not reproduce accumulated CNN noise. At least three levels of
depth must remain. No EvalSign, FHEW bootstrapping keys, or reverse conversion
are included. Do not compare these numbers to the complete activation time.

Smaller S may need more ciphertexts for a fixed CNN workload. Evaluate total
layer time including repacking before adopting it in the actual CNN.
