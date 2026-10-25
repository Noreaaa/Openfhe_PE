# Host snapshot: main-branch experiments

This backup preserves the experiment files that previously lived outside the Git repository under `/root/benchmarks`. No experiment is active. The recorded source baseline is main commit `08e56ef`; the earlier feature-branch archive remains at `018d59259d1478dd1859c17d4e4670007d0d3224` on `feature/sparse-conv2d`.

## Reports and preserved state

- [Original main baseline shape tests](main_original_full_encrypted_shapes_20261025/REPORT.md): completed 16384-ring cases for 64 and 128 channels; cancelled during the 256-channel case. Full records, source snapshots, missing-header compatibility declaration and input-image backup are included.
- [Repeated six-stage baseline](main_baseline_six_ring16384_20261025/REPORT.md): original attempt failed with heap corruption in stage 2. The standalone bounds fix passed a two-convolution numerical regression test; the rerun reached stage 5 activation and failed when the original modulus chain was exhausted. It is **not** a successful six-stage result. Both attempts, instrumentation, exact patches and raw memory/time records are included.
- [Archived feature-branch summaries and grouped estimates](feature_report_archive_20261025/): estimates are labelled and must not be reported as measurements.
- [ReLU roundtrip microbenchmarks](relu-roundtrip/): original local logs and NumPy reference/output arrays.
- `bootstrap65536-probe` and `security-depth-probe`: standalone Linux probe executables retained because they existed outside the repository. Their shared-library/runtime requirements are machine-specific; they are not portable source builds.

Generated build trees, compiler caches, Python caches and lock files are omitted. Original report/log contents and embedded absolute paths are retained as provenance. `snapshot_metadata.json` records the source commit and machine details; `MANIFEST.sha256` verifies snapshot file contents.

## Recovery on another machine

Install a compatible C++17 compiler, CMake, OpenMP, zlib, cnpy and OpenFHE **1.2.4, native 64-bit**. Supply installation prefixes via `CMAKE_PREFIX_PATH`, or use `$HOME/local/openFHE` and `$HOME/local/cnpy`. The recovery CMake project builds from preserved sources; it does not need the old build directories or static libraries.

From the repository root:

```bash
SNAPSHOT=experiments/host_snapshot_20261026
(cd "$SNAPSHOT" && sha256sum -c MANIFEST.sha256)
mkdir -p datasets/imagenet-mini
cp "$SNAPSHOT/main_original_full_encrypted_shapes_20261025/input_image_backup.bin" \
   datasets/imagenet-mini/ILSVRC2012_val_00000029.bin
cmake -S "$SNAPSHOT" -B build-main-recovery -DCMAKE_BUILD_TYPE=Release
cmake --build build-main-recovery -j12
./build-main-recovery/snapshot_bounds_regression
```

The original programs use paths relative to their working directory. Run the experiment executables **from `build-main-recovery`** so `../datasets` and `../python_model` resolve to the repository data/model directories. Set `LD_LIBRARY_PATH` to your OpenFHE/cnpy library directories if required by your installation.

```bash
cd build-main-recovery
OMP_NUM_THREADS=96 OMP_DYNAMIC=FALSE OMP_MAX_ACTIVE_LEVELS=1 \
  /usr/bin/time -v ./snapshot_original_baseline \
  --model baseline_test --ringDim 16384 --batch 8192 --act 1 \
  --top 0 --bottom 55 --left 0 --right 55 --nums 64
```

`snapshot_baseline_six` accepts the same arguments plus `--baseline-layers 6`. Its original depth-23 / 30/40-bit configuration is expected to expose the recorded limitations; the bounds fix does not fix modulus exhaustion or polynomial approximation accuracy. The original code also initializes both bootstrap and scheme-switch key families even for approximation.

The historical Python runners and their CMake files contain original absolute host paths. They are archived records; use the recovery CMake project above for relocation, or update paths explicitly before attempting to reuse those runners.
