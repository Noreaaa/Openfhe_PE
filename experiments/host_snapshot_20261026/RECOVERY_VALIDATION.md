# Recovery validation

Validated on the original host before upload:

- Recovery CMake configuration succeeded using installed OpenFHE 1.2.4 and cnpy.
- All three targets rebuilt from source: `snapshot_original_baseline`, `snapshot_baseline_six`, `snapshot_bounds_regression`.
- The two-convolution retained-slot bounds regression passed; maximum absolute errors were below 1e-9 for both stages.
- The 87 selected files from `/root/benchmarks` were copied byte-for-byte.
- Historical failed/cancelled results remain labelled as such. No full experiment was restarted during this backup.

The recovery build is independent of the omitted original static libraries and build trees. The archived Python runners retain old paths; relocation instructions are in README.md.
