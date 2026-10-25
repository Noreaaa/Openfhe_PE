# Main-branch experiment backup

The host snapshot is preserved in [experiments/host_snapshot_20261026](experiments/host_snapshot_20261026/README.md).

- [Original fully encrypted shape experiments](experiments/host_snapshot_20261026/main_original_full_encrypted_shapes_20261025/REPORT.md): cancelled; completed cases and raw data retained.
- [Six-stage baseline](experiments/host_snapshot_20261026/main_baseline_six_ring16384_20261025/REPORT.md): the original attempt crashed due to retained-index buffer overflow; the bounds-fixed attempt stopped in stage 5 activation when the original modulus budget was exhausted. These are not successful six-stage inference results.
- [Bounds fix and instrumentation](experiments/host_snapshot_20261026/main_baseline_six_ring16384_20261025/source/): standalone sources and exact patches.
- [Earlier reports and grouped estimates](experiments/host_snapshot_20261026/feature_report_archive_20261025/).

The larger feature-branch experiment archive is already committed at `018d59259d1478dd1859c17d4e4670007d0d3224` on `feature/sparse-conv2d`.
