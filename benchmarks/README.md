# Experiment reports and raw measurements

This directory preserves the experiment state at the Git upload checkpoint. Completed, failed and cancelled runs are retained separately. Incomplete measurements are not valid completed results; estimates are labelled in their reports. All experiment processes were stopped at the user's request before upload.

Tracked artifacts include reports, configurations, timing CSVs, RSS samples, logs, figures and source snapshots. Generated ELF executables and lock files remain local and are ignored by Git. Configuration files retain executable hashes where recorded. Memory units and accounting boundaries vary across historical campaigns; consult each report before comparing results.

| Experiment | Status at checkpoint | Report |
|---|---|---|
| bootstrap65536_20261023 | See report | [README.md](bootstrap65536_20261023/README.md) |
| channelwise_approx_boot_shapes_r16384_r32768_20261025 | cancelled_by_user | [REPORT.md](channelwise_approx_boot_shapes_r16384_r32768_20261025/REPORT.md) |
| channelwise_approx_bootstrap_ring32768_20261025 | cancelled_by_user | [REPORT.md](channelwise_approx_bootstrap_ring32768_20261025/REPORT.md) |
| channelwise_conv_shapes_r16384_r32768_20261025 | failed | [REPORT.md](channelwise_conv_shapes_r16384_r32768_20261025/REPORT.md) |
| channelwise_full_ring32768_20261025 | finished | [REPORT.md](channelwise_full_ring32768_20261025/REPORT.md) |
| conv_memory_optimization_20261023 | See report | [README.md](conv_memory_optimization_20261023/README.md) |
| first_layer_code_audit_20261024 | See report | [README.md](first_layer_code_audit_20261024/README.md) |
| legacy_pfhe_conv_shapes_r16384_r32768_20261025 | cancelled_scope_corrected | [REPORT.md](legacy_pfhe_conv_shapes_r16384_r32768_20261025/REPORT.md) |
| relu_roundtrip_20261022 | See report | [README.md](relu_roundtrip_20261022/README.md) |
| resnet_client_first_ring2048_slots1024_area10_20261024 | finished | [REPORT.md](resnet_client_first_ring2048_slots1024_area10_20261024/REPORT.md) |
| resnet_client_first_slots8192_20261024 | finished | [REPORT.md](resnet_client_first_slots8192_20261024/REPORT.md) |
| resnet_repeated_30_40_d6_depth40_20261023 | cancelled | [REPORT.md](resnet_repeated_30_40_d6_depth40_20261023/REPORT.md) |
| resnet_ring2048_depth30_area10_layers6_20261025 | finished | [REPORT.md](resnet_ring2048_depth30_area10_layers6_20261025/REPORT.md) |
| resnet_ring2048_depth30_decrypted_mask_area10_layers6_20261025 | finished | [REPORT.md](resnet_ring2048_depth30_decrypted_mask_area10_layers6_20261025/REPORT.md) |
| resnet_ring4096_depth30_decrypted_mask_area10_layers6_20261025 | finished | [REPORT.md](resnet_ring4096_depth30_decrypted_mask_area10_layers6_20261025/REPORT.md) |
| resnet_ring4096_depth30_encrypted_mask_areas10_40_layers6_20261025 | finished | [REPORT.md](resnet_ring4096_depth30_encrypted_mask_areas10_40_layers6_20261025/REPORT.md) |
| resnet_ring4096_depth30_encrypted_mask_topleft_areas10_40_layers6_20261025 | finished | [REPORT.md](resnet_ring4096_depth30_encrypted_mask_topleft_areas10_40_layers6_20261025/REPORT.md) |
| resnet_ring8192_depth30_decrypted_mask_area10_layers6_20261025 | finished | [REPORT.md](resnet_ring8192_depth30_decrypted_mask_area10_layers6_20261025/REPORT.md) |
| resnet_ring_comparison_slots2048_20261023 | finished | [REPORT.md](resnet_ring_comparison_slots2048_20261023/REPORT.md) |
| reverse_active_20261023 | See report | [README.md](reverse_active_20261023/README.md) |
| security_depth_20261023 | See report | [README.md](security_depth_20261023/README.md) |
| sparse_conv_level_encoding_20261025 | See report | [REPORT.md](sparse_conv_level_encoding_20261025/REPORT.md) |
| sparse_conv_optimization_20261025 | cancelled_by_user | [REPORT.md](sparse_conv_optimization_20261025/REPORT.md) |
