# Sparse convolution optimization: matched-input benchmark

Real ResNet weights and the same image are used for both executables. Ring dimension 16384, 8192 slots, depth 40, 30/40-bit moduli, HYBRID digits 6, 96 threads; direct rotation keys and no channel-concurrency cap. The original-image sensitive region is 40×40 and is propagated through the preceding plaintext Conv+BN+ReLU stages. This tests individual convolution calls, not complete encrypted prefixes. Scheme-switch keys are not resident in either process. Input levels are matched to the archived prefix (0, 12 and 24 for layers 2, 3 and 7). One observation per case; timings are not statistical estimates.

Both implementations perform the same public convolution, encrypted convolution, packing and folded BN. The optimized path reduces only occupied channel positions, uses one-level baby-step/giant-step stride-2 compaction, and performs owned accumulator additions in place. Output layout and multiplicative-level consumption must match.

| Layer | Before (s) | After (s) | Speedup | Before / after output level | Before / after whole-process peak (GiB) | Before / after max absolute error |
|---:|---:|---:|---:|---|---|---|
| 2 | 182.622 | 119.979 | 1.52× | 1 / 1 | 24.895 / 25.783 | 0.000606445 / 0.000556144 |

The new `CONV_PROFILE` log separates public convolution, preparation/encoding, encrypted processing, and final public-value insertion. Decryption/reference checks occur outside convolution timers. Full-process memory includes setup/verification and is not isolated cloud RSS. Archived full-prefix measurements are not overwritten.

## Correctness checks

The sparse convolution suite passed after extending the wide-region case (31-pixel width, seven input channels, nonzero starting column), covering kernels 1/3/7, strides 1/2, channel-block tails, BN/bias, and chained convolutions. A separate 10000-case randomized index check verified the compaction map. Each real-weight case below also decrypts and compares the complete output to plaintext Conv+BN.

## Cancelled by user

The optimization benchmark was cancelled. Completed measurements above are retained. The interrupted layer-3 optimized run is not a valid timing measurement (it was paused and then terminated). Layer 7 was not run. No automatic restart is scheduled. Production code changes are retained; this report is not a completed three-case benchmark.
