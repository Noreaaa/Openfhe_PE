# Fully Encrypted Channel-Wise ResNet Prefix Baselines

Updated UTC: 2026-10-25T00:46:05.880005+00:00

## Scope and implementation

Original convolution 1 (Conv + BN + exact ReLU) executes on the simulated client. All 64 × 112 × 112 resulting features are encrypted, using one ciphertext per channel and 16384 logical slots. The server executes original convolutions 2–8, including the convolution that replaces max pooling. This is the same sequential prefix as previous experiments; residual additions and the classifier are not included. There is no plaintext feature bypass and no 10×10–80×80 partial-encryption sweep.

The channel-wise layout is derived from the existing `Conv2d_C` in `src/cnn/conv2d.cpp`. The benchmark adapter adds correct same padding and folded batch normalization, and caches rotations across output channels. It retains spatial indices across strided convolutions rather than repacking channels. This is an adapted implementation, not an unmodified historical executable.

Parameters: ring dimension 32768, 16384 slots, 30/40-bit scaling/first moduli, HYBRID numLargeDigits=6, 96 OpenMP threads. Each activation mode runs once in a separate process. Scheme switching uses depth 40; polynomial approximation uses depth 36. Both omit CKKS bootstrapping. FHEW sign bootstrapping remains part of the scheme-switch activation. Experimental HEStd_NotSet / FHEW TOY parameters do not establish 128-bit security.

Scheme-switch ReLU follows the current encrypted-mask implementation: CKKS → FHEW → encrypted sign → CKKS → ciphertext multiplication. No sign decryption occurs in the activation. Ciphertexts are processed sequentially for scheme switching; sign evaluations and library kernels use the thread pool.

Approximation reuses the historical polynomial p(z)=0.0229645+0.5z+1.4340124z²−2.089185z⁴+1.1551725z⁶. Although the historical comment calls it degree 7, its highest nonzero degree is 6. For numerical stability, each convolution output is divided by R=max(1,1.1×max|plaintext preactivation|) via folded coefficients; the activation returns R×p(x/R). Bounds are calibrated offline on this single benchmark image and this polynomial trajectory, not computed by decrypting server intermediate results. They are not dataset-wide guarantees. Unlike the historical `Relu_appx`, this adapter does not bootstrap after every activation; it reserves sufficient initial depth instead. Constants are injected only at active feature indices.

## Measurement conventions

Operator wall times exclude plaintext reference construction, calibration and decryption-based verification. Client computation, encryption and key generation are separately marked in logs/RSS samples and excluded from server forward totals. RSS is sampled every 0.5 s; per-layer operator-phase mean/peak and lifetime GNU time peak are distinct. All phases share a process: resident keys, allocator retention and reference tensors remain included in operator-phase RSS. Memory is not measured in an isolated server process. No allocator trimming or forced peak reset is used. No network transfer is timed. Arithmetic checks compare encrypted convolution and activation outputs with the matching plaintext path. Approximation-versus-exact-ReLU error is reported separately on the current polynomial-path preactivation; it is not end-to-end classifier accuracy.

## Summary

| Mode | Status | Server forward (min) | Conv+BN (min) | Activation (min) | Operator RSS mean / peak (GiB) | Whole-process peak (GiB) | Process wall (min) |
|---|---|---:|---:|---:|---:|---:|---:|
| approximation | success (7/7 layers) | 14.372 | 13.916 | 0.456 | 15.010 / 19.488 | 19.898 | 15.153 |
| scheme_switch | success (7/7 layers) | 376.758 | 37.374 | 339.384 | 91.826 / 100.956 | 101.315 | 378.480 |

## Per-layer results

| Mode | Original layer | Output C×H×W | Conv+BN (s) | Activation (s) | Cumulative (s) | Operator RSS mean / peak (GiB) | Level / towers | Calibration R | Conv max error | Activation arithmetic RMSE | vs exact ReLU RMSE |
|---|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| approximation | 2 | 64×56×56 | 91.421 | 6.956 | 98.377 | 18.487 / 19.488 | 4 / 33 | 12.0299 | 0.000451027 | 0.000215862 | 0.17099 |
| approximation | 3 | 64×56×56 | 184.311 | 5.792 | 288.480 | 16.029 / 17.007 | 9 / 28 | 5.4360 | 9.78962e-05 | 8.10318e-05 | 0.0713925 |
| approximation | 4 | 64×56×56 | 120.891 | 4.424 | 413.795 | 15.093 / 15.189 | 14 / 23 | 4.5377 | 0.000230423 | 9.46962e-05 | 0.0634626 |
| approximation | 5 | 64×56×56 | 96.675 | 3.477 | 513.947 | 15.087 / 15.110 | 19 / 18 | 1.6333 | 0.000246787 | 3.65092e-05 | 0.0261836 |
| approximation | 6 | 64×56×56 | 77.684 | 2.479 | 594.110 | 14.775 / 14.821 | 24 / 13 | 2.4323 | 0.000466134 | 8.97783e-05 | 0.036616 |
| approximation | 7 | 128×28×28 | 118.306 | 2.905 | 715.321 | 13.770 / 13.890 | 29 / 8 | 1.0000 | 0.000333111 | 2.86295e-05 | 0.0154007 |
| approximation | 8 | 128×28×28 | 145.665 | 1.320 | 862.306 | 12.405 / 12.516 | 34 / 3 | 2.4283 | 0.000158408 | 4.42038e-05 | 0.0334555 |
| scheme_switch | 2 | 64×56×56 | 111.947 | 2401.607 | 2513.554 | 98.327 / 100.956 | 12 / 29 | 1.0000 | 0.00337937 | 0.00394432 | 0.00394923 |
| scheme_switch | 3 | 64×56×56 | 145.464 | 2481.118 | 5140.136 | 92.146 / 93.477 | 14 / 27 | 1.0000 | 0.0248687 | 0.00200633 | 0.00200699 |
| scheme_switch | 4 | 64×56×56 | 128.081 | 2535.294 | 7803.511 | 91.582 / 92.807 | 16 / 25 | 1.0000 | 0.0494013 | 0.00276212 | 0.00275464 |
| scheme_switch | 5 | 64×56×56 | 122.966 | 2505.893 | 10432.371 | 90.165 / 91.513 | 18 / 23 | 1.0000 | 0.0149447 | 0.00127379 | 0.00127776 |
| scheme_switch | 6 | 64×56×56 | 119.105 | 2493.246 | 13044.722 | 89.496 / 90.790 | 20 / 21 | 1.0000 | 0.0583625 | 0.00303059 | 0.00302824 |
| scheme_switch | 7 | 128×28×28 | 427.227 | 3995.281 | 17467.230 | 89.567 / 90.763 | 22 / 19 | 1.0000 | 0.0149755 | 0.00100701 | 0.00100743 |
| scheme_switch | 8 | 128×28×28 | 1187.654 | 3950.609 | 22605.493 | 92.588 / 92.944 | 24 / 17 | 1.0000 | 0.015162 | 0.00128767 | 0.0012884 |

Raw artifacts per mode: `run.log`, `layers.csv`, `rss.csv`, `time.txt`, `result.json`, `command.json`. Source/binary SHA256 values are recorded in `configuration.json`. Pending or failed cases must not be interpreted as completed performance results.

## Post-reboot integrity check

Both configurations were already complete before the server shutdown. A post-reboot audit verified the executable SHA256 against the saved configuration, seven CSV rows per mode (original layers 2–8), zero exit codes in both process records, final phase markers, all seven completion log entries, and cumulative timing consistency. No interrupted configuration needs to be resumed or rerun. See `recovery_check.json` for the audit timestamp and results.

## Why convolution times differ between activation modes

The convolution function and channel-wise packing are shared, but the ciphertext states are not held fixed. Polynomial approximation consumes levels faster, shrinking its remaining RNS basis. The independently chosen initial depths (36 versus 40) also differ. Thus these are two full pipeline measurements, not a controlled benchmark of identical ciphertext convolution inputs.

| Original layer | Approx. input level / towers | Scheme-switch input level / towers | Approx. convolution (s) | Scheme-switch convolution (s) |
|---:|---|---|---:|---:|
| 2 | 0 / 37 | 0 / 41 | 91.421 | 111.947 |
| 3 | 4 / 33 | 12 / 29 | 184.311 | 145.464 |
| 4 | 9 / 28 | 14 / 27 | 120.891 | 128.081 |
| 5 | 14 / 23 | 16 / 25 | 96.675 | 122.966 |
| 6 | 19 / 18 | 18 / 23 | 77.684 | 119.105 |
| 7 | 24 / 13 | 20 / 21 | 118.306 | 427.227 |
| 8 | 29 / 8 | 22 / 19 | 145.665 | 1187.654 |

RNS tower counts explain part of the difference, especially late in the prefix: more towers mean more polynomial arithmetic, larger ciphertexts, and more data moved. However, the final-layer time ratio (about 8.15×) is much larger than the input tower-count ratio (19/8). It cannot be attributed to tower counts alone. Scheme-switch operator RSS remains around 90–100 GiB versus 12–19 GiB for approximation; resident switching keys, allocator behavior and cache/memory-bandwidth effects are plausible contributors, not established causal measurements. These runs did not record historical swap activity, page faults or memory-bandwidth counters, so those causes cannot be asserted. A controlled follow-up should time the shared convolution at matched depth/level and layout in separate processes with and without resident scheme-switch keys, and capture hardware/OS counters.
