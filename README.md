# PFHE: Partially Encrypted CNNs and ResNet Prefix Benchmarks

PFHE uses OpenFHE CKKS to encrypt a rectangular region of an image and evaluates convolutions and ReLU on mixed plaintext/ciphertext features. This repository contains the implementation, correctness tests, standalone microbenchmarks, sweep scripts, and raw experimental results.

## Implementation overview

The current `feature/sparse-conv2d` branch focuses on sparse row packing and sequential convolution prefixes using exported ResNet weights. The main execution path is:

```text
src/main.cpp: parse options, configure CKKS/FHEW, and generate keys
  -> src/init/resnet18.cpp: load the image/weights and assemble the prefix
  -> Encrypt_MCSR_P: encrypt the selected rectangle and zero its public entries
  -> Network::predict_P: alternate SparseConv2dBN_P and sparse_activation
  -> per-layer CSV profiles and process logs
```

Each encrypted row is split into channel blocks. For a feature width `W` and `S` logical slots, a ciphertext holds `floor(S / W)` complete channel rows; unused slots are zero-filled. Convolution propagates the encrypted region to every output position whose receptive field intersects it. Batch normalization is folded into the convolution computation. The optimized path reuses spatial rotations, reduces channel contributions, and repacks output rows.

This prefix is **not a complete ResNet-18 classifier**: it has no residual additions or classification head. Its second convolution loads `maxpool_0_weight.npy` and the corresponding batch-normalization parameters; it does not execute standard ResNet max pooling. The `basic_blk` and `skip_connect` source/header files are empty placeholders.

The current sparse ReLU scales the CKKS values by 1000, switches to FHEW, evaluates encrypted signs with `EvalSign(..., true)`, and switches the negative indicators back to CKKS. It computes the encrypted mask `1 - negative` and multiplies it by the original CKKS input. **ReLU evaluation does not decrypt signs or retain a FHEW secret key.** Constructor key arguments remain for source compatibility; secret keys are still needed during setup and for optional test/reference decryption. The input encryption routine still prints encoded input plaintexts to the log.

The roundtrip path uses a 50-bit scale and 60-bit first modulus on native 64-bit builds. The original 30/40-bit settings were insufficiently stable in the minimal reverse-conversion check. OpenMP defaults to all CPUs available to the process (96 logical CPUs / 48 physical cores on the current server), with nested teams disabled. Ciphertext blocks are processed sequentially in sparse ReLU so each conversion can use internal parallelism without multiplying reverse-transform working sets.


## Source and file guide

Matching `.hpp` files declare the interfaces implemented by their `.cpp` files, unless noted otherwise.

| File or directory | Purpose |
|---|---|
| `CMakeLists.txt` | Builds the `encryption` and `cnn` libraries, the main `test` executable, and the real-weight benchmark/accuracy executables. |
| `src/main.cpp` | Command-line entry point, parameter validation, cryptographic context/key setup, model selection, and peak-memory reporting. |
| `src/init/resnet18.{cpp,hpp}` | Builds 1–10 sequential Conv+BN+ReLU stages, selects the sparse fast path, and optionally compares each prefix with a plaintext reference. |
| `src/init/hcnn.{cpp,hpp}` | Loads the legacy HCNN layers and weights. |
| `src/init/baseline_test.{cpp,hpp}` | Legacy synthetic encrypted convolution/activation baseline. |
| `src/encryption/setparam.{cpp,hpp}` | CKKS ring/slot/scale/depth settings and FHEW scheme-switching parameters. |
| `src/encryption/encryption.{cpp,hpp}` | Partial encryption and multiple-channel single-row packing, plus legacy compact/baseline layouts. |
| `src/cnn/layer.{cpp,hpp}` | Layer types and the virtual interfaces for plaintext/ciphertext layouts. |
| `src/cnn/model.{cpp,hpp}` | Layer sequencing, per-layer timing, CSV output, and optional feature observers. |
| `src/cnn/sparse_conv2d.{cpp,hpp}` | Sparse Conv+BN reference/optimized paths and shape/slot validation. Only `SparseConv2dBN_P::forward` has a working convolution path; the other sparse convolution variants and compact `forward_C` throw unimplemented errors. |
| `src/cnn/sparse_activation.{cpp,hpp}` | Sparse row-packed encrypted-mask ReLU and amplification/forward/sign/reverse/complement/product timing counters; compact packing is unimplemented. |
| `src/cnn/encrypted_relu.hpp` | Shared encrypted sign-to-CKKS-mask conversion for sparse and legacy scheme-switch ReLUs. |
| `src/cnn/conv2d.{cpp,hpp}` | Legacy convolution/Conv+BN layouts, plaintext `GoldenConv2d`/`GoldenBN`, and encrypted-region propagation helpers. |
| `src/cnn/conv2d_l.{cpp,hpp}` | Deprecated convolution variant for later ImageNet layers. |
| `src/cnn/activation.{cpp,hpp}` | Legacy square, scheme-switch ReLU, polynomial ReLU, and plaintext activation helpers. |
| `src/cnn/pool.{cpp,hpp}` | Sum/average pooling implementations and plaintext pooling helpers. |
| `src/cnn/bootstrap.{cpp,hpp}` | CKKS bootstrapping layer for partially encrypted tensors; not used by the current scheme-switch ResNet prefix. |
| `src/cnn/linear.{cpp,hpp}` | Legacy linear layers, including ciphertext reconstruction/decryption before plaintext fully connected computation. |
| `src/cnn/region_update.{cpp,hpp}` | Legacy encrypted-region bookkeeping helpers. |
| `src/cnn/basic_blk.{cpp,hpp}`, `src/cnn/skip_connect.{cpp,hpp}` | Empty placeholders for residual blocks and skip connections. |
| `src/utils/dataloader.{cpp,hpp}` | Loads NumPy weights/features, CIFAR input, and raw ImageNet image tensors. |
| `src/utils/helper.{cpp,hpp}` | Region propagation, range checks, vector rotations, tensor printing, and synthetic inputs. |
| `src/utils/globals.{cpp,hpp}`, `src/utils/types.hpp` | Shared cryptographic/layout state and multidimensional tensor aliases. |
| `src/utils/Resnet-18.txt` | Text description of a ResNet-18 model. |
| `src/include/cmdline.h`, `src/include/cnpy.h` | Bundled command-line parser and cnpy declarations; the cnpy library is still required. |
| `tests/sparse_packing/CMakeLists.txt` | Independent build for the packing benchmark and synthetic convolution/activation tests. |
| `tests/sparse_packing/sparse_packing.cpp` | CKKS-to-FHEW microbenchmark varying logical slots, extraction count, and ring dimension. |
| `tests/sparse_packing/sparse_conv2d_test.cpp` | Synthetic convolution/BN checks across slot counts, kernels, strides, and chained convolutions. |
| `tests/sparse_packing/sparse_activation_test.cpp` | Sparse ReLU checks covering the last slot, partial channel blocks, and Conv-to-ReLU chaining. |
| `tests/sparse_packing/sparse_real_accuracy.cpp` | Real-image third-convolution/BN/ReLU check against an independent long-double reference; saves tensors and error reports. |
| `tests/sparse_packing/resnet_conv3_bench.cpp` | Real-weight third-convolution/ReLU benchmark with legacy, reference, and optimized variants. |
| `tests/aff_conv.cpp`, `tests/verify_py.py` | C++/Python checks of the output region affected by a changed input rectangle. |
| `tests/innerproduct/`, `tests/schemeswitch/` | Separate CMake projects for legacy CKKS inner-product and scheme-switching examples. |
| `tests/test_ring_dimension.cpp` | Incomplete legacy ring-dimension experiment, outside the main build. |
| `tests/test_vec.txt` | Stored test-vector data. |
| `scripts/run_resnet_sweep.py` | Serial sweep runner with CPU affinity, resume signatures, GNU time metrics, and persistent status/results. |
| `scripts/start_resnet_sweep_tmux.sh` | Builds the main target and launches runner/monitor windows in tmux. |
| `python_model/HCNN.py` | PyTorch CIFAR-10 HCNN training/evaluation and parameter export. |
| `python_model/resnet-18.py` | CIFAR-100 ResNet-style training with square activations; separate from the C++ prefix experiment. |
| `python_model/resnet-12.py` | Experimental CIFAR-100 convolutional training script. |
| `python_model/resnet-18-8.py` | Extracts a torchvision ResNet-18 feature map after `layer2` for later-layer experiments. |
| `python_model/data_reader.cpp` | Empty placeholder. |
| `python_model/saved_models/` | HCNN checkpoint/NumPy parameters, exported ResNet-18 parameters, and a ResNet-8 NPZ archive with tensor metadata. |
| `python_model/histories/` | TensorBoard training event files. |
| `datasets/imagenet-mini/` | The single raw input tensor retained for reproduction. |
| `test_results/` | Archived sweep configuration, per-case profiles/logs, summary, and result analysis. |
| `artifacts/` | Split historical archive, checksums, packaging-time environment metadata, and a legacy runner snapshot. |
| `.gitignore`, `LICENSE`, `readme.md` | Generated-file exclusions, license, and a short link to this README. |

Binary model files, event files, input tensors, and archive parts are data artifacts rather than source code. Legacy executables such as `tests/aff_test` and `tests/innerproduct/test` may also be present; rebuild from source on the target machine.

## Environment and dependencies

- Linux, a C++17 compiler, CMake >= 3.16, and OpenMP.
- OpenFHE 1.2.4, cnpy, and zlib development libraries.
- Python 3 (the sweep runner uses only the standard library), tmux, and GNU `/usr/bin/time`; the monitoring window uses `watch`.
- The archived full experiment used 32 threads and reached approximately 34.7 GiB peak process memory. Allow additional memory for the operating system. All 40 serial cases took approximately 35 hours 25 minutes; runtime varies by machine.

After installing OpenFHE and cnpy, specify their installation prefixes through `CMAKE_PREFIX_PATH`. The build also searches `$HOME/local/openFHE` and `$HOME/local/cnpy` by default. If shared libraries cannot be loaded, add their installation `lib` directories (or `lib64` on some systems) to `LD_LIBRARY_PATH`.

```bash
git clone --branch feature/sparse-conv2d https://github.com/Noreaaa/Openfhe_PE.git
cd Openfhe_PE
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/local/openFHE;$HOME/local/cnpy"
cmake --build build -j 4
```

If cnpy is not found automatically, pass `-DCNPY_INCLUDE_DIR=/path/include -DCNPY_LIB=/path/lib/libcnpy.so` explicitly.

## Models and input

ResNet parameters are stored in `python_model/saved_models/ResNet-18/parameters/`. The single image tensor used for the archived experiment is `datasets/imagenet-mini/ILSVRC2012_val_00000029.bin`, so reproducing this experiment does not require downloading the entire dataset. The full ImageNet-mini dataset is not included in this checkout.

The input is a **3×224×224 tensor of native float32 values in CHW order**, totaling 602112 bytes; it is not a JPEG. The current ResNet prefix always reads this file, so `--nums` does not make this experiment iterate over multiple images.

## Build and correctness checks

See the [minimal encrypted-ReLU measurements](benchmarks/relu_roundtrip_20261022/README.md) for operation latency, before/after peak memory, raw logs, and reproduction commands.

The synthetic tests in the standalone project do not require cnpy, model weights, or image files:

```bash
cmake -S tests/sparse_packing -B build-sparse -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/local/openFHE"
cmake --build build-sparse -j 4
ctest --test-dir build-sparse --output-on-failure
```

Run the independent plaintext-reference check on the real image (more expensive than the synthetic tests):

```bash
OMP_NUM_THREADS=4 ./build/sparse_real_accuracy "$PWD" test_results/accuracy-new
```

This checks the third convolution/BN and its ReLU after deriving the first two stages in plaintext. It does not validate every full-prefix sweep case. The optional `--verify` observer in the main executable prints per-prefix maximum absolute error and RMSE; its current implementation rejects non-finite outputs but does not enforce a numerical error threshold. The standalone correctness executables enforce a `1e-3` absolute-error tolerance.

## Run a single case

The following commands use a central 50×50 encrypted region and the first 10 convolutions with their ReLUs. Validate the configuration before running:

```bash
./build/test --model ResNet-18 --layers 10 --depth 46 \
  --ringDim 16384 --batch 1024 --act 0 --encrypt-size 50 \
  --data-root "$PWD" --check-config

mkdir -p test_results/single-new
OMP_NUM_THREADS=96 OMP_THREAD_LIMIT=96 OMP_DYNAMIC=FALSE \
  ./build/test --model ResNet-18 --layers 10 --depth 46 \
  --ringDim 16384 --batch 1024 --act 0 --encrypt-size 50 \
  --data-root "$PWD" --profile test_results/single-new/layers.csv \
  > test_results/single-new/run.log 2>&1
```

- `--layers`: execute the first 1–10 sequential convolutions, each followed by ReLU. These are convolution prefixes, not full ResNet-18 classification evaluations; classification accuracy cannot be inferred from them.
- `--depth`: initial multiplicative depth. All archived cases use 35 to keep the modulus chain fixed across layer-count comparisons. The current encrypted-mask path conservatively requires at least `max(20, 3*layers+14)`; automatic depth selection uses `max(23, 3*layers+16)`. The old depth budget does not cover reverse conversion and ciphertext multiplication.
- `--ringDim`: CKKS ring dimension; `--batch`: logical encoding slots.
- `--client-first-layer`: execute original convolution 1, batch normalization and ReLU in plaintext before encryption. Propagate the original sensitive rectangle through its receptive field and encrypt the affected feature rectangle. `--layers 8` then executes original convolutions 2–8 on the server; layer numbers remain unchanged. Requires scheme-switch ReLU and at least two total layers.
- `--act 0`: scheme-switch ReLU; `--act 1`: polynomial approximation, which should not be mixed into the archived experiment's results.
- `--encrypt-size`: side length of the central encrypted square. Alternatively, specify a rectangle using `--top/--bottom/--left/--right`; all bounds are inclusive.
- `--profile`: per-layer timing and layout CSV; `--verify`: additional per-prefix plaintext-reference comparisons that increase whole-process runtime.
- `--check-config`: checks the parameter budget without performing expensive inference. It does not validate model files or every runtime constraint.
- `--experimental-depth`: permits an explicitly requested non-bootstrap scheme-switch ResNet depth below the conservative margin, subject to the experimental lower bound. Use per-layer verification; the option does not guarantee numerical correctness or increase the requested depth.

The experimental parameters use `HEStd_NotSet` and FHEW `TOY`; they do not establish a production security level.

The client-first experiment runner is `python3 scripts/run_resnet_client_first.py`. It validates a two-layer prefix before running original layers 1–8 once for each central area from 10×10 to 80×80, using ring dimension 16384, 8192 slots, depth 40, 30/40-bit moduli and 96 threads. Results are written to `benchmarks/resnet_client_first_slots8192_20261024/`. Client computation and encryption are timed separately; `rss.csv` labels execution phases so that server sampling excludes key-generation and verification time windows. All phases share a process, so these RSS values are not isolated server-process measurements.

## Run a new 40-case sweep

The archived sweep used the former decrypted-sign ReLU, 30/40-bit scale/modulus settings, depth 35, and 32 threads. To reproduce that implementation, use source revision `08ec29d402e717cbc48131740244db64de1118c8` in a separate checkout and follow its README. The commands below exercise the new encrypted-mask path and will not reproduce the archived timings.

Use a **new output directory** to preserve the completed raw results:

```bash
bash scripts/start_resnet_sweep_tmux.sh \
  test_results/resnet_sweep-reproduction resnet_reproduction \
  --depth 46 --threads "$(nproc)" --max-area 80

tmux attach -t resnet_reproduction
# Inside tmux, press Ctrl-b followed by d to detach and leave the task running.
```

The launcher first builds `build/test`, then runs the combinations of square side lengths 10, 20, …, 80 and convolution counts 6, 7, 8, 9, 10 serially: **40 cases** in total. An earlier informal description called this “50 tests,” but the actual configuration contains 40. `--max-area 100` produces 50 cases; `--max-area 50` produces 25. Despite its name, `--max-area` specifies a square's side length, not its area.

Without tmux:

```bash
python3 -u scripts/run_resnet_sweep.py \
  --output test_results/resnet_sweep-reproduction \
  --depth 46 --threads "$(nproc)" --max-area 80
```

Inspect status:

```bash
python3 scripts/run_resnet_sweep.py --status \
  --output test_results/resnet_sweep-reproduction
```

After interruption, restart the runner with the same options and output directory. Successful cases with matching commands, binary fingerprints, and CPU affinity are skipped; interrupted cases run again. `--rerun` forces reruns. The tmux launcher does not replace an existing session of the same name; choose a new name if necessary. Binary or option changes can trigger recomputation, so do not direct new experiments into the archived result directory. Use `--dry-run` to print planned commands.

`--threads` limits OpenMP and binds child processes to the first N CPUs available to the runner; it does not reserve those CPUs exclusively. Choose a value no greater than the job's CPU allocation. The runner defaults are depth 46, all CPUs in its affinity mask, and maximum side length 100. Explicit `--threads` values remain supported by the runner through CPU affinity.

## Results and interpretation

These results predate the encrypted-mask change. The raw experiment is in [`test_results/resnet_sweep_t32_d35_max80/`](test_results/resnet_sweep_t32_d35_max80/), with 40/40 cases successful. See the [results and analysis](test_results/README.md) (currently in Chinese).

- `summary.csv`: inference, convolution, ReLU, whole-process runtime, and peak memory for each case.
- `configuration.json`: depth, threads, ring dimension, slots, CPU affinity, and binary SHA256.
- `areaNNN/layersNN/layers.csv`: per-layer convolution/ReLU times, cumulative times, encrypted region, ciphertext count, level, and remaining towers.
- Per-case `run.log`, `result.json`, and `time.txt`: full output, command, exit status, and GNU time measurements.
- `runner.log` and `live.json`: scheduling history and final state. Absolute paths in archived records refer to the original machine; locate files by their relative paths on another machine.

`*_ms` values are milliseconds, `*_seconds` values are seconds, and `peak_rss_gib` is measured in GiB. `inference_seconds` is the cumulative convolution+ReLU forward time; `wall_seconds` covers the entire test subprocess, including initialization.

Each configuration was run once. Machine-load variation between runs means that a longer prefix appearing faster does not demonstrate a speedup from adding layers. To compare cumulative work, prefer prefixes within the **same 10-layer run**. Formal performance comparisons should repeat measurements under stable resource conditions and report medians.

## Other tests and historical archives

- [`tests/sparse_packing/README.md`](tests/sparse_packing/README.md): standalone CKKS-to-FHEW microbenchmark for slot counts and output counts.
- `build/resnet_conv3_bench "$PWD" 16384`: microbenchmark using actual weights from the third convolution in the ResNet prefix.
- `tests/schemeswitch/`: original scheme-switching example, built as a separate CMake project.
- `artifacts/run_bigger.legacy.sh`: older central-region sweep script retained for reference. Its inclusive bounds actually produce regions of size+1, so use the new runner for this experiment.
- [`artifacts/legacy-results.tar.gz.parts/`](artifacts/legacy-results.tar.gz.parts/): split compressed archive (up to 45 MiB per part) containing other historical `test_results/`, `old_results/`, and root-level slot-experiment logs. It includes older configurations and failed trials; do not combine these with the archived 40-case sweep.

```bash
# Inspect the archive; extract into a new directory to avoid overwriting results.
cat artifacts/legacy-results.tar.gz.parts/part-* | tar -tzf -
mkdir -p recovered-results
cat artifacts/legacy-results.tar.gz.parts/part-* | tar -xzf - -C recovered-results
```

Build directories and generated build products are excluded by `.gitignore`. The experiment's source and results are retained; rebuild the executables on the target machine.
