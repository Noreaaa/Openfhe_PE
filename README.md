# PFHE：部分加密 CNN 与 ResNet 前缀测试

使用 OpenFHE CKKS 对图像中的矩形区域加密，在明文/密文混合特征上执行卷积和 ReLU。本仓库包含实现、正确性测试、独立微基准、批量测试脚本和原始结果。

## 环境与依赖

- Linux、C++17 编译器、CMake >= 3.16、OpenMP。
- OpenFHE 1.2.4、cnpy、zlib 开发库。
- Python 3（批量脚本仅使用标准库）；tmux、GNU `/usr/bin/time`；监控窗口使用 `watch`。
- 本次完整实验使用 32 线程，最高进程峰值内存约 34.7 GiB；应预留额外系统内存。完整 40 项串行实验约耗时 35 小时 25 分钟，其他机器可能不同。

安装 OpenFHE 和 cnpy 后，通过 `CMAKE_PREFIX_PATH` 指定其安装目录。默认也搜索 `$HOME/local/openFHE` 和 `$HOME/local/cnpy`。动态库加载失败时，将安装目录的 `lib`（某些系统是 `lib64`）加入 `LD_LIBRARY_PATH`。

```bash
git clone --branch feature/sparse-conv2d https://github.com/Noreaaa/Openfhe_PE.git
cd Openfhe_PE
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/local/openFHE;$HOME/local/cnpy"
cmake --build build -j 4
```

如果 cnpy 未被自动找到，可显式传入 `-DCNPY_INCLUDE_DIR=/path/include -DCNPY_LIB=/path/lib/libcnpy.so`。

## 模型与输入

ResNet 参数在 `python_model/saved_models/ResNet-18/parameters/`。本次测试使用的单张输入保存在 `datasets/imagenet-mini/ILSVRC2012_val_00000029.bin`，无需下载整个数据集即可重跑此实验。完整 ImageNet-mini 数据集不包含在此提交中。

输入文件是 **3×224×224、CHW 排列的原生 float32**，共 602112 字节；它不是 JPEG。当前 ResNet 前缀路径固定读取上述文件，因此 `--nums` 不代表本实验会遍历多张图片。

## 先验证构建与正确性

独立测试项目中的合成数据测试不依赖 cnpy、模型权重或图片：

```bash
cmake -S tests/sparse_packing -B build-sparse -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/local/openFHE"
cmake --build build-sparse -j 4
ctest --test-dir build-sparse --output-on-failure
```

真实图片的独立明文参考检查（比合成测试更耗时）：

```bash
OMP_NUM_THREADS=4 ./build/sparse_real_accuracy "$PWD" test_results/accuracy-new
```

## 单项运行

以下命令运行中心 50×50 区域、前 10 个卷积及对应 ReLU。先做配置检查，然后运行：

```bash
./build/test --model ResNet-18 --layers 10 --depth 35 \
  --ringDim 16384 --batch 1024 --act 0 --encrypt-size 50 \
  --data-root "$PWD" --check-config

mkdir -p test_results/single-new
OMP_NUM_THREADS=32 OMP_THREAD_LIMIT=32 OMP_DYNAMIC=FALSE \
  ./build/test --model ResNet-18 --layers 10 --depth 35 \
  --ringDim 16384 --batch 1024 --act 0 --encrypt-size 50 \
  --data-root "$PWD" --profile test_results/single-new/layers.csv \
  > test_results/single-new/run.log 2>&1
```

- `--layers`：执行前 1～10 个顺序卷积，每个卷积后跟 ReLU。这里是卷积前缀，不是完整 ResNet-18 分类评估，也不能据此报告分类准确率。
- `--depth`：初始乘法深度；本次全部固定为 35，保证层数比较使用同一模数链。当前 scheme-switch ReLU 路径要求至少 `3*layers+4`。
- `--ringDim`：CKKS 环维数；`--batch`：逻辑编码 slots。
- `--act 0`：scheme-switch ReLU；`--act 1`：多项式近似路径，不能直接混作本次实验结果。
- `--encrypt-size`：输入中心加密正方形边长。也可用 `--top/--bottom/--left/--right` 显式指定矩形，边界均包含端点。
- `--profile`：逐层耗时和布局 CSV；`--verify`：增加逐层明文参考检查，会增加整进程耗时。
- `--check-config`：仅检查参数预算，不进行昂贵的推理。

当前实验参数使用 `HEStd_NotSet` 和 FHEW `TOY`，用于研究测试，不表示满足生产安全等级。

## 重跑本次 40 项实验

使用**新的输出目录**保留已经完成的原始结果：

```bash
bash scripts/start_resnet_sweep_tmux.sh \
  test_results/resnet_sweep-reproduction resnet_reproduction \
  --depth 35 --threads 32 --max-area 80

tmux attach -t resnet_reproduction
# 在 tmux 内按 Ctrl-b 再按 d，退出查看但保持任务运行。
```

脚本先构建 `build/test`，再串行运行边长 10、20、…、80 和层数 6、7、8、9、10 的组合，共 **40 项**。虽然之前口头称“50 个测试”，实际配置是 40 项。`--max-area 100` 才是 50 项。仅需到 50×50 时使用 `--max-area 50`（25 项）。

不使用 tmux：

```bash
python3 -u scripts/run_resnet_sweep.py \
  --output test_results/resnet_sweep-reproduction \
  --depth 35 --threads 32 --max-area 80
```

查看状态：

```bash
python3 scripts/run_resnet_sweep.py --status \
  --output test_results/resnet_sweep-reproduction
```

中断后，用相同参数和输出目录重启 runner：命令、二进制指纹和 CPU 绑定一致的成功项会跳过，中断项重跑。`--rerun` 强制重跑。tmux 启动器不会覆盖同名 session；仍存在时换一个 session 名称。二进制或参数改变可能导致重新计算，因此不要把新实验写入本次归档目录。`--dry-run` 可打印计划命令。

`--threads` 同时限制 OpenMP 并绑定当前允许使用的前 N 个 CPU；不代表独占这些 CPU。选择不超过作业分配的 CPU 数量。

## 结果在哪里、如何理解

原始实验在 [`test_results/resnet_sweep_t32_d35_max80/`](test_results/resnet_sweep_t32_d35_max80/)，40/40 成功。详见 [结果与分析](test_results/README.md)。

- `summary.csv`：整项推理、卷积、ReLU、整进程耗时与峰值内存。
- `configuration.json`：深度、线程数、环维数、slots、CPU 绑定和二进制 SHA256。
- `areaNNN/layersNN/layers.csv`：每层卷积/ReLU 耗时、累计耗时、加密区域、密文数、level 和剩余 towers。
- 每项的 `run.log`、`result.json`、`time.txt`：完整输出、命令、退出状态和 GNU time 数据。
- `runner.log`、`live.json`：调度过程与最终状态。历史记录中的绝对路径是原机器的溯源信息，在其他机器上请从此目录按相对路径找文件。

`*_ms` 单位为毫秒；`*_seconds` 为秒；`peak_rss_gib` 为 GiB。`inference_seconds` 是累计卷积+ReLU forward 计时，`wall_seconds` 是整个测试子进程耗时，包含初始化等开销。

每组合仅运行一次。跨次运行会有机器负载波动，不能把相邻层数的倒挂解释为增加层数能加速。比较累计计算量时，优先比较**同一次 10 层运行**内的前缀；正式性能比较应在稳定资源上重复测量并报告中位数。

## 其他测试及历史归档

- [`tests/sparse_packing/README.md`](tests/sparse_packing/README.md)：独立 CKKS→FHEW slots/输出数微基准。
- `build/resnet_conv3_bench "$PWD" 16384`：使用真实 ResNet 第三个卷积权重的微基准。
- `tests/schemeswitch/`：原有 scheme switching 示例，单独用 CMake 构建。
- `artifacts/run_bigger.legacy.sh`：旧版中心区域脚本，保留作历史参考；其包含端点的边界实际产生 size+1 的区域，不用于本次实验，请使用新的 runner。
- [`artifacts/legacy-results.tar.gz.parts/`](artifacts/legacy-results.tar.gz.parts/)：分卷压缩归档（每卷最多 45 MiB），包含其他历史 `test_results/`、`old_results/` 和根目录 slots 实验日志；包含旧配置和失败试验，不应与本次 40 项混合统计。

```bash
# 查看历史归档；解压到新目录以免覆盖已有结果。
cat artifacts/legacy-results.tar.gz.parts/part-* | tar -tzf -
mkdir -p recovered-results
cat artifacts/legacy-results.tar.gz.parts/part-* | tar -xzf - -C recovered-results
```

构建目录和编译产物由 `.gitignore` 排除。本次提交保留源码与结果，二进制需在目标环境重新构建。
