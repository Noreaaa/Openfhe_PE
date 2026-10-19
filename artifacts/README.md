# 历史结果归档

`legacy-results.tar.gz.parts/part-000`～`part-002` 按名称顺序拼接后是一个 gzip tar 包，包含其他历史测试日志、准确性测试张量、`old_results/` 和根目录 slots 实验输出。它们可能包含失败试验和不同配置；本次正式 40 项结果保存在 `../test_results/resnet_sweep_t32_d35_max80/`。

从仓库根目录执行：

```bash
(cd artifacts && sha256sum -c SHA256SUMS)
cat artifacts/legacy-results.tar.gz.parts/part-* | tar -tzf -
mkdir -p recovered-results
cat artifacts/legacy-results.tar.gz.parts/part-* | tar -xzf - -C recovered-results
```

`environment.json` 是打包时采集的机器、编译器信息和本次输入 SHA256，不是实验运行期间的持续资源监控。原始实验参数与二进制指纹以结果目录的 `configuration.json` 为准。

`run_bigger.legacy.sh` 是旧脚本快照，参数和包含端点的边界计算与新 runner 不同，保留供追溯；重跑应使用主 README 中的命令。
