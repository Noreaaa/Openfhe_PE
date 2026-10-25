# ResNet repeated prefix benchmark

Updated UTC: 2026-10-22T19:31:09.046040+00:00

Completed successful cases: 0/24; recorded cases: 0.

Configuration: 30/40 bits, depth 40, HYBRID digits 6, 1024 logical slots, 96 OpenMP threads, no CKKS bootstrap. Center square sides 10..80 step 10; target 8 convolutions; three repeats. Each case uses a fresh process and fresh keys. Plaintext verification enabled in every case.

Whole-process wall time includes setup, encryption and verification. Forward time sums Conv+BN and ReLU timers and excludes the plaintext observer. Peak RSS includes setup and verification. Per-layer peak is cumulative process high-water RSS, not isolated layer working memory.

The prefix is sequential Conv+BN+ReLU without residual additions/downsample branches. Layer 2 replaces maxpool with convolution. Layers 6/7/8 map to layer1.1.conv2 / layer2.0.conv1 / layer2.0.conv2. This experiment does not establish full ResNet classification accuracy or 128-bit security.

Ring dimensions are summarized separately. Memory-pressure fallback triggers below 12 GiB host available RAM, on std::bad_alloc, or an OOM kill; each affected case retries with ring 16384. Failed and partial attempts remain available.

## Independent runs (means over repeats)

| Area | Target | Ring | n | Wall mean (s) | Forward mean (s) | Conv mean (s) | ReLU mean (s) | Peak mean / max (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Pooled prefix timings

At completion, per area and absent fallback/failure: every prefix from 1 through 8 has 3 samples. Prefix samples from one process are correlated; CSV also provides target-stratified means.

| Area | Prefix | Ring | n | Mean (s) | Std (s) | Min (s) | Max (s) | Peak max (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Per-layer forward means

| Area | Layer | Ring | n | Conv+BN mean (s) | ReLU mean (s) | Layer mean (s) | Layer std (s) |
|---|---:|---:|---:|---:|---:|---:|---:|

## Individual runs

| Repeat | Area | Target | Ring | Status | Process wall (s) | Forward (s) | Peak RSS (GiB) |
|---:|---|---:|---:|---|---:|---:|---:|

Sum of completed attempt process wall times (including failed/fallback attempts): 0.000 s.
Sum of successful forward times: 0.000 s.
