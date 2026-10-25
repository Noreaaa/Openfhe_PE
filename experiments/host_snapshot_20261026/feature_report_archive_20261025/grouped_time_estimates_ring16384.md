# Estimated grouped convolution time: ring dimension 16384

Method: fully encrypted channel-wise convolution. Ring 16384, slots 8192, depth 30, 30/40-bit moduli, 96 threads. Each representative convolution was measured independently with a fresh level-0 input.

For each requested group, multiply its representative-layer Conv+BN time by four. These are estimates, not measured four-layer executions. ReLU, bootstrapping, key generation, encryption and communication are excluded. Transition layers 7, 11 and 15 change channels and resolution, so they are not identical to the chosen same-channel stride-1 representatives. Actual sequential execution also changes ciphertext levels.

| Group | Representative layer | Measured Conv+BN (s) | Multiplier | Estimated group time (s) | Estimated group time (min) |
|---|---:|---:|---:|---:|---:|
| 7–10 | 8 | 199.769 | 4 | 799.075 | 13.318 |
| 11–14 | 12 | 733.350 | 4 | 2933.399 | 48.890 |
| 15–18 | 16 | 2916.835 | 4 | 11667.338 | 194.456 |

Total estimate for groups 7–18: 15399.813 s = 256.664 min.

For comparison, the completed ring-16384 approximation-plus-bootstrap layer-8 measurement was 353.286403537 s for Conv+BN + polynomial + bootstrap; multiplying by four estimates group 7–10 at 1413.145614148 s (23.552426902 min). Corresponding completed measurements for representative layers 12 and 16 are unavailable; they are not filled from convolution-only timings.
