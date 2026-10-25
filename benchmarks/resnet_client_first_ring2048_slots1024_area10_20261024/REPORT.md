# 10x10 client-first memory experiment

Ring dimension 2048, 1024 slots, depth 40, 30/40-bit moduli, HYBRID digits 6, 96 threads. Client computes original Conv1+BN+ReLU; server executes original layers 2–8 once. No bootstrap. All seven server layer numerical checks passed.

Server forward: 405.811 s. Client + encryption + forward: 406.477 s. Whole-process wall time: 411.140 s.

Operator-phase sampled RSS mean/peak: 2.032 / 2.920 GiB. Whole-process peak: 2.948 GiB.

RSS phase sampling excludes setup/verification time windows but shares their process and resident allocations. One repetition; no network measured. These experimental security settings do not establish 128-bit security.

| Original layer | Conv+BN (s) | ReLU (s) | Ciphertexts |
|---:|---:|---:|---:|
| 2 | 7.331 | 23.595 | 20 |
| 3 | 3.731 | 37.039 | 28 |
| 4 | 3.993 | 48.124 | 36 |
| 5 | 4.927 | 66.459 | 44 |
| 6 | 4.960 | 85.129 | 52 |
| 7 | 6.206 | 44.906 | 28 |
| 8 | 4.562 | 64.850 | 36 |
