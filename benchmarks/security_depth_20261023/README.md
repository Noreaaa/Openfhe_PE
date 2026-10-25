# CKKS depth limits at classical 128-bit security — 2026-10-23

Measured using `GenCryptoContext` in OpenFHE 1.2.4, `HEStd_128_classic`, UNIFORM_TERNARY, FLEXIBLEAUTO, HYBRID, `numLargeDigits=3`, and 1024 logical slots. No keys or inference were generated. Each candidate uses `multiplicativeDepth=depth`, `scalingModSize=scale`, and `firstModSize=scale+10`.

| Ring dimension | 30/40 bits max depth | 40/50 bits max depth | 50/60 bits max depth |
|---:|---:|---:|---:|
| 16384 | 8 | 5 | 5 |
| 32768 | 20 | 14 | 11 |
| 65536 | 40 | 30 | 24 |

For the current ring dimension 16384 and 40/50-bit moduli, **maximum depth is 5**, and depth 6 is rejected as requiring ring dimension 32768. The ternary-secret standard table permits 438 bits. HYBRID security uses Q*P, not Q alone. At depth 5: Q approximately 250 bits, P 120 bits, QP approximately 370 bits. At depth 6, the three-digit partition requires three Q towers per digit; the largest digit is 50+40+40=130 bits, requiring three 60-bit auxiliary primes (P=180 bits). Q approximately 290 bits and QP approximately 470 bits exceed 438. Parameter generation adds a conservative extra bit in its bound, yielding 371 and 471 bits, respectively.

These are parameter-generation limits for the specified configuration, not counts of CNN layers. Changing numLargeDigits, scale bits, key-switch technique, or ring dimension changes the result. For example, at 40/50 bits depth 14 is accepted with ring dimension 32768; depth 34 exceeds even the ring-65536 limit of 30.

The CNN currently uses HEStd_NotSet and FHEW TOY. This probe overrides security only within its own standalone contexts. The CKKS result does not certify the complete CKKS/FHEW pipeline at 128 bits: the FHEW parameter set must also be replaced with an appropriate security configuration.

Source: `probe.cpp`; raw accept/reject records: `run.log`; structured boundary results: `summary.json`. Some very small depths fail the three-digit partition constraint independently of security, so the probe skips those errors and stops only at the first security rejection.
