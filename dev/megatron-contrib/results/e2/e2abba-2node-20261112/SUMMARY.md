# E2 ABBA: 2-node benchmark, arms off / main-on / p1-on

## Conclusion

The large-T slowdown comes from P1, not from the reduce-side split or from run order. For train GA=4 step at T=1024 and T=4096, p1-on is 1.78% and 2.95% slower than main-on on d4096, and 0.40% and 2.17% slower on d8192. The reduce-side split (main-on vs off) changes that step by at most +0.63% at these sizes, and repeat differences for large-T step times are at most 0.29%, so drift does not explain the gap. The nostep control shows the same P1 effect: p1-on is 1.9 to 3.2% slower than main-on at T≥1024 on both models, while main-on stays within 0.7% of off. The extra time sits in the forward (fwd_rest, +6.2% to +8.5% for p1-on vs main-on), so P1 with the flag on slows the forward even when no outer all-gather runs in the timed steps; P1 lowers exposed time E, but not enough to offset this at large T. These runs do not identify the mechanism.

Runs in time order: 1-off, 2-main-on, 3-p1-on, 4-p1-on, 5-main-on, 6-off.

Arms:
- off: P1 code, overlap_outer_communication=False.
- main-on: main code, flag True. Reduce-side stream split only (#6973).
- p1-on: P1 code, flag True. Reduce-side split plus the DP-outer all-gather stream (P1).

Units: ms. E = JSON `max.exposed`. step and fwd_rest are JSON `max` values. Each `max` value is a per-rank median, then the max over 16 ranks.

## 1. Sanity

### 1a. Arm, code snapshot and meta flag

| run | arm | node0 snapshot | node1 snapshot | flag (d4096 / d8192) | status |
|---|---|---|---|---|---|
| 1-off | off | p1-20261112 | p1-20261112 | False / False | PASS |
| 2-main-on | main-on | main-d2b0e35cc | main-d2b0e35cc | True / True | PASS |
| 3-p1-on | p1-on | p1-20261112 | p1-20261112 | True / True | PASS |
| 4-p1-on | p1-on | p1-20261112 | p1-20261112 | True / True | PASS |
| 5-main-on | main-on | main-d2b0e35cc | main-d2b0e35cc | True / True | PASS |
| 6-off | off | p1-20261112 | p1-20261112 | False / False | PASS |

Expected: off -> p1 snapshot, flag False; main-on -> main snapshot, flag True; p1-on -> p1 snapshot, flag True.
Meta other than the flag identical across the six runs (e1_d4096_L32.json): PASS.
Meta other than the flag identical across the six runs (e1_d8192_L16.json): PASS.

Arm/flag/snapshot failures: 0.

### 1b. Loss bit-identity across the six runs

Compared per (model, mode, T, GA): raw_rank0.losses and per_rank.loss_sum (16 values), exact float equality against run 1-off.

Rows checked: 18. Rows bit-identical in all six runs: 18/18.

### 1c. Peak memory (JSON max.peak_mem_gb, GB)

| model | mode | T | GA | 1-off (off) | 2-main-on (main-on) | 3-p1-on (p1-on) | 4-p1-on (p1-on) | 5-main-on (main-on) | 6-off (off) | spread (GB) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e1_d4096_L32.json | nostep | 256 | 4 | 4.6192 | 4.6192 | 4.6192 | 4.6192 | 4.6192 | 4.6192 | 0.0000 | identical in all six runs |
| e1_d4096_L32.json | nostep | 1024 | 4 | 6.2891 | 6.2891 | 6.2891 | 6.2891 | 6.2891 | 6.2891 | 0.0000 | identical in all six runs |
| e1_d4096_L32.json | nostep | 4096 | 4 | 12.9688 | 12.9688 | 12.9688 | 12.9688 | 12.9688 | 12.9688 | 0.0000 | identical in all six runs |
| e1_d4096_L32.json | train | 256 | 1 | 4.6133 | 4.6133 | 4.6133 | 4.6133 | 4.6133 | 4.6133 | 0.0000 | identical in all six runs |
| e1_d4096_L32.json | train | 256 | 4 | 4.6192 | 4.6192 | 4.6192 | 4.6192 | 4.6192 | 4.6192 | 0.0000 | identical in all six runs |
| e1_d4096_L32.json | train | 1024 | 1 | 6.2656 | 6.2656 | 6.2656 | 6.2656 | 6.2656 | 6.2656 | 0.0000 | identical in all six runs |
| e1_d4096_L32.json | train | 1024 | 4 | 6.2891 | 6.2891 | 6.2891 | 6.2891 | 6.2891 | 6.2891 | 0.0000 | identical in all six runs |
| e1_d4096_L32.json | train | 4096 | 1 | 12.8750 | 12.8750 | 12.8750 | 12.8750 | 12.8750 | 12.8750 | 0.0000 | identical in all six runs |
| e1_d4096_L32.json | train | 4096 | 4 | 12.9688 | 12.9688 | 12.9688 | 12.9688 | 12.9688 | 12.9688 | 0.0000 | identical in all six runs |
| e1_d8192_L16.json | nostep | 256 | 4 | 10.6133 | 10.6133 | 10.6133 | 10.6133 | 10.6133 | 10.6133 | 0.0000 | identical in all six runs |
| e1_d8192_L16.json | nostep | 1024 | 4 | 12.2657 | 12.2657 | 12.2657 | 12.2657 | 12.2657 | 12.2657 | 0.0000 | identical in all six runs |
| e1_d8192_L16.json | nostep | 4096 | 4 | 18.8750 | 18.8750 | 18.8750 | 18.8750 | 18.8750 | 18.8750 | 0.0000 | identical in all six runs |
| e1_d8192_L16.json | train | 256 | 1 | 10.6016 | 10.6016 | 10.6016 | 10.6016 | 10.6016 | 10.6016 | 0.0000 | identical in all six runs |
| e1_d8192_L16.json | train | 256 | 4 | 10.6133 | 10.6133 | 10.6133 | 10.6133 | 10.6133 | 10.6133 | 0.0000 | identical in all six runs |
| e1_d8192_L16.json | train | 1024 | 1 | 12.2188 | 12.2188 | 12.2188 | 12.2188 | 12.2188 | 12.2188 | 0.0000 | identical in all six runs |
| e1_d8192_L16.json | train | 1024 | 4 | 12.2657 | 12.2657 | 12.2657 | 12.2657 | 12.2657 | 12.2657 | 0.0000 | identical in all six runs |
| e1_d8192_L16.json | train | 4096 | 1 | 18.6875 | 18.6875 | 18.6875 | 18.6875 | 18.6875 | 18.6875 | 0.0000 | identical in all six runs |
| e1_d8192_L16.json | train | 4096 | 4 | 18.8750 | 18.8750 | 18.8750 | 18.8750 | 18.8750 | 18.8750 | 0.0000 | identical in all six runs |

Rows with identical peak memory in all six runs: 18/18.

### 1d. Sweep logs vs JSON, and E recomputed from raw rank-0 data

| run | model | log rows | JSON rows | values compared | outside tolerance | max abs diff |
|---|---|---|---|---|---|---|
| 1-off | e1_d4096_L32.json | 9 | 9 | 48 | 0 | 0.0344 |
| 1-off | e1_d8192_L16.json | 9 | 9 | 48 | 0 | 0.0343 |
| 2-main-on | e1_d4096_L32.json | 9 | 9 | 48 | 0 | 0.0344 |
| 2-main-on | e1_d8192_L16.json | 9 | 9 | 48 | 0 | 0.0343 |
| 3-p1-on | e1_d4096_L32.json | 9 | 9 | 48 | 0 | 0.0344 |
| 3-p1-on | e1_d8192_L16.json | 9 | 9 | 48 | 0 | 0.0343 |
| 4-p1-on | e1_d4096_L32.json | 9 | 9 | 48 | 0 | 0.0344 |
| 4-p1-on | e1_d8192_L16.json | 9 | 9 | 48 | 0 | 0.0343 |
| 5-main-on | e1_d4096_L32.json | 9 | 9 | 48 | 0 | 0.0344 |
| 5-main-on | e1_d8192_L16.json | 9 | 9 | 48 | 0 | 0.0343 |
| 6-off | e1_d4096_L32.json | 9 | 9 | 48 | 0 | 0.0344 |
| 6-off | e1_d8192_L16.json | 9 | 9 | 48 | 0 | 0.0343 |

Log-vs-JSON mismatches: 0.

Recomputed E (median over steps of fwd[mb0] - median(fwd[mb1..])) vs per_rank.exposed[0], GA=4 rows:

| run | model | rows | max abs diff (ms) |
|---|---|---|---|
| 1-off | e1_d4096_L32.json | 6 | 0.00e+00 |
| 1-off | e1_d8192_L16.json | 6 | 0.00e+00 |
| 2-main-on | e1_d4096_L32.json | 6 | 0.00e+00 |
| 2-main-on | e1_d8192_L16.json | 6 | 0.00e+00 |
| 3-p1-on | e1_d4096_L32.json | 6 | 0.00e+00 |
| 3-p1-on | e1_d8192_L16.json | 6 | 0.00e+00 |
| 4-p1-on | e1_d4096_L32.json | 6 | 0.00e+00 |
| 4-p1-on | e1_d8192_L16.json | 6 | 0.00e+00 |
| 5-main-on | e1_d4096_L32.json | 6 | 0.00e+00 |
| 5-main-on | e1_d8192_L16.json | 6 | 0.00e+00 |
| 6-off | e1_d4096_L32.json | 6 | 0.00e+00 |
| 6-off | e1_d8192_L16.json | 6 | 0.00e+00 |

Largest recomputed-E difference: 0.00e+00 ms.

## 2. Drift: repeat-to-repeat difference per arm (noise floor)

Each cell: first value -> second value (% difference, (second - first) / first). Repeats: off 1-off -> 6-off, main-on 2-main-on -> 5-main-on, p1-on 3-p1-on -> 4-p1-on.

### e1_d4096_L32.json

| metric | T | off (1 -> 6) | main-on (2 -> 5) | p1-on (3 -> 4) |
|---|---|---|---|---|
| E (train GA=4) | 256 | 22.5 -> 23.0 (+2.12%) | 22.5 -> 24.4 (+8.72%) | 2.6 -> 2.4 (-9.56%) |
| E (train GA=4) | 1024 | 14.9 -> 16.2 (+8.13%) | 15.1 -> 16.0 (+5.37%) | 6.4 -> 6.2 (-3.38%) |
| E (train GA=4) | 4096 | 16.9 -> 16.4 (-3.13%) | 17.6 -> 17.4 (-1.31%) | 12.7 -> 12.2 (-3.95%) |
| fwd_rest (train GA=4) | 256 | 33.0 -> 32.9 (-0.13%) | 32.9 -> 32.9 (-0.08%) | 32.7 -> 32.7 (-0.02%) |
| fwd_rest (train GA=4) | 1024 | 78.6 -> 78.6 (-0.10%) | 78.6 -> 78.6 (+0.00%) | 84.5 -> 84.7 (+0.19%) |
| fwd_rest (train GA=4) | 4096 | 269.0 -> 269.2 (+0.10%) | 269.3 -> 269.1 (-0.06%) | 292.3 -> 291.8 (-0.19%) |
| step (train GA=1) | 256 | 133.8 -> 134.6 (+0.56%) | 125.9 -> 125.7 (-0.19%) | 105.7 -> 105.9 (+0.20%) |
| step (train GA=1) | 1024 | 253.3 -> 253.1 (-0.08%) | 254.3 -> 254.9 (+0.23%) | 249.8 -> 249.8 (-0.03%) |
| step (train GA=1) | 4096 | 830.8 -> 829.3 (-0.18%) | 839.7 -> 837.9 (-0.21%) | 846.7 -> 846.3 (-0.04%) |
| step (train GA=4) | 256 | 430.3 -> 430.1 (-0.04%) | 419.3 -> 420.9 (+0.38%) | 400.6 -> 400.3 (-0.09%) |
| step (train GA=4) | 1024 | 953.3 -> 953.3 (-0.00%) | 953.8 -> 954.6 (+0.08%) | 971.2 -> 971.1 (-0.01%) |
| step (train GA=4) | 4096 | 3251.8 -> 3251.5 (-0.01%) | 3255.1 -> 3254.6 (-0.01%) | 3352.2 -> 3349.3 (-0.09%) |
| step (nostep GA=4) | 256 | 406.0 -> 405.1 (-0.22%) | 394.5 -> 394.8 (+0.09%) | 396.2 -> 396.1 (-0.02%) |
| step (nostep GA=4) | 1024 | 937.6 -> 935.4 (-0.24%) | 936.6 -> 936.6 (+0.00%) | 963.4 -> 963.3 (-0.02%) |
| step (nostep GA=4) | 4096 | 3238.9 -> 3237.7 (-0.04%) | 3238.4 -> 3239.1 (+0.02%) | 3342.4 -> 3341.7 (-0.02%) |

### e1_d8192_L16.json

| metric | T | off (1 -> 6) | main-on (2 -> 5) | p1-on (3 -> 4) |
|---|---|---|---|---|
| E (train GA=4) | 256 | 43.6 -> 43.5 (-0.28%) | 43.8 -> 44.2 (+1.07%) | 6.2 -> 5.5 (-11.11%) |
| E (train GA=4) | 1024 | 36.4 -> 34.9 (-4.10%) | 34.8 -> 34.1 (-2.13%) | 13.4 -> 13.2 (-1.52%) |
| E (train GA=4) | 4096 | 30.8 -> 28.6 (-7.37%) | 28.9 -> 29.3 (+1.41%) | 21.0 -> 21.4 (+1.95%) |
| fwd_rest (train GA=4) | 256 | 62.5 -> 62.4 (-0.10%) | 62.5 -> 62.5 (+0.02%) | 62.4 -> 62.4 (-0.05%) |
| fwd_rest (train GA=4) | 1024 | 147.8 -> 148.1 (+0.23%) | 148.2 -> 148.0 (-0.19%) | 157.1 -> 157.4 (+0.17%) |
| fwd_rest (train GA=4) | 4096 | 533.7 -> 534.0 (+0.06%) | 534.1 -> 533.9 (-0.03%) | 570.1 -> 570.7 (+0.11%) |
| step (train GA=1) | 256 | 255.8 -> 255.8 (+0.01%) | 237.6 -> 238.6 (+0.42%) | 199.2 -> 199.4 (+0.09%) |
| step (train GA=1) | 1024 | 528.9 -> 528.0 (-0.16%) | 508.9 -> 509.5 (+0.11%) | 477.2 -> 479.6 (+0.50%) |
| step (train GA=1) | 4096 | 1674.9 -> 1666.7 (-0.49%) | 1673.4 -> 1676.3 (+0.17%) | 1661.1 -> 1662.4 (+0.08%) |
| step (train GA=4) | 256 | 801.5 -> 800.8 (-0.09%) | 783.8 -> 784.0 (+0.03%) | 750.6 -> 749.6 (-0.13%) |
| step (train GA=4) | 1024 | 1833.0 -> 1831.2 (-0.09%) | 1845.9 -> 1841.5 (-0.24%) | 1850.9 -> 1851.3 (+0.02%) |
| step (train GA=4) | 4096 | 6469.9 -> 6451.3 (-0.29%) | 6450.5 -> 6458.6 (+0.13%) | 6595.1 -> 6594.4 (-0.01%) |
| step (nostep GA=4) | 256 | 751.8 -> 751.7 (-0.01%) | 734.8 -> 734.5 (-0.04%) | 738.6 -> 738.5 (-0.01%) |
| step (nostep GA=4) | 1024 | 1786.7 -> 1784.6 (-0.12%) | 1797.7 -> 1798.7 (+0.06%) | 1831.2 -> 1832.3 (+0.06%) |
| step (nostep GA=4) | 4096 | 6405.0 -> 6400.0 (-0.08%) | 6389.3 -> 6376.3 (-0.20%) | 6548.3 -> 6547.2 (-0.02%) |

Noise floor summary: absolute repeat difference (%) per arm.

| metric | arm | cells | median abs | max abs |
|---|---|---|---|---|
| E (train GA=4) | off | 6 | 3.61% | 8.13% |
| E (train GA=4) | main-on | 6 | 1.77% | 8.72% |
| E (train GA=4) | p1-on | 6 | 3.66% | 11.11% |
| fwd_rest (train GA=4) | off | 6 | 0.10% | 0.23% |
| fwd_rest (train GA=4) | main-on | 6 | 0.05% | 0.19% |
| fwd_rest (train GA=4) | p1-on | 6 | 0.14% | 0.19% |
| step (train GA=1) | off | 6 | 0.17% | 0.56% |
| step (train GA=1) | main-on | 6 | 0.20% | 0.42% |
| step (train GA=1) | p1-on | 6 | 0.09% | 0.50% |
| step (train GA=4) | off | 6 | 0.06% | 0.29% |
| step (train GA=4) | main-on | 6 | 0.10% | 0.38% |
| step (train GA=4) | p1-on | 6 | 0.05% | 0.13% |
| step (nostep GA=4) | off | 6 | 0.10% | 0.24% |
| step (nostep GA=4) | main-on | 6 | 0.05% | 0.20% |
| step (nostep GA=4) | p1-on | 6 | 0.02% | 0.06% |

All step metrics (GA=1, GA=4, nostep GA=4; both models; all T; all arms): median abs 0.08%, max abs 0.56%.

## 3. Main table (mean of the two repeats per arm, ms)

Delta = (b - a) / a, positive = slower. A delta is marked (within noise) when its absolute value is below the larger of the two arms' repeat differences for the same row (section 2). Each repeat difference is a single pair, so each threshold is a rough estimate.

### e1_d4096_L32.json

| T | metric | off | main-on | p1-on | main-on vs off | p1-on vs main-on (P1) | p1-on vs off (total) |
|---|---|---|---|---|---|---|---|
| 256 | E (train GA=4) | 22.7 | 23.5 | 2.5 | +3.27% (within noise) | -89.42% | -89.07% |
| 256 | fwd_rest (train GA=4) | 33.0 | 32.9 | 32.7 | -0.13% (within noise) | -0.77% | -0.90% |
| 256 | step (train GA=1) | 134.2 | 125.8 | 105.8 | -6.23% | -15.89% | -21.13% |
| 256 | step (train GA=4) | 430.2 | 420.1 | 400.4 | -2.35% | -4.68% | -6.92% |
| 256 | step (nostep GA=4) | 405.6 | 394.6 | 396.2 | -2.69% | +0.38% | -2.32% |
| 1024 | E (train GA=4) | 15.5 | 15.5 | 6.3 | +0.03% (within noise) | -59.54% | -59.53% |
| 1024 | fwd_rest (train GA=4) | 78.6 | 78.6 | 84.6 | -0.00% (within noise) | +7.58% | +7.58% |
| 1024 | step (train GA=1) | 253.2 | 254.6 | 249.8 | +0.55% | -1.88% | -1.34% |
| 1024 | step (train GA=4) | 953.3 | 954.2 | 971.1 | +0.09% | +1.78% | +1.87% |
| 1024 | step (nostep GA=4) | 936.5 | 936.6 | 963.4 | +0.02% (within noise) | +2.85% | +2.87% |
| 4096 | E (train GA=4) | 16.7 | 17.5 | 12.5 | +5.15% | -28.85% | -25.19% |
| 4096 | fwd_rest (train GA=4) | 269.1 | 269.2 | 292.0 | +0.05% (within noise) | +8.48% | +8.53% |
| 4096 | step (train GA=1) | 830.1 | 838.8 | 846.5 | +1.05% | +0.91% | +1.97% |
| 4096 | step (train GA=4) | 3251.7 | 3254.8 | 3350.8 | +0.10% | +2.95% | +3.05% |
| 4096 | step (nostep GA=4) | 3238.3 | 3238.7 | 3342.0 | +0.01% (within noise) | +3.19% | +3.20% |

### e1_d8192_L16.json

| T | metric | off | main-on | p1-on | main-on vs off | p1-on vs main-on (P1) | p1-on vs off (total) |
|---|---|---|---|---|---|---|---|
| 256 | E (train GA=4) | 43.5 | 44.0 | 5.8 | +1.12% | -86.71% | -86.56% |
| 256 | fwd_rest (train GA=4) | 62.4 | 62.5 | 62.4 | +0.05% (within noise) | -0.09% | -0.04% (within noise) |
| 256 | step (train GA=1) | 255.8 | 238.1 | 199.3 | -6.92% | -16.28% | -22.08% |
| 256 | step (train GA=4) | 801.1 | 783.9 | 750.1 | -2.15% | -4.31% | -6.36% |
| 256 | step (nostep GA=4) | 751.7 | 734.7 | 738.5 | -2.27% | +0.52% | -1.76% |
| 1024 | E (train GA=4) | 35.7 | 34.5 | 13.3 | -3.47% (within noise) | -61.28% | -62.62% |
| 1024 | fwd_rest (train GA=4) | 147.9 | 148.1 | 157.2 | +0.13% (within noise) | +6.17% | +6.30% |
| 1024 | step (train GA=1) | 528.4 | 509.2 | 478.4 | -3.64% | -6.05% | -9.47% |
| 1024 | step (train GA=4) | 1832.1 | 1843.7 | 1851.1 | +0.63% | +0.40% | +1.04% |
| 1024 | step (nostep GA=4) | 1785.6 | 1798.2 | 1831.8 | +0.70% | +1.86% | +2.58% |
| 4096 | E (train GA=4) | 29.7 | 29.1 | 21.2 | -2.02% (within noise) | -27.07% | -28.54% |
| 4096 | fwd_rest (train GA=4) | 533.8 | 534.0 | 570.4 | +0.03% (within noise) | +6.82% | +6.85% |
| 4096 | step (train GA=1) | 1670.8 | 1674.9 | 1661.8 | +0.24% (within noise) | -0.78% | -0.54% |
| 4096 | step (train GA=4) | 6460.6 | 6454.5 | 6594.7 | -0.09% (within noise) | +2.17% | +2.08% |
| 4096 | step (nostep GA=4) | 6402.5 | 6382.8 | 6547.8 | -0.31% | +2.58% | +2.27% |

## 4. Attribution

Step-time deltas per T. Verdict per cell: slower / faster only when the delta is beyond the noise threshold.

### e1_d4096_L32.json

| T | metric | main-on vs off (reduce-side split) | p1-on vs main-on (P1) | p1-on vs off (total) |
|---|---|---|---|---|
| 256 | step (train GA=1) | -6.23% (faster) | -15.89% (faster) | -21.13% (faster) |
| 256 | step (train GA=4) | -2.35% (faster) | -4.68% (faster) | -6.92% (faster) |
| 256 | step (nostep GA=4) | -2.69% (faster) | +0.38% (slower) | -2.32% (faster) |
| 1024 | step (train GA=1) | +0.55% (slower) | -1.88% (faster) | -1.34% (faster) |
| 1024 | step (train GA=4) | +0.09% (slower) | +1.78% (slower) | +1.87% (slower) |
| 1024 | step (nostep GA=4) | +0.02% (within noise) | +2.85% (slower) | +2.87% (slower) |
| 4096 | step (train GA=1) | +1.05% (slower) | +0.91% (slower) | +1.97% (slower) |
| 4096 | step (train GA=4) | +0.10% (slower) | +2.95% (slower) | +3.05% (slower) |
| 4096 | step (nostep GA=4) | +0.01% (within noise) | +3.19% (slower) | +3.20% (slower) |

### e1_d8192_L16.json

| T | metric | main-on vs off (reduce-side split) | p1-on vs main-on (P1) | p1-on vs off (total) |
|---|---|---|---|---|
| 256 | step (train GA=1) | -6.92% (faster) | -16.28% (faster) | -22.08% (faster) |
| 256 | step (train GA=4) | -2.15% (faster) | -4.31% (faster) | -6.36% (faster) |
| 256 | step (nostep GA=4) | -2.27% (faster) | +0.52% (slower) | -1.76% (faster) |
| 1024 | step (train GA=1) | -3.64% (faster) | -6.05% (faster) | -9.47% (faster) |
| 1024 | step (train GA=4) | +0.63% (slower) | +0.40% (slower) | +1.04% (slower) |
| 1024 | step (nostep GA=4) | +0.70% (slower) | +1.86% (slower) | +2.58% (slower) |
| 4096 | step (train GA=1) | +0.24% (within noise) | -0.78% (faster) | -0.54% (faster) |
| 4096 | step (train GA=4) | -0.09% (within noise) | +2.17% (slower) | +2.08% (slower) |
| 4096 | step (nostep GA=4) | -0.31% (faster) | +2.58% (slower) | +2.27% (slower) |

Large-T attribution (T=1024 and T=4096 both required):

- e1_d4096_L32.json, step (train GA=1), reduce-side split (main-on vs off): slower at both large T. T=1024 +0.55% (noise 0.23%); T=4096 +1.05% (noise 0.21%).
- e1_d4096_L32.json, step (train GA=1), P1 (p1-on vs main-on): not slower at both large T. T=1024 -1.88% (noise 0.23%); T=4096 +0.91% (noise 0.21%).
- e1_d4096_L32.json, step (train GA=4), reduce-side split (main-on vs off): slower at both large T. T=1024 +0.09% (noise 0.08%); T=4096 +0.10% (noise 0.01%).
- e1_d4096_L32.json, step (train GA=4), P1 (p1-on vs main-on): slower at both large T. T=1024 +1.78% (noise 0.08%); T=4096 +2.95% (noise 0.09%).
- e1_d4096_L32.json, step (nostep GA=4), reduce-side split (main-on vs off): not slower at both large T. T=1024 +0.02% (noise 0.24%); T=4096 +0.01% (noise 0.04%).
- e1_d4096_L32.json, step (nostep GA=4), P1 (p1-on vs main-on): slower at both large T. T=1024 +2.85% (noise 0.02%); T=4096 +3.19% (noise 0.02%).
- e1_d8192_L16.json, step (train GA=1), reduce-side split (main-on vs off): not slower at both large T. T=1024 -3.64% (noise 0.16%); T=4096 +0.24% (noise 0.49%).
- e1_d8192_L16.json, step (train GA=1), P1 (p1-on vs main-on): faster at both large T. T=1024 -6.05% (noise 0.50%); T=4096 -0.78% (noise 0.17%).
- e1_d8192_L16.json, step (train GA=4), reduce-side split (main-on vs off): not slower at both large T. T=1024 +0.63% (noise 0.24%); T=4096 -0.09% (noise 0.29%).
- e1_d8192_L16.json, step (train GA=4), P1 (p1-on vs main-on): slower at both large T. T=1024 +0.40% (noise 0.24%); T=4096 +2.17% (noise 0.13%).
- e1_d8192_L16.json, step (nostep GA=4), reduce-side split (main-on vs off): not slower at both large T. T=1024 +0.70% (noise 0.12%); T=4096 -0.31% (noise 0.20%).
- e1_d8192_L16.json, step (nostep GA=4), P1 (p1-on vs main-on): slower at both large T. T=1024 +1.86% (noise 0.06%); T=4096 +2.58% (noise 0.20%).

### 4b. fwd_rest, train and nostep, GA=4 (mean of the two repeats, ms)

fwd_rest is the forward time of microbatches 1..3 on rank 0's max. Delta = (b - a) / a. Threshold = larger repeat difference of the two arms for the same row.

#### e1_d4096_L32.json

| T | mode | off | main-on | p1-on | main-on vs off | p1-on vs main-on | p1-on vs off |
|---|---|---|---|---|---|---|---|
| 256 | train | 33.0 | 32.9 | 32.7 | -0.13% (within noise) | -0.77% | -0.90% |
| 256 | nostep | 33.0 | 32.9 | 32.6 | -0.17% (within noise) | -0.95% | -1.12% |
| 1024 | train | 78.6 | 78.6 | 84.6 | -0.00% (within noise) | +7.58% | +7.58% |
| 1024 | nostep | 78.4 | 78.5 | 84.3 | +0.08% (within noise) | +7.39% | +7.48% |
| 4096 | train | 269.1 | 269.2 | 292.0 | +0.05% (within noise) | +8.48% | +8.53% |
| 4096 | nostep | 269.2 | 269.4 | 291.9 | +0.05% | +8.36% | +8.41% |

#### e1_d8192_L16.json

| T | mode | off | main-on | p1-on | main-on vs off | p1-on vs main-on | p1-on vs off |
|---|---|---|---|---|---|---|---|
| 256 | train | 62.4 | 62.5 | 62.4 | +0.05% (within noise) | -0.09% | -0.04% (within noise) |
| 256 | nostep | 62.4 | 62.4 | 62.4 | -0.02% (within noise) | -0.05% (within noise) | -0.07% (within noise) |
| 1024 | train | 147.9 | 148.1 | 157.2 | +0.13% (within noise) | +6.17% | +6.30% |
| 1024 | nostep | 147.3 | 147.5 | 156.8 | +0.10% | +6.31% | +6.42% |
| 4096 | train | 533.8 | 534.0 | 570.4 | +0.03% (within noise) | +6.82% | +6.85% |
| 4096 | nostep | 530.3 | 529.7 | 568.7 | -0.11% (within noise) | +7.35% | +7.24% |

## 5. Per-step variance on rank 0

train GA=4 step time from raw_rank0.step (8 measured steps). Cell: median / IQR, ms.

### e1_d4096_L32.json

| run | arm | T=256 | T=1024 | T=4096 |
|---|---|---|---|---|
| 1-off | off | 430.3 / 2.00 | 953.3 / 0.63 | 3251.6 / 2.94 |
| 2-main-on | main-on | 419.2 / 0.69 | 953.7 / 2.83 | 3254.9 / 1.38 |
| 3-p1-on | p1-on | 400.5 / 1.02 | 971.0 / 1.52 | 3351.9 / 2.15 |
| 4-p1-on | p1-on | 400.2 / 0.41 | 971.1 / 1.62 | 3349.2 / 2.55 |
| 5-main-on | main-on | 420.8 / 2.95 | 954.5 / 1.54 | 3254.4 / 3.52 |
| 6-off | off | 430.1 / 1.23 | 953.0 / 1.73 | 3251.3 / 1.81 |

### e1_d8192_L16.json

| run | arm | T=256 | T=1024 | T=4096 |
|---|---|---|---|---|
| 1-off | off | 801.4 / 3.06 | 1832.8 / 4.65 | 6469.8 / 20.22 |
| 2-main-on | main-on | 783.7 / 1.74 | 1845.7 / 2.23 | 6450.3 / 15.80 |
| 3-p1-on | p1-on | 750.5 / 5.55 | 1850.7 / 3.03 | 6594.7 / 3.59 |
| 4-p1-on | p1-on | 749.5 / 2.59 | 1851.2 / 4.14 | 6594.3 / 6.49 |
| 5-main-on | main-on | 783.9 / 2.22 | 1841.4 / 5.32 | 6458.5 / 17.74 |
| 6-off | off | 800.6 / 1.28 | 1831.1 / 1.05 | 6451.3 / 14.25 |

