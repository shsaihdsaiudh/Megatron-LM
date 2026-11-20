# E2 diagnostic summary

All 10 runs passed the checks: each run's code snapshot, CUDA_DEVICE_MAX_CONNECTIONS setting and flag matched its arm, losses were bit-identical in every run for all four rows, and peak memory was identical. An extra unused stream (main-dummy) changes no forward time beyond noise (fwd_rest -0.13% at T=1024 and -0.01% at T=4096, train). P1 (p1-on) raises fwd_rest by +8.6% at T=1024 and +10.2% at T=4096 in train rows, and by +8.7% and +10.1% in nostep rows, with repeat differences under 0.3%, so the slowdown is not run-to-run noise. CUDA_DEVICE_MAX_CONNECTIONS=32 does not remove it (p1-on-c32 fwd_rest +7.5% and +8.4% in train). This set has no flag-off arm, so every delta here is against main-on, and the flag itself is not tested.

---

# E2 diagnostic: 10 runs, 5 arms, d4096 L32, 2 nodes x 8 GPUs (16 ranks)

Arms: main-on (main code, flag on); main-dummy (main plus one never-used stream when the flag is on); p1-on (P1 code, flag on); main-on-c32 and p1-on-c32 (same code as main-on and p1-on, with CUDA_DEVICE_MAX_CONNECTIONS=32). Each arm has two runs.

Metrics (JSON `max` values, each a per-rank median, then the max over 16 ranks): fwd_mb0, E (exposed), fwd_rest (microbatches 1..3 forward), bwd_rest (microbatches 1..3 backward), step. Units: ms.

## 1. Checks

### 1a. Snapshot, sweep header and meta flag per run

| run | arm | node0 snapshot | node1 snapshot | CUDA_DEVICE_MAX_CONNECTIONS (sweep header) | meta flag | status |
|---|---|---|---|---|---|---|
| 01-main-on | main-on | main-d2b0e35cc | main-d2b0e35cc | None | True | PASS |
| 02-main-dummy | main-dummy | main-dummystream-d2b0e35cc | main-dummystream-d2b0e35cc | None | True | PASS |
| 03-p1-on | p1-on | p1-20261112 | p1-20261112 | None | True | PASS |
| 04-main-on-c32 | main-on-c32 | main-d2b0e35cc | main-d2b0e35cc | 32 | True | PASS |
| 05-p1-on-c32 | p1-on-c32 | p1-20261112 | p1-20261112 | 32 | True | PASS |
| 06-p1-on-c32 | p1-on-c32 | p1-20261112 | p1-20261112 | 32 | True | PASS |
| 07-main-on-c32 | main-on-c32 | main-d2b0e35cc | main-d2b0e35cc | 32 | True | PASS |
| 08-p1-on | p1-on | p1-20261112 | p1-20261112 | None | True | PASS |
| 09-main-dummy | main-dummy | main-dummystream-d2b0e35cc | main-dummystream-d2b0e35cc | None | True | PASS |
| 10-main-on | main-on | main-d2b0e35cc | main-d2b0e35cc | None | True | PASS |

Expected snapshot and CUDA setting per arm: main-on: main-d2b0e35cc, None; main-dummy: main-dummystream-d2b0e35cc, None; p1-on: p1-20261112, None; main-on-c32: main-d2b0e35cc, 32; p1-on-c32: p1-20261112, 32.

Snapshot/header/flag failures: 0.

### 1b. Loss bit-identity across all 10 runs

Per row: raw_rank0.losses and per_rank.loss_sum (16 values) compared by exact equality with run 01-main-on.

| mode | T | GA | runs identical to 01-main-on | differing runs |
|---|---|---|---|---|
| train | 1024 | 4 | 10/10 | none |
| train | 4096 | 4 | 10/10 | none |
| nostep | 1024 | 4 | 10/10 | none |
| nostep | 4096 | 4 | 10/10 | none |

Rows that differ from run 01-main-on in any run: 0 run-row pairs.

### 1c. Peak memory per row (max.peak_mem_gb, GB)

| mode | T | GA | min | max | spread | verdict |
|---|---|---|---|---|---|---|
| train | 1024 | 4 | 6.2891 | 6.2891 | 0.0000 | equal in all 10 runs |
| train | 4096 | 4 | 12.9688 | 12.9688 | 0.0000 | equal in all 10 runs |
| nostep | 1024 | 4 | 6.2891 | 6.2891 | 0.0000 | equal in all 10 runs |
| nostep | 4096 | 4 | 12.9688 | 12.9688 | 0.0000 | equal in all 10 runs |

Rows with equal peak memory in all 10 runs: 4/4.

## 2. Results (ms)

Cell: mean of the two repeats (Δ vs main-on in percent, positive = larger; rep = repeat difference of the two runs in the arm, (second - first) / first). A "~" marks |Δ| below max(|rep| of this arm, |rep| of main-on) for the same metric.

### train T=1024 GA=4

| arm (runs) | fwd_mb0 | E | fwd_rest | bwd_rest | step |
|---|---|---|---|---|---|
| main-on (01, 10) | 93.82 (base, rep +0.10%) | 15.15 (base, rep +0.79%) | 78.76 (base, rep -0.25%) | 155.07 (base, rep -0.03%) | 954.99 (base, rep -0.02%) |
| main-dummy (02, 09) | 93.46 (Δ -0.39%, rep +0.10%) | 14.86 (Δ -1.94%, rep -0.03%) | 78.66 (Δ -0.13% ~, rep -0.03%) | 154.93 (Δ -0.09%, rep +0.02%) | 953.87 (Δ -0.12%, rep +0.04%) |
| p1-on (03, 08) | 90.64 (Δ -3.39%, rep +0.49%) | 5.39 (Δ -64.40%, rep +11.35%) | 85.52 (Δ +8.59%, rep +0.06%) | 155.49 (Δ +0.27%, rep +0.01%) | 973.70 (Δ +1.96%, rep +0.09%) |
| main-on-c32 (04, 07) | 95.59 (Δ +1.88%, rep +0.66%) | 16.87 (Δ +11.32%, rep +3.98%) | 78.70 (Δ -0.07% ~, rep +0.01%) | 162.73 (Δ +4.94%, rep -0.14%) | 980.50 (Δ +2.67%, rep +0.21%) |
| p1-on-c32 (05, 06) | 90.37 (Δ -3.67%, rep -0.60%) | 6.42 (Δ -57.62%, rep -7.25%) | 84.68 (Δ +7.52%, rep +0.20%) | 155.55 (Δ +0.31%, rep +0.05%) | 971.66 (Δ +1.75%, rep +0.04%) |

### train T=4096 GA=4

| arm (runs) | fwd_mb0 | E | fwd_rest | bwd_rest | step |
|---|---|---|---|---|---|
| main-on (01, 10) | 285.74 (base, rep -0.01%) | 16.55 (base, rep +1.11%) | 269.27 (base, rep +0.01%) | 539.46 (base, rep +0.04%) | 3256.08 (base, rep -0.03%) |
| main-dummy (02, 09) | 288.11 (Δ +0.83%, rep -0.58%) | 19.27 (Δ +16.46%, rep -8.32%) | 269.24 (Δ -0.01% ~, rep -0.02%) | 539.35 (Δ -0.02% ~, rep -0.02%) | 3257.08 (Δ +0.03% ~, rep -0.05%) |
| p1-on (03, 08) | 303.02 (Δ +6.05%, rep +0.13%) | 7.22 (Δ -56.38%, rep -4.87%) | 296.84 (Δ +10.24%, rep +0.17%) | 542.40 (Δ +0.55%, rep +0.02%) | 3363.62 (Δ +3.30%, rep +0.08%) |
| main-on-c32 (04, 07) | 300.03 (Δ +5.00%, rep -0.34%) | 30.72 (Δ +85.64%, rep -2.66%) | 269.33 (Δ +0.02%, rep -0.00%) | 579.17 (Δ +7.36%, rep -0.10%) | 3389.00 (Δ +4.08%, rep -0.00%) |
| p1-on-c32 (05, 06) | 302.76 (Δ +5.96%, rep +0.32%) | 12.24 (Δ -26.04%, rep -2.03%) | 291.89 (Δ +8.40%, rep -0.10%) | 542.44 (Δ +0.55%, rep -0.05%) | 3349.96 (Δ +2.88%, rep +0.02%) |

### nostep T=1024 GA=4

| arm (runs) | fwd_mb0 | E | fwd_rest | bwd_rest | step |
|---|---|---|---|---|---|
| main-on (01, 10) | 80.36 (base, rep -0.67%) | 1.89 (base, rep -15.26%) | 78.56 (base, rep -0.25%) | 155.06 (base, rep -0.01%) | 938.15 (base, rep -0.17%) |
| main-dummy (02, 09) | 80.30 (Δ -0.07% ~, rep -0.03%) | 1.92 (Δ +1.74% ~, rep +1.74%) | 78.50 (Δ -0.07% ~, rep -0.05%) | 155.00 (Δ -0.04% ~, rep -0.14%) | 937.68 (Δ -0.05% ~, rep -0.11%) |
| p1-on (03, 08) | 86.98 (Δ +8.24%, rep +0.63%) | 1.91 (Δ +0.98% ~, rep +36.00%) | 85.42 (Δ +8.73%, rep +0.23%) | 155.54 (Δ +0.31%, rep -0.11%) | 967.10 (Δ +3.09%, rep +0.05%) |
| main-on-c32 (04, 07) | 80.32 (Δ -0.05% ~, rep -0.29%) | 1.85 (Δ -2.01% ~, rep -1.55%) | 78.55 (Δ -0.02% ~, rep -0.19%) | 162.55 (Δ +4.83%, rep +0.17%) | 960.05 (Δ +2.33%, rep -0.00%) |
| p1-on-c32 (05, 06) | 87.22 (Δ +8.54%, rep -0.82%) | 3.06 (Δ +61.87%, rep -1.35%) | 84.66 (Δ +7.77%, rep -0.16%) | 155.76 (Δ +0.45%, rep -0.02%) | 965.03 (Δ +2.87%, rep -0.02%) |

### nostep T=4096 GA=4

| arm (runs) | fwd_mb0 | E | fwd_rest | bwd_rest | step |
|---|---|---|---|---|---|
| main-on (01, 10) | 272.86 (base, rep -0.01%) | 3.52 (base, rep -14.95%) | 269.44 (base, rep +0.15%) | 539.48 (base, rep +0.08%) | 3241.17 (base, rep +0.07%) |
| main-dummy (02, 09) | 273.24 (Δ +0.14% ~, rep -0.26%) | 4.21 (Δ +19.42% ~, rep -19.50%) | 269.58 (Δ +0.05% ~, rep -0.07%) | 539.78 (Δ +0.06% ~, rep -0.05%) | 3241.87 (Δ +0.02% ~, rep -0.06%) |
| p1-on (03, 08) | 298.09 (Δ +9.24%, rep +0.08%) | 2.15 (Δ -39.02% ~, rep +52.71%) | 296.55 (Δ +10.06%, rep -0.13%) | 542.43 (Δ +0.55%, rep -0.17%) | 3355.83 (Δ +3.54%, rep -0.13%) |
| main-on-c32 (04, 07) | 274.43 (Δ +0.57%, rep -0.13%) | 4.82 (Δ +36.71%, rep -8.86%) | 269.98 (Δ +0.20% ~, rep +0.21%) | 578.95 (Δ +7.32%, rep -0.28%) | 3360.18 (Δ +3.67%, rep -0.04%) |
| p1-on-c32 (05, 06) | 297.84 (Δ +9.15%, rep -0.44%) | 7.20 (Δ +104.33%, rep -4.12%) | 292.14 (Δ +8.42%, rep -0.20%) | 542.49 (Δ +0.56%, rep -0.21%) | 3344.32 (Δ +3.18%, rep -0.13%) |

### Repeat noise floor (|rep| %, all rows, all metrics)

| arm | median | max |
|---|---|---|
| main-on | 0.09 | 15.26 |
| main-dummy | 0.06 | 19.50 |
| p1-on | 0.13 | 52.71 |
| main-on-c32 | 0.20 | 8.86 |
| p1-on-c32 | 0.20 | 7.25 |

