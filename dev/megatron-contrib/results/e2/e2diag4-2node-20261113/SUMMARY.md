# E2 diagnostic: 12 runs, 6 arms, d4096 L32, 2 nodes x 8 GPUs (16 ranks)

## Findings

Train T=4096 GA=4, fwd_rest (mean of 2 repeats): main-on 269.14 ms, p1-on 297.67 ms (+10.60%). Neither env var moves p1 near 269 ms. p1-lo is 297.41 ms (+10.49% vs main-lo 269.16 ms), and p1-gm0 is 297.19 ms (+10.44% vs main-gm0 269.10 ms). The p1 vs main gap for fwd_rest is +10.4% to +10.6% at train T=4096 and +10.1% to +10.3% at nostep T=4096. Repeat differences for fwd_rest at train T=4096 are at most 0.34%. Step at train T=4096 is +3.3% to +3.4% for every p1 arm vs its main partner. The env vars are present in the env.*.txt files of all 12 runs with the expected values. Loss_sum is bit-identical across all 12 runs and peak memory is identical across all runs.

Arms: main-on, p1-on (no extra env); main-lo, p1-lo (NCCL_LAUNCH_ORDER_IMPLICIT=1); main-gm0, p1-gm0 (NCCL_GRAPH_MIXING_SUPPORT=0). main-* = baseline code (snapshot main-d2b0e35cc); p1-* = P1 code (snapshot p1-20261112, outer all-gather on its own CUDA stream). Each arm has two runs.

Metrics (JSON `max` values, ms): fwd_mb0, fwd_rest (microbatches 1..3 forward), bwd_rest (microbatches 1..3 backward), exposed, step.

## 1. Checks

### 1a. Env vars per run (env.*.txt, all node files)

Expected: lo arms NCCL_LAUNCH_ORDER_IMPLICIT=1 and NCCL_GRAPH_MIXING_SUPPORT not set; gm0 arms NCCL_GRAPH_MIXING_SUPPORT=0 and NCCL_LAUNCH_ORDER_IMPLICIT not set; on arms neither set. Cell = value (number of env files with it).

| run | arm | env files | NCCL_LAUNCH_ORDER_IMPLICIT | NCCL_GRAPH_MIXING_SUPPORT | status |
|---|---|---|---|---|---|
| 01-main-on | main-on | 2 | not set (2/2) | not set (2/2) | PASS |
| 02-p1-on | p1-on | 2 | not set (2/2) | not set (2/2) | PASS |
| 03-main-lo | main-lo | 2 | =1 (2/2) | not set (2/2) | PASS |
| 04-p1-lo | p1-lo | 2 | =1 (2/2) | not set (2/2) | PASS |
| 05-main-gm0 | main-gm0 | 2 | not set (2/2) | =0 (2/2) | PASS |
| 06-p1-gm0 | p1-gm0 | 2 | not set (2/2) | =0 (2/2) | PASS |
| 07-p1-gm0 | p1-gm0 | 2 | not set (2/2) | =0 (2/2) | PASS |
| 08-main-gm0 | main-gm0 | 2 | not set (2/2) | =0 (2/2) | PASS |
| 09-p1-lo | p1-lo | 2 | =1 (2/2) | not set (2/2) | PASS |
| 10-main-lo | main-lo | 2 | =1 (2/2) | not set (2/2) | PASS |
| 11-p1-on | p1-on | 2 | not set (2/2) | not set (2/2) | PASS |
| 12-main-on | main-on | 2 | not set (2/2) | not set (2/2) | PASS |

Sweep logs and sysinfo files that mention either variable: 0 of 48 files. NCCL nccl.*.log files that mention either variable: 0 of 192 files. The env.*.txt files are the only per-run record of these variables.

Env var failures: 0.

### 1b. Snapshot path per node, sweep header and meta flag

Snapshot = the `megatron_fsdp` line in node{0,1}.sysinfo.txt. Expected: main-* in `snapshots/main-d2b0e35cc/`, p1-* in `snapshots/p1-20261112/`. Sweep header CUDA_DEVICE_MAX_CONNECTIONS expected None.

| run | arm | node0 snapshot path | node1 snapshot path | sweep CUDA_DEVICE_MAX_CONNECTIONS | meta flag | status |
|---|---|---|---|---|---|---|
| 01-main-on | main-on | /planning-users/yangyeyuan/megatron-contrib/snapshots/main-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | /planning-users/yangyeyuan/megatron-contrib/snapshots/main-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | None | True | PASS |
| 02-p1-on | p1-on | /planning-users/yangyeyuan/megatron-contrib/snapshots/p1-20261112/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | /planning-users/yangyeyuan/megatron-contrib/snapshots/p1-20261112/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | None | True | PASS |
| 03-main-lo | main-lo | /planning-users/yangyeyuan/megatron-contrib/snapshots/main-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | /planning-users/yangyeyuan/megatron-contrib/snapshots/main-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | None | True | PASS |
| 04-p1-lo | p1-lo | /planning-users/yangyeyuan/megatron-contrib/snapshots/p1-20261112/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | /planning-users/yangyeyuan/megatron-contrib/snapshots/p1-20261112/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | None | True | PASS |
| 05-main-gm0 | main-gm0 | /planning-users/yangyeyuan/megatron-contrib/snapshots/main-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | /planning-users/yangyeyuan/megatron-contrib/snapshots/main-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | None | True | PASS |
| 06-p1-gm0 | p1-gm0 | /planning-users/yangyeyuan/megatron-contrib/snapshots/p1-20261112/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | /planning-users/yangyeyuan/megatron-contrib/snapshots/p1-20261112/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | None | True | PASS |
| 07-p1-gm0 | p1-gm0 | /planning-users/yangyeyuan/megatron-contrib/snapshots/p1-20261112/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | /planning-users/yangyeyuan/megatron-contrib/snapshots/p1-20261112/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | None | True | PASS |
| 08-main-gm0 | main-gm0 | /planning-users/yangyeyuan/megatron-contrib/snapshots/main-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | /planning-users/yangyeyuan/megatron-contrib/snapshots/main-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | None | True | PASS |
| 09-p1-lo | p1-lo | /planning-users/yangyeyuan/megatron-contrib/snapshots/p1-20261112/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | /planning-users/yangyeyuan/megatron-contrib/snapshots/p1-20261112/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | None | True | PASS |
| 10-main-lo | main-lo | /planning-users/yangyeyuan/megatron-contrib/snapshots/main-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | /planning-users/yangyeyuan/megatron-contrib/snapshots/main-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | None | True | PASS |
| 11-p1-on | p1-on | /planning-users/yangyeyuan/megatron-contrib/snapshots/p1-20261112/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | /planning-users/yangyeyuan/megatron-contrib/snapshots/p1-20261112/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | None | True | PASS |
| 12-main-on | main-on | /planning-users/yangyeyuan/megatron-contrib/snapshots/main-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | /planning-users/yangyeyuan/megatron-contrib/snapshots/main-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py | None | True | PASS |

Snapshot/header/flag failures: 0.

### 1c. Loss bit-identity across all 12 runs

Per row: per_rank.loss_sum (16 values) and max.loss_sum compared by exact equality with run 01-main-on.

| mode | T | GA | runs identical to 01-main-on | differing runs |
|---|---|---|---|---|
| train | 1024 | 4 | 12/12 | none |
| train | 4096 | 4 | 12/12 | none |
| nostep | 1024 | 4 | 12/12 | none |
| nostep | 4096 | 4 | 12/12 | none |

Run-row pairs that differ from run 01-main-on: 0.

### 1d. Peak memory per arm (max.peak_mem_gb, GB)

| mode | T | GA | arm | repeat 1 | repeat 2 |
|---|---|---|---|---|---|
| train | 1024 | 4 | main-on | 6.2891 | 6.2891 |
| train | 1024 | 4 | p1-on | 6.2891 | 6.2891 |
| train | 1024 | 4 | main-lo | 6.2891 | 6.2891 |
| train | 1024 | 4 | p1-lo | 6.2891 | 6.2891 |
| train | 1024 | 4 | main-gm0 | 6.2891 | 6.2891 |
| train | 1024 | 4 | p1-gm0 | 6.2891 | 6.2891 |
| train | 4096 | 4 | main-on | 12.9688 | 12.9688 |
| train | 4096 | 4 | p1-on | 12.9688 | 12.9688 |
| train | 4096 | 4 | main-lo | 12.9688 | 12.9688 |
| train | 4096 | 4 | p1-lo | 12.9688 | 12.9688 |
| train | 4096 | 4 | main-gm0 | 12.9688 | 12.9688 |
| train | 4096 | 4 | p1-gm0 | 12.9688 | 12.9688 |
| nostep | 1024 | 4 | main-on | 6.2891 | 6.2891 |
| nostep | 1024 | 4 | p1-on | 6.2891 | 6.2891 |
| nostep | 1024 | 4 | main-lo | 6.2891 | 6.2891 |
| nostep | 1024 | 4 | p1-lo | 6.2891 | 6.2891 |
| nostep | 1024 | 4 | main-gm0 | 6.2891 | 6.2891 |
| nostep | 1024 | 4 | p1-gm0 | 6.2891 | 6.2891 |
| nostep | 4096 | 4 | main-on | 12.9688 | 12.9688 |
| nostep | 4096 | 4 | p1-on | 12.9688 | 12.9688 |
| nostep | 4096 | 4 | main-lo | 12.9688 | 12.9688 |
| nostep | 4096 | 4 | p1-lo | 12.9688 | 12.9688 |
| nostep | 4096 | 4 | main-gm0 | 12.9688 | 12.9688 |
| nostep | 4096 | 4 | p1-gm0 | 12.9688 | 12.9688 |

- train T=1024: min 6.2891, max 6.2891, spread 0.0000: identical in all 12 runs
- train T=4096: min 12.9688, max 12.9688, spread 0.0000: identical in all 12 runs
- nostep T=1024: min 6.2891, max 6.2891, spread 0.0000: identical in all 12 runs
- nostep T=4096: min 12.9688, max 12.9688, spread 0.0000: identical in all 12 runs

Rows with identical peak memory in all 12 runs: 4/4.

## 2. Results (ms)

Cell: mean of the two repeats (Δ vs main-on in percent, positive = larger; rep = repeat difference of the two runs in the arm, (second - first) / first). A "~" marks |Δ| below max(|rep| of this arm, |rep| of main-on) for the same metric.

### train T=1024 GA=4

| arm (runs) | fwd_mb0 | fwd_rest | bwd_rest | exposed | step |
|---|---|---|---|---|---|
| main-on (01, 12) | 93.58 (base, rep -0.05%) | 78.75 (base, rep -0.20%) | 155.19 (base, rep -0.06%) | 14.92 (base, rep +0.27%) | 955.18 (base, rep -0.08%) |
| p1-on (02, 11) | 90.22 (Δ -3.59%, rep -0.31%) | 85.67 (Δ +8.79%, rep +0.01%) | 155.60 (Δ +0.26%, rep -0.01%) | 4.80 (Δ -67.86%, rep -5.03%) | 974.82 (Δ +2.06%, rep -0.02%) |
| main-lo (03, 10) | 93.94 (Δ +0.39%, rep -0.26%) | 78.67 (Δ -0.09% ~, rep -0.06%) | 155.19 (Δ -0.00% ~, rep -0.07%) | 15.33 (Δ +2.77%, rep -1.42%) | 955.46 (Δ +0.03% ~, rep -0.04%) |
| p1-lo (04, 09) | 91.73 (Δ -1.98%, rep -1.05%) | 85.67 (Δ +8.79%, rep -0.01%) | 155.65 (Δ +0.30%, rep -0.04%) | 6.25 (Δ -58.10%, rep -10.87%) | 975.95 (Δ +2.17%, rep -0.09%) |
| main-gm0 (05, 08) | 93.61 (Δ +0.03% ~, rep +0.79%) | 78.66 (Δ -0.10% ~, rep -0.04%) | 155.15 (Δ -0.02% ~, rep -0.02%) | 14.96 (Δ +0.26% ~, rep +5.02%) | 954.76 (Δ -0.04% ~, rep +0.07%) |
| p1-gm0 (06, 07) | 90.18 (Δ -3.63%, rep +0.04%) | 85.58 (Δ +8.68%, rep -0.05%) | 155.59 (Δ +0.26%, rep +0.00%) | 4.72 (Δ -68.34%, rep +1.19%) | 974.24 (Δ +2.00%, rep -0.04%) |

### train T=4096 GA=4

| arm (runs) | fwd_mb0 | fwd_rest | bwd_rest | exposed | step |
|---|---|---|---|---|---|
| main-on (01, 12) | 287.42 (base, rep -0.02%) | 269.14 (base, rep -0.01%) | 539.38 (base, rep -0.06%) | 18.47 (base, rep +3.89%) | 3257.03 (base, rep -0.04%) |
| p1-on (02, 11) | 303.86 (Δ +5.72%, rep -0.26%) | 297.67 (Δ +10.60%, rep -0.34%) | 542.53 (Δ +0.58%, rep -0.02%) | 7.04 (Δ -61.90%, rep -7.40%) | 3367.42 (Δ +3.39%, rep -0.14%) |
| main-lo (03, 10) | 287.01 (Δ -0.14%, rep +0.04%) | 269.16 (Δ +0.01% ~, rep -0.00%) | 539.50 (Δ +0.02% ~, rep -0.04%) | 18.29 (Δ -1.00% ~, rep -0.01%) | 3257.01 (Δ -0.00% ~, rep -0.00%) |
| p1-lo (04, 09) | 302.32 (Δ +5.19%, rep +0.00%) | 297.41 (Δ +10.50%, rep -0.09%) | 542.57 (Δ +0.59%, rep -0.03%) | 5.78 (Δ -68.70%, rep +12.69%) | 3365.97 (Δ +3.34%, rep -0.07%) |
| main-gm0 (05, 08) | 287.45 (Δ +0.01% ~, rep +2.40%) | 269.10 (Δ -0.01% ~, rep +0.00%) | 539.37 (Δ -0.00% ~, rep -0.01%) | 18.65 (Δ +0.97% ~, rep +47.86%) | 3256.55 (Δ -0.01% ~, rep +0.22%) |
| p1-gm0 (06, 07) | 302.60 (Δ +5.28%, rep +0.01%) | 297.19 (Δ +10.42%, rep -0.08%) | 542.59 (Δ +0.60%, rep -0.03%) | 6.17 (Δ -66.59%, rep -9.25%) | 3365.13 (Δ +3.32%, rep -0.04%) |

### nostep T=1024 GA=4

| arm (runs) | fwd_mb0 | fwd_rest | bwd_rest | exposed | step |
|---|---|---|---|---|---|
| main-on (01, 12) | 80.51 (base, rep -0.99%) | 78.59 (base, rep -0.24%) | 155.07 (base, rep -0.27%) | 1.92 (base, rep -30.87%) | 938.69 (base, rep -0.32%) |
| p1-on (02, 11) | 86.28 (Δ +7.17%, rep +0.14%) | 85.43 (Δ +8.69%, rep -0.13%) | 155.48 (Δ +0.27% ~, rep -0.19%) | 1.23 (Δ -35.93%, rep +5.60%) | 966.89 (Δ +3.00%, rep -0.10%) |
| main-lo (03, 10) | 80.41 (Δ -0.13% ~, rep +0.18%) | 78.53 (Δ -0.09% ~, rep -0.02%) | 154.87 (Δ -0.13% ~, rep -0.02%) | 2.03 (Δ +5.73% ~, rep -0.92%) | 937.29 (Δ -0.15% ~, rep +0.03%) |
| p1-lo (04, 09) | 86.75 (Δ +7.75%, rep +0.64%) | 85.39 (Δ +8.65%, rep -0.06%) | 155.43 (Δ +0.23% ~, rep +0.07%) | 1.56 (Δ -18.81% ~, rep +36.90%) | 966.03 (Δ +2.91%, rep +0.15%) |
| main-gm0 (05, 08) | 79.88 (Δ -0.78% ~, rep +0.37%) | 78.53 (Δ -0.08% ~, rep +0.05%) | 154.85 (Δ -0.14% ~, rep -0.08%) | 1.45 (Δ -24.55% ~, rep +21.66%) | 937.15 (Δ -0.16% ~, rep +0.04%) |
| p1-gm0 (06, 07) | 87.00 (Δ +8.06%, rep -0.80%) | 85.41 (Δ +8.68%, rep -0.10%) | 155.32 (Δ +0.16% ~, rep -0.06%) | 1.68 (Δ -12.51% ~, rep -43.99%) | 966.27 (Δ +2.94%, rep -0.12%) |

### nostep T=4096 GA=4

| arm (runs) | fwd_mb0 | fwd_rest | bwd_rest | exposed | step |
|---|---|---|---|---|---|
| main-on (01, 12) | 272.83 (base, rep -0.49%) | 269.31 (base, rep +0.09%) | 539.15 (base, rep -0.08%) | 3.61 (base, rep -18.67%) | 3239.60 (base, rep -0.17%) |
| p1-on (02, 11) | 297.33 (Δ +8.98%, rep -0.01%) | 296.44 (Δ +10.07%, rep -0.09%) | 541.81 (Δ +0.49%, rep +0.07%) | 1.83 (Δ -49.23%, rep -12.08%) | 3352.17 (Δ +3.47%, rep -0.03%) |
| main-lo (03, 10) | 271.64 (Δ -0.44% ~, rep +0.20%) | 268.83 (Δ -0.18% ~, rep +0.31%) | 538.60 (Δ -0.10% ~, rep +0.11%) | 3.11 (Δ -13.82% ~, rep -10.73%) | 3233.97 (Δ -0.17% ~, rep +0.21%) |
| p1-lo (04, 09) | 298.07 (Δ +9.25%, rep -0.13%) | 296.39 (Δ +10.06%, rep -0.17%) | 541.46 (Δ +0.43%, rep +0.17%) | 2.76 (Δ -23.53%, rep -17.91%) | 3350.33 (Δ +3.42%, rep +0.12%) |
| main-gm0 (05, 08) | 271.15 (Δ -0.61%, rep +0.39%) | 268.44 (Δ -0.32% ~, rep +0.77%) | 537.34 (Δ -0.34% ~, rep +0.57%) | 2.91 (Δ -19.27%, rep -13.52%) | 3227.82 (Δ -0.36% ~, rep +0.49%) |
| p1-gm0 (06, 07) | 298.11 (Δ +9.27%, rep +0.11%) | 296.06 (Δ +9.93%, rep -0.03%) | 540.94 (Δ +0.33%, rep -0.21%) | 2.50 (Δ -30.80% ~, rep +33.25%) | 3346.88 (Δ +3.31%, rep -0.15%) |

### Repeat noise floor (|rep| %, all rows, all metrics)

| arm | median | max |
|---|---|---|
| main-on | 0.19 | 30.87 |
| p1-on | 0.13 | 12.08 |
| main-lo | 0.07 | 10.73 |
| p1-lo | 0.12 | 36.90 |
| main-gm0 | 0.38 | 47.86 |
| p1-gm0 | 0.09 | 43.99 |

## 2b. P1 vs matching main (ms)

Δ = (p1 mean - main mean) / main mean, positive = p1 slower. rep = repeat difference of each arm. A "~" marks |Δ| below the larger of the two repeat differences.

### train T=1024 GA=4

| p1 vs main | fwd_rest main | fwd_rest p1 | Δ fwd_rest | step main | step p1 | Δ step |
|---|---|---|---|---|---|---|
| p1-on vs main-on | 78.75 (rep -0.20%) | 85.67 (rep +0.01%) | +8.79% | 955.18 (rep -0.08%) | 974.82 (rep -0.02%) | +2.06% |
| p1-lo vs main-lo | 78.67 (rep -0.06%) | 85.67 (rep -0.01%) | +8.89% | 955.46 (rep -0.04%) | 975.95 (rep -0.09%) | +2.14% |
| p1-gm0 vs main-gm0 | 78.66 (rep -0.04%) | 85.58 (rep -0.05%) | +8.79% | 954.76 (rep +0.07%) | 974.24 (rep -0.04%) | +2.04% |

### train T=4096 GA=4

| p1 vs main | fwd_rest main | fwd_rest p1 | Δ fwd_rest | step main | step p1 | Δ step |
|---|---|---|---|---|---|---|
| p1-on vs main-on | 269.14 (rep -0.01%) | 297.67 (rep -0.34%) | +10.60% | 3257.03 (rep -0.04%) | 3367.42 (rep -0.14%) | +3.39% |
| p1-lo vs main-lo | 269.16 (rep -0.00%) | 297.41 (rep -0.09%) | +10.49% | 3257.01 (rep -0.00%) | 3365.97 (rep -0.07%) | +3.35% |
| p1-gm0 vs main-gm0 | 269.10 (rep +0.00%) | 297.19 (rep -0.08%) | +10.44% | 3256.55 (rep +0.22%) | 3365.13 (rep -0.04%) | +3.33% |

### nostep T=1024 GA=4

| p1 vs main | fwd_rest main | fwd_rest p1 | Δ fwd_rest | step main | step p1 | Δ step |
|---|---|---|---|---|---|---|
| p1-on vs main-on | 78.59 (rep -0.24%) | 85.43 (rep -0.13%) | +8.69% | 938.69 (rep -0.32%) | 966.89 (rep -0.10%) | +3.00% |
| p1-lo vs main-lo | 78.53 (rep -0.02%) | 85.39 (rep -0.06%) | +8.74% | 937.29 (rep +0.03%) | 966.03 (rep +0.15%) | +3.07% |
| p1-gm0 vs main-gm0 | 78.53 (rep +0.05%) | 85.41 (rep -0.10%) | +8.76% | 937.15 (rep +0.04%) | 966.27 (rep -0.12%) | +3.11% |

### nostep T=4096 GA=4

| p1 vs main | fwd_rest main | fwd_rest p1 | Δ fwd_rest | step main | step p1 | Δ step |
|---|---|---|---|---|---|---|
| p1-on vs main-on | 269.31 (rep +0.09%) | 296.44 (rep -0.09%) | +10.07% | 3239.60 (rep -0.17%) | 3352.17 (rep -0.03%) | +3.47% |
| p1-lo vs main-lo | 268.83 (rep +0.31%) | 296.39 (rep -0.17%) | +10.25% | 3233.97 (rep +0.21%) | 3350.33 (rep +0.12%) | +3.60% |
| p1-gm0 vs main-gm0 | 268.44 (rep +0.77%) | 296.06 (rep -0.03%) | +10.29% | 3227.82 (rep +0.49%) | 3346.88 (rep -0.15%) | +3.69% |

