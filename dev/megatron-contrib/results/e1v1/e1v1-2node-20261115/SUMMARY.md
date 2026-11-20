# E1 v1 vs v2: 10 runs, 5 arms, d4096 L32, 2 nodes x 8 GPUs (16 ranks), GA=4

Arms: v1-base (v1, outer all-gather on own stream, inner waits); v1-5985 (v1, hfsdp_param_gather_overlap=True, outer on own stream, prefetched); v1-same (v1 patched, outer all-gather on the inner stream, baseline for v1); v2-off (v2, outer on the inner stream, baseline for v2); v2-p1 (v2, overlap_outer_communication=True, outer on own stream). Each arm has two runs.

Metrics are the JSON `max` values in ms (per-rank value, max over 16 ranks): fwd_mb0, fwd_rest (microbatches 1..3 forward), bwd_mb0, bwd_rest, exposed, step.

## Findings

fwd_rest, mean of two repeats, delta % vs the version baseline (v1-same for the v1 arms, v2-off for v2-p1). "~" means |delta| is below the threshold, the larger |rep| of the two arms compared. Earlier jobs showed about +8% to +10% at T=4096 (quoted from the request, not re-measured here).

- train, T=1024: v1-base vs v1-same -0.08% ~ (threshold 0.23%); v1-5985 vs v1-same +0.08% ~ (threshold 0.12%); v2-p1 vs v2-off +8.67% (threshold 0.35%).
- train, T=4096: v1-base vs v1-same -0.05% ~ (threshold 0.25%); v1-5985 vs v1-same -0.05% ~ (threshold 0.12%); v2-p1 vs v2-off +10.05% (threshold 0.17%).
- nostep, T=1024: v1-base vs v1-same +0.07% (threshold 0.05%); v1-5985 vs v1-same +0.04% ~ (threshold 0.10%); v2-p1 vs v2-off +8.62% (threshold 0.46%).
- nostep, T=4096: v1-base vs v1-same +0.01% ~ (threshold 0.22%); v1-5985 vs v1-same -0.09% ~ (threshold 0.22%); v2-p1 vs v2-off +10.52% (threshold 0.37%).

## 1. Checks

### 1a. Meta flags and sweep header per run

| run | arm | meta (expected) | meta check | sweep header vs meta | status |
|---|---|---|---|---|---|
| 01-v2-off | v2-off | version=<absent>, overlap_outer_communication=False | ok | overlap_outer_communication=False | PASS |
| 02-v2-p1 | v2-p1 | version=<absent>, overlap_outer_communication=True | ok | overlap_outer_communication=True | PASS |
| 03-v1-base | v1-base | version=v1, outer_ag_on_ag_stream=False, hfsdp_param_gather_overlap=False | ok | hfsdp_param_gather_overlap=False, outer_ag_is_ag_stream=False | PASS |
| 04-v1-5985 | v1-5985 | version=v1, outer_ag_on_ag_stream=False, hfsdp_param_gather_overlap=True | ok | hfsdp_param_gather_overlap=True, outer_ag_is_ag_stream=False | PASS |
| 05-v1-same | v1-same | version=v1, outer_ag_on_ag_stream=True, hfsdp_param_gather_overlap=False | ok | hfsdp_param_gather_overlap=False, outer_ag_is_ag_stream=True | PASS |
| 06-v1-same | v1-same | version=v1, outer_ag_on_ag_stream=True, hfsdp_param_gather_overlap=False | ok | hfsdp_param_gather_overlap=False, outer_ag_is_ag_stream=True | PASS |
| 07-v1-5985 | v1-5985 | version=v1, outer_ag_on_ag_stream=False, hfsdp_param_gather_overlap=True | ok | hfsdp_param_gather_overlap=True, outer_ag_is_ag_stream=False | PASS |
| 08-v1-base | v1-base | version=v1, outer_ag_on_ag_stream=False, hfsdp_param_gather_overlap=False | ok | hfsdp_param_gather_overlap=False, outer_ag_is_ag_stream=False | PASS |
| 09-v2-p1 | v2-p1 | version=<absent>, overlap_outer_communication=True | ok | overlap_outer_communication=True | PASS |
| 10-v2-off | v2-off | version=<absent>, overlap_outer_communication=False | ok | overlap_outer_communication=False | PASS |

### 1b. megatron_fsdp snapshot path per run (sysinfo)

| run | arm | node0 | node1 |
|---|---|---|---|
| 01-v2-off | v2-off | `/planning-users/yangyeyuan/megatron-contrib/snapshots/p1-20261112/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py` | `/planning-users/yangyeyuan/megatron-contrib/snapshots/p1-20261112/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py` |
| 02-v2-p1 | v2-p1 | `/planning-users/yangyeyuan/megatron-contrib/snapshots/p1-20261112/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py` | `/planning-users/yangyeyuan/megatron-contrib/snapshots/p1-20261112/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py` |
| 03-v1-base | v1-base | `/planning-users/yangyeyuan/megatron-contrib/snapshots/main-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py` | `/planning-users/yangyeyuan/megatron-contrib/snapshots/main-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py` |
| 04-v1-5985 | v1-5985 | `/planning-users/yangyeyuan/megatron-contrib/snapshots/main-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py` | `/planning-users/yangyeyuan/megatron-contrib/snapshots/main-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py` |
| 05-v1-same | v1-same | `/planning-users/yangyeyuan/megatron-contrib/snapshots/v1-outer-on-ag-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py` | `/planning-users/yangyeyuan/megatron-contrib/snapshots/v1-outer-on-ag-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py` |
| 06-v1-same | v1-same | `/planning-users/yangyeyuan/megatron-contrib/snapshots/v1-outer-on-ag-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py` | `/planning-users/yangyeyuan/megatron-contrib/snapshots/v1-outer-on-ag-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py` |
| 07-v1-5985 | v1-5985 | `/planning-users/yangyeyuan/megatron-contrib/snapshots/main-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py` | `/planning-users/yangyeyuan/megatron-contrib/snapshots/main-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py` |
| 08-v1-base | v1-base | `/planning-users/yangyeyuan/megatron-contrib/snapshots/main-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py` | `/planning-users/yangyeyuan/megatron-contrib/snapshots/main-d2b0e35cc/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py` |
| 09-v2-p1 | v2-p1 | `/planning-users/yangyeyuan/megatron-contrib/snapshots/p1-20261112/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py` | `/planning-users/yangyeyuan/megatron-contrib/snapshots/p1-20261112/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py` |
| 10-v2-off | v2-off | `/planning-users/yangyeyuan/megatron-contrib/snapshots/p1-20261112/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py` | `/planning-users/yangyeyuan/megatron-contrib/snapshots/p1-20261112/megatron/core/distributed/fsdp/src/megatron_fsdp/__init__.py` |

### 1c. loss_sum (max.loss_sum)

Bit-identity is exact float equality of the JSON values, pooled over the runs of one version (v1: 03, 04, 05, 06, 07, 08; v2: 01, 02, 09, 10). The cross-version columns are informational.

| mode | T | GA | v1 identical | v2 identical | v1 loss_sum (first run) | v2 loss_sum (first run) | v1 == v2 | abs diff |
|---|---|---|---|---|---|---|---|---|
| train | 256 | 4 | yes | yes | 173.80037212371826 | 173.80035734176636 | no | 1.48e-05 |
| train | 1024 | 4 | yes | yes | 173.90694117546082 | 173.90693974494934 | no | 1.43e-06 |
| train | 4096 | 4 | yes | yes | 173.66075086593628 | 173.66074752807617 | no | 3.34e-06 |
| nostep | 256 | 4 | yes | yes | 173.79852294921875 | 173.79848098754883 | no | 4.2e-05 |
| nostep | 1024 | 4 | yes | yes | 173.9063093662262 | 173.90632247924805 | no | 1.31e-05 |
| nostep | 4096 | 4 | yes | yes | 173.66077327728271 | 173.66078901290894 | no | 1.57e-05 |

### 1d. Peak memory per arm (max.peak_mem_gb, GB)

Cell: repeat 1 / repeat 2 in time order. Spread is max - min over all 10 runs.

| mode | T | v1-base | v1-5985 | v1-same | v2-off | v2-p1 | spread | verdict |
|---|---|---|---|---|---|---|---|---|
| train | 256 | 5.2071 / 5.2071 | 5.2071 / 5.2071 | 5.2071 / 5.2071 | 4.6192 / 4.6192 | 4.6192 / 4.6192 | 0.5879 | differs |
| train | 1024 | 6.7657 / 6.7657 | 6.7657 / 6.7657 | 6.7657 / 6.7657 | 6.2891 / 6.2891 | 6.2891 / 6.2891 | 0.4766 | differs |
| train | 4096 | 13.0938 / 13.0938 | 13.0938 / 13.0938 | 13.0938 / 13.0938 | 12.9688 / 12.9688 | 12.9688 / 12.9688 | 0.1250 | differs |
| nostep | 256 | 5.2071 / 5.2071 | 5.2071 / 5.2071 | 5.2071 / 5.2071 | 4.6192 / 4.6192 | 4.6192 / 4.6192 | 0.5879 | differs |
| nostep | 1024 | 6.7657 / 6.7657 | 6.7657 / 6.7657 | 6.7657 / 6.7657 | 6.2891 / 6.2891 | 6.2891 / 6.2891 | 0.4766 | differs |
| nostep | 4096 | 13.0938 / 13.0938 | 13.0938 / 13.0938 | 13.0938 / 13.0938 | 12.9688 / 12.9688 | 12.9688 / 12.9688 | 0.1250 | differs |

## 2. Results (ms)

### train T=256 GA=4

#### Table A: v1 arms (baseline v1-same)

| arm (runs) | fwd_mb0 | fwd_rest | bwd_mb0 | bwd_rest | exposed | step |
|---|---|---|---|---|---|---|
| v1-base (03, 08) | 42.53 (Δ -0.46% ~, rep +1.26%) | 32.57 (Δ +0.10%, rep -0.07%) | 72.56 (Δ -0.14%, rep +0.07%) | 71.36 (Δ +0.20%, rep -0.01%) | 10.00 (Δ -2.26% ~, rep +5.58%) | 431.00 (Δ -0.02% ~, rep +0.11%) |
| v1-5985 (04, 07) | 35.90 (Δ -15.97%, rep +0.78%) | 32.57 (Δ +0.10% ~, rep -0.15%) | 70.66 (Δ -2.75%, rep -0.12%) | 71.19 (Δ -0.03% ~, rep -0.17%) | 3.40 (Δ -66.73%, rep +7.27%) | 422.37 (Δ -2.02%, rep +0.03%) |
| v1-same (05, 06) | 42.73 (base, rep -0.97%) | 32.54 (base, rep -0.00%) | 72.66 (base, rep +0.11%) | 71.21 (base, rep +0.18%) | 10.23 (base, rep -3.89%) | 431.08 (base, rep -0.09%) |

#### Table B: v2 arms (baseline v2-off)

| arm (runs) | fwd_mb0 | fwd_rest | bwd_mb0 | bwd_rest | exposed | step |
|---|---|---|---|---|---|---|
| v2-off (01, 10) | 55.32 (base, rep -0.24%) | 32.96 (base, rep -0.04%) | 65.25 (base, rep +0.06%) | 64.95 (base, rep +0.01%) | 22.48 (base, rep -0.16%) | 429.81 (base, rep -0.17%) |
| v2-p1 (02, 09) | 35.17 (Δ -36.42%, rep +0.38%) | 32.58 (Δ -1.16%, rep -0.06%) | 66.17 (Δ +1.41%, rep +0.24%) | 65.57 (Δ +0.96%, rep +0.01%) | 2.65 (Δ -88.22%, rep +8.19%) | 399.66 (Δ -7.02%, rep +0.02%) |

#### Table C: all 5 arms, mean fwd_rest and step (ms, informational)

| metric | v1-base | v1-5985 | v1-same | v2-off | v2-p1 |
|---|---|---|---|---|---|
| fwd_rest | 32.57 | 32.57 | 32.54 | 32.96 | 32.58 |
| step | 431.00 | 422.37 | 431.08 | 429.81 | 399.66 |

### train T=1024 GA=4

#### Table A: v1 arms (baseline v1-same)

| arm (runs) | fwd_mb0 | fwd_rest | bwd_mb0 | bwd_rest | exposed | step |
|---|---|---|---|---|---|---|
| v1-base (03, 08) | 94.20 (Δ -0.39% ~, rep +0.58%) | 85.34 (Δ -0.08% ~, rep -0.23%) | 177.93 (Δ +0.49%, rep -0.27%) | 176.19 (Δ +0.16% ~, rep -0.57%) | 8.87 (Δ -5.52% ~, rep +9.88%) | 1058.28 (Δ +0.03% ~, rep -0.20%) |
| v1-5985 (04, 07) | 90.16 (Δ -4.66%, rep +1.33%) | 85.48 (Δ +0.08% ~, rep -0.11%) | 167.49 (Δ -5.41%, rep +0.51%) | 176.30 (Δ +0.22%, rep -0.09%) | 5.00 (Δ -46.72%, rep +31.38%) | 1045.05 (Δ -1.22%, rep -0.00%) |
| v1-same (05, 06) | 94.57 (base, rep -0.85%) | 85.41 (base, rep -0.12%) | 177.06 (base, rep -0.08%) | 175.91 (base, rep +0.22%) | 9.39 (base, rep -10.79%) | 1057.96 (base, rep +0.18%) |

#### Table B: v2 arms (baseline v2-off)

| arm (runs) | fwd_mb0 | fwd_rest | bwd_mb0 | bwd_rest | exposed | step |
|---|---|---|---|---|---|---|
| v2-off (01, 10) | 93.51 (base, rep -0.07%) | 78.66 (base, rep -0.18%) | 154.26 (base, rep -0.25%) | 154.47 (base, rep -0.21%) | 14.99 (base, rep +0.80%) | 952.52 (base, rep -0.22%) |
| v2-p1 (02, 09) | 90.07 (Δ -3.68%, rep -0.95%) | 85.48 (Δ +8.67%, rep -0.35%) | 155.60 (Δ +0.87%, rep -0.27%) | 155.40 (Δ +0.60%, rep -0.18%) | 4.75 (Δ -68.33%, rep -12.72%) | 972.89 (Δ +2.14%, rep -0.33%) |

#### Table C: all 5 arms, mean fwd_rest and step (ms, informational)

| metric | v1-base | v1-5985 | v1-same | v2-off | v2-p1 |
|---|---|---|---|---|---|
| fwd_rest | 85.34 | 85.48 | 85.41 | 78.66 | 85.48 |
| step | 1058.28 | 1045.05 | 1057.96 | 952.52 | 972.89 |

### train T=4096 GA=4

#### Table A: v1 arms (baseline v1-same)

| arm (runs) | fwd_mb0 | fwd_rest | bwd_mb0 | bwd_rest | exposed | step |
|---|---|---|---|---|---|---|
| v1-base (03, 08) | 312.48 (Δ +0.04% ~, rep -0.52%) | 295.75 (Δ -0.05% ~, rep -0.25%) | 583.76 (Δ +0.64%, rep -0.25%) | 583.54 (Δ -0.01% ~, rep -0.21%) | 17.77 (Δ -2.11% ~, rep -9.57%) | 3530.67 (Δ +0.11% ~, rep -0.21%) |
| v1-5985 (04, 07) | 303.77 (Δ -2.75%, rep -0.26%) | 295.76 (Δ -0.05% ~, rep -0.12%) | 564.34 (Δ -2.71%, rep +0.63%) | 583.30 (Δ -0.05% ~, rep -0.49%) | 8.87 (Δ -51.14%, rep -6.26%) | 3498.92 (Δ -0.79%, rep +0.04%) |
| v1-same (05, 06) | 312.37 (base, rep +0.09%) | 295.90 (base, rep -0.04%) | 580.06 (base, rep +0.18%) | 583.58 (base, rep -0.14%) | 18.15 (base, rep +3.65%) | 3526.77 (base, rep +0.00%) |

#### Table B: v2 arms (baseline v2-off)

| arm (runs) | fwd_mb0 | fwd_rest | bwd_mb0 | bwd_rest | exposed | step |
|---|---|---|---|---|---|---|
| v2-off (01, 10) | 284.14 (base, rep +0.14%) | 268.81 (base, rep -0.05%) | 537.90 (base, rep -0.16%) | 538.45 (base, rep -0.13%) | 16.32 (base, rep +12.45%) | 3248.33 (base, rep -0.12%) |
| v2-p1 (02, 09) | 302.72 (Δ +6.54%, rep -0.28%) | 295.81 (Δ +10.05%, rep -0.17%) | 540.21 (Δ +0.43% ~, rep -0.47%) | 540.44 (Δ +0.37% ~, rep -0.40%) | 8.11 (Δ -50.32%, rep -7.70%) | 3351.52 (Δ +3.18%, rep -0.39%) |

#### Table C: all 5 arms, mean fwd_rest and step (ms, informational)

| metric | v1-base | v1-5985 | v1-same | v2-off | v2-p1 |
|---|---|---|---|---|---|
| fwd_rest | 295.75 | 295.76 | 295.90 | 268.81 | 295.81 |
| step | 3530.67 | 3498.92 | 3526.77 | 3248.33 | 3351.52 |

### nostep T=256 GA=4

#### Table A: v1 arms (baseline v1-same)

| arm (runs) | fwd_mb0 | fwd_rest | bwd_mb0 | bwd_rest | exposed | step |
|---|---|---|---|---|---|---|
| v1-base (03, 08) | 34.07 (Δ -0.15% ~, rep -0.23%) | 32.51 (Δ -0.10%, rep +0.08%) | 70.49 (Δ +0.10% ~, rep -0.22%) | 71.06 (Δ -0.03%, rep +0.00%) | 1.57 (Δ -2.37% ~, rep -8.34%) | 418.54 (Δ -0.01% ~, rep +0.02%) |
| v1-5985 (04, 07) | 33.99 (Δ -0.40%, rep +0.15%) | 32.53 (Δ -0.06% ~, rep +0.06%) | 70.46 (Δ +0.04% ~, rep +0.08%) | 71.08 (Δ -0.01% ~, rep -0.07%) | 1.57 (Δ -2.29% ~, rep +11.42%) | 418.45 (Δ -0.03% ~, rep -0.04%) |
| v1-same (05, 06) | 34.13 (base, rep -0.31%) | 32.54 (base, rep -0.05%) | 70.43 (base, rep +0.10%) | 71.09 (base, rep +0.00%) | 1.60 (base, rep -4.63%) | 418.59 (base, rep -0.10%) |

#### Table B: v2 arms (baseline v2-off)

| arm (runs) | fwd_mb0 | fwd_rest | bwd_mb0 | bwd_rest | exposed | step |
|---|---|---|---|---|---|---|
| v2-off (01, 10) | 33.65 (base, rep -0.49%) | 32.94 (base, rep -0.04%) | 65.13 (base, rep +0.05%) | 64.94 (base, rep +0.05%) | 0.81 (base, rep -10.36%) | 405.26 (base, rep -0.09%) |
| v2-p1 (02, 09) | 33.58 (Δ -0.21% ~, rep +2.41%) | 32.54 (Δ -1.20%, rep -0.10%) | 65.89 (Δ +1.18%, rep -0.08%) | 65.46 (Δ +0.80%, rep +0.03%) | 1.08 (Δ +32.27% ~, rep +107.64%) | 395.09 (Δ -2.51%, rep +0.10%) |

#### Table C: all 5 arms, mean fwd_rest and step (ms, informational)

| metric | v1-base | v1-5985 | v1-same | v2-off | v2-p1 |
|---|---|---|---|---|---|
| fwd_rest | 32.51 | 32.53 | 32.54 | 32.94 | 32.54 |
| step | 418.54 | 418.45 | 418.59 | 405.26 | 395.09 |

### nostep T=1024 GA=4

#### Table A: v1 arms (baseline v1-same)

| arm (runs) | fwd_mb0 | fwd_rest | bwd_mb0 | bwd_rest | exposed | step |
|---|---|---|---|---|---|---|
| v1-base (03, 08) | 87.01 (Δ +0.14% ~, rep +0.26%) | 85.27 (Δ +0.07%, rep +0.04%) | 166.42 (Δ +0.11% ~, rep -0.21%) | 175.80 (Δ -0.09% ~, rep -0.17%) | 2.06 (Δ +11.84%, rep +5.28%) | 1037.73 (Δ +0.09% ~, rep +0.09%) |
| v1-5985 (04, 07) | 87.25 (Δ +0.41%, rep +0.10%) | 85.25 (Δ +0.04% ~, rep +0.10%) | 166.57 (Δ +0.21% ~, rep +0.54%) | 175.74 (Δ -0.12%, rep -0.10%) | 2.28 (Δ +23.85%, rep +1.65%) | 1036.78 (Δ -0.00% ~, rep +0.00%) |
| v1-same (05, 06) | 86.89 (base, rep +0.31%) | 85.21 (base, rep +0.05%) | 166.23 (base, rep +0.43%) | 175.96 (base, rep +0.03%) | 1.84 (base, rep +8.48%) | 1036.81 (base, rep +0.11%) |

#### Table B: v2 arms (baseline v2-off)

| arm (runs) | fwd_mb0 | fwd_rest | bwd_mb0 | bwd_rest | exposed | step |
|---|---|---|---|---|---|---|
| v2-off (01, 10) | 80.08 (base, rep -0.13%) | 78.53 (base, rep -0.46%) | 153.98 (base, rep -0.60%) | 154.24 (base, rep -0.60%) | 1.63 (base, rep +9.28%) | 934.84 (base, rep -0.45%) |
| v2-p1 (02, 09) | 86.87 (Δ +8.48%, rep +0.71%) | 85.30 (Δ +8.62%, rep -0.02%) | 154.94 (Δ +0.62%, rep -0.42%) | 154.89 (Δ +0.42% ~, rep -0.43%) | 1.79 (Δ +10.02% ~, rep +48.55%) | 963.95 (Δ +3.11%, rep -0.18%) |

#### Table C: all 5 arms, mean fwd_rest and step (ms, informational)

| metric | v1-base | v1-5985 | v1-same | v2-off | v2-p1 |
|---|---|---|---|---|---|
| fwd_rest | 85.27 | 85.25 | 85.21 | 78.53 | 85.30 |
| step | 1037.73 | 1036.78 | 1036.81 | 934.84 | 963.95 |

### nostep T=4096 GA=4

#### Table A: v1 arms (baseline v1-same)

| arm (runs) | fwd_mb0 | fwd_rest | bwd_mb0 | bwd_rest | exposed | step |
|---|---|---|---|---|---|---|
| v1-base (03, 08) | 297.51 (Δ -0.18% ~, rep -0.10%) | 295.60 (Δ +0.01% ~, rep +0.05%) | 561.52 (Δ -0.56%, rep +0.07%) | 583.42 (Δ -0.10% ~, rep +0.16%) | 2.40 (Δ -22.42% ~, rep -29.78%) | 3488.24 (Δ -0.10%, rep +0.02%) |
| v1-5985 (04, 07) | 297.43 (Δ -0.20% ~, rep +0.04%) | 295.29 (Δ -0.09% ~, rep +0.04%) | 564.67 (Δ -0.01% ~, rep +0.46%) | 583.37 (Δ -0.11% ~, rep -0.19%) | 3.10 (Δ +0.09% ~, rep +7.89%) | 3488.93 (Δ -0.08% ~, rep -0.11%) |
| v1-same (05, 06) | 298.03 (base, rep +0.28%) | 295.57 (base, rep +0.22%) | 564.70 (base, rep +0.09%) | 583.99 (base, rep -0.10%) | 3.09 (base, rep +6.91%) | 3491.72 (base, rep +0.07%) |

#### Table B: v2 arms (baseline v2-off)

| arm (runs) | fwd_mb0 | fwd_rest | bwd_mb0 | bwd_rest | exposed | step |
|---|---|---|---|---|---|---|
| v2-off (01, 10) | 270.99 (base, rep -0.05%) | 267.58 (base, rep -0.37%) | 535.82 (base, rep -0.25%) | 536.12 (base, rep -0.26%) | 3.79 (base, rep +17.93%) | 3220.22 (base, rep -0.27%) |
| v2-p1 (02, 09) | 297.44 (Δ +9.76%, rep +0.51%) | 295.74 (Δ +10.52%, rep +0.13%) | 539.50 (Δ +0.69%, rep -0.26%) | 539.72 (Δ +0.67%, rep -0.24%) | 2.24 (Δ -40.87% ~, rep +86.82%) | 3341.06 (Δ +3.75%, rep -0.14%) |

#### Table C: all 5 arms, mean fwd_rest and step (ms, informational)

| metric | v1-base | v1-5985 | v1-same | v2-off | v2-p1 |
|---|---|---|---|---|---|
| fwd_rest | 295.60 | 295.29 | 295.57 | 267.58 | 295.74 |
| step | 3488.24 | 3488.93 | 3491.72 | 3220.22 | 3341.06 |

### Repeat noise floor (|rep| %, all rows, all metrics)

| arm | median | max |
|---|---|---|
| v1-base | 0.21 | 29.78 |
| v1-5985 | 0.14 | 31.38 |
| v1-same | 0.13 | 10.79 |
| v2-off | 0.19 | 17.93 |
| v2-p1 | 0.31 | 107.64 |

## 3. Anomalies

None.

