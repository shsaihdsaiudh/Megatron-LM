# E1v1 Nsight Systems comparison: forward of microbatches 1..3

Script: `bench/e1v1_nsys_compare.py`. Inputs: sqlite exports of `d4096_L32_T4096_ga4.rank{0,8}.nsys-rep` for v2-off, v2-p1 and v1-same. Rank 8 is primary and rank 0 is secondary.

All per-microbatch numbers average over the 6 forward instances (mb1, mb2, mb3 in each of 2 steps). Membership is CPU launch time inside the NVTX range (primary). The GPU-window check is in section 1.

## Rank 8 (primary)

Arms in this group, baseline v2-off: v2-off = `01-v2-off/nsys/d4096_L32_T4096_ga4.rank8.sqlite`, v2-p1 = `02-v2-p1/nsys/d4096_L32_T4096_ga4.rank8.sqlite`, v1-same = `03-v1-same/nsys/d4096_L32_T4096_ga4.rank8.sqlite`.

### 1. Detection summary

### v2-off (01-v2-off/nsys/d4096_L32_T4096_ga4.rank8.sqlite)

- Compute stream: 7 (1536 nvjet kernels; other streams with nvjet: none).
- AG-I: stream 39, gridX 24, ncclDevKernel_AllGather_RING_LL.
- Outer AG: stream 39, gridX 4, ncclDevKernel_AllGather_RING_LL.
- Forward CPU membership: thread-matched / any-thread launches inside the range: mb0_fwd 526/526, mb1_fwd 460/460, mb2_fwd 460/460, mb3_fwd 460/460.

| stream | NCCL kernel | gridX | role | mb0 fwd per mb | mb1..3 fwd per mb | mb*_bwd per mb | outside any phase | total |
|---|---|---|---|---|---|---|---|---|
| 35 | ReduceScatter_Sum_bf16_RING_LL | 1 | ReduceScatter | 0.00 | 0.00 | 1.25 | 0 | 10 |
| 35 | ReduceScatter_Sum_bf16_RING_LL | 4 | ReduceScatter | 0.00 | 0.00 | 8.00 | 0 | 64 |
| 35 | ReduceScatter_Sum_bf16_RING_LL | 24 | ReduceScatter | 0.00 | 0.00 | 32.00 | 0 | 256 |
| 39 | AllGather_RING_LL | 1 | other AllGather | 2.00 | 1.00 | 1.00 | 0 | 18 |
| 39 | AllGather_RING_LL | 4 | outer AG | 32.00 | 0.00 | 0.00 | 0 | 64 |
| 39 | AllGather_RING_LL | 24 | AG-I | 32.00 | 32.00 | 32.00 | 0 | 512 |

### v2-p1 (02-v2-p1/nsys/d4096_L32_T4096_ga4.rank8.sqlite)

- Compute stream: 7 (1536 nvjet kernels; other streams with nvjet: none).
- AG-I: stream 39, gridX 24, ncclDevKernel_AllGather_RING_LL.
- Outer AG: stream 47, gridX 4, ncclDevKernel_AllGather_RING_LL.
- Forward CPU membership: thread-matched / any-thread launches inside the range: mb0_fwd 526/526, mb1_fwd 460/460, mb2_fwd 460/460, mb3_fwd 460/460.

| stream | NCCL kernel | gridX | role | mb0 fwd per mb | mb1..3 fwd per mb | mb*_bwd per mb | outside any phase | total |
|---|---|---|---|---|---|---|---|---|
| 35 | ReduceScatter_Sum_bf16_RING_LL | 1 | ReduceScatter | 0.00 | 0.00 | 1.00 | 0 | 8 |
| 35 | ReduceScatter_Sum_bf16_RING_LL | 24 | ReduceScatter | 0.00 | 0.00 | 32.00 | 0 | 256 |
| 39 | AllGather_RING_LL | 1 | other AllGather | 1.00 | 1.00 | 1.00 | 0 | 16 |
| 39 | AllGather_RING_LL | 24 | AG-I | 32.00 | 32.00 | 32.00 | 0 | 512 |
| 43 | ReduceScatter_Sum_bf16_RING_LL | 1 | ReduceScatter | 0.00 | 0.00 | 0.25 | 0 | 2 |
| 43 | ReduceScatter_Sum_bf16_RING_LL | 4 | ReduceScatter | 0.00 | 0.00 | 8.00 | 0 | 64 |
| 47 | AllGather_RING_LL | 1 | other AllGather | 1.00 | 0.00 | 0.00 | 0 | 2 |
| 47 | AllGather_RING_LL | 4 | outer AG | 32.00 | 0.00 | 0.00 | 0 | 64 |

### v1-same (03-v1-same/nsys/d4096_L32_T4096_ga4.rank8.sqlite)

- Compute stream: 7 (1536 nvjet kernels; other streams with nvjet: none).
- AG-I: stream 43, gridX 24, ncclDevKernel_AllGather_RING_LL.
- Outer AG: stream 51, gridX 4, ncclDevKernel_AllGather_RING_LL.
- Forward CPU membership: thread-matched / any-thread launches inside the range: mb0_fwd 526/526, mb1_fwd 458/458, mb2_fwd 458/458, mb3_fwd 458/458.

| stream | NCCL kernel | gridX | role | mb0 fwd per mb | mb1..3 fwd per mb | mb*_bwd per mb | outside any phase | total |
|---|---|---|---|---|---|---|---|---|
| 43 | AllGather_RING_LL | 1 | other AllGather | 1.00 | 0.00 | 0.00 | 0 | 2 |
| 43 | AllGather_RING_LL | 24 | AG-I | 32.00 | 32.00 | 32.00 | 0 | 512 |
| 47 | ReduceScatter_Sum_bf16_RING_LL | 1 | ReduceScatter | 0.00 | 0.00 | 1.00 | 0 | 8 |
| 47 | ReduceScatter_Sum_bf16_RING_LL | 24 | ReduceScatter | 0.00 | 0.00 | 32.00 | 0 | 256 |
| 51 | AllGather_RING_LL | 1 | other AllGather | 1.00 | 0.00 | 0.00 | 0 | 2 |
| 51 | AllGather_RING_LL | 4 | outer AG | 32.00 | 0.00 | 8.00 | 0 | 128 |
| 55 | ReduceScatter_Sum_bf16_RING_LL | 1 | ReduceScatter | 0.00 | 0.00 | 0.25 | 0 | 2 |
| 55 | ReduceScatter_Sum_bf16_RING_LL | 4 | ReduceScatter | 0.00 | 0.00 | 8.00 | 0 | 64 |

Membership check (CPU launch primary vs GPU window secondary):

| metric (mb1..3 forward, per microbatch) | v2-off | v2-p1 | v1-same |
|---|---|---|---|
| CPU launch (primary): all ops per mb | 230.0 | 230.0 | 229.0 |
| GPU window (secondary): all ops per mb | 232.8 | 231.2 | 229.0 |
| CPU launch (primary): compute-stream ops per mb | 197.0 | 197.0 | 197.0 |
| GPU window (secondary): compute-stream ops per mb | 197.0 | 197.0 | 197.0 |
| CPU launch (primary): compute-stream op ms per mb | 268.09 | 294.71 | 295.41 |
| GPU window (secondary): compute-stream op ms per mb | 268.09 | 294.71 | 295.41 |
| CPU launch (primary): AG-I per mb | 32.0 | 32.0 | 32.0 |
| GPU window (secondary): AG-I per mb | 33.0 | 32.2 | 32.0 |
| ops in both, per mb | 230.0 | 230.0 | 229.0 |

Structural checks:

- v2-off: no structural anomalies
- v2-p1: AllGather families appearing in mb0 forward only (candidates for outer AG): stream 47 gridX 1, stream 47 gridX 4; chosen: stream 47 gridX 4
- v1-same: AllGather families appearing in mb0 forward only (candidates for outer AG): stream 51 gridX 1, stream 43 gridX 1, stream 51 gridX 4; chosen: stream 51 gridX 4

### 2. GPU span and idle time of mb1..3 forward

| metric (mean per microbatch instance) | v2-off | v2-p1 | v1-same |
|---|---|---|---|
| Compute-stream ops per microbatch | 197.0 | 197.0 | 197.0 |
| GPU span (first op start to last op end), ms | 269.308 | 296.172 | 296.911 |
| Busy (union of op intervals), ms | 268.092 | 294.707 | 295.407 |
| Idle inside span (span minus busy), ms | 1.215 | 1.465 | 1.503 |
| Idle share of span | 0.0045 | 0.0049 | 0.0051 |
| Sum of op durations, ms | 268.092 | 294.707 | 295.407 |
| Idle gaps per microbatch (count) | 196.0 | 196.0 | 196.0 |
| Instances used | 6 | 6 | 6 |

Top 5 idle gaps, v2-off (ms, instance, previous op, next op):

| gap ms | instance | previous op | next op |
|---|---|---|---|
| 0.749 | mb1_fwd#1 | elementwise_kernel | memset |
| 0.735 | mb3_fwd#0 | elementwise_kernel | memset |
| 0.731 | mb2_fwd#0 | elementwise_kernel | memset |
| 0.729 | mb2_fwd#1 | elementwise_kernel | memset |
| 0.726 | mb3_fwd#1 | elementwise_kernel | memset |

Top 5 idle gaps, v2-p1 (ms, instance, previous op, next op):

| gap ms | instance | previous op | next op |
|---|---|---|---|
| 0.805 | mb2_fwd#1 | elementwise_kernel | memset |
| 0.763 | mb1_fwd#1 | elementwise_kernel | memset |
| 0.705 | mb3_fwd#0 | elementwise_kernel | memset |
| 0.677 | mb3_fwd#1 | elementwise_kernel | memset |
| 0.664 | mb2_fwd#0 | elementwise_kernel | memset |

Top 5 idle gaps, v1-same (ms, instance, previous op, next op):

| gap ms | instance | previous op | next op |
|---|---|---|---|
| 0.759 | mb2_fwd#0 | elementwise_kernel | memset |
| 0.753 | mb1_fwd#1 | elementwise_kernel | memset |
| 0.739 | mb3_fwd#1 | elementwise_kernel | memset |
| 0.738 | mb1_fwd#0 | elementwise_kernel | memset |
| 0.735 | mb2_fwd#1 | elementwise_kernel | memset |

### 3. GEMM and non-GEMM compute kernels (mb1..3 forward)

### v2-off

Top 10 compute-stream kernel names by total time (all compute-stream ops, per microbatch):

| kernel | count per mb | total ms per mb | median ms |
|---|---|---|---|
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 64.00 | 260.10 | 4.064 |
| vectorized_elementwise_kernel [gx65536] | 32.00 | 6.78 | 0.214 |
| vectorized_elementwise_kernel [gx16384] | 33.00 | 0.90 | 0.027 |
| memset | 65.00 | 0.11 | 0.002 |
| unrolled_elementwise_kernel [gx32768] | 1.00 | 0.08 | 0.081 |
| elementwise_kernel [gx32768] | 1.00 | 0.07 | 0.072 |
| reduce_kernel [gx1] | 1.00 | 0.05 | 0.053 |

GEMM families on the compute stream, split by whether any NCCL kernel on another stream overlaps them:

| GEMM kernel | overlapped count per mb | overlapped median ms | overlapped total ms per mb | not overlapped count per mb | not overlapped median ms | not overlapped total ms per mb |
|---|---|---|---|---|---|---|
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 62.00 | 4.066 | 251.99 | 2.00 | 4.059 | 8.12 |

Aggregates on the compute stream (per microbatch):

| group | count per mb | total ms per mb | overlapped count per mb | overlapped median ms | overlapped total ms per mb | not overlapped count per mb | not overlapped median ms | not overlapped total ms per mb |
|---|---|---|---|---|---|---|---|---|
| all GEMM | 64.00 | 260.10 | 62.00 | 4.066 | 251.99 | 2.00 | 4.059 | 8.12 |
| non-GEMM compute ops | 133.00 | 7.99 | 66.83 | 0.053 | 6.94 | 66.17 | 0.003 | 1.05 |

| GEMM duration (all GEMM on compute stream) | mean ms | p50 ms | p90 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| all GEMM | 4.064 | 3.999 | 4.152 | 4.165 | 4.173 |

GEMMs split at 4.5 ms (a fixed cut between the two duration clusters in the data), crossed with NCCL overlap:

| GEMM class | overlapped count per mb | overlapped total ms per mb | overlapped mean ms | not overlapped count per mb | not overlapped total ms per mb | not overlapped mean ms |
|---|---|---|---|---|---|---|
| <= 4.5 ms | 62.00 | 251.99 | 4.064 | 2.00 | 8.12 | 4.058 |
| > 4.5 ms | 0.00 | 0.00 | n/a | 0.00 | 0.00 | n/a |

### v2-p1

Top 10 compute-stream kernel names by total time (all compute-stream ops, per microbatch):

| kernel | count per mb | total ms per mb | median ms |
|---|---|---|---|
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 64.00 | 287.70 | 4.142 |
| vectorized_elementwise_kernel [gx65536] | 32.00 | 5.78 | 0.196 |
| vectorized_elementwise_kernel [gx16384] | 33.00 | 0.90 | 0.027 |
| elementwise_kernel [gx32768] | 1.00 | 0.10 | 0.110 |
| memset | 65.00 | 0.10 | 0.002 |
| unrolled_elementwise_kernel [gx32768] | 1.00 | 0.08 | 0.081 |
| reduce_kernel [gx1] | 1.00 | 0.04 | 0.043 |

GEMM families on the compute stream, split by whether any NCCL kernel on another stream overlaps them:

| GEMM kernel | overlapped count per mb | overlapped median ms | overlapped total ms per mb | not overlapped count per mb | not overlapped median ms | not overlapped total ms per mb |
|---|---|---|---|---|---|---|
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 49.33 | 4.224 | 227.23 | 14.67 | 4.131 | 60.47 |

Aggregates on the compute stream (per microbatch):

| group | count per mb | total ms per mb | overlapped count per mb | overlapped median ms | overlapped total ms per mb | not overlapped count per mb | not overlapped median ms | not overlapped total ms per mb |
|---|---|---|---|---|---|---|---|---|
| all GEMM | 64.00 | 287.70 | 49.33 | 4.224 | 227.23 | 14.67 | 4.131 | 60.47 |
| non-GEMM compute ops | 133.00 | 7.01 | 58.50 | 0.002 | 4.22 | 74.50 | 0.026 | 2.79 |

| GEMM duration (all GEMM on compute stream) | mean ms | p50 ms | p90 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| all GEMM | 4.495 | 4.142 | 5.510 | 5.768 | 5.773 |

GEMMs split at 4.5 ms (a fixed cut between the two duration clusters in the data), crossed with NCCL overlap:

| GEMM class | overlapped count per mb | overlapped total ms per mb | overlapped mean ms | not overlapped count per mb | not overlapped total ms per mb | not overlapped mean ms |
|---|---|---|---|---|---|---|
| <= 4.5 ms | 26.67 | 108.55 | 4.070 | 14.67 | 60.47 | 4.123 |
| > 4.5 ms | 22.67 | 118.68 | 5.236 | 0.00 | 0.00 | n/a |

### v1-same

Top 10 compute-stream kernel names by total time (all compute-stream ops, per microbatch):

| kernel | count per mb | total ms per mb | median ms |
|---|---|---|---|
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 64.00 | 288.57 | 4.144 |
| vectorized_elementwise_kernel [gx65536] | 32.00 | 5.64 | 0.148 |
| vectorized_elementwise_kernel [gx16384] | 33.00 | 0.91 | 0.027 |
| memset | 65.00 | 0.10 | 0.002 |
| unrolled_elementwise_kernel [gx32768] | 1.00 | 0.08 | 0.080 |
| elementwise_kernel [gx32768] | 1.00 | 0.07 | 0.072 |
| reduce_kernel [gx1] | 1.00 | 0.03 | 0.032 |

GEMM families on the compute stream, split by whether any NCCL kernel on another stream overlaps them:

| GEMM kernel | overlapped count per mb | overlapped median ms | overlapped total ms per mb | not overlapped count per mb | not overlapped median ms | not overlapped total ms per mb |
|---|---|---|---|---|---|---|
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 46.67 | 4.335 | 217.02 | 17.33 | 4.137 | 71.55 |

Aggregates on the compute stream (per microbatch):

| group | count per mb | total ms per mb | overlapped count per mb | overlapped median ms | overlapped total ms per mb | not overlapped count per mb | not overlapped median ms | not overlapped total ms per mb |
|---|---|---|---|---|---|---|---|---|
| all GEMM | 64.00 | 288.57 | 46.67 | 4.335 | 217.02 | 17.33 | 4.137 | 71.55 |
| non-GEMM compute ops | 133.00 | 6.84 | 49.67 | 0.002 | 3.40 | 83.33 | 0.026 | 3.43 |

| GEMM duration (all GEMM on compute stream) | mean ms | p50 ms | p90 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| all GEMM | 4.509 | 4.144 | 5.559 | 5.769 | 5.772 |

GEMMs split at 4.5 ms (a fixed cut between the two duration clusters in the data), crossed with NCCL overlap:

| GEMM class | overlapped count per mb | overlapped total ms per mb | overlapped mean ms | not overlapped count per mb | not overlapped total ms per mb | not overlapped mean ms |
|---|---|---|---|---|---|---|
| <= 4.5 ms | 24.50 | 99.91 | 4.078 | 17.33 | 71.55 | 4.128 |
| > 4.5 ms | 22.17 | 117.11 | 5.283 | 0.00 | 0.00 | n/a |

### 4. AG-I timing (mb1..3 forward)

### v2-off

| metric | value |
|---|---|
| AG-I kernels per microbatch | 32.00 |
| AG-I median duration, ms | 0.788 |
| AG-I total time per microbatch, ms | 25.20 |
| Fraction of AG-I time overlapping a GEMM | 0.695 |
| Fraction of AG-I time overlapping any compute-stream op | 0.961 |

Compute op running when each AG-I starts (top 5, count over both steps and mb1..3):

| running op at AG-I start | count |
|---|---|
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 186 |
| elementwise_kernel [gx32768] | 6 |

AG-I duration by what runs on the compute stream at its start:

| class at AG-I start | count per mb | median ms | total ms per mb |
|---|---|---|---|
| GEMM | 31.00 | 0.789 | 24.47 |
| other compute op | 1.00 | 0.729 | 0.73 |

### v2-p1

| metric | value |
|---|---|
| AG-I kernels per microbatch | 32.00 |
| AG-I median duration, ms | 2.900 |
| AG-I total time per microbatch, ms | 102.12 |
| Fraction of AG-I time overlapping a GEMM | 0.951 |
| Fraction of AG-I time overlapping any compute-stream op | 0.990 |

Compute op running when each AG-I starts (top 5, count over both steps and mb1..3):

| running op at AG-I start | count |
|---|---|
| (compute stream idle) | 100 |
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 90 |
| elementwise_kernel [gx32768] | 2 |

AG-I duration by what runs on the compute stream at its start:

| class at AG-I start | count per mb | median ms | total ms per mb |
|---|---|---|---|
| GEMM | 15.00 | 1.118 | 21.08 |
| compute stream idle | 16.67 | 4.812 | 80.78 |
| other compute op | 0.33 | 0.780 | 0.26 |

### v1-same

| metric | value |
|---|---|
| AG-I kernels per microbatch | 32.00 |
| AG-I median duration, ms | 3.728 |
| AG-I total time per microbatch, ms | 103.09 |
| Fraction of AG-I time overlapping a GEMM | 0.958 |
| Fraction of AG-I time overlapping any compute-stream op | 0.990 |

Compute op running when each AG-I starts (top 5, count over both steps and mb1..3):

| running op at AG-I start | count |
|---|---|
| (compute stream idle) | 112 |
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 80 |

AG-I duration by what runs on the compute stream at its start:

| class at AG-I start | count per mb | median ms | total ms per mb |
|---|---|---|---|
| GEMM | 13.33 | 0.852 | 15.33 |
| compute stream idle | 18.67 | 4.798 | 87.76 |

### 5. Compute-stream kernel diff vs baseline (mb1..3 forward)

### v2-p1 minus v2-off (compute-stream kernel time per microbatch)

| kernel | v2-off count/mb | v2-p1 count/mb | v2-off ms/mb | v2-p1 ms/mb | difference ms/mb |
|---|---|---|---|---|---|
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 64.00 | 64.00 | 260.104 | 287.699 | +27.595 |
| vectorized_elementwise_kernel [gx65536] | 32.00 | 32.00 | 6.779 | 5.781 | -0.999 |
| elementwise_kernel [gx32768] | 1.00 | 1.00 | 0.072 | 0.100 | +0.028 |
| memset | 65.00 | 65.00 | 0.111 | 0.098 | -0.013 |
| vectorized_elementwise_kernel [gx16384] | 33.00 | 33.00 | 0.895 | 0.904 | +0.009 |
| reduce_kernel [gx1] | 1.00 | 1.00 | 0.050 | 0.045 | -0.005 |
| unrolled_elementwise_kernel [gx32768] | 1.00 | 1.00 | 0.081 | 0.081 | +0.000 |
| TOTAL compute stream (all kernels) |  |  | 268.09 | 294.71 | +26.615 |

### v1-same minus v2-off (compute-stream kernel time per microbatch)

| kernel | v2-off count/mb | v1-same count/mb | v2-off ms/mb | v1-same ms/mb | difference ms/mb |
|---|---|---|---|---|---|
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 64.00 | 64.00 | 260.104 | 288.570 | +28.465 |
| vectorized_elementwise_kernel [gx65536] | 32.00 | 32.00 | 6.779 | 5.644 | -1.135 |
| reduce_kernel [gx1] | 1.00 | 1.00 | 0.050 | 0.033 | -0.017 |
| vectorized_elementwise_kernel [gx16384] | 33.00 | 33.00 | 0.895 | 0.906 | +0.011 |
| memset | 65.00 | 65.00 | 0.111 | 0.102 | -0.009 |
| unrolled_elementwise_kernel [gx32768] | 1.00 | 1.00 | 0.081 | 0.081 | +0.000 |
| elementwise_kernel [gx32768] | 1.00 | 1.00 | 0.072 | 0.072 | +0.000 |
| TOTAL compute stream (all kernels) |  |  | 268.09 | 295.41 | +27.315 |

### 6. Other streams (non-compute, non-NCCL), mb1..3 forward

### v2-off

None. Every mb1..3 forward kernel, memcpy and memset outside NCCL is on the compute stream.

### v2-p1

None. Every mb1..3 forward kernel, memcpy and memset outside NCCL is on the compute stream.

### v1-same

None. Every mb1..3 forward kernel, memcpy and memset outside NCCL is on the compute stream.

## Rank 0 (secondary)

Arms in this group, baseline v2-off: v2-off = `01-v2-off/nsys/d4096_L32_T4096_ga4.rank0.sqlite`, v2-p1 = `02-v2-p1/nsys/d4096_L32_T4096_ga4.rank0.sqlite`, v1-same = `03-v1-same/nsys/d4096_L32_T4096_ga4.rank0.sqlite`.

### 1. Detection summary

### v2-off (01-v2-off/nsys/d4096_L32_T4096_ga4.rank0.sqlite)

- Compute stream: 7 (1536 nvjet kernels; other streams with nvjet: none).
- AG-I: stream 39, gridX 24, ncclDevKernel_AllGather_RING_LL.
- Outer AG: stream 39, gridX 4, ncclDevKernel_AllGather_RING_LL.
- Forward CPU membership: thread-matched / any-thread launches inside the range: mb0_fwd 526/526, mb1_fwd 460/460, mb2_fwd 460/460, mb3_fwd 460/460.

| stream | NCCL kernel | gridX | role | mb0 fwd per mb | mb1..3 fwd per mb | mb*_bwd per mb | outside any phase | total |
|---|---|---|---|---|---|---|---|---|
| 35 | ReduceScatter_Sum_bf16_RING_LL | 1 | ReduceScatter | 0.00 | 0.00 | 1.25 | 0 | 10 |
| 35 | ReduceScatter_Sum_bf16_RING_LL | 4 | ReduceScatter | 0.00 | 0.00 | 8.00 | 0 | 64 |
| 35 | ReduceScatter_Sum_bf16_RING_LL | 24 | ReduceScatter | 0.00 | 0.00 | 32.00 | 0 | 256 |
| 39 | AllGather_RING_LL | 1 | other AllGather | 2.00 | 1.00 | 1.00 | 0 | 18 |
| 39 | AllGather_RING_LL | 4 | outer AG | 32.00 | 0.00 | 0.00 | 0 | 64 |
| 39 | AllGather_RING_LL | 24 | AG-I | 32.00 | 32.00 | 32.00 | 0 | 512 |

### v2-p1 (02-v2-p1/nsys/d4096_L32_T4096_ga4.rank0.sqlite)

- Compute stream: 7 (1536 nvjet kernels; other streams with nvjet: none).
- AG-I: stream 39, gridX 24, ncclDevKernel_AllGather_RING_LL.
- Outer AG: stream 47, gridX 4, ncclDevKernel_AllGather_RING_LL.
- Forward CPU membership: thread-matched / any-thread launches inside the range: mb0_fwd 526/526, mb1_fwd 460/460, mb2_fwd 460/460, mb3_fwd 460/460.

| stream | NCCL kernel | gridX | role | mb0 fwd per mb | mb1..3 fwd per mb | mb*_bwd per mb | outside any phase | total |
|---|---|---|---|---|---|---|---|---|
| 35 | ReduceScatter_Sum_bf16_RING_LL | 1 | ReduceScatter | 0.00 | 0.00 | 1.00 | 0 | 8 |
| 35 | ReduceScatter_Sum_bf16_RING_LL | 24 | ReduceScatter | 0.00 | 0.00 | 32.00 | 0 | 256 |
| 39 | AllGather_RING_LL | 1 | other AllGather | 1.00 | 1.00 | 1.00 | 0 | 16 |
| 39 | AllGather_RING_LL | 24 | AG-I | 32.00 | 32.00 | 32.00 | 0 | 512 |
| 43 | ReduceScatter_Sum_bf16_RING_LL | 1 | ReduceScatter | 0.00 | 0.00 | 0.25 | 0 | 2 |
| 43 | ReduceScatter_Sum_bf16_RING_LL | 4 | ReduceScatter | 0.00 | 0.00 | 8.00 | 0 | 64 |
| 47 | AllGather_RING_LL | 1 | other AllGather | 1.00 | 0.00 | 0.00 | 0 | 2 |
| 47 | AllGather_RING_LL | 4 | outer AG | 32.00 | 0.00 | 0.00 | 0 | 64 |

### v1-same (03-v1-same/nsys/d4096_L32_T4096_ga4.rank0.sqlite)

- Compute stream: 7 (1536 nvjet kernels; other streams with nvjet: none).
- AG-I: stream 43, gridX 24, ncclDevKernel_AllGather_RING_LL.
- Outer AG: stream 51, gridX 4, ncclDevKernel_AllGather_RING_LL.
- Forward CPU membership: thread-matched / any-thread launches inside the range: mb0_fwd 526/526, mb1_fwd 458/458, mb2_fwd 458/458, mb3_fwd 458/458.

| stream | NCCL kernel | gridX | role | mb0 fwd per mb | mb1..3 fwd per mb | mb*_bwd per mb | outside any phase | total |
|---|---|---|---|---|---|---|---|---|
| 43 | AllGather_RING_LL | 1 | other AllGather | 1.00 | 0.00 | 0.00 | 0 | 2 |
| 43 | AllGather_RING_LL | 24 | AG-I | 32.00 | 32.00 | 32.00 | 0 | 512 |
| 47 | ReduceScatter_Sum_bf16_RING_LL | 1 | ReduceScatter | 0.00 | 0.00 | 1.00 | 0 | 8 |
| 47 | ReduceScatter_Sum_bf16_RING_LL | 24 | ReduceScatter | 0.00 | 0.00 | 32.00 | 0 | 256 |
| 51 | AllGather_RING_LL | 1 | other AllGather | 1.00 | 0.00 | 0.00 | 0 | 2 |
| 51 | AllGather_RING_LL | 4 | outer AG | 32.00 | 0.00 | 8.00 | 0 | 128 |
| 55 | ReduceScatter_Sum_bf16_RING_LL | 1 | ReduceScatter | 0.00 | 0.00 | 0.25 | 0 | 2 |
| 55 | ReduceScatter_Sum_bf16_RING_LL | 4 | ReduceScatter | 0.00 | 0.00 | 8.00 | 0 | 64 |

Membership check (CPU launch primary vs GPU window secondary):

| metric (mb1..3 forward, per microbatch) | v2-off | v2-p1 | v1-same |
|---|---|---|---|
| CPU launch (primary): all ops per mb | 230.0 | 230.0 | 229.0 |
| GPU window (secondary): all ops per mb | 232.3 | 231.2 | 229.0 |
| CPU launch (primary): compute-stream ops per mb | 197.0 | 197.0 | 197.0 |
| GPU window (secondary): compute-stream ops per mb | 197.0 | 197.0 | 197.0 |
| CPU launch (primary): compute-stream op ms per mb | 267.67 | 294.78 | 295.24 |
| GPU window (secondary): compute-stream op ms per mb | 267.67 | 294.78 | 295.24 |
| CPU launch (primary): AG-I per mb | 32.0 | 32.0 | 32.0 |
| GPU window (secondary): AG-I per mb | 32.8 | 32.2 | 32.0 |
| ops in both, per mb | 230.0 | 230.0 | 229.0 |

Structural checks:

- v2-off: no structural anomalies
- v2-p1: AllGather families appearing in mb0 forward only (candidates for outer AG): stream 47 gridX 1, stream 47 gridX 4; chosen: stream 47 gridX 4
- v1-same: AllGather families appearing in mb0 forward only (candidates for outer AG): stream 51 gridX 1, stream 43 gridX 1, stream 51 gridX 4; chosen: stream 51 gridX 4

### 2. GPU span and idle time of mb1..3 forward

| metric (mean per microbatch instance) | v2-off | v2-p1 | v1-same |
|---|---|---|---|
| Compute-stream ops per microbatch | 197.0 | 197.0 | 197.0 |
| GPU span (first op start to last op end), ms | 268.891 | 296.262 | 296.755 |
| Busy (union of op intervals), ms | 267.669 | 294.780 | 295.241 |
| Idle inside span (span minus busy), ms | 1.223 | 1.482 | 1.515 |
| Idle share of span | 0.0045 | 0.0050 | 0.0051 |
| Sum of op durations, ms | 267.669 | 294.780 | 295.241 |
| Idle gaps per microbatch (count) | 196.0 | 196.0 | 196.0 |
| Instances used | 6 | 6 | 6 |

Top 5 idle gaps, v2-off (ms, instance, previous op, next op):

| gap ms | instance | previous op | next op |
|---|---|---|---|
| 0.789 | mb3_fwd#0 | elementwise_kernel | memset |
| 0.774 | mb1_fwd#0 | elementwise_kernel | memset |
| 0.762 | mb2_fwd#1 | elementwise_kernel | memset |
| 0.754 | mb1_fwd#1 | elementwise_kernel | memset |
| 0.746 | mb3_fwd#1 | elementwise_kernel | memset |

Top 5 idle gaps, v2-p1 (ms, instance, previous op, next op):

| gap ms | instance | previous op | next op |
|---|---|---|---|
| 0.802 | mb2_fwd#0 | elementwise_kernel | memset |
| 0.758 | mb1_fwd#1 | elementwise_kernel | memset |
| 0.753 | mb3_fwd#1 | elementwise_kernel | memset |
| 0.744 | mb2_fwd#1 | elementwise_kernel | memset |
| 0.728 | mb1_fwd#0 | elementwise_kernel | memset |

Top 5 idle gaps, v1-same (ms, instance, previous op, next op):

| gap ms | instance | previous op | next op |
|---|---|---|---|
| 0.760 | mb2_fwd#0 | elementwise_kernel | memset |
| 0.756 | mb3_fwd#0 | elementwise_kernel | memset |
| 0.755 | mb1_fwd#1 | elementwise_kernel | memset |
| 0.754 | mb1_fwd#0 | elementwise_kernel | memset |
| 0.752 | mb2_fwd#1 | elementwise_kernel | memset |

### 3. GEMM and non-GEMM compute kernels (mb1..3 forward)

### v2-off

Top 10 compute-stream kernel names by total time (all compute-stream ops, per microbatch):

| kernel | count per mb | total ms per mb | median ms |
|---|---|---|---|
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 64.00 | 259.73 | 4.059 |
| vectorized_elementwise_kernel [gx65536] | 32.00 | 6.73 | 0.213 |
| vectorized_elementwise_kernel [gx16384] | 33.00 | 0.90 | 0.027 |
| memset | 65.00 | 0.11 | 0.002 |
| unrolled_elementwise_kernel [gx32768] | 1.00 | 0.08 | 0.081 |
| elementwise_kernel [gx32768] | 1.00 | 0.07 | 0.072 |
| reduce_kernel [gx1] | 1.00 | 0.04 | 0.046 |

GEMM families on the compute stream, split by whether any NCCL kernel on another stream overlaps them:

| GEMM kernel | overlapped count per mb | overlapped median ms | overlapped total ms per mb | not overlapped count per mb | not overlapped median ms | not overlapped total ms per mb |
|---|---|---|---|---|---|---|
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 62.00 | 4.059 | 251.62 | 2.00 | 4.055 | 8.11 |

Aggregates on the compute stream (per microbatch):

| group | count per mb | total ms per mb | overlapped count per mb | overlapped median ms | overlapped total ms per mb | not overlapped count per mb | not overlapped median ms | not overlapped total ms per mb |
|---|---|---|---|---|---|---|---|---|
| all GEMM | 64.00 | 259.73 | 62.00 | 4.059 | 251.62 | 2.00 | 4.055 | 8.11 |
| non-GEMM compute ops | 133.00 | 7.94 | 66.67 | 0.050 | 6.89 | 66.33 | 0.002 | 1.05 |

| GEMM duration (all GEMM on compute stream) | mean ms | p50 ms | p90 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| all GEMM | 4.058 | 3.993 | 4.142 | 4.160 | 4.178 |

GEMMs split at 4.5 ms (a fixed cut between the two duration clusters in the data), crossed with NCCL overlap:

| GEMM class | overlapped count per mb | overlapped total ms per mb | overlapped mean ms | not overlapped count per mb | not overlapped total ms per mb | not overlapped mean ms |
|---|---|---|---|---|---|---|
| <= 4.5 ms | 62.00 | 251.62 | 4.058 | 2.00 | 8.11 | 4.056 |
| > 4.5 ms | 0.00 | 0.00 | n/a | 0.00 | 0.00 | n/a |

### v2-p1

Top 10 compute-stream kernel names by total time (all compute-stream ops, per microbatch):

| kernel | count per mb | total ms per mb | median ms |
|---|---|---|---|
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 64.00 | 287.78 | 4.137 |
| vectorized_elementwise_kernel [gx65536] | 32.00 | 5.79 | 0.196 |
| vectorized_elementwise_kernel [gx16384] | 33.00 | 0.90 | 0.027 |
| memset | 65.00 | 0.10 | 0.002 |
| unrolled_elementwise_kernel [gx32768] | 1.00 | 0.08 | 0.081 |
| elementwise_kernel [gx32768] | 1.00 | 0.08 | 0.073 |
| reduce_kernel [gx1] | 1.00 | 0.04 | 0.043 |

GEMM families on the compute stream, split by whether any NCCL kernel on another stream overlaps them:

| GEMM kernel | overlapped count per mb | overlapped median ms | overlapped total ms per mb | not overlapped count per mb | not overlapped median ms | not overlapped total ms per mb |
|---|---|---|---|---|---|---|
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 49.50 | 4.154 | 228.03 | 14.50 | 4.128 | 59.75 |

Aggregates on the compute stream (per microbatch):

| group | count per mb | total ms per mb | overlapped count per mb | overlapped median ms | overlapped total ms per mb | not overlapped count per mb | not overlapped median ms | not overlapped total ms per mb |
|---|---|---|---|---|---|---|---|---|
| all GEMM | 64.00 | 287.78 | 49.50 | 4.154 | 228.03 | 14.50 | 4.128 | 59.75 |
| non-GEMM compute ops | 133.00 | 7.00 | 59.33 | 0.002 | 4.28 | 73.67 | 0.026 | 2.71 |

| GEMM duration (all GEMM on compute stream) | mean ms | p50 ms | p90 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| all GEMM | 4.497 | 4.137 | 5.657 | 5.764 | 5.767 |

GEMMs split at 4.5 ms (a fixed cut between the two duration clusters in the data), crossed with NCCL overlap:

| GEMM class | overlapped count per mb | overlapped total ms per mb | overlapped mean ms | not overlapped count per mb | not overlapped total ms per mb | not overlapped mean ms |
|---|---|---|---|---|---|---|
| <= 4.5 ms | 26.83 | 108.99 | 4.062 | 14.50 | 59.75 | 4.121 |
| > 4.5 ms | 22.67 | 119.04 | 5.252 | 0.00 | 0.00 | n/a |

### v1-same

Top 10 compute-stream kernel names by total time (all compute-stream ops, per microbatch):

| kernel | count per mb | total ms per mb | median ms |
|---|---|---|---|
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 64.00 | 288.19 | 4.138 |
| vectorized_elementwise_kernel [gx65536] | 32.00 | 5.87 | 0.196 |
| vectorized_elementwise_kernel [gx16384] | 33.00 | 0.90 | 0.027 |
| memset | 65.00 | 0.10 | 0.001 |
| unrolled_elementwise_kernel [gx32768] | 1.00 | 0.08 | 0.080 |
| elementwise_kernel [gx32768] | 1.00 | 0.07 | 0.072 |
| reduce_kernel [gx1] | 1.00 | 0.03 | 0.033 |

GEMM families on the compute stream, split by whether any NCCL kernel on another stream overlaps them:

| GEMM kernel | overlapped count per mb | overlapped median ms | overlapped total ms per mb | not overlapped count per mb | not overlapped median ms | not overlapped total ms per mb |
|---|---|---|---|---|---|---|
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 51.33 | 4.268 | 236.01 | 12.67 | 4.128 | 52.18 |

Aggregates on the compute stream (per microbatch):

| group | count per mb | total ms per mb | overlapped count per mb | overlapped median ms | overlapped total ms per mb | not overlapped count per mb | not overlapped median ms | not overlapped total ms per mb |
|---|---|---|---|---|---|---|---|---|
| all GEMM | 64.00 | 288.19 | 51.33 | 4.268 | 236.01 | 12.67 | 4.128 | 52.18 |
| non-GEMM compute ops | 133.00 | 7.05 | 56.33 | 0.002 | 4.36 | 76.67 | 0.026 | 2.69 |

| GEMM duration (all GEMM on compute stream) | mean ms | p50 ms | p90 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| all GEMM | 4.503 | 4.138 | 5.616 | 5.761 | 5.765 |

GEMMs split at 4.5 ms (a fixed cut between the two duration clusters in the data), crossed with NCCL overlap:

| GEMM class | overlapped count per mb | overlapped total ms per mb | overlapped mean ms | not overlapped count per mb | not overlapped total ms per mb | not overlapped mean ms |
|---|---|---|---|---|---|---|
| <= 4.5 ms | 27.83 | 113.22 | 4.068 | 12.67 | 52.18 | 4.119 |
| > 4.5 ms | 23.50 | 122.79 | 5.225 | 0.00 | 0.00 | n/a |

### 4. AG-I timing (mb1..3 forward)

### v2-off

| metric | value |
|---|---|
| AG-I kernels per microbatch | 32.00 |
| AG-I median duration, ms | 0.797 |
| AG-I total time per microbatch, ms | 26.73 |
| Fraction of AG-I time overlapping a GEMM | 0.714 |
| Fraction of AG-I time overlapping any compute-stream op | 0.963 |

Compute op running when each AG-I starts (top 5, count over both steps and mb1..3):

| running op at AG-I start | count |
|---|---|
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 186 |
| elementwise_kernel [gx32768] | 6 |

AG-I duration by what runs on the compute stream at its start:

| class at AG-I start | count per mb | median ms | total ms per mb |
|---|---|---|---|
| GEMM | 31.00 | 0.798 | 25.98 |
| other compute op | 1.00 | 0.757 | 0.76 |

### v2-p1

| metric | value |
|---|---|
| AG-I kernels per microbatch | 32.00 |
| AG-I median duration, ms | 3.081 |
| AG-I total time per microbatch, ms | 104.02 |
| Fraction of AG-I time overlapping a GEMM | 0.952 |
| Fraction of AG-I time overlapping any compute-stream op | 0.990 |

Compute op running when each AG-I starts (top 5, count over both steps and mb1..3):

| running op at AG-I start | count |
|---|---|
| (compute stream idle) | 98 |
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 89 |
| elementwise_kernel [gx32768] | 5 |

AG-I duration by what runs on the compute stream at its start:

| class at AG-I start | count per mb | median ms | total ms per mb |
|---|---|---|---|
| GEMM | 14.83 | 1.103 | 22.04 |
| compute stream idle | 16.33 | 5.102 | 81.35 |
| other compute op | 0.83 | 0.751 | 0.63 |

### v1-same

| metric | value |
|---|---|
| AG-I kernels per microbatch | 32.00 |
| AG-I median duration, ms | 2.721 |
| AG-I total time per microbatch, ms | 103.54 |
| Fraction of AG-I time overlapping a GEMM | 0.948 |
| Fraction of AG-I time overlapping any compute-stream op | 0.990 |

Compute op running when each AG-I starts (top 5, count over both steps and mb1..3):

| running op at AG-I start | count |
|---|---|
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 98 |
| (compute stream idle) | 94 |

AG-I duration by what runs on the compute stream at its start:

| class at AG-I start | count per mb | median ms | total ms per mb |
|---|---|---|---|
| GEMM | 16.33 | 1.614 | 26.94 |
| compute stream idle | 15.67 | 5.040 | 76.60 |

### 5. Compute-stream kernel diff vs baseline (mb1..3 forward)

### v2-p1 minus v2-off (compute-stream kernel time per microbatch)

| kernel | v2-off count/mb | v2-p1 count/mb | v2-off ms/mb | v2-p1 ms/mb | difference ms/mb |
|---|---|---|---|---|---|
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 64.00 | 64.00 | 259.733 | 287.785 | +28.052 |
| vectorized_elementwise_kernel [gx65536] | 32.00 | 32.00 | 6.735 | 5.794 | -0.941 |
| memset | 65.00 | 65.00 | 0.108 | 0.097 | -0.010 |
| elementwise_kernel [gx32768] | 1.00 | 1.00 | 0.072 | 0.078 | +0.006 |
| vectorized_elementwise_kernel [gx16384] | 33.00 | 33.00 | 0.896 | 0.901 | +0.005 |
| reduce_kernel [gx1] | 1.00 | 1.00 | 0.045 | 0.044 | -0.001 |
| unrolled_elementwise_kernel [gx32768] | 1.00 | 1.00 | 0.081 | 0.081 | +0.000 |
| TOTAL compute stream (all kernels) |  |  | 267.67 | 294.78 | +27.112 |

### v1-same minus v2-off (compute-stream kernel time per microbatch)

| kernel | v2-off count/mb | v1-same count/mb | v2-off ms/mb | v1-same ms/mb | difference ms/mb |
|---|---|---|---|---|---|
| 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] | 64.00 | 64.00 | 259.733 | 288.187 | +28.454 |
| vectorized_elementwise_kernel [gx65536] | 32.00 | 32.00 | 6.735 | 5.872 | -0.863 |
| memset | 65.00 | 65.00 | 0.108 | 0.095 | -0.012 |
| reduce_kernel [gx1] | 1.00 | 1.00 | 0.045 | 0.033 | -0.012 |
| vectorized_elementwise_kernel [gx16384] | 33.00 | 33.00 | 0.896 | 0.902 | +0.005 |
| unrolled_elementwise_kernel [gx32768] | 1.00 | 1.00 | 0.081 | 0.080 | -0.000 |
| elementwise_kernel [gx32768] | 1.00 | 1.00 | 0.072 | 0.072 | +0.000 |
| TOTAL compute stream (all kernels) |  |  | 267.67 | 295.24 | +27.572 |

### 6. Other streams (non-compute, non-NCCL), mb1..3 forward

### v2-off

None. Every mb1..3 forward kernel, memcpy and memset outside NCCL is on the compute stream.

### v2-p1

None. Every mb1..3 forward kernel, memcpy and memset outside NCCL is on the compute stream.

### v1-same

None. Every mb1..3 forward kernel, memcpy and memset outside NCCL is on the compute stream.

## 7. Findings

Rank 8, mb1..3 forward, per microbatch. The +26.6 ms (v2-p1) and +27.3 ms (v1-same) over v2-off is GEMM time. It is not idle time or other kernels. Compute-stream busy time is 268.1, 294.7 and 295.4 ms, and idle time inside the span is 1.2, 1.5 and 1.5 ms. Non-GEMM compute ops take 8.0, 7.0 and 6.8 ms. All arms run the same GEMM tile (128x256_64x4_2x1_v_bz_coopA_TNN), 64 per microbatch. In v2-off every GEMM takes 4.06 ms on average and none exceeds 4.5 ms. In v2-p1, 22.7 GEMMs per microbatch take 5.24 ms on average (118.7 ms in total), and every one of them overlaps an NCCL kernel on another stream. The other 41.3 take 4.07 ms, and 14.7 of those have no NCCL overlap. In v1-same, 22.2 GEMMs take 5.28 ms, every one overlapped, and 41.8 take 4.07 ms. Relative to 4.06 ms per GEMM, the slow GEMMs account for 26.5 ms of the v2-p1 difference and 27.0 ms of the v1-same difference. v2-p1 and v1-same match on GEMM total (287.7 and 288.6 ms), slow-GEMM count and mean, and AG-I total time (102.1 and 103.1 ms). They differ in AG-I median duration (2.90 and 3.73 ms) and in AG-I starts while the compute stream is idle (16.7 and 18.7 per microbatch). In v2-off, 31 of 32 AG-I start during a GEMM and none start while the compute stream is idle. The AG-I median is 0.79 ms, the total is 25.2 ms per microbatch, and GEMMs cover 69.5% of AG-I time. In v2-p1 and v1-same, the AG-I that start while the compute stream is idle have median durations of 4.81 and 4.80 ms. One rank cannot show whether AG-I durations include waiting for peers on node 1.
