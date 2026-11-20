# E2 skew: inner all-gather spin versus late arrival, 2-node nsys

Script: `bench/e2skew_compare.py`. Inputs: nsys sqlite exports of `d4096_L32_T4096_ga4.rank{0..15}` for `01-v2-off` (fast) and `02-v2-p1` (slow). Inner all-gather (AG-I) = NCCL AllGather on the inner stream with gridX 24. Phase tags come from CPU-launch membership inside the NVTX ranges. Cross-rank comparisons use per-rank clock offsets estimated from AG-I end times (section 1). Node 0 and node 1 are never compared.

## 1. Sanity

### fwd_rest in train mode at T=4096 (ms, max over ranks)

| source | off | p1 | p1 - off |
|---|---|---|---|
| train: sweep, no nsys (e1_d4096_L32.json) | 269.08 | 296.68 | +27.61 |
| train: nsys run (nsys_d4096_L32_T4096_ga4.json) | 269.48 | 297.59 | +28.11 |
| nostep: sweep, no nsys (e1_d4096_L32.json) | 269.11 | 295.83 | +26.72 |

The nsys run logs (`node{0,1}.nsys_d4096_T4096_ga4.log`) contain 1 (off) and 1 (p1) `fwd_rest` lines, so the nsys row uses `nsys_d4096_L32_T4096_ga4.json`.

### AG-I per rank per phase (count, min-max over 16 ranks)

| phase (CPU-launch membership) | off | p1 |
|---|---|---|
| mb0_fwd | 64-64 | 64-64 |
| mb1_fwd | 64-64 | 64-64 |
| mb2_fwd | 64-64 | 64-64 |
| mb3_fwd | 64-64 | 64-64 |
| mb0_bwd | 64-64 | 64-64 |
| mb1_bwd | 64-64 | 64-64 |
| mb2_bwd | 64-64 | 64-64 |
| mb3_bwd | 64-64 | 64-64 |
| outside | 0-0 | 0-0 |

Expected: 64 per phase per rank (2 steps x 32 AG-I per microbatch). AG-I per rank in total: off 512, p1 512.

### Clock offsets within each node

Offsets are relative to the first rank of the node. Raw end skew is the max-min of the 8 raw end times of one collective. Corrected skew uses the shifted times.

| arm | rank | node | offset (ms) | offset, step 0 (ms) | offset, step 1 (ms) | step 1 minus step 0 (ms) |
|---|---|---|---|---|---|---|
| off | 0 | node 0 | +0.000 | +0.000 | +0.000 | +0.0000 |
| off | 1 | node 0 | -0.086 | -0.088 | -0.044 | +0.0443 |
| off | 2 | node 0 | +0.047 | +0.016 | +0.115 | +0.0987 |
| off | 3 | node 0 | +694.157 | +694.120 | +694.233 | +0.1128 |
| off | 4 | node 0 | -23.451 | -23.475 | -23.379 | +0.0961 |
| off | 5 | node 0 | +162.716 | +162.694 | +162.767 | +0.0734 |
| off | 6 | node 0 | +1845.887 | +1845.864 | +1845.926 | +0.0621 |
| off | 7 | node 0 | -0.437 | -0.442 | -0.434 | +0.0080 |
| off | 8 | node 1 | +0.000 | +0.000 | +0.000 | +0.0000 |
| off | 9 | node 1 | -51.766 | -51.766 | -51.766 | +0.0006 |
| off | 10 | node 1 | +1015.244 | +1015.239 | +1015.246 | +0.0068 |
| off | 11 | node 1 | -266.326 | -266.338 | -266.316 | +0.0224 |
| off | 12 | node 1 | -263.139 | -263.158 | -263.124 | +0.0344 |
| off | 13 | node 1 | -233.423 | -233.434 | -233.414 | +0.0194 |
| off | 14 | node 1 | -250.768 | -250.770 | -250.768 | +0.0018 |
| off | 15 | node 1 | -300.122 | -300.122 | -300.122 | -0.0003 |
| p1 | 0 | node 0 | +0.000 | +0.000 | +0.000 | +0.0000 |
| p1 | 1 | node 0 | +1958.930 | +1958.918 | +1958.954 | +0.0362 |
| p1 | 2 | node 0 | -186.765 | -186.792 | -186.751 | +0.0413 |
| p1 | 3 | node 0 | -268.140 | -268.155 | -268.114 | +0.0414 |
| p1 | 4 | node 0 | -283.033 | -283.045 | -283.023 | +0.0216 |
| p1 | 5 | node 0 | -99.108 | -99.116 | -99.094 | +0.0227 |
| p1 | 6 | node 0 | -267.022 | -267.023 | -267.022 | +0.0014 |
| p1 | 7 | node 0 | -21.414 | -21.413 | -21.415 | -0.0014 |
| p1 | 8 | node 1 | +0.000 | +0.000 | +0.000 | +0.0000 |
| p1 | 9 | node 1 | -21.458 | -21.458 | -21.458 | +0.0003 |
| p1 | 10 | node 1 | +1821.258 | +1821.256 | +1821.259 | +0.0030 |
| p1 | 11 | node 1 | +193.785 | +193.785 | +193.785 | +0.0002 |
| p1 | 12 | node 1 | -1.092 | -1.091 | -1.093 | -0.0018 |
| p1 | 13 | node 1 | -0.598 | -0.596 | -0.601 | -0.0054 |
| p1 | 14 | node 1 | -1.049 | -1.038 | -1.052 | -0.0135 |
| p1 | 15 | node 1 | +172.812 | +172.812 | +172.811 | -0.0015 |

| arm | node | raw end skew median (ms) | raw start skew median (ms) | corrected end skew median (ms) | corrected end skew p90 (ms) | corrected end skew max (ms) | corrected start skew median (ms) |
|---|---|---|---|---|---|---|---|
| off | node 0 | 1869.323 | 1869.370 | 0.121 | 0.162 | 0.198 | 0.323 |
| off | node 1 | 1315.358 | 1315.464 | 0.091 | 0.146 | 0.238 | 0.291 |
| p1 | node 0 | 2241.975 | 2241.952 | 0.127 | 0.208 | 0.252 | 1.082 |
| p1 | node 1 | 1842.707 | 1842.898 | 0.150 | 0.193 | 0.219 | 1.207 |

## 2. Skew and duration per node

Start skew is max(start) - min(start) over the 8 ranks of one collective, on corrected times. End skew is max(end) - min(end). `together`: start skew < 0.1 ms. `tight`: start skew >= 0.1 ms and end skew <= 0.2 x start skew, so the ends line up and the kernel waited for the last rank. `loose`: other.

### Start and end skew (ms, corrected)

| arm | node | phase group | collectives | start skew median | start skew p90 | start skew max | end skew median | end skew p90 | end skew max | together | tight | loose |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | node 0 | mb0 fwd | 64 | 0.164 | 0.381 | 3.893 | 0.096 | 0.140 | 0.151 | 5 | 5 | 54 |
| off | node 0 | mb1..3 fwd | 192 | 0.400 | 0.466 | 3.867 | 0.128 | 0.168 | 0.198 | 10 | 34 | 148 |
| off | node 0 | bwd, mb0..3 | 256 | 0.299 | 0.388 | 3.905 | 0.125 | 0.154 | 0.190 | 11 | 12 | 233 |
| off | node 1 | mb0 fwd | 64 | 0.137 | 1.685 | 4.893 | 0.084 | 0.130 | 0.206 | 17 | 12 | 35 |
| off | node 1 | mb1..3 fwd | 192 | 0.373 | 0.447 | 0.508 | 0.110 | 0.182 | 0.238 | 9 | 48 | 135 |
| off | node 1 | bwd, mb0..3 | 256 | 0.265 | 0.336 | 3.885 | 0.083 | 0.131 | 0.214 | 14 | 49 | 193 |
| p1 | node 0 | mb0 fwd | 64 | 5.551 | 6.547 | 7.406 | 0.165 | 0.222 | 0.252 | 0 | 64 | 0 |
| p1 | node 0 | mb1..3 fwd | 192 | 5.351 | 6.283 | 7.241 | 0.157 | 0.220 | 0.250 | 2 | 186 | 4 |
| p1 | node 0 | bwd, mb0..3 | 256 | 0.282 | 0.417 | 4.026 | 0.107 | 0.159 | 0.209 | 7 | 21 | 228 |
| p1 | node 1 | mb0 fwd | 64 | 5.287 | 6.367 | 7.282 | 0.177 | 0.199 | 0.219 | 0 | 64 | 0 |
| p1 | node 1 | mb1..3 fwd | 192 | 5.251 | 6.210 | 7.169 | 0.173 | 0.198 | 0.211 | 4 | 186 | 2 |
| p1 | node 1 | bwd, mb0..3 | 256 | 0.269 | 0.415 | 3.909 | 0.106 | 0.161 | 0.209 | 9 | 35 | 212 |

### AG-I duration (ms)

Durations are over all 8 ranks of each collective. Last-starter duration is the duration of the rank that started last. Others median is the median of the 7 other ranks per collective. The last two columns are medians over collectives.

| arm | node | phase group | AG-I dur median | AG-I dur p90 | AG-I dur max | last-starter dur median | others median dur | others minus last (median) |
|---|---|---|---|---|---|---|---|---|
| off | node 0 | mb0 fwd | 1.084 | 1.200 | 4.626 | 1.018 | 1.090 | 0.062 |
| off | node 0 | mb1..3 fwd | 1.066 | 1.132 | 4.610 | 0.792 | 1.078 | 0.287 |
| off | node 0 | bwd, mb0..3 | 0.916 | 1.042 | 4.591 | 0.781 | 0.940 | 0.155 |
| off | node 1 | mb0 fwd | 1.050 | 1.176 | 5.608 | 0.984 | 1.054 | 0.076 |
| off | node 1 | mb1..3 fwd | 1.054 | 1.118 | 1.196 | 0.789 | 1.073 | 0.286 |
| off | node 1 | bwd, mb0..3 | 0.909 | 1.040 | 4.601 | 0.788 | 0.939 | 0.150 |
| p1 | node 0 | mb0 fwd | 3.127 | 6.164 | 8.043 | 0.791 | 3.930 | 3.146 |
| p1 | node 0 | mb1..3 fwd | 2.910 | 6.036 | 7.968 | 0.789 | 3.633 | 2.849 |
| p1 | node 0 | bwd, mb0..3 | 0.922 | 1.049 | 4.649 | 0.779 | 0.940 | 0.163 |
| p1 | node 1 | mb0 fwd | 3.214 | 6.105 | 8.001 | 0.785 | 4.088 | 3.294 |
| p1 | node 1 | mb1..3 fwd | 3.147 | 6.075 | 7.776 | 0.784 | 4.052 | 3.272 |
| p1 | node 1 | bwd, mb0..3 | 0.923 | 1.045 | 4.643 | 0.785 | 0.950 | 0.160 |

### AG-I duration by collective kind (mb1..3 forward)

| arm | kind | collectives | AG-I dur median, all 8 ranks | last-starter dur median | others median dur |
|---|---|---|---|---|---|
| off | together | 19 | 0.758 | 0.758 | 0.757 |
| off | tight | 82 | 1.058 | 0.787 | 1.067 |
| off | loose | 283 | 1.067 | 0.792 | 1.078 |
| p1 | together | 6 | 0.772 | 0.757 | 0.769 |
| p1 | tight | 372 | 3.226 | 0.788 | 4.001 |
| p1 | loose | 6 | 0.778 | 0.749 | 0.789 |

### Spin check: duration of non-last ranks against the wait for the last rank (mb1..3 forward)

For a rank that started early, a pure wait gives duration = (latest corrected end) - (own corrected start). The fit is duration = a + b x (latest end - own start), over all non-last ranks. A pure wait gives b near 1 and a near 0.

| arm | non-last ranks | slope b | intercept a (ms) | median duration minus wait (ms) | p90 of |duration minus wait| (ms) |
|---|---|---|---|---|---|
| off | 2688 | 0.943 | 0.006 | -0.043 | 0.128 |
| p1 | 2688 | 0.996 | -0.076 | -0.084 | 0.171 |

## 3. Which rank starts last (mb1..3 forward, corrected times)

Histogram of the last-starting rank over all mb1..3 forward collectives (both steps):

| arm | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 87 | 5 | 0 | 81 | 0 | 1 | 16 | 2 | 71 | 0 | 15 | 1 | 98 | 0 | 7 | 0 |
| p1 | 25 | 16 | 31 | 32 | 16 | 25 | 24 | 23 | 31 | 18 | 24 | 15 | 34 | 20 | 25 | 25 |

| arm | node | most often last (share) | top rank per step | same top rank in both steps |
|---|---|---|---|---|
| off | node 0 | rank 0: 87/192 | step 0: rank 0 (45/96); step 1: rank 3 (47/96) | no |
| off | node 1 | rank 12: 98/192 | step 0: rank 8 (46/96); step 1: rank 12 (53/96) | no |
| p1 | node 0 | rank 3: 32/192 | step 0: rank 3 (19/96); step 1: rank 2 (19/96) | no |
| p1 | node 1 | rank 12: 34/192 | step 0: rank 12 (21/96); step 1: rank 8 (19/96) | no |

## 4. Why the last-starting rank started late (mb1..3 forward)

The class of each collective is decided on its own rank, so it uses raw times. Lead = GPU start minus CUPTI runtime-call end (enqueue). pgap = GPU start minus end of the previous op on the inner stream. run left = end of the compute-stream op running at the start, minus the start. Rules in `classify()`: (a) the previous op ended within 0.02 ms before start and the collective was enqueued before that end; (b) enqueued within 0.05 ms before start with the stream idle before; (c) enqueued at least 0.05 ms before start with the stream idle for at least 0.02 ms before start.

| arm | class | last starters in class | median lead (ms) | median pgap (ms) | median AG-I dur of last starter (ms) | compute stream on GEMM at start | compute stream idle at start | running op ends within 0.05 ms after start |
|---|---|---|---|---|---|---|---|---|
| off | (a) stream-serialized: enqueued behind the previous op on the inner stream | 0 / 384 | n/a | n/a | n/a | 0 | 0 | 0 |
| off | (b) CPU-bound: enqueued just before start, inner stream idle before | 0 / 384 | n/a | n/a | n/a | 0 | 0 | 0 |
| off | (c) launched early, GPU start delayed while the inner stream was idle | 384 / 384 | 1395.690 | 7.601 | 0.790 | 372 | 0 | 12 |
| off | unclassifiable (no previous op on the stream) | 0 / 384 | n/a | n/a | n/a | 0 | 0 | 0 |
| p1 | (a) stream-serialized: enqueued behind the previous op on the inner stream | 0 / 384 | n/a | n/a | n/a | 0 | 0 | 0 |
| p1 | (b) CPU-bound: enqueued just before start, inner stream idle before | 0 / 384 | n/a | n/a | n/a | 0 | 0 | 0 |
| p1 | (c) launched early, GPU start delayed while the inner stream was idle | 384 / 384 | 1434.190 | 8.526 | 0.787 | 370 | 2 | 12 |
| p1 | unclassifiable (no previous op on the stream) | 0 / 384 | n/a | n/a | n/a | 0 | 0 | 0 |

Class (c) detail. The waited-for event is the record with the same event id and sync id. Its predecessor is the last op on the recording stream before the record. `gating` means that predecessor ends within 0.05 ms of the AG start.

| arm | class (c) last starters | no cudaStreamWaitEvent in window | gating (event ends at AG start) | event completed before AG start | event ends after AG start | no gating op found |
|---|---|---|---|---|---|---|
| off | 384 | 0 | 0 | 360 | 0 | 24 |
| p1 | 384 | 0 | 2 | 358 | 0 | 24 |

- off: no gating event.
- p1: gating events were recorded after: vectorized_elementwise_kernel [gx16384] (2).

Blocking GEMM: the compute-stream op running on the rank at the AG-I start (mb1..3 forward). `ends within 0.05 ms` counts AG-I that start at most 0.05 ms before that op ends.

| arm | set | AG-I | GEMM running at start | GEMM ends within 0.05 ms after start | blocking GEMM median dur (ms) | blocking GEMM slow (>4.5 ms) | median GEMM time left at start (ms), p10-p90 |
|---|---|---|---|---|---|---|---|
| off | every rank | 3072 | 2975 | 94 | 3.951 | 0 | 0.146 (0.142-0.149) |
| off | non-last ranks | 2688 | 2603 | 82 | 3.949 | 0 | 0.146 (0.142-0.148) |
| off | last starters | 384 | 372 | 12 | 3.982 | 0 | 0.147 (0.142-0.150) |
| p1 | every rank | 3072 | 1476 | 54 | 3.954 | 0 | 0.146 (0.143-0.149) |
| p1 | non-last ranks | 2688 | 1106 | 42 | 3.953 | 0 | 0.146 (0.143-0.148) |
| p1 | last starters | 384 | 370 | 12 | 3.957 | 0 | 0.146 (0.143-0.149) |


Collective kind (corrected) and class of the last starter (mb1..3 forward):

- off: loose kind, class c: 283; tight kind, class c: 82; together kind, class c: 19.
- p1: loose kind, class c: 6; tight kind, class c: 372; together kind, class c: 6.

## 5. GEMM slowdown and early arrival of the overlapping AG-I (mb1..3 forward)

Each GEMM on the compute stream is matched to the AG-I on the same rank that overlaps it most. That AG-I is `early` when its rank started at least 0.1 ms before the last rank of its node (corrected times). Otherwise it is `last or near-last`. Slow means duration above 4.5 ms. Per-microbatch values average over the 6 instances (2 steps x mb1..3).

### off: per rank, per microbatch

| rank | node | slow GEMM per mb | GEMM per mb | early AG-I per mb | rank was last starter (share of its AG-I) | AG-I median dur (ms) |
|---|---|---|---|---|---|---|
| 0 | node 0 | 0.17 | 64.0 | 6.83 | 0.453 | 0.796 |
| 1 | node 0 | 0.17 | 64.0 | 29.00 | 0.026 | 1.103 |
| 2 | node 0 | 0.33 | 64.0 | 29.83 | 0.000 | 1.087 |
| 3 | node 0 | 0.17 | 64.0 | 11.17 | 0.422 | 0.797 |
| 4 | node 0 | 0.17 | 64.0 | 29.83 | 0.000 | 1.079 |
| 5 | node 0 | 0.17 | 64.0 | 29.33 | 0.005 | 1.081 |
| 6 | node 0 | 0.17 | 64.0 | 27.67 | 0.083 | 1.070 |
| 7 | node 0 | 0.17 | 64.0 | 28.33 | 0.010 | 1.072 |
| 8 | node 1 | 0.00 | 64.0 | 9.33 | 0.370 | 0.777 |
| 9 | node 1 | 0.00 | 64.0 | 30.00 | 0.000 | 1.081 |
| 10 | node 1 | 0.00 | 64.0 | 23.33 | 0.078 | 1.010 |
| 11 | node 1 | 0.00 | 64.0 | 29.67 | 0.005 | 1.094 |
| 12 | node 1 | 0.00 | 64.0 | 7.50 | 0.510 | 0.795 |
| 13 | node 1 | 0.00 | 64.0 | 29.33 | 0.000 | 1.079 |
| 14 | node 1 | 0.00 | 64.0 | 29.33 | 0.036 | 1.079 |
| 15 | node 1 | 0.00 | 64.0 | 30.33 | 0.000 | 1.081 |

### p1: per rank, per microbatch

| rank | node | slow GEMM per mb | GEMM per mb | early AG-I per mb | rank was last starter (share of its AG-I) | AG-I median dur (ms) |
|---|---|---|---|---|---|---|
| 0 | node 0 | 23.00 | 64.0 | 26.00 | 0.130 | 2.934 |
| 1 | node 0 | 24.83 | 64.0 | 27.83 | 0.083 | 2.839 |
| 2 | node 0 | 23.17 | 64.0 | 25.50 | 0.161 | 3.936 |
| 3 | node 0 | 22.50 | 64.0 | 25.17 | 0.167 | 2.269 |
| 4 | node 0 | 25.00 | 64.0 | 27.67 | 0.083 | 3.384 |
| 5 | node 0 | 24.00 | 64.0 | 25.67 | 0.130 | 2.998 |
| 6 | node 0 | 24.33 | 64.0 | 26.00 | 0.125 | 2.660 |
| 7 | node 0 | 24.00 | 64.0 | 26.50 | 0.120 | 2.801 |
| 8 | node 1 | 22.67 | 64.0 | 25.50 | 0.161 | 3.346 |
| 9 | node 1 | 24.33 | 64.0 | 27.50 | 0.094 | 2.934 |
| 10 | node 1 | 23.17 | 64.0 | 26.00 | 0.125 | 2.841 |
| 11 | node 1 | 26.17 | 64.0 | 27.00 | 0.078 | 2.964 |
| 12 | node 1 | 22.33 | 64.0 | 25.00 | 0.177 | 2.707 |
| 13 | node 1 | 24.67 | 64.0 | 26.83 | 0.104 | 2.948 |
| 14 | node 1 | 22.83 | 64.0 | 26.17 | 0.130 | 3.839 |
| 15 | node 1 | 23.83 | 64.0 | 25.33 | 0.130 | 3.514 |

### GEMM by the overlapping AG-I on the same rank (16 ranks, per microbatch)

| arm | GEMM category | GEMM per mb | slow per mb | slow fraction | GEMM median dur (ms) | slow GEMM median dur (ms) |
|---|---|---|---|---|---|---|
| off | overlapping AG-I early | 47.57 | 0.09 | 0.002 | 4.085 | 4.699 |
| off | overlapping AG-I last or near-last | 14.42 | 0.00 | 0.000 | 4.043 | n/a |
| off | no AG-I overlap | 2.01 | 0.00 | 0.000 | 4.079 | n/a |
| p1 | overlapping AG-I early | 40.42 | 23.80 | 0.589 | 4.681 | 5.273 |
| p1 | overlapping AG-I last or near-last | 9.71 | 0.00 | 0.000 | 4.084 | n/a |
| p1 | no AG-I overlap | 13.88 | 0.00 | 0.000 | 4.103 | n/a |

Pearson r over (rank, step, mb) points, x = early AG-I count and y = slow GEMM count:

| arm | points | Pearson r |
|---|---|---|
| off | 96 | -0.010 |
| p1 | 96 | 0.547 |

## 6. Timeline excerpt: node 1, step index 1, mb1 forward, first 6 AG-I (corrected times)

Step index 1 is the second profiled step (NVTX `train_T4096_ga4_step4` in these files; the first is `step3`). t0 is the earliest corrected start of the first AG-I of the excerpt across the 8 ranks. Launch is the CUPTI runtime-call start. `running at start` is the compute-stream op on that rank at the AG-I start.

### off (collective indices k = 320..325 on each rank)

| AG-I # | rank | launch (ms) | start (ms) | end (ms) | dur (ms) | running at start |
|---|---|---|---|---|---|---|
| 1 | 8 | -736.547 | +0.029 | +0.755 | 0.725 | elementwise_kernel [gx32768] |
| 1 | 9 | -731.260 | +0.053 | +0.812 | 0.759 | elementwise_kernel [gx32768] |
| 1 | 10 | -737.917 | +0.015 | +0.875 | 0.860 | (compute stream idle) |
| 1 | 11 | -740.074 | +0.095 | +0.916 | 0.821 | elementwise_kernel [gx32768] |
| 1 | 12 | -736.090 | +0.077 | +0.884 | 0.806 | elementwise_kernel [gx32768] |
| 1 | 13 | -735.053 | +0.095 | +0.875 | 0.779 | elementwise_kernel [gx32768] |
| 1 | 14 | -739.080 | +0.110 | +0.859 | 0.750 | elementwise_kernel [gx32768] |
| 1 | 15 | -740.677 | +0.000 | +0.784 | 0.784 | (compute stream idle) |
| 2 | 8 | -736.080 | +4.609 | +5.380 | 0.771 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 2 | 9 | -730.825 | +4.625 | +5.395 | 0.770 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 2 | 10 | -737.496 | +4.705 | +5.460 | 0.755 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 2 | 11 | -739.594 | +4.720 | +5.487 | 0.767 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 2 | 12 | -735.639 | +4.717 | +5.503 | 0.786 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 2 | 13 | -734.589 | +4.698 | +5.493 | 0.795 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 2 | 14 | -738.670 | +4.665 | +5.465 | 0.800 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 2 | 15 | -740.261 | +4.589 | +5.380 | 0.791 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 3 | 8 | -735.255 | +13.028 | +13.794 | 0.766 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 3 | 9 | -730.108 | +12.944 | +13.795 | 0.850 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 3 | 10 | -736.748 | +13.071 | +13.859 | 0.788 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 3 | 11 | -738.844 | +13.045 | +13.872 | 0.827 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 3 | 12 | -734.882 | +13.108 | +13.898 | 0.789 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 3 | 13 | -733.814 | +13.026 | +13.889 | 0.863 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 3 | 14 | -737.919 | +12.960 | +13.861 | 0.901 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 3 | 15 | -739.474 | +12.882 | +13.800 | 0.918 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 4 | 8 | -734.507 | +21.448 | +22.211 | 0.763 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 4 | 9 | -729.412 | +21.262 | +22.209 | 0.946 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 4 | 10 | -736.012 | +21.461 | +22.248 | 0.788 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 4 | 11 | -738.195 | +21.363 | +22.231 | 0.868 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 4 | 12 | -734.158 | +21.484 | +22.243 | 0.759 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 4 | 13 | -733.037 | +21.365 | +22.234 | 0.870 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 4 | 14 | -737.240 | +21.257 | +22.216 | 0.959 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 4 | 15 | -738.746 | +21.169 | +22.218 | 1.048 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 5 | 8 | -733.779 | +29.866 | +30.649 | 0.783 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 5 | 9 | -728.724 | +29.574 | +30.647 | 1.073 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 5 | 10 | -735.303 | +29.844 | +30.657 | 0.813 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 5 | 11 | -737.501 | +29.666 | +30.658 | 0.991 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 5 | 12 | -733.461 | +29.890 | +30.683 | 0.794 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 5 | 13 | -732.330 | +29.712 | +30.674 | 0.963 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 5 | 14 | -736.534 | +29.563 | +30.647 | 1.084 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 5 | 15 | -738.022 | +29.540 | +30.646 | 1.106 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 6 | 8 | -733.034 | +38.285 | +39.071 | 0.787 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 6 | 9 | -728.088 | +37.947 | +39.071 | 1.124 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 6 | 10 | -734.590 | +38.242 | +39.063 | 0.820 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 6 | 11 | -736.785 | +37.984 | +39.103 | 1.118 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 6 | 12 | -732.812 | +38.299 | +39.131 | 0.832 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 6 | 13 | -731.610 | +38.043 | +39.123 | 1.081 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 6 | 14 | -735.789 | +37.986 | +39.097 | 1.111 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 6 | 15 | -737.365 | +37.922 | +39.073 | 1.151 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |

### p1 (collective indices k = 320..325 on each rank)

| AG-I # | rank | launch (ms) | start (ms) | end (ms) | dur (ms) | running at start |
|---|---|---|---|---|---|---|
| 1 | 8 | -750.603 | +0.054 | +0.788 | 0.734 | elementwise_kernel [gx32768] |
| 1 | 9 | -756.773 | +0.006 | +0.795 | 0.789 | (compute stream idle) |
| 1 | 10 | -754.344 | +0.092 | +0.847 | 0.755 | elementwise_kernel [gx32768] |
| 1 | 11 | -749.421 | +0.076 | +0.837 | 0.760 | elementwise_kernel [gx32768] |
| 1 | 12 | -756.860 | +0.000 | +0.813 | 0.813 | (compute stream idle) |
| 1 | 13 | -754.459 | +0.079 | +0.801 | 0.721 | elementwise_kernel [gx32768] |
| 1 | 14 | -755.506 | +0.035 | +0.808 | 0.773 | (compute stream idle) |
| 1 | 15 | -748.978 | +0.022 | +0.792 | 0.770 | (compute stream idle) |
| 2 | 8 | -750.180 | +0.792 | +5.301 | 4.509 | (compute stream idle) |
| 2 | 9 | -756.328 | +4.598 | +5.375 | 0.776 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 2 | 10 | -753.948 | +0.852 | +5.391 | 4.540 | (compute stream idle) |
| 2 | 11 | -748.982 | +4.643 | +5.431 | 0.789 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 2 | 12 | -756.448 | +0.817 | +5.398 | 4.581 | (compute stream idle) |
| 2 | 13 | -754.053 | +0.806 | +5.359 | 4.553 | (compute stream idle) |
| 2 | 14 | -755.073 | +0.812 | +5.331 | 4.519 | (compute stream idle) |
| 2 | 15 | -748.522 | +4.595 | +5.353 | 0.757 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 3 | 8 | -749.389 | +10.437 | +13.624 | 3.186 | (compute stream idle) |
| 3 | 9 | -755.622 | +12.910 | +13.705 | 0.795 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 3 | 10 | -753.211 | +10.484 | +13.696 | 3.212 | (compute stream idle) |
| 3 | 11 | -748.195 | +12.955 | +13.736 | 0.782 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 3 | 12 | -755.774 | +10.478 | +13.702 | 3.223 | (compute stream idle) |
| 3 | 13 | -753.295 | +10.374 | +13.662 | 3.288 | (compute stream idle) |
| 3 | 14 | -754.349 | +10.394 | +13.635 | 3.241 | (compute stream idle) |
| 3 | 15 | -747.762 | +9.102 | +13.615 | 4.514 | (compute stream idle) |
| 4 | 8 | -748.694 | +19.646 | +24.142 | 4.496 | (compute stream idle) |
| 4 | 9 | -754.885 | +17.418 | +24.211 | 6.793 | (compute stream idle) |
| 4 | 10 | -752.487 | +23.502 | +24.282 | 0.780 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 4 | 11 | -747.495 | +21.267 | +24.289 | 3.022 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 4 | 12 | -755.058 | +23.548 | +24.314 | 0.767 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 4 | 13 | -752.582 | +23.438 | +24.296 | 0.858 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 4 | 14 | -753.617 | +23.423 | +24.288 | 0.864 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 4 | 15 | -746.997 | +22.441 | +24.233 | 1.792 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 5 | 8 | -747.973 | +29.300 | +32.384 | 3.083 | (compute stream idle) |
| 5 | 9 | -754.148 | +27.705 | +32.346 | 4.641 | (compute stream idle) |
| 5 | 10 | -751.827 | +28.066 | +32.376 | 4.310 | (compute stream idle) |
| 5 | 11 | -746.781 | +30.407 | +32.428 | 2.021 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 5 | 12 | -754.389 | +28.113 | +32.462 | 4.349 | (compute stream idle) |
| 5 | 13 | -751.917 | +31.731 | +32.523 | 0.792 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 5 | 14 | -752.936 | +31.736 | +32.524 | 0.788 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 5 | 15 | -746.251 | +31.295 | +32.470 | 1.176 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 6 | 8 | -747.224 | +42.341 | +43.137 | 0.796 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 6 | 9 | -753.488 | +37.323 | +43.138 | 5.815 | (compute stream idle) |
| 6 | 10 | -751.116 | +37.650 | +43.100 | 5.451 | (compute stream idle) |
| 6 | 11 | -746.056 | +35.480 | +43.037 | 7.556 | (compute stream idle) |
| 6 | 12 | -753.725 | +37.725 | +42.970 | 5.245 | (compute stream idle) |
| 6 | 13 | -751.190 | +40.041 | +43.009 | 2.968 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 6 | 14 | -752.272 | +40.066 | +43.093 | 3.026 | 128x256_64x4_2x1_v_bz_coopA_TNN [gx2] |
| 6 | 15 | -745.504 | +35.992 | +43.122 | 7.130 | (compute stream idle) |

## 7. Findings

In mb1..3 forward, the long p1 AG-I kernels are waits for the last-arriving rank. They are not slow kernels. Median corrected start skew within a node is 5.35 ms (node 0) and 5.25 ms (node 1) in p1, against 0.40 ms and 0.37 ms in off. Median end skew is 0.16 ms and 0.17 ms in p1, and 0.13 ms and 0.11 ms in off. For non-last ranks, duration fits the wait for the last rank with slope 0.996 and intercept -0.08 ms (2688 ranks; off slope 0.943). The last starter's AG-I median is 0.79 ms in both arms, while the other ranks' median is 3.63 ms (node 0) and 4.05 ms (node 1) in p1. When all 8 ranks start together (6 collectives in p1, 19 in off), the AG-I median is 0.76 ms in off and 0.77 ms in p1.

The latest rank's delay is on the GPU, not the CPU and not an event. Its AG-I was enqueued a median 1.43 s before it started (p1; 1.40 s off). None of the 384 p1 last starters had a previous inner-stream op end just before its start (class a) or a late enqueue (class b). Of the waited events, 358 had completed earlier, 2 gated the start, and 24 had no gating op in the trace. Its AG-I started a median 0.146 ms before the end of the GEMM running on its compute stream (p10-p90 0.143-0.149 ms), and 370 of 384 had a GEMM running at start. That GEMM had a median duration of 3.96 ms, and none was slow. Early ranks in p1 had a GEMM running at AG start in 1106 of 2688 cases, against 2603 of 2688 in off, so 59% of early-rank AG starts in p1 found no GEMM on the compute stream. Slow GEMMs (>4.5 ms) are 23.8 per microbatch across 16 ranks in p1. Of those GEMMs, 58.9% overlapped an early AG-I, and none overlapped a last-or-near-last AG-I. Per-rank Pearson r between early-AG-I count and slow-GEMM count is 0.55 (96 points, p1) and -0.01 (off). Whether the 0.15 ms tail is SM availability held by the GEMM is inferred from this fixed lead and was not measured.

## Anomalies and checks

- off node 0: AG-I count 512 on each of the 8 ranks; no mismatch.
- off node 1: AG-I count 512 on each of the 8 ranks; no mismatch.
- p1 node 0: AG-I count 512 on each of the 8 ranks; no mismatch.
- p1 node 1: AG-I count 512 on each of the 8 ranks; no mismatch.
- Collectives whose 8 ranks disagree on (step, phase): 0.
- off: inner AG (stream, gridX) over the 16 files: [(39, 24)]. Outer AG: ['(39, 4)'].
- off: unmatched ops 0; multi-phase matches 0; duplicate runtime correlation ids 0; duplicate event records 0; cudaStreamWaitEvent rows with another name 0; train NVTX ranges per file [2].
- off: AG-I outside any phase 0; outside any step 0; last-starter unclassifiable 0 (mb1..3 forward); waited event completes after the AG start 0; AG-I whose waited event has no gating op (record not in trace, or no earlier op on the recording stream) 1312 (all phases).
- p1: inner AG (stream, gridX) over the 16 files: [(39, 24)]. Outer AG: ['(47, 4)'].
- p1: unmatched ops 0; multi-phase matches 0; duplicate runtime correlation ids 0; duplicate event records 0; cudaStreamWaitEvent rows with another name 0; train NVTX ranges per file [2].
- p1: AG-I outside any phase 0; outside any step 0; last-starter unclassifiable 0 (mb1..3 forward); waited event completes after the AG start 0; AG-I whose waited event has no gating op (record not in trace, or no earlier op on the recording stream) 202 (all phases).
