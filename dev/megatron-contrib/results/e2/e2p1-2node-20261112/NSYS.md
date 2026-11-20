# E2 Nsight Systems: DP-outer (O) vs DP-inner (I) all-gather streams, flag OFF vs ON

Setup: 2 nodes, 16 GPUs, mesh (2, 8), hidden 4096, 32 layers, GA=4. Traces are
`off|on/nsys/d4096_L32_T{256,1024}_ga4.rank{0,8}.nsys-rep`. Numbers are from rank 8
unless a line says otherwise. Rank 0 is compared at the end of each T section.
Analysis script: `bench/e2_nsys_streams.py` (`python3 e2_nsys_streams.py OFF.sqlite ON.sqlite`,
prints Markdown). Exported sqlite files and per-rank output are in `/tmp/e2an/`.

## Method and assumptions

- **Steps.** Each file holds two profiled steps (NVTX `train_*_step3`, `train_*_step4`). Each step
  has `mb0..mb3` forward and backward ranges and an `optimizer` range. Values are per-step means
  over the two steps.
- **Kernel membership.** A kernel belongs to an NVTX range if its CPU launch call starts inside
  the range. NVTX ranges are CPU-side. The GPU runs well behind them: at T=1024 the forward of
  mb1 is about 22 ms on the CPU but 79 ms on the GPU. "GPU span" below means first kernel start to
  last kernel end among the kernels launched in a range. Forward GPU spans of consecutive
  microbatches do not overlap.
- **O vs I.** The NCCL kernel `gridX` value identifies the communicator. `gridX=4` is the 2-rank
  outer communicator (O). `gridX=24` is the 8-rank inner communicator (I). The NVTX
  communicator-ID payload is not exported to sqlite, so this is inferred from the data:
  - `gridX=4` all-gathers occur only in `mb0_fwd`, 33 per step, alternating with `gridX=24`.
  - `mb1..3` forward and every backward range contain only `gridX=24` all-gathers (validation table).
  - `gridX=1` kernels are tiny root-level unshards. In each `mb0_fwd` the first one is O. In ON it
    runs on stream 47, which confirms the rule. The same ordinal is used for OFF.
- **Reduce-scatter.** The same rule applies: `gridX=4` is O-RS, which is on stream 43 in ON and
  stream 35 in OFF. O-RS only occurs in `bwd3`.
- **GEMM** means kernels whose name contains `nvjet`. The compute stream is 7 in both files.
- **Overlap fractions** are the mean of per-step ratios.

Validation, mean all-gather kernels per range and step (T=1024 and T=256 identical):

| file | range | all-gather kernels (comm@stream = count) |
|---|---|---|
| OFF | mb0_fwd | I@39 = 33, O@39 = 33 |
| OFF | mb1_fwd, mb3_fwd, mb0_bwd, mb3_bwd | I@39 = 33 |
| ON | mb0_fwd | I@39 = 33, O@47 = 33 |
| ON | mb1_fwd, mb3_fwd, mb0_bwd, mb3_bwd | I@39 = 33 |

## Q1. Streams carrying all-gather kernels in mb0_fwd

| T | file | stream | comm | kernels/step | total ms/step |
|---|---|---|---|---|---|
| 256 | OFF | 39 | I | 33 | 24.29 |
| 256 | OFF | 39 | O | 33 | 26.63 |
| 256 | ON | 39 | I | 33 | 28.22 |
| 256 | ON | 47 | O | 33 | 19.98 |
| 1024 | OFF | 39 | I | 33 | 27.14 |
| 1024 | OFF | 39 | O | 33 | 27.28 |
| 1024 | ON | 39 | I | 33 | 40.05 |
| 1024 | ON | 47 | O | 33 | 25.89 |

ON puts O on stream 47 and I on stream 39. OFF puts both on stream 39. This matches the
expected layout. In ON the outer reduce-scatter also moves to stream 43.

Rank 0 agrees on streams and counts. Its O total in OFF at T=1024 (62.9 ms) is inflated by one
root-level kernel that waits about 74 ms in the first step. Use medians for rank 0.

## Q2. O/I overlap in mb0_fwd and single-kernel durations

| metric | T=256 OFF | T=256 ON | T=1024 OFF | T=1024 ON |
|---|---|---|---|---|
| total O kernel time, ms | 26.63 | 19.98 | 27.28 | 25.89 |
| time an O and an I kernel run together, ms | 0.00 | 16.73 | 0.00 | 12.46 |
| concurrent time / total O kernel time | 0.000 | 0.827 | 0.000 | 0.485 |
| O kernels overlapping any I kernel | 0 of 33 | 31 of 33 | 0 of 33 | 22 of 33 |
| O mean / median duration, ms | 0.807 / 0.764 | 0.606 / 0.627 | 0.827 / 0.803 | 0.785 / 0.726 |
| I mean / median duration in mb0_fwd, ms | 0.736 / 0.739 | 0.855 / 0.878 | 0.822 / 0.782 | 1.214 / 1.019 |
| I mean / median duration in mb1..3 fwd (no O traffic), ms | 0.813 / 0.842 | 0.856 / 0.855 | 0.839 / 0.865 | 1.291 / 1.138 |

Running concurrently does not stretch O. Its median is equal or lower in ON.
I is stretched in ON. At T=1024 the stretch also appears in mb1..3, where no O kernel runs, so O
overlap is not the only cause. At T=256 the mb1..3 I durations barely change (median 0.842 to 0.855 ms).

Rank 0 agrees on the streams, the 33 kernels per step, and the concurrency pattern (T=1024
concurrent fraction 0.538, T=256 0.621). Its mean O durations are skewed by the root wait noted in Q1.

## Q3. mb0_fwd wall time and GEMM work

| metric | T=256 OFF | T=256 ON | T=1024 OFF | T=1024 ON |
|---|---|---|---|---|
| NVTX mb0_fwd duration (CPU side), ms | 30.47 | 30.00 | 29.36 | 32.79 |
| GPU span of kernels launched in mb0_fwd, ms | 57.01 | 34.26 | 92.72 | 87.96 |
| GEMM kernels per step in mb0_fwd | 64 | 64 | 64 | 64 |
| sum GEMM kernel time in mb0_fwd, ms | 24.13 | 27.53 | 76.43 | 82.75 |
| GEMM time overlapping any NCCL kernel, ms | 18.57 | 25.11 | 38.61 | 48.85 |
| GEMM time overlapping NCCL / GEMM time | 0.770 | 0.912 | 0.505 | 0.590 |
| mean GEMM kernel duration in mb0_fwd, ms | 0.377 | 0.430 | 1.194 | 1.293 |
| mean GEMM kernel duration in mb1..3 fwd, ms | 0.375 | 0.416 | 1.175 | 1.279 |

The NVTX mb0_fwd duration is not the GPU time, so the GPU span is the wall-time figure.
At T=256 the flag shortens the GPU span of mb0_fwd by 22.8 ms (57.0 to 34.3 ms). The O and I
all-gathers no longer queue behind each other on one stream, which is the likely source of this gain.
At T=1024 the gain is 4.8 ms (92.7 to 88.0 ms).
The ON GEMMs are longer, and more of their time is covered by NCCL (0.59 vs 0.51 at T=1024).

Rank 0 gives the same direction at T=256 (GPU span 54.9 vs 51.2 ms, with a 34 ms root wait in the
ON first step), so the rank 0 span is less reliable than rank 8.

## Q4. Forward of mb1..3 (NVTX mb1_fwd, mb2_fwd, mb3_fwd)

All three ranges exist, so no fallback is needed. The GPU forward of one microbatch (span of
mb1..3 divided by 3) is:

| T | OFF GPU fwd per microbatch, ms | ON GPU fwd per microbatch, ms | JSON fwd_rest (rank 0), OFF / ON |
|---|---|---|---|
| 256 | 32.9 | 32.5 | 32.92 / 32.53 |
| 1024 | 78.6 | 85.4 | 78.57 / 85.43 |

The JSON fwd_rest matches the GPU span, not the CPU NVTX time. The prompt quoted 78.8 and 84.4 ms.
This report uses the values in the JSON files (78.6 and 85.4 ms).

Summary of the mb1..3 forward window (mean per step, three windows):

| T | file | GEMM kernels | GEMM with no NCCL overlap: count / mean ms | GEMM with NCCL overlap: count / mean ms | GPU span, ms | compute-stream busy, ms | compute-stream idle, ms | idle gaps >= 0.05 ms: count / sum ms |
|---|---|---|---|---|---|---|---|---|
| 256 | OFF | 192 | 6.0 / 0.252 | 186.0 / 0.379 | 98.61 | 76.59 | 22.02 | 95.0 / 21.37 |
| 256 | ON | 192 | 6.0 / 0.254 | 186.0 / 0.422 | 97.65 | 83.87 | 13.78 | 52.0 / 12.01 |
| 1024 | OFF | 192 | 6.0 / 1.003 | 186.0 / 1.181 | 235.78 | 231.71 | 4.07 | 3.0 / 2.27 |
| 1024 | ON | 192 | 15.5 / 1.005 | 176.5 / 1.303 | 256.29 | 251.38 | 4.92 | 3.0 / 2.25 |

Per-GEMM duration by share of the GEMM covered by any NCCL kernel (median ms, count per step):

| T | share covered | OFF count | OFF median | ON count | ON median |
|---|---|---|---|---|---|
| 256 | 0% | 6.0 | 0.252 | 6.0 | 0.254 |
| 256 | <50% | 92.0 | 0.253 | 47.0 | 0.254 |
| 256 | 50-99% | 0.5 | 0.504 | 27.0 | 0.507 |
| 256 | 100% | 93.5 | 0.503 | 112.0 | 0.504 |
| 1024 | 0% | 6.0 | 1.002 | 15.5 | 1.005 |
| 1024 | <50% | 107.0 | 1.006 | 91.0 | 1.012 |
| 1024 | 50-99% | 78.5 | 1.357 | 47.5 | 1.488 |
| 1024 | 100% | 0.5 | 1.557 | 38.0 | 1.558 |

Top GEMM names in mb1..3 (median ms, all overlap states):

| T | kernel | OFF count | OFF median | ON count | ON median |
|---|---|---|---|---|---|
| 256 | nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_splitK_TNT | 96 | 0.503 | 96 | 0.507 |
| 256 | nvjet_sm90_tst_144x128_64x6_1x2_h_bz_TNN | 96 | 0.252 | 96 | 0.255 |
| 1024 | nvjet_sm90_tst_144x128_64x6_1x2_h_bz_TNN | 96 | 1.353 | 96 | 1.328 |
| 1024 | nvjet_sm90_tst_192x128_64x5_1x2_h_bz_coopB_TNT | 96 | 1.006 | 96 | 1.012 |

NCCL kernels that overlap the mb1..3 forward windows (mean per step, full kernel durations):

| T | class | stream | kernel | OFF count | OFF total ms | ON count | ON total ms |
|---|---|---|---|---|---|---|---|
| 256 | AG-I | 39 | ncclDevKernel_AllGather_RING_LL | 102.5 | 81.01 | 103.5 | 85.99 |
| 1024 | AG-I | 39 | ncclDevKernel_AllGather_RING_LL | 102.5 | 83.59 | 104.0 | 129.56 |

Only inner all-gathers overlap these windows. No O kernel and no reduce-scatter runs there
(reduce-scatter only happens in backward, and O-RS only in `bwd3`).

**Decision (T=1024).** Option (a), slower GEMMs with no communication, is not supported. GEMMs
with no NCCL overlap run at the same speed in both files (median 1.002 vs 1.005 ms). Option (b),
extra communication kernels, is not supported either. The window contains no extra kernels. What
changes is the same inner all-gather. In ON it runs longer in mb1..3 (mean 1.29 ms vs 0.84 ms,
and union of its kernel intervals about 128 vs 83 ms per step). It therefore covers more GEMM time.
GEMMs that are 50-99% covered have a median of 1.49 ms in ON and 1.36 ms in OFF. Fully covered GEMMs
take 1.56 ms in ON (38 per step) and almost never occur in OFF (0.5 per step). Uncovered GEMMs take 1.00 ms. Option (c) is minor: compute-stream idle is 4.9 ms per step in ON and
4.1 ms in OFF.

Why the inner all-gather is longer in ON is not determined. Rank 0 shows the same pattern (inner
all-gather mean 1.30 ms in ON vs 0.84 ms in OFF). Both ranks sit in different nodes, which argues
against one slow GPU on one node, but this does not identify the cause. Whether the
GEMMs take SMs away from the gather is not measured here. The jump from about 1.0 to 1.56 ms looks
discrete, which fits fewer SMs for a GEMM with fixed tile waves, but the trace does not show SM occupancy.

**T=256.** The forward is not slower in ON (32.9 vs 32.5 ms per microbatch). Fully covered GEMMs run
at 0.503 vs 0.504 ms. Compute-stream idle is lower in ON (13.8 vs 22.0 ms per step).

Rank 0 confirms the same picture: T=1024 forward per microbatch 78.6 vs 85.4 ms. The uncovered
bucket medians agree within 0.01 ms. The 50-99% bucket is 1.361 ms (OFF) and 1.435 ms (ON), against
1.357 and 1.488 ms on rank 8.

## Q5. JSON `exposed` values

The `exposed` field is defined in `bench/e1_exposure.py` as E = T_fwd(mb0) minus the median
T_fwd(mb1..3), with CUDA events on the compute stream. The JSON reports the maximum over ranks of
the per-rank median.

| T | flag | exposed max | rank 0 | rank 8 | median of the other 14 ranks | fwd_mb0 max | rank 0 fwd_mb0 (per step) |
|---|---|---|---|---|---|---|---|
| 256 | OFF | 1166.3 | 23.0 | 23.3 | 1166.2 | 1199.2 | 55.9 / 54.6 / 163.0 |
| 256 | ON | 1169.1 | 2.1 | 2.2 | 1168.9 | 1201.7 | 68.9 / 34.5 / 34.6 |
| 1024 | OFF | 1206.2 | 87.9 | 14.7 | 1206.1 | 1284.8 | 166.5 / 94.0 / 246.6 |
| 1024 | ON | 1142.2 | 5.1 | 1.8 | 1141.8 | 1227.6 | 90.4 / 90.4 / 95.0 |

For ranks 0 and 8 the values are real and match the trace. The E from the trace (mb0 GPU span
minus mean mb1..3 GPU span) is 24.1 ms (OFF) and 1.8 ms (ON) at T=256, against JSON 23.3 and 2.2.
At T=1024 it is 14.1 ms (OFF) and 2.6 ms (ON), against JSON 14.7 and 1.8. In this metric the
flag removes most of the exposed outer all-gather for this rank pair.

The roughly 1.2 s values belong to the other 14 ranks, and they appear in both flags at both
sizes. The JSON keeps raw data for rank 0 only, and the traces cover only ranks 0 and 8. The 1.2 s
cannot be checked here, and no cause is established. Do not use the max value as steady-state
exposed communication. Rank 0 also shows large single-step mb0 values (OFF 163 and 247 ms), which
look like startup or peer-wait effects.

## Not determined

- Why inner all-gathers are longer in ON during mb1..3 at T=1024. Rank 0 and rank 8 agree, so it
  is not one slow peer, but the cause is not identified.
- SM occupancy. The contention claim rests on timing overlap and on the discrete GEMM slowdown,
  not on counters.
- The no-overlap comparison is small: 6 to 15.5 GEMMs per step. Treat the 1.002 vs 1.005 ms
  comparison as indicative.
- The 1.2 s `exposed` values on ranks other than 0 and 8.
- O/I classification relies on `gridX` and the layout checks above, not on NCCL communicator IDs,
  which are not in the sqlite export.
