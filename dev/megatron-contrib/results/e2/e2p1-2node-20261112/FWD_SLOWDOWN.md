# E2 forward slowdown: OFF vs ON traces, microbatches 1..3 forward, T=1024

Sources: `/tmp/e2an/{off,on}_T1024_rank{8,0}.sqlite` (nsys exports). Script: `bench/e2_fwd_slowdown.py`. Rank 8 is the main pair; rank 0 is the confirmation pair (second section).

## Findings

1. **Kernel names do not change.** Every inner all-gather in forward and backward mb1..3 is `ncclDevKernel_AllGather_RING_LL` (LL protocol, gridX 24, stream 39), 32 per microbatch, in both files and both ranks.
2. **CPU work in mb1..3 forward does not change.** Runtime calls per microbatch are identical in every category (kernel launches 165, cudaStreamWaitEvent 133, cudaEventRecord 201, cudaMemsetAsync 65, no cudaMalloc, cudaFree, cudaMemcpyAsync or synchronize). The OFF/ON difference is in mb0 only (+34 cudaStreamWaitEvent and +34 cudaEventRecord, from the outer all-gather path).
3. **No extra GPU work in mb1..3 forward.** No kernels run outside the compute stream (7) and stream 39. Memset only runs on stream 7 (65 per microbatch in both files). No memcpy is launched.
4. **The CPU is far ahead of the GPU.** Launch to GPU start is about 330 to 380 ms for inner all-gathers in both files, so launch timing does not gate them.
5. **Inner all-gather start timing changes.** In OFF, 7 of 192 inner all-gathers start at the time their dependency completes (rank 0: 6 of 192). Otherwise the all-gather waits about 0.9 ms, and it starts when the running coopB GEMM is at about 89% of its duration. In ON, 97 of 192 start at their dependency time (rank 0: 100 of 192), and about half start while no GEMM runs on the compute stream. Median all-gather duration rises from 0.87 ms to 1.16 to 1.18 ms.
6. **GEMM time rises, and only for the coopB tile.** Per microbatch, the coopB GEMM (`192x128_64x5_1x2_h_bz_coopB_TNT`) takes +8.3 ms (rank 8) and +8.6 ms (rank 0) longer in ON. The TNN GEMM takes 1.6 to 2.1 ms less. Net GEMM time rises by +6.6 ms (rank 8) and +6.5 ms (rank 0). The forward slowdown in the same profiled runs is +6.8 ms per microbatch (NSYS.md, rank 0 JSON values 78.6 to 85.4 ms). The non-profiled JSON runs show 78.8 to 84.4 ms (max over ranks).
7. **coopB GEMMs slow only when an inner all-gather covers almost all of them.** When covered at least 90%, a coopB GEMM takes 1.56 ms (ON: 14 to 15 per microbatch; OFF: 0 to 0.2). When partly covered or uncovered, it takes 1.00 ms in both files. TNN GEMMs take 1.33 to 1.36 ms when partly covered in both files.
8. **Launch configuration.** The GPU has 78 SMs, 233,472 B shared memory per SM and 65,536 registers per SM. A coopB GEMM launches 78 CTAs (one per SM) with 384 threads, 221,420 B shared memory and 64,512 registers per CTA. An inner all-gather launches 24 CTAs with 544 threads, 119,040 B shared memory and 52,224 registers per CTA. One GEMM CTA and one inner all-gather CTA together exceed the SM limits on both shared memory and registers, so they cannot share an SM.

## Most likely mechanism

In ON, about half of the inner all-gathers start while the compute stream is idle. Each of those starts a median 7 µs before the next coopB GEMM is dispatched (rank 8), so the all-gather takes SMs before that GEMM is placed. Because the GEMM needs one SM per CTA and an all-gather CTA cannot share an SM with it, the GEMM runs in more waves. The coopB GEMM slows from 1.00 to 1.56 ms, and the all-gather itself slows to about 1.75 ms while the GEMM shares the GPU. In OFF, the GEMM is dispatched first, and the all-gather starts only when GEMM CTAs free SMs near the end of the GEMM. The net GEMM increase of +6.6 ms per microbatch, mostly from coopB, matches the forward slowdown (+6.8 ms per microbatch).

A simple occupancy model predicts a slowdown between 1.44x and 2.0x for a GEMM that loses the SMs held by 24 all-gather CTAs. The measured 1.56x falls in that range. This assumes each all-gather CTA holds its own SM, and the trace has no SM occupancy counters, so the model is not verified.

## Not determined

- Why the dispatch order changes in ON. The CPU calls, stream waits and kernel launch configurations are the same in both files. The only trace difference before mb1 is the mb0 outer path. The traces cannot show what changes the dispatch order.
- Whether SM occupancy is the cause of the 1.56 ms GEMM. This is the most likely explanation from the resource arithmetic and the coverage split, but no counter confirms it.
- Why an all-gather that starts first lasts about 1.75 ms. One rank cannot show whether it waits for peer ranks.
- Layer examples are not uniform. On rank 8 (layers 15 and 16) the all-gather starts at the layer start in ON and the coopB GEMM slows. On rank 0 (layers 15 and 16) the all-gather starts at the GEMM tail in ON and the TNN GEMM is slower instead. Layer 17 matches OFF on rank 8. The per-GEMM counts above are the evidence, not the layer examples.

---

---

# Rank pair 1: OFF vs ON

## OFF

- File: `off_T1024_rank8.sqlite`. Compute stream: 7. Inner all-gather: AllGather with gridX 24 on stream 39.
- Operations with no matching CUDA runtime row: 0.
- Instances (thread-matched ops in the NVTX range / ops with a launch time in the range, all threads): mb0_fwd n=2, 526/526; mb1_fwd n=2, 460/460; mb2_fwd n=2, 460/460; mb3_fwd n=2, 460/460; mb0_bwd n=2, 4/1056; mb1_bwd n=2, 4/990; mb2_bwd n=2, 4/990; mb3_bwd n=2, 4/1056.

## ON

- File: `on_T1024_rank8.sqlite`. Compute stream: 7. Inner all-gather: AllGather with gridX 24 on stream 39.
- Operations with no matching CUDA runtime row: 0.
- Instances (thread-matched ops in the NVTX range / ops with a launch time in the range, all threads): mb0_fwd n=2, 526/526; mb1_fwd n=2, 460/460; mb2_fwd n=2, 460/460; mb3_fwd n=2, 460/460; mb0_bwd n=2, 4/1056; mb1_bwd n=2, 4/990; mb2_bwd n=2, 4/990; mb3_bwd n=2, 4/1056.

### Q1. Full NCCL names of inner all-gathers (per microbatch, mean over instances)

| phase | shortName | demangled name (before template args) | gridX | stream | OFF per microbatch | ON per microbatch |
|---|---|---|---|---|---|---|
| bwd mb1..3 | ncclDevKernel_AllGather_RING_LL | ncclDevKernel_AllGather_RING_LL | 24 | 39 | 32.00 | 32.00 |
| fwd mb1..3 | ncclDevKernel_AllGather_RING_LL | ncclDevKernel_AllGather_RING_LL | 24 | 39 | 32.00 | 32.00 |

Inner all-gathers per microbatch: OFF fwd 32.00, bwd 32.00; ON fwd 32.00, bwd 32.00.
Distinct inner all-gather shortNames across fwd and bwd, both files: ['ncclDevKernel_AllGather_RING_LL'].

### Q2. CUDA runtime API calls per forward microbatch (CPU thread, inside the NVTX range)

Averages per microbatch: mb0 over 2 instances, mb1..3 over 6 instances per file.

| category | OFF mb0 | ON mb0 | OFF mb1..3 | ON mb1..3 | Δ mb1..3 (ON - OFF) |
|---|---|---|---|---|---|
| kernel launch (cudaLaunchKernel / cuLaunchKernelEx) | 198.0 | 198.0 | 165.0 | 165.0 | +0.0 |
| cudaStreamWaitEvent | 199.0 | 233.0 | 133.0 | 133.0 | +0.0 |
| cudaEventRecord (incl. WithFlags) | 333.0 | 367.0 | 201.0 | 201.0 | +0.0 |
| cudaMalloc* | 0.0 | 0.0 | 0.0 | 0.0 | +0.0 |
| cudaFree* | 0.0 | 0.0 | 0.0 | 0.0 | +0.0 |
| cudaMemcpyAsync | 0.0 | 0.0 | 0.0 | 0.0 | +0.0 |
| cudaMemsetAsync | 65.0 | 65.0 | 65.0 | 65.0 | +0.0 |
| *Synchronize* | 0.0 | 0.0 | 0.0 | 0.0 | +0.0 |
| other | 930.0 | 1066.0 | 699.0 | 699.0 | +0.0 |

Per runtime-call name (top 30 by mb1..3 count):

| runtime call | OFF mb0 | ON mb0 | OFF mb1..3 | ON mb1..3 | Δ mb1..3 |
|---|---|---|---|---|---|
| cudaStreamIsCapturing | 268.0 | 336.0 | 202.0 | 202.0 | +0.0 |
| cudaStreamWaitEvent | 199.0 | 233.0 | 133.0 | 133.0 | +0.0 |
| cuKernelGetAttribute | 128.0 | 128.0 | 128.0 | 128.0 | +0.0 |
| cudaEventCreateWithFlags | 135.0 | 169.0 | 102.0 | 102.0 | +0.0 |
| cudaEventRecordWithFlags | 135.0 | 169.0 | 102.0 | 102.0 | +0.0 |
| cudaEventDestroy | 133.0 | 167.0 | 100.0 | 100.0 | +0.0 |
| cudaEventRecord | 198.0 | 198.0 | 99.0 | 99.0 | +0.0 |
| cuLaunchKernelEx | 130.0 | 130.0 | 97.0 | 97.0 | +0.0 |
| cudaLaunchKernel | 68.0 | 68.0 | 68.0 | 68.0 | +0.0 |
| cuKernelGetName | 68.0 | 68.0 | 68.0 | 68.0 | +0.0 |
| cudaStreamGetCaptureInfo | 132.0 | 132.0 | 66.0 | 66.0 | +0.0 |
| cudaMemsetAsync | 65.0 | 65.0 | 65.0 | 65.0 | +0.0 |
| cudaGetFuncBySymbol | 66.0 | 66.0 | 33.0 | 33.0 | +0.0 |

### Q3. Inner all-gathers in mb1..3 forward: launch, queue delay, and the compute stream

| metric | OFF | ON |
|---|---|---|
| inner AG kernels analysed (both steps, mb1..3) | 192 | 192 |
| launch to GPU start (queue delay), median ms | 338.358 | 328.708 |
| launch to GPU start, p90 ms | 403.771 | 415.139 |
| AG kernel duration, median ms | 0.865 | 1.158 |
| AG starts while a GEMM runs on the compute stream | 0.964 | 0.495 |
| AG starts while the compute stream is idle | 0.005 | 0.484 |
| AG starts while another compute-stream op runs (memset etc.) | 0.031 | 0.021 |
| GEMM elapsed fraction at AG start, median | 0.890 | 0.889 |
| AG duration median, when a GEMM runs at start, ms | 0.866 | 0.884 |
| AG duration median, when compute stream idle at start, ms | 1.724 | 1.743 |
| gap from previous stream-39 op end to AG start, median ms | 1.575 | 1.230 |

GEMM elapsed fraction bins at AG start (count):

| bin | OFF | ON |
|---|---|---|
| 0-25% | 0 | 0 |
| 25-50% | 0 | 0 |
| 50-75% | 0 | 0 |
| 75-100% | 185 | 95 |

Op immediately before the AG on stream 39 (top 3, count):

- OFF: [('AG-I RING_LL', 186), ('AG grid1 RING_LL', 6)]
- ON: [('AG-I RING_LL', 186), ('AG grid1 RING_LL', 6)]

Binding dependency of each AG (the op whose completion set the stream-39 frontier before the AG). Lag = AG GPU start minus that op's end. 'Starts at once' counts lag < 0.1 ms; the rest wait.

| binding dependency | OFF count | OFF starts at once | OFF lag median ms | ON count | ON starts at once | ON lag median ms |
|---|---|---|---|---|---|---|
| other op on compute stream | 180 | 0 | 0.902 | 180 | 87 | 0.899 |
| previous AG on stream 39 | 12 | 7 | 0.020 | 12 | 10 | 0.013 |

AG duration by the GEMM that runs during it (share of the AG duration overlapped by the GEMM class):

| overlap class | OFF count | OFF median duration ms | ON count | ON median duration ms |
|---|---|---|---|---|
| coopB GEMM covers >=50% of the AG | 1 | 1.724 | 91 | 1.745 |
| no GEMM for >=50% of the AG | 6 | 0.750 | 6 | 0.751 |
| other GEMM covers >=50% of the AG | 185 | 0.866 | 95 | 0.884 |

Compute-stream op classes at AG start that are not GEMM or idle (top 3): OFF [('other elementwise_kernel', 6)]; ON [('other elementwise_kernel', 4)].

### Q3b. GEMM duration by tile and by how much inner all-gather covers it (mb1..3 forward)

Counts are per microbatch. Coverage is the share of the GEMM's duration overlapped by any inner all-gather.

| GEMM tile (name after nvjet_sm90_tst_) | coverage | OFF count | OFF median ms | ON count | ON median ms |
|---|---|---|---|---|---|
| 144x128_64x6_1x2_h_bz_TNN | none | 1.0 | 1.001 | 4.2 | 1.005 |
| 144x128_64x6_1x2_h_bz_TNN | partial (<90%) | 31.0 | 1.354 | 27.8 | 1.332 |
| 192x128_64x5_1x2_h_bz_coopB_TNT | full (>=90%) | 0.2 | 1.557 | 14.3 | 1.559 |
| 192x128_64x5_1x2_h_bz_coopB_TNT | none | 1.0 | 1.005 | 1.0 | 1.008 |
| 192x128_64x5_1x2_h_bz_coopB_TNT | partial (<90%) | 30.8 | 1.006 | 16.7 | 1.007 |

GEMM time per microbatch by tile (ms, sum of kernel durations):

| GEMM tile | OFF | ON | ON - OFF |
|---|---|---|---|
| 144x128_64x6_1x2_h_bz_TNN | 42.93 | 41.29 | -1.63 |
| 192x128_64x5_1x2_h_bz_coopB_TNT | 32.28 | 40.55 | +8.27 |
| all GEMMs | 75.21 | 81.84 | +6.63 |

### Q6. Launch configuration and SM limits (context for the overlap mechanism)

GPU (TARGET_INFO_GPU): 78 SMs, 233472 B shared memory per SM, 65536 registers per SM.

| file | kernel | CTAs (gridX×Y×Z) | threads per CTA | registers per CTA | shared memory per CTA (B) |
|---|---|---|---|---|---|
| OFF | coopB GEMM | 2×39×1 = 78 | 384 | 64512 | 221420 |
| OFF | TNN GEMM | 2×39×1 = 78 | 384 | 64512 | 225524 |
| OFF | inner AG | 24×1×1 = 24 | 544 | 52224 | 119040 |
| ON | coopB GEMM | 2×39×1 = 78 | 384 | 64512 | 221420 |
| ON | TNN GEMM | 2×39×1 = 78 | 384 | 64512 | 225524 |
| ON | inner AG | 24×1×1 = 24 | 544 | 52224 | 119040 |

One coopB GEMM CTA plus one inner AG CTA need 340460 B shared memory (SM limit 233472; exceeds: True) and 116736 registers (SM limit 65536; exceeds: True). The coopB GEMM launches 78 CTAs, which is at least as many as the 78 SMs.

### Q4. Activity outside the compute stream and stream 39 in mb1..3 forward (per microbatch)

Kernels on other streams:

| stream | kernel | OFF count | OFF total ms | ON count | ON total ms |
|---|---|---|---|---|---|
| none | none | 0 | 0 | 0 | 0 |

memcpy and memset launched in mb1..3 forward (all streams):

| table | stream | op | OFF count | OFF bytes (MB) | ON count | ON bytes (MB) |
|---|---|---|---|---|---|---|
| MEMSET | 7 | memset | 65.00 | 0.00 | 65.00 | 0.00 |

### Q5. Per-layer timeline, mb1 forward, first instance (layers 15, 16, 17)

Offsets are GPU start minus the first op of the layer, in ms. Rows are aligned by position.

#### Layer 15

| # | OFF stream | OFF op | OFF offset | OFF dur | ON stream | ON op | ON offset | ON dur |
|---|---|---|---|---|---|---|---|---|
| 0 | 7 | memset | 0.000 | 0.001 | 39 | AG-I RING_LL | 0.000 | 1.897 |
| 1 | 7 | GEMM 192x128_64x5_1x2_h_bz_coopB_TNT | 0.003 | 1.005 | 7 | memset | 0.002 | 0.001 |
| 2 | 39 | AG-I RING_LL | 0.896 | 0.843 | 7 | GEMM 192x128_64x5_1x2_h_bz_coopB_TNT | 0.005 | 1.565 |
| 3 | 7 | vectorized_elementwise_kernel | 1.010 | 0.055 | 7 | vectorized_elementwise_kernel | 1.573 | 0.057 |
| 4 | 7 | memset | 1.068 | 0.002 | 7 | memset | 1.634 | 0.002 |
| 5 | 7 | GEMM 144x128_64x6_1x2_h_bz_TNN | 1.073 | 1.333 | 7 | GEMM 144x128_64x6_1x2_h_bz_TNN | 1.639 | 1.256 |
| 6 | 7 | vectorized_elementwise_kernel | 2.408 | 0.007 | 7 | vectorized_elementwise_kernel | 2.898 | 0.007 |

CPU-side calls in the layer range: OFF {'other': 21, 'cudaStreamWaitEvent': 4, 'cudaEventRecord (incl. WithFlags)': 6, 'kernel launch (cudaLaunchKernel / cuLaunchKernelEx)': 5, 'cudaMemsetAsync': 2}; ON {'other': 21, 'cudaStreamWaitEvent': 4, 'cudaEventRecord (incl. WithFlags)': 6, 'kernel launch (cudaLaunchKernel / cuLaunchKernelEx)': 5, 'cudaMemsetAsync': 2}.
cudaStreamWaitEvent (syncType 2) by waiting stream: OFF {7: 1, 39: 2, 153: 1}; ON {7: 1, 39: 2, 153: 1}.

#### Layer 16

| # | OFF stream | OFF op | OFF offset | OFF dur | ON stream | ON op | ON offset | ON dur |
|---|---|---|---|---|---|---|---|---|
| 0 | 7 | memset | 0.000 | 0.001 | 39 | AG-I RING_LL | 0.000 | 1.608 |
| 1 | 7 | GEMM 192x128_64x5_1x2_h_bz_coopB_TNT | 0.003 | 1.006 | 7 | memset | 0.003 | 0.001 |
| 2 | 39 | AG-I RING_LL | 0.901 | 0.867 | 7 | GEMM 192x128_64x5_1x2_h_bz_coopB_TNT | 0.006 | 1.562 |
| 3 | 7 | vectorized_elementwise_kernel | 1.011 | 0.055 | 7 | vectorized_elementwise_kernel | 1.573 | 0.040 |
| 4 | 7 | memset | 1.069 | 0.002 | 7 | memset | 1.614 | 0.001 |
| 5 | 7 | GEMM 144x128_64x6_1x2_h_bz_TNN | 1.073 | 1.360 | 7 | GEMM 144x128_64x6_1x2_h_bz_TNN | 1.617 | 1.006 |
| 6 | 7 | vectorized_elementwise_kernel | 2.435 | 0.006 | 7 | vectorized_elementwise_kernel | 2.627 | 0.006 |

CPU-side calls in the layer range: OFF {'other': 21, 'cudaStreamWaitEvent': 4, 'cudaEventRecord (incl. WithFlags)': 6, 'kernel launch (cudaLaunchKernel / cuLaunchKernelEx)': 5, 'cudaMemsetAsync': 2}; ON {'other': 21, 'cudaStreamWaitEvent': 4, 'cudaEventRecord (incl. WithFlags)': 6, 'kernel launch (cudaLaunchKernel / cuLaunchKernelEx)': 5, 'cudaMemsetAsync': 2}.
cudaStreamWaitEvent (syncType 2) by waiting stream: OFF {7: 1, 39: 2, 153: 1}; ON {7: 1, 39: 2, 153: 1}.

#### Layer 17

| # | OFF stream | OFF op | OFF offset | OFF dur | ON stream | ON op | ON offset | ON dur |
|---|---|---|---|---|---|---|---|---|
| 0 | 7 | memset | 0.000 | 0.001 | 7 | memset | 0.000 | 0.001 |
| 1 | 7 | GEMM 192x128_64x5_1x2_h_bz_coopB_TNT | 0.003 | 1.004 | 7 | GEMM 192x128_64x5_1x2_h_bz_coopB_TNT | 0.003 | 1.012 |
| 2 | 39 | AG-I RING_LL | 0.901 | 0.881 | 39 | AG-I RING_LL | 0.901 | 0.912 |
| 3 | 7 | vectorized_elementwise_kernel | 1.009 | 0.055 | 7 | vectorized_elementwise_kernel | 1.018 | 0.058 |
| 4 | 7 | memset | 1.067 | 0.002 | 7 | memset | 1.080 | 0.002 |
| 5 | 7 | GEMM 144x128_64x6_1x2_h_bz_TNN | 1.073 | 1.373 | 7 | GEMM 144x128_64x6_1x2_h_bz_TNN | 1.084 | 1.395 |
| 6 | 7 | vectorized_elementwise_kernel | 2.447 | 0.007 | 7 | vectorized_elementwise_kernel | 2.482 | 0.007 |

CPU-side calls in the layer range: OFF {'other': 21, 'cudaStreamWaitEvent': 4, 'cudaEventRecord (incl. WithFlags)': 6, 'kernel launch (cudaLaunchKernel / cuLaunchKernelEx)': 5, 'cudaMemsetAsync': 2}; ON {'other': 21, 'cudaStreamWaitEvent': 4, 'cudaEventRecord (incl. WithFlags)': 6, 'kernel launch (cudaLaunchKernel / cuLaunchKernelEx)': 5, 'cudaMemsetAsync': 2}.
cudaStreamWaitEvent (syncType 2) by waiting stream: OFF {7: 1, 39: 2, 153: 1}; ON {7: 1, 39: 2, 153: 1}.


# Rank pair 2: OFF vs ON (confirmation)

## OFF

- File: `off_T1024_rank0.sqlite`. Compute stream: 7. Inner all-gather: AllGather with gridX 24 on stream 39.
- Operations with no matching CUDA runtime row: 0.
- Instances (thread-matched ops in the NVTX range / ops with a launch time in the range, all threads): mb0_fwd n=2, 526/526; mb1_fwd n=2, 460/460; mb2_fwd n=2, 460/460; mb3_fwd n=2, 460/460; mb0_bwd n=2, 4/1056; mb1_bwd n=2, 4/990; mb2_bwd n=2, 4/990; mb3_bwd n=2, 4/1056.

## ON

- File: `on_T1024_rank0.sqlite`. Compute stream: 7. Inner all-gather: AllGather with gridX 24 on stream 39.
- Operations with no matching CUDA runtime row: 0.
- Instances (thread-matched ops in the NVTX range / ops with a launch time in the range, all threads): mb0_fwd n=2, 526/526; mb1_fwd n=2, 460/460; mb2_fwd n=2, 460/460; mb3_fwd n=2, 460/460; mb0_bwd n=2, 4/1056; mb1_bwd n=2, 4/990; mb2_bwd n=2, 4/990; mb3_bwd n=2, 4/1056.

### Q1. Full NCCL names of inner all-gathers (per microbatch, mean over instances)

| phase | shortName | demangled name (before template args) | gridX | stream | OFF per microbatch | ON per microbatch |
|---|---|---|---|---|---|---|
| bwd mb1..3 | ncclDevKernel_AllGather_RING_LL | ncclDevKernel_AllGather_RING_LL | 24 | 39 | 32.00 | 32.00 |
| fwd mb1..3 | ncclDevKernel_AllGather_RING_LL | ncclDevKernel_AllGather_RING_LL | 24 | 39 | 32.00 | 32.00 |

Inner all-gathers per microbatch: OFF fwd 32.00, bwd 32.00; ON fwd 32.00, bwd 32.00.
Distinct inner all-gather shortNames across fwd and bwd, both files: ['ncclDevKernel_AllGather_RING_LL'].

### Q2. CUDA runtime API calls per forward microbatch (CPU thread, inside the NVTX range)

Averages per microbatch: mb0 over 2 instances, mb1..3 over 6 instances per file.

| category | OFF mb0 | ON mb0 | OFF mb1..3 | ON mb1..3 | Δ mb1..3 (ON - OFF) |
|---|---|---|---|---|---|
| kernel launch (cudaLaunchKernel / cuLaunchKernelEx) | 198.0 | 198.0 | 165.0 | 165.0 | +0.0 |
| cudaStreamWaitEvent | 199.0 | 233.0 | 133.0 | 133.0 | +0.0 |
| cudaEventRecord (incl. WithFlags) | 333.0 | 367.0 | 201.0 | 201.0 | +0.0 |
| cudaMalloc* | 0.0 | 0.0 | 0.0 | 0.0 | +0.0 |
| cudaFree* | 0.0 | 0.0 | 0.0 | 0.0 | +0.0 |
| cudaMemcpyAsync | 0.0 | 0.0 | 0.0 | 0.0 | +0.0 |
| cudaMemsetAsync | 65.0 | 65.0 | 65.0 | 65.0 | +0.0 |
| *Synchronize* | 0.0 | 0.0 | 0.0 | 0.0 | +0.0 |
| other | 930.0 | 1066.0 | 699.0 | 699.0 | +0.0 |

Per runtime-call name (top 30 by mb1..3 count):

| runtime call | OFF mb0 | ON mb0 | OFF mb1..3 | ON mb1..3 | Δ mb1..3 |
|---|---|---|---|---|---|
| cudaStreamIsCapturing | 268.0 | 336.0 | 202.0 | 202.0 | +0.0 |
| cudaStreamWaitEvent | 199.0 | 233.0 | 133.0 | 133.0 | +0.0 |
| cuKernelGetAttribute | 128.0 | 128.0 | 128.0 | 128.0 | +0.0 |
| cudaEventCreateWithFlags | 135.0 | 169.0 | 102.0 | 102.0 | +0.0 |
| cudaEventRecordWithFlags | 135.0 | 169.0 | 102.0 | 102.0 | +0.0 |
| cudaEventDestroy | 133.0 | 167.0 | 100.0 | 100.0 | +0.0 |
| cudaEventRecord | 198.0 | 198.0 | 99.0 | 99.0 | +0.0 |
| cuLaunchKernelEx | 130.0 | 130.0 | 97.0 | 97.0 | +0.0 |
| cudaLaunchKernel | 68.0 | 68.0 | 68.0 | 68.0 | +0.0 |
| cuKernelGetName | 68.0 | 68.0 | 68.0 | 68.0 | +0.0 |
| cudaStreamGetCaptureInfo | 132.0 | 132.0 | 66.0 | 66.0 | +0.0 |
| cudaMemsetAsync | 65.0 | 65.0 | 65.0 | 65.0 | +0.0 |
| cudaGetFuncBySymbol | 66.0 | 66.0 | 33.0 | 33.0 | +0.0 |

### Q3. Inner all-gathers in mb1..3 forward: launch, queue delay, and the compute stream

| metric | OFF | ON |
|---|---|---|
| inner AG kernels analysed (both steps, mb1..3) | 192 | 192 |
| launch to GPU start (queue delay), median ms | 377.331 | 338.707 |
| launch to GPU start, p90 ms | 404.163 | 416.187 |
| AG kernel duration, median ms | 0.867 | 1.183 |
| AG starts while a GEMM runs on the compute stream | 0.969 | 0.479 |
| AG starts while the compute stream is idle | 0.005 | 0.510 |
| AG starts while another compute-stream op runs (memset etc.) | 0.026 | 0.010 |
| GEMM elapsed fraction at AG start, median | 0.889 | 0.889 |
| AG duration median, when a GEMM runs at start, ms | 0.868 | 0.905 |
| AG duration median, when compute stream idle at start, ms | 0.728 | 1.750 |
| gap from previous stream-39 op end to AG start, median ms | 1.572 | 1.093 |

GEMM elapsed fraction bins at AG start (count):

| bin | OFF | ON |
|---|---|---|
| 0-25% | 0 | 0 |
| 25-50% | 0 | 0 |
| 50-75% | 0 | 0 |
| 75-100% | 186 | 92 |

Op immediately before the AG on stream 39 (top 3, count):

- OFF: [('AG-I RING_LL', 186), ('AG grid1 RING_LL', 6)]
- ON: [('AG-I RING_LL', 186), ('AG grid1 RING_LL', 6)]

Binding dependency of each AG (the op whose completion set the stream-39 frontier before the AG). Lag = AG GPU start minus that op's end. 'Starts at once' counts lag < 0.1 ms; the rest wait.

| binding dependency | OFF count | OFF starts at once | OFF lag median ms | ON count | ON starts at once | ON lag median ms |
|---|---|---|---|---|---|---|
| other op on compute stream | 180 | 0 | 0.900 | 180 | 91 | 0.012 |
| previous AG on stream 39 | 12 | 6 | 0.458 | 12 | 9 | 0.004 |

AG duration by the GEMM that runs during it (share of the AG duration overlapped by the GEMM class):

| overlap class | OFF count | OFF median duration ms | ON count | ON median duration ms |
|---|---|---|---|---|
| coopB GEMM covers >=50% of the AG | 0 | n/a | 94 | 1.755 |
| no GEMM for >=50% of the AG | 6 | 0.738 | 6 | 0.752 |
| other GEMM covers >=50% of the AG | 186 | 0.868 | 92 | 0.905 |

Compute-stream op classes at AG start that are not GEMM or idle (top 3): OFF [('other elementwise_kernel', 5)]; ON [('other elementwise_kernel', 2)].

### Q3b. GEMM duration by tile and by how much inner all-gather covers it (mb1..3 forward)

Counts are per microbatch. Coverage is the share of the GEMM's duration overlapped by any inner all-gather.

| GEMM tile (name after nvjet_sm90_tst_) | coverage | OFF count | OFF median ms | ON count | ON median ms |
|---|---|---|---|---|---|
| 144x128_64x6_1x2_h_bz_TNN | none | 1.0 | 1.001 | 3.8 | 1.000 |
| 144x128_64x6_1x2_h_bz_TNN | partial (<90%) | 31.0 | 1.358 | 28.2 | 1.333 |
| 192x128_64x5_1x2_h_bz_coopB_TNT | full (>=90%) | 0.0 | n/a | 15.0 | 1.559 |
| 192x128_64x5_1x2_h_bz_coopB_TNT | none | 1.0 | 1.003 | 1.0 | 1.004 |
| 192x128_64x5_1x2_h_bz_coopB_TNT | partial (<90%) | 31.0 | 1.004 | 16.0 | 1.004 |

GEMM time per microbatch by tile (ms, sum of kernel durations):

| GEMM tile | OFF | ON | ON - OFF |
|---|---|---|---|
| 144x128_64x6_1x2_h_bz_TNN | 43.13 | 41.01 | -2.11 |
| 192x128_64x5_1x2_h_bz_coopB_TNT | 32.13 | 40.77 | +8.64 |
| all GEMMs | 75.26 | 81.79 | +6.53 |

### Q6. Launch configuration and SM limits (context for the overlap mechanism)

GPU (TARGET_INFO_GPU): 78 SMs, 233472 B shared memory per SM, 65536 registers per SM.

| file | kernel | CTAs (gridX×Y×Z) | threads per CTA | registers per CTA | shared memory per CTA (B) |
|---|---|---|---|---|---|
| OFF | coopB GEMM | 2×39×1 = 78 | 384 | 64512 | 221420 |
| OFF | TNN GEMM | 2×39×1 = 78 | 384 | 64512 | 225524 |
| OFF | inner AG | 24×1×1 = 24 | 544 | 52224 | 119040 |
| ON | coopB GEMM | 2×39×1 = 78 | 384 | 64512 | 221420 |
| ON | TNN GEMM | 2×39×1 = 78 | 384 | 64512 | 225524 |
| ON | inner AG | 24×1×1 = 24 | 544 | 52224 | 119040 |

One coopB GEMM CTA plus one inner AG CTA need 340460 B shared memory (SM limit 233472; exceeds: True) and 116736 registers (SM limit 65536; exceeds: True). The coopB GEMM launches 78 CTAs, which is at least as many as the 78 SMs.

### Q4. Activity outside the compute stream and stream 39 in mb1..3 forward (per microbatch)

Kernels on other streams:

| stream | kernel | OFF count | OFF total ms | ON count | ON total ms |
|---|---|---|---|---|---|
| none | none | 0 | 0 | 0 | 0 |

memcpy and memset launched in mb1..3 forward (all streams):

| table | stream | op | OFF count | OFF bytes (MB) | ON count | ON bytes (MB) |
|---|---|---|---|---|---|---|
| MEMSET | 7 | memset | 65.00 | 0.00 | 65.00 | 0.00 |

### Q5. Per-layer timeline, mb1 forward, first instance (layers 15, 16, 17)

Offsets are GPU start minus the first op of the layer, in ms. Rows are aligned by position.

#### Layer 15

| # | OFF stream | OFF op | OFF offset | OFF dur | ON stream | ON op | ON offset | ON dur |
|---|---|---|---|---|---|---|---|---|
| 0 | 7 | memset | 0.000 | 0.001 | 7 | memset | 0.000 | 0.001 |
| 1 | 7 | GEMM 192x128_64x5_1x2_h_bz_coopB_TNT | 0.003 | 1.008 | 7 | GEMM 192x128_64x5_1x2_h_bz_coopB_TNT | 0.003 | 1.003 |
| 2 | 39 | AG-I RING_LL | 0.899 | 0.897 | 39 | AG-I RING_LL | 0.893 | 0.970 |
| 3 | 7 | vectorized_elementwise_kernel | 1.013 | 0.055 | 7 | vectorized_elementwise_kernel | 1.009 | 0.056 |
| 4 | 7 | memset | 1.070 | 0.002 | 7 | memset | 1.068 | 0.002 |
| 5 | 7 | GEMM 144x128_64x6_1x2_h_bz_TNN | 1.075 | 1.387 | 7 | GEMM 144x128_64x6_1x2_h_bz_TNN | 1.073 | 1.454 |
| 6 | 7 | vectorized_elementwise_kernel | 2.463 | 0.007 | 7 | vectorized_elementwise_kernel | 2.530 | 0.007 |

CPU-side calls in the layer range: OFF {'other': 21, 'cudaStreamWaitEvent': 4, 'cudaEventRecord (incl. WithFlags)': 6, 'kernel launch (cudaLaunchKernel / cuLaunchKernelEx)': 5, 'cudaMemsetAsync': 2}; ON {'other': 21, 'cudaStreamWaitEvent': 4, 'cudaEventRecord (incl. WithFlags)': 6, 'kernel launch (cudaLaunchKernel / cuLaunchKernelEx)': 5, 'cudaMemsetAsync': 2}.
cudaStreamWaitEvent (syncType 2) by waiting stream: OFF {7: 1, 39: 2, 153: 1}; ON {39: 2, 153: 1, 7: 1}.

#### Layer 16

| # | OFF stream | OFF op | OFF offset | OFF dur | ON stream | ON op | ON offset | ON dur |
|---|---|---|---|---|---|---|---|---|
| 0 | 7 | memset | 0.000 | 0.002 | 7 | memset | 0.000 | 0.001 |
| 1 | 7 | GEMM 192x128_64x5_1x2_h_bz_coopB_TNT | 0.003 | 1.006 | 7 | GEMM 192x128_64x5_1x2_h_bz_coopB_TNT | 0.002 | 1.002 |
| 2 | 39 | AG-I RING_LL | 0.897 | 0.882 | 39 | AG-I RING_LL | 0.899 | 1.155 |
| 3 | 7 | vectorized_elementwise_kernel | 1.011 | 0.057 | 7 | vectorized_elementwise_kernel | 1.009 | 0.056 |
| 4 | 7 | memset | 1.070 | 0.002 | 7 | memset | 1.067 | 0.002 |
| 5 | 7 | GEMM 144x128_64x6_1x2_h_bz_TNN | 1.076 | 1.370 | 7 | GEMM 144x128_64x6_1x2_h_bz_TNN | 1.072 | 1.645 |
| 6 | 7 | vectorized_elementwise_kernel | 2.447 | 0.006 | 7 | vectorized_elementwise_kernel | 2.720 | 0.007 |

CPU-side calls in the layer range: OFF {'other': 21, 'cudaStreamWaitEvent': 4, 'cudaEventRecord (incl. WithFlags)': 6, 'kernel launch (cudaLaunchKernel / cuLaunchKernelEx)': 5, 'cudaMemsetAsync': 2}; ON {'other': 21, 'cudaStreamWaitEvent': 4, 'cudaEventRecord (incl. WithFlags)': 6, 'kernel launch (cudaLaunchKernel / cuLaunchKernelEx)': 5, 'cudaMemsetAsync': 2}.
cudaStreamWaitEvent (syncType 2) by waiting stream: OFF {7: 1, 39: 2, 153: 1}; ON {7: 1, 39: 2, 153: 1}.

#### Layer 17

| # | OFF stream | OFF op | OFF offset | OFF dur | ON stream | ON op | ON offset | ON dur |
|---|---|---|---|---|---|---|---|---|
| 0 | 7 | memset | 0.000 | 0.001 | 7 | memset | 0.000 | 0.001 |
| 1 | 7 | GEMM 192x128_64x5_1x2_h_bz_coopB_TNT | 0.003 | 1.005 | 7 | GEMM 192x128_64x5_1x2_h_bz_coopB_TNT | 0.003 | 1.003 |
| 2 | 39 | AG-I RING_LL | 0.897 | 0.863 | 39 | AG-I RING_LL | 0.897 | 1.106 |
| 3 | 7 | vectorized_elementwise_kernel | 1.009 | 0.057 | 7 | vectorized_elementwise_kernel | 1.009 | 0.055 |
| 4 | 7 | memset | 1.068 | 0.002 | 7 | memset | 1.068 | 0.002 |
| 5 | 7 | GEMM 144x128_64x6_1x2_h_bz_TNN | 1.074 | 1.352 | 7 | GEMM 144x128_64x6_1x2_h_bz_TNN | 1.072 | 1.594 |
| 6 | 7 | vectorized_elementwise_kernel | 2.428 | 0.007 | 7 | vectorized_elementwise_kernel | 2.670 | 0.007 |

CPU-side calls in the layer range: OFF {'other': 21, 'cudaStreamWaitEvent': 4, 'cudaEventRecord (incl. WithFlags)': 6, 'kernel launch (cudaLaunchKernel / cuLaunchKernelEx)': 5, 'cudaMemsetAsync': 2}; ON {'other': 21, 'cudaStreamWaitEvent': 4, 'cudaEventRecord (incl. WithFlags)': 6, 'kernel launch (cudaLaunchKernel / cuLaunchKernelEx)': 5, 'cudaMemsetAsync': 2}.
cudaStreamWaitEvent (syncType 2) by waiting stream: OFF {7: 1, 39: 2, 153: 1}; ON {7: 1, 39: 2, 153: 1}.

