## Gather side update: results for P1 (DP-outer AG on its own stream) and an unexpected finding

### Setup

- Hardware: 2 nodes × 8 H20 (78 SMs), cross-node RDMA
- Mesh: dp_outer=2, dp_inner=8
- Software: torch 2.14.1+cu130, NCCL 2.30.7
- Model: 32 blocks, d=4096, 256 MB of bf16 parameters per block, GEMMs and elementwise ops only
- Metrics:
  - `fwd_mb0` is the forward time of the first microbatch
  - `fwd_rest` is the median forward time of mb1..3
  - Every number is the max over the 16 ranks, in ms
- Each set of arms runs in forward then reversed order (ABBA); repeat noise is below 0.4%

### 1. What P1 does

P1 moves the DP-outer all-gather (O) to its own stream, so in the first microbatch it can overlap the DP-inner all-gather (I).

- Losses are bit-exact in every configuration; memory is unchanged.
- Small T gains clearly:

  | T | mb0 exposed time | step, GA=1 | step, GA=4 |
  |---|---|---|---|
  | 256 | 22.8 → 2.4 | 21–25% faster | 4.7% faster |

- Large T with GA>1 gets slower:

  | T | fwd_rest | step, GA=4 |
  |---|---|---|
  | 1024 | 78.7 → 85.5 (+8.7%) | 2.1% slower |
  | 4096 | 269 → 297 (+10.4%) | 3.3% slower |

  There is no O in mb1..3, yet their forward gets slower.

### 2. Causes ruled out (T=4096, fwd_rest)

| Hypothesis | Experiment | fwd_rest | Result |
|---|---|---|---|
| Reduce-side split (#6973) | #6973 on / off | no difference | ruled out |
| Just creating an extra stream | main plus one unused stream | 269 | ruled out |
| Hardware queue aliasing | `CUDA_DEVICE_MAX_CONNECTIONS=32` | p1 292 | ruled out |
| Cross-stream event fork/join | events only, NCCL stays on its stream | 269 | ruled out |
| O running concurrently with I | O on its own stream but fully serialized with I (0 overlap in nsys) | 296 | ruled out |
| NCCL launch order / graph mixing | `NCCL_LAUNCH_ORDER_IMPLICIT=1`, `NCCL_GRAPH_MIXING_SUPPORT=0` | still +10.4% | ruled out |
| Ranks already out of step at the end of mb0 | `synchronize` + `barrier` before mb1 | 297 | ruled out |

The only consistent trigger so far: an NCCL kernel has run on a stream other than the inner AG stream.

### 3. Mechanism (nsys on all 8 ranks of a node)

The extra ~28 ms is entirely GEMMs running longer. Idle time and other kernels differ by less than 1 ms.

- When inner all-gather k (AG-I k) finishes, two kernels become ready at once: the layer-k GEMM on the compute stream, and AG-I k+1 on the AG stream.
- **Flag off**: on all 8 ranks the GEMM gets the SMs first, and AG-I starts only as the GEMM finishes. Ranks stay aligned; AG-I median is 0.79 ms and the GEMM 4.06 ms.
- **P1**: on some ranks AG-I gets the SMs first, then spins inside the kernel waiting for the other ranks.
  - Within a node, the median spread of start times of the same AG-I across ranks is 5.3 ms (0.4 ms with the flag off), while end times differ by only 0.16 ms.
  - On the ranks that start early, the GEMM runs on the remaining SMs and takes 5.2–5.3 ms. About 22 of the 64 GEMMs per microbatch are affected.
  - The last rank to arrive runs AG-I in 0.79 ms, so the NCCL kernel itself is not slow.
- The race repeats at every layer, so the misalignment persists. This also explains why a single barrier before mb1 does not help.

Still unexplained: why the GEMM always wins with the flag off. The CPU-side API call sequence is identical in both configurations.

### 4. Comparison with v1 (including #5985)

Same model, same mesh, same data:

| | v2 off | v2 P1 | v1 | v1 + `hfsdp_param_gather_overlap` |
|---|---|---|---|---|
| fwd_rest, T=4096 | **268.8** | 295.8 | 295.8 | 295.8 |
| fwd_rest, T=1024 | **78.7** | 85.5 | 85.3 | 85.5 |
| step, T=4096 | 3248 | 3352 | 3531 | 3499 |

- **v1 is in the "slow" state in every configuration.**
  - In v1 the inner AG uses `async_ops=True` and is issued on ProcessGroupNCCL's internal stream, while the outer AG is synchronous and runs on the current stream. So the two always run on different streams.
  - I also tested v1 with its outer stream set to `ag_stream`; the result is the same, because the inner AG does not actually run on `ag_stream`.
- v1's nsys signature matches P1:
  - 22 slow GEMMs per microbatch
  - AG-I median 3.7 ms
  - About half of the AG-I kernels start while the compute stream is idle

  For v1 only local rank 0 of each node was profiled, so the cross-rank misalignment was not measured directly.
- So both arms of the ~5% gain reported in #5985 already include this cost, and v1 never had a "fast" baseline that would have exposed it.
- P1's step is still faster than v1, mostly from the backward pass.

### 5. Next steps

1. Add a diagnostic switch that makes AG-I k+1 explicitly wait until the layer-k GEMM has started, pinning the order seen with the flag off instead of relying on how the SMs happen to be scheduled.
   - Test it on both v2 P1 and v1.
   - If fwd_rest returns to about 269, make it part of the P1 PR; v1 may also recover about 10% of fwd_rest.
2. The P1 PR is on hold until then.

### Questions

- @wujingyue @rapatel could you check whether the fwd_rest gap between flag off and P1 reproduces on H100/B200? H20 has only 78 SMs and this GEMM uses exactly 78 CTAs, so one wave fills every SM and the SM race may be more pronounced than on other GPUs. I can share the benchmark scripts.
- Does anyone know how the hardware assigns SMs when kernels on two streams become ready at the same time, and why having run an NCCL kernel on another stream would change the outcome?
