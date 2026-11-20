# Issue #6714: MFSDP v2: overlap DP-inner and DP-outer communication

- URL: https://github.com/NVIDIA/Megatron-LM/issues/6714
- Author: @wujingyue  State: open
- Created: 2026-08-20T19:22:47Z

## Body

Follow-up to #5616: explore overlapping DP-inner and DP-outer communication in MFSDP v2.

The two communications use different network domains, and early experience suggests that overlapping them could reduce latency significantly. The implementation should follow the runtime scheduling model in `megatron/core/distributed/fsdp/src/docs/runtime_schedule.md`.

We should first measure the benefit and use those results to guide the implementation and scheduling design.

## Conversation

### @Achyuthan-S 2026-08-21T06:10:36Z

@wujingyue Thanks — I'll take this. Agreed on measure-first: I'll quantify the DP-outer reduction's current cost on the critical path and the latency a separate DP-outer stream recovers, on an HSDP/HFSDP config following runtime_schedule.md, and post numbers here before proposing the implementation. I'll prototype the separate stream just enough to get a real before/after, then let the results shape the production design.

### @Achyuthan-S 2026-08-24T06:28:02Z

Hey @wujingyue ,
Measured the DP-inner/DP-outer collective-overlap benefit (measure-first). Setup: 2×2 DP mesh, distinct NCCL communicators per axis (mesh.get_group), DP-inner reduce-scatter vs DP-outer all-reduce on separate CUDA streams, CUDA-event timed.

 Single-node, 4×A100-80GB / 4×H100-80GB:

| size    | serial         | overlapped     | savings   |
|---------|----------------|----------------|-----------|
| 64 MiB  | 1.65 / 1.07 ms | 1.05 / 0.73 ms | 37% / 31% |
| 256 MiB | 6.14 / 3.83 ms | 3.92 / 2.56 ms | 36% / 33% |
| 1 GiB   | 23.3 / 14.4 ms | 14.8 / 9.8 ms  | 37% / 32% |

Overlapped ≈ max(RS, AR) — the reduce-scatter hides fully behind the all-reduce, recovering ~⅓ of combined collective time, consistent across sizes and both GPUs. In the actual HSDP path the DP-outer all-reduce is fully serialized behind DP-inner today (0/N overlap in the profiler, both on the reduce-scatter stream), so that third is entirely on the critical path.

Caveats: single-node (both axes on NVLink) — they overlap cleanly here with little contention; a multi-node run (DP-outer cross-node on IB) is the natural follow-up for the cross-domain number. And the microbench runs both collectives every iteration, whereas in training DP-outer is last-microbatch only, so the realized per-step win needs the actual separate-stream implementation to pin down.

Given the collectives clearly overlap on distinct communicators+streams and are serialized today, I'd like to proceed to the separate DP-outer stream implementation. Does that sequencing work, or would you prefer a multi-node measurement first?

### @wujingyue 2026-08-24T06:57:27Z

> I'd like to proceed to the separate DP-outer stream implementation.

Let's do that. Then, I can measure the impact on some larger, more realistic settings. 

### @Achyuthan-S 2026-08-24T07:59:49Z

 @wujingyue Starting the DP-outer overlap (#6714). To run the last-microbatch finalize on a separate stream without tripping the #6441 allocation-stream asserts, I'm moving main_grad ownership to the dp_outer reduce stream (alloc in __init__ + on the HFSDP reset there, with an RS→dp_outer wait after the DP-inner reduce). 

Is that the ownership model you intended, or would you rather the DP-outer collective target a separate finalize buffer and leave main_grad on the reduce-scatter stream?

### @wujingyue 2026-08-24T18:02:44Z

Can you prototype multiple approaches? When we overlap outer and inner, it seems inevitable to keep both the outer-reduced buffer and all-reduced buffer alive. Therefore, we could allocate one DBuffer on outer-rs stream and another on the inner-rs stream. 

### @Achyuthan-S 2026-08-26T12:56:37Z

@wujingyue Got it — two buffers it is: keep main_grad (accumulation) on the inner stream and add a separate DP-outer-reduced buffer on the outer stream, finalizing via redistribute(out=...) so both stay alive and alloc/free stay same-stream. Nice side effect: main_grad never gets reassigned, so the reset path stays trivial. I'll prototype two lifetime variants for the finalized buffer — persistent vs allocated-per-last-microbatch-and-freed-after-step — and compare parity, overlap (event counts), and peak memory. Will share numbers.

### @wujingyue 2026-08-26T15:13:49Z

> Will share numbers

Caveat:

I don't know if you have the right hardware to measure this. This overlapping is mostly useful when dp-outer and dp-inner map to different communication channels, e.g., RDMA vs nvlink. 

When you have a prototype and have verified the overlapping is indeed achieved, @rapatel and I can measure the impact on a larger, more realistic setting. 

### @rapatel 2026-08-26T15:23:52Z

Isn't this what #5985 does - which overlaps the outer-DP AGs with the inner-DP AGs.

### @Achyuthan-S 2026-08-26T15:37:37Z

> Isn't this what [#5985](https://github.com/NVIDIA/Megatron-LM/pull/5985) does - which overlaps the outer-DP AGs with the inner-DP AGs.

@rapatel Thanks for the pointer — I went through #5985. I think they're complementary rather than the same piece:
 
- #5985 pipelines HFSDP parameter all-gathers across the DP-outer/DP-inner domains — prefetching the DP-outer gather a unit ahead of the DP-inner frontier (hfsdp_param_gather_overlap) — on the mfsdp_v1 path (param_and_grad_buffer.py), tracking #4130.
- This work (#6714) overlaps the DP-outer gradient reduction — the last-microbatch all-reduce (HSDP) / reduce-scatter (HFSDP) — with the DP-inner reductions, on the mfsdp_v2 experimental path.

  So it's a different collective (parameter gathers vs gradient reductions) on a different code path (v1 vs v2) — the gather side and the reduce side of the DP-outer/DP-inner overlap. Let me know if you see a place they'd actually collide.

### @Achyuthan-S 2026-08-26T15:39:07Z

> > Will share numbers
> 
> Caveat:
> 
> I don't know if you have the right hardware to measure this. This overlapping is mostly useful when dp-outer and dp-inner map to different communication channels, e.g., RDMA vs nvlink.
> 
> When you have a prototype and have verified the overlapping is indeed achieved, [@rapatel](https://github.com/rapatel) and I can measure the impact on a larger, more realistic setting.

Thanks @wujingyue  — understood on the hardware. 

Plan: use the prototype to verify the overlap is actually achieved (DP-outer reduction running concurrently with DP-inner on its own stream, via profiler traces) plus bit-identical parity, then hand it to you and @rapatel to measure impact on a realistic multi-channel setting. 
Will share parity + overlap-verification here.

### @rapatel 2026-08-26T16:10:21Z

@Achyuthan-S Got it - thanks for the explanation, makes sense. That's great - so with your PR and the PR I mentioned, we can overlap all of the outer-DP comms with the inner-DP comms!

### @wujingyue 2026-08-26T16:57:21Z

@rapatel this issue is for v2, not v1. (I think you already realized that -- just in case)

### @rapatel 2026-08-26T17:35:55Z

@wujingyue Yes I got that. Just wanted to point out the other PR because it's a similar optimization (just on the other side of the optimizer step during the first microbatch). Both features should be supported in V2.

### @Achyuthan-S 2026-08-27T01:43:26Z

Agreed — this handles the reduce side (last-microbatch DP-outer reduction) in v2; the gather side is the natural counterpart. Landing the reduce-side overlap here first, parity + overlap-verification coming shortly.


### @Achyuthan-S 2026-08-29T11:21:15Z

@wujingyue — opened the reduce-side DP-outer overlap here (two-buffer design, off by default, per your suggestion). 

Whenever you get a chance, could you mark it ready + /ok to test? The mfsdp_v2 unit bucket runs the HSDP/HFSDP overlap parity for both zero_grad modes — I'd like to confirm that correctness before building out the persistent-vs-transient buffer-lifetime variant for the memory comparison. I'll follow up with the overlap-achieved benchmark separately.

### @shsaihdsaiudh 2026-11-09T14:21:02Z

Hi @wujingyue, I'll take the v2 gather-side counterpart of #5985: running the post-optimizer DP-outer all-gather on its own stream so it overlaps the DP-inner all-gather. `context.py` still notes that outer all-gather separation isn't implemented yet.

I have 2 nodes × 8 H20 (NVLink intra-node, RDMA inter-node). Following the measure-first approach here, I'll first quantify how much of the first microbatch's DP-outer all-gather is exposed on the critical path in HFSDP, with and without CUDA graphs, and post the numbers before writing the implementation.

