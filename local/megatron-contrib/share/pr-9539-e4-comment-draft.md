<!-- Draft 1: comment on #9539 -->

@wujingyue @rapatel I ran #9539 on a real model to check the known issue in the PR description. The forward slowdown from the synthetic benchmark does not show up there. There is no measurable gain on this config either.

### Setup

- 2 nodes × 8 H20e (141 GB, 78 SMs, same SM count as the H20 in the benchmark), NVLink inside a node, RDMA across nodes.
- `pretrain_gpt.py` with a Llama3-8B-shaped GPT: 32 layers, hidden 4096, FFN 14336 (SwiGLU), GQA with 8 groups, TE layers, mock data, bf16.
- Sequence length 4096 and micro batch size 1, so 4096 tokens per microbatch. That is the case where the benchmark showed the slowdown.
- MFSDP v2 HFSDP on a (2, 8) mesh, with DP-outer across nodes: `--use-megatron-fsdp --megatron-fsdp-version 2 --data-parallel-sharding-strategy optim_grads_params --num-distributed-optimizer-instances 2 --outer-dp-sharding-strategy optim --overlap-grad-reduce --overlap-param-gather --grad-reduce-in-bf16`.
- No CUDA graphs. `CUDA_DEVICE_MAX_CONNECTIONS` unset.
- Code: this PR before the rebase (29824b2df) and its base on main (d2b0e35cc). The rebase did not change the logic.
- Arms, as in the benchmark:
  - off: this PR, flag off.
  - main-on: main with `--overlap-outer-communication`.
  - pr-on: this PR with `--overlap-outer-communication`.

### Iteration time

Each value is the median iteration time over iterations 6–20 (GA 4) or 6–15 (GA 8), in ms. GA 4 ran in ABBA order with two runs per arm.

| GA | off | main-on | pr-on |
|---|---|---|---|
| 4 | 6304.6 / 6309.5 | 6300.1 / 6299.4 | 6299.6 / 6299.3 |
| 8 | 12555.1 | 12556.0 | 12555.0 |

All arms are within 0.12% of each other.

### nsys (iterations 10–11, ranks 0 and 8)

- With the flag on, this PR runs the DP-outer all-gather on its own stream, as intended. With the flag off, it shares the DP-inner all-gather stream.
- The race I described in #6714 happens here in both arms. In the race, a DP-inner all-gather starts in the gap before the next compute kernel. It happens for 24–25% of DP-inner all-gathers with the flag off and 26–30% with this PR (528 per report). In the benchmark the split was 0% vs about 50%.
- The race costs nothing here. Total GEMM time on the compute stream is the same in both arms: 11939.5 vs 11935.6 ms on rank 0, and 11931.7 vs 11940.8 ms on rank 8.
- The DP-outer all-gather kernels take about 30 ms per step on rank 0, out of a 6.3 s step. So on this config the most this PR could save is about 0.5%, which matches the result.

So the slowdown in my earlier comment comes from the synthetic model: two GEMMs per `fully_shard` unit and nothing else. With attention, norms, and TE GEMMs in each layer, it disappears. I'll update the "Known issue" section of the PR description.

On H20, compute is slow enough to hide most of the DP-outer all-gather at 4096 tokens per microbatch. I'd expect a gain only where that all-gather is a larger share of the step: fewer tokens per microbatch, faster GPUs, or CUDA graphs. That is the kind of setting you and @rapatel offered to test, so could you run this PR on one of your configs?

Next, I'll run `tests/unit_tests/distributed/mfsdp_v2/` on 8 GPUs on the rebased branch and post the result here. If you'd rather keep the all-gather side behind its own option instead of `overlap_outer_communication`, I can split it out.

<!-- Draft 2: replacement for the "Known issue" section in the PR description -->

### Known issue (synthetic benchmark only)

In the synthetic benchmark from #6714 (32 residual MLP blocks, no attention, 4096 tokens per microbatch), the forward of mb1..3 is about 8–10% slower with this change. On a real model the slowdown does not appear. The real-model run used a Llama3-8B-shaped GPT via `pretrain_gpt.py` on 2 × 8 H20e with mesh (2, 8). Iteration time with this PR is within 0.12% of flag off and of main with the flag on, at GA 4 and GA 8. Details are in [this comment](<link to the comment above>).
