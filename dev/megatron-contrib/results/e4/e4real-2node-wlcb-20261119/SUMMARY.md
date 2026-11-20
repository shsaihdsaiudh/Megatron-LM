# E4: real-model A/B for #9539 (wlcb, 2 x 8 H20e)

Job `t-20261119174043-nw3zf`, queue 13192 (<redacted-queue>). Full logs and nsys reports are on
the wlcb volume: <redacted>

Setup: `pretrain_gpt.py`, Llama3-8B shape (32 layers, hidden 4096, FFN 14336, GQA 8), TE, mock data,
seq 4096, micro batch 1 (4096 tokens per microbatch), MFSDP v2 HFSDP: `--num-distributed-optimizer-instances 2
--outer-dp-sharding-strategy optim`, inner `optim_grads_params`, mesh (2, 8), DP-outer across nodes.
`CUDA_DEVICE_MAX_CONNECTIONS` unset. Snapshots: PR 29824b2df, main d2b0e35cc.

## Iteration time (median over iterations 6..N, ms)

| GA | off | main-on | pr-on |
|---|---|---|---|
| 4 (ABBA, 2 runs each) | 6304.6 / 6309.5 | 6300.1 / 6299.4 | 6299.6 / 6299.3 |
| 8 (1 run each) | 12555.1 | 12556.0 | 12555.0 |

All arms are within 0.12% of each other. No slowdown and no gain. About 125 TFLOP/s/GPU.

## nsys (iterations 10-11, ranks 0 and 8)

- pr-on really splits the streams: DP-outer AllGather (gridX 4) on its own stream, DP-outer
  ReduceScatter on another. In off both share the DP-inner streams.
- DP-outer AllGather kernel time is about 28-33 ms per step on rank 0 (66 kernels over 2 steps),
  about 0.5% of a 6.3 s step, so the possible gain at this config is small.
- The race from #6714 (DP-inner AG starting in the gap before the next compute kernel) happens in
  both arms: off 24-25%, pr-on 26-30% of 528 AG-I. In the benchmark it was 0% (off) vs about 50% (P1).
- Total GEMM kernel time on the compute stream is the same: rank 0 11939.5 (off) vs 11935.6 ms
  (pr-on); rank 8 11931.7 vs 11940.8 ms. Compute-stream busy time is the same too.

## Caveats

- H20e (141 GB), not H20 (96 GB); both have 78 SMs.
- One shape at 4096 tokens per microbatch, the regime where the benchmark showed the slowdown.
  The small-T regime where #9539 is meant to help was not run with the real model.
- Losses differ slightly even between runs of the same arm (no `--deterministic-mode`), so this run
  does not check bit-exact parity.

Scripts: `cml/e4_entry.sh`, `cml/e4-2node-wlcb.yaml`, `bench/e4_summary.py`,
`bench/e4_nsys_overview.py`, `bench/e4_nsys_race.py`.
