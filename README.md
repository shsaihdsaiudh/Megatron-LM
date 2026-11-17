# nsys traces for #6714 (gather side)

Rank 0 of a 2 node × 8 H20 run, HFSDP mesh (dp_outer=2, dp_inner=8).
Synthetic benchmark: 32 residual MLP blocks (`x + W2·gelu(W1·x)`), d=4096, bf16, one FSDP unit per block.
T=4096 tokens per microbatch, 4 microbatches per step (gradient accumulation).
Each file profiles two steps (`train_T4096_ga4_step3`, `train_T4096_ga4_step4`);
every microbatch has NVTX ranges `mb{i}_fwd` and `mb{i}_bwd`.

| File | Arm | Code | Flag |
|---|---|---|---|
| `v2_off/d4096_L32_T4096_ga4.rank0.nsys-rep` | MFSDP v2, outer AG on the all-gather stream | P1 branch | none |
| `v2_p1/d4096_L32_T4096_ga4.rank0.nsys-rep` | MFSDP v2, outer AG on its own stream | P1 branch | `--overlap-outer-communication` |
| `v1_base/d4096_L32_T4096_ga4.rank0.nsys-rep` | MFSDP v1 | main @ d2b0e35cc | none |
| `v1_5985/d4096_L32_T4096_ga4.rank0.nsys-rep` | MFSDP v1 + #5985 | main @ d2b0e35cc | `--hfsdp-param-gather-overlap` |

P1 branch: https://github.com/shsaihdsaiudh/Megatron-LM/tree/mfsdp-v2-outer-allgather-stream

Streams on rank 0:

| File | Compute | DP-inner all-gather |
|---|---|---|
| v2 | stream 7 | stream 39 |
| v1 | stream 7 | stream 43 |
