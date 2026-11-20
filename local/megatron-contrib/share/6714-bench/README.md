# Benchmark for Megatron-LM #6714 / #9539: DP-outer all-gather on its own stream

A synthetic HFSDP benchmark that keeps only the communication pattern and the compute volume of the case in [#6714](https://github.com/NVIDIA/Megatron-LM/issues/6714). It compares three arms on one node:

| arm | code | `overlap_outer_communication` | DP-outer reduction | DP-outer all-gather |
|---|---|---|---|---|
| `off` | #9539 branch | off | shared stream | on the all-gather stream |
| `main-on` | main | on | own stream (#6973) | on the all-gather stream |
| `pr-on` | #9539 branch | on | own stream | own stream (#9539) |

## Files

- `bench_hfsdp_outer_ag.py`: the benchmark. It needs only PyTorch and `megatron_fsdp` from a Megatron-LM checkout.
- `run.sh`: runs the arms in the order off, main-on, pr-on, pr-on, main-on, off, so that drift affects every arm equally. It then prints a summary.
- `summarize.py`: prints one line per run and the mean per arm. `run.sh` calls it.

## Model and measurement

- 32 residual MLP blocks, `x + W2·gelu(W1·x)`, d=4096, bf16, no attention. Each block is one `fully_shard` unit (256 MB of bf16 parameters).
- HFSDP placements on a `(dp_outer, dp_inner)` mesh. Parameters are replicated across DP-outer and sharded across DP-inner. Optimizer state and fp32 main weights are sharded across both. Gradients are reduce-scattered across DP-inner and averaged across DP-outer.
- Each step runs 4 microbatches with gradient accumulation (mb0..mb3), each with T random tokens, then one SGD step. After the step, the compute weights are stale, so mb0 runs the DP-outer all-gather and mb1..3 do not.
- Timings use CUDA events on the compute stream. Each number is the per-rank median over 8 steps after 3 warm-up steps, then the max over ranks, in ms.
  - `exposed` = forward of mb0 minus the median forward of mb1..3. This is the DP-outer all-gather time on the critical path.
  - `fwd_rest` = median forward of mb1..3.
- `nostep` rows skip the optimizer step, so no DP-outer all-gather runs. Their `exposed` should be close to 0.

## Run

Requirements: one node with 8 GPUs and PyTorch with NCCL. We used torch 2.14.1+cu130 and NCCL 2.30.7.

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git megatron-main
git clone -b mfsdp-6714-outer-ag-overlap https://github.com/shsaihdsaiudh/Megatron-LM.git megatron-9539

PR_REPO=$PWD/megatron-9539 MAIN_REPO=$PWD/megatron-main bash run.sh
```

- With 8 GPUs and `MESH_OUTER=2` (the default), the mesh is (2, 4).
- Results go to `./results-<timestamp>/`. Each run writes a JSON file and a log. The directory also holds `env.txt` (commits, torch, NCCL, GPU) and `summary.txt`.
- `NSYS=1` adds two short runs, off and pr-on, at T=4096 and GA=4, with nsys on rank 0. They are run separately so that nsys overhead does not bias the timings.
- Other options: `NPROC`, `MESH_OUTER`, `OUT_DIR`, `ARMS` (e.g. `ARMS="off pr-on"`), and `BENCH_ARGS` (extra arguments for the benchmark, see `--help`).
- `CUDA_DEVICE_MAX_CONNECTIONS` is taken from the environment. Our runs left it unset; the value used is printed and recorded.

## What to compare

- `fwd_rest` of `train T=4096 GA=4`, across the arms. This is where the slowdown shows.
- `exposed` of `train T=256 GA=4`. This is where #9539 is meant to help.

Our results, on 2 nodes × 8 H20 with mesh (2, 8) (DP-outer across nodes over RDMA), two repeats per arm:

| arm | `exposed`, T=256 GA=4 | `fwd_rest`, T=4096 GA=4 |
|---|---|---|
| off | 22.5 / 23.0 | 269.0 / 269.2 |
| main-on | 22.5 / 24.4 | 269.3 / 269.1 |
| pr-on | 2.6 / 2.4 | 292.3 / 291.8 |

On one node, DP-outer runs over NVLink, so `exposed` will be much smaller than in our results. The `fwd_rest` slowdown happens in microbatches that run no DP-outer all-gather, so it is a more direct check of the stream effect.

Losses are bit-identical across all arms. `summary.txt` reports this on its last line.

Please share `summary.txt` and `env.txt`. If you ran with `NSYS=1`, the `.nsys-rep` files help too.
