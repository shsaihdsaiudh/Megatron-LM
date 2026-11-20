# Megatron-LM contribution: MFSDP v2 gather-side DP-outer overlap

Handoff note for work on the dev machine (`cloudml`). Prepared from WSL on 2026-11-09.

## Goal

Claimed in https://github.com/NVIDIA/Megatron-LM/issues/6714#issuecomment-6742364179
(posted as GitHub user `shsaihdsaiudh`; no maintainer reply yet as of 2026-11-09).

Implement the MFSDP v2 counterpart of v1 PR #5985: run the post-optimizer DP-outer all-gather
on its own CUDA stream so it overlaps the DP-inner all-gather. **Measure first**, post numbers
on #6714, then implement.

### What the code does today (main @ d2b0e35cc)

Code lives in `megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/`.

- HFSDP placements on a `[DP-outer, DP-inner]` mesh: optimizer / main weight `[Flat, Flat]`,
  resident compute weight `model_weight` `[Replicate, Flat]`, unsharded for compute
  `[Replicate, Replicate]`.
- After the optimizer step, `ParameterGroup.sync_model_weight_from_main_weight()` casts into
  `post_optimizer_model_weight` (still `[Flat, Flat]`) and marks `_model_weight_is_stale`.
- In the first microbatch, `ParameterGroup.unshard_parameters()` does, per group and on the same
  `allgather_stream`:
  1. `post_optimizer_model_weight.redistribute(model_weight.placements, out=model_weight)`
     (DP-outer all-gather, inter-node RDMA)
  2. `_allgather_model_weight()` (DP-inner all-gather, intra-node NVLink)
  So layer i+1's outer gather queues behind layer i's inner gather.
- `context.py` (`FsdpContext.__init__`) says: "Outer all-gather separation is not implemented
  yet; the outer option currently affects only reduction." `RuntimePolicy.overlap_outer_communication`
  only moves the DP-outer *reduction* to `outer_reduction_stream` (added by #6973).
- Later microbatches only do the DP-inner gather, so the optimization only affects microbatch 1
  of each step.

### Not to be confused with `hfsdp_param_gather_overlap`

The public docs and `megatron/core/distributed/fsdp/src/README.md` describe
`hfsdp_param_gather_overlap` (DP-Outer prefetched one FSDP unit beyond the DP-Inner frontier,
off by default). That is the **v1** option added by #5985 (2026-08-31). It lives in
`megatron_fsdp/fully_shard.py`, `param_and_grad_buffer.py`, both `distributed_data_parallel_config.py`
files, and `tests/unit_tests/distributed/mfsdp_v1/`. Nothing under `experimental/` or
`tests/unit_tests/distributed/mfsdp_v2/` references it, and the "not implemented yet" note in
`experimental/context.py` is still there (checked at d2b0e35cc). It is the design to port, not
a v2 implementation. Re-check with `git grep` after each `sync.sh`.

### Expectations to keep in mind

- Reduce-side overlap (#6973) measured 0.06% on 256 GPUs DeepSeek-V3 HFSDP+EP with CUDA graphs
  (wujingyue) and ~2% on 128 GB200 DeepSeek-V4-Flash (Autumn1998). Gather side is likely the
  same order of magnitude. A near-zero result is still worth reporting.
- H20 has low compute relative to H100/B200, so communication is easier to hide; say so when
  reporting numbers.
- The maintainers run with training and optimizer CUDA graphs. Measure with and without them.
- Maintainer's process: read `megatron/core/distributed/fsdp/src/docs/runtime_schedule.md`
  first; small PRs; option off by default; parity tests (bit-identical losses) plus a test that
  verifies the overlap really happens.

### Measurement plan

1. Run the MFSDP v2 unit tests (`tests/unit_tests/distributed/mfsdp_v2/`) on 8 GPUs, then on
   2 nodes x 8.
2. HFSDP, mesh `(2, 8)`, Llama-8B-sized GPT, several microbatches per step.
3. nsys: compare forward time of microbatch 1 vs microbatch 2. Same compute; the difference is
   roughly the exposed DP-outer all-gather time, i.e. the upper bound of what this change saves.
   Confirm by looking for compute-stream gaps waiting on `allgather_stream`.
4. Vary microbatch count and model size; report the difference as a fraction of step time.

## Layout

- Repo: `/planning-users/yangyeyuan/Megatron-LM` (cloned from a bundle; `origin` points at
  GitHub but **the dev machine cannot reach GitHub**).
  - `origin/main`: upstream main. Pushed from WSL.
  - `pr/<N>`: PR heads: 5985 (v1 gather-side prefetch, the design to port), 6973 (v2
    reduce-side overlap, the pattern to follow), 8395, 8787, 9105.
- Discussions: `/planning-users/yangyeyuan/megatron-contrib/github/*.md`, exported issue/PR
  conversations and inline review comments: issue-5616, issue-6714, issue-7814, pr-5985,
  pr-6973, pr-8395, pr-8787, pr-9105. Read pr-6973.md for what reviewers asked for.

## What has to happen in WSL, not on the dev machine

The dev machine has no GitHub / nvcr.io access. From WSL:

- `~/src/megatron-contrib/sync.sh [numbers...]`: fetch upstream main and PR heads, push them to the
  dev clone, refresh the markdown exports. Pass extra PR/issue numbers to add more.
- Posting comments, opening PRs (needs a fork under `shsaihdsaiudh` and pushing from
  `~/src/Megatron-LM`).

## Environment (not set up yet)

- Dev machine: 1x H20, Python 3.10, torch 2.6.0+cu126, no Transformer Engine, no nvcc.
  Megatron main requires Python >= 3.12 (`pyproject.toml`); upstream CI uses
  `nvcr.io/nvidia/pytorch:26.09-py3`.
- pip works through `<redacted-url>`.
- Job images must be under `<redacted-image>`. Closest platform preset:
  `<redacted-image>`
  (Python 3.12, torch 2.10, CUDA 13.1). MFSDP v2 code itself is pure PyTorch; TE is only needed
  for TE layers.
- Jobs: `cml` works on the dev machine (`~/.cloudml-cli/bin/cml`, not on PATH).
  `cml custom_train template` gives a yaml template. A reference export of an earlier job is
  at `cml/reference-job.yaml`. H20 queue: `12790` (`<redacted-queue>`), resource
  `<redacted-resource>`, 8 GPUs per node. For 2 nodes set `nodeNumber: 2` and `rdma: true`.
  Mount `planning-users` via `juiceFsMountConfigs` as in the reference.
