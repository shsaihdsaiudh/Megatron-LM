"""E1 for MFSDP v1: exposed DP-outer all-gather and later-microbatch forward cost (#6714).

Same model, mesh, data, metrics, JSON layout as e1_exposure.py (v2), on the v1 API
(``megatron_fsdp.fully_shard_model``): zero_dp_strategy=3 (inner shards params,
grads, optimizer state), outer_dp_sharding_strategy="optim" (HFSDP), one FSDP unit
per Block. v1 runs the DP-outer all-gather in the first forward after
``optimizer.step()`` (``microbatch_count == 0``), on its own
``outer_fsdp_group_param_gather_stream``, then the inner all-gather on the
param-gather side stream after waiting for it. ``--hfsdp-param-gather-overlap``
enables #5985 (DP-outer prefetched one unit beyond the inner frontier).

Control ``nostep``: no optimizer.step(); gradients are still synced
(finish_grad_sync) and microbatch_count is then set to 1, so mb0 runs no outer
all-gather, as in the v2 control.

Launch with torchrun; mesh is (MESH_OUTER, WORLD_SIZE / MESH_OUTER).
"""

import argparse
import contextlib
import json
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from e1_exposure import Model, all_ranks, summarize
from megatron_fsdp import fully_shard_model, fully_shard_optimizer
from megatron_fsdp.mixed_precision import MixedPrecisionPolicy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--mesh-outer", type=int, default=2)
    p.add_argument("--tokens", type=str, default="256,1024,4096")
    p.add_argument("--ga", type=str, default="4")
    p.add_argument("--control-ga", type=int, default=4)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--hfsdp-param-gather-overlap", action="store_true")
    p.add_argument("--profile", type=str, default="")
    p.add_argument("--profile-steps", type=int, default=2)
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--out-name", type=str, default="")
    return p.parse_args()


def run_config(model, optimizer, d, tokens, ga, mode, warmup, steps, profile, device):
    """Same loop and timing points as e1_exposure.run_config, with v1 GA controls."""
    gen = torch.Generator(device=device).manual_seed(1234 + dist.get_rank())
    xs = [torch.randn(tokens, d, device=device, dtype=torch.bfloat16, generator=gen) for _ in range(ga)]
    ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731
    rec = {"fwd": [], "bwd": [], "step": [], "opt": []}
    losses = []
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(warmup + steps):
        measured = step >= warmup
        if profile and step == warmup:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
        f_s, f_e, b_e = [ev() for _ in range(ga)], [ev() for _ in range(ga)], [ev() for _ in range(ga)]
        s_s, o_s, s_e = ev(), ev(), ev()
        s_s.record()
        torch.cuda.nvtx.range_push(f"{mode}_T{tokens}_ga{ga}_step{step}")
        optimizer.zero_grad(set_to_none=False)
        for mb in range(ga):
            last = mb == ga - 1
            ctx = contextlib.nullcontext() if last else model.no_sync()
            # Last microbatch: launch the DP-outer gradient reduction (v1 reads this flag).
            model.is_last_microbatch = last
            with ctx:
                torch.cuda.nvtx.range_push(f"mb{mb}_fwd")
                f_s[mb].record()
                loss = model(xs[mb]).float().pow(2).mean()
                f_e[mb].record()
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push(f"mb{mb}_bwd")
                (loss / ga).backward()
                b_e[mb].record()
                torch.cuda.nvtx.range_pop()
            losses.append(loss.detach())
        o_s.record()
        if mode == "train":
            torch.cuda.nvtx.range_push("optimizer")
            optimizer.step()  # finish_grad_sync, step, install weights; microbatch_count = 0
            torch.cuda.nvtx.range_pop()
        else:
            model.finish_grad_sync()
            # Weights did not change: skip the DP-outer all-gather in the next mb0.
            model.microbatch_count = 1
        s_e.record()
        torch.cuda.nvtx.range_pop()
        if profile and step == warmup + profile - 1:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
        if measured:
            torch.cuda.synchronize()
            rec["fwd"].append([f_s[i].elapsed_time(f_e[i]) for i in range(ga)])
            rec["bwd"].append([f_e[i].elapsed_time(b_e[i]) for i in range(ga)])
            rec["opt"].append(o_s.elapsed_time(s_e))
            rec["step"].append(s_s.elapsed_time(s_e))
    torch.cuda.synchronize()
    loss_vals = torch.stack(losses).float()
    rec["finite_losses"] = bool(torch.isfinite(loss_vals).all())
    rec["losses"] = loss_vals.tolist()
    rec["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
    return rec


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world % args.mesh_outer == 0
    inner = world // args.mesh_outer
    # Same dims and flattening as tests/unit_tests/distributed/mfsdp_v1 (CP = TP = 1).
    mesh = init_device_mesh(
        "cuda", (args.mesh_outer, inner, 1, 1), mesh_dim_names=("dp_outer", "dp_shard", "cp", "tp")
    )
    mesh[("dp_outer", "dp_shard")]._flatten("dp")
    mesh[("dp_shard", "cp")]._flatten("dp_shard_cp")
    mesh[("dp_outer", "dp_shard", "cp")]._flatten("hsdp")
    d, layers = args.hidden, args.layers

    torch.manual_seed(0)
    with torch.device(device):
        module = Model(d, layers).to(torch.bfloat16)
    block_type = type(module.blocks[0])
    model = fully_shard_model(
        module=module,
        device_mesh=mesh,
        dp_shard_dim="dp_shard_cp",
        dp_outer_dim="dp_outer",
        tp_dim="tp",
        hybrid_fsdp_group=mesh["hsdp"].get_group(),
        fsdp_unit_modules=[block_type],
        zero_dp_strategy=3,
        outer_dp_sharding_strategy="optim",
        mixed_precision_policy=MixedPrecisionPolicy(main_params_dtype=torch.float32),
        sync_model_each_microbatch=False,
        hfsdp_param_gather_overlap=args.hfsdp_param_gather_overlap,
    )
    torch.cuda.empty_cache()
    optimizer = fully_shard_optimizer(torch.optim.SGD(model.parameters(), lr=1e-6))

    block_mb = 8 * d * d * 2 / 2**20
    if rank == 0:
        agp = model.all_gather_pipeline
        print(
            f"[e1v1] world={world} mesh=({args.mesh_outer},{inner}) d={d} layers={layers} "
            f"block={block_mb:.0f}MB hfsdp_param_gather_overlap={args.hfsdp_param_gather_overlap} "
            f"suggested_AG_prefetch_size={model.suggested_AG_prefetch_size} "
            f"outer_ag_is_ag_stream={agp.outer_fsdp_group_param_gather_stream is agp.ag_stream} "
            f"CUDA_DEVICE_MAX_CONNECTIONS={os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS')}",
            flush=True,
        )

    profile_cfg = tuple(int(v) for v in args.profile.split(",")) if args.profile else None
    configs = [("train", t, g) for t in map(int, args.tokens.split(",")) for g in map(int, args.ga.split(","))]
    if args.control_ga > 0:
        configs += [("nostep", t, args.control_ga) for t in map(int, args.tokens.split(","))]

    results = []
    for mode, tokens, ga in configs:
        t0 = time.time()
        profile = args.profile_steps if profile_cfg == (tokens, ga) and mode == "train" else 0
        rec = run_config(model, optimizer, d, tokens, ga, mode, args.warmup, args.steps, profile, device)
        per_rank = all_ranks(summarize(rec, ga), device)
        row = {
            "mode": mode,
            "tokens": tokens,
            "ga": ga,
            "max": {k: max(v) for k, v in per_rank.items()},
            "rank0": {k: v[0] for k, v in per_rank.items()},
            "per_rank": per_rank,
            "raw_rank0": rec if rank == 0 else None,
        }
        results.append(row)
        if rank == 0:
            m = row["max"]
            e = f"E={m['exposed']:7.2f} fwd_rest={m['fwd_rest']:7.2f} " if ga > 1 else " " * 33
            print(
                f"[e1] {mode:6s} T={tokens:5d} GA={ga} fwd_mb0={m['fwd_mb0']:7.2f} {e}"
                f"step={m['step']:8.2f} opt={m['opt']:6.2f} mem={m['peak_mem_gb']:5.1f}GB "
                f"finite={bool(m['finite'])} ({time.time() - t0:.0f}s)",
                flush=True,
            )

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, args.out_name or f"e1_d{d}_L{layers}.json")
        meta = {
            "version": "v1",
            "world": world,
            "mesh": [args.mesh_outer, inner],
            "hidden": d,
            "layers": layers,
            "block_mb_bf16": block_mb,
            "warmup": args.warmup,
            "steps": args.steps,
            "hfsdp_param_gather_overlap": args.hfsdp_param_gather_overlap,
            "outer_ag_on_ag_stream": (
                model.all_gather_pipeline.outer_fsdp_group_param_gather_stream
                is model.all_gather_pipeline.ag_stream
            ),
            "torch": torch.__version__,
            "nccl": list(torch.cuda.nccl.version()),
            "units": "ms",
        }
        with open(path, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=1)
        print(f"[e1] wrote {path}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
