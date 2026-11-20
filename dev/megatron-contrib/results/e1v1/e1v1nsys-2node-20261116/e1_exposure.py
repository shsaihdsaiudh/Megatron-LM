"""E1: exposed DP-outer all-gather in MFSDP v2 HFSDP baseline (#6714 gather side).

Unmodified main. N residual MLP blocks, each fully_shard-ed, plus the root, on a
(dp_outer, dp_inner) mesh with HFSDP placements. After optimizer.step() every
parameter group's compute weight is stale, so the first microbatch's
pre-forward unshard runs the DP-outer all-gather (redistribute of
post_optimizer_model_weight) followed by the DP-inner all-gather, both on the
all-gather stream. Later microbatches only run the inner all-gather. With equal
compute per microbatch:

    E = T_fwd(mb0) - median(T_fwd(mb1..))   # exposed outer all-gather time

Control mode ``nostep`` skips optimizer.step(): weights never go stale, mb0 has
no outer all-gather, and E should be ~0. This checks that E is attributable to
the outer gather and not to other first-microbatch effects.

Timings are CUDA events on the compute stream; per-rank medians over the measured
steps, then MAX over ranks (the slowest rank sets the step time).

Launch with torchrun; the mesh is (MESH_OUTER, WORLD_SIZE / MESH_OUTER).
"""

import argparse
import json
import math
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Partial, Replicate, Shard

from megatron_fsdp.experimental import (
    FsdpContext,
    Placements,
    RuntimePolicy,
    finalize_gradients,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron_fsdp.mixed_precision import MixedPrecisionPolicy


class Block(nn.Module):
    """Residual MLP: x + W2 gelu(W1 x). 8 d^2 parameters."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class Model(nn.Module):
    """Blocks plus a root-owned scale so the root FsdpModule has its own group."""

    def __init__(self, d: int, layers: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.blocks = nn.ModuleList([Block(d) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.scale
        for block in self.blocks:
            x = block(x)
        return x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--mesh-outer", type=int, default=2)
    p.add_argument("--tokens", type=str, default="128,256,512,1024,2048,4096")
    p.add_argument("--ga", type=str, default="1,4,8")
    p.add_argument("--control-ga", type=int, default=4, help="GA for the nostep control; 0 disables")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--overlap-outer-communication", action="store_true")
    p.add_argument(
        "--profile",
        type=str,
        default="",
        help="'tokens,ga' to bracket that config's measured steps with cudaProfilerStart/Stop",
    )
    p.add_argument("--profile-steps", type=int, default=2)
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--out-name", type=str, default="", help="JSON file name; default e1_d<d>_L<l>.json")
    return p.parse_args()


def run_config(model, optimizer, context, d, tokens, ga, mode, warmup, steps, profile, device):
    """Run warmup + measured steps; return this rank's per-step timing lists (ms)."""
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
            with finalize_gradients(context, enabled=mb == ga - 1):
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
            optimizer.step()
            torch.cuda.nvtx.range_pop()
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
    # Every microbatch loss of every step (warmup included), for bit-exact comparison
    # between runs with and without overlap_outer_communication.
    rec["losses"] = loss_vals.tolist()
    rec["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
    return rec


def summarize(rec, ga):
    """Per-rank scalars: median over measured steps."""
    med = statistics.median
    out = {
        "fwd_mb0": med(s[0] for s in rec["fwd"]),
        "bwd_mb0": med(s[0] for s in rec["bwd"]),
        "step": med(rec["step"]),
        "opt": med(rec["opt"]),
        "peak_mem_gb": rec["peak_mem_gb"],
        "finite": float(rec["finite_losses"]),
        # float64 sum of float32 losses: equal across runs iff (practically) losses match.
        "loss_sum": math.fsum(rec["losses"]),
    }
    if ga > 1:
        out["fwd_rest"] = med(x for s in rec["fwd"] for x in s[1:])
        out["bwd_rest"] = med(x for s in rec["bwd"] for x in s[1:])
        # Per-step difference first, then median: robust to slow drifting steps.
        out["exposed"] = med(s[0] - statistics.median(s[1:]) for s in rec["fwd"])
    return out


def all_ranks(values: dict, device) -> dict:
    """Gather per-rank scalar dicts -> {key: [rank0, rank1, ...]}."""
    keys = sorted(values)
    t = torch.tensor([values[k] for k in keys], device=device, dtype=torch.float64)
    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    return {k: [g[i].item() for g in gathered] for i, k in enumerate(keys)}


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world % args.mesh_outer == 0
    inner = world // args.mesh_outer
    mesh = init_device_mesh("cuda", (args.mesh_outer, inner), mesh_dim_names=("dp_outer", "dp_inner"))
    d, layers = args.hidden, args.layers

    torch.manual_seed(0)
    with torch.device(device):
        model = Model(d, layers).to(torch.bfloat16)
    placements = Placements(
        dp_axes=[0, 1],
        parameter=[Replicate(), Shard(0)],
        gradient=[Partial("avg"), Shard(0)],
        optimizer=[Shard(0), Shard(0)],
    )
    mp = MixedPrecisionPolicy(main_params_dtype=torch.float32)
    policy = RuntimePolicy(overlap_outer_communication=args.overlap_outer_communication)
    with fully_shard_context(FsdpContext(device=device, runtime_policy=policy)) as context:
        for block in model.blocks:
            fully_shard(block, mesh=mesh, placements=placements, mixed_precision_policy=mp)
        fully_shard(model, mesh=mesh, placements=placements, mixed_precision_policy=mp)
    torch.cuda.empty_cache()
    # Tiny LR: weights stay well-scaled; only the communication pattern matters.
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
    fully_shard_optimizer(optimizer)

    block_mb = 8 * d * d * 2 / 2**20
    if rank == 0:
        print(
            f"[e1] world={world} mesh=({args.mesh_outer},{inner}) d={d} layers={layers} "
            f"block={block_mb:.0f}MB bf16 total={layers * block_mb / 1024:.1f}GB "
            f"overlap_outer_communication={args.overlap_outer_communication} "
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
        rec = run_config(model, optimizer, context, d, tokens, ga, mode, args.warmup, args.steps, profile, device)
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
            "world": world,
            "mesh": [args.mesh_outer, inner],
            "hidden": d,
            "layers": layers,
            "block_mb_bf16": block_mb,
            "warmup": args.warmup,
            "steps": args.steps,
            "overlap_outer_communication": args.overlap_outer_communication,
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
