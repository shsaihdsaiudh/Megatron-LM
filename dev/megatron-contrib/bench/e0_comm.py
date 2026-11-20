#!/usr/bin/env python3
"""E0 for #6714 (gather side): DP-outer vs DP-inner all-gather microbenchmark.

Pure torch.distributed; no Megatron import. Message sizes follow one HFSDP block
of `Linear(h, m*h) -> Linear(m*h, h)` on a [dp_outer, dp_inner] mesh:

  O (DP-outer AG): [Flat, Flat] -> [Replicate, Flat], in place, like
      post_optimizer_model_weight.redistribute(out=model_weight).
  I (DP-inner AG): [Replicate, Flat] -> [Replicate, Replicate].

Modes, each timed per iteration and reduced with MAX over ranks:
  outer       O alone
  inner       I alone
  serial      O then I on one stream (what v2 does today)
  concurrent  O and I on two streams, joined every iteration (the target)

Launch with torch.distributed.run; see ../cml/e0_entry.sh.
"""

import argparse
import json
import os
import socket
import statistics

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mesh-outer", type=int, default=2, help="DP-outer size")
    parser.add_argument(
        "--hidden", type=int, nargs="+", default=[2048, 4096, 8192], help="block hidden sizes"
    )
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20, help="collectives per timed sample")
    parser.add_argument("--repeats", type=int, default=5, help="timed samples; median reported")
    parser.add_argument("--out-dir", default=None, help="rank 0 writes e0_results.json here")
    return parser.parse_args()


def check_topology(mesh, world_size):
    """Confirm DP-outer groups cross nodes and DP-inner groups stay within a node."""
    hosts = [None] * world_size
    dist.all_gather_object(hosts, socket.gethostname())
    layout = mesh.mesh  # shape (outer, inner), entries are global ranks
    outer_groups = [layout[:, j].tolist() for j in range(layout.shape[1])]
    inner_groups = [layout[i, :].tolist() for i in range(layout.shape[0])]
    return {
        "hosts": hosts,
        "outer_groups": outer_groups,
        "inner_groups": inner_groups,
        "outer_cross_node": all(len({hosts[r] for r in g}) == len(g) for g in outer_groups),
        "inner_intra_node": all(len({hosts[r] for r in g}) == 1 for g in inner_groups),
    }


def measure(fn, args, device):
    """Median over repeats of per-iteration time in ms, MAX-reduced over ranks."""
    for _ in range(args.warmup):
        fn()
    samples = []
    for _ in range(args.repeats):
        torch.cuda.synchronize(device)
        dist.barrier(device_ids=[device.index])
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            fn()
        end.record()
        torch.cuda.synchronize(device)
        sample = torch.tensor([start.elapsed_time(end) / args.iters], device=device)
        dist.all_reduce(sample, op=dist.ReduceOp.MAX)
        samples.append(sample.item())
    return statistics.median(samples)


def bench_hidden(hidden, args, mesh, device):
    outer = mesh.size(0)
    inner = mesh.size(1)
    world = outer * inner
    outer_group = mesh.get_group("dp_outer")
    inner_group = mesh.get_group("dp_inner")
    outer_rank = mesh.get_local_rank("dp_outer")
    dtype = DTYPES[args.dtype]

    numel = 2 * args.ffn_mult * hidden * hidden
    numel = -(-numel // world) * world  # pad to a multiple of the world size
    element_size = torch.tensor([], dtype=dtype).element_size()

    # O: in-place gather into this rank's [Replicate, Flat] shard.
    o_out = torch.empty(numel // inner, dtype=dtype, device=device)
    o_chunk = numel // world
    o_in = o_out.narrow(0, outer_rank * o_chunk, o_chunk)
    # I: separate buffers, as I_i reads layer i while O_{i+1} writes layer i+1.
    i_in = torch.empty(numel // inner, dtype=dtype, device=device)
    i_out = torch.empty(numel, dtype=dtype, device=device)

    def run_outer():
        dist.all_gather_into_tensor(o_out, o_in, group=outer_group)

    def run_inner():
        dist.all_gather_into_tensor(i_out, i_in, group=inner_group)

    def run_serial():
        run_outer()
        run_inner()

    main_stream = torch.cuda.current_stream(device)
    outer_stream = torch.cuda.Stream(device)
    inner_stream = torch.cuda.Stream(device)

    def run_concurrent():
        outer_stream.wait_stream(main_stream)
        inner_stream.wait_stream(main_stream)
        with torch.cuda.stream(outer_stream):
            run_outer()
        with torch.cuda.stream(inner_stream):
            run_inner()
        main_stream.wait_stream(outer_stream)
        main_stream.wait_stream(inner_stream)

    times = {}
    for name, fn in (
        ("outer", run_outer),
        ("inner", run_inner),
        ("serial", run_serial),
        ("concurrent", run_concurrent),
    ):
        torch.cuda.nvtx.range_push(f"e0_h{hidden}_{name}")
        times[name] = measure(fn, args, device)
        torch.cuda.nvtx.range_pop()

    o_ms, i_ms = times["outer"], times["inner"]
    o_bytes = o_out.numel() * element_size
    i_bytes = i_out.numel() * element_size
    hidden_ms = o_ms + i_ms - times["concurrent"]
    return {
        "hidden": hidden,
        "block_numel": numel,
        "block_mb": numel * element_size / 2**20,
        "outer_out_mb": o_bytes / 2**20,
        "inner_out_mb": i_bytes / 2**20,
        **{f"{name}_ms": value for name, value in times.items()},
        "max_ms": max(o_ms, i_ms),
        "sum_ms": o_ms + i_ms,
        # 1.0 means the shorter collective is fully hidden behind the longer one.
        "overlap_efficiency": hidden_ms / min(o_ms, i_ms) if min(o_ms, i_ms) > 0 else 0.0,
        # NCCL-tests convention for all-gather: busbw = out_bytes / t * (n - 1) / n.
        "outer_busbw_gbps": o_bytes / (o_ms * 1e-3) * (outer - 1) / outer / 1e9,
        "inner_busbw_gbps": i_bytes / (i_ms * 1e-3) * (inner - 1) / inner / 1e9,
    }


def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size % args.mesh_outer != 0:
        raise ValueError(f"world size {world_size} is not divisible by {args.mesh_outer}")
    mesh = init_device_mesh(
        "cuda",
        (args.mesh_outer, world_size // args.mesh_outer),
        mesh_dim_names=("dp_outer", "dp_inner"),
    )

    topology = check_topology(mesh, world_size)
    env = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": ".".join(map(str, torch.cuda.nccl.version())),
        "gpu": torch.cuda.get_device_name(device),
        "world_size": world_size,
        "mesh": [mesh.size(0), mesh.size(1)],
        "dtype": args.dtype,
        "CUDA_DEVICE_MAX_CONNECTIONS": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"),
    }
    if rank == 0:
        print(f"[e0] env: {json.dumps(env)}", flush=True)
        print(
            f"[e0] outer_cross_node={topology['outer_cross_node']} "
            f"inner_intra_node={topology['inner_intra_node']} "
            f"outer_groups={topology['outer_groups']}",
            flush=True,
        )

    results = [bench_hidden(hidden, args, mesh, device) for hidden in args.hidden]

    if rank == 0:
        header = (
            f"{'hidden':>6} {'blockMB':>8} {'O ms':>7} {'I ms':>7} {'serial':>7} "
            f"{'concur':>7} {'max':>7} {'eff':>5} {'O GB/s':>7} {'I GB/s':>7}"
        )
        print(header, flush=True)
        for r in results:
            print(
                f"{r['hidden']:>6} {r['block_mb']:>8.1f} {r['outer_ms']:>7.3f} "
                f"{r['inner_ms']:>7.3f} {r['serial_ms']:>7.3f} {r['concurrent_ms']:>7.3f} "
                f"{r['max_ms']:>7.3f} {r['overlap_efficiency']:>5.2f} "
                f"{r['outer_busbw_gbps']:>7.1f} {r['inner_busbw_gbps']:>7.1f}",
                flush=True,
            )
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            path = os.path.join(args.out_dir, "e0_results.json")
            with open(path, "w") as f:
                json.dump(
                    {"args": vars(args), "env": env, "topology": topology, "results": results},
                    f,
                    indent=2,
                )
            print(f"[e0] wrote {path}", flush=True)

    dist.barrier(device_ids=[device.index])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
