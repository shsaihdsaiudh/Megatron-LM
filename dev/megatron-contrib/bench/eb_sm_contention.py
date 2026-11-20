"""Experiment B (#6714 gather side): do NCCL all-gathers wait for SMs held by GEMMs?

No FSDP. One node, all ranks in one group (NVLink), shapes from E1 (d=4096 MLP
block: fc1 [T,d]x[d,4d], gelu, fc2 [T,4d]x[4d,d]; AG of one 256 MB bf16 block,
32 MB in / 256 MB out per rank, like the DP-inner gather in E1).

Per config (tokens T, cuBLAS SM carveout, stream priority) it measures:

- gemm_alone: L-layer compute chain, no communication.
- ag_alone:   L back-to-back AGs, idle compute. Per-AG time A0.
- independent: compute chain runs freely. AG_{i+1} becomes ready when layer
  i-1 finishes, i.e. exactly when layer i's fc1 is about to start, the pattern
  seen in E1. On the AG stream: wait(compute) -> record(ready) -> AG ->
  record(done). AG delay = (done - ready) - A0, i.e. how much longer the AG
  takes than on an idle GPU (start delay + slowdown). Compute is never blocked.
- fsdp_like: same issue order (prefetch one layer ahead, like MFSDP), and layer
  i waits for AG_i before fc1. chain_overhead = chain - gemm_alone chain is the
  exposed communication.

H1 (SM contention) predicts AG delay ~ GEMM-tail wait without carveout and a
drop to ~0 with carveout. Stream priority tests whether the block scheduler
prefers the AG once SMs free up. NCCL_MAX_CTAS (set per launch by the entry
script) tests whether a smaller AG grid starts sooner.

Results: medians over reps per rank, MAX over ranks. Writes eb_<tag>.json.
"""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist
import torch.nn.functional as F

med = statistics.median


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--tokens", type=str, default="256,1024,4096")
    p.add_argument("--carveout", type=str, default="0,8,16")
    p.add_argument("--priority", type=str, default="normal,high")
    p.add_argument("--modes", type=str, default="independent,fsdp_like")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--tag", type=str, default="default")
    p.add_argument("--profile", action="store_true", help="cudaProfilerStart/Stop around the whole sweep")
    p.add_argument("--out-dir", type=str, default=".")
    return p.parse_args()


def ev():
    return torch.cuda.Event(enable_timing=True)


class Bench:
    def __init__(self, args, device):
        d, self.L = args.hidden, args.layers
        self.d, self.device = d, device
        world = dist.get_world_size()
        bf = dict(device=device, dtype=torch.bfloat16)
        self.w1 = [torch.randn(4 * d, d, **bf) * 0.01 for _ in range(self.L)]
        self.w2 = [torch.randn(d, 4 * d, **bf) * 0.01 for _ in range(self.L)]
        block = 8 * d * d
        self.ag_in = torch.randn(block // world, **bf)
        self.ag_out = [torch.empty(block, **bf) for _ in range(2)]
        hi = dist.ProcessGroupNCCL.Options()
        hi.is_high_priority_stream = True
        lo_prio, hi_prio = torch.cuda.Stream.priority_range()
        self.groups = {"normal": dist.new_group(), "high": dist.new_group(pg_options=hi)}
        self.streams = {
            "normal": torch.cuda.Stream(device, priority=lo_prio),
            "high": torch.cuda.Stream(device, priority=hi_prio),
        }

    def ag(self, i, prio):
        dist.all_gather_into_tensor(self.ag_out[i % 2], self.ag_in, group=self.groups[prio])

    def layer(self, x, i):
        return F.linear(F.gelu(F.linear(x, self.w1[i])), self.w2[i])

    def run(self, mode, tokens, prio, reps, warmup, label):
        x = torch.randn(tokens, self.d, device=self.device, dtype=torch.bfloat16)
        comp = torch.cuda.current_stream()
        ags = self.streams[prio]
        out = {"chain": [], "ag": []}
        for rep in range(warmup + reps):
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f"{label}_rep{rep}")
            c_s, c_e = ev(), ev()
            ready = [ev() for _ in range(self.L)]
            done = [ev() for _ in range(self.L)]
            if mode == "ag_alone":
                with torch.cuda.stream(ags):
                    for i in range(self.L):
                        ready[i].record()
                        self.ag(i, prio)
                        done[i].record()
                comp.wait_stream(ags)
            else:
                c_s.record()
                ags.wait_stream(comp)
                comm = mode != "gemm_alone"

                def issue(j):
                    with torch.cuda.stream(ags):
                        ready[j].record()
                        self.ag(j, prio)
                        done[j].record()

                if comm:
                    issue(0)
                h = x
                for i in range(self.L):
                    if comm and i + 1 < self.L:
                        # Prefetch one layer ahead, like MFSDP: AG_{i+1} becomes ready when
                        # layer i-1 has finished (everything enqueued on compute so far),
                        # so it can overlap layer i.
                        ags.wait_stream(comp)
                        issue(i + 1)
                    if mode == "fsdp_like":
                        comp.wait_event(done[i])
                    torch.cuda.nvtx.range_push(f"layer{i}")
                    h = self.layer(h, i)
                    torch.cuda.nvtx.range_pop()
                c_e.record()
                comp.wait_stream(ags)
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            if rep < warmup:
                continue
            if mode != "ag_alone":
                out["chain"].append(c_s.elapsed_time(c_e))
            if mode != "gemm_alone":
                # ag_alone: skip AG_0 (warm). Chains: skip AG_0/AG_1, ready before any GEMM ran.
                first = 1 if mode == "ag_alone" else 2
                out["ag"].append([ready[i].elapsed_time(done[i]) for i in range(first, self.L)])
        return out


def max_over_ranks(vals: dict, device) -> dict:
    keys = sorted(vals)
    t = torch.tensor([vals[k] for k in keys], device=device, dtype=torch.float64)
    g = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(g, t)
    return {k: max(x[i].item() for x in g) for i, k in enumerate(keys)}


def main():
    args = parse_args()
    lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    device = torch.device("cuda", lr)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    b = Bench(args, device)
    tokens = [int(t) for t in args.tokens.split(",")]
    carveouts = [int(c) for c in args.carveout.split(",")]
    prios = args.priority.split(",")
    modes = args.modes.split(",")
    ctas = os.environ.get("NCCL_MAX_CTAS", "default")
    if rank == 0:
        print(f"[eb] world={dist.get_world_size()} d={args.hidden} L={args.layers} tag={args.tag} "
              f"NCCL_MAX_CTAS={ctas} CUDA_DEVICE_MAX_CONNECTIONS={os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS')} "
              f"stream priority range={torch.cuda.Stream.priority_range()}", flush=True)
    if args.profile:
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
    rows = []
    for co in carveouts:
        torch._C._set_sm_carveout_experimental(co)
        for prio in prios:
            a0 = b.run("ag_alone", tokens[0], prio, args.reps, args.warmup, f"ag_alone_c{co}_{prio}")
            a0_med = med(x for r in a0["ag"] for x in r)
            for t in tokens:
                g0 = b.run("gemm_alone", t, prio, args.reps, args.warmup, f"gemm_alone_T{t}_c{co}_{prio}")
                g0_med = med(g0["chain"])
                for mode in modes:
                    r = b.run(mode, t, prio, args.reps, args.warmup, f"{mode}_T{t}_c{co}_{prio}")
                    ag_all = [x for rep in r["ag"] for x in rep]
                    delays = sorted(x - a0_med for x in ag_all)
                    vals = {
                        "ag_alone": a0_med,
                        "gemm_chain": g0_med,
                        "chain": med(r["chain"]),
                        "chain_overhead": med(r["chain"]) - g0_med,
                        "ag_delay_med": med(delays),
                        "ag_delay_p90": delays[int(0.9 * (len(delays) - 1))],
                    }
                    m = max_over_ranks(vals, device)
                    row = dict(mode=mode, tokens=t, carveout=co, priority=prio, nccl_max_ctas=ctas, max=m,
                               raw_rank0={"chain": r["chain"], "ag": r["ag"]} if rank == 0 else None)
                    rows.append(row)
                    if rank == 0:
                        print(f"[eb] {mode:11s} T={t:5d} carveout={co:2d} prio={prio:6s} "
                              f"ag_alone={m['ag_alone']:.3f} ag_delay med={m['ag_delay_med']:.3f} "
                              f"p90={m['ag_delay_p90']:.3f} gemm_chain={m['gemm_chain']:.2f} "
                              f"chain={m['chain']:.2f} overhead={m['chain_overhead']:.2f}", flush=True)
    torch._C._set_sm_carveout_experimental(None)
    if args.profile:
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, f"eb_{args.tag}.json")
        meta = dict(world=dist.get_world_size(), hidden=args.hidden, layers=args.layers, warmup=args.warmup,
                    reps=args.reps, nccl_max_ctas=ctas, torch=torch.__version__,
                    nccl=list(torch.cuda.nccl.version()), units="ms",
                    note="ag_delay = (done-ready) - ag_alone median; chain_overhead = chain - gemm_alone chain")
        with open(path, "w") as f:
            json.dump({"meta": meta, "rows": rows}, f, indent=1)
        print(f"[eb] wrote {path}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
