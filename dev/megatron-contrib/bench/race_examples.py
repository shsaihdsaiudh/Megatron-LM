#!/usr/bin/env python3
"""Print concrete AG-I race examples (timestamps) from one rank's nsys sqlite, for viewing in the GUI.

Compute stream 7 and inner AG-I on stream 39 (ncclDevKernel_AllGather_RING_LL, gridX 24), as detected in
results/e2/e2skew-2node-20261116/RACE2.md. Phases: mb1..3 forward of the first profiled step, ops joined
to the NVTX range by CPU launch time (correlationId).

For each AG-I start A in the phase:
  idle  - no compute-stream op spans A. X = last compute op ending before A, R = X.end.
  busy  - A falls inside a compute op (GEMM). X = the op two before it (X, memset N, GEMM G).
Times: absolute in seconds from session start (as in the GUI timeline), gaps in us from R.

Usage: python3 bench/race_examples.py SQLITE [--n 3] [--ag-stream 39]
"""

import argparse
import sqlite3

COMPUTE = 7


def load(path):
    c = sqlite3.connect(path)
    names = dict(c.execute("select id, value from StringIds"))
    tid = c.execute("select globalTid from NVTX_EVENTS where text='mb1_fwd'").fetchone()[0]
    ranges = list(
        c.execute(
            "select text, start, end from NVTX_EVENTS where text in ('mb1_fwd','mb2_fwd','mb3_fwd') "
            "and globalTid=? order by start limit 3",
            (tid,),
        )
    )
    rt = dict(c.execute("select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME"))
    ops = []
    for s, e, st, corr, nm, gx in c.execute(
        "select start, end, streamId, correlationId, shortName, gridX from CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        ops.append((s, e, st, corr, names.get(nm, str(nm)), gx))
    for s, e, st, corr in c.execute(
        "select start, end, streamId, correlationId from CUPTI_ACTIVITY_KIND_MEMSET"
    ):
        ops.append((s, e, st, corr, "memset", 0))
    ops.sort()
    return ranges, rt, ops


def is_gemm(name):
    return "nvjet" in name or "gemm" in name.lower() or "128x256" in name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--ag-stream", type=int, default=39, help="inner AG-I stream (v2: 39, v1: 43)")
    a = ap.parse_args()
    ranges, rt, ops = load(a.sqlite)
    shown = {"idle": 0, "busy": 0}
    counts = {"idle": 0, "busy": 0}
    for text, rs, re_ in ranges:
        mem = [o for o in ops if o[3] in rt and rs <= rt[o[3]] <= re_]
        comp = [o for o in mem if o[2] == COMPUTE]
        ag = [o for o in mem if o[2] == a.ag_stream and "AllGather" in o[4] and o[5] == 24]
        for A in ag:
            a0 = A[0]
            span = [i for i, o in enumerate(comp) if o[0] <= a0 < o[1]]
            if span:
                kind = "busy"
                i = span[0]
                if i < 2:
                    continue
                X, N, G = comp[i - 2], comp[i - 1], comp[i]
            else:
                kind = "idle"
                prev = [i for i, o in enumerate(comp) if o[1] <= a0]
                if not prev:
                    continue
                i = prev[-1]
                X = comp[i]
                rest = comp[i + 1 :]
                N = rest[0] if rest else None
                G = next((o for o in rest if is_gemm(o[4])), None)
            counts[kind] += 1
            if shown[kind] >= a.n or N is None or G is None:
                continue
            shown[kind] += 1
            R = X[1]
            us = lambda t: (t - R) / 1e3
            print(f"[{kind}] {text}  R = X.end = {R / 1e9:.6f} s")
            print(f"    X  {X[4][:60]}  {X[0] / 1e9:.6f} - {X[1] / 1e9:.6f} s")
            print(f"    N  {N[4][:60]}  start R+{us(N[0]):.1f} us")
            print(f"    G  {G[4][:60]}  start R+{us(G[0]):.1f} us, dur {(G[1] - G[0]) / 1e6:.3f} ms")
            print(f"    A  AG-I k+1 start R+{us(a0):.1f} us, dur {(A[1] - A[0]) / 1e6:.3f} ms")
    print(f"counts in mb1..3 fwd of first step: AG-I start while compute idle {counts['idle']}, "
          f"during a compute op {counts['busy']}")


if __name__ == "__main__":
    main()
