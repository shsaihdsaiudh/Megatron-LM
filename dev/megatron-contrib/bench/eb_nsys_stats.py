#!/usr/bin/env python3
"""Experiment B nsys stats: when do NCCL all-gather (AG) kernels start, and is a GEMM running then?

Input: an Nsight Systems report exported to sqlite:
    nsys export --type sqlite -f true -o OUT.sqlite IN.nsys-rep
Usage:
    python3 eb_nsys_stats.py OUT.sqlite [--select last|all] [--warmup 2] [--rows 0]

NVTX ranges are named <mode>_<config>_rep<N> (mode: independent, fsdp_like, gemm_alone,
ag_alone). By default the highest N of each label is used (--select last). With
--select all, every rep with N >= --warmup is used.

Per AG kernel whose launch call falls inside a range. AG streams are every non-GEMM stream
that carries AllGather kernels, so normal and high priority streams are both covered.
- ready and start delay (start - ready) come from e1_nsys_ready.analyze_ag, which resolves
  stream order and stream waits through CUDA_EVENT records (see bench/e1_nsys_ready.py).
- gemm_running: an nvjet kernel on the GEMM stream is active at AG start.
  gemm_prog: that kernel's progress (t - start) / (end - start) at AG start.
- dur: AG kernel duration. On this rank-0 trace it includes waiting for peer ranks.
- delay+dur-alone: delay + dur - median ag_alone duration of the matching config. This is the
  per-AG equivalent of the sweep's ag_delay, so it can be compared with eb_*.json.

Table 1 (per AG label): start delay statistics, share of AGs that start while a GEMM runs,
GEMM progress at those starts, and AG duration against ag_alone.
Table 2 (per GEMM label): total nvjet kernel time per rep against gemm_alone of the same
config. This is the GEMM slowdown while AGs overlap.
"""

import argparse
import os
import re
import sqlite3
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_nsys_ready as e1  # noqa: E402

AG_MARK = e1.AG_MARK
GEMM_MARK = e1.GEMM_MARK
DELAY_EPS_MS = e1.DELAY_EPS_MS
LAST_FRAC = e1.LAST_FRAC
RANGE_RE = re.compile(r"^((independent|fsdp_like|gemm_alone|ag_alone)_.+)_rep(\d+)$")
ALONE_RE = re.compile(r"_c(\d+)_(normal|high)$")
CFG_RE = re.compile(r"(T\d+_c\d+_\w+)$")


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def p90(xs):
    return e1.pct(xs, 0.9) if xs else float("nan")


def fmt(x, nd=3):
    return "-" if x != x else f"{x:.{nd}f}"


def select_ranges(names, path, select, warmup):
    c = sqlite3.connect(path)
    spans = []
    for s, e, text, tid in c.execute(
        "select start, end, text, textId from NVTX_EVENTS where end is not null order by start"
    ):
        lab = text if text is not None else names.get(tid)
        m = RANGE_RE.match(lab or "")
        if m:
            spans.append((m.group(1), int(m.group(3)), s, e))
    c.close()
    if select == "all":
        return [x for x in spans if x[1] >= warmup]
    best = {}
    for x in spans:
        if x[0] not in best or x[1] > best[x[0]][1]:
            best[x[0]] = x
    return list(best.values())


def gemm_at(gemms, t):
    """Latest-starting GEMM active at time t, or None."""
    best = None
    for g in gemms:
        if g.start <= t < g.end and (best is None or g.start > best.start):
            best = g
    return best


def alone_ref(base):
    if base.startswith("ag_alone_"):
        return base
    m = ALONE_RE.search(base)
    return f"ag_alone_c{m.group(1)}_{m.group(2)}" if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite")
    ap.add_argument("--select", choices=["last", "all"], default="last")
    ap.add_argument("--warmup", type=int, default=2, help="with --select all: lowest rep index kept")
    ap.add_argument("--rows", type=int, default=0, help="print the first N raw AG rows of each label")
    args = ap.parse_args()

    T = e1.load(args.sqlite)
    spans = select_ranges(T.names, args.sqlite, args.select, args.warmup)
    if not spans:
        raise SystemExit("no matching NVTX ranges")

    # GEMM streams are the streams with nvjet kernels (the compute stream). AG streams are the
    # other streams with AllGather kernels (one per priority). AG kernels on the GEMM stream
    # (here 8 from the max_over_ranks gathers, outside all ranges) are not benchmark AGs.
    all_gemms = [o for lst in T.by_stream.values() for o in lst if GEMM_MARK in o.name]
    gemm_streams = sorted({o.stream for o in all_gemms})
    ag_streams = sorted(
        sid
        for sid, lst in T.by_stream.items()
        if sid not in gemm_streams and any(AG_MARK in o.name for o in lst)
    )
    gemms = sorted(all_gemms, key=lambda o: o.start)
    stray = sum(
        1
        for sid in gemm_streams
        for o in T.by_stream[sid]
        if AG_MARK in o.name and o.launch is not None and any(lo <= o.launch <= hi for _, _, lo, hi in spans)
    )

    recs = []
    gemm_time = defaultdict(list)
    gemm_count = defaultdict(list)
    for base, rep, lo, hi in sorted(spans, key=lambda x: (x[0], x[1])):
        T.t0 = lo
        rows = []
        for sid in ag_streams:
            T.ag = sid
            rows += e1.analyze_ag(T, lo, hi)
        for r in rows:
            g = gemm_at(gemms, r["start"])
            r["base"], r["rep"] = base, rep
            r["gemm_prog"] = None if g is None else (r["start"] - g.start) / (g.end - g.start)
            recs.append(r)
        ks = [o for o in all_gemms if o.launch is not None and lo <= o.launch <= hi]
        gemm_time[base].append(sum(o.end - o.start for o in ks) / 1e6)
        gemm_count[base].append(len(ks))

    by = defaultdict(list)
    for r in recs:
        by[r["base"]].append(r)
    alone = {b: mean([r["dur"] for r in lst]) for b, lst in by.items() if b.startswith("ag_alone_")}

    print(f"file: {args.sqlite}")
    print(f"AG streams: {ag_streams}   GEMM streams: {gemm_streams}   select={args.select}   ranges used: {len(spans)}")
    print(f"AG kernels on GEMM streams launched inside ranges (expected 0): {stray}")
    print()
    print("Table 1: AG kernels (times in ms)")
    print()
    print(
        "| label | reps | AG/rep | start delay mean | start delay med | p90 | delayed share | "
        "GEMM running at start | GEMM prog at start med | prog>=0.6 (of GEMM-running) | AG dur mean | "
        "dur - alone mean | delay+dur-alone med |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for base in sorted(by):
        lst = by[base]
        reps = len({r["rep"] for r in lst})
        ready = [r["delay"] for r in lst if r["delay"] is not None]
        delayed = sum(d >= DELAY_EPS_MS for d in ready)
        gr = [r["gemm_prog"] for r in lst if r["gemm_prog"] is not None]
        last = sum(p >= LAST_FRAC for p in gr)
        durs = [r["dur"] for r in lst]
        ref = alone_ref(base)
        has_ref = ref in alone
        excess = mean(durs) - alone[ref] if has_ref else float("nan")
        # Per-AG equivalent of the sweep's ag_delay = (done - ready) - ag_alone median.
        eq = [r["delay"] + r["dur"] - alone[ref] for r in lst if r["delay"] is not None] if has_ref else []
        print(
            f"| {base} | {reps} | {len(lst) / reps:.1f} | {fmt(mean(ready))} | {fmt(med(ready))} | {fmt(p90(ready))} | "
            f"{fmt(delayed / len(ready) if ready else float('nan'), 2)} | "
            f"{fmt(len(gr) / len(lst) if lst else float('nan'), 2)} | {fmt(med(gr), 2)} | "
            f"{fmt(last / len(gr) if gr else float('nan'), 2)} | {fmt(mean(durs))} | {fmt(excess)} | {fmt(med(eq))} |"
        )

    print()
    print("Table 2: GEMM kernels (nvjet, per rep, summed kernel time)")
    print()
    print("| label | GEMM kernels/rep | GEMM kernel time med (ms) | vs gemm_alone, same config |")
    print("|---|---|---|---|")
    gref = {}
    for base in gemm_time:
        m = CFG_RE.search(base)
        if base.startswith("gemm_alone_") and m:
            gref[m.group(1)] = med(gemm_time[base])
    for base in sorted(gemm_time):
        if base.startswith("ag_alone_"):
            continue
        m = CFG_RE.search(base)
        ref = gref.get(m.group(1)) if m else None
        ratio = "-" if ref is None else f"{med(gemm_time[base]) / ref:.3f}"
        print(f"| {base} | {med(gemm_count[base]):.0f} | {fmt(med(gemm_time[base]), 2)} | {ratio} |")

    if args.rows:
        print()
        for base in sorted(by):
            print(f"[raw rows] {base}")
            print(f"{'rep':>3} {'t_start':>9} {'dur':>7} {'ready':>9} {'delay':>8} {'bind':>8} {'gemm':>8} {'prog':>6}")
            for r in by[base][: args.rows]:
                ready = "n/a" if r["ready_rel"] is None else f"{r['ready_rel']:9.3f}"
                delay = "n/a" if r["delay"] is None else f"{r['delay']:8.3f}"
                prog = "-" if r["gemm_prog"] is None else f"{r['gemm_prog']:.2f}"
                gemm = "-"
                g = gemm_at(gemms, r["start"])
                if g is not None:
                    gemm = e1.label(g.name)
                print(
                    f"{r['rep']:>3} {r['t_rel']:9.3f} {r['dur']:7.3f} {ready:>9} {delay:>8} "
                    f"{r['binding']:>8} {gemm:>8} {prog:>6}"
                )


if __name__ == "__main__":
    main()
