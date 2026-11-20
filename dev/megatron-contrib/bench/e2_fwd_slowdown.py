#!/usr/bin/env python3
"""E2 forward slowdown: compare the flag-OFF and flag-ON (P1) Nsight Systems traces, forward of mb1..3.

Usage:
  python3 e2_fwd_slowdown.py OFF.sqlite ON.sqlite [--also OFF0.sqlite ON0.sqlite] [--head FILE.md] [--out FILE.md]

Inputs are sqlite exports of nsys reports (`nsys export --type sqlite`). The script prints Markdown.
--also runs the same report on a second pair (for example rank 0) after the first.

Definitions
- Instance: one NVTX range occurrence on the CPU. mb{i}_fwd and mb{i}_bwd each occur twice per file
  (two profiled steps). Per-microbatch numbers are totals over the instances, divided by the number of
  instances.
- Membership: an operation (kernel, memcpy, memset) or a CUDA runtime call belongs to an instance if its
  CPU launch time (CUPTI runtime start, joined to the operation by correlationId) falls inside the
  instance and on the same CPU thread (globalTid).
- Inner all-gather (AG-I): NCCL AllGather kernel with gridX 24. The 8-rank communicator runs on stream 39
  in both files. Outer all-gather (O) has gridX 4 and occurs only in mb0 forward.
- Compute stream: the stream with the most nvjet (GEMM) kernels.
- Times are in ms. "GPU" times are CUPTI start/end. "Launch" is the CUPTI runtime call start.
- Layer k (Q5): CPU range "MFSDP blocks.k forward" inside the first mb1_fwd instance. Operations are
  listed in GPU start order, offsets from the first operation of that layer.
"""

import argparse
import bisect
import re
import sqlite3
import statistics
from collections import Counter, defaultdict

NS = 1e6  # ns per ms
PREFIX = "nvjet_sm90_tst_"
INNER_GRID = 24
INNER_STREAM = 39
FWD = ["mb1_fwd", "mb2_fwd", "mb3_fwd"]
BWD = ["mb1_bwd", "mb2_bwd", "mb3_bwd"]
LAYERS = [15, 16, 17]


class Op:
    __slots__ = ("table", "stream", "start", "end", "corr", "short", "full", "grid", "kind", "bytes",
                 "launch", "tid", "cfg")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def pctl(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def fm(x, nd=3):
    return "n/a" if x != x else f"{x:.{nd}f}"


def base(name):
    return re.sub(r"_v\d+$", "", name)


def is_ag(o):
    return o.table == "KERNEL" and o.short.startswith("ncclDevKernel_AllGather")


def is_inner_ag(o):
    return is_ag(o) and o.grid == INNER_GRID


def is_gemm(o):
    return o.table == "KERNEL" and "nvjet" in o.short


def alg_of(o):
    return o.short.split("AllGather_")[-1] if "AllGather_" in o.short else o.short


def label(o):
    if o.table == "MEMCPY":
        return f"memcpy(kind {o.kind})"
    if o.table == "MEMSET":
        return "memset"
    if is_inner_ag(o):
        return f"AG-I {alg_of(o)}"
    if is_ag(o):
        return f"AG grid{o.grid} {alg_of(o)}"
    if is_gemm(o):
        return "GEMM " + o.short[len(PREFIX):][:44]
    return o.short[:44]


def category(name):
    b = base(name)
    if b in ("cudaLaunchKernel", "cuLaunchKernelEx"):
        return "kernel launch (cudaLaunchKernel / cuLaunchKernelEx)"
    if b == "cudaStreamWaitEvent":
        return "cudaStreamWaitEvent"
    if b.startswith("cudaEventRecord"):
        return "cudaEventRecord (incl. WithFlags)"
    if b.startswith("cudaMalloc"):
        return "cudaMalloc*"
    if b.startswith("cudaFree"):
        return "cudaFree*"
    if b == "cudaMemcpyAsync":
        return "cudaMemcpyAsync"
    if b == "cudaMemsetAsync":
        return "cudaMemsetAsync"
    if "Synchronize" in b:
        return "*Synchronize*"
    return "other"


CATS = [
    "kernel launch (cudaLaunchKernel / cuLaunchKernelEx)",
    "cudaStreamWaitEvent",
    "cudaEventRecord (incl. WithFlags)",
    "cudaMalloc*",
    "cudaFree*",
    "cudaMemcpyAsync",
    "cudaMemsetAsync",
    "*Synchronize*",
    "other",
]


def load(path):
    c = sqlite3.connect(path)
    S = dict(c.execute("select id, value from StringIds"))

    nvtx = defaultdict(list)
    for s, e, t, tid, g in c.execute(
        "select start, end, text, textId, globalTid from NVTX_EVENTS where end is not null"
    ):
        lab = t if t is not None else S.get(tid)
        if lab:
            nvtx[lab].append((s, e, g))
    for v in nvtx.values():
        v.sort()

    rt = {}
    rt_rows = []
    for corr, s, nid, g in c.execute("select correlationId, start, nameId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"):
        name = S.get(nid, "")
        rt[corr] = (s, g, name)
        rt_rows.append((s, g, name, corr))
    rt_rows.sort()
    rt_starts = [r[0] for r in rt_rows]

    ops = []
    for sid, s, e, corr, sn, dn, gx in c.execute(
        "select streamId, start, end, correlationId, shortName, demangledName, gridX from CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        o = Op()
        o.table, o.stream, o.start, o.end, o.corr = "KERNEL", sid, s, e, corr
        o.short, o.full, o.grid, o.kind, o.bytes = S.get(sn, ""), S.get(dn, ""), gx, None, None
        ops.append(o)
    for s, e, sid, corr, b, kind in c.execute(
        "select start, end, streamId, correlationId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"
    ):
        o = Op()
        o.table, o.stream, o.start, o.end, o.corr = "MEMCPY", sid, s, e, corr
        o.short, o.full, o.grid, o.kind, o.bytes = "memcpy", "", None, kind, b
        ops.append(o)
    for s, e, sid, corr, b, val in c.execute(
        "select start, end, streamId, correlationId, bytes, value from CUPTI_ACTIVITY_KIND_MEMSET"
    ):
        o = Op()
        o.table, o.stream, o.start, o.end, o.corr = "MEMSET", sid, s, e, corr
        o.short, o.full, o.grid, o.kind, o.bytes = "memset", "", None, val, b
        ops.append(o)

    unmatched = 0
    kcfg = {}
    for corr, gy, gz, bx, regs, sm, dm in c.execute(
        "select correlationId, gridY, gridZ, blockX, registersPerThread, staticSharedMemory, dynamicSharedMemory "
        "from CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        kcfg[corr] = (gy, gz, bx, regs, sm + dm)
    gpu = list(c.execute("select smCount, maxShmemPerSm, maxRegistersPerSm from TARGET_INFO_GPU"))[0]
    for o in ops:
        o.cfg = kcfg.get(o.corr) if o.table == "KERNEL" else None
    unmatched = 0
    for o in ops:
        r = rt.get(o.corr)
        if r is None:
            unmatched += 1
            o.launch, o.tid = None, None
        else:
            o.launch, o.tid = r[0], r[1]
    ops.sort(key=lambda o: o.start)

    sync = defaultdict(list)
    for corr, sid, st in c.execute("select correlationId, streamId, syncType from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"):
        sync[corr].append((sid, st))
    rec_rows = list(c.execute("select correlationId, streamId, eventSyncId from CUPTI_ACTIVITY_KIND_CUDA_EVENT"))
    wait_rows = list(
        c.execute(
            "select correlationId, streamId, eventSyncId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where syncType = 2"
        )
    )

    compute_n = Counter(o.stream for o in ops if is_gemm(o))
    compute = compute_n.most_common(1)[0][0]
    by_stream = defaultdict(list)
    for o in ops:
        by_stream[o.stream].append(o)
    cstarts = [o.start for o in by_stream[compute]]
    s39starts = [o.start for o in by_stream[INNER_STREAM]]

    launched = sorted([o for o in ops if o.launch is not None], key=lambda o: o.launch)
    D = {
        "name": path.split("/")[-1],
        "nvtx": nvtx,
        "rt_rows": rt_rows,
        "rt_starts": rt_starts,
        "rt": rt,
        "ops": ops,
        "unmatched": unmatched,
        "sync": sync,
        "compute": compute,
        "by_stream": by_stream,
        "cstarts": cstarts,
        "s39starts": s39starts,
        "launched": launched,
        "launch_t": [o.launch for o in launched],
        "rec_rows": rec_rows,
        "wait_rows": wait_rows,
        "gpu": gpu,
    }
    D["ready"] = gate_table(D)
    return D


def gate_table(D):
    """For each op on the inner stream: the time its stream frontier was complete before it ran, and the
    op that completed last (the binding dependency). Streams run in order, so the frontier of a stream is
    the latest completion of everything enqueued on it. Items are processed in correlationId (enqueue) order.
    A stream wait sets the frontier of the waiting stream to the event completion value, and an event
    record stores the frontier of its stream. Wait-to-record matching uses eventSyncId."""
    items = [(o.corr, 0, o) for o in D["ops"]]
    items += [(corr, 1, (sid, esid)) for corr, sid, esid in D["rec_rows"]]
    items += [(corr, 2, (sid, esid)) for corr, sid, esid in D["wait_rows"]]
    items.sort(key=lambda x: (x[0], x[1]))
    done = {}
    rec = {}
    ready = {}
    for corr, kind, p in items:
        if kind == 0:
            o = p
            if o.stream == INNER_STREAM:
                ready[o.corr] = done.get(o.stream, (0, None))
            cur = done.get(o.stream, (0, None))
            if o.end > cur[0]:
                done[o.stream] = (o.end, o)
        elif kind == 1:
            sid, esid = p
            rec[esid] = done.get(sid, (0, None))
        else:
            sid, esid = p
            v = rec.get(esid)
            if v is not None and v[0] > done.get(sid, (0, None))[0]:
                done[sid] = v
    return ready


def origin_class(o, compute):
    """Class of the binding dependency of an inner all-gather."""
    if o is None:
        return "none"
    if o.stream == INNER_STREAM:
        return "previous AG on stream 39" if is_ag(o) else "other op on stream 39"
    if o.stream == compute:
        if is_gemm(o):
            return "GEMM coopB on compute stream" if "coopB" in o.short else "GEMM plain on compute stream"
        return "other op on compute stream"
    return "op on another stream"


def instances(D, label_name):
    return D["nvtx"].get(label_name, [])


def window(D, lo, hi, tid):
    a = bisect.bisect_left(D["launch_t"], lo)
    b = bisect.bisect_right(D["launch_t"], hi)
    if tid is None:
        return D["launched"][a:b]
    return [o for o in D["launched"][a:b] if o.tid == tid]


def rt_window(D, lo, hi, tid):
    a = bisect.bisect_left(D["rt_starts"], lo)
    b = bisect.bisect_right(D["rt_starts"], hi)
    return [r for r in D["rt_rows"][a:b] if r[1] == tid]


def ops_for(D, insts):
    res = []
    for n, (s, e, tid) in enumerate(insts):
        for o in window(D, s, e, tid):
            res.append((n, o))
    return res


def insts_for(D, lab):
    """Backward kernels launch on a different CPU thread than the mb*_bwd NVTX ranges, so backward
    instances match by time window only (tid None)."""
    insts = instances(D, lab)
    if lab.endswith("_bwd"):
        return [(s, e, None) for s, e, _ in insts]
    return insts


def all_insts(D, labels):
    out = []
    for lab in labels:
        out += insts_for(D, lab)
    return out


def validation(D):
    lines = []
    for lab in ["mb0_fwd"] + FWD + ["mb0_bwd"] + BWD:
        insts = instances(D, lab)
        n_thread = n_any = 0
        for s, e, tid in insts:
            n_thread += len(window(D, s, e, tid))
            a = bisect.bisect_left(D["launch_t"], s)
            b = bisect.bisect_right(D["launch_t"], e)
            n_any += b - a
        lines.append((lab, len(insts), n_thread, n_any))
    return lines


# Q1
def q1(D, phases):
    counts = Counter()
    per = {}
    for phase, labels in phases.items():
        insts = all_insts(D, labels)
        for n, o in ops_for(D, insts):
            if is_inner_ag(o):
                key = (phase, o.short, base(o.full).split("(")[0], o.grid, o.stream)
                counts[key] += 1
        per[phase] = len(insts)
    total = Counter()
    for (phase, *_), v in counts.items():
        total[phase] += v
    return {k: v / per[k[0]] for k, v in counts.items()}, {k: total[k] / per[k] for k in per}


def q1_names(D, phases):
    names = set()
    for labels in phases.values():
        for n, o in ops_for(D, all_insts(D, labels)):
            if is_inner_ag(o):
                names.add(o.short)
    return names


# Q2
def q2_counts(D, labels):
    insts = all_insts(D, labels)
    c = Counter()
    for s, e, tid in insts:
        for r in rt_window(D, s, e, tid):
            c[base(r[2])] += 1
    return {k: v / len(insts) for k, v in c.items()}, len(insts)


# Q3
def q3_records(D, insts):
    comp = D["by_stream"][D["compute"]]
    cs = D["cstarts"]
    s39 = D["by_stream"][INNER_STREAM]
    s39s = D["s39starts"]
    recs = []
    for n, o in ops_for(D, insts):
        if not is_inner_ag(o):
            continue
        r = {"delay": (o.start - o.launch) / NS, "dur": (o.end - o.start) / NS, "inst": n}
        i = bisect.bisect_right(cs, o.start) - 1
        r["run_cls"] = "idle"
        r["frac"] = None
        if i >= 0 and comp[i].end > o.start:
            c = comp[i]
            if is_gemm(c):
                r["run_cls"] = "GEMM"
                r["frac"] = (o.start - c.start) / (c.end - c.start)
            else:
                r["run_cls"] = "other " + label(c)
        j = bisect.bisect_left(s39s, o.start) - 1
        prev = s39[j] if j >= 0 else None
        r["prev"] = label(prev) if prev is not None else "none"
        r["gap"] = (o.start - prev.end) / NS if prev is not None else float("nan")
        origin = D["ready"].get(o.corr, (0, None))[1]
        r["origin"] = origin_class(origin, D["compute"])
        r["lag"] = (o.start - origin.end) / NS if origin is not None else float("nan")
        dur_ns = o.end - o.start
        ov = {"coopB": 0.0, "other GEMM": 0.0, "non-GEMM": 0.0}
        k = max(0, bisect.bisect_right(cs, o.start) - 1)
        while k < len(comp) and comp[k].start < o.end:
            c_ = comp[k]
            x = min(c_.end, o.end) - max(c_.start, o.start)
            if x > 0:
                if is_gemm(c_) and "coopB" in c_.short:
                    ov["coopB"] += x
                elif is_gemm(c_):
                    ov["other GEMM"] += x
                else:
                    ov["non-GEMM"] += x
            k += 1
        if ov["coopB"] >= 0.5 * dur_ns:
            r["ov_cls"] = "coopB GEMM covers >=50% of the AG"
        elif ov["other GEMM"] >= 0.5 * dur_ns:
            r["ov_cls"] = "other GEMM covers >=50% of the AG"
        else:
            r["ov_cls"] = "no GEMM for >=50% of the AG"
        recs.append(r)
    return recs


def q3_summary(recs):
    gemm = [r for r in recs if r["run_cls"] == "GEMM"]
    idle = [r for r in recs if r["run_cls"] == "idle"]
    other = [r for r in recs if r["run_cls"].startswith("other")]
    n = len(recs)
    fr = [r["frac"] for r in gemm]
    bins = Counter()
    for f in fr:
        bins["0-25%" if f < 0.25 else "25-50%" if f < 0.5 else "50-75%" if f < 0.75 else "75-100%"] += 1
    prev_top = Counter(r["prev"] for r in recs).most_common(3)
    origin = defaultdict(list)
    for r in recs:
        origin[r["origin"]].append(r["lag"])
    ovc = defaultdict(list)
    for r in recs:
        ovc[r["ov_cls"]].append(r["dur"])
    return {
        "ov": {k: (len(v), med(v)) for k, v in ovc.items()},
        "origin": {k: (len(v), med(v), sum(1 for x in v if x < 0.1)) for k, v in origin.items()},
        "n": n,
        "delay_med": med([r["delay"] for r in recs]),
        "delay_p90": pctl([r["delay"] for r in recs], 0.9),
        "dur_med": med([r["dur"] for r in recs]),
        "gemm_share": len(gemm) / n if n else float("nan"),
        "idle_share": len(idle) / n if n else float("nan"),
        "other_share": len(other) / n if n else float("nan"),
        "frac_med": med(fr),
        "bins": bins,
        "dur_gemm": med([r["dur"] for r in gemm]),
        "dur_idle": med([r["dur"] for r in idle]),
        "gap_med": med([r["gap"] for r in recs]),
        "prev_top": prev_top,
        "other_top": Counter(r["run_cls"] for r in other).most_common(3),
    }


# Q4
def q4(D, insts):
    comp = D["compute"]
    kc, kt = Counter(), defaultdict(float)
    mc, mb = Counter(), defaultdict(float)
    for n, o in ops_for(D, insts):
        if o.table == "KERNEL" and o.stream not in (comp, INNER_STREAM):
            key = (o.stream, label(o))
            kc[key] += 1
            kt[key] += (o.end - o.start) / NS
        if o.table in ("MEMCPY", "MEMSET"):
            key = (o.table, o.stream, label(o))
            mc[key] += 1
            mb[key] += o.bytes or 0
    k = len(insts)
    return (
        {key: (v / k, kt[key] / k) for key, v in kc.items()},
        {key: (v / k, mb[key] / k) for key, v in mc.items()},
    )


# Q5
def layer_ops(D, mb_inst, layer):
    s, e, tid = mb_inst
    lab = f"MFSDP blocks.{layer} forward"
    blocks = [b for b in D["nvtx"].get(lab, []) if s <= b[0] and b[1] <= e and b[2] == tid]
    if not blocks:
        return None
    bs, be, btid = blocks[0]
    ops = sorted(window(D, bs, be, btid), key=lambda o: o.start)
    api = Counter(category(r[2]) for r in rt_window(D, bs, be, btid))
    corrs = {r[3] for r in rt_window(D, bs, be, btid)}
    waits = Counter()
    for corr in corrs:
        for sid, st in D["sync"].get(corr, []):
            if st == 2:
                waits[sid] += 1
    return ops, api, waits


def gpu_cat_rows(ops):
    if not ops:
        return []
    t0 = ops[0].start
    return [(o.stream, label(o), (o.start - t0) / NS, (o.end - o.start) / NS) for o in ops]


def report_one(D, title):
    p = print
    p(f"## {title}")
    p("")
    p(f"- File: `{D['name']}`. Compute stream: {D['compute']}. Inner all-gather: AllGather with gridX {INNER_GRID} on stream {INNER_STREAM}.")
    p(f"- Operations with no matching CUDA runtime row: {D['unmatched']}.")
    p("- Instances (thread-matched ops in the NVTX range / ops with a launch time in the range, all threads): "
      + "; ".join(f"{lab} n={n}, {m}/{a}" for lab, n, m, a in validation(D)) + ".")
    p("")
    return


def section_q1(off, on):
    phases = {"fwd mb1..3": FWD, "bwd mb1..3": BWD}
    off_c, off_tot = q1(off, phases)
    on_c, on_tot = q1(on, phases)
    keys = sorted(set(off_c) | set(on_c), key=lambda k: (k[0], k[1], k[4]))
    rows = []
    for k in keys:
        rows.append((k[0], k[1], k[2], k[3], k[4], f"{off_c.get(k, 0):.2f}", f"{on_c.get(k, 0):.2f}"))
    print("### Q1. Full NCCL names of inner all-gathers (per microbatch, mean over instances)\n")
    print("| phase | shortName | demangled name (before template args) | gridX | stream | OFF per microbatch | ON per microbatch |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print("")
    print(f"Inner all-gathers per microbatch: OFF fwd {off_tot['fwd mb1..3']:.2f}, bwd {off_tot['bwd mb1..3']:.2f}; "
          f"ON fwd {on_tot['fwd mb1..3']:.2f}, bwd {on_tot['bwd mb1..3']:.2f}.")
    names = q1_names(off, phases) | q1_names(on, phases)
    print(f"Distinct inner all-gather shortNames across fwd and bwd, both files: {sorted(names)}.")
    print("")


def section_q2(off, on):
    o0, n0 = q2_counts(off, ["mb0_fwd"])
    o13, n13 = q2_counts(off, FWD)
    e0, _ = q2_counts(on, ["mb0_fwd"])
    e13, _ = q2_counts(on, FWD)
    print("### Q2. CUDA runtime API calls per forward microbatch (CPU thread, inside the NVTX range)\n")
    print(f"Averages per microbatch: mb0 over {n0} instances, mb1..3 over {n13} instances per file.\n")
    print("| category | OFF mb0 | ON mb0 | OFF mb1..3 | ON mb1..3 | Δ mb1..3 (ON - OFF) |")
    print("|---|---|---|---|---|---|")
    for cat in CATS:
        def tot(d):
            return sum(v for k, v in d.items() if category(k) == cat)
        a0, b0, a1, b1 = tot(o0), tot(e0), tot(o13), tot(e13)
        print(f"| {cat} | {a0:.1f} | {b0:.1f} | {a1:.1f} | {b1:.1f} | {b1 - a1:+.1f} |")
    print("")
    names = set(o13) | set(e13)
    top = sorted(names, key=lambda k: -(o13.get(k, 0) + e13.get(k, 0)))[:30]
    print("Per runtime-call name (top 30 by mb1..3 count):\n")
    print("| runtime call | OFF mb0 | ON mb0 | OFF mb1..3 | ON mb1..3 | Δ mb1..3 |")
    print("|---|---|---|---|---|---|")
    for k in top:
        a0, b0 = o0.get(k, 0), e0.get(k, 0)
        a1, b1 = o13.get(k, 0), e13.get(k, 0)
        print(f"| {k} | {a0:.1f} | {b0:.1f} | {a1:.1f} | {b1:.1f} | {b1 - a1:+.1f} |")
    print("")


def section_q3(off, on):
    recs_off = q3_records(off, all_insts(off, FWD))
    recs_on = q3_records(on, all_insts(on, FWD))
    so, sn = q3_summary(recs_off), q3_summary(recs_on)
    print("### Q3. Inner all-gathers in mb1..3 forward: launch, queue delay, and the compute stream\n")
    print("| metric | OFF | ON |")
    print("|---|---|---|")
    rows = [
        ("inner AG kernels analysed (both steps, mb1..3)", so["n"], sn["n"]),
        ("launch to GPU start (queue delay), median ms", fm(so["delay_med"]), fm(sn["delay_med"])),
        ("launch to GPU start, p90 ms", fm(so["delay_p90"]), fm(sn["delay_p90"])),
        ("AG kernel duration, median ms", fm(so["dur_med"]), fm(sn["dur_med"])),
        ("AG starts while a GEMM runs on the compute stream", f"{so['gemm_share']:.3f}", f"{sn['gemm_share']:.3f}"),
        ("AG starts while the compute stream is idle", f"{so['idle_share']:.3f}", f"{sn['idle_share']:.3f}"),
        ("AG starts while another compute-stream op runs (memset etc.)", f"{so['other_share']:.3f}", f"{sn['other_share']:.3f}"),
        ("GEMM elapsed fraction at AG start, median", fm(so["frac_med"]), fm(sn["frac_med"])),
        ("AG duration median, when a GEMM runs at start, ms", fm(so["dur_gemm"]), fm(sn["dur_gemm"])),
        ("AG duration median, when compute stream idle at start, ms", fm(so["dur_idle"]), fm(sn["dur_idle"])),
        ("gap from previous stream-39 op end to AG start, median ms", fm(so["gap_med"]), fm(sn["gap_med"])),
    ]
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} |")
    print("")
    print("GEMM elapsed fraction bins at AG start (count):\n")
    print("| bin | OFF | ON |")
    print("|---|---|---|")
    for b in ["0-25%", "25-50%", "50-75%", "75-100%"]:
        print(f"| {b} | {so['bins'][b]} | {sn['bins'][b]} |")
    print("")
    print("Op immediately before the AG on stream 39 (top 3, count):\n")
    print(f"- OFF: {so['prev_top']}")
    print(f"- ON: {sn['prev_top']}")
    print("")
    print("Binding dependency of each AG (the op whose completion set the stream-39 frontier before the AG). "
          "Lag = AG GPU start minus that op's end. 'Starts at once' counts lag < 0.1 ms; the rest wait.\n")
    print("| binding dependency | OFF count | OFF starts at once | OFF lag median ms | ON count | ON starts at once | ON lag median ms |")
    print("|---|---|---|---|---|---|---|")
    keys = sorted(set(so["origin"]) | set(sn["origin"]))
    for k in keys:
        a = so["origin"].get(k, (0, float("nan"), 0))
        b = sn["origin"].get(k, (0, float("nan"), 0))
        print(f"| {k} | {a[0]} | {a[2]} | {fm(a[1])} | {b[0]} | {b[2]} | {fm(b[1])} |")
    print("")
    print("AG duration by the GEMM that runs during it (share of the AG duration overlapped by the GEMM class):\n")
    print("| overlap class | OFF count | OFF median duration ms | ON count | ON median duration ms |")
    print("|---|---|---|---|---|")
    keys = sorted(set(so["ov"]) | set(sn["ov"]))
    for k in keys:
        a = so["ov"].get(k, (0, float("nan")))
        b = sn["ov"].get(k, (0, float("nan")))
        print(f"| {k} | {a[0]} | {fm(a[1])} | {b[0]} | {fm(b[1])} |")
    print("")
    print(f"Compute-stream op classes at AG start that are not GEMM or idle (top 3): OFF {so['other_top']}; ON {sn['other_top']}.")
    print("")


def gemm_coverage(D, insts):
    """Inner all-gather intervals (all steps, all ranks' ops in this file) merged, then per GEMM in insts the share
    of its duration covered by them."""
    ivs = sorted((o.start, o.end) for o in D["ops"] if is_inner_ag(o))
    merged = []
    for s, e in ivs:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    ms = [m[0] for m in merged]
    me = [m[1] for m in merged]
    out = []
    for n, o in ops_for(D, insts):
        if not is_gemm(o):
            continue
        dur = o.end - o.start
        ov = 0
        i = bisect.bisect_right(me, o.start)
        while i < len(ms) and ms[i] < o.end:
            ov += min(o.end, me[i]) - max(o.start, ms[i])
            i += 1
        cov = ov / dur if dur else 0.0
        bin_ = "none" if cov == 0 else ("partial (<90%)" if cov < 0.9 else "full (>=90%)")
        out.append((o.short[len(PREFIX):], bin_, dur / NS))
    return out


def section_q3b(off, on):
    insts_off = all_insts(off, FWD)
    insts_on = all_insts(on, FWD)
    co = gemm_coverage(off, insts_off)
    cn = gemm_coverage(on, insts_on)
    mb_off, mb_on = len(insts_off), len(insts_on)
    print("### Q3b. GEMM duration by tile and by how much inner all-gather covers it (mb1..3 forward)\n")
    print("Counts are per microbatch. Coverage is the share of the GEMM's duration overlapped by any inner all-gather.\n")
    print("| GEMM tile (name after nvjet_sm90_tst_) | coverage | OFF count | OFF median ms | ON count | ON median ms |")
    print("|---|---|---|---|---|---|")
    keys = sorted({(t, b) for t, b, _ in co} | {(t, b) for t, b, _ in cn})
    for t, b in keys:
        a = [d for tt, bb, d in co if tt == t and bb == b]
        c = [d for tt, bb, d in cn if tt == t and bb == b]
        print(f"| {t} | {b} | {len(a) / mb_off:.1f} | {fm(med(a))} | {len(c) / mb_on:.1f} | {fm(med(c))} |")
    print("")

    def gemm_time(rows, mb):
        out = defaultdict(float)
        for t, _, d in rows:
            out[t] += d / mb
        return out

    go, gn = gemm_time(co, mb_off), gemm_time(cn, mb_on)
    print("GEMM time per microbatch by tile (ms, sum of kernel durations):\n")
    print("| GEMM tile | OFF | ON | ON - OFF |")
    print("|---|---|---|---|")
    for t in sorted(set(go) | set(gn)):
        print(f"| {t} | {go.get(t, 0):.2f} | {gn.get(t, 0):.2f} | {gn.get(t, 0) - go.get(t, 0):+.2f} |")
    print(f"| all GEMMs | {sum(go.values()):.2f} | {sum(gn.values()):.2f} | {sum(gn.values()) - sum(go.values()):+.2f} |")
    print("")


def section_q6(off, on):
    print("### Q6. Launch configuration and SM limits (context for the overlap mechanism)\n")
    smc, shm, regs_sm = off["gpu"]
    print(f"GPU (TARGET_INFO_GPU): {smc} SMs, {shm} B shared memory per SM, {regs_sm} registers per SM.\n")
    print("| file | kernel | CTAs (gridX×Y×Z) | threads per CTA | registers per CTA | shared memory per CTA (B) |")
    print("|---|---|---|---|---|---|")
    picks = {}
    for name, D in (("OFF", off), ("ON", on)):
        found = {}
        for o in D["ops"]:
            if o.table != "KERNEL" or o.cfg is None:
                continue
            if "coopB" in o.short and "nvjet" in o.short and "coopB GEMM" not in found:
                found["coopB GEMM"] = o
            elif "TNN" in o.short and "nvjet" in o.short and "TNN GEMM" not in found:
                found["TNN GEMM"] = o
            elif is_inner_ag(o) and "inner AG" not in found:
                found["inner AG"] = o
        picks[name] = found
        for kind in ("coopB GEMM", "TNN GEMM", "inner AG"):
            o = found.get(kind)
            if o is None:
                continue
            gy, gz, bx, regs, smem = o.cfg
            print(f"| {name} | {kind} | {o.grid}×{gy}×{gz} = {o.grid * gy * gz} | {bx} | {regs * bx} | {smem} |")
    print("")
    f = picks["OFF"]
    if "coopB GEMM" in f and "inner AG" in f:
        g, a = f["coopB GEMM"].cfg, f["inner AG"].cfg
        smem_sum = g[4] + a[4]
        regs_sum = g[3] * g[2] + a[3] * a[2]
        print(f"One coopB GEMM CTA plus one inner AG CTA need {smem_sum} B shared memory (SM limit {shm}; "
              f"exceeds: {smem_sum > shm}) and {regs_sum} registers (SM limit {regs_sm}; exceeds: {regs_sum > regs_sm}). "
              f"The coopB GEMM launches {f['coopB GEMM'].grid * g[0] * g[1]} CTAs, which is {'at least as many as' if f['coopB GEMM'].grid * g[0] * g[1] >= smc else 'fewer than'} the {smc} SMs.")
    print("")




def section_q4(off, on):
    ok, ob = q4(off, all_insts(off, FWD))
    nk, nb = q4(on, all_insts(on, FWD))
    print("### Q4. Activity outside the compute stream and stream 39 in mb1..3 forward (per microbatch)\n")
    print("Kernels on other streams:\n")
    print("| stream | kernel | OFF count | OFF total ms | ON count | ON total ms |")
    print("|---|---|---|---|---|---|")
    keys = sorted(set(ok) | set(nk), key=lambda k: (k[0], k[1]))
    if not keys:
        print("| none | none | 0 | 0 | 0 | 0 |")
    for k in keys:
        a = ok.get(k, (0, 0))
        b = nk.get(k, (0, 0))
        print(f"| {k[0]} | {k[1]} | {a[0]:.2f} | {a[1]:.3f} | {b[0]:.2f} | {b[1]:.3f} |")
    print("")
    print("memcpy and memset launched in mb1..3 forward (all streams):\n")
    print("| table | stream | op | OFF count | OFF bytes (MB) | ON count | ON bytes (MB) |")
    print("|---|---|---|---|---|---|---|")
    keys = sorted(set(ob) | set(nb), key=lambda k: (k[0], k[1], k[2]))
    if not keys:
        print("| none | | | 0 | 0 | 0 | 0 |")
    for k in keys:
        a = ob.get(k, (0, 0))
        b = nb.get(k, (0, 0))
        print(f"| {k[0]} | {k[1]} | {k[2]} | {a[0]:.2f} | {a[1] / 1e6:.2f} | {b[0]:.2f} | {b[1] / 1e6:.2f} |")
    print("")


def section_q5(off, on):
    print("### Q5. Per-layer timeline, mb1 forward, first instance (layers 15, 16, 17)\n")
    print("Offsets are GPU start minus the first op of the layer, in ms. Rows are aligned by position.\n")
    mo = instances(off, "mb1_fwd")[0]
    mn = instances(on, "mb1_fwd")[0]
    for k in LAYERS:
        lo = layer_ops(off, mo, k)
        ln = layer_ops(on, mn, k)
        if lo is None or ln is None:
            print(f"Layer {k}: block range not found.\n")
            continue
        ro = gpu_cat_rows(lo[0])
        rn = gpu_cat_rows(ln[0])
        print(f"#### Layer {k}\n")
        print("| # | OFF stream | OFF op | OFF offset | OFF dur | ON stream | ON op | ON offset | ON dur |")
        print("|---|---|---|---|---|---|---|---|---|")
        for i in range(max(len(ro), len(rn))):
            a = ro[i] if i < len(ro) else None
            b = rn[i] if i < len(rn) else None
            fa = ("", "", "", "") if a is None else (a[0], a[1], f"{a[2]:.3f}", f"{a[3]:.3f}")
            fb = ("", "", "", "") if b is None else (b[0], b[1], f"{b[2]:.3f}", f"{b[3]:.3f}")
            print(f"| {i} | {fa[0]} | {fa[1]} | {fa[2]} | {fa[3]} | {fb[0]} | {fb[1]} | {fb[2]} | {fb[3]} |")
        print("")
        print(f"CPU-side calls in the layer range: OFF {dict(lo[1])}; ON {dict(ln[1])}.")
        print(f"cudaStreamWaitEvent (syncType 2) by waiting stream: OFF {dict(lo[2])}; ON {dict(ln[2])}.")
        print("")


def run(off_path, on_path, title):
    on = load(on_path)
    off = load(off_path)
    print(f"# {title}\n")
    report_one(off, "OFF")
    report_one(on, "ON")
    section_q1(off, on)
    section_q2(off, on)
    section_q3(off, on)
    section_q3b(off, on)
    section_q6(off, on)
    section_q4(off, on)
    section_q5(off, on)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("off")
    ap.add_argument("on")
    ap.add_argument("--also", nargs=2, metavar=("OFF0", "ON0"), default=None)
    ap.add_argument("--head", default="", help="text file printed first")
    ap.add_argument("--out", default="", help="also write the Markdown to this file")
    args = ap.parse_args()

    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run(args.off, args.on, "Rank pair 1: OFF vs ON")
        if args.also:
            print("")
            run(args.also[0], args.also[1], "Rank pair 2: OFF vs ON (confirmation)")
    body = buf.getvalue()
    head = ""
    if args.head:
        with open(args.head) as f:
            head = f.read().rstrip() + "\n\n---\n\n"
    text = head + body
    print(text, end="")
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
